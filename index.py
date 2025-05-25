# index.py

# Copyright 2025 Kenny Peng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import accumulate, tee
import json
import logging
import re
from pathlib import Path
from shutil import rmtree, copy
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Generator, Literal, TypedDict
import warnings

from datasets import Dataset, disable_progress_bars, disable_caching
from datasets.config import HF_DATASETS_CACHE
from datasets.fingerprint import Hasher
import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
from faiss.contrib.ondisk import merge_ondisk
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from utils.env_utils import CACHE
from utils.gpu_utils import imap, imap_multi_gpu, iunsqueeze

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks
OPQ_PATTERN = re.compile(r"OPQ([0-9]+)(?:_([0-9]+))?")
RR_PATTERN = re.compile(r"(?:PCAR|RR)([0-9]+)")  # RR <==> PCAR without the PCA
GPU_OPQ_WIDTHS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96]  # GPU widths
BATCH_SIZE = 1024
SHARD_SIZE = 1048576  # keep temporary shard sizes small to save on RAM

logger = logging.getLogger(__name__)


class IndexParameters(TypedDict):
    recall: float  # in this case 10-recall@10
    exec_time: float  # seconds (raw faiss measure is in milliseconds)
    param_string: str  # pass directly to faiss index


class Params(TypedDict):
    dimensions: int | None
    normalize: bool
    optimal_params: list[IndexParameters]


def parse_args() -> Namespace:
    parser = ArgumentParser("train.py", "Trains the FAISS index.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    clean = subparsers.add_parser("clean")
    clean.add_argument("-s", "--source", default=None, type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("source", type=Path)
    train.add_argument("dest", type=Path)
    train.add_argument("-d", "--dimensions", default=None, type=int)  # matryoshka
    train.add_argument("-N", "--normalize", action="store_true")  # also if normalize_d_
    train.add_argument("-p", "--preprocess", default="OPQ96_384")
    train.add_argument("-k", "--intersection", default=None, type=int)  # 1R@1 else kR@k
    train.add_argument("-c", "--clusters", default=None, type=int)
    train.add_argument("-q", "--queries", default=8192, type=int)
    train.add_argument("-P", "--progress", action="store_true")
    train.add_argument("--use-cache", action="store_true")  # for experiments only

    return parser.parse_args()


@dataclass
class CleanArgs:
    mode: Literal["clean"]
    source: Path | None

    @classmethod
    def from_namespace(cls, namespace: Namespace):
        return cls(**vars(namespace))


@dataclass
class TrainArgs:
    mode: Literal["train"]
    source: Path
    dest: Path
    dimensions: int | None
    normalize: bool
    preprocess: str
    intersection: int | None
    clusters: int | None
    queries: int
    progress: bool
    use_cache: bool

    # not args
    ivf_encoding: str = field(init=False, compare=False)
    encoding_width: int = field(init=False, compare=False)
    one_recall_at_one: bool = field(init=False, compare=False)
    k: int = field(init=False, compare=False)

    @classmethod
    def from_namespace(cls, namespace: Namespace):
        return cls(**vars(namespace))

    def __post_init__(self):
        if not self.source.exists():
            raise ValueError(f'source path "{self.source}" does not exist')

        if self.dimensions is not None and not self.normalize:
            self.normalize = True
            warnings.warn("inferring --normalize from --dimension")

        if (match := OPQ_PATTERN.match(self.preprocess)):
            self.ivf_encoding = f"PQ{match[1]}"
            self.encoding_width = int(match[1])
            if self.encoding_width not in GPU_OPQ_WIDTHS:
                raise ValueError(f"OPQ width {self.encoding_width} is not valid")
        elif (match := RR_PATTERN.match(self.preprocess)):
            self.ivf_encoding = "SQ8"
            self.encoding_width = int(match[1])
        else:
            raise ValueError(f'preprocessing string "{self.preprocess}" is not valid')

        if self.intersection is None:
            self.one_recall_at_one = True
            self.k = 1
        else:
            self.one_recall_at_one = False
            self.k = self.intersection


@contextmanager
def del_on_exc(path: Path | Iterable[Path]) -> Generator[None, None, None]:
    paths = [path] if isinstance(path, Path) else path
    try:
        yield
    except (KeyboardInterrupt, Exception):
        for p in paths:
            if not p.exists():
                continue
            if p.is_dir():
                rmtree(p)
            else:
                p.unlink()
        raise


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in dir.glob("*.parquet")]
    dataset: Dataset = Dataset.from_parquet(paths)  # type: ignore
    ids = np.arange(len(dataset), dtype=np.int32)  # add unique integer IDs for later
    return dataset.add_column("index", ids)  # type: ignore  (wrong func signature)


def hash(parameters: list) -> str:
    h = Hasher()
    for parameter in parameters:
        h.update(parameter)
    return h.hexdigest()


def iter_tensors(
    dataset: Dataset
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    with dataset.formatted_as("torch", columns=["index", "embedding"]):
        for batch in dataset.iter(BATCH_SIZE):
            yield batch["index"], batch["embedding"]  # type: ignore


# NOTE: ground truth is computed with the full embedding length
def make_ground_truth(
    dataset: Dataset, queries: Dataset, cache_dir: Path, args: TrainArgs
) -> Dataset:
    cache_identifier = hash(
        [dataset._fingerprint, queries._fingerprint, args.normalize, args.k]
    )
    cache_path = cache_dir / f"gt_{cache_identifier}"
    if cache_path.exists():
        return Dataset.load_from_disk(cache_path)

    with queries.formatted_as("torch", columns=["embedding", "index"]):
        q_embeddings: torch.Tensor = queries["embedding"]  # type: ignore
        q_ids: torch.Tensor = queries["index"]  # type: ignore

        if args.normalize:
            q_embeddings = torch.nn.functional.normalize(q_embeddings)

    # make available a local copy to each GPU
    n_devices = torch.cuda.device_count()
    q_embeddings_copy = [q_embeddings.to(f"cuda:{i}") for i in range(n_devices)]
    q_ids_copy = [q_ids.to(f"cuda:{i}") for i in range(n_devices)]

    # for unit vectors, the L2 minimizing is also the inner-product maximizing
    inner_product_search = args.normalize

    def get_length(ids: torch.Tensor, _: torch.Tensor) -> int:
        return len(ids)

    def local_topk(
        device: torch.device, ids: torch.Tensor, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # send to GPU asynchronously
        embeddings = embeddings.to(device, non_blocking=True)
        ids = ids.to(device, non_blocking=True)

        # don't consider the queries themselves as possible ground truth
        not_in_queries = torch.isin(ids, q_ids_copy[device.index], invert=True)
        embeddings = embeddings[not_in_queries]
        ids = ids[not_in_queries]

        if inner_product_search:
            # ensure that the vectors are unit-length
            embeddings = torch.nn.functional.normalize(embeddings)

            # becomes a matmult for multiple data
            scores = q_embeddings_copy[device.index] @ embeddings.T
        else:
            # prefer direct calc over following the quadratic form with matmult
            scores = torch.cdist(
                q_embeddings_copy[device.index],
                embeddings,
                compute_mode="donot_use_mm_for_euclid_dist"
            )

        # only yield k from this batch, in the extreme this k replaces all running k
        top_scores, argtop = torch.topk(
            scores, args.k, dim=1, largest=inner_product_search
        )
        top_ids = ids[argtop.flatten()].reshape(argtop.shape)

        if n_devices > 1:
            return top_ids.cpu(), top_scores.cpu()
        else:
            return top_ids, top_scores  # reduce step is on this GPU

    def reduce_topk(
        gt: tuple[torch.Tensor, torch.Tensor],
        batch_top: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_ids, gt_scores = gt
        batch_ids, batch_scores = batch_top

        batch_ids = batch_ids.cuda(non_blocking=True)
        batch_scores = batch_scores.cuda(non_blocking=True)

        # update the top k for each query
        gt_scores = torch.hstack((gt_scores, batch_scores))
        gt_ids = torch.hstack((gt_ids, batch_ids))
        gt_scores, argtop = torch.topk(
            gt_scores, args.k, dim=1, largest=inner_product_search
        )
        gt_ids = torch.gather(gt_ids, 1, argtop)

        return gt_ids, gt_scores

    # initialize the top k
    n_q, _ = q_embeddings.shape
    shape = (n_q, args.k)
    gt_ids = torch.full(shape, -1, dtype=torch.int32).cuda()
    if inner_product_search:
        gt_scores = torch.zeros(shape, dtype=torch.float32).cuda()
    else:
        gt_scores = torch.full(shape, torch.inf, dtype=torch.float32).cuda()

    with tqdm(
        desc="make_ground_truth", total=len(dataset), disable=(not args.progress)
    ) as counter:
        batches = iter_tensors(dataset)
        batches, batches_copy = tee(batches, 2)
        lengths = imap(batches_copy, get_length, 0)
        batches = imap_multi_gpu(batches, local_topk)
        batches = accumulate(batches, reduce_topk, initial=(gt_ids, gt_scores))
        batches = zip(lengths, batches)
        for length, (gt_ids, _) in batches:
            counter.update(length)

    gt_ids = gt_ids.cpu()

    ground_truth = Dataset.from_dict(
        {
            "embedding": q_embeddings,
            "gt_ids": gt_ids.numpy(),
        }
    )
    ground_truth.save_to_disk(cache_path)
    return ground_truth


def create_memmap(
    dataset: Dataset, cache_dir: Path, args: TrainArgs
) -> np.memmap[Any, np.dtype[np.float32]]:
    n = len(dataset)
    d = len(dataset[0]["embedding"]) if args.dimensions is None else args.dimensions
    shape = (n, d)

    cache_identifier = hash([dataset._fingerprint, args.dimensions, args.normalize])
    cache_path = cache_dir / f"train_{cache_identifier}.memmap"
    if cache_path.exists():
        return np.memmap(cache_path, np.float32, mode="r", shape=shape)

    def preproc(_, embeddings: torch.Tensor) -> torch.Tensor:
        if args.dimensions is not None:
            embeddings = embeddings[:, :args.dimensions]
        if args.normalize:
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    i = 0
    memmap = np.memmap(cache_path, np.float32, mode="w+", shape=shape)
    with (
        del_on_exc(cache_path),
        tqdm(
            desc="create_memmap", total=len(dataset), disable=(not args.progress)
        ) as counter
    ):
        batches = iter_tensors(dataset)
        batches = imap(batches, preproc, -1)
        for embeddings_batch in batches:
            # save batches to disk by assigning to memmap slices
            n_batch = len(embeddings_batch)
            memmap[i:(i + n_batch)] = embeddings_batch.numpy()
            i += n_batch
            counter.update(n_batch)

    # flush from RAM to disk, then destroy the object on RAM and recreate from disk
    memmap.flush()
    return np.memmap(cache_path, np.float32, mode="r", shape=shape)


def to_gpu(index: faiss.Index, device: int = 0) -> faiss.Index:
    opts = faiss.GpuClonerOptions()
    opts.useFloat16 = True  # float16 is necessary for codes sized 56 bits and over
    env = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(env, device, index, opts)


def to_cpu(index: faiss.Index) -> faiss.Index:
    return faiss.index_gpu_to_cpu(index)


def train_index(
    train: Dataset, factory_string: str, cache_dir: Path, args: TrainArgs
) -> Path:
    cache_identifier = hash(  # includes create_memmap parameters
        [train._fingerprint, factory_string, args.dimensions, args.normalize]
    )
    cache_path = cache_dir / f"empty_{cache_identifier}.faiss"
    if cache_path.exists():
        return cache_path

    train_memmap = create_memmap(train, cache_dir, args)

    # doing a bit of testing seems to show that passing METRIC_L2 is superior to passing
    # METRIC_INNER_PRODUCT for the same factory string, even for normalized embeddings
    _, d = train_memmap.shape
    index: faiss.Index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)

    index = to_gpu(index)
    index.train(train_memmap)  # type: ignore (monkey-patched)
    index = to_cpu(index)

    faiss.write_index(index, str(cache_path))
    return cache_path


def make_index(
    dir: Path,
    trained_path: Path,
    dataset: Dataset,
    holdouts: torch.Tensor | None,
    cache_dir: Path,
    args: TrainArgs,
) -> None:
    holdout_list = [] if holdouts is None else holdouts.tolist()
    cache_identifier = hash([trained_path, dataset._fingerprint, *holdout_list])

    index_cache_path = cache_dir / f"filled_{cache_identifier}.faiss"
    ondisk_cache_path = cache_dir / f"ondisk_{cache_identifier}.ivfdata"
    if index_cache_path.exists() and ondisk_cache_path.exists():
        copy(index_cache_path, dir / "index.faiss")
        copy(ondisk_cache_path, dir / "ondisk.ivfdata")
        return

    # clone trained index for filling on the GPU and merging on the CPU
    trained_index = faiss.read_index(str(trained_path))
    on_gpus: list[faiss.Index] = [
        to_gpu(trained_index) for _ in range(torch.cuda.device_count())
    ]
    index: faiss.Index = faiss.clone_index(trained_index)

    def preproc(
        ids: torch.Tensor, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if holdouts is not None:
            not_in = torch.isin(ids, holdouts, invert=True)
            ids = ids[not_in]
            embeddings = embeddings[not_in]
        if args.dimensions is not None:
            embeddings = embeddings[:, :args.dimensions]
        if args.normalize:
            embeddings = torch.nn.functional.normalize(embeddings)
        return ids, embeddings

    def add_with_gpu(
        device: torch.device, ids: torch.Tensor, embeddings: torch.Tensor
    ) -> int:
        on_gpu = on_gpus[device.index]
        on_gpu.add_with_ids(embeddings.numpy(), ids.numpy())  # type: ignore
        return len(ids)  # yield the number of embeddings added

    def transfer_and_reset(on_gpu: faiss.Index) -> faiss.Index:
        on_cpu = to_cpu(on_gpu)
        on_gpu.reset()
        return on_cpu

    # write shards to disk because holding them all in RAM causes OOM-kill
    shard_paths: list[Path] = []
    try:
        n_ids = (len(dataset) - len(holdouts)) if holdouts is not None else len(dataset)
        with tqdm(desc="make_index", total=n_ids, disable=(not args.progress)) as c:
            for i_shard, row_start in enumerate(range(0, len(dataset), SHARD_SIZE)):
                shard = dataset.select(  # yields another Datset not rows
                    range(row_start, min(row_start + SHARD_SIZE, len(dataset)))
                )

                batches = iter_tensors(shard)
                batches = imap(batches, preproc, -1)
                counts = imap_multi_gpu(batches, add_with_gpu)
                for count in counts:
                    c.update(count)

                # transfer takes time, so do this across all GPUs in parallel
                for on_cpu in imap(iunsqueeze(on_gpus), transfer_and_reset, -1):
                    index.merge_from(on_cpu)

                shard_path = dir / f"shard_{i_shard:03d}.faiss"
                faiss.write_index(index, str(shard_path))
                shard_paths.append(shard_path)

                index.reset()

        # merge_ondisk takes file _names_ and only puts in working directory...
        ondisk_relpath = Path("./ondisk.ivfdata")
        merge_ondisk(index, [str(p) for p in shard_paths], ondisk_relpath.name)
        ondisk_relpath.rename(dir / ondisk_relpath)  # ... so move it after
    finally:
        for p in shard_paths:
            p.unlink()

    # write the index (which points to `ondisk.ivfdata`) and drop the shards
    faiss.write_index(index, str(dir / "index.faiss"))

    # copy into the cache
    copy(dir / "index.faiss", index_cache_path)
    copy(dir / "ondisk.ivfdata", ondisk_cache_path)


def open_ondisk(dir: Path) -> faiss.Index:
    # without IO_FLAG_ONDISK_SAME_DIR, read_index gets on-disk indices in working dir
    return faiss.read_index(str(dir / "index.faiss"), faiss.IO_FLAG_ONDISK_SAME_DIR)


def tune_index(
    filled_index: faiss.Index, ground_truth: Dataset, args: TrainArgs
) -> list[IndexParameters]:
    with ground_truth.formatted_as("numpy"):
        q: npt.NDArray[np.float32] = ground_truth["embedding"]  # type: ignore
        gt_ids: npt.NDArray[np.int32] = ground_truth["gt_ids"]  # type: ignore

    if args.dimensions is not None:
        q = q[:, :args.dimensions]
    if args.normalize:
        q = q / np.linalg.norm(q, ord=2, axis=1)[:, np.newaxis]
    gt_ids_int64 = gt_ids.astype(np.int64)  # faiss expects int64

    # init with ground-truth IDs but not ground-truth distances because faiss doesn't
    # use them anyway (see faiss/AutoTune.cpp)
    if args.one_recall_at_one:
        criterion = faiss.OneRecallAtRCriterion(len(ground_truth), 1)
    else:
        criterion = faiss.IntersectionCriterion(len(ground_truth), args.k)
    criterion.set_groundtruth(None, gt_ids_int64)  # type: ignore (monkey-patched)

    p_space = faiss.ParameterSpace()
    p_space.verbose = args.progress
    p_space.initialize(filled_index)
    results: faiss.OperatingPoints = p_space.explore(  # type: ignore (monkey-patched)
        filled_index, q, criterion
    )

    pareto_vector: faiss.OperatingPointVector = results.optimal_pts
    optimal_params: list[IndexParameters] = []
    for i in range(pareto_vector.size()):
        point: faiss.OperatingPoint = pareto_vector.at(i)
        params = IndexParameters(  # converts from ms to seconds
            recall=point.perf, exec_time=(0.001 * point.t), param_string=point.key
        )
        optimal_params.append(params)

    return optimal_params


def save_ids(path: Path, dataset: Dataset):
    # only the id column is needed to run the index
    dataset.select_columns("id").to_parquet(path, BATCH_SIZE, compression="lz4")


def save_params(
    path: Path,
    dimensions: int | None,
    normalize: bool,
    optimal_params: list[IndexParameters]
):
    params = Params(
        dimensions=dimensions, normalize=normalize, optimal_params=optimal_params
    )
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def clean_cache(args: CleanArgs, cache_dir: Path):
    if cache_dir.exists():
        rmtree(cache_dir)

    # get cache directory path by following the path to an individual cache file
    # NOTE: if the cache wasn't created, this will create then delete the cache
    if args.source is not None and args.source.exists():
        dataset = load_dataset(args.source)
        file_0_path = Path(dataset.cache_files[0]["filename"])
        del dataset

        # parts[0] -> dataset ("parquet" by default)
        # parts[1] -> cache (pseudorandom, seeded with stuff like file metadata)
        # since this is a low-level detail, sanity-check the above facts for change
        file_0_path_rel = file_0_path.relative_to(HF_DATASETS_CACHE)
        dataset_name = file_0_path_rel.parts[0]
        cache_name = file_0_path_rel.parts[1]
        if not (
            dataset_name == "parquet"
            and "default-" in cache_name
        ):
            print("error: path integrity check failed", file=stderr)
            return 1

        # remove the cache directory
        hf_cache_dir = HF_DATASETS_CACHE / dataset_name / cache_name
        rmtree(hf_cache_dir)

        # remove its associated lock
        for lock in HF_DATASETS_CACHE.iterdir():
            if not lock.suffix == ".lock":
                continue
            if cache_name in str(lock):
                lock.unlink()


def ensure_trained(dataset: Dataset, args: TrainArgs) -> tuple[Path, Path]:
    trained_dest_path = args.dest / "empty.faiss"
    params_path = args.dest / "params.json"
    if trained_dest_path.exists() and params_path.exists():
        return trained_dest_path, params_path

    # extract a "train" set and a "test" set, which is actually ground-truth
    # queries to be held out from the making of a provisional index
    if args.clusters is None:
        clusters = (len(dataset) - args.queries) // TRAIN_SIZE_MULTIPLE
    else:
        clusters = args.clusters
    factory_string = f"{args.preprocess},IVF{clusters},{args.ivf_encoding}"
    train_size = TRAIN_SIZE_MULTIPLE * clusters
    splits = dataset.train_test_split(args.queries, train_size, seed=42)
    train = splits["train"]
    queries = splits["test"]

    # train index and Pareto-optimal params from splits
    with TemporaryDirectory(dir=CACHE) as tmpdir:
        tmpdir = Path(tmpdir)
        working_dir = CACHE if args.use_cache else tmpdir

        ground_truth = make_ground_truth(dataset, queries, working_dir, args)
        trained_path = train_index(train, factory_string, working_dir, args)

        with queries.formatted_as("torch"):
            q_ids: torch.Tensor = queries["index"]  # type: ignore
        make_index(tmpdir, trained_path, dataset, q_ids, working_dir, args)
        merged_index = open_ondisk(tmpdir)
        optimal_params = tune_index(merged_index, ground_truth, args)

        with del_on_exc([trained_dest_path, params_path]):
            copy(trained_path, trained_dest_path)
            save_params(
                params_path,
                args.dimensions,
                args.normalize,
                optimal_params
            )
    
    return trained_dest_path, params_path


def ensure_filled(
    dataset: Dataset, trained_path: Path, args: TrainArgs
) -> tuple[Path, Path, Path]:
    # TODO: make make_index take index_paths entries?
    ids_path = args.dest / "ids.parquet"
    index_paths = (args.dest / "index.faiss", args.dest / "ondisk.ivfdata")
    with del_on_exc([ids_path, *index_paths]):
        save_ids(ids_path, dataset)
        make_index(args.dest, trained_path, dataset, None, CACHE, args)
    return ids_path, *index_paths


def main():
    args = parse_args()

    if args.mode == "clean":
        args = CleanArgs.from_namespace(args)
        clean_cache(args, CACHE)
        return 0

    CACHE.mkdir(exist_ok=True)

    try:
        args = TrainArgs.from_namespace(args)
    except ValueError as e:
        print("error:", e.args[0], file=stderr)
        return 1

    # run through global huggingface datasets settings
    if not args.use_cache:
        disable_caching()
    if not args.progress:
        disable_progress_bars()

    # prepare source dataset and destination directory
    dataset = load_dataset(args.source)
    if not args.dest.exists():
        args.dest.mkdir()

    trained_path, _ = ensure_trained(dataset, args)
    _ = ensure_filled(dataset, trained_path, args)


if __name__ == "__main__":
    exit(main())
