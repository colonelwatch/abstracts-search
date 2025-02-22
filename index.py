# index.py

# Copyright 2024 Kenny Peng
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
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import accumulate, tee
import json
import logging
import os
import re
from pathlib import Path
from shutil import rmtree
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Literal, TypedDict
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

from utils.gpu_utils import imap, imap_multi_gpu

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks
OPQ_PATTERN = re.compile(r"OPQ([0-9]+)(?:_([0-9]+))?")
RR_PATTERN = re.compile(r"(?:PCAR|RR)([0-9]+)")  # RR <==> PCAR without the PCA
GPU_OPQ_WIDTHS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96]  # GPU widths

logger = logging.getLogger(__name__)


class IndexParameters(TypedDict):
    recall: float  # in this case 10-recall@10
    exec_time: float  # seconds (raw faiss measure is in milliseconds)
    param_string: str  # pass directly to faiss index


def get_env_var[T, U](
    key: str, type_: Callable[[str], T] = str, default: U = None
) -> T | U:
    var = os.getenv(key)
    if var is not None:
        var = type_(var)
    else:
        var = default
    return var


def parse_args() -> Namespace:
    parser = ArgumentParser("train.py", "Trains the FAISS index.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    clean = subparsers.add_parser("clean")
    clean.add_argument("-s", "--source", default=None, type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("source", type=Path)
    train.add_argument("dest", type=Path)
    train.add_argument("-d", "--dimensions", default=None, type=int)  # matryoshka
    train.add_argument("-N", "--normalize", action="store_true")
    train.add_argument("-I", "--inner-product", action="store_true")
    train.add_argument("-p", "--preprocess", default="OPQ96_384")
    train.add_argument("-k", "--intersection", default=None, type=int)  # 1R@1 else kR@k
    train.add_argument("-c", "--clusters", default=None, type=int)
    train.add_argument("-q", "--queries", default=8192, type=int)
    train.add_argument("-P", "--progress", action="store_true")
    train.add_argument("--batch-size", default=1024, type=int)
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
    inner_product: bool
    preprocess: str
    intersection: int | None
    clusters: int | None
    queries: int
    progress: bool
    batch_size: int
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

        if self.dest.exists():
            raise ValueError(f'destination path "{self.dest}" exists')

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

        self.one_recall_at_one = (self.intersection is None)
        self.k = 1 if self.intersection is None else self.intersection


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in dir.glob("*.parquet")]
    dataset: Dataset = Dataset.from_parquet(paths)  # type: ignore
    ids = np.arange(len(dataset), dtype=np.int32)  # add unique integer IDs for later
    return dataset.add_column("index", ids)  # type: ignore  (wrong func signature)


def splits(
    dataset: Dataset, train_size: int, test_size: int, seed: int = 42
) -> tuple[Dataset, Dataset]:
    splits = dataset.train_test_split(test_size, train_size, seed=seed)
    return splits["train"], splits["test"]  # result: two sets of indices, one file


def hash(parameters: list) -> str:
    h = Hasher()
    for parameter in parameters:
        h.update(parameter)
    return h.hexdigest()


def iter_tensors(  # noqa: E302
    dataset: Dataset, batch_size: int
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    with dataset.formatted_as("torch", columns=["index", "embedding"]):
        for batch in dataset.iter(batch_size):
            yield batch["index"], batch["embedding"]  # type: ignore


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

    memmap = np.memmap(cache_path, np.float32, mode="w+", shape=shape)
    try:
        batches = iter_tensors(dataset, args.batch_size)
        batches = imap(batches, preproc, -1)
        with tqdm(
            desc="create_memmap", total=len(dataset), disable=(not args.progress)
        ) as counter:
            i = 0  # save batches to disk by assigning to memmap slices
            for embeddings_batch in batches:
                n_batch = len(embeddings_batch)
                memmap[i:(i + n_batch)] = embeddings_batch.numpy()
                i += n_batch
                counter.update(n_batch)
    except (KeyboardInterrupt, Exception):
        cache_path.unlink()
        raise

    # flush from RAM to disk, then destroy the object on RAM and recreate from disk
    memmap.flush()
    return np.memmap(cache_path, np.float32, mode="r", shape=shape)


# NOTE: ground truth is computed with the full embedding length
def make_ground_truth(
    dataset: Dataset, queries: Dataset, cache_dir: Path, args: TrainArgs
) -> Dataset:
    cache_identifier = hash(
        [
            dataset._fingerprint,
            queries._fingerprint,
            args.normalize,
            args.batch_size,
            args.k,
            args.inner_product
        ]
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

        if args.normalize:
            embeddings = torch.nn.functional.normalize(embeddings)

        if args.inner_product:
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
            scores, args.k, dim=1, largest=args.inner_product  # min L2 or max IP
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
            gt_scores, args.k, dim=1, largest=args.inner_product
        )
        gt_ids = torch.gather(gt_ids, 1, argtop)

        return gt_ids, gt_scores

    # initialize the top k
    n_q, _ = q_embeddings.shape
    shape = (n_q, args.k)
    gt_ids = torch.full(shape, -1, dtype=torch.int32).cuda()
    if args.inner_product:
        gt_scores = torch.zeros(shape, dtype=torch.float32).cuda()
    else:
        gt_scores = torch.full(shape, torch.inf, dtype=torch.float32).cuda()

    with tqdm(
        desc="make_ground_truth", total=len(dataset), disable=(not args.progress)
    ) as counter:
        batches = iter_tensors(dataset, args.batch_size)
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
            "gt_ids": gt_ids.numpy().astype(np.int64),  # faiss expects int64
        }
    )
    ground_truth.save_to_disk(cache_path)
    return ground_truth


def to_gpu(index: faiss.Index, device: int = 0) -> faiss.Index:
    opts = faiss.GpuClonerOptions()
    opts.useFloat16 = True  # float16 is necessary for codes sized 56 bits and over
    env = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(env, device, index, opts)


def to_cpu(index: faiss.Index) -> faiss.Index:
    return faiss.index_gpu_to_cpu(index)


def train_index(
    embeddings: npt.NDArray[np.float32] | np.memmap[Any, np.dtype[np.float32]],
    factory_string: str,
    inner_product: bool,
) -> faiss.Index:
    _, d = embeddings.shape
    metric = faiss.METRIC_INNER_PRODUCT if inner_product else faiss.METRIC_L2
    index: faiss.Index = faiss.index_factory(d, factory_string, metric)

    index = to_gpu(index)
    index.train(embeddings)  # type: ignore (monkey-patched)
    index = to_cpu(index)

    return index


def make_index_shards(
    dir: Path,
    trained_index: faiss.Index,
    dataset: Dataset,
    holdout_ids: npt.NDArray | None,
    args: TrainArgs,
) -> None:
    # Determine n_dataset and holdouts (and copy them to GPUs)
    if holdout_ids is not None:
        holdouts = torch.from_numpy(holdout_ids)
        n_dataset = len(dataset) - len(holdout_ids)
    else:
        holdouts = None
        n_dataset = len(dataset)

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

    # write shards to disk so that file sizes stay under 4GB but also because
    # holding them all in RAM causes OOM-kill (on my machine with 16GB + 16GB swap!)
    dir.mkdir()
    try:
        empty_path = dir / "empty.faiss"
        faiss.write_index(trained_index, str(empty_path))

        # set shard sizes such that file size is under the FAT32 limit (4 GiB),
        # including disk usage at zero entries and the 64-bit ID
        usage_at_zero = empty_path.stat().st_size
        entry_size = args.encoding_width + 8
        shard_size = (4 * (1024 * 1024 * 1024) - usage_at_zero) // entry_size

        # copy trained index
        on_gpus: deque[faiss.Index] = deque(
            to_gpu(trained_index) for _ in range(torch.cuda.device_count())
        )
        index: faiss.Index = faiss.clone_index(trained_index)

        with tqdm(desc="fill_index", total=n_dataset, disable=(not args.progress)) as c:
            for i_shard, row_start in enumerate(range(0, n_dataset, shard_size)):
                shard = dataset.select(  # yields another Datset not rows
                    range(row_start, min(row_start + shard_size, len(dataset)))
                )

                batches = iter_tensors(shard, args.batch_size)
                batches = imap(batches, preproc, -1)
                for ids, embds in batches:
                    on_gpus[0].add_with_ids(embds.numpy(), ids.numpy())  # type: ignore
                    on_gpus.rotate(-1)
                    c.update(len(ids))

                for on_gpu in on_gpus:
                    index.merge_from(to_cpu(on_gpu))
                    on_gpu.reset()
                faiss.write_index(index, str(dir / f"shard_{i_shard:03d}.faiss"))
                index.reset()
    except (KeyboardInterrupt, Exception):
        rmtree(dir)
        raise


@contextmanager
def merged(dir: Path) -> Generator[faiss.Index, None, None]:
    empty_path = dir / "empty.faiss"
    shard_paths = [str(p) for p in dir.glob("shard_*.faiss")]
    index = faiss.read_index(str(empty_path))
    try:
        merge_ondisk(index, shard_paths, "temp.ivfdata")
        yield index
    finally:
        Path("temp.ivfdata").unlink(missing_ok=True)


def tune_index(
    filled_index: faiss.Index, ground_truth: Dataset, args: TrainArgs
) -> list[IndexParameters]:
    with ground_truth.formatted_as("numpy"):
        q: npt.NDArray[np.float32] = ground_truth["embedding"]  # type: ignore
        gt_ids: npt.NDArray[np.int64] = ground_truth["gt_ids"]  # type: ignore

    if args.dimensions is not None:
        q = q[:, :args.dimensions]
    if args.normalize:
        q = q / np.linalg.norm(q, ord=2, axis=1)[:, np.newaxis]

    # init with ground-truth IDs but not ground-truth distances because faiss doesn't
    # use them anyway (see faiss/AutoTune.cpp)
    if args.one_recall_at_one:
        criterion = faiss.OneRecallAtRCriterion(len(ground_truth), 1)
    else:
        criterion = faiss.IntersectionCriterion(len(ground_truth), args.k)
    criterion.set_groundtruth(None, gt_ids)  # type: ignore (monkey-patched)

    params = faiss.ParameterSpace()
    params.verbose = args.progress
    params.initialize(filled_index)
    results: faiss.OperatingPoints = params.explore(  # type: ignore (monkey-patched)
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


def save_ids(path: Path, dataset: Dataset, batch_size: int):
    # the columns themselves aren't needed to run the index
    dataset.remove_columns("embedding").to_parquet(path, batch_size, compression="lz4")


def save_optimal_params(
    path: Path,
    dimensions: int | None,
    normalize: bool,
    optimal_params: list[IndexParameters]
):
    params = {
        "dimensions": dimensions,
        "normalize": normalize,
        "optimal_params": optimal_params
    }
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


def main():
    cache_dir = get_env_var(
        "ABSTRACTS_SEARCH_CACHE", Path, Path.home() / ".cache/abstracts-search"
    )
    args = parse_args()

    if args.mode == "clean":
        args = CleanArgs.from_namespace(args)
        clean_cache(args, cache_dir)
        return 0

    cache_dir.mkdir(exist_ok=True)

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

    dataset = load_dataset(args.source)

    if args.clusters is None:
        clusters = (len(dataset) - args.queries) // TRAIN_SIZE_MULTIPLE
    else:
        clusters = args.clusters
    factory_string = f"{args.preprocess},IVF{clusters},{args.ivf_encoding}"
    train_size = TRAIN_SIZE_MULTIPLE * clusters

    train, queries = splits(dataset, train_size, args.queries)

    with TemporaryDirectory(dir=cache_dir) as tmpdir:
        working_dir = cache_dir if args.use_cache else Path(tmpdir)
        ground_truth = make_ground_truth(dataset, queries, working_dir, args)
        train_memmap = create_memmap(train, working_dir, args)

        faiss_index = train_index(train_memmap, factory_string, args.inner_product)

        shards_dir = Path(tmpdir) / "shards"
        with queries.formatted_as("numpy"):
            q_ids: npt.NDArray = queries["index"]  # type: ignore
        make_index_shards(shards_dir, faiss_index, dataset, q_ids, args)
        with merged(shards_dir) as merged_index:
            optimal_params = tune_index(merged_index, ground_truth, args)

    args.dest.mkdir()
    try:
        save_ids(args.dest / "ids.parquet", dataset, args.batch_size)
        save_optimal_params(
            args.dest / "params.json", args.dimensions, args.normalize, optimal_params
        )
        make_index_shards(args.dest / "shards", faiss_index, dataset, None, args)
    except (KeyboardInterrupt, Exception):
        rmtree(args.dest)
        raise


if __name__ == "__main__":
    exit(main())
