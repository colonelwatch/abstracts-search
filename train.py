# train.py

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
from itertools import accumulate, tee
import json
import logging
import os
from pathlib import Path
from shutil import rmtree, move
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Literal, overload, TypedDict

from datasets import Dataset, disable_progress_bars, disable_caching
from datasets.config import HF_DATASETS_CACHE
from datasets.fingerprint import Hasher
import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
from faiss.contrib.ondisk import merge_ondisk
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from utils.gpu_utils import imap, imap_multi_gpu, iunsqueeze

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks

logger = logging.getLogger(__name__)
torch_norm = torch.nn.functional.normalize


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
    train.add_argument("-k", "--intersection", default=None, type=int)
    train.add_argument("-c", "--clusters", default=None, type=int)  # TODO: raise
    train.add_argument("-q", "--queries", default=16384, type=int)
    train.add_argument("-t", "--tasks", default=None, type=int)
    train.add_argument("-P", "--progress", action="store_true")
    train.add_argument("--truncate", default=None, type=int)
    train.add_argument("--batch-size", default=1024, type=int)
    train.add_argument("--shard-size", default=4194304, type=int)
    train.add_argument("--use-cache", action="store_true")

    return parser.parse_args()


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in dir.glob("*.parquet")]
    dataset: Dataset = Dataset.from_parquet(paths)  # type: ignore
    ids = np.arange(len(dataset), dtype=np.int32)
    return dataset.add_column("ids", ids)  # type: ignore (new_fingerprint not required)


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


@overload
def iter_tensors(
    dataset: Dataset, batch_size: int, embeddings_only: Literal[False] = False
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    ...

@overload  # noqa: E302
def iter_tensors(
    dataset: Dataset, batch_size: int, embeddings_only: Literal[True]
) -> Generator[torch.Tensor, None, None]:
    ...

def iter_tensors(  # noqa: E302
    dataset: Dataset, batch_size: int, embeddings_only: bool = False
) -> (
    Generator[tuple[torch.Tensor, torch.Tensor], None, None] |
    Generator[torch.Tensor, None, None]
):
    columns = ["embeddings"] if embeddings_only else ["ids", "embeddings"]
    with dataset.formatted_as("torch", columns=columns):
        for batch in dataset.iter(batch_size):
            if embeddings_only:
                yield batch["embeddings"]  # type: ignore
            else:
                yield batch["ids"], batch["embeddings"]  # type: ignore


def create_memmap(
    dataset: Dataset,
    dimensions: int | None,
    normalize: bool,
    cache_dir: Path,
    batch_size: int,
    n_tasks: int,
    progress: bool
) -> np.memmap[Any, np.dtype[np.float32]]:
    n = len(dataset)
    d = len(dataset[0]["embeddings"]) if dimensions is None else dimensions
    shape = (n, d)

    cache_identifier = hash([dataset._fingerprint, dimensions, normalize])
    cache_path = cache_dir / f"train_{cache_identifier}.memmap"
    if cache_path.exists():
        return np.memmap(cache_path, np.float32, mode="r", shape=shape)

    def preproc(device: torch.device, x: torch.Tensor) -> torch.Tensor:
        if dimensions is not None:
            x = x[:, :dimensions]
        x = x.to(device, non_blocking=True)
        if normalize:
            x = torch.nn.functional.normalize(x)
        return x.cpu()

    memmap = np.memmap(cache_path, np.float32, mode="w+", shape=shape)
    try:
        batches = iter_tensors(dataset, batch_size, embeddings_only=True)
        batches = iunsqueeze(batches)
        batches = imap_multi_gpu(batches, preproc, n_tasks)
        with tqdm(total=len(dataset), disable=(not progress)) as counter:
            i = 0
            for embeddings_batch in batches:
                n_batch = len(embeddings_batch)
                memmap[i:(i + n_batch)] = embeddings_batch.numpy()
                i += n_batch
                counter.update(n_batch)
    except (KeyboardInterrupt, Exception):
        cache_path.unlink()
        raise

    memmap.flush()
    return np.memmap(cache_path, np.float32, mode="r", shape=shape)


# NOTE: ground truth is computed with the full embedding length
def make_ground_truth(
    dataset: Dataset,
    queries: Dataset,
    normalize: bool,
    batch_size: int,
    k: int | None,
    inner_product: bool,
    cache_dir: Path,
    n_tasks: int,
    progress: bool,
) -> Dataset:
    cache_identifier = hash(
        [
            dataset._fingerprint,
            queries._fingerprint,
            normalize,
            batch_size,
            k,
            inner_product
        ]
    )
    cache_path = cache_dir / f"gt_{cache_identifier}"
    if cache_path.exists():
        return Dataset.load_from_disk(cache_path)

    with queries.formatted_as("torch", columns=["embeddings", "ids"]):
        q: torch.Tensor = queries["embeddings"]  # type: ignore
        q_ids: torch.Tensor = queries["ids"]  # type: ignore

        if normalize:
            q = torch.nn.functional.normalize(q)

    n_devices = torch.cuda.device_count()
    q_copy = [q.to(f"cuda:{i}") for i in range(n_devices)]
    q_ids_copy = [q_ids.to(f"cuda:{i}") for i in range(n_devices)]

    if k is None:
        k = 1  # 1-Recall @ 1

    def get_length(ids: torch.Tensor, _: torch.Tensor) -> int:
        return len(ids)

    def local_topk(
        device: torch.device, d_batch_ids: torch.Tensor, d_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_batch = d_batch.to(device, non_blocking=True)
        d_batch_ids = d_batch_ids.to(device, non_blocking=True)

        not_in_queries = torch.isin(d_batch_ids, q_ids_copy[device.index], invert=True)
        d_batch = d_batch[not_in_queries]
        d_batch_ids = d_batch_ids[not_in_queries]

        if normalize:
            d_batch = torch.nn.functional.normalize(d_batch)

        if inner_product:
            scores_batch = q_copy[device.index] @ d_batch.T
        else:
            # prefer direct calc over following the quadratic form with matmult
            scores_batch = torch.cdist(
                q_copy[device.index],
                d_batch,
                compute_mode="donot_use_mm_for_euclid_dist"
            )

        top_scores_batch, argtop = torch.topk(scores_batch, k, 1, inner_product)
        top_ids_batch = d_batch_ids[argtop.flatten()].reshape(argtop.shape)

        if n_devices > 1:
            return top_ids_batch.cpu(), top_scores_batch.cpu()
        else:
            return top_ids_batch, top_scores_batch

    def reduce_topk(
        gt: tuple[torch.Tensor, torch.Tensor],
        local_top: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_ids, gt_scores = gt
        top_ids_batch, top_scores_batch = local_top

        top_ids_batch = top_ids_batch.cuda(non_blocking=True)
        top_scores_batch = top_scores_batch.cuda(non_blocking=True)

        gt_scores = torch.hstack((gt_scores, top_scores_batch))
        gt_ids = torch.hstack((gt_ids, top_ids_batch))
        gt_scores, argtop = torch.topk(gt_scores, k, 1, inner_product)
        gt_ids = torch.gather(gt_ids, 1, argtop)

        return gt_ids, gt_scores

    n_q, _ = q.shape
    gt_ids = torch.full((n_q, k), -1, dtype=torch.int32).cuda()
    if inner_product:
        gt_scores = torch.zeros((n_q, k), dtype=torch.float32).cuda()
    else:
        gt_scores = torch.full((n_q, k), torch.inf, dtype=torch.float32).cuda()

    with tqdm(total=(len(dataset) - len(queries)), disable=(not progress)) as counter:
        batches = iter_tensors(dataset, batch_size)
        batches, batches_copy = tee(batches, 2)
        lengths = imap(batches_copy, get_length, None)
        batches = imap_multi_gpu(batches, local_topk, n_tasks)
        batches = accumulate(batches, reduce_topk, initial=(gt_ids, gt_scores))
        batches = zip(lengths, batches)
        for length, (gt_ids, _) in batches:
            counter.update(length)

    gt_ids = gt_ids.cpu()

    ground_truth = Dataset.from_dict(
        {
            "embeddings": queries["embeddings"],
            "gt_ids": gt_ids.numpy().astype(np.int64),
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


def fill_index(
    index_path: Path,
    ivf_path: Path,
    trained_index: faiss.Index,
    dataset: Dataset,
    holdout_ids: npt.NDArray | None,
    dimensions: int | None,
    normalize: bool,
    cache_dir: Path,
    batch_size: int,
    shard_size: int,
    n_tasks: int,
    progress: bool,
) -> faiss.Index:
    if holdout_ids is not None:
        holdouts_cpu = torch.from_numpy(holdout_ids)
        holdouts = [
            holdouts_cpu.to(f"cuda:{i}")
            for i in range(torch.cuda.device_count())
        ]
        n_dataset = len(dataset) - len(holdout_ids)
    else:
        holdouts = None
        n_dataset = len(dataset)

    def get_length(ids: torch.Tensor, _: torch.Tensor) -> int:
        return len(ids)

    def preproc(
        device: torch.device, ids: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dimensions is not None:
            x = x[:, :dimensions]
        x = x.to(device, non_blocking=True)
        if normalize:
            x = torch.nn.functional.normalize(x)
        if holdouts is not None:
            ids = ids.to(device, non_blocking=True)
            not_in = torch.isin(ids, holdouts[device.index], invert=True)
            ids = ids[not_in]
            x = x[not_in]
        return ids.cpu(), x.cpu()

    def make_shard_index(device: torch.device, shard_start: int) -> faiss.Index:
        dev_id = device.index

        on_gpu = to_gpu(trained_index, device=dev_id)
        shard = dataset.select(
            range(shard_start, min(shard_start + shard_size, len(dataset)))
        )

        batches = iter_tensors(shard, batch_size)
        batches, batches_copy = tee(batches, 2)
        lengths = imap(batches_copy, get_length, None)
        batches = imap(batches, lambda ids, x: preproc(device, ids, x), None)
        batches = zip(lengths, batches)
        with tqdm(total=len(shard), disable=(not progress), position=(dev_id + 1)) as c:
            for n_batch, (ids, x) in batches:
                on_gpu.add_with_ids(x.numpy(), ids.numpy())  # type: ignore
                c.update(n_batch)
        
        return to_cpu(on_gpu)

    with TemporaryDirectory(dir=cache_dir) as tmpdir:
        shards = range(0, n_dataset, shard_size)
        n_shards = len(shards)
        shards = iunsqueeze(shards)
        shards = imap_multi_gpu(shards, make_shard_index, n_tasks)
        shards = tqdm(shards, total=n_shards, disable=(not progress), position=0)

        shard_paths: list[Path] = []
        for i_shard, shard_index in enumerate(shards):
            path = Path(tmpdir) / f"index_{i_shard:03d}.faiss'"
            faiss.write_index(shard_index, str(path))
            shard_paths.append(path)

        temp_ivf_path = Path(ivf_path.name)
        try:
            index = faiss.clone_index(trained_index)
            merge_ondisk(index, [str(p) for p in shard_paths], str(temp_ivf_path))
            move(temp_ivf_path, ivf_path)
            faiss.write_index(index, str(index_path))
        except (KeyboardInterrupt, Exception):
            temp_ivf_path.unlink(missing_ok=True)
            ivf_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
            raise

    return index


def tune_index(
    filled_index: faiss.Index,
    ground_truth: Dataset,
    dimensions: int | None,
    normalize: bool,
    k: int | None,
    progress: bool
) -> list[IndexParameters]:
    with ground_truth.formatted_as("numpy"):
        q: npt.NDArray[np.float32] = ground_truth["embeddings"]  # type: ignore
        gt_ids: npt.NDArray[np.int64] = ground_truth["gt_ids"]  # type: ignore

    if dimensions is not None:
        q = q[:, :dimensions]
    if normalize:
        q = q / np.linalg.norm(q, ord=2, axis=1)[:, np.newaxis]

    # init with ground-truth IDs but not ground-truth distances because faiss doesn't
    # use them anyway (see faiss/AutoTune.cpp)
    if k is None:
        criterion = faiss.OneRecallAtRCriterion(len(ground_truth), 1)
    else:
        criterion = faiss.IntersectionCriterion(len(ground_truth), k)
    criterion.set_groundtruth(None, gt_ids)  # type: ignore (monkey-patched)

    params = faiss.ParameterSpace()
    params.verbose = progress
    params.initialize(filled_index)
    results: faiss.OperatingPoints = params.explore(  # type: ignore (monkey-patched)
        filled_index, q, criterion
    )

    pareto_vector: faiss.OperatingPointVector = results.optimal_pts
    optimal_params: list[IndexParameters] = []
    for i in range(pareto_vector.size()):
        point: faiss.OperatingPoint = pareto_vector.at(i)
        params = IndexParameters(
            recall=point.perf, exec_time=(0.001 * point.t), param_string=point.key
        )
        optimal_params.append(params)

    return optimal_params


def save_ids(path: Path, dataset: Dataset, batch_size: int):
    dataset.remove_columns("embeddings").to_parquet(path, batch_size, compression="lz4")


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


def main():
    cache_dir = get_env_var(
        "ABSTRACTS_SEARCH_CACHE", Path, Path.home() / ".cache/abstracts-search"
    )
    args = parse_args()

    if args.mode == "clean":
        if cache_dir.exists():
            rmtree(cache_dir)

        # get cache directory path by following the path to an individual cache file
        # NOTE: if the cache wasn't created, this will create then delete the cache
        clean_source: Path | None = args.source
        if clean_source is not None and clean_source.exists():
            dataset = load_dataset(clean_source)
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

        return 0

    cache_dir.mkdir(exist_ok=True)

    source: Path = args.source
    if not source.exists():
        print(f'error: source path "{source}" does not exist', file=stderr)
        return 1

    dest: Path = args.dest
    if dest.exists():
        print(f'error: destination path "{dest}" exists', file=stderr)
        return 1

    dimensions: int | None = args.dimensions
    normalize: bool = args.normalize
    if args.dimensions is not None and not args.normalize:
        print("warning: inferring --normalize from --dimension", file=stderr)
        normalize = True

    # run through global huggingface datasets settings
    use_cache: bool = args.use_cache
    progress: bool = args.progress
    if not use_cache:
        disable_caching()  # caching is good for experimention, not otherwise
    if not progress:
        disable_progress_bars()

    n_tasks: int | None = args.tasks
    if n_tasks is None:
        n_tasks = torch.cuda.device_count() + 2

    dataset = load_dataset(source)
    truncate: int | None = args.truncate
    if truncate is not None:
        dataset = dataset.take(truncate)

    n_clusters: int | None = args.clusters
    n_queries: int = args.queries
    if n_clusters is None:
        n_clusters = (len(dataset) - n_queries) // TRAIN_SIZE_MULTIPLE
    factory_string = f"OPQ96,IVF{n_clusters},PQ96"
    train_size = TRAIN_SIZE_MULTIPLE * n_clusters

    train, queries = splits(dataset, train_size, args.queries)
    ground_truth = make_ground_truth(
        dataset,
        queries,
        normalize,
        args.batch_size,
        args.intersection,
        args.inner_product,
        cache_dir,
        n_tasks,
        progress,
    )

    train_memmap = create_memmap(
        train, dimensions, normalize, cache_dir, args.batch_size, n_tasks, progress
    )

    faiss_index = train_index(train_memmap, factory_string, args.inner_product)

    with TemporaryDirectory(dir=cache_dir) as tmpdir:
        with queries.formatted_as("numpy"):
            q_ids: npt.NDArray = queries["ids"]  # type: ignore
        index = fill_index(
            Path(tmpdir) / "index.faiss",
            Path(tmpdir) / "index.ivfdata",
            faiss_index,
            dataset,
            q_ids,
            dimensions,
            normalize,
            cache_dir,
            args.batch_size,
            args.shard_size,
            n_tasks,
            progress,
        )
        optimal_params = tune_index(
            index, ground_truth, dimensions, normalize, args.intersection, progress
        )

    dest.mkdir()
    try:
        save_ids(dest / "ids.parquet", dataset, args.batch_size)
        save_optimal_params(dest / "params.json", dimensions, normalize, optimal_params)
        fill_index(
            dest / "index.faiss",
            dest / "index.ivfdata",
            faiss_index,
            dataset,
            None,
            dimensions,
            normalize,
            cache_dir,
            args.batch_size,
            args.shard_size,
            n_tasks,
            progress,
        )
    except (KeyboardInterrupt, Exception):
        rmtree(dest)
        raise


if __name__ == "__main__":
    main()
