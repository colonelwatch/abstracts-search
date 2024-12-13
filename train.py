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
import json
import logging
from pathlib import Path
from shutil import rmtree, move
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

from datasets import Dataset, disable_progress_bars
from datasets.fingerprint import Hasher
import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
from faiss.contrib.ondisk import merge_ondisk
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks

logger = logging.getLogger(__name__)


class IndexParameters(TypedDict):
    recall: float  # in this case 10-recall@10
    exec_time: float  # seconds (raw faiss measure is in milliseconds)
    param_string: str  # pass directly to faiss index


def parse_args() -> Namespace:
    parser = ArgumentParser("train.py", "Trains the FAISS index.")
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("-w", "--working-dir", default="splits", type=Path)
    parser.add_argument("-N", "--normalize", action="store_true")
    parser.add_argument("-i", "--inner-product", action="store_true")
    parser.add_argument("-k", "--intersection", default=10, type=int)
    parser.add_argument("-c", "--clusters", default=None, type=int)  # TODO: raise
    parser.add_argument("-q", "--queries", default=16384, type=int)
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("-P", "--progress", action="store_true")
    parser.add_argument("-t", "--truncate", default=None, type=int)
    parser.add_argument("--shard-size", default=4194304, type=int)
    return parser.parse_args()


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in dir.glob("*.parquet")]
    dataset: Dataset = Dataset.from_parquet(paths)  # type: ignore
    return dataset


def add_id_column(dataset: Dataset) -> Dataset:
    ids = np.arange(len(dataset), dtype=np.int32)
    return dataset.add_column("ids", ids)  # type: ignore (new_fingerprint not required)


def splits(
    dataset: Dataset,
    train_size: int,
    test_size: int,
    seed: int = 42
) -> tuple[Dataset, Dataset]:
    # TODO: investigate whether taking queries from the end of the shuffle is
    # significantly slower than `train_test_split` or not
    splits = dataset.train_test_split(test_size, train_size, seed=seed)
    return splits["train"], splits["test"]  # result: two sets of indices, one file


def create_memmap(
    working_dir: Path,
    dataset: Dataset,
    normalize: bool,
    batch_size: int,
    progress: bool
) -> np.memmap[Any, np.dtype[np.float32]]:
    n = len(dataset)
    d = len(dataset[0]["embeddings"])
    shape = (n, d)

    cache_path = working_dir / f"train_{dataset._fingerprint}.memmap"
    if cache_path.exists():
        return np.memmap(cache_path, np.float32, mode="r", shape=shape)

    memmap = np.memmap(cache_path, np.float32, mode="w+", shape=shape)
    try:
        with (
            dataset.formatted_as("torch", columns=["embeddings"]),
            tqdm(total=len(dataset), disable=(not progress)) as counter
        ):
            i = 0
            for batch in dataset.iter(batch_size):
                embeddings_batch: torch.Tensor = batch["embeddings"]  # type: ignore
                n_batch = len(embeddings_batch)

                embeddings_batch = embeddings_batch.cuda(non_blocking=True)

                if normalize:
                    embeddings_batch = torch.nn.functional.normalize(embeddings_batch)

                memmap[i:(i + n_batch)] = embeddings_batch.cpu().numpy()
                i += n_batch

                counter.update(n_batch)
    except KeyboardInterrupt:
        cache_path.unlink()
        raise

    memmap.flush()
    return np.memmap(cache_path, np.float32, mode="r", shape=shape)


# TODO: multithread?
def make_ground_truth(
    working_dir: Path,
    dataset: Dataset,
    queries: Dataset,
    normalize: bool,
    batch_size: int,
    k: int,
    inner_product: bool,
    progress: bool,
) -> Dataset:
    h = Hasher()
    h.update(dataset._fingerprint)
    h.update(queries._fingerprint)
    h.update(normalize)
    h.update(batch_size)
    h.update(k)
    h.update(inner_product)
    cache_identifier = h.hexdigest()
    cache_path = working_dir / f"gt_{cache_identifier}"
    if cache_path.exists():
        return Dataset.load_from_disk(cache_path)

    with queries.formatted_as("torch", columns=["embeddings", "ids"], device="cuda"):
        q: torch.Tensor = queries["embeddings"]  # type: ignore
        q_ids: torch.Tensor = queries["ids"]  # type: ignore

    if normalize:
        q = torch.nn.functional.normalize(q)

    n_q, _ = q.shape
    gt_ids = torch.full((n_q, k), -1, dtype=torch.int32, device="cuda")
    if inner_product:
        gt_scores = torch.zeros((n_q, k), dtype=torch.float32, device="cuda")
    else:
        gt_scores = torch.full((n_q, k), torch.inf, dtype=torch.float32, device="cuda")

    with (
        dataset.formatted_as("torch", columns=["embeddings", "ids"]),
        tqdm(total=(len(dataset) - len(queries)), disable=(not progress)) as counter,
    ):
        for batch in dataset.iter(batch_size):
            d_batch: torch.Tensor = batch["embeddings"]  # type: ignore
            d_batch_ids: torch.Tensor = batch["ids"]  # type: ignore

            d_batch = d_batch.cuda(non_blocking=True)
            d_batch_ids = d_batch_ids.cuda(non_blocking=True)

            not_in_queries = torch.isin(
                d_batch_ids, q_ids, assume_unique=True, invert=True
            )
            d_batch = d_batch[not_in_queries]
            d_batch_ids = d_batch_ids[not_in_queries]

            if normalize:
                d_batch = torch.nn.functional.normalize(d_batch)

            if inner_product:
                scores_batch = q @ d_batch.T
                largest = True
            else:
                # prefer direct calc over following the quadratic form with matmult
                scores_batch = torch.cdist(
                    q, d_batch, compute_mode="donot_use_mm_for_euclid_dist"
                )
                largest = False

            top_scores_batch, argtop = torch.topk(scores_batch, k, 1, largest)
            top_ids_batch = d_batch_ids[argtop.flatten()].reshape(argtop.shape)

            gt_scores = torch.hstack((gt_scores, top_scores_batch))
            gt_ids = torch.hstack((gt_ids, top_ids_batch))
            gt_scores, argtop = torch.topk(gt_scores, k, 1, largest)
            gt_ids = torch.gather(gt_ids, 1, argtop)

            counter.update(len(d_batch))

    ground_truth = Dataset.from_dict(
        {
            "embeddings": queries["embeddings"],
            "gt_ids": gt_ids.cpu().numpy().astype(np.int64),
        }
    )
    ground_truth.save_to_disk(cache_path)
    return ground_truth


def to_gpu(index: faiss.Index) -> faiss.Index:
    opts = faiss.GpuClonerOptions()
    opts.useFloat16 = True  # float16 is necessary for codes sized 56 bits and over
    env = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(env, 0, index, opts)


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
    index_dir: Path,
    index_filename: str,
    vectors_filename: str,
    dataset: Dataset,
    trained_index: faiss.Index,
    batch_size: int,
    shard_size: int,
    holdout_ids: npt.NDArray | None = None,
) -> faiss.Index:
    with TemporaryDirectory() as tmpdir:
        chunk_paths: list[Path] = []
        on_gpu = to_gpu(trained_index)
        for i_shard, shard_start in enumerate(range(0, len(dataset), shard_size)):
            shard = dataset.select(
                range(shard_start, min(shard_start + shard_size, len(dataset)))
            )

            for batch_start in range(0, len(shard), batch_size):
                batch = shard.select(
                    range(batch_start, min(batch_start + batch_size, len(shard)))
                )
                with batch.formatted_as("numpy"):
                    batch_ids: npt.NDArray = batch["ids"]  # type: ignore
                    batch_embeddings: npt.NDArray = batch["embeddings"]  # type: ignore

                # TODO: is this slower than precalculating a mask?
                if holdout_ids is not None:
                    not_in_test_set = np.isin(
                        batch_ids, holdout_ids, assume_unique=True, invert=True
                    )
                    batch_ids = batch_ids[not_in_test_set]
                    batch_embeddings = batch_embeddings[not_in_test_set]

                on_gpu.add_with_ids(batch_embeddings, batch_ids)  # type: ignore

            shard_index = to_cpu(on_gpu)
            path = Path(tmpdir) / f"index_{i_shard:03d}.faiss'"
            faiss.write_index(shard_index, str(path))

            chunk_paths.append(path)
            on_gpu.reset()

        # TODO: investigate why I build in current working directory first
        temp_vectors_path = Path(vectors_filename)
        vectors_path = index_dir / vectors_filename
        index_path = index_dir / index_filename
        try:
            index = faiss.clone_index(trained_index)
            merge_ondisk(index, [str(p) for p in chunk_paths], str(temp_vectors_path))
            move(temp_vectors_path, vectors_path)
            faiss.write_index(index, str(index_path))
        except KeyboardInterrupt:
            temp_vectors_path.unlink(missing_ok=True)
            vectors_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
            raise

    return index


def tune_index(
    filled_index: faiss.Index, ground_truth: Dataset, k: int, progress: bool
) -> list[IndexParameters]:
    with ground_truth.formatted_as("numpy"):
        q: npt.NDArray[np.float32] = ground_truth["embeddings"]  # type: ignore
        gt_ids: npt.NDArray[np.int64] = ground_truth["gt_ids"]  # type: ignore

    # init with ground-truth IDs but not ground-truth distances because faiss doesn't
    # use them anyway (see faiss/AutoTune.cpp)
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


def save_optimal_params(path: Path, optimal_params: list[IndexParameters]):
    with open(path, "w") as f:
        json.dump(optimal_params, f, indent=4)


def main():
    args = parse_args()

    source: Path = args.source
    if not source.exists():
        print(f'error: source path "{source}" does not exist', file=stderr)
        return 1

    dest: Path = args.dest
    if dest.exists():
        print(f'error: destination path "{dest}" exists', file=stderr)
        return 1

    working_dir: Path = args.working_dir
    working_dir.mkdir(exist_ok=True)

    progress: bool = args.progress
    if not progress:
        disable_progress_bars()

    dataset = load_dataset(source)
    dataset = add_id_column(dataset)
    truncate: int | None = args.truncate
    if truncate is not None:
        dataset = dataset.take(truncate)

    n_clusters: int | None = args.clusters
    n_queries: int = args.queries
    if n_clusters is None:
        n_clusters = (len(dataset) - n_queries) // TRAIN_SIZE_MULTIPLE
    factory_string = f"OPQ64_256,IVF{n_clusters},PQ64"
    train_size = TRAIN_SIZE_MULTIPLE * n_clusters

    train, queries = splits(dataset, train_size, args.queries)
    ground_truth = make_ground_truth(
        working_dir,
        dataset,
        queries,
        args.normalize,
        args.batch_size,
        args.intersection,
        args.inner_product,
        progress,
    )

    train_memmap = create_memmap(
        working_dir, train, args.normalize, args.batch_size, progress
    )

    faiss_index = train_index(train_memmap, factory_string, args.inner_product)

    with TemporaryDirectory() as tmpdir:
        with queries.formatted_as("numpy"):
            q_ids: npt.NDArray = queries["ids"]  # type: ignore
        index = fill_index(
            Path(tmpdir),
            "index.faiss",
            "index.ivfdata",
            dataset,
            faiss_index,
            args.batch_size,
            args.shard_size,
            holdout_ids=q_ids
        )
        optimal_params = tune_index(index, ground_truth, args.intersection, progress)

    dest.mkdir()
    try:
        save_ids(dest / "ids.parquet", dataset, args.batch_size)
        save_optimal_params(dest / "params.json", optimal_params)
        fill_index(
            dest,
            "index.faiss",
            "index.ivfdata",
            dataset,
            faiss_index,
            args.batch_size,
            args.shard_size,
        )
    except KeyboardInterrupt:
        rmtree(dest)
        raise


if __name__ == "__main__":
    main()
