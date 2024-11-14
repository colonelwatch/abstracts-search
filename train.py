from argparse import ArgumentParser, Namespace
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from datasets import Dataset
from datasets.fingerprint import Hasher
import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

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
    parser.add_argument("-i", "--inner-product", action="store_true")
    parser.add_argument("-k", "--intersection", default=10, type=int)
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("--train-size", default=6291456, type=int)  # hangups past this?
    parser.add_argument("--test-size", default=16384, type=int)
    parser.add_argument("--validation-size", default=None)
    parser.add_argument("--truncate", default=None, type=int)
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
    validation_size: int | None,
    seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:
    if validation_size is None:
        validation_size = test_size
    total_test_size = validation_size + test_size
    splits_0 = dataset.train_test_split(total_test_size, train_size, seed=seed)
    splits_1 = splits_0["test"].train_test_split(test_size, validation_size, seed=seed)

    train = splits_0["train"]
    validation = splits_1["train"]
    test = splits_1["test"]

    # result: three sets of indices, indexing one underlying file
    return train, validation, test


def create_memmap(
    path: Path, dataset: Dataset, batch_size: int
) -> np.memmap[Any, np.dtype[np.float32]]:
    n = len(dataset)
    d = len(dataset[0]["embeddings"])
    memmap = np.memmap(path, np.float32, mode="w+", shape=(n, d))

    with dataset.formatted_as("numpy", columns=["embeddings"]):
        counter = 0
        for batch in dataset.iter(batch_size):
            embeddings_batch: npt.NDArray = batch["embeddings"]  # type: ignore

            n_batch = len(embeddings_batch)
            memmap[counter:(counter + n_batch)] = embeddings_batch
            counter += n_batch

    memmap.flush()
    return np.memmap(path, np.float32, mode="r", shape=(n, d))


# TODO: multithread?
def make_ground_truth(
    working_dir: Path, dataset: Dataset, queries: Dataset, batch_size: int, k: int
) -> Dataset:
    h = Hasher()
    h.update(dataset._fingerprint)
    h.update(queries._fingerprint)
    h.update(batch_size)
    h.update(k)
    cache_identifier = h.hexdigest()
    cache_path = working_dir / f"gt_{cache_identifier}"
    if cache_path.exists():
        return Dataset.load_from_disk(cache_path)

    queries._indices
    with queries.formatted_as("torch", columns=["embeddings", "ids"]):
        q: torch.Tensor = queries["embeddings"]  # type: ignore
        q_ids: torch.Tensor = queries["ids"]  # type: ignore

    q = q.cuda()
    q_ids = q_ids.cuda()

    n_q, _ = q.shape
    gt_scores = torch.zeros((n_q, k), dtype=torch.float32).cuda()
    gt_ids = torch.full((n_q, k), -1, dtype=torch.int32).cuda()
    with (
        dataset.formatted_as("torch", columns=["embeddings", "ids"]),
        tqdm(total=(len(dataset) - len(queries))) as counter,
    ):
        for batch in dataset.iter(batch_size):
            d_batch: torch.Tensor = batch["embeddings"]  # type: ignore
            d_batch_ids: torch.Tensor = batch["ids"]  # type: ignore

            d_batch = d_batch.cuda(non_blocking=True)
            d_batch_ids = d_batch_ids.cuda(non_blocking=True)

            not_in_queries = torch.isin(d_batch_ids, q_ids, invert=True)
            d_batch = d_batch[not_in_queries]
            d_batch_ids = d_batch_ids[not_in_queries]

            scores_batch = q @ d_batch.T

            top_scores_batch, argtop = torch.topk(scores_batch, k, dim=1)
            top_ids_batch = d_batch_ids[argtop.flatten()].reshape(argtop.shape)

            gt_scores = torch.hstack((gt_scores, top_scores_batch))
            gt_ids = torch.hstack((gt_ids, top_ids_batch))
            gt_scores, argtop = torch.topk(gt_scores, k, dim=1)
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


def train_index(
    embeddings: npt.NDArray[np.float32] | np.memmap[Any, np.dtype[np.float32]],
    factory_string: str,
    inner_product: bool,
) -> faiss.Index:
    _, d = embeddings.shape
    metric = faiss.METRIC_INNER_PRODUCT if inner_product else faiss.METRIC_L2
    index: faiss.Index = faiss.index_factory(d, factory_string, metric)

    gpu_env = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_env, 0, index)
    index.train(embeddings)  # type: ignore (monkey-patched)
    index = faiss.index_gpu_to_cpu(index)

    return index


def fill_index(
    dataset: Dataset, trained_index: faiss.Index, batch_size: int
) -> faiss.Index:
    # fill the index on the gpu, using the dataset, then restore the dataset's state
    gpu_env = faiss.StandardGpuResources()
    on_gpu = faiss.index_cpu_to_gpu(gpu_env, 0, trained_index)
    dataset.add_faiss_index(
        "embeddings", custom_index=on_gpu, batch_size=batch_size
    )
    dataset.drop_index("embeddings")
    return faiss.index_gpu_to_cpu(on_gpu)


def tune_index(
    filled_index: faiss.Index, ground_truth: Dataset, k: int
) -> list[IndexParameters]:
    with ground_truth.formatted_as("numpy"):
        q: npt.NDArray[np.float32] = ground_truth["embeddings"]  # type: ignore
        gt_ids: npt.NDArray[np.int64] = ground_truth["gt_ids"]  # type: ignore

    # init with ground-truth IDs but not ground-truth distances because faiss doesn't
    # use them anyway (see faiss/AutoTune.cpp)
    criterion = faiss.IntersectionCriterion(len(ground_truth), k)
    criterion.set_groundtruth(None, gt_ids)  # type: ignore (monkey-patched)

    params = faiss.ParameterSpace()
    params.verbose = False  # TODO: control verbosity
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


def save_index(path: Path, index: faiss.Index):
    faiss.write_index(index, str(path))


def save_optimal_params(path: Path, optimal_params: list[IndexParameters]):
    with open(path, "w") as f:
        json.dump(optimal_params, f, indent=4)


def main():
    args = parse_args()

    working_dir: Path = args.working_dir
    working_dir.mkdir(exist_ok=True)

    dataset = load_dataset(args.source)
    dataset = add_id_column(dataset)
    truncate: int | None = args.truncate
    if truncate is not None:
        dataset = dataset.take(truncate)

    # TODO: come up with some scheme for the test split?
    train, validation, _ = splits(
        dataset, args.train_size, args.test_size, args.validation_size
    )
    ground_truth = make_ground_truth(
        working_dir, dataset, validation, args.batch_size, args.intersection
    )

    train_memmap = create_memmap(working_dir / "train.memmap", train, args.batch_size)

    faiss_index = train_index(
        train_memmap, "OPQ64_256,IVF131072,PQ64", args.inner_product
    )
    # faiss_index = train_index(train_memmap, "OPQ32_128,IVF4096,PQ32", True)
    index = fill_index(dataset, faiss_index, args.batch_size)
    optimal_params = tune_index(index, ground_truth, args.intersection)

    save_ids(args.dest / "ids.parquet", dataset, args.batch_size)
    save_optimal_params(args.dest / "params.json", optimal_params)
    save_index(args.dest / "index.faiss", index)


if __name__ == "__main__":
    main()
