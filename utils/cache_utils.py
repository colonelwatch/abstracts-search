from pathlib import Path
from shutil import rmtree
from sys import stderr
from tempfile import TemporaryDirectory
from threading import Lock

from datasets import Dataset, disable_caching
from datasets.config import HF_DATASETS_CACHE

from .env_utils import CACHE

_cache_dir: Path = CACHE
_tmpdir: TemporaryDirectory | None = None
_lock = Lock()


def _get_cache_dir_nolock() -> Path:
    _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


def get_cache_dir() -> Path:
    with _lock:
        return _get_cache_dir_nolock()


def seal_persistent_cache() -> None:
    global _tmpdir, _cache_dir
    with _lock:
        # replace with a TemporaryDirectory (cleanup upon interpreter exit)
        cache_dir = _get_cache_dir_nolock()
        if cache_dir != CACHE:
            return
        _tmpdir = TemporaryDirectory(dir=str(cache_dir))
        _cache_dir = Path(_tmpdir.name)


def clean_persistent_cache() -> None:
    rmtree(CACHE)


def seal_hf_cache() -> None:
    disable_caching()


def clean_hf_cache(dataset: Dataset):
    # get cache directory path by following the path to an individual cache file
    file_0_path = Path(dataset.cache_files[0]["filename"])
    del dataset

    # parts[0] -> dataset ("parquet" by default)
    # parts[1] -> cache (pseudorandom, seeded with stuff like file metadata)
    # since this is a low-level detail, sanity-check the above facts for change
    file_0_path_rel = file_0_path.relative_to(HF_DATASETS_CACHE)
    dataset_name = file_0_path_rel.parts[0]
    cache_name = file_0_path_rel.parts[1]
    if not (dataset_name == "parquet" and "default-" in cache_name):
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
