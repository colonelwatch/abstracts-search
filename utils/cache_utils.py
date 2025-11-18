from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from .env_utils import CACHE

_cache_dir: Path = CACHE
_tmpdir: TemporaryDirectory | None = None
_lock = Lock()


def seal_persistent_cache() -> None:
    global _tmpdir, _cache_dir
    with _lock:
        # replace with a TemporaryDirectory (cleanup upon interpreter exit)
        cache_dir = _get_cache_dir_nolock()
        if cache_dir != CACHE:
            return
        _tmpdir = TemporaryDirectory(dir=str(cache_dir))
        _cache_dir = Path(_tmpdir.name)


def get_cache_dir() -> Path:
    with _lock:
        return _get_cache_dir_nolock()


def _get_cache_dir_nolock() -> Path:
    _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir
