import os
from pathlib import Path
from typing import Callable


def get_env_var[T, U](
    key: str, type_: Callable[[str], T] = str, default: U = None
) -> T | U:
    var = os.getenv(key)
    if var is not None:
        var = type_(var)
    else:
        var = default
    return var


def bool_from_str(value: str) -> bool:
    if value in ["0", "false", "False", "FALSE"]:
        return False
    elif value in ["1", "true", "True", "TRUE"]:
        return True
    else:
        raise ValueError(f'invalid value "{value}", expected bool-like')


CACHE = get_env_var("ABSEARCH_CACHE", Path, Path.home() / ".cache/abstracts-search")
MODEL = get_env_var("ABSEARCH_MODEL", default="all-MiniLM-L6-v2")
TRUST_REMOTE_CODE = get_env_var("ABSEARCH_TRUST_REMOTE_CODE", bool_from_str, False)
FP16 = get_env_var("ABSEARCH_FP16", bool_from_str, False)
BF16 = not FP16  # support only bf16 or fp16 for simplicity
