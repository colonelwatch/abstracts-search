import sqlite3
from typing import Iterable, Literal

import numpy as np
import numpy.typing as npt
import torch

from .env_utils import BF16


class VectorConverter:
    def __init__(self, bf16: bool, to_dtype: Literal["fp32", "fp16"] | None):
        self.bf16 = bf16  # else fp16
        self.to_dtype = to_dtype

    def from_sql_binary(self, val: bytes) -> npt.NDArray:
        if self.bf16:   # do bf16 -> fp32/16 (TODO: do this with pure numpy code?)
            arr = np.frombuffer(val, dtype=np.uint16)
            t = torch.tensor(arr.copy())  # PyTorch complains about read-only memory
            arr = t.view(torch.bfloat16)
            arr = t.half() if self.to_dtype == "fp16" else t.float()
            arr = t.numpy()
        else:
            arr = np.frombuffer(val, dtype=np.float16)
            if self.to_dtype == "fp32":
                arr = arr.astype(np.float32)
        return arr


def to_sql_binary(vect: torch.Tensor) -> sqlite3.Binary:
    if vect.dtype == torch.bfloat16:
        vect = vect.view(torch.uint16)
    return vect.numpy().data


def create_embeddings_table(conn: sqlite3.Connection, bf16: bool):
    conn.execute(
        "CREATE TABLE embeddings(id TEXT PRIMARY KEY, embedding vector)"
    )
    conn.execute("CREATE TABLE properties(key TEXT, value TEXT)")
    conn.execute(
        "INSERT INTO properties VALUES(?, ?)", ("dtype", "bf16" if bf16 else "fp16")
    )


def insert_embeddings(
    oa_ids: Iterable[str], embeddings: Iterable[torch.Tensor], conn: sqlite3.Connection
):
    conn.executemany(
        "INSERT INTO embeddings VALUES(?, ?) "
        "ON CONFLICT(id) DO UPDATE SET embedding=excluded.embedding",
        zip(oa_ids, embeddings)
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from sys import stderr

    parser = argparse.ArgumentParser("table_utils.py", "Init the embeddings table.")
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    target: Path = args.target
    if target.exists():
        print("error: target already exists", file=stderr)

    with sqlite3.connect(target) as conn:
        create_embeddings_table(conn, BF16)
