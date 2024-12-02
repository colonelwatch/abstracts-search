# encode.py  TODO: rename to dump.py?

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
from pathlib import Path
import sqlite3
from typing import Generator

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def parse_args() -> Namespace:
    parser = ArgumentParser("encode.py", description="Repartitions the embeddings.")
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("-s", "--shard-size", default=4194304)  # 2^22 is under 4GB
    parser.add_argument("--fp16", action="store_false", dest="bf16")  # fp16 or bf16
    return parser.parse_args()


class VectorConverter:
    def __init__(self, bf16: bool):
        self.bf16 = bf16  # else fp16

    def from_sql_binary(self, val: bytes) -> npt.NDArray:
        if self.bf16:  # do bf16 -> fp32 (TODO: do this with pure numpy code?)
            arr = np.frombuffer(val, dtype=np.uint16)
            t = torch.tensor(arr.copy())  # PyTorch complains about read-only memory
            arr = t.view(torch.bfloat16).float().numpy()
        else:
            arr = np.frombuffer(val, dtype=np.float16)
        return arr


def to_arrays(
    idxs: list[str], embeddings: list[npt.NDArray]
) -> tuple[pa.Array, pa.Array]:
    dim = embeddings[0].shape[0]
    flattened = np.hstack(embeddings)
    embeddings_arr = pa.FixedSizeListArray.from_arrays(flattened, dim)
    idxs_arr = pa.array(idxs, pa.string())
    return idxs_arr, embeddings_arr


def exact_shards(
    dataset: sqlite3.Cursor,
    size: int
) -> Generator[tuple[pa.Array, pa.Array], None, None]:
    idxs_batch: list[str] = []
    embeddings_batch: list[npt.NDArray] = []
    for idx, embedding in dataset:
        idxs_batch.append(idx)
        embeddings_batch.append(embedding)

        if len(idxs_batch) >= size:
            yield to_arrays(idxs_batch, embeddings_batch)
            idxs_batch = []
            embeddings_batch = []

    if idxs_batch:
        yield to_arrays(idxs_batch, embeddings_batch)


def write_table(shard: pa.Table, path: str | Path, bf16: bool) -> None:
    if bf16:
        # the conversion from bfloat16 to float32 leaves 16 bits of mantissa which are
        # completely zero. Exploit this with byte-stream split and lz4 compression
        pq.write_table(
            shard,
            str(path),
            compression="lz4",
            use_byte_stream_split=["embeddings"]  # type: ignore (documented option)
        )
    else:
        # otherwise, compressing float embeddings isn't worth it
        pq.write_table(shard, str(path), compression="none")


def main():
    args = parse_args()

    dest: Path = args.dest
    dest.mkdir()

    converter = VectorConverter(args.bf16)
    sqlite3.register_converter("vector", converter.from_sql_binary)
    with sqlite3.connect(args.source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.execute("SELECT * FROM embeddings")
        shards = exact_shards(cursor, args.shard_size)
        for id_, (idxs_shard, embd_shard) in enumerate(shards):
            shard = pa.table([idxs_shard, embd_shard], names=["idxs", "embeddings"])
            write_table(shard, dest / f"data_{id_:03}.parquet", args.bf16)


if __name__ == "__main__":
    main()
