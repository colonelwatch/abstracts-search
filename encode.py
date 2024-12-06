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
from shutil import rmtree
import sqlite3
from sys import stderr
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
    parser.add_argument("-s", "--shard-size", default=4194304, type=int)  # under 4GB
    parser.add_argument("--row-group-size", default=1048576, type=int)  # 1024 * 1024
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


def to_chunks(
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


def open_parquet(path: str | Path, dim: int, bf16: bool) -> pq.ParquetWriter:
    schema = {"idxs": pa.string()}
    if bf16:
        # the conversion from bfloat16 to float32 leaves 16 bits of mantissa which are
        # completely zero. Exploit this with byte-stream split and lz4 compression
        schema["embeddings"] = pa.list_(pa.float32(), dim)
        writer = pq.ParquetWriter(
            str(path),
            pa.schema(schema),
            compression="lz4",
            use_byte_stream_split=["embeddings"]  # type: ignore (documented option)
        )
    else:
        # otherwise, compressing float embeddings isn't worth it
        schema["embeddings"] = pa.list_(pa.float16(), dim)
        writer = pq.ParquetWriter(str(path), pa.schema(schema), compression="none")
    return writer


def write_to_parquet(
    idxs_chunk: pa.Array, embd_chunk: pa.Array, writer: pq.ParquetWriter
) -> None:
    batch = pa.table([idxs_chunk, embd_chunk], schema=writer.schema)
    writer.write_table(batch, row_group_size=len(idxs_chunk))


def dump_database(
    source: Path,
    dest: Path,
    shard_size: int,
    row_group_size: int,
    bf16: bool,
):
    converter = VectorConverter(bf16)
    sqlite3.register_converter("vector", converter.from_sql_binary)
    with sqlite3.connect(source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.execute("SELECT * FROM embeddings")
        chunks = to_chunks(cursor, row_group_size)

        embedding = conn.execute("SELECT embedding FROM embeddings").fetchone()[0]
        dim = embedding.shape[0]

        id_ = 0
        counter = 0
        shard = open_parquet(dest / f"data_{id_:03}.parquet", dim, bf16)
        for idxs_chunk, embd_chunk in chunks:
            counter += len(idxs_chunk)

            while counter >= shard_size:
                excess = counter - shard_size

                cutoff = len(idxs_chunk) - excess
                write_to_parquet(idxs_chunk[:cutoff], embd_chunk[:cutoff], shard)
                idxs_chunk = idxs_chunk[cutoff:]
                embd_chunk = embd_chunk[cutoff:]

                id_ += 1
                counter = excess
                shard = open_parquet(dest / f"data_{id_:03}.parquet", dim, bf16)

            if counter:
                write_to_parquet(idxs_chunk, embd_chunk, shard)


def main() -> int:
    args = parse_args()

    source: Path = args.source
    if not source.exists():
        print(f'error: source path "{source}" does not exist', file=stderr)
        return 1

    dest: Path = args.dest
    if dest.exists():
        print(f'error: destination path "{dest}" exists', file=stderr)
        return 1
    dest.mkdir()

    try:
        dump_database(source, dest, args.shard_size, args.row_group_size, args.bf16)
    except KeyboardInterrupt:
        rmtree(dest)
        raise

    return 0


if __name__ == "__main__":
    ret = main()
    exit(ret)
