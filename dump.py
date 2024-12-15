# dump.py

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
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from utils.table_utils import (
    create_embeddings_table, insert_embeddings, to_sql_binary, VectorConverter
)


def parse_args() -> Namespace:
    parser = ArgumentParser("dump.py", description="Dumps embeddings to parquet files.")
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("-s", "--shard-size", default=4194304, type=int)  # under 4GB
    parser.add_argument("--row-group-size", default=1048576, type=int)  # 1024 * 1024
    parser.add_argument("--fp16", action="store_false", dest="bf16")  # fp16 or bf16
    return parser.parse_args()


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
    if not (source.suffix == ".sqlite" and dest.suffix == ""):
        raise ValueError("invalid source and dest types")

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


def dump_dataset(source: Path, dest: Path, batch_size: int) -> ds.Dataset:
    if not (source.suffix == "" and dest.suffix == ".sqlite"):
        raise ValueError("invalid source and dest types")

    paths = [str(path) for path in source.glob("*.parquet")]
    dataset: ds.Dataset = ds.dataset(paths)

    embeddings_col_type = dataset.schema.field("embeddings").type
    length = embeddings_col_type.list_size  # poorly documented!
    dtype = embeddings_col_type.value_type
    if dtype == pa.float32():
        bf16 = True
    elif dtype == pa.float16():
        bf16 = False
    else:
        raise ValueError(f'invalid embeddings type "{dtype}"')

    sqlite3.register_adapter(torch.Tensor, to_sql_binary)
    with sqlite3.connect(dest) as conn:
        create_embeddings_table(conn)
        for batch in dataset.to_batches(batch_size=batch_size):
            embeddings_np: npt.NDArray = (
                batch["embeddings"].flatten().to_numpy().reshape((-1, length))
            )
            embeddings = torch.from_numpy(embeddings_np.copy())  # no read-only memory
            if bf16:
                embeddings = embeddings.bfloat16()
            insert_embeddings(batch["idxs"].to_pylist(), embeddings, conn)


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

    if source.suffix == ".sqlite" and dest.suffix == "":
        dest.mkdir()
        try:
            dump_database(source, dest, args.shard_size, args.row_group_size, args.bf16)
        except (KeyboardInterrupt, Exception):
            rmtree(dest)
            raise
    elif source.suffix == "" and dest.suffix == ".sqlite":
        try:
            dump_dataset(source, dest, args.batch_size)
        except (KeyboardInterrupt, Exception):
            dest.unlink()
            raise
    else:
        print("error: invalid source and destination types", file=stderr)
        return 1

    return 0


if __name__ == "__main__":
    ret = main()
    exit(ret)
