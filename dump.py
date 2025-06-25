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

import sqlite3
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import rmtree
from sys import stderr
from typing import Generator, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from utils.env_utils import BF16
from utils.table_utils import (
    VectorConverter,
    create_embeddings_table,
    insert_embeddings,
    to_sql_binary,
)


def parse_args() -> Namespace:
    parser = ArgumentParser("dump.py", description="Dumps embeddings to parquet files.")
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("-s", "--shard-size", default=4194304, type=int)  # under 4GB
    parser.add_argument("--row-group-size", default=262144, type=int)  # around 128MB
    parser.add_argument(
        "--no-enforce-dtype", action="store_false", dest="enforce_dtype"
    )
    return parser.parse_args()


def to_arrays(
    ids: list[str], embeddings: list[npt.NDArray]
) -> tuple[pa.Array, pa.Array]:
    dim = embeddings[0].shape[0]
    flattened = np.hstack(embeddings)
    embeddings_arr = pa.FixedSizeListArray.from_arrays(flattened, dim)
    ids_arr = pa.array(ids, pa.string())
    return ids_arr, embeddings_arr


def to_chunks(
    dataset: sqlite3.Cursor, size: int
) -> Generator[tuple[pa.Array, pa.Array], None, None]:
    ids_batch: list[str] = []
    embeddings_batch: list[npt.NDArray] = []
    for id_, embedding in dataset:
        ids_batch.append(id_)
        embeddings_batch.append(embedding)

        if len(ids_batch) >= size:
            yield to_arrays(ids_batch, embeddings_batch)
            ids_batch = []
            embeddings_batch = []

    if ids_batch:
        yield to_arrays(ids_batch, embeddings_batch)


def open_parquet(path: str | Path, dim: int, bf16: bool) -> pq.ParquetWriter:
    schema = {"id": pa.string()}
    if bf16:
        # the conversion from bfloat16 to float32 leaves 16 bits of mantissa which are
        # completely zero. Exploit this with byte-stream split and lz4 compression
        schema["embedding"] = pa.list_(pa.float32(), dim)
        writer = pq.ParquetWriter(
            str(path),
            pa.schema(schema),
            compression="lz4",
            use_byte_stream_split=["embedding"],  # type: ignore (documented option)
        )
    else:
        # otherwise, compressing float embeddings isn't worth it
        schema["embedding"] = pa.list_(pa.float16(), dim)
        writer = pq.ParquetWriter(str(path), pa.schema(schema), compression="none")
    return writer


def write_to_parquet(
    ids_chunk: pa.Array, embd_chunk: pa.Array, writer: pq.ParquetWriter
) -> None:
    batch = pa.table([ids_chunk, embd_chunk], schema=writer.schema)
    writer.write_table(batch, row_group_size=len(ids_chunk))


def dump_database(
    source: Path,
    dest: Path,
    shard_size: int,
    row_group_size: int,
    enforce: Literal["bf16", "fp16"] | None = None,
):
    if not (source.suffix == ".sqlite" and dest.suffix == ""):
        raise ValueError("invalid source and dest types")

    # detect the type used in the database
    with sqlite3.connect(source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        (dtype,) = conn.execute(
            "SELECT value FROM properties where key = 'dtype'"
        ).fetchone()
        if dtype == "bf16":
            bf16 = True
        elif dtype == "fp16":
            bf16 = False
        else:
            raise ValueError("database contains an invalid dtype value")

    # VectorConverter does torch.bfloat16 to np.float32
    to_dtype = "fp32" if enforce == "bf16" else enforce
    converter = VectorConverter(bf16, to_dtype)
    sqlite3.register_converter("vector", converter.from_sql_binary)

    # To save RAM, push chunks of row_group_size into shards of shard_size one-by-one
    with sqlite3.connect(source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        # get the dimension by querying the first row and checking its length
        embedding = conn.execute("SELECT embedding FROM embeddings LIMIT 1")
        dim = embedding.fetchone()[0].shape[0]

        # iterate through this massive query in chunks
        cursor = conn.execute("SELECT * FROM embeddings")
        chunks = to_chunks(cursor, row_group_size)

        id_ = 0  # shard id
        counter = 0  # the number of rows the current shard will have
        shard = open_parquet(dest / f"data_{id_:03}.parquet", dim, bf16)
        for ids_chunk, embd_chunk in chunks:
            # start by assuming this shard will get the whole chunk
            counter += len(ids_chunk)

            # open new shard(s) and write so that the remainder fits in one shard
            while counter >= shard_size:
                excess = counter - shard_size

                cutoff = len(ids_chunk) - excess  # != shard_size perhaps only at first
                write_to_parquet(ids_chunk[:cutoff], embd_chunk[:cutoff], shard)
                ids_chunk = ids_chunk[cutoff:]
                embd_chunk = embd_chunk[cutoff:]

                id_ += 1
                counter = excess
                shard = open_parquet(dest / f"data_{id_:03}.parquet", dim, bf16)

            if counter:  # if counter didn't happen to be a multiple of shard_size
                write_to_parquet(ids_chunk, embd_chunk, shard)


def dump_dataset(
    source: Path,
    dest: Path,
    batch_size: int,
    enforce: Literal["bf16", "fp16"] | None = None,
) -> ds.Dataset:
    if not (source.suffix == "" and dest.suffix == ".sqlite"):
        raise ValueError("invalid source and dest types")

    paths = [str(path) for path in source.glob("*.parquet")]
    dataset: ds.Dataset = ds.dataset(paths)

    # extract the vector dtype and length from the schema
    embeddings_col_type = dataset.schema.field("embedding").type
    dtype = embeddings_col_type.value_type
    length = embeddings_col_type.list_size  # poorly documented!

    if not enforce:
        if dtype == pa.float32():
            bf16 = True  # assume this was converted from bfloat16
        elif dtype == pa.float16():
            bf16 = False
        else:
            raise ValueError(f'invalid embeddings type "{dtype}"')
    else:
        bf16 = True if enforce == "bf16" else False

    sqlite3.register_adapter(torch.Tensor, to_sql_binary)
    with sqlite3.connect(dest) as conn:
        create_embeddings_table(conn, bf16)
        for batch in dataset.to_batches(batch_size=batch_size):
            embeddings_np: npt.NDArray = (  # this makes the conversion zero-copy
                batch["embedding"].flatten().to_numpy().reshape((-1, length))
            )
            embeddings = torch.from_numpy(embeddings_np.copy())  # no read-only memory
            if bf16:
                embeddings = embeddings.bfloat16()
            insert_embeddings(batch["id"].to_pylist(), embeddings, conn)


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

    if args.enforce_dtype:
        enforce = "bf16" if BF16 else "fp16"
    else:
        enforce = None

    if source.suffix == ".sqlite" and dest.suffix == "":
        dest.mkdir()
        try:
            dump_database(source, dest, args.shard_size, args.row_group_size, enforce)
        except (KeyboardInterrupt, Exception):
            rmtree(dest)
            raise
    elif source.suffix == "" and dest.suffix == ".sqlite":
        try:
            dump_dataset(source, dest, args.batch_size, enforce)
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
