# encode.py

# Copyright 2023 Kenny Peng
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
import os
from typing import Generator

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


def parse_args() -> Namespace:
    parser = ArgumentParser("encode.py", description="Repartitions the embeddings.")
    parser.add_argument("source")
    parser.add_argument("dest")
    parser.add_argument("-s", "--shard-size", default=4194304)  # 2^22 is under 4GB
    parser.add_argument("--fp16", action="store_false", dest="bf16")  # fp16 or bf16
    return parser.parse_args()


def exact_shards(dataset: ds.Dataset, size: int) -> Generator[pa.Array, None, None]:
    counter = 0
    idxs_batches: list[pa.RecordBatch] = []
    embeddings_batches: list[pa.RecordBatch] = []
    for batch in dataset.to_batches():
        idxs_batches.append(batch["idxs"])
        embeddings_batches.append(batch["embeddings"])
        counter += len(batch)

        while counter >= size:
            idxs_concat = pa.concat_arrays(idxs_batches)
            embeddings_concat = pa.concat_arrays(embeddings_batches)
            yield idxs_concat[:size], embeddings_concat[:size]

            idxs_batches = [idxs_concat[size:]]
            embeddings_batches = [embeddings_concat[size:]]
            counter -= size

    if counter:
        idxs_concat = pa.concat_arrays(idxs_batches)
        embeddings_concat = pa.concat_arrays(embeddings_batches)
        yield idxs_concat, embeddings_concat


def write_table(shard: pa.Table, path: str, bf16: bool) -> None:
    if bf16:
        pq.write_table(
            shard,
            path,
            compression="lz4",
            use_byte_stream_split=["embeddings"]  # type: ignore (documented option)
        )
    else:
        pq.write_table(shard, path, compression="none")


def main():
    args = parse_args()
    d = ds.dataset(args.source)
    os.mkdir(args.dest)
    for id_, (idxs_shard, embd_shard) in enumerate(exact_shards(d, args.shard_size)):
        shard = pa.table([idxs_shard, embd_shard], names=["idxs", "embeddings"])
        write_table(shard, f"{args.dest}/data_{id_:03}.parquet", args.bf16)


if __name__ == "__main__":
    main()
