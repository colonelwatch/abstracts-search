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

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

SHARD_SIZE = 4194304  # (2^22), puts the shard size a bit under 4 GB


def parse_args() -> Namespace:
    parser = ArgumentParser("encode.py", description="Repartitions the embeddings.")
    parser.add_argument("old_path")
    parser.add_argument("new_path")
    parser.add_argument("-s", "--shard-size")
    return parser.parse_args()


def take_shard(
    idxs: list[pa.Array], embeddings: list[pa.Array], shards_path: str, shard_id: int
) -> tuple[list[pa.Array], list[pa.Array]]:
    idxs_concat = pa.concat_arrays(idxs)
    embeddings_concat = pa.concat_arrays(embeddings)

    # TODO: make the shard schema a literal
    shard = pa.table(
        [idxs_concat[:SHARD_SIZE], embeddings_concat[:SHARD_SIZE]],
        names=["idxs", "embeddings"]
    )
    pq.write_table(
        shard,
        f"{shards_path}/data_{shard_id:03}.parquet",
        compression="none"
    )

    return [idxs_concat[SHARD_SIZE:]], [embeddings_concat[SHARD_SIZE:]]


def main():
    args = parse_args()

    os.mkdir(args.new_path)

    # merge the partial works tables
    shard_id = 0
    idxs = []
    embeddings = []
    for batch in ds.dataset(args.old_path).to_batches():
        idxs.append(batch['idxs'])
        embeddings.append(batch['embeddings'])
        while len(idxs) >= SHARD_SIZE:
            idxs, embeddings = take_shard(idxs, embeddings, args.new_path, shard_id)
            shard_id += 1
    _, _ = take_shard(idxs, embeddings, args.new_path, shard_id)


if __name__ == "__main__":
    main()
