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

import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

SHARD_SIZE = 4194304  # (2^22), puts the shard size a bit under 4 GB


def take_shard(idxs, embeddings, shards_path, shard_id):
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


try:
    parts_path = sys.argv[1]
except IndexError:
    print("parts_path not provided")
    exit(0)

try:
    shards_path = sys.argv[2]
except IndexError:
    print("shards_path not provided")
    exit(0)

os.mkdir(shards_path)

# merge the partial works tables
shard_id = 0
idxs = []
embeddings = []
for batch in ds.dataset(parts_path).to_batches():
    idxs.append(batch['idxs'])
    embeddings.append(batch['embeddings'])
    while len(idxs) >= SHARD_SIZE:
        idxs, embeddings = take_shard(idxs, embeddings, shards_path, shard_id)
        shard_id += 1
_, _ = take_shard(idxs, embeddings, shards_path, shard_id)
