# build.py

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

import json
from pathlib import Path
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 1024 # number of works to process at a time
D = 384 # dimension of the embeddings

# TODO: dynamically determine while GPU to use depending on concurrent processes
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0').half()

try:
    parquet_path = sys.argv[1]
except IndexError:
    print("parquet_path not given")
    exit(0)

idxs = []
embeddings_chunks = []

documents_chunk = []
for line in tqdm(sys.stdin, desc="works", leave=False):
    row = json.loads(line)
    idxs.append(row["id"])
    documents_chunk.append(row["document"])

    if len(documents_chunk) >= CHUNK_SIZE:
        embeddings_chunk = model.encode(documents_chunk, batch_size=CHUNK_SIZE)
        embeddings_chunks.append(embeddings_chunk)
        documents_chunk = []

if documents_chunk:
    embeddings_chunk = model.encode(documents_chunk, batch_size=CHUNK_SIZE)
    embeddings_chunks.append(embeddings_chunk)
    documents_chunk = []

# merge embeddings chunks into a single array
embeddings = np.vstack(embeddings_chunks) if embeddings_chunks else np.empty((0, D), dtype=np.float16)

idxs = pa.array(idxs, pa.string())
embeddings = pa.FixedSizeListArray.from_arrays(embeddings.reshape(-1), D)
table = pa.Table.from_arrays([idxs, embeddings], names=['idxs', 'embeddings'])

# compressing float16 embeddings isn't worth it
Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
pq.write_table(table,  parquet_path, compression='none')
