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

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 1024 # number of works to process at a time
D = 384 # dimension of the embeddings
SHARD_SIZE = 4194304  # (2^22), puts the shard size a bit under 4 GB

def _recover_abstract(inverted_index):
    abstract_size = max([max(appearances) for appearances in inverted_index.values()])+1

    abstract = [None]*abstract_size
    for word, appearances in inverted_index.items(): # yes, this is a second iteration over inverted_index
        for appearance in appearances:
            abstract[appearance] = word

    abstract = [word for word in abstract if word is not None]
    abstract = ' '.join(abstract)
    return abstract

def _build_document(row):
    if row['title']:
        return f'{row["title"]} {_recover_abstract(row["abstract_inverted_index"])}'
    else:
        return _recover_abstract(row['abstract_inverted_index'])

# TODO: dynamically determine while GPU to use depending on concurrent processes
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0').half()

try:
    works_url = sys.argv[1]
except IndexError:
    print("works_url not given")
    exit(0)

try:
    parquet_path = sys.argv[2]
except IndexError:
    print("parquet_path not given")
    exit(0)

idxs = []
embeddings_chunks = []

works_counter = tqdm.tqdm(desc=works_url)  # TODO: same issue for tqdm
chunks_reader = pd.read_json(works_url, lines=True, chunksize=CHUNK_SIZE)
with works_counter, chunks_reader:
    for works_chunk in chunks_reader:
        # drop unnecessary columns and works with no abstract early to save time and memory
        works_chunk = works_chunk[works_chunk['language'] == 'en']
        works_chunk = works_chunk[works_chunk['abstract_inverted_index'].notnull()]
        works_chunk = works_chunk[(works_chunk['abstract_inverted_index'].astype(str) != '{}')]
        works_chunk = works_chunk[['id', 'title', 'abstract_inverted_index']]

        if len(works_chunk) == 0:
            continue

        documents_chunk = []
        for _, row in works_chunk.iterrows():
            idxs.append(row['id'])
            documents_chunk.append(_build_document(row))
        
        # build the idxs and embeddings for this chunk
        embeddings_chunk = model.encode(documents_chunk, batch_size=128, show_progress_bar=False)

        embeddings_chunks.append(embeddings_chunk)
        works_counter.update(len(documents_chunk))

# merge embeddings chunks into a single array
embeddings = np.vstack(embeddings_chunks) if embeddings_chunks else np.empty((0, D), dtype=np.float16)

idxs = pa.array(idxs, pa.string())
embeddings = pa.FixedSizeListArray.from_arrays(embeddings.reshape(-1), D)
table = pa.Table.from_arrays([idxs, embeddings], names=['idxs', 'embeddings'])

# compressing float16 embeddings isn't worth it
Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
pq.write_table(table,  parquet_path, compression='none')
