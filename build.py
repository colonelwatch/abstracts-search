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

from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

N_TASKS = 2
CHUNK_SIZE = 256  # number of works to process at a time
D = 384  # dimension of the embeddings


# TODO: for now, assuming no prompts
def encode_faster(model: SentenceTransformer, sentences: list[str]):
    model.eval()
    features = model.tokenize(sentences)
    features = {k: v.to(model.device, non_blocking=True) for k, v in features.items()}
    with torch.no_grad():
        out_features = model.forward(features)
        embeddings = out_features["sentence_embedding"]
    return embeddings.float().cpu().numpy()


try:
    parquet_path = sys.argv[1]
except IndexError:
    print("parquet_path not given")
    exit(0)

# TODO: dynamically determine while GPU to use depending on concurrent processes
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0").bfloat16()

if model.default_prompt_name is not None:
    prompt = model.prompts.get(model.default_prompt_name, None)
    if prompt is not None:
        raise NotImplementedError("No fast encoding with a model that uses prompts")

idxs = []
embeddings_chunks = []
task_queue = deque[Future]()

with tqdm(desc="works", leave=False) as counter, ThreadPoolExecutor() as executor:
    documents_chunk = []
    for line in sys.stdin:
        row = json.loads(line)
        idxs.append(row["id"])
        documents_chunk.append(row["document"])

        # for efficiency, encode multiple chunks in parallel
        if len(documents_chunk) >= CHUNK_SIZE:
            # clear out the task queue of completed tasks, then wait until there's room
            while (task_queue and task_queue[0].done()) or len(task_queue) > N_TASKS:
                embeddings_chunk = task_queue.popleft().result()
                embeddings_chunks.append(embeddings_chunk)
                counter.update(len(embeddings_chunk))

            # encode in a task so cpu-to-gpu and gpu-to-cpu transfers are both async
            task_queue.append(executor.submit(encode_faster, model, documents_chunk))
            documents_chunk = []

    # wait for the remaining tasks to finish
    while task_queue:
        embeddings_chunk = task_queue.popleft().result()
        embeddings_chunks.append(embeddings_chunk)
        counter.update(len(embeddings_chunk))

    # encode the remaining documents
    if documents_chunk:
        embeddings_chunk = encode_faster(model, documents_chunk)
        embeddings_chunks.append(embeddings_chunk)
        counter.update(len(embeddings_chunk))

# merge embeddings chunks into a single array
if embeddings_chunks:
    embeddings = np.vstack(embeddings_chunks)
else:
    embeddings = np.empty((0, D), dtype=np.float32)

idxs = pa.array(idxs, pa.string())
embeddings = pa.FixedSizeListArray.from_arrays(embeddings.reshape(-1), D)
table = pa.Table.from_arrays([idxs, embeddings], names=["idxs", "embeddings"])

# compressing float16 embeddings isn't worth it
Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
pq.write_table(
    table, parquet_path, compression="lz4", use_byte_stream_split=["embeddings"]
)
