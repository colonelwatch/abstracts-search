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

TRUST_REMOTE_CODE = False
N_TASKS = 2
CHUNK_SIZE = 256  # number of works to process at a time
D = 384  # dimension of the embeddings


def encode_faster(model: SentenceTransformer, sentences: list[str], prompt: str | None):
    model.eval()

    if prompt is None:
        features = {}
    else:
        sentences = [prompt + sentence for sentence in sentences]

        tokenized_prompt = model.tokenize([prompt])
        if "input_ids" in tokenized_prompt:
            features = {"input_ids": tokenized_prompt["input_ids"].shape[-1] - 1}
        else:
            features = {}

    features |= {
        k: v.to(model.device, non_blocking=True)
        for k, v in model.tokenize(sentences).items()
    }    

    with torch.no_grad():
        out_features = model.forward(features)
        embeddings = out_features["sentence_embedding"]

    return embeddings.float().cpu().numpy()


try:
    parquet_path = sys.argv[1]
except IndexError:
    print("parquet_path not given")
    exit(0)

try:
    model_name = sys.argv[2]
except IndexError:
    print("model_name not given")
    exit(0)

try:
    prompt_name = sys.argv[3]
except IndexError:
    prompt_name = None

# TODO: dynamically determine while GPU to use depending on concurrent processes
model = SentenceTransformer(
    model_name, device="cuda:0", trust_remote_code=TRUST_REMOTE_CODE
)
model = model.bfloat16()
prompt = model.prompts[prompt_name] if prompt_name is not None else None

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
            task_queue.append(
                executor.submit(encode_faster, model, documents_chunk, prompt)
            )
            documents_chunk = []

    # wait for the remaining tasks to finish
    while task_queue:
        embeddings_chunk = task_queue.popleft().result()
        embeddings_chunks.append(embeddings_chunk)
        counter.update(len(embeddings_chunk))

    # encode the remaining documents
    if documents_chunk:
        embeddings_chunk = encode_faster(model, documents_chunk, prompt)
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
