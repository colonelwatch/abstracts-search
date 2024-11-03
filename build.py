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
from subprocess import Popen, PIPE

from filelock import FileLock
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch

TRUST_REMOTE_CODE = False
FP16 = True
N_TASKS = 2
CHUNK_SIZE = 256  # number of works to process at a time


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

    if not FP16:
        # convert to float32 numpy to preserve all bits
        return embeddings.float().cpu().numpy()
    else:
        return embeddings.cpu().numpy()


try:
    parquet_path = sys.argv[1]
except IndexError:
    print("parquet_path not given")
    exit(-1)

try:
    model_name = sys.argv[2]
except IndexError:
    print("model_name not given")
    exit(-1)

try:
    prompt_name = sys.argv[3]
except IndexError:
    prompt_name = None

# Find the first GPU that isn't occupied by python then occupy it with the model
with FileLock("/tmp/abstracts-search-gpu.lock"):
    p1 = Popen(
        ["nvidia-smi", "--query-gpu=gpu_bus_id,index", "--format=csv,noheader"],
        stdout=PIPE
    )
    p2 = Popen(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,name", "--format=csv,noheader"],
        stdout=PIPE
    )

    bus_id_to_index: dict[str, str] = {}
    with p1:
        for line in p1.stdout:
            gpu_bus_id, index = [v.strip() for v in line.decode().split(",")]
            bus_id_to_index[gpu_bus_id] = int(index)

    proc_count = [0] * len(bus_id_to_index)
    with p2:
        for line in p2.stdout:
            gpu_bus_id, proc_name = [v.strip() for v in line.decode().split(",")]
            if "python" in proc_name:
                proc_count[bus_id_to_index[gpu_bus_id]] += 1

    selected_index = min(range(len(proc_count)), key=(lambda i: proc_count[i]))
    model = SentenceTransformer(
        model_name,
        device=f"cuda:{selected_index}",
        trust_remote_code=TRUST_REMOTE_CODE
    )

embedding_dim = model.get_sentence_embedding_dimension()
if embedding_dim is None:
    print("model doesn't have exact embedding dim")
    exit(-1)

model = model.bfloat16() if not FP16 else model.half()
prompt = model.prompts[prompt_name] if prompt_name is not None else None

idxs = []
embeddings_chunks = []
task_queue = deque[Future]()

with ThreadPoolExecutor(N_TASKS) as executor:
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

            # encode in a task so cpu-to-gpu and gpu-to-cpu transfers are both async
            task_queue.append(
                executor.submit(encode_faster, model, documents_chunk, prompt)
            )
            documents_chunk = []

    # wait for the remaining tasks to finish
    while task_queue:
        embeddings_chunk = task_queue.popleft().result()
        embeddings_chunks.append(embeddings_chunk)

    # encode the remaining documents
    if documents_chunk:
        embeddings_chunk = encode_faster(model, documents_chunk, prompt)
        embeddings_chunks.append(embeddings_chunk)

# merge embeddings chunks into a single array
if embeddings_chunks:
    embeddings = np.vstack(embeddings_chunks)
else:
    embeddings = np.empty(
        (0, embedding_dim),
        dtype=np.float32 if not FP16 else np.float16
    )

idxs = pa.array(idxs, pa.string())
embeddings = pa.FixedSizeListArray.from_arrays(embeddings.reshape(-1), embedding_dim)
table = pa.Table.from_arrays([idxs, embeddings], names=["idxs", "embeddings"])

Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
if not FP16:
    # the conversion from bfloat16 to float32 leaves 16 bits of mantissa which are
    # completely zero. Exploit this with byte-stream split and lz4 compression
    pq.write_table(
        table, parquet_path, compression="lz4", use_byte_stream_split=["embeddings"]
    )
else:
    # compressing float16 embeddings isn't worth it
    pq.write_table(table, parquet_path, compression="none")
