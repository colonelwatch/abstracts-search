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

from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import json
from pathlib import Path
import sys
from subprocess import Popen, PIPE
from typing import TextIO, BinaryIO, Iterable, overload, Literal

from filelock import FileLock
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch

TRUST_REMOTE_CODE = False
N_TASKS = 2
CHUNK_SIZE = 256  # number of works to process at a time


def parse_args():
    parser = ArgumentParser("build.py", description="Embeds titles and abstracts.")
    parser.add_argument("parquet_path")
    parser.add_argument("-m", "--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("-p", "--prompt-name", default=None)
    parser.add_argument("--fp16", action="store_false", dest="bf16")
    args = parser.parse_args()
    return args


def get_model(model_name: str, bf16: bool):
    # start queries in parallel
    p1 = Popen(
        ["nvidia-smi", "--query-gpu=gpu_bus_id,index", "--format=csv,noheader"],
        stdout=PIPE
    )
    p2 = Popen(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,name", "--format=csv,noheader"],
        stdout=PIPE
    )

    # get "cuda:X" device indices for each GPU
    bus_id_to_index: dict[str, str] = {}
    with p1:
        for line in p1.stdout:
            gpu_bus_id, index = [v.strip() for v in line.decode().split(",")]
            bus_id_to_index[gpu_bus_id] = int(index)

    # get the processes on each "cuda:X" device
    proc_count = [0] * len(bus_id_to_index)
    with p2:
        for line in p2.stdout:
            gpu_bus_id, proc_name = [v.strip() for v in line.decode().split(",")]
            if "python" in proc_name:
                proc_count[bus_id_to_index[gpu_bus_id]] += 1

    # Find the first device that isn't occupied by python then occupy it with the model
    selected_index = min(range(len(proc_count)), key=(lambda i: proc_count[i]))
    model = SentenceTransformer(
        model_name,
        device=f"cuda:{selected_index}",
        trust_remote_code=TRUST_REMOTE_CODE,
        model_kwargs={"torch_dtype": torch.bfloat16 if bf16 else torch.float16}
    )

    return model


def load_oajsonl_chunked(f: TextIO | BinaryIO, chunk_size: int):
    idxs_chunk: list[str] = []
    documents_chunk: list[str] = []
    for line in f:
        row = json.loads(line)

        idxs_chunk.append(row["id"])
        documents_chunk.append(row["document"])
        if len(documents_chunk) >= chunk_size:
            yield idxs_chunk, documents_chunk
            idxs_chunk = []
            documents_chunk = []

    if documents_chunk:
        yield idxs_chunk, documents_chunk


def encode_faster(
    model: SentenceTransformer,
    sentences: list[str],
    prompt: str | None,
    bf16: bool
):
    model.eval()

    features = {}
    if prompt is not None:
        sentences = [prompt + sentence for sentence in sentences]

        tokenized_prompt = model.tokenize([prompt])
        if "input_ids" in tokenized_prompt:
            features |= {"input_ids": tokenized_prompt["input_ids"].shape[-1] - 1}

    features |= {
        k: v.to(model.device, non_blocking=True)
        for k, v in model.tokenize(sentences).items()
    }

    with torch.no_grad():
        out_features = model.forward(features)
        embeddings = out_features["sentence_embedding"]

    if bf16:
        # convert to float32 numpy to preserve all bits
        return embeddings.float().cpu().numpy()
    else:
        return embeddings.cpu().numpy()


def encode_pipelined(
    chunks: Iterable[tuple[list[str], list[str]]],
    model: SentenceTransformer,
    prompt: str,
    bf16: bool,
    n_tasks: int
):
    idxs_chunks = deque[list[str]]()
    embed_tasks = deque[Future[npt.NDArray]]()
    with ThreadPoolExecutor(n_tasks) as executor:
        for idxs_chunk, documents_chunk in chunks:
            # clear out the task queue of completed tasks, then wait until there's room
            while (embed_tasks and embed_tasks[0].done()) or len(embed_tasks) > n_tasks:
                yield idxs_chunks.popleft(), embed_tasks.popleft().result()

            # encode in a task so cpu-to-gpu and gpu-to-cpu transfers are both async
            idxs_chunks.append(idxs_chunk)
            embed_tasks.append(
                executor.submit(encode_faster, model, documents_chunk, prompt, bf16)
            )

        # wait for the remaining tasks to finish
        while embed_tasks:
            yield idxs_chunks.popleft(), embed_tasks.popleft().result()


@overload
def write_parquet(
    path: str, idxs: list[str], embeddings: npt.NDArray[np.float32], bf16: Literal[True]
): ...


@overload
def write_parquet(
    path: str, idxs: list[str], embeddings: npt.NDArray, bf16: Literal[False] = False
): ...


def write_parquet(
    path: str, idxs: list[str], embeddings: npt.NDArray, bf16: bool = False
):
    if bf16 and embeddings.dtype != np.float32:
        raise ValueError("took bf16 path without passing an array promoted to float32")

    _, dim = embeddings.shape
    idxs_arr = pa.array(idxs, pa.string())
    embed_arr = pa.FixedSizeListArray.from_arrays(embeddings.reshape(-1), dim)
    table = pa.Table.from_arrays([idxs_arr, embed_arr], names=["idxs", "embeddings"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if bf16:
        # the conversion from bfloat16 to float32 leaves 16 bits of mantissa which are
        # completely zero. Exploit this with byte-stream split and lz4 compression
        pq.write_table(
            table, path, compression="lz4", use_byte_stream_split=["embeddings"]
        )
    else:
        # otherwise, compressing float embeddings isn't worth it
        pq.write_table(table, path, compression="none")


def main():
    args = parse_args()

    # Get model with file lock to ensure next process will see this one
    with FileLock("/tmp/abstracts-search-gpu.lock"):
        model = get_model(args.model_name, args.bf16)

    if args.prompt_name is None:
        prompt = None
    else:
        prompt = model.prompts[args.prompt_name]

    embedding_dim = model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        print("model doesn't have exact embedding dim")
        exit(-1)

    chunks = load_oajsonl_chunked(sys.stdin, CHUNK_SIZE)
    chunks = encode_pipelined(chunks, model, prompt, args.bf16, N_TASKS)

    idxs_chunks: list[list[str]] = []
    embeddings_chunks: list[npt.NDArray] = []
    for idxs_chunk, embeddings_chunk in chunks:
        idxs_chunks.append(idxs_chunk)
        embeddings_chunks.append(embeddings_chunk)

    # merge chunks
    idxs = [idx for idxs_chunk in idxs_chunks for idx in idxs_chunk]
    if embeddings_chunks:
        embeddings = np.vstack(embeddings_chunks)
    else:
        embeddings = np.empty(
            (0, embedding_dim),
            dtype=np.float32 if args.bf16 else np.float16
        )

    write_parquet(args.parquet_path, idxs, embeddings, bf16=args.bf16)


if __name__ == "__main__":
    main()
