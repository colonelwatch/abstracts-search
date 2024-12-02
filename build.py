# build.py

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

from argparse import ArgumentParser, Namespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import json
from pathlib import Path
import sys
from subprocess import Popen, PIPE
import sqlite3
from typing import TextIO, BinaryIO, Iterable, Generator

from filelock import FileLock
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser("build.py", description="Embeds titles and abstracts.")
    parser.add_argument("data_path", type=Path)
    parser.add_argument("-m", "--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("-p", "--prompt-name", default=None)
    parser.add_argument("-t", "--tasks", default=2, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--fp16", action="store_false", dest="bf16")  # fp16 or bf16
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("-P", "--progress", action="store_true")
    args = parser.parse_args()
    return args


def get_model(
    model_name: str, bf16: bool, trust_remote_code: bool
) -> SentenceTransformer:
    # start queries in parallel
    p1 = Popen(
        ["nvidia-smi", "--query-gpu=gpu_bus_id,index", "--format=csv,noheader"],
        stdout=PIPE
    )
    p2 = Popen(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,name", "--format=csv,noheader"],
        stdout=PIPE
    )
    assert p1.stdout is not None
    assert p2.stdout is not None

    # get "cuda:X" device indices for each GPU
    bus_id_to_index: dict[str, int] = {}
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
        trust_remote_code=trust_remote_code,
        model_kwargs={"torch_dtype": torch.bfloat16 if bf16 else torch.float16}
    )

    return model


def load_oajsonl_batched(
    f: TextIO | BinaryIO, batch_size: int
) -> Generator[tuple[list[str], list[str]], None, None]:
    idxs_batch: list[str] = []
    documents_batch: list[str] = []
    for line in f:
        row = json.loads(line)

        idxs_batch.append(row["id"])
        documents_batch.append(row["document"])
        if len(documents_batch) >= batch_size:
            yield idxs_batch, documents_batch
            idxs_batch = []
            documents_batch = []

    if documents_batch:
        yield idxs_batch, documents_batch


# built from SentenceTransformer.encode but with non-blocking CPU-to-GPU transfers
def encode_faster(
    model: SentenceTransformer,
    sentences: list[str],
    prompt: str | None,
) -> torch.Tensor:
    model.eval()

    # if given a prompt, add it to the sentences
    features = {}
    if prompt is not None:
        sentences = [prompt + sentence for sentence in sentences]

        # SentenceTransformers expects this feature if a prompt is used
        tokenized_prompt = model.tokenize([prompt])
        if "input_ids" in tokenized_prompt:
            features |= {"input_ids": tokenized_prompt["input_ids"].shape[-1] - 1}

    # Tokenize (which yields a dict) then do a non-blocking transfer
    features |= {
        k: v.to(model.device, non_blocking=True)
        for k, v in model.tokenize(sentences).items()
    }

    with torch.no_grad():
        out_features = model.forward(features)
        embeddings = out_features["sentence_embedding"]

    return embeddings.cpu()


def encode_pipelined(
    batches: Iterable[tuple[list[str], list[str]]],
    model: SentenceTransformer,
    prompt: str | None,
    n_tasks: int,
    progress: bool,
) -> Generator[tuple[list[str], torch.Tensor], None, None]:
    idxs_batches = deque[list[str]]()
    embed_tasks = deque[Future[torch.Tensor]]()
    with ThreadPoolExecutor(n_tasks) as executor, tqdm(disable=(not progress)) as count:
        for idxs_batch, documents_batch in batches:
            # clear out the task queue of completed tasks, then wait until there's room
            while (embed_tasks and embed_tasks[0].done()) or len(embed_tasks) > n_tasks:
                embeddings_batch = embed_tasks.popleft().result()
                count.update(len(embeddings_batch))
                yield idxs_batches.popleft(), embeddings_batch

            # encode in a task so cpu-to-gpu and gpu-to-cpu transfers are both async
            idxs_batches.append(idxs_batch)
            embed_tasks.append(
                executor.submit(encode_faster, model, documents_batch, prompt)
            )

        # wait for the remaining tasks to finish
        while embed_tasks:
            embeddings_batch = embed_tasks.popleft().result()
            count.update(len(embeddings_batch))
            yield idxs_batches.popleft(), embeddings_batch


def to_sql_binary(vect: torch.Tensor) -> sqlite3.Binary:
    if vect.dtype == torch.bfloat16:
        vect = vect.view(torch.uint16)
    return vect.numpy().data


def main():
    args = parse_args()

    # Get model with file lock to ensure next process will see this one
    with FileLock("/tmp/abstracts-search-gpu.lock"):
        model = get_model(args.model_name, args.bf16, args.trust_remote_code)

    if args.prompt_name is None:
        prompt = None
    else:
        prompt = model.prompts[args.prompt_name]

    embedding_dim = model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        print("model doesn't have exact embedding dim")
        exit(-1)

    batches = load_oajsonl_batched(sys.stdin.buffer, args.batch_size)
    batches = encode_pipelined(batches, model, prompt, args.tasks, args.progress)

    idxs_batches: list[list[str]] = []
    embeddings_batches: list[torch.Tensor] = []
    for idxs_batch, embeddings_batch in batches:
        idxs_batches.append(idxs_batch)
        embeddings_batches.append(embeddings_batch)

    sqlite3.register_adapter(torch.Tensor, to_sql_binary)
    with sqlite3.connect(
        args.data_path,
        isolation_level="EXCLUSIVE",
        autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL,  # type: ignore
    ) as conn:
        for idxs_batch, embeddings_batch in zip(idxs_batches, embeddings_batches):
            tups = [
                (idx, embedding) for idx, embedding
                in zip(idxs_batch, embeddings_batch)
            ]
            conn.executemany(
                "INSERT INTO embeddings VALUES(?, ?) "
                "ON CONFLICT(oa_id) DO UPDATE SET embedding=excluded.embedding",
                tups
            )


if __name__ == "__main__":
    main()
