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

from utils.gpu_utils import imap, iunsqueeze, iunzip
from utils.table_utils import insert_embeddings, to_sql_binary


def parse_args() -> Namespace:
    parser = ArgumentParser("build.py", description="Embeds titles and abstracts.")
    parser.add_argument("data_path", type=Path)
    parser.add_argument("-m", "--model-name", default="all-MiniLM-L6-v2")
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
    ids_batch: list[str] = []
    documents_batch: list[str] = []
    for line in f:
        row = json.loads(line)

        ids_batch.append(row["id"])
        documents_batch.append(row["document"])
        if len(documents_batch) >= batch_size:
            yield ids_batch, documents_batch
            ids_batch = []
            documents_batch = []

    if documents_batch:
        yield ids_batch, documents_batch


# built from SentenceTransformer.encode but with non-blocking CPU-to-GPU transfers
def encode_faster(
    model: SentenceTransformer,
    sentences: list[str],
) -> torch.Tensor:
    model.eval()

    # Tokenize (which yields a dict) then do a non-blocking transfer
    features = {
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
    n_tasks: int,
    progress: bool,
) -> Generator[tuple[list[str], torch.Tensor], None, None]:
    with tqdm(disable=(not progress)) as count:
        ids_batches, documents_batches = iunzip(batches, 2)
        documents_batches = iunsqueeze(documents_batches)
        embeddings_batches = imap(
            documents_batches, lambda x: encode_faster(model, x), n_tasks
        )
        batches_out = zip(ids_batches, embeddings_batches)
        for ids_batch, embeddings_batch in batches_out:
            count.update(len(embeddings_batch))
            yield ids_batch, embeddings_batch


def main():
    args = parse_args()

    # Get model with file lock to ensure next process will see this one
    with FileLock("/tmp/abstracts-search-gpu.lock"):
        model = get_model(args.model_name, args.bf16, args.trust_remote_code)

    embedding_dim = model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        print("error: model doesn't have exact embedding dim", file=sys.stderr)
        exit(1)

    batches = load_oajsonl_batched(sys.stdin.buffer, args.batch_size)
    batches = encode_pipelined(batches, model, args.tasks, args.progress)

    batches = list(batches)  # collect now to commit them all at once

    sqlite3.register_adapter(torch.Tensor, to_sql_binary)
    with sqlite3.connect(
        args.data_path,
        isolation_level="EXCLUSIVE",
        autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL,  # type: ignore
    ) as conn:
        for ids_batch, embeddings_batch in batches:
            insert_embeddings(ids_batch, embeddings_batch, conn)


if __name__ == "__main__":
    main()
