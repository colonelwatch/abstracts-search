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
from concurrent.futures import ThreadPoolExecutor
import json
import queue
from pathlib import Path
import sys
from subprocess import Popen, PIPE
import sqlite3
from threading import Thread, Event
from typing import cast, TextIO, Iterable, Generator, Self

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
    parser.add_argument("--filter-tasks", default=5, type=int)
    parser.add_argument("--filter-batch-size", default=1024, type=int)
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


class OaJsonlBatched:
    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._worker = Thread(target=self._load_routine)
        self._queue: queue.Queue[tuple[list[str], list[str]] | None] = queue.Queue(3)
        self._halt = Event()

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._halt.set()
        self._worker.join()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.stop()

    def __iter__(self) -> Generator[tuple[list[str], list[str]], None, None]:
        while True:
            batch = self._queue.get()
            if batch is None:
                break
            yield batch

    def _load_routine(self):
        stdin_cast = cast(TextIO, sys.stdin)
        
        ids_batch: list[str] = []
        documents_batch: list[str] = []
        for line in stdin_cast.buffer:
            row = json.loads(line)

            ids_batch.append(row["id"])
            documents_batch.append(row["document"])
            if len(documents_batch) >= self._batch_size:
                self._queue.put((ids_batch, documents_batch))
                ids_batch = []
                documents_batch = []

            if self._halt.is_set():
                return

        if documents_batch:
            self._queue.put((ids_batch, documents_batch))

        self._queue.put(None)


class SharedConnection:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._worker = ThreadPoolExecutor(1)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        fut = self._worker.submit(sqlite3.connect, self._path, autocommit=False)
        self._conn = fut.result()

    def close(self) -> None:
        if self._conn is None:
            raise RuntimeError("closed with a not-open connection")
        fut = self._worker.submit(self._conn.close)
        fut.result()
        self._worker.shutdown()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def pick_existing(self, ids: list[str]) -> list[str]:
        def _pick_existing():
            conn = self._ensure_conn()
            placeholders = ", ".join("?" * len(ids))
            return conn.execute(
                f"SELECT id from embeddings WHERE id IN ({placeholders})", ids
            ).fetchall()

        fut = self._worker.submit(_pick_existing)
        res: list[tuple[str]] = fut.result()

        return [id_ for (id_,) in res]

    def insert_async(
        self, oa_ids: Iterable[str], embeddings: Iterable[torch.Tensor]
    ) -> None:
        def _insert():
            conn = self._ensure_conn()
            insert_embeddings(oa_ids, embeddings, conn)
            conn.commit()
        _ = self._worker.submit(_insert)

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("called with a not-open connection")
        return self._conn


def filter_batched(
    batches: Iterable[tuple[list[str], list[str]]],
    conn: SharedConnection,
    batch_size: int,
    n_tasks: int,
    progress: bool,
) -> Generator[tuple[list[str], list[str]], None, None]:
    filtereds: list[tuple[str, str]] = []

    def roll(drain: bool):
        nonlocal filtereds

        while len(filtereds) >= batch_size or (drain and filtereds):
            out = filtereds[:batch_size]
            ids_out = [id_ for id_, _ in out]
            documents_out = [document for _, document in out]
            yield ids_out, documents_out

            filtereds = filtereds[batch_size:]

    with tqdm(disable=(not progress)) as count:
        def filt(ids: list[str], documents: list[str]):
            batch = {id_: document for id_, document in zip(ids, documents)}
            for id_ in conn.pick_existing(ids):
                del batch[id_]
            count.update(len(ids))  # update counter with the unfiltered count
            return batch

        for filtered in imap(batches, filt, n_tasks):
            filtereds.extend(filtered.items())
            yield from roll(False)
        yield from roll(True)


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
) -> Generator[tuple[list[str], torch.Tensor], None, None]:
    ids_batches, documents_batches = iunzip(batches, 2)
    documents_batches = iunsqueeze(documents_batches)
    embeddings_batches = imap(
        documents_batches, lambda x: encode_faster(model, x), n_tasks
    )
    batches_out = zip(ids_batches, embeddings_batches)
    for ids_batch, embeddings_batch in batches_out:
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

    sqlite3.register_adapter(torch.Tensor, to_sql_binary)
    with (
        OaJsonlBatched(args.filter_batch_size) as batches,
        SharedConnection(args.data_path) as conn
    ):
        batches = filter_batched(
            batches, conn, args.batch_size, args.filter_tasks, args.progress
        )
        batches = encode_pipelined(batches, model, args.tasks)
        for ids_batch, embeddings_batch in batches:
            conn.insert_async(ids_batch, embeddings_batch)


if __name__ == "__main__":
    main()
