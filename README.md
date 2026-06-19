# abstracts-search

`abstracts-search` is a project about indexing 200+ million academic publications into a single semantic search engine. The method behind it is to take the publicly-available abstracts in the [OpenAlex](https://openalex.org) dataset, generate embeddings using the [stella_en_1.5B_v5](https://huggingface.co/NovaSearch/stella_en_1.5B_v5) model, then train a fast index using [faiss](https://github.com/facebookresearch/faiss). Furthermore, this can be repeated to keep the index synced with the quarterly snapshots of the OpenAlex dataset.

The project is split into four repositories:

- `abstracts-search`: An orchestration Makefile and the helper program `oa_jsonl`.
- `abstracts-embeddings`: The raw embeddings (released under CC0) as a [Hugging Face Dataset](https://huggingface.co/datasets/colonelwatch/abstracts-embeddings).
- `abstracts-index`: The search interface as a [Hugging Face Space](https://huggingface.co/spaces/colonelwatch/abstracts-index) (also released under CC0).
- `abstracts-faiss`: The trained index as a [Hugging Face Dataset](https://huggingface.co/datasets/colonelwatch/abstracts-faiss) (also released under CC0).

Additionally, this project loads [`sidecar-search`](https://github.com/colonelwatch/sidecar-search) (originally spun off from this project), a set of CLI build tools for building sidecar indexes.

## Running

An Internet connection is always needed. All data associated with the publications (titles, abstracts, authors, etc) is retrieved from OpenAlex and is not provided with this project. `abstracts-faiss` is only a trained index that maps query embeddings to directly to OpenAlex IDs, ranked by semantic similarity. These IDs are then used to query OpenAlex.

For running the search interface alone, without regular sync (last synced on March 30th, 2026 as of the time of writing), the only repo that needs to be cloned is `abstracts-index`:

```
git clone https://huggingface.co/spaces/colonelwatch/abstracts-index
cd abstracts-index

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

MODEL_NAME="NovaSearch/stella_en_1.5B_v5" PROMPT_NAME="s2p_query" TRUST_REMOTE_CODE=1 python3 app.py
```

## Building

Building of `abstracts-embeddings` was originally done on a Google Cloud machine with four L4 GPUs over one week, with all subsequent building done on a machine with 16 GB of RAM (plus 16 GB swap), an RTX 3060 12GB, and a 2 TB scratch disk, so altogether this poses the minimum requirements for now. That initial build can be bypassed by using `abstracts-embeddings` as a base.

Ensure `hf` ([Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)), `jq`, and `mbuffer` are available. Then, start with the following setup.

```
sudo nvidia-persistenced

git clone https://github.com/colonelwatch/abstracts-search
cd abstracts-search

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- To proceed by building from the `abstracts-embeddings` repository as a base, and run the following recovery commands.

```
hf download colonelwatch/abstracts-embeddings --repo-type dataset --local-dir abstracts-embeddings
make recover
```

- Otherwise, immediately proceed to the next step.

Finally, build the index by running the following `make` command.

```
SIDECARSEARCH_MODEL="NovaSearch/stella_en_1.5B_v5" SIDECARSEARCH_TRUST_REMOTE_CODE=1 make BUILDFLAGS='-b 32' DUMPFLAGS='--shard-size 2097152 --row-group-size 65536' TRAINFLAGS='-N -c 65536'
```

## Syncing

To sync, rerun the above `make` command.
