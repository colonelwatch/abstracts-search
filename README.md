# abstracts-search

`abstracts-search` is a project about indexing 110 million academic publications into a single semantic search engine. The method behind it is to take the publicly-available abstracts in the [OpenAlex](https://openalex.org) dataset, generate embeddings using the [stella_en_1.5B_v5](https://huggingface.co/NovaSearch/stella_en_1.5B_v5) model, then train a fast index using [faiss](https://github.com/facebookresearch/faiss). Furthermore, this can be repeated to keep the index synced with the quarterly snapshots of the OpenAlex dataset.

The project is split into four repositories:

* `abstracts-search`: Hosts the embedding and indexing scripts, along with a Makefile for describing the entire build
* `abstracts-embeddings`: Hosts the raw embeddings (released under CC0) as a [Hugging Face Dataset](https://huggingface.co/datasets/colonelwatch/abstracts-embeddings)
* `abstracts-index`: Hosts `app.py`, the search interface, as a [Hugging Face Space](https://huggingface.co/spaces/colonelwatch/abstracts-index) (also released under CC0)
* `abstracts-faiss`: Hosts the index, which `app.py` accesses, as a [Hugging Face Dataset](https://huggingface.co/spaces/colonelwatch/abstracts-faiss) (also released under CC0)

# Running

An Internet connection is always needed. In regards to setup, packages are downloaded with conda or pip, and the model is downloaded from Hugging Face. All data associated with the publications (titles, abstracts, authors, etc) is retrieved from OpenAlex and are not provided with this project. `abstracts-faiss` is only a trained index that takes query embeddings and outputs OpenAlex IDs. These IDs are used in a query sent to OpenAlex.

For running the search interface alone, without regular sync (last synced on Jan 1, 2025), the only repo that needs to be cloned is `abstracts-index`:

```
git lfs install
git clone https://huggingface.co/spaces/colonelwatch/abstracts-index

cd abstracts-index
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

MODEL_NAME="dunzhang/stella_en_1.5B_v5" PROMPT_NAME="s2p_query" TRUST_REMOTE_CODE=1 python3 app.py
```

# Building

Building of `abstracts-embeddings` was done on a Google Cloud machine with four L4 GPUs over one week, and subsequent building was done on a machine with 16 GB of RAM (plus 16 GB swap), an RTX 3060 12GB, and a 2 TB scratch disk, so altogether this poses the minimum requirements for now. An upcoming objective to rebuild the sync so it can be done on just the latter machine, using `abstracts-embeddings` as a base.

There are two ways to build the index: from the `abstracts-embeddings` (with sync) or from scratch, using the OpenAlex S3 bucket.

Either way, make sure `aws` (AWS CLI), `git-lfs`, `mbuffer`, and `conda` are available, and make sure Git over SSH is set up. Then, start with the following setup.

```
sudo nvidia-persistenced
git lfs install

git clone https://github.com/colonelwatch/abstracts-search
cd abstracts-search

conda env create -f environment.yml
conda activate abstracts-search
```

The above commands do not yet download the generated embeddings or the built index, which are in `abstracts-embeddings` and `abstracts-faiss`.

* To proceed building as usual, download the submodules, and then a recovery command needs to be run.

```
git submodule update --init --recursive
make recover
huggingface-cli lfs-enable-largefiles abstracts-faiss
```

* To build from scratch, the submodules can be downloaded but with those things skipped.


```
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
git submodule update --init abstracts-embeddings
rm -r abstracts-embeddings/data abstracts-embeddings/events abstracts-faiss/index
huggingface-cli lfs-enable-largefiles abstracts-faiss
```

Finally, build the index by running the following `make` command.

```
make -j 4 BUILDFLAGS='-m "NovaSearch/stella_en_1.5B_v5" -b 32 --trust-remote-code' DUMPFLAGS='--shard-size 2097152 --row-group-size 65536' TRAINFLAGS='-d 384 -N -p OPQ96 -c 131072'
```

# Syncing

To sync, rerun the above `make` command.

# Experimenting

`abstracts-seach` is structured as a collection of embedding and indexing scripts. Each script is self-contained (besides a few shared utils) and controllable via environment variables, arguments, and options. To adjust the index building or run specific operations, look in the Makefile and scripts for how to run the scripts.
