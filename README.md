# abstracts-search

`abstracts-search` is a project about indexing 95 million academic publications into a single semantic search engine. The method behind it is to take the publicly-available abstracts in the [OpenAlex](https://openalex.org) dataset and generate embeddings using the `all-MiniLM-L6-v2` model provided by [sentence-transformers](https://www.sbert.net/).

The project is split into three repositories:

* `abstracts-search`: Hosts `build.py` and `train.py`, the embedding and indexing scripts respectively
* `abstracts-embeddings`: Hosts the raw embeddings (released under CC0) as a [Hugging Face Dataset](https://huggingface.co/datasets/colonelwatch/abstracts-embeddings)
* `abstracts-index`: Hosts the index and `app.py`, the search interface, as a [Hugging Face Space](https://huggingface.co/spaces/colonelwatch/abstracts-index)

# Running Locally

All data associated with the publications (titles, abstracts, authors, etc) is not provided by this project. Rather, it only contains in embeddings labeled with OpenAlex IDs, and the IDs are using to fetch that data from the OpenAlex API. An internet connection is therefore always needed. Still, running the semantic search locally may be desirable.

If that is the case, the only repo that needs to be cloned is `abstracts-index`:

```
git lfs install
git clone https://huggingface.co/spaces/colonelwatch/abstracts-index

cd abstracts-index
pip3 install -r requirements.txt

python3 app.py
```

# Building

All building was done on a machine with 16 GB of RAM (plus 16 GB swap), an RTX 2060 6GB, and a 1 TB scratch disk, so this poses the minimum requirements for now.

There are two ways to build the index: from the `abstracts-embeddings` (recommended) or from the OpenAlex S3 bucket.

To build from `abstracts-embeddings`, make sure `conda` and `gcc-12` are available:

```
git lfs install

git clone https://github.com/colonelwatch/abstracts-search

env CC=gcc-12 conda env create -f environment.yml
conda activate abstracts-search

git submodule update --init abstracts-embeddings
cd abstracts-embeddings
cat embeddings_*.memmap > embeddings.memmap
cd ..

env GIT_LFS_SKIP_SMUDGE=1 git submodule update --init abstracts-index
python train.py
```

If you're looking to build from the OpenAlex S3 bucket, you may also be interested in retaining the git history. Again, make sure `conda` and `gcc-12` are available.

```
git lfs install

git clone https://github.com/colonelwatch/abstracts-search

env CC=gcc-12 conda env create -f environment.yml
conda activate abstracts-search

env GIT_LFS_SKIP_SMUDGE=1 git submodule update --init abstracts-embeddings
rm abstracts-embeddings/embeddings_*.memmap
rm abstracts-embeddings/openalex_ids.txt
python build.py

env GIT_LFS_SKIP_SMUDGE=1 git submodule update --init abstracts-index
python train.py
```