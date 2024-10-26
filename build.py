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

import os
import shutil
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, Pool, Manager
from itertools import repeat
import tqdm

CHUNK_SIZE = 1024 # number of works to process at a time
D = 384 # dimension of the embeddings

def _recover_abstract(inverted_index):
    abstract_size = max([max(appearances) for appearances in inverted_index.values()])+1

    abstract = [None]*abstract_size
    for word, appearances in inverted_index.items(): # yes, this is a second iteration over inverted_index
        for appearance in appearances:
            abstract[appearance] = word

    abstract = [word for word in abstract if word is not None]
    abstract = ' '.join(abstract)
    return abstract

def _build_document(row):
    if row['title']:
        return f'{row["title"]} {_recover_abstract(row["abstract_inverted_index"])}'
    else:
        return _recover_abstract(row['abstract_inverted_index'])

def works_url_routine(args):
    os.nice(10) # lower the priority of this process to avoid slowing down the rest of the system

    # TODO: dynamically determine while GPU to use depending on concurrent processes
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0').half()

    i_task, works_url = args
       
    idxs = []
    embeddings_chunks = []

    works_counter = tqdm.tqdm(desc=f'works_{i_task}', position=1, leave=False)
    chunks_reader = pd.read_json(works_url, lines=True, chunksize=CHUNK_SIZE)
    with works_counter, chunks_reader:
        for works_chunk in chunks_reader:
            # drop unnecessary columns and works with no abstract early to save time and memory
            works_chunk = works_chunk[works_chunk['language'] == 'en']
            works_chunk = works_chunk[works_chunk['abstract_inverted_index'].notnull()]
            works_chunk = works_chunk[(works_chunk['abstract_inverted_index'].astype(str) != '{}')]
            works_chunk = works_chunk[['id', 'title', 'abstract_inverted_index']]

            if len(works_chunk) == 0:
                continue

            documents_chunk = []
            for _, row in works_chunk.iterrows():
                idxs.append(row['id'])
                documents_chunk.append(_build_document(row))
            
            # build the idxs and embeddings for this chunk
            embeddings_chunk = model.encode(documents_chunk, batch_size=128, show_progress_bar=False)

            embeddings_chunks.append(embeddings_chunk)
            works_counter.update(len(documents_chunk))
    
    # merge all the idxs and embeddings chunks into a single list and array
    embeddings = np.vstack(embeddings_chunks) if embeddings_chunks else np.empty((0, D), dtype=np.float16)

    # save the idxs and embeddings built from this chunk to the disk
    np.save(f'partial_works/embeddings_{i_task}.npy', embeddings)
    with open(f'partial_works/idxs_{i_task}.txt', 'w') as f:
        f.write('\n'.join(idxs))
    
    return len(idxs) # return the number of works processed

if __name__ == '__main__':
    # Identify the state of the works table
    if os.path.exists('abstracts-embeddings/embeddings.memmap'):
        print('Completed works table found, exiting...')
        exit()
    elif os.path.exists('abstracts-embeddings/embeddings_000.memmap'):
        print('Completed works table found, but it is split into chunks. Have you called "cat embeddings_*.memmap > embeddings.memmap"? Exiting...')
        exit()
    elif os.path.exists('partial_works/checkpoint.txt'):
        print('Partial works table found, resuming works download...')
        with open('partial_works/checkpoint.txt') as f:
            last_downloaded_url = f.read()
    else:
        print('No works table found, starting works download...')

        last_downloaded_url = None
        shutil.rmtree('partial_works', ignore_errors=True)
        os.makedirs('partial_works')


    # Compare the checkpoint to the works manifest to determine which works to download
    works_manifest = requests.get('https://openalex.s3.amazonaws.com/data/works/manifest').json()
    works_urls = [entry['url'] for entry in works_manifest['entries']]

    i_last_completed_task = works_urls.index(last_downloaded_url) if last_downloaded_url else -1
    i_next_task = i_last_completed_task+1


    work_urls_counter = tqdm.tqdm(desc='work_urls', initial=i_next_task, total=len(works_urls), position=0)
    with Manager() as manager, Pool(1) as pool, work_urls_counter:
        i_iter = range(i_next_task, len(works_urls))
        args_iter = zip(i_iter, works_urls[i_next_task:])
        for n_works in pool.imap(works_url_routine, args_iter):
            i_last_completed_task += 1
            work_urls_counter.update(1)
            with open('partial_works/checkpoint.txt', 'w') as f:
                f.write(works_urls[i_last_completed_task])

    
    print('Merging partial works idxs lists...')
    idxs_chunks_paths = [f'partial_works/idxs_{i}.txt' for i in range(len(works_urls))]
    with open('abstracts-embeddings/openalex_ids.txt', 'w') as f:
        for i, idxs_chunk_path in tqdm.tqdm(enumerate(idxs_chunks_paths), desc='idxs_chunks'):
            with open(idxs_chunk_path) as g:
                idxs_chunk = g.read()
            f.write(idxs_chunk)
            if idxs_chunk and i != len(idxs_chunks_paths)-1:
                f.write('\n')


    # merge the partial works tables
    print('Merging partial works tables...')
    
    n_rows = 0
    embeddings_chunks_paths = [f'partial_works/embeddings_{i}.npy' for i in range(len(works_urls))]
    for embeddings_chunk_path in embeddings_chunks_paths:
        embeddings_chunk = np.load(embeddings_chunk_path)
        n_rows += len(embeddings_chunk)

    embeddings = np.memmap('abstracts-embeddings/embeddings.memmap', dtype=np.float16, mode='w+', shape=(n_rows, D))
    
    rows_ptr = 0
    for embeddings_chunk_path in tqdm.tqdm(embeddings_chunks_paths, desc='embeddings_chunks'):
        embeddings_chunk = np.load(embeddings_chunk_path)
        embeddings[rows_ptr:rows_ptr+len(embeddings_chunk)] = embeddings_chunk
        rows_ptr += len(embeddings_chunk)
        embeddings.flush()

    print('Done!')
