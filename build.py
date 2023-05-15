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
from fasttext.FastText import _FastText as FastText # importing this way to bypass a forced print to stderr
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, Pool, Manager
from itertools import repeat
import tqdm

N_GPU_WORKERS = 1
N_CPU_WORKERS = 6
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

def model_routine(i_gpu, in_queue, out_queues): # a batch labelled with i_cpu is sent out on out_queues[i_cpu]
    os.nice(10) # lower the priority of this process to avoid slowing down the rest of the system
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=f'cuda:{i_gpu}').half()
    while True:
        documents, i_cpu = in_queue.get()
        embeddings = model.encode(documents, batch_size=128, show_progress_bar=False)
        out_queues[i_cpu].put(embeddings)

def works_url_routine(args):
    os.nice(10) # lower the priority of this process to avoid slowing down the rest of the system
    
    i_task, works_url, model_in_queue, model_out_queues = args
    i_cpu = i_task%N_CPU_WORKERS # infer a unique CPU id from i_task (valid approach as long as we use the pool.imap)
       
    idxs_chunks = []
    embeddings_chunks = []

    lang_detector = FastText('lid.176.bin')
    works_counter = tqdm.tqdm(desc=f'works_{i_task}', position=i_cpu+1, leave=False)
    chunks_reader = pd.read_json(works_url, lines=True, chunksize=CHUNK_SIZE)
    with works_counter, chunks_reader:
        for works_chunk in chunks_reader:
            # drop unnecessary columns and works with no abstract early to save time and memory
            works_chunk = works_chunk[works_chunk['abstract_inverted_index'].notnull()]
            works_chunk = works_chunk[(works_chunk['abstract_inverted_index'].astype(str) != '{}')]
            works_chunk = works_chunk[['id', 'title', 'abstract_inverted_index']]

            idxs_chunk = []
            documents_chunk = []
            for _, row in works_chunk.iterrows():
                idx = row['id']
                document = _build_document(row)

                cleaned_document = document.replace('\n', '').replace('\r', '') # FastText doesn't accept newlines
                __label__lang = lang_detector.predict(cleaned_document)[0][0]

                if __label__lang == '__label__en':
                    idxs_chunk.append(idx)
                    documents_chunk.append(document)
            
            if len(idxs_chunk) == 0:
                continue
            
            # build the idxs and embeddings for this chunk
            model_in_queue.put((documents_chunk, i_cpu))
            embeddings_chunk = model_out_queues[i_cpu].get()

            idxs_chunks.append(idxs_chunk)
            embeddings_chunks.append(embeddings_chunk)
            works_counter.update(len(idxs_chunk))
    
    # merge all the idxs and embeddings chunks into a single list and array
    idxs = sum(idxs_chunks, []) # flatten the list of lists
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

    # if lid.176.bin doesn't exist, download it
    if not os.path.exists('lid.176.bin'):
        print('Downloading FastText language detector...')
        os.system('wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin')


    # Compare the checkpoint to the works manifest to determine which works to download
    works_manifest = requests.get('https://openalex.s3.amazonaws.com/data/works/manifest').json()
    works_urls = [entry['url'] for entry in works_manifest['entries']]

    i_last_completed_task = works_urls.index(last_downloaded_url) if last_downloaded_url else -1
    i_next_task = i_last_completed_task+1


    work_urls_counter = tqdm.tqdm(desc='work_urls', initial=i_next_task, total=len(works_urls), position=0)
    with Manager() as manager, Pool(N_CPU_WORKERS) as pool, work_urls_counter:
        model_in_queue = manager.Queue()
        model_out_queues = [manager.Queue() for _ in range(N_CPU_WORKERS)]
        model_workers = [
            Process(target=model_routine, args=(i_gpu, model_in_queue, model_out_queues))
            for i_gpu in range(N_GPU_WORKERS)
        ]
        for i_gpu in range(N_GPU_WORKERS):
            model_workers[i_gpu].start()

        i_iter = range(i_next_task, len(works_urls))
        model_in_queue_iter = repeat(model_in_queue) # queues aren't pickleable, but they can be passed as an arg
        model_out_queues_iter = repeat(model_out_queues)
        args_iter = zip(i_iter, works_urls[i_next_task:], model_in_queue_iter, model_out_queues_iter)
        for n_works in pool.imap(works_url_routine, args_iter):
            i_last_completed_task += 1
            work_urls_counter.update(1)
            with open('partial_works/checkpoint.txt', 'w') as f:
                f.write(works_urls[i_last_completed_task])
        
        for i_gpu in range(N_GPU_WORKERS):
            model_workers[i_gpu].terminate()
            model_workers[i_gpu].join()

    
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
