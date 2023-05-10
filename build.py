import os
import shutil
import requests
import numpy as np
import pandas as pd
from fasttext.FastText import _FastText as FastText # importing this way to bypass a forced print to stderr
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, Pool, Manager
import tqdm

N_GPU_WORKERS = 1
N_CPU_WORKERS = 4
CHUNK_SIZE = 128 # number of works to process at a time
D = 384 # dimension of the embeddings

# below are some helper functions

def infer_source_id(row):
    try:
        source_object = row['primary_location']
        return source_object['id']
    except (TypeError, KeyError): # row['primary_location'] may be NaN or a dict without an 'id' key
        try:
            source_object = row['host_venue']
            return source_object['id']
        except (TypeError, KeyError):
            return None

def recover_abstract(inverted_index):
    abstract_size = max([max(appearances) for appearances in inverted_index.values()])+1

    abstract = [None]*abstract_size
    for word, appearances in inverted_index.items(): # yes, this is a second iteration over inverted_index
        for appearance in appearances:
            abstract[appearance] = word

    abstract = [word for word in abstract if word is not None]
    abstract = ' '.join(abstract)
    return abstract

def build_document(row):
    if row['title']:
        return f'{row["title"]} {recover_abstract(row["abstract_inverted_index"])}'
    else:
        return recover_abstract(row['abstract_inverted_index'])

def model_routine(i_gpu, in_queue, out_queues): # a batch labelled with i_cpu is sent out on out_queues[i_cpu]
    model = SentenceTransformer('all-MiniLM-L6-v2', device=f'cuda:{i_gpu}').half()
    while True:
        documents, i_cpu = in_queue.get()
        embeddings = model.encode(documents, show_progress_bar=False)
        out_queues[i_cpu].put(embeddings)

# the below code tries to replicate the following SQL query:
# SELECT id, title, abstract_inverted_index FROM works INNER JOIN sources ON 
# works.primary_location.id = sources.id WHERE sources.country_code = 'US' 
# AND abstract_inverted_index IS NOT NULL AND abstract_inverted_index::text 
# <> '{}';
def works_url_routine(i_cpu, i_task, works_url, model_in_queue, model_out_queue):
    lang_detector = FastText('lid.176.bin')
    works_counter = tqdm.tqdm(desc=f'works_{i_task}', position=i_cpu+1, leave=False)
    chunks_reader = pd.read_json(works_url, lines=True, chunksize=CHUNK_SIZE)
    with works_counter, chunks_reader:
        idxs_chunks = []
        embeddings_chunks = []
        idxs_chunk = []
        documents_chunk = []
        for works_chunk in chunks_reader:
            works_chunk = works_chunk[ \
                (works_chunk['abstract_inverted_index'].notnull()) & \
                (works_chunk['abstract_inverted_index'].astype(str) != '{}') \
            ] # filter out works with no abstract early to save time and memory
            works_chunk = works_chunk[['id', 'title', 'abstract_inverted_index']] # also drop all other columns to save memory

            for _, row in works_chunk.iterrows():
                idx = row['id']
                document = build_document(row)

                cleaned_document = document.replace('\n', '').replace('\r', '') # FastText doesn't accept newlines
                __label__lang = lang_detector.predict(cleaned_document)[0][0]

                if __label__lang == '__label__en':
                    idxs_chunk.append(idx)
                    documents_chunk.append(document)
            
            if len(idxs_chunk) > CHUNK_SIZE:
                assert len(idxs_chunk) == len(documents_chunk)

                # build the idxs and embeddings for this chunk
                model_in_queue.put((documents_chunk, i_cpu))
                embeddings_chunk = model_out_queue.get()

                idxs_chunks.append(idxs_chunk)
                embeddings_chunks.append(embeddings_chunk)

                works_counter.update(len(idxs_chunk))
                idxs_chunk = []
                documents_chunk = []

        # build the idxs and embeddings for the last chunk
        if len(idxs_chunk) > 0:
            assert len(idxs_chunk) == len(documents_chunk)

            # build the idxs and embeddings for this chunk
            model_in_queue.put((documents_chunk, i_cpu))
            embeddings_chunk = model_out_queue.get()

            idxs_chunks.append(idxs_chunk)
            embeddings_chunks.append(embeddings_chunk)

            works_counter.update(len(idxs_chunk))
    
    idxs = []
    for idxs_chunk in idxs_chunks:
        idxs.extend(idxs_chunk)

    if len(embeddings_chunks) == 0:
        embeddings = np.empty((0, D), dtype=np.float16)
    else:
        embeddings = np.vstack(embeddings_chunks)

    # save the idxs and embeddings built from this chunk to the disk
    np.save(f'partial_works/embeddings_{i_task}.npy', embeddings)
    with open(f'partial_works/idxs_{i_task}.txt', 'w') as f:
        f.write('\n'.join(idxs))
    
    return len(idxs) # return the number of works processed

if __name__ == '__main__':
    # Identify the state of the works table
    if os.path.exists('abstracts-embeddings/idxs.txt') and os.path.exists('abstracts-embeddings/embeddings.memmap'):
        print('Completed works table found, exiting...')

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
    n_tasks_remaining = len(works_urls)-i_next_task


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

        # assign the first N_CPU_WORKERS tasks unless there are fewer than N_CPU_WORKERS tasks
        work_futures = [None]*N_CPU_WORKERS
        for i_cpu in range(min(n_tasks_remaining, N_CPU_WORKERS)):
            work_futures[i_cpu] = pool.apply_async(
                works_url_routine, 
                (i_cpu, i_next_task, works_urls[i_next_task], model_in_queue, model_out_queues[i_cpu])
            )
            i_next_task += 1
        
        i_cpu = 0
        while n_tasks_remaining > 0:
            n_abstracts = work_futures[i_cpu].get() # wait for the next result to come in
            n_tasks_remaining -= 1
            i_last_completed_task += 1

            # update progress bars and checkpoint
            work_urls_counter.update(1)
            with open('partial_works/checkpoint.txt', 'w') as f:
                f.write(works_urls[i_last_completed_task])
            
            # if all remaining tasks have been assigned, don't assign more
            if i_next_task < len(works_urls):
                work_futures[i_cpu] = pool.apply_async(
                    works_url_routine, 
                    (i_cpu, i_next_task, works_urls[i_next_task], model_in_queue, model_out_queues[i_cpu])
                )
                i_next_task += 1

            i_cpu = (i_cpu+1)%N_CPU_WORKERS
        
        for i_gpu in range(N_GPU_WORKERS):
            model_workers[i_gpu].terminate()
            model_workers[i_gpu].join()

    
    print('Merging partial works idxs lists...')
    idxs_chunks_paths = [f'partial_works/idxs_{i}.txt' for i in range(len(works_urls))]
    with open('abstracts-embeddings/idxs.txt', 'w') as f:
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