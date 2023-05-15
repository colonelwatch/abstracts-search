# train.py
# Trains and tests an index on the works embeddings, yielding a trained index 
#  and a set of Pareto-optimal pairs of recall (10@10) scores and time taken for 
#  that index.

import os
import numpy as np
import torch
from tqdm import tqdm, trange
import faiss
from faiss.contrib.ondisk import merge_ondisk
import gc

TRAIN_SIZE = 4194304
TEST_SIZE = 16384 # the size of the test set that will be held out until after the index is evaluated
CHUNK_SIZE = 1048576 # keep at a small number to avoid running out of memory
D = 384 # dimension of the embeddings
FACTORY_STRING = 'OPQ64_256,IVF65536,PQ64'

def purge_from_memory(obj):
    del obj
    gc.collect()


works_path = 'abstracts-embeddings/embeddings.memmap'
works_train_path = 'abstracts-embeddings/train/train.memmap'
idxs_train_path = 'abstracts-embeddings/train/idxs_train.npy'
idxs_queries_path = 'abstracts-embeddings/train/idxs_queries.npy'
idxs_gt_path = 'abstracts-embeddings/train/idxs_gt.npy'
index_path = 'abstracts-index/index.faiss'
ivfdata_path = 'abstracts-index/index.ivfdata'

os.makedirs('abstracts-embeddings/train', exist_ok=True)
os.makedirs('partial_indices', exist_ok=True)


if not os.path.exists(works_path) and os.path.exists('abstracts-embeddings/embeddings_000.memmap'):
    print('Embeddings found, but it is split into chunks. Have you called "cat embeddings_*.memmap > embeddings.memmap"? Exiting...')

    exit()


# The sampling should be random, so we'll use a reproducible shuffle to do it
embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
n_embeddings = len(embeddings)
purge_from_memory(embeddings)

np.random.seed(42)
idxs_permuted = np.random.permutation(n_embeddings)


idxs_train = idxs_permuted[:TRAIN_SIZE]
idxs_train = np.sort(idxs_train)

train_set_found = False
if os.path.exists(idxs_train_path):
    idxs_train = np.load(idxs_train_path)
    train_set_length = len(idxs_train)
    if train_set_length >= TRAIN_SIZE:
        train_set_found = True

if train_set_found:
    print('Valid train set found, skipping train set generation...')
else:
    print('Train set not found or invalid, generating train set...')

    embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
    embeddings_train = np.memmap(works_train_path, dtype=np.float32, mode='w+', shape=(TRAIN_SIZE, D))

    # references to the embeddings array should destroyed immediately after use
    for i in trange(0, TRAIN_SIZE, CHUNK_SIZE):
        idxs_train_chunk = idxs_train[i:i+CHUNK_SIZE]
        embeddings_train[i:i+CHUNK_SIZE] = embeddings[idxs_train_chunk]
        embeddings_train.flush()
    
    np.save(idxs_train_path, idxs_train) # should be saved AFTER generation to signify the generation was performed

    purge_from_memory(embeddings_train)
    purge_from_memory(embeddings)


idxs_queries = idxs_permuted[-TEST_SIZE:]
idxs_queries = np.sort(idxs_queries)

gt_found = False
if os.path.exists(idxs_gt_path):
    idxs_gt = np.load(idxs_gt_path)
    gt_set_length = len(idxs_gt)
    if gt_set_length >= TEST_SIZE:
        gt_found = True

if gt_found:
    print('Valid ground truth set found, skipping ground truth generation...')
else:
    print('Ground truth set not found or invalid, generating ground truth set...')

    embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
    embeddings_queries = embeddings[idxs_queries].copy()
    embeddings_queries = torch.from_numpy(embeddings_queries).cuda()

    scores_gt = np.zeros((TEST_SIZE, 10), dtype=np.float16)
    idxs_gt = np.empty((TEST_SIZE, 10), dtype=np.int64)
    for i in trange(0, len(embeddings), 65536): # 65536 is the limit on the chunk size, determined by GPU VRAM
        idxs_chunk = np.arange(i, min(i+65536, len(embeddings)))
        in_test_set = np.isin(idxs_chunk, idxs_queries)
        idxs_chunk = idxs_chunk[~in_test_set]

        embeddings_chunk = torch.from_numpy(embeddings[idxs_chunk]).cuda()
        scores_chunk = embeddings_queries @ embeddings_chunk.T

        scores_gt_chunk, arg_scores_gt_chunk = torch.topk(scores_chunk, 10, dim=1, largest=True)
        scores_gt_chunk = scores_gt_chunk.cpu().numpy()
        idxs_gt_chunk = idxs_chunk[arg_scores_gt_chunk.cpu().numpy()]

        for j in range(TEST_SIZE):
            scores = np.hstack((scores_gt[j], scores_gt_chunk[j]))
            idxs = np.hstack((idxs_gt[j], idxs_gt_chunk[j]))

            top_10_filter = np.argsort(scores)[10:]
            top_10_filter = top_10_filter[::-1]

            scores_gt[j] = scores[top_10_filter]
            idxs_gt[j] = idxs[top_10_filter]

    np.save(idxs_queries_path, idxs_queries)
    np.save(idxs_gt_path, idxs_gt)

    purge_from_memory(embeddings_queries) # this purge is about leaked GPU VRAM, not the RAM a memmap would leak
    purge_from_memory(embeddings)
    torch.cuda.empty_cache()


print('Training index...')

index = faiss.index_factory(D, FACTORY_STRING, faiss.METRIC_INNER_PRODUCT)
faiss.ParameterSpace().set_index_parameter(index, 'verbose', 1) # turn on verbose logging

gpu_env = faiss.StandardGpuResources()
gpu_options = faiss.GpuClonerOptions()
gpu_options.useFloat16 = True
index = faiss.index_cpu_to_gpu(gpu_env, 0, index, gpu_options)

embeddings_train = np.memmap(works_train_path, dtype=np.float32, mode='r', shape=(TRAIN_SIZE, D))
index.train(embeddings_train)
faiss.ParameterSpace().set_index_parameter(index, 'verbose', 0) # turn off verbose logging

index = faiss.index_gpu_to_cpu(index)
faiss.write_index(index, 'partial_indices/empty.faiss')

purge_from_memory(embeddings_train)


print('Addings embeddings (except the queries set) to trained index...')

embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)

n_chunks = 0
temp_index_gpu = faiss.index_cpu_to_gpu(gpu_env, 0, index, gpu_options)
for i_chunk, i in enumerate(trange(0, len(embeddings), CHUNK_SIZE)):
    idxs_chunk = np.arange(i, min(i+CHUNK_SIZE, len(embeddings)))
    in_queries_set = np.isin(idxs_chunk, idxs_queries)
    idxs_chunk = idxs_chunk[~in_queries_set]

    temp_index_gpu.add_with_ids(embeddings[idxs_chunk], idxs_chunk)
    temp_index = faiss.index_gpu_to_cpu(temp_index_gpu)
    faiss.write_index(temp_index, f'partial_indices/index_{i_chunk:03d}.faiss')

    n_chunks += 1
    temp_index_gpu.reset()

index = faiss.read_index('partial_indices/empty.faiss')
partial_index_paths = [f'partial_indices/index_{i_chunk:03d}.faiss' for i_chunk in range(n_chunks)]
merge_ondisk(index, partial_index_paths, ivfdata_path.replace('abstracts-index/', ''))
faiss.write_index(index, index_path)

purge_from_memory(embeddings)
purge_from_memory(temp_index_gpu)
purge_from_memory(temp_index)


print('Evaluating index...')

embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
embeddings_queries = embeddings[idxs_queries].astype(np.float32, copy=True)
purge_from_memory(embeddings) # after getting the test set, we don't need the whole embeddings array anymore

idxs_gt = np.load(idxs_gt_path) # in case gt_set_length > TEST_SIZE, the last TEST_SIZE rows are the correct ones
idxs_gt = idxs_gt[-TEST_SIZE:]
tuning_criterion = faiss.IntersectionCriterion(TEST_SIZE, 10)
tuning_criterion.set_groundtruth(None, idxs_gt)

param_space = faiss.ParameterSpace()
param_space.initialize(index)
param_values = param_space.explore(index, embeddings_queries, tuning_criterion)

param_values.display()


print('Adding test set to index...')

partial_test_index = faiss.read_index('partial_indices/empty.faiss')
partial_test_index.add_with_ids(embeddings_queries, idxs_queries)
faiss.write_index(partial_test_index, 'partial_indices/index_test.faiss')

partial_index_paths.append('partial_indices/index_test.faiss')

index = faiss.read_index('partial_indices/empty.faiss')
merge_ondisk(index, partial_index_paths, ivfdata_path.replace('abstracts-index/', ''))
faiss.write_index(index, index_path)


print('Copying idxs.txt to abstracts-index...')
lines_counter = tqdm(total=n_embeddings, desc='lines copied')
with open('abstracts-embeddings/idxs.txt', 'r') as f, open('abstracts-index/idxs.txt', 'w') as g:
    for line in f:
        lines_counter.update()
        g.write(line)


print('Migrating ivfdata to abstracts-index...')
os.rename(ivfdata_path.replace('abstracts-index/', ''), ivfdata_path)