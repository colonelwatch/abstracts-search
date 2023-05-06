# train.py
# Trains and tests an index on the works embeddings, yielding a trained index 
#  and a set of Pareto-optimal pairs of recall (10@10) scores and time taken for 
#  that index.

import os
import numpy as np
import torch
from tqdm import trange
from tqdm.contrib.concurrent import thread_map
import faiss
import gc

TRAIN_SIZE = 4194304
TEST_SIZE = 131072 # the size of the test set that will be held out until after the index is evaluated
CHUNK_SIZE = 1048576 # keep at a small number to avoid running out of memory
D = 384 # dimension of the embeddings
FACTORY_STRING = 'OPQ32_128,IVF65536,PQ32'

def purge_from_memory(obj):
    del obj
    gc.collect()


works_path = 'abstracts-embeddings/embeddings.memmap'
works_train_path = 'abstracts-embeddings/train/train.memmap'
idxs_train_path = 'abstracts-embeddings/train/idxs_train.npy'
idxs_queries_path = 'abstracts-embeddings/train/idxs_queries.npy'
idxs_gt_path = 'abstracts-embeddings/train/idxs_gt.npy'
index_path = 'abstracts-embeddings/index.faiss'

os.makedirs('abstracts-embeddings/train', exist_ok=True)


# The sampling should be random, so we'll use a reproducible shuffle to do it
embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
n_embeddings = len(embeddings)
purge_from_memory(embeddings)

np.random.seed(42)
idxs_permuted = np.random.permutation(n_embeddings)


train_set_found = False
if os.path.exists(works_train_path):
    train_set_length = len(np.load(idxs_train_path))
    if train_set_length == TRAIN_SIZE:
        train_set_found = True

if train_set_found:
    print('Valid train set found, skipping train set generation...')

    idxs_train = np.load(idxs_train_path) # expose idxs_train to the rest of the script
else:
    print('Train set not found or invalid, generating train set...')

    print('Loading embeddings...')
    embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
    n_embeddings = len(embeddings)

    print('Generating train set...')
    idxs_train = idxs_permuted[:TRAIN_SIZE]
    idxs_train = np.sort(idxs_train)
    np.save(idxs_train_path, idxs_train)

    embeddings_train = np.memmap(works_train_path, dtype=np.float32, mode='w+', shape=(TRAIN_SIZE, D))
    for i in trange(TRAIN_SIZE//CHUNK_SIZE+1):
        idxs_train_chunk = idxs_train[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        embeddings_train[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = embeddings[idxs_train_chunk]
        embeddings_train.flush()

    purge_from_memory(embeddings_train)
    purge_from_memory(embeddings)


BATCH_SIZE = 4096
N_VECTORS_COMPARED = 65536
def search_routine(idxs_queries_batch):
    embeddings_test_batch = embeddings[idxs_queries_batch]
    embeddings_test_batch = torch.from_numpy(embeddings_test_batch).cuda()
    
    top_10_scores_batch = np.zeros((BATCH_SIZE, 10), dtype=np.float16)
    top_10_idxs_batch = np.empty((BATCH_SIZE, 10), dtype=np.int64)
    for j in range(0, n_embeddings, N_VECTORS_COMPARED):
        idxs_chunk = np.arange(j, min(j+N_VECTORS_COMPARED, n_embeddings))
        in_test_set = np.isin(idxs_chunk, idxs_queries_batch)
        idxs_chunk = idxs_chunk[~in_test_set]

        embeddings_chunk = torch.from_numpy(embeddings[idxs_chunk]).cuda()
        scores_batch_chunk = embeddings_test_batch @ embeddings_chunk.T

        top_10_scores_batch_chunk, arg_top_10_scores_batch_chunk = torch.topk(scores_batch_chunk, 10, dim=1, largest=True)
        top_10_scores_batch_chunk = top_10_scores_batch_chunk.cpu().numpy()
        top_10_idxs_batch_chunk = idxs_chunk[arg_top_10_scores_batch_chunk.cpu().numpy()]

        for i in range(BATCH_SIZE):
            scores = np.hstack((top_10_scores_batch[i], top_10_scores_batch_chunk[i]))
            idxs = np.hstack((top_10_idxs_batch[i], top_10_idxs_batch_chunk[i]))

            top_10_filter = np.argsort(scores)[10:]
            top_10_filter = top_10_filter[::-1]

            top_10_scores_batch[i] = scores[top_10_filter]
            top_10_idxs_batch[i] = idxs[top_10_filter]
    
    return top_10_idxs_batch

queries_set_found = False
if os.path.exists(idxs_queries_path):
    queries_set_length = len(np.load(idxs_queries_path))
    if queries_set_length == TEST_SIZE:
        queries_set_found = True

# The precomputed train set also needs to be valid. If it's not, the precomputed 
# test and queries sets might be for the wrong slice of idxs_permuted
if train_set_found and queries_set_found:
    print('Queries test set found, skipping queries and ground truth sets generation...')

    idxs_queries = np.load(idxs_queries_path) # expose idxs_queries to the rest of the script
    idxs_gt = np.load(idxs_gt_path) # expose idxs_gt to the rest of the script
else:
    # the remaining cases is !T!Q, !TQ, and T!Q, but only !TQ needs a special explanation
    if not train_set_found and queries_set_found:
        print('Queries test set might be invalid, generating queries and ground truth sets...')
    else:
        print('Queries test set not found or invalid, generating queries and ground truth sets...')

    # TODO: right now, search_routine counts in this reference existing, but this reference actually shows up in the 
    #  code AFTER search_routine is defined. This is a confusing design.
    embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
    n_embeddings = len(embeddings)

    idxs_queries = idxs_permuted[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    idxs_queries = np.sort(idxs_queries)

    idxs_queries_batches = np.array_split(idxs_queries, TEST_SIZE//BATCH_SIZE)
    idxs_gt_batches = thread_map(search_routine, idxs_queries_batches, max_workers=3, desc='query batches')
    idxs_gt = np.vstack(idxs_gt_batches)

    np.save(idxs_queries_path, idxs_queries)
    np.save(idxs_gt_path, idxs_gt)

    purge_from_memory(embeddings)
    torch.cuda.empty_cache()


print('Training index...')

gpu_env = faiss.StandardGpuResources()
gpu_options = faiss.GpuClonerOptions()
gpu_options.useFloat16 = True

embeddings_train = np.memmap(works_train_path, dtype=np.float32, mode='r', shape=(TRAIN_SIZE, D))

index = faiss.index_factory(D, FACTORY_STRING, faiss.METRIC_INNER_PRODUCT)
index = faiss.index_cpu_to_gpu(gpu_env, 0, index, gpu_options)
index.train(embeddings_train)
index = faiss.index_gpu_to_cpu(index)
purge_from_memory(embeddings_train)


print('Adding embeddings (except the test set) to trained index...')

embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
n_embeddings = len(embeddings)
purge_from_memory(embeddings)

index = faiss.index_cpu_to_gpu(gpu_env, 0, index, gpu_options)
for i in trange(len(embeddings)//CHUNK_SIZE+1):
    offset = i*CHUNK_SIZE
    shape = (min(CHUNK_SIZE, n_embeddings-offset), D)
    n_bytes_per_elem = np.dtype(np.float16).itemsize
    shard = np.memmap(works_path, dtype=np.float16, mode='r', offset=offset*D*n_bytes_per_elem, shape=shape)
    shard = shard.astype(np.float32, copy=True)

    faiss_ids = np.arange(offset, offset+shape[0])
    in_test_set = np.isin(faiss_ids, idxs_queries)
    faiss_ids = faiss_ids[~in_test_set]
    shard = shard[~in_test_set]

    index.add_with_ids(shard, faiss_ids)
    purge_from_memory(shard)
index = faiss.index_gpu_to_cpu(index)
faiss.write_index(index, index_path)


print('Evaluating index...')

embeddings = np.memmap(works_path, dtype=np.float16, mode='r').reshape(-1, D)
embeddings_test = embeddings[idxs_queries].astype(np.float32, copy=True)
purge_from_memory(embeddings) # after getting the test set, we don't need the whole embeddings array anymore

tuning_criterion = faiss.IntersectionCriterion(TEST_SIZE, 10)
tuning_criterion.set_groundtruth(None, idxs_gt)

index = faiss.read_index(index_path)
param_space = faiss.ParameterSpace()
param_space.initialize(index)
param_values = param_space.explore(index, embeddings_test, tuning_criterion)

param_values.display()


print('Adding test set to index...')
index = faiss.read_index(index_path)
index = faiss.index_cpu_to_gpu(gpu_env, 0, index, gpu_options)
index.add_with_ids(embeddings_test, idxs_queries)
index = faiss.index_gpu_to_cpu(index)
faiss.write_index(index, index_path)