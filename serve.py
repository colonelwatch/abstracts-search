# serve.py
# Loads all completed shards and finds the most similar vector to a given query vector.

import requests
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

works_ids_path = 'abstracts-embeddings/idxs.txt'
with open(works_ids_path) as f:
    idxs = f.read().splitlines()
index = faiss.read_index('abstracts-embeddings/index.faiss')

ps = faiss.ParameterSpace()
ps.initialize(index)
ps.set_index_parameters(index, 'nprobe=32,ht=512')


def _recover_abstract(inverted_index):
    abstract_size = max([max(appearances) for appearances in inverted_index.values()])+1

    abstract = [None]*abstract_size
    for word, appearances in inverted_index.items(): # yes, this is a second iteration over inverted_index
        for appearance in appearances:
            abstract[appearance] = word

    abstract = [word for word in abstract if word is not None]
    abstract = ' '.join(abstract)
    return abstract

def search(query):
    global model, index, idxs

    query_embedding = model.encode(query)
    query_embedding = query_embedding.reshape(1, -1)
    distances, faiss_ids = index.search(query_embedding, 10)

    distances = distances[0]
    faiss_ids = faiss_ids[0]

    openalex_ids = [idxs[faiss_id] for faiss_id in faiss_ids]
    search_filter = f'openalex_id:{"|".join(openalex_ids)}'
    search_select = 'id,title,abstract_inverted_index,authorships,primary_location,publication_year,cited_by_count,doi'

    neighbors = [(distance, openalex_id) for distance, openalex_id in zip(distances, openalex_ids)]
    request_str = f'https://api.openalex.org/works?filter={search_filter}&select={search_select}'

    return neighbors, request_str

def execute_request(request_str):
    response = requests.get(request_str).json()
    return response

def format_response(neighbors, response):
    response = {doc['id']: doc for doc in response['results']}
    
    result_string = ''
    for distance, openalex_id in neighbors:
        doc = response[openalex_id]

        # collect attributes from openalex doc for the given openalex_id
        title = doc['title']
        abstract = _recover_abstract(doc['abstract_inverted_index'])
        author_names = [authorship['author']['display_name'] for authorship in doc['authorships']]
        # journal_name = doc['primary_location']['source']['display_name']
        publication_year = doc['publication_year']
        citation_count = doc['cited_by_count']
        doi = doc['doi']

        # try to get journal name or else set it to None
        try:
            journal_name = doc['primary_location']['source']['display_name']
        except (TypeError, KeyError):
            journal_name = None

        # abstract: knock out escape sequences, then truncate to 1500 characters if necessary
        abstract = abstract.replace('\n', '\\n').replace('\r', '\\r')
        if len(abstract) > 2000:
            abstract = abstract[:2000] + '...'
        
        # authors: truncate to 3 authors if necessary
        if len(author_names) >= 3:
            authors_str = ', '.join(author_names[:3]) + ', ...'
        else:
            authors_str = ', '.join(author_names)

        
        entry_string = ''

        if doi: # edge case: for now, no doi -> no link
            entry_string += f'## [{title}]({doi})\n'
        else:
            entry_string += f'## {title}\n'
        
        if journal_name:
            entry_string += f'**{authors_str} - {journal_name}, {publication_year}**\n'
        else:
            entry_string += f'**{authors_str}, {publication_year}**\n'
        
        entry_string += f'{abstract}\n'
        
        if citation_count: # edge case: we shouldn't tack "Cited-by count: 0" onto someone's paper
            entry_string += f'*Cited-by count: {citation_count}*'
            entry_string += '&nbsp;&nbsp;&nbsp;&nbsp;'
        
        if doi: # list the doi if it exists
            entry_string += f'*DOI: {doi.replace("https://doi.org/", "")}*'
            entry_string += '&nbsp;&nbsp;&nbsp;&nbsp;'
        
        entry_string += f'*Similarity: {distance:.2f}*'
        entry_string += '&nbsp;&nbsp;&nbsp;&nbsp;\n'

        result_string += entry_string
    
    return result_string

with gr.Blocks() as demo:
    gr.Markdown('# abstracts-search demo')
    gr.Markdown(
        'Explore 21 million academic publications selected from the [OpenAlex](https://openalex.org) dataset. This '
        'project is an index of the embeddings generated from their titles and abstracts. The embeddings were '
        'generated using the `all-MiniLM-L6-v2` model provided by the [sentence-transformers](https://www.sbert.net/) '
        'module, and the index was built using the [faiss](https://github.com/facebookresearch/faiss) module.'
    )

    neighbors_var = gr.State()
    request_str_var = gr.State()
    response_var = gr.State()
    query = gr.Textbox(lines=1, placeholder='Enter your query here', show_label=False)
    btn = gr.Button('Search')
    with gr.Box():
        results = gr.Markdown()
    
    query.submit(search, inputs=[query], outputs=[neighbors_var, request_str_var]) \
        .success(execute_request, inputs=[request_str_var], outputs=[response_var]) \
        .success(format_response, inputs=[neighbors_var, response_var], outputs=[results])
    btn.click(search, inputs=[query], outputs=[neighbors_var, request_str_var]) \
        .success(execute_request, inputs=[request_str_var], outputs=[response_var]) \
        .success(format_response, inputs=[neighbors_var, response_var], outputs=[results])

demo.queue()
demo.launch()