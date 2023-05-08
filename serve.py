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


def lookup_openalex_id(idx):
    global idxs
    return idxs[idx]

def recover_abstract(inverted_index):
    abstract_size = max([max(appearances) for appearances in inverted_index.values()])+1

    abstract = [None]*abstract_size
    for word, appearances in inverted_index.items(): # yes, this is a second iteration over inverted_index
        for appearance in appearances:
            abstract[appearance] = word

    abstract = [word for word in abstract if word is not None]
    abstract = ' '.join(abstract)
    return abstract

def search(query):
    global model, index
    
    query_embedding = model.encode(query)
    query_embedding = query_embedding.reshape(1, -1)
    distances, faiss_ids = index.search(query_embedding, 10)

    result_string = ''
    for distance, faiss_id in zip(distances[0], faiss_ids[0]):
        openalex_id = lookup_openalex_id(faiss_id)

        response = requests.get(f'https://api.openalex.org/works/{openalex_id}').json()

        # collect attributes from openalex response for the given openalex_id
        title = response['title']
        abstract = recover_abstract(response['abstract_inverted_index'])
        author_names = [authorship['author']['display_name'] for authorship in response['authorships']]
        # journal_name = response['primary_location']['source']['display_name']
        publication_year = response['publication_year']
        citation_count = response['cited_by_count']
        doi = response['doi']

        # try to get journal name or else set it to None
        try:
            journal_name = response['primary_location']['source']['display_name']
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

        yield result_string

with gr.Blocks() as demo:
    gr.Markdown('# abstracts-search demo')
    gr.Markdown(
        'Explore 21 million academic publications selected from the [OpenAlex](https://openalex.org) dataset. This '
        'project is an index of the embeddings generated from their titles and abstracts. The embeddings were '
        'generated using the `all-MiniLM-L6-v2` model provided by the [sentence-transformers](https://www.sbert.net/) '
        'module, and the index was built using the [faiss](https://github.com/facebookresearch/faiss) module.'
    )
    query = gr.Textbox(lines=1, placeholder='Enter your query here', show_label=False)
    btn = gr.Button('Search')
    with gr.Box():
        results = gr.Markdown()
    
    btn.click(search, inputs=[query], outputs=[results])

demo.queue()
demo.launch()