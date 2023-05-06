# serve.py
# Loads all completed shards and finds the most similar vector to a given query vector.

import re
import requests
from sentence_transformers import SentenceTransformer
import faiss
import openai
import gradio as gr

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

works_ids_path = 'abstracts-embeddings/idxs.txt'
with open(works_ids_path) as f:
    idxs = f.read().splitlines()
index = faiss.read_index('abstracts-embeddings/index.faiss')

ps = faiss.ParameterSpace()
ps.initialize(index)
ps.set_index_parameters(index, 'nprobe=16,ht=256')


def generate_pseudodocument(query, api_key):
    openai.api_key = api_key
    prompt = open('_prompt.md').read()

    parsed_output = None
    attempts_count = 0
    while not parsed_output and attempts_count < 5:
        raw_output = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt.replace('[REP]', query),
            # temperature=0.7,
            temperature=1.0,
            max_tokens=300,
            # top_p=1,
            top_p = 0.95,
            frequency_penalty=0,
            presence_penalty=0
        ).choices[0].text
        parsed_output = re.search(
            f'.*\\n(?:T|t)itle ?::? ?(.*)\\n(?:T|t)ext ?::? ?(.*)', 
            raw_output
        )
        if not parsed_output:
            attempts_count += 1
            print('Warning: LLM output is not as expected:')
            print('Query:', query)
            print('Raw LLM output:')
            print(raw_output)
    if not parsed_output:
        raise Exception('Too many attempts to get a valid LLM output')

    title = parsed_output.group(1).strip()
    text = parsed_output.group(2).strip()
    document = {'title': title, 'text': text}

    return document

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

def search(query, api_key):
    global model, index
    
    pseudodocument = generate_pseudodocument(query, api_key)
    pseudodocument_string = f'"{pseudodocument["title"]}": {pseudodocument["text"]}'
    yield pseudodocument_string, 'Loading search results...'

    pseudodocument_probe = f'{pseudodocument["title"]} {pseudodocument["text"]}'
    query_embedding = model.encode(pseudodocument_probe)
    query_embedding = query_embedding.reshape(1, -1)
    distances, faiss_ids = index.search(query_embedding, 10)

    result_string = ''
    for distance, faiss_id in zip(distances[0], faiss_ids[0]):
        openalex_id = lookup_openalex_id(faiss_id)

        response = requests.get(f'https://api.openalex.org/works/{openalex_id}').json()
        title = response['title']
        inverted_index = response['abstract_inverted_index']
        abstract = recover_abstract(inverted_index)

        abstract = abstract.replace('\n', '\\n').replace('\r', '\\r')

        result_string += f'[{distance:.2f}] "{title}": {abstract}\n\n'
        yield pseudodocument_string, result_string

with gr.Blocks() as demo:
    gr.Markdown('# OpenAlex Generative Search Demo')
    gr.Markdown(
        'Enter your OpenAI API key and a query to search the OpenAlex corpus. '
        'The demo will use a GPT-3 model to generate a pseudodocument, which '
        'will then be used to search the OpenAlex corpus. The results will be '
        'sorted by cosine similarity between the pseudodocument and the abstracts '
        'of the OpenAlex documents.'
    )
    with gr.Row():
        with gr.Column():
            api_key = gr.Textbox(
                lines=1, 
                placeholder='Enter your OpenAI API key here', 
                label='OpenAI API Key'
            )
            query = gr.Textbox(
                lines=3, 
                placeholder='Enter your query here', 
                label='Query'
            )
            btn = gr.Button('Search')
            pseudodocument = gr.Textbox(
                max_lines=10, 
                placeholder='Awaiting query...', 
                label='Pseudodocument'
            )
        with gr.Column():
            results = gr.Textbox(
                max_lines=24, 
                placeholder='Awaiting query...', 
                label='Results'
            )
    
    btn.click(search, inputs=[query, api_key], outputs=[pseudodocument, results])

demo.queue()
demo.launch()