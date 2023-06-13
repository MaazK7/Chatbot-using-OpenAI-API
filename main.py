# package the semantic search into an API with front end
# set up a flask app below

import collections
from flask import Flask, render_template, request
import openai
import pandas as pd
import numpy as np


global FILE_NAME
FILE_NAME = 'tgg.parquet'
EMBEDDINGS_MODEL = "text-embedding-ada-002"
CHAT_COMPLETIONS_MODEL = "gpt-3.5-turbo-0301"
openai.api_key = "" #Enter your OPENAI API KEY!!


def parse_dataset():
    """
    Parse a dataset of preprocessed text and embeddings.

    Returns:
    - An instance of the collections.namedtuple 'Engine', containing the following attributes:
        * page_text_corpus: a list of strings representing the preprocessed text data.
        * page_text_corpus_embeddings: a numpy array of shape (n, m) representing the precomputed embeddings for each text data.
        * page_numbers_list: a list of integers representing the page numbers for each text data.
        * chapter_number_list: a list of strings representing the chapter numbers for each text data.

    Example:
    >>> engine = parse_dataset()
    """

    print("Loading embedding file...")
    tgg_file = pd.read_parquet(FILE_NAME)
    print("Converting to np array...")
    page_numbers_list = tgg_file['page_number'].tolist()
    chapter_number_list = tgg_file['chapter'].tolist()
    page_text_corpus = tgg_file['page'].tolist()
    page_text_corpus_embeddings = np.array(tgg_file['embedding'].tolist(), dtype=float)
    print("done converting to np array")
    return collections.namedtuple('Engine', 
    ['page_text_corpus', 
    'page_text_corpus_embeddings', 
    'page_numbers_list',
    'chapter_number_list'])(
        page_text_corpus, 
        page_text_corpus_embeddings, 
        page_numbers_list,
        chapter_number_list)

def get_query_embedding_openai(prompt):
    """
    Generate a vector embedding for a given prompt using OpenAI's embeddings model.

    Args:
    - prompt: a string representing the input text to be embedded.

    Returns:
    - A numpy array representing the vector embedding for the input text.
    """
    response = openai.Embedding.create(
        model=EMBEDDINGS_MODEL,
        input=prompt
    )
    return response['data'][0]['embedding']

def prepare_contexts(dataset):
    """
    Create a dictionary of document section embeddings.

    Args:
    - dataset: contains preprocessed text data and their embeddings.

    Returns:
    - A dictionary where each key is a tuple representing a document section consisting of (page_text, page_number, chapter_number), 
    and each value is the corresponding embedding.
    """
    contexts = {}
    for page_text, page_number, chapter_number, embedding in zip(
        dataset.page_text_corpus, 
        dataset.page_numbers_list, 
        dataset.chapter_number_list, 
        dataset.page_text_corpus_embeddings
    ):
        contexts[(page_text, page_number, chapter_number)] = embedding
    return contexts

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query_embedding: str, contexts: dict[(str, int, int), np.array]) -> list[(float, (str, int, int))]:
    """
    Compare query embedding against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def get_semantic_suggestions(prompt):
    """
    Generate a list of semantic suggestions based on a given prompt.

    Args:
    - prompt: a string representing the user's query.

    Returns:
    - A list of dictionaries containing the top-k most relevant document sections to the query.
        Each dictionary has the following keys:
            * page: a string representing the text of the document section.
            * chapter_number: a string representing the chapter number of the document section.
            * page_number: an integer representing the page number of the document section.
    """
    dataset_with_embeddings = parse_dataset()
    query_embedding = np.array(get_query_embedding_openai(prompt), dtype=float)
    relevant_sections = order_document_sections_by_query_similarity(
        query_embedding, 
        prepare_contexts(dataset=dataset_with_embeddings)
    )
    top_three = relevant_sections[:3]
    final = []
    for _, (page_text, page_number, chapter_number) in top_three:
        final.append(
            {
                'page': page_text,
                'chapter_number': chapter_number,
                'page_number': page_number
            })
    return final 

# experimental version of the prompt construction function
def construct_completions_prompt_exp(question):
    """
    Construct a prompt for OpenAI's chatbot model that simulates a book research assistance scenario.

    Args:
    - question: a string representing the student's question to the research assistant.

    Returns:
    - A dictionary containing the user's prompt and the system's response.
        The dictionary has the following keys:
            * 'user': a string representing the student's question.
            * 'system': a string representing the research assistant's response, including the available context.
    """
    system_prompt = """
        You are assisting a student's book research for a paper. 
        The student will ask you questions about the book. You will answer the student's questions while referencing the book.
        The book references have been pre-processed and are available to you in the context section below.
        Do not refer to your prior knowledge on the subject. Only answer the student's questions with the excerpts in the context section below.
        The student requires citations so you must include the chapter and page number for each reference.
        
        Context:
        *insert text*
    """
    user_prompt = """
        Q: *insert question*
    """
    edited_user_prompt = user_prompt.replace("*insert question*", question)
    # on the next line write code similar to the previous line but for the "insert text" part
    page_results = get_semantic_suggestions(question)
    page_composite_string = ""
    for page_result in page_results:
        page_composite_string += f"'{page_result['page'].strip()}', (Chapter {page_result['chapter_number']}, Page {page_result['page_number']})\n"
    edited_system_prompt = system_prompt.replace("*insert text*", page_composite_string)
    return {"user": edited_user_prompt, "system": edited_system_prompt}

# experimental version of the answer function
def get_answer_exp(prompt_obj):
    """
    Generate an answer to a prompt using OpenAI's ChatGPT model.

    Args:
    - prompt_obj: a dictionary containing the user's prompt and the system's response.
        The dictionary should have the following keys:
            * 'user': a string representing the user's prompt.
            * 'system': a string representing the chatbot's response to the user's prompt.

    Returns:
    - A string representing the answer to the prompt.
    """
    return openai.ChatCompletion.create(
        messages=[
            {
                "role": "user",
                "content": prompt_obj['user']
            },
            {
                "role": "system",
                "content": prompt_obj['system']
            }
        ],
        model=CHAT_COMPLETIONS_MODEL,
        temperature=0.8
    )["choices"][0]["message"]["content"].strip(" \n")

def run_flask_app():
    """
    Runs a Flask app that listens for incoming HTTP requests at port 8080 on all network interfaces.
    The app has two routes:
    - '/' returns a rendered HTML template.
    - '/ask' accepts a POST request with a 'question' parameter, constructs a completion prompt expression
      using the question, passes the prompt expression to a function to retrieve an answer,
      combines the prompt and answer into a string, and returns the combined string as JSON.
    """
    app = Flask(__name__,template_folder='Template')

    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/ask', methods=['POST'])
    def ask():
        question = request.form.get('question')
        if question:
            prompt_obj = construct_completions_prompt_exp(question) 
            answer = get_answer_exp(prompt_obj)
            combined_prompt = f"User: {prompt_obj['user']}\nSystem: {prompt_obj['system']}"
            return {"answer": answer} 
        else:
              return render_template('index.html')
    app.run(debug=False, host="0.0.0.0", port=8080, threaded=True)


if __name__ == '__main__':
    run_flask_app()
