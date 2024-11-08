__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import argparse

# from flask import Flask, current_app, jsonify, request
# from flask_cors import CORS
from langchain_community.vectorstores.chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from services.rag.embedding import get_embedding_function

# ==========================
# Global ML Models Registry
# ==========================
checkpoint = "microsoft/Phi-3.5-mini-instruct"
device = "cuda" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)
generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    # "do_sample": False,
}

# =========
# Constants
# =========
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
# ===============
# Message History
# ===============
messages = [{"role": "system", "content": "You are an AI assistant."}]

def main():
    # Create LLM
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    query_text = input("Your query: ")
    response = query_rag(query_text)
    print(response)


def prompt_formatter(query: str) -> list[dict]:
    """
    Augments query with context items.
    """
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query, k=5)

    context = "- " + "\n- ".join([doc.page_content for doc, _score in results])

    # print(f"[DEBUG]: {context}")
    
    base_prompt = """Now use the following context items to answer the user query: {context}\n
    User query: {query}\n
    Answer:
    """
    base_prompt = base_prompt.format(context=context, query=query)
    
    return [{"role": "user", "content": base_prompt}]


def query_rag(query_text: str):
    prompt = prompt_formatter(query_text)
    messages += prompt

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']


main()
# app = Flask(__name__)
# CORS(app)

# @app.route("/generate", methods=["POST"])
# def generate():

#     global model
#     global tokenizer

#     data = request.get_json()

#     messages = [
#         {"role": "system", "content": "You are a news anchor"},
#         {"role": "user", "content": data["query"]}
#     ]

#     output = pipe(messages, **generation_args)

#     response = jsonify({"response": output[0]['generated_text']})
#     return response

