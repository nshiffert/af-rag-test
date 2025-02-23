# backend.py

import streamlit as st
from pinecone import Pinecone  # New Pinecone client class
import openai

# --- Configuration ---
# Pinecone settings
PINECONE_API_KEY = st.secrets["general"]["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "auctic-rag"  # Correct index name from your dashboard
PINECONE_ENVIRONMENT = "us-east-1-aws"  # Updated environment value for AWS

# OpenAI settings
OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Initialize Pinecone Client ---
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Retrieve the index instance using the correct index name.
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text: str):
    """
    Generate an embedding for the input text using OpenAI.
    """
    openai.api_key = OPENAI_API_KEY
    response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
    embedding = response["data"][0]["embedding"]
    return embedding

def search_video_segments(embedding: list, top_k: int = 5):
    """
    Query the Pinecone index using the provided embedding and return matching segments.
    """
    query_response = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return query_response["matches"]

# python -c 'import backend; print(backend.delete_all_pinecone_vectors())'
def delete_all_pinecone_vectors():

    # Delete all vectors
    index.delete(delete_all=True)

    print("All vectors deleted successfully.")
