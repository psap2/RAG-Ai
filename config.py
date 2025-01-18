import os
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai
import chromadb

def initialize_config():
    load_dotenv()
    ai_key = os.getenv("AI_KEY")
    genai.configure(api_key=ai_key)
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=ai_key)
    
    chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
    collection_name = "document_qa_collection"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=google_ef
    )
    return chroma_client, collection, google_ef, collection_name
