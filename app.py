import os
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

load_dotenv()

ai_key = os.getenv("AI_KEY")

genai.configure(api_key=ai_key)
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=ai_key)

# Initialize a persistent chroma client
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=google_ef
)