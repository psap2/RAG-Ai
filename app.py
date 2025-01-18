import os
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

#configuration of gemini and embedding function
load_dotenv()
ai_key = os.getenv("AI_KEY")
genai.configure(api_key=ai_key)
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=ai_key)

# initializing a persistent (want to store data to reuse collections query) chroma client
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=google_ef
)

#load document from file directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

#split texts into chunks for vectorization
def split_text(text, chunk_size = 1000, overlap = 20): #want overlap in order to maintain contextual meaning of text in between chunks
    chunks = []
    start = 0 
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks 

#load_documents
directory = "./news_articles"
documents = load_documents_from_directory(directory)
print(len(documents)) 

#creating chunks
chunked_documents = []
for document in documents:
    chunks = split_text(document["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{document['id']}_chunk{i+1}", "text": chunk})
