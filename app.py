from config import initialize_config
from document_management import load_documents_from_directory, split_text
from embedding_management import create_embedding

chroma_client, collection, google_ef, collection_name = initialize_config()

#load articles
directory = "./news_articles"
documents = load_documents_from_directory(directory)
print(len(documents))

#create chunks for articles
chunked_documents = []
for document in documents:
    chunks = split_text(document["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{document['id']}_chunk{i+1}", "text": chunk})

