import google.generativeai as genai

def create_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text)
    embedding = result['embedding']
    return embedding
