# step5_search.py
# Goal: take a question, embed it, and find the most similar stored chunks

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="my_documents")

question = "Why do we split documents into chunks?"
question_embedding = model.encode([question]).tolist()

results = collection.query(query_embeddings=question_embedding, n_results=2)

print(f"Question: {question}\n")
print("Most relevant chunks found:\n")
for doc in results["documents"][0]:
    print("-", doc[:150], "...\n")