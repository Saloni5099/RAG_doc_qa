# step6_generate_gemini.py
# Same as before, but using Google's free Gemini API instead of Claude

import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="my_documents")

question = "Why do we split documents into chunks?"

# Retrieve
question_embedding = model.encode([question]).tolist()
results = collection.query(query_embeddings=question_embedding, n_results=2)
retrieved_chunks = results["documents"][0]

# Build prompt
context = "\n\n".join(retrieved_chunks)
prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:"""

# Generate
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
response = gemini_model.generate_content(prompt)

print("Answer:", response.text)