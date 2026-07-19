# query.py
# Combines Steps 5-7: Retrieve -> Build prompt -> Generate answer
# Now lets you ask multiple questions in a loop

import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

DB_DIR = "chroma_db"
COLLECTION_NAME = "my_documents"
TOP_K = 5

client_gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def retrieve_relevant_chunks(question, model, collection, top_k=TOP_K):
    question_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    chunks = results["documents"][0]
#   print("\n--- DEBUG: Retrieved chunk text ---")
#   for c in chunks:
#      print(c[:300])
#      print("...")
#   print("--- END DEBUG ---\n")
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return chunks,sources


def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(chunks)
    return f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know based on the provided documents.

Context:
{context}

Question: {question}
Answer:"""


def main():
    print("Loading model and database...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Ready! Type a question, or 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        chunks, sources = retrieve_relevant_chunks(question, model, collection)
        print(f"\n[Retrieved from: {set(sources)}]")

        prompt = build_prompt(question, chunks)
        response = client_gemini.models.generate_content(
          model="gemini-2.5-flash",
          contents=prompt
        )
        print(f"\nAnswer: {response.text}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()