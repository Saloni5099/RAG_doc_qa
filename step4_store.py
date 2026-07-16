import chromadb
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# 1. Load
with open("documents/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Chunk
chunks = chunk_text(text)
ids = [f"chunk_{i}" for i in range(len(chunks))]

# 3. Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# 4. Store
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="my_documents")

collection.add(ids=ids, embeddings=embeddings, documents=chunks)

print(f"Stored {len(chunks)} chunks in ChromaDB.")
print("Check your folder — you should see a new 'chroma_db' directory.")