# build_index.py
# Combines Steps 1-4: Load -> Chunk -> Embed -> Store

import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DOCUMENTS_DIR = "documents"
DB_DIR = "chroma_db"
COLLECTION_NAME = "my_documents"


def load_documents(folder):
    """Read every .txt and .pdf file in the folder."""
    docs = {}
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs[filename] = f.read()

        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            docs[filename] = text

    return docs


def chunk_text(text, chunk_size=120, overlap=20):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def main():
    print("Step 1: Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)
    print(f"  Loaded {len(documents)} document(s): {list(documents.keys())}")

    print("Step 2: Chunking documents...")
    all_chunks, all_ids, all_metadatas = [], [], []

    for filename, text in documents.items():
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_chunk_{i}")
            all_metadatas.append({"source": filename, "chunk_index": i})

    print(f"  Created {len(all_chunks)} chunks total.")

    print("Step 3: Embedding chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    print("Step 4: Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # Delete old collection if it exists, so re-running doesn't duplicate data
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(ids=all_ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas)

    print(f"\nDone. {len(all_chunks)} chunks indexed and saved to '{DB_DIR}/'.")


if __name__ == "__main__":
    main()