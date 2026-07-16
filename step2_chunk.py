# step2_chunk.py
# Goal: split the document into overlapping word chunks

def chunk_text(text, chunk_size=60, overlap=10):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but overlap a bit
    return chunks

with open("documents/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks\n")
for i, c in enumerate(chunks):
    print(f"--- Chunk {i} ({len(c.split())} words) ---")
    print(c)
    print()