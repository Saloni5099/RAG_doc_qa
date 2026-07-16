# step3_embed.py
# Goal: convert text into numeric vectors and prove that similar
# sentences produce similar vectors

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads once, ~80MB
# This loads a pre-trained AI model whose only job is: take in text, output a list of numbers that represents its meaning. Think of it like a translator that converts sentences into a "meaning code" that computers can compare mathematically.
sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",       # similar meaning to sentence 1
    "The stock market crashed today."    # unrelated meaning
]

embeddings = model.encode(sentences)

print(f"Each embedding has {len(embeddings[0])} numbers\n")

sim_matrix = cosine_similarity(embeddings)

print("Similarity between sentence 1 and 2 (should be HIGH):", sim_matrix[0][1])
print("Similarity between sentence 1 and 3 (should be LOW):", sim_matrix[0][2])