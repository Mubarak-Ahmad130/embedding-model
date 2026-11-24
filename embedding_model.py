
from sentence_transformers import SentenceTransformer

# 1. Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create some sentences
sentences = [
    "I love machine learning",
    "Artificial intelligence is interesting",
    "I like eating pizza",
    "Deep learning is a part of machine learning"
]

# 3. Convert sentences into embeddings
embeddings = model.encode(sentences)

# 4. Print embeddings
print("Embeddings:")
print(embeddings)

# 5. Print embedding shape
print("\nEmbedding Shape:")
print(embeddings.shape)


model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love machine learning",
    "I enjoy studying artificial intelligence",
    "I like eating pizza"
]

embeddings = model.encode(sentences)

# Calculate similarity
similarity = model.similarity(embeddings, embeddings)

print(similarity)