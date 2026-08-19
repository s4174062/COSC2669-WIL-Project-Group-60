"""
Given a query, returns the top-k most similar chunks from the vector store.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))

from sentence_transformers import SentenceTransformer
from embed import get_collection, MODEL_NAME


def retrieve(query: str, top_k: int = 3) -> list[str]:
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    #requires embed.py to run first so the store isn't empty
    query = "How long do I have to apply for special consideration?"
    for i, chunk in enumerate(retrieve(query)):
        print(f"--- result {i} ---\n{chunk}\n")
