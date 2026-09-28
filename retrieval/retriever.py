"""
Given a query, returns the top-k most similar chunks from the vector store.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DEFAULT_TOP_K,
    EMPTY_COLLECTION_MESSAGE,
    EmptyCollectionError,
    add_project_paths,
)

add_project_paths()

from embed import get_collection, get_embedding_model

# Load the embedding model once and reuse it, rather than reloading it on
# every query (slow, and it floods the console with loading messages).
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

# Load the embedding model once and reuse it, rather than reloading it on
# every query (slow, and it floods the console with loading messages).
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def retrieve(query: str, top_k: int = 3) -> list[str]:
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise EmptyCollectionError(EMPTY_COLLECTION_MESSAGE)

    model = get_embedding_model()
    encoded = model.encode([query])
    query_embedding = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results.get("distances") else [None] * len(documents)

    hits = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append(
            {
                "text": doc,
                "metadata": meta or {},
                "distance": dist,
            }
        )
    return hits


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
    """Compatibility helper: return only passage text from retrieve_hits()."""
    return [hit["text"] for hit in retrieve_hits(query, top_k=top_k)]


if __name__ == "__main__":
    #requires embed.py to run first so the store isn't empty
    query = "How long do I have to apply for special consideration?"
    for i, chunk in enumerate(retrieve(query)):
        print(f"--- result {i} ---\n{chunk}\n")
