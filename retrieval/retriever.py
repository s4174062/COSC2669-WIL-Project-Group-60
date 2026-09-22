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


def retrieve_hits(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Query Chroma and return structured hits: text, metadata, distance."""
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
    query = "How long before an assessment due date do I need to apply for an extension?"
    for i, hit in enumerate(retrieve_hits(query)):
        meta = hit["metadata"]
        print(f"--- result {i} | {meta.get('title')} | {meta.get('heading')} ---")
        print(hit["text"])
        print()
