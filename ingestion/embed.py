"""
Embeds text chunks and stores them in a local Chroma collection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL

MODEL_NAME = EMBEDDING_MODEL
_EMBEDDING_MODEL = None


def get_embedding_model():
    """Load MiniLM once and reuse it for ingest and retrieval."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDING_MODEL


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def reset_collection():
    """Drop and recreate the policy collection so ingest can rebuild cleanly."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(COLLECTION_NAME)


def add_chunks(chunks: list[str], source: str = "unknown", metadatas: list[dict] | None = None):
    """Embed and add a list of text chunks to the vector store."""
    model = get_embedding_model()
    collection = get_collection()

    encoded = model.encode(chunks)
    embeddings = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    ids = [f"{source}_{i}" for i in range(len(chunks))]
    if metadatas is None:
        metadatas = [{"source": source} for _ in chunks]
    else:
        metadatas = [{**{"source": source}, **meta} for meta in metadatas]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"Added {len(chunks)} chunks from '{source}' to the vector store.")


def replace_all_chunks(records: list[dict]):
    """Rebuild the collection from records with text/id/metadata keys."""
    if not records:
        raise ValueError("No chunks to index.")

    collection = reset_collection()
    model = get_embedding_model()
    documents = [record["text"] for record in records]
    ids = [record["id"] for record in records]
    metadatas = [record["metadata"] for record in records]
    encoded = model.encode(documents)
    embeddings = encoded.tolist() if hasattr(encoded, "tolist") else encoded

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"Indexed {len(records)} chunks into '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    print("Placeholder embedding is disabled. Run ingestion/ingest.py instead.")
