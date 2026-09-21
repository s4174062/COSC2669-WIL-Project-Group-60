"""
Embeds text chunks and stores them in a local Chroma collection.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "policy_chunks"
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(REPO_ROOT / "chroma_db")


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def reset_collection():
    """Drop and recreate the policy collection so ingest can rebuild cleanly."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(COLLECTION_NAME)


def add_chunks(chunks: list[str], source: str = "unknown", metadatas: list[dict] | None = None):
    """Embed and add a list of text chunks to the vector store."""
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()

    embeddings = model.encode(chunks).tolist()
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
    model = SentenceTransformer(MODEL_NAME)
    documents = [record["text"] for record in records]
    ids = [record["id"] for record in records]
    metadatas = [record["metadata"] for record in records]
    embeddings = model.encode(documents).tolist()

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
