"""
Embeds text chunks and stores them in a local Chroma collection.

Run this to (re)build the vector store from every .txt file in data/raw/.
It clears any existing collection first so re-running is always a clean
rebuild rather than an accumulation of duplicate/stale chunks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(__file__))
from chunking import chunk_text

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "policy_chunks"
DB_PATH = "./chroma_db"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def add_chunks(chunks: list[str], source: str = "unknown"):
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


def document_title(text: str, fallback: str) -> str:
    """First non-empty line of the file, which is the policy's title."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return fallback


def ingest_all_raw_documents():
    collection = get_collection(reset=True)

    if not os.path.isdir(RAW_DATA_DIR):
        print(f"No data/raw directory found at {RAW_DATA_DIR}")
        return

    txt_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")]
    if not txt_files:
        print("No .txt files found in data/raw/ — nothing to embed.")
        return

    for filename in sorted(txt_files):
        path = os.path.join(RAW_DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        source_name = os.path.splitext(filename)[0]
        title = document_title(text, source_name)

        # Contextual header: put the document title in front of every chunk.
        # Policies use abbreviations ("LOA") inside the body, so a chunk like
        # "LOA is not normally available in the first semester" never contains
        # the words a student actually types ("leave of absence"). The title
        # restores that context for embedding and for the model reading it.
        chunks = [f"[{title}]\n{c}" for c in chunk_text(text)]
        add_chunks(chunks, source=source_name, collection=collection)

    print(f"\nDone. Total documents indexed: {len(txt_files)}")


if __name__ == "__main__":
    print("Placeholder embedding is disabled. Run ingestion/ingest.py instead.")
