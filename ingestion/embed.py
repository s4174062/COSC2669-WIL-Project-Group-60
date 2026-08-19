"""
Embeds text chunks and stores them in a local Chroma collection.
"""

import chromadb
from sentence_transformers import SentenceTransformer

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
    metadatas = [{"source": source} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"Added {len(chunks)} chunks from '{source}' to the vector store.")


if __name__ == "__main__":
    #test
    test_chunks = [
        "Special Consideration applications must be submitted within two "
        "working days of the assessment date.",
        "Extensions of up to seven days can be granted by course "
        "coordinators without supporting documentation.",
    ]
    add_chunks(test_chunks, source="placeholder_policy")
