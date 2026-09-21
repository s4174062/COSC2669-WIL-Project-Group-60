"""
Ingest official RMIT policy HTML into chunks and the Chroma vector store.

Run from the repository root:

    python ingestion/ingest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INGESTION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INGESTION_DIR))

from chunk import chunk_clauses
from embed import MODEL_NAME, get_collection, replace_all_chunks
from extract import extract_from_file
from sentence_transformers import SentenceTransformer

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SOURCES_PATH = REPO_ROOT / "data" / "sources.json"

SANITY_QUERIES = [
    "How long before an assessment due date do I need to apply for an extension?",
    "What is the maximum length of a leave of absence?",
    "How is eligibility for a fee refund determined?",
    "Can I enrol after the published deadline?",
]


def load_sources() -> list[dict]:
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def ingest() -> list[dict]:
    sources = load_sources()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    processed_docs = []

    for source in sources:
        html_path = RAW_DIR / source["file"]
        if not html_path.exists():
            raise FileNotFoundError(f"Missing source file: {html_path}")

        clauses = extract_from_file(html_path)
        chunks = chunk_clauses(clauses, title=source["title"])
        doc_records = []
        for i, chunk in enumerate(chunks):
            record = {
                "id": f"{source['id']}_{i}",
                "text": chunk["text"],
                "metadata": {
                    "source": source["id"],
                    "title": source["title"],
                    "domain": source["domain"],
                    "url": source["url"],
                    "section": chunk["section"],
                    "subsection": chunk["subsection"],
                    "heading": chunk["heading"],
                    "clauses": chunk["clauses"],
                },
            }
            doc_records.append(record)
            all_records.append(record)

        processed_path = PROCESSED_DIR / f"{source['id']}.json"
        processed_path.write_text(
            json.dumps(
                {
                    "source": source,
                    "clause_count": len(clauses),
                    "chunk_count": len(doc_records),
                    "chunks": doc_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        processed_docs.append(
            {
                "id": source["id"],
                "title": source["title"],
                "clauses": len(clauses),
                "chunks": len(doc_records),
            }
        )
        print(f"{source['title']}: {len(clauses)} clauses -> {len(doc_records)} chunks")

    replace_all_chunks(all_records)
    summary_path = PROCESSED_DIR / "ingest_summary.json"
    summary_path.write_text(
        json.dumps({"documents": processed_docs, "total_chunks": len(all_records)}, indent=2),
        encoding="utf-8",
    )
    return all_records


def sanity_check(top_k: int = 3) -> None:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        print("Vector store is empty; skip retrieval check.")
        return

    model = SentenceTransformer(MODEL_NAME)
    print(f"\nRetrieval sanity check ({count} chunks, top-{top_k}):")
    for query in SANITY_QUERIES:
        embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        print(f"\nQ: {query}")
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            preview = " ".join(doc.split())[:180]
            print(
                f"  {i + 1}. {meta.get('title')} | {meta.get('section')} | "
                f"{meta.get('heading') or meta.get('subsection')} "
                f"(clauses {meta.get('clauses') or 'n/a'}, dist={dist:.3f})"
            )
            print(f"     {preview}...")


if __name__ == "__main__":
    ingest()
    sanity_check()
