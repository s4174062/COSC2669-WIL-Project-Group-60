"""
Simple CLI entry point that ties the pipeline together end to end.
Run ingestion/ingest.py first so the vector store contains the four policies.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "generation"))

from retriever import retrieve, retrieve_hits
from generate import generate_answer


def ask(question: str, top_k: int = 3) -> str:
    context_chunks = retrieve(question, top_k=top_k)
    if not context_chunks:
        return "I don't have enough information to answer that."
    return generate_answer(question, context_chunks)


if __name__ == "__main__":
    print("RMIT policy assistant (baseline RAG). Type 'quit' to exit.\n")
    while True:
        q = input("Ask a question: ")
        if q.strip().lower() == "quit":
            break
        hits = retrieve_hits(q)
        if not hits:
            print("\nI don't have enough information to answer that.\n")
            continue
        print("\nRetrieved from:")
        for hit in hits:
            meta = hit["metadata"]
            location = meta.get("heading") or meta.get("subsection") or meta.get("section")
            print(f"  - {meta.get('title')} | {location} (clauses {meta.get('clauses') or 'n/a'})")
        print("\n" + generate_answer(q, [hit["text"] for hit in hits]) + "\n")
