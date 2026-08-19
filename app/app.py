"""
Simple CLI entry point that ties the pipeline together end to end
Can be run once ingestion/embed.py has populated the vector store
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "generation"))

from retriever import retrieve
from generate import generate_answer


def ask(question: str, top_k: int = 3) -> str:
    context_chunks = retrieve(question, top_k=top_k)
    if not context_chunks:
        return "I don't have enough information to answer that."
    return generate_answer(question, context_chunks)


if __name__ == "__main__":
    print("Policy Assistant (placeholder data). Type 'quit' to exit.\n")
    while True:
        q = input("Ask a question: ")
        if q.strip().lower() == "quit":
            break
        print("\n" + ask(q) + "\n")
