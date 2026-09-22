"""
Baseline CLI: retrieve once, show those sources, then generate from the same hits.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DEFAULT_TOP_K,
    EMPTY_QUESTION_MESSAGE,
    NO_RESULTS_MESSAGE,
    EmptyCollectionError,
    OllamaModelMissingError,
    OllamaNotRunningError,
    add_project_paths,
)

add_project_paths()

from generate import generate_answer
from retriever import retrieve_hits


def run_baseline(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Run one baseline retrieve-then-generate call.

    Retrieval happens once. The same hits are used for display and generation.
    """
    if question is None or not str(question).strip():
        raise ValueError(EMPTY_QUESTION_MESSAGE)

    hits = retrieve_hits(question, top_k=top_k)
    if not hits:
        return {
            "answer": NO_RESULTS_MESSAGE,
            "hits": [],
            "top_k": top_k,
        }

    answer = generate_answer(question, [hit["text"] for hit in hits])
    return {
        "answer": answer,
        "hits": hits,
        "top_k": top_k,
    }


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    return run_baseline(question, top_k=top_k)["answer"]


def _print_sources(hits: list[dict]) -> None:
    print("\nRetrieved from:")
    for hit in hits:
        meta = hit["metadata"]
        location = meta.get("heading") or meta.get("subsection") or meta.get("section")
        print(f"  - {meta.get('title')} | {location} (clauses {meta.get('clauses') or 'n/a'})")


if __name__ == "__main__":
    print("RMIT policy assistant (baseline RAG). Type 'quit' to exit.\n")
    while True:
        q = input("Ask a question: ")
        if q.strip().lower() == "quit":
            break
        try:
            result = run_baseline(q, top_k=DEFAULT_TOP_K)
        except ValueError as exc:
            print(f"\n{exc}\n")
            continue
        except (EmptyCollectionError, OllamaNotRunningError, OllamaModelMissingError) as exc:
            print(f"\n{exc}\n")
            continue

        if not result["hits"]:
            print(f"\n{result['answer']}\n")
            continue
        _print_sources(result["hits"])
        print("\n" + result["answer"] + "\n")
