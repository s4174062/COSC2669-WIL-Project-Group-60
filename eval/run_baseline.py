"""
Run every labelled question through baseline RAG and record the outputs.

Does not score correctness. Manual review uses the saved file.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    EVAL_QUESTIONS_PATH,
    EVAL_RESULTS_DIR,
    GENERATION_MODEL,
    add_project_paths,
)

add_project_paths()

from app import run_baseline


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _serialise_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "text": hit.get("text"),
            "metadata": hit.get("metadata") or {},
            "distance": hit.get("distance"),
        }
        for hit in hits
    ]


def evaluate_questions(questions: list[dict], top_k: int = DEFAULT_TOP_K) -> dict:
    rows = []
    for index, item in enumerate(questions):
        question = item.get("question", "")
        row = {
            "index": index,
            "question": question,
            "expected_answer": item.get("expected_answer"),
            "expected_source": item.get("source_doc"),
            "question_type": item.get("type"),
            "generated_response": None,
            "retrieved_passages": [],
            "retrieved_metadata": [],
            "similarity_distances": [],
            "model_name": GENERATION_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": top_k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }
        try:
            result = run_baseline(question, top_k=top_k)
            hits = result.get("hits") or []
            row["generated_response"] = result.get("answer")
            row["retrieved_passages"] = [hit.get("text") for hit in hits]
            row["retrieved_metadata"] = [hit.get("metadata") or {} for hit in hits]
            row["similarity_distances"] = [hit.get("distance") for hit in hits]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        rows.append(row)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": top_k,
        "question_count": len(questions),
        "error_count": sum(1 for row in rows if row["error"]),
        "results": rows,
    }


def save_results(payload: dict) -> Path:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = EVAL_RESULTS_DIR / f"baseline_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    questions = load_questions()
    payload = evaluate_questions(questions)
    output_path = save_results(payload)
    print(f"Wrote {payload['question_count']} baseline results to {output_path}")
    if payload["error_count"]:
        print(f"Recorded errors for {payload['error_count']} question(s).")
