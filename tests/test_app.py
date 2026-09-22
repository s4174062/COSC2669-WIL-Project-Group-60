import pytest

from config import EMPTY_QUESTION_MESSAGE, NO_RESULTS_MESSAGE
import app as baseline_app


HITS = [
    {
        "text": "Applications are submitted at least one working day before the due date.",
        "metadata": {
            "title": "Assessment and Assessment Flexibility Policy",
            "heading": "Extensions",
            "clauses": "35",
        },
        "distance": 0.31,
    }
]


def test_run_baseline_rejects_empty_question():
    with pytest.raises(ValueError, match="empty"):
        baseline_app.run_baseline("   ")


def test_cli_uses_same_hits_for_display_and_generation(monkeypatch):
    captured = {}

    def fake_retrieve(query, top_k=3):
        captured["query"] = query
        captured["top_k"] = top_k
        return HITS

    def fake_generate(question, context_chunks):
        captured["context"] = context_chunks
        return "Apply at least one working day before the due date."

    monkeypatch.setattr(baseline_app, "retrieve_hits", fake_retrieve)
    monkeypatch.setattr(baseline_app, "generate_answer", fake_generate)

    result = baseline_app.run_baseline("How long before the due date?", top_k=3)

    assert captured["top_k"] == 3
    assert captured["context"] == [HITS[0]["text"]]
    assert result["hits"] == HITS
    assert result["answer"] == "Apply at least one working day before the due date."
    assert result["hits"][0]["text"] == captured["context"][0]


def test_run_baseline_no_results_message(monkeypatch):
    monkeypatch.setattr(baseline_app, "retrieve_hits", lambda query, top_k=3: [])
    result = baseline_app.run_baseline("What is the capital of France?")
    assert result["hits"] == []
    assert result["answer"] == NO_RESULTS_MESSAGE
    assert EMPTY_QUESTION_MESSAGE
