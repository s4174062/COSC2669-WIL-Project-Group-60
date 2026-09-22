"""
Takes retrieved chunks + a question, builds a strict grounded prompt,
and calls a local Ollama model.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama

from config import (
    GENERATION_MODEL,
    OLLAMA_MODEL_MISSING_MESSAGE,
    OLLAMA_NOT_RUNNING_MESSAGE,
    OllamaModelMissingError,
    OllamaNotRunningError,
)

MODEL_NAME = GENERATION_MODEL

PROMPT_TEMPLATE = """You are a university policy assistant. Answer the question
using ONLY the context below. If the answer is not contained in the context,
say "I don't have enough information to answer that" rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def _raise_ollama_error(exc: Exception) -> None:
    """Map known Ollama failures; re-raise anything unexpected."""
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    connection_markers = (
        "connection refused",
        "connecterror",
        "connect error",
        "connectionerror",
        "failed to connect",
        "winerror 10061",
        "actively refused",
        "name or service not known",
    )
    if any(marker in message or marker in type(exc).__name__.lower() for marker in connection_markers):
        raise OllamaNotRunningError(OLLAMA_NOT_RUNNING_MESSAGE) from exc
    if status == 404 or "not found" in message or ("pull" in message and "model" in message):
        raise OllamaModelMissingError(OLLAMA_MODEL_MISSING_MESSAGE) from exc
    raise exc


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        _raise_ollama_error(exc)

    return response["message"]["content"]


if __name__ == "__main__":
    fake_context = [
        "Special Consideration applications must be submitted within two "
        "working days of the assessment date."
    ]
    answer = generate_answer(
        "How long do I have to apply for special consideration?", fake_context
    )
    print(answer)
