"""
Single pipeline, two configurations:
  - verification_enabled=False -> baseline: retrieve, generate, done
  - verification_enabled=True  -> enhanced: ambiguity check, then
    retrieve/generate, then faithfulness check that can WITHHOLD the answer

The faithfulness check acts on its result: if too few of the answer's
sentences are supported by the retrieved context (score below
faithfulness_threshold), the draft answer is replaced with the standard
"I don't have enough information" refusal instead of being shown.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "generation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "verification"))

from retriever import retrieve
from generate import generate_answer
from ambiguity_check import check_ambiguity
from faithfulness_check import check_faithfulness

#fraction of answer sentences that must be supported by retrieved
#context for the answer to be shown. The overlap checker is crude (e.g. a
#hedging preamble like "Based on the context, I would say..." counts as
#unsupported), so this is lenient.
FAITHFULNESS_THRESHOLD = 0.5

REFUSAL_TEXT = "I don't have enough information to answer that."


def ask(
    question: str,
    verification_enabled: bool = False,
    top_k: int = 3,
    ambiguity_style: str = "rephrase",
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
) -> dict:
    """
    Returns a dict describing what happened, so the eval harness can
    inspect the decision path, not just the final text:
    {
        "type": "clarification" | "answer" | "no_context",
        "text": str,                    # what the user actually sees
        "faithfulness": dict | None,    # score of the DRAFT answer (enhanced only)
        "suppressed": bool,             # True if the draft was withheld
        "original_answer": str | None,  # the withheld draft, if any
        "context_used": list[str],
    }

    ambiguity_style: "rephrase" (default) asks the user to restate the
    whole question — appropriate here since this pipeline has no memory
    between calls. pipeline_conversational.py overrides this to "direct"
    since it merges the answer automatically.
    """
    if verification_enabled:
        clarification = check_ambiguity(question, style=ambiguity_style)
        if clarification:
            return {
                "type": "clarification",
                "text": clarification,
                "faithfulness": None,
                "suppressed": False,
                "original_answer": None,
                "context_used": [],
            }

    context_chunks = retrieve(question, top_k=top_k)
    if not context_chunks:
        return {
            "type": "no_context",
            "text": REFUSAL_TEXT,
            "faithfulness": None,
            "suppressed": False,
            "original_answer": None,
            "context_used": [],
        }

    answer = generate_answer(question, context_chunks)

    faithfulness = None
    suppressed = False
    original_answer = None
    if verification_enabled:
        faithfulness = check_faithfulness(answer, context_chunks)
        if faithfulness["score"] < faithfulness_threshold:
            suppressed = True
            original_answer = answer
            answer = REFUSAL_TEXT

    return {
        "type": "answer",
        "text": answer,
        "faithfulness": faithfulness,
        "suppressed": suppressed,
        "original_answer": original_answer,
        "context_used": context_chunks,
    }


def print_result(result: dict) -> None:
    """Shared console output for the interactive loops."""
    print(f"\n[{result['type']}] {result['text']}")
    if result["faithfulness"]:
        print(f"  faithfulness score: {result['faithfulness']['score']}")
        if result["faithfulness"]["unsupported_sentences"]:
            print(f"  unsupported: {result['faithfulness']['unsupported_sentences']}")
    if result["suppressed"]:
        print("  (draft answer withheld: not enough of it is supported by the retrieved sources)")
        print(f"  withheld draft: {result['original_answer']}")
    print()


if __name__ == "__main__":
    print("Policy Assistant. Type 'quit' to exit.")
    print("Commands: 'baseline <question>' or 'enhanced <question>'\n")
    while True:
        raw = input("> ")
        if raw.strip().lower() == "quit":
            break
        if raw.startswith("baseline "):
            result = ask(raw[len("baseline "):], verification_enabled=False)
        elif raw.startswith("enhanced "):
            result = ask(raw[len("enhanced "):], verification_enabled=True)
        else:
            print("Prefix your question with 'baseline ' or 'enhanced '.\n")
            continue

        print_result(result)
