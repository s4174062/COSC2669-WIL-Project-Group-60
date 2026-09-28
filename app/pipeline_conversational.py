"""
Conversational variant of the enhanced pipeline.

Where pipeline.py's "enhanced" config stops as soon as it detects
ambiguity (returns a clarification question and nothing more), this
version goes one step further: it obtains the user's clarifying
response, merges it into the original question, and re-runs the
pipeline to produce a final, resolved answer.

Deliberately reuses ask() from pipeline.py unchanged rather than
duplicating retrieval/generation/verification logic — the merged,
clarified question naturally passes the ambiguity check on the second
pass because it now contains a qualifying term (e.g. "exam"), so no
special-casing is needed anywhere else in the pipeline.

This gives you three comparable conditions for evaluation:
  1. baseline                 -> pipeline.ask(verification_enabled=False)
  2. enhanced (detect-only)   -> pipeline.ask(verification_enabled=True)
  3. enhanced (resolved)      -> ask_conversational() in this file
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
from pipeline import ask, print_result


def ask_conversational(
    question: str,
    verification_enabled: bool = True,
    top_k: int = 3,
    clarification_response_fn=None,
    max_rounds: int = 3,
) -> dict:
    """
    Runs the pipeline, looping through as many clarification rounds as
    needed (up to max_rounds) rather than just one. Some questions are
    ambiguous in more than one way at once — e.g. both the assessment
    type AND whether the student wants an extension vs. special
    consideration — and resolving one doesn't necessarily resolve the
    other. Each round merges the latest answer into the running question
    and re-checks; it only proceeds to retrieval/generation once
    check_ambiguity finds nothing left to clarify, or max_rounds is hit.

    Returns the final result dict with extra keys:
      "clarification_asked": bool
      "clarification_rounds": int (how many rounds actually happened)
      "merged_question": str | None (the final combined question, if any rounds happened)
    """
    current_question = question
    rounds = 0

    while True:
        # Merged multi-round queries are messier text than a clean single
        # question, so widen the retrieval net a little to compensate —
        # a partial mitigation, not a fix for the underlying limitation.
        effective_top_k = top_k if rounds == 0 else top_k + 2
        result = ask(current_question, verification_enabled=verification_enabled, top_k=effective_top_k, ambiguity_style="direct")

        if result["type"] != "clarification":
            result["clarification_asked"] = rounds > 0
            result["clarification_rounds"] = rounds
            result["merged_question"] = current_question if rounds > 0 else None
            return result

        rounds += 1
        if rounds > max_rounds:
            result["clarification_asked"] = True
            result["clarification_rounds"] = rounds - 1
            result["merged_question"] = current_question
            result["text"] = (
                result["text"]
                + f" (Note: still ambiguous after {max_rounds} clarification "
                "attempts — stopping rather than guessing.)"
            )
            return result

        if clarification_response_fn is not None:
            clarification_answer = clarification_response_fn(result["text"])
        else:
            print(f"\n[clarification, round {rounds}] {result['text']}")
            clarification_answer = input("> ")

        current_question = f"{current_question} {clarification_answer}"


if __name__ == "__main__":
    print("Policy Assistant (conversational). Type 'quit' to exit.")
    print("Commands: 'baseline <question>' or 'enhanced <question>'\n")
    while True:
        raw = input("> ")
        if raw.strip().lower() == "quit":
            break
        if raw.startswith("baseline "):
            result = ask_conversational(raw[len("baseline "):], verification_enabled=False)
        elif raw.startswith("enhanced "):
            result = ask_conversational(raw[len("enhanced "):], verification_enabled=True)
        else:
            print("Prefix your question with 'baseline ' or 'enhanced '.\n")
            continue

        if result["type"] == "clarification":
            print(f"\n[unresolved after {result['clarification_rounds']} rounds] {result['text']}\n")
        else:
            if result.get("clarification_asked"):
                print(f"\n(resolved after {result['clarification_rounds']} clarification round(s))")
            print_result(result)