"""
Sanity-checks eval/test_questions.json WITHOUT needing Ollama or the vector
store, so teammates can run it every time they add questions:

    python eval/validate_test_set.py

It reports:
  - schema problems (missing/wrongly-typed fields, duplicates)
  - the balance of ambiguous vs unambiguous vs out-of-scope questions
  - whether each ambiguous question's scripted clarification_response
    actually resolves every ambiguity trigger in one go (if it doesn't,
    the conversational eval would loop and give up)
  - a quick preview of where the rule-based ambiguity detector disagrees
    with the labels (these are detector misses to expect in the eval, or
    labels worth a second look — not schema errors)
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "verification"))
from ambiguity_check import check_ambiguity

DEFAULT_PATH = os.path.join(HERE, "test_questions.json")

REQUIRED_FIELDS = {
    "question": str,
    "is_ambiguous": bool,
    "expected_clarification": (str, type(None)),
    "clarification_response": (str, type(None)),
    "expected_answer": (str, type(None)),
    "source_doc": str,
}


def validate(path: str = DEFAULT_PATH) -> int:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    errors = []
    warnings = []
    seen_questions = set()

    for i, item in enumerate(items, start=1):
        label = f"#{i} \"{str(item.get('question', ''))[:50]}\""

        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in item:
                errors.append(f"{label}: missing field '{field}'")
            elif not isinstance(item[field], expected_type):
                errors.append(f"{label}: field '{field}' has the wrong type")

        q = item.get("question", "")
        if not isinstance(q, str) or not q.strip():
            errors.append(f"{label}: question is empty")
            continue
        if q.strip().lower() in seen_questions:
            errors.append(f"{label}: duplicate question")
        seen_questions.add(q.strip().lower())

        if item.get("is_ambiguous") is True:
            response = item.get("clarification_response")
            if not response:
                errors.append(f"{label}: ambiguous question needs a clarification_response")
            else:
                combined = f"{q} {response}"
                still_ambiguous = check_ambiguity(combined, style="direct")
                if still_ambiguous:
                    errors.append(
                        f"{label}: clarification_response does not resolve every ambiguity "
                        f"(detector would still ask: \"{still_ambiguous[:70]}...\")"
                    )
            if not item.get("expected_clarification"):
                warnings.append(f"{label}: ambiguous question has no expected_clarification note")
        else:
            if item.get("clarification_response"):
                warnings.append(f"{label}: unambiguous question has a clarification_response (unused)")
            if not item.get("expected_answer"):
                warnings.append(f"{label}: unambiguous question has no expected_answer")

    # Balance summary
    total = len(items)
    ambiguous = sum(1 for it in items if it.get("is_ambiguous") is True)
    out_of_scope = sum(
        1 for it in items
        if str(it.get("expected_answer") or "").lower().startswith("i don't have enough information")
    )
    answerable = total - ambiguous - out_of_scope
    by_source = Counter(it.get("source_doc", "?") for it in items)

    print(f"Test set: {path}")
    print(f"  Total questions:         {total}")
    print(f"  Ambiguous:               {ambiguous}")
    print(f"  Clearly answerable:      {answerable}")
    print(f"  Out-of-scope (refusal):  {out_of_scope}")
    print("  By source document:")
    for source, count in sorted(by_source.items()):
        print(f"    {count:>3}  {source}")

    if total and ambiguous / total < 0.25:
        warnings.append(
            f"only {ambiguous}/{total} questions are ambiguous; recall on so few "
            "is not meaningful. Aim for roughly a third."
        )
    if ambiguous and answerable and ambiguous < 3:
        warnings.append("fewer than 3 ambiguous questions: precision/recall will swing wildly on a single miss")

    # Detector preview (no LLM needed)
    tp = fp = tn = fn = 0
    disagreements = []
    for item in items:
        detected = check_ambiguity(item.get("question", ""), style="direct") is not None
        labelled = item.get("is_ambiguous") is True
        if labelled and detected:
            tp += 1
        elif labelled and not detected:
            fn += 1
            disagreements.append(f"labelled ambiguous but detector misses it: \"{item['question']}\"")
        elif not labelled and detected:
            fp += 1
            disagreements.append(f"labelled unambiguous but detector asks anyway: \"{item['question']}\"")
        else:
            tn += 1
    print("\n  Rule-based detector vs labels (preview, no LLM):")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    for d in disagreements:
        print(f"    - {d}")
    if fn:
        print("    (misses mean ambiguity_check.py needs a pattern for that question)")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ! {w}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  X {e}")
        return 1

    print("\nNo schema errors.")
    return 0


if __name__ == "__main__":
    sys.exit(validate(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH))
