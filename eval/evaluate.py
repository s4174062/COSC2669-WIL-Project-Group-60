"""
Runs the labeled test set through THREE conditions on identical questions:

  1. baseline               - retrieve + generate, no verification
  2. enhanced (detect-only) - stateless pipeline; stops at the clarification
  3. enhanced (resolved)    - conversational pipeline; the scripted
                              clarification_response is fed back in and the
                              question is re-run to a final answer

and reports:
  - ambiguity detection precision/recall (condition 2 vs the labels)
  - whether clarification rounds actually ended in an answer (condition 3)
  - faithfulness of generated answers for BOTH baseline and enhanced, so the
    two can be compared, plus how many low-faithfulness answers each one
    actually showed the user
  - retrieval Hit@3 and NDCG@3 vs labelled source_doc (document-level)
  - a CSV with blank grade columns for manual correctness grading

Run from anywhere:  python eval/evaluate.py
Requires the vector store to be built (python ingestion/ingest.py) and Ollama
to be running. Redirect output to keep a record:
    python eval/evaluate.py > eval/results/run_output.txt
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))
sys.path.append(os.path.join(HERE, "..", "app"))
sys.path.append(os.path.join(HERE, "..", "verification"))
sys.path.append(os.path.join(HERE, "..", "retrieval"))
sys.path.append(os.path.join(HERE, "..", "ingestion"))
sys.path.append(HERE)

from config import DEFAULT_TOP_K, add_project_paths

add_project_paths()

from pipeline import ask, FAITHFULNESS_THRESHOLD
from pipeline_conversational import ask_conversational
from faithfulness_check import check_faithfulness
from retriever import retrieve_hits
from retrieval_metrics import labelled_source, score_hits

TEST_SET_PATH = os.path.join(HERE, "test_questions.json")
RESULTS_DIR = os.path.join(HERE, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "eval_results.csv")


def load_test_set(path: str = TEST_SET_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, digits: int = 3) -> str:
    return "n/a" if x is None or x != x else f"{x:.{digits}f}"


def mean(values: list):
    return sum(values) / len(values) if values else float("nan")


def make_scripted_responder(response):
    """Answers any clarification question with the scripted response."""
    fallback = "no further detail available"

    def responder(_clarification_text: str) -> str:
        return response if response else fallback

    return responder


def _retrieval_scores(question: str, source_doc: str | None, top_k: int) -> dict | None:
    expected = labelled_source(source_doc)
    if expected is None:
        return None
    hits = retrieve_hits(question, top_k=top_k)
    return score_hits(hits, expected, k=DEFAULT_TOP_K)


def run_baseline(question: str) -> dict:
    """Baseline has no verification layer, so score its answer here with the
    same checker the enhanced config uses; otherwise there is nothing to
    compare the enhanced config's faithfulness against."""
    result = ask(question, verification_enabled=False)
    score = None
    if result["type"] == "answer":
        score = check_faithfulness(result["text"], result["context_used"])["score"]
    result["faithfulness_score"] = score
    return result


def evaluate(test_path: str = TEST_SET_PATH, csv_path: str = CSV_PATH) -> None:
    test_set = load_test_set(test_path)

    tp = fp = tn = fn = 0
    rows = []

    baseline_scores = []            # raw faithfulness of baseline answers
    enhanced_scores = []            # raw faithfulness of enhanced (resolved) DRAFT answers
    baseline_low_delivered = 0      # baseline answers shown with score < threshold
    enhanced_low_delivered = 0      # enhanced answers shown with score < threshold
    enhanced_withheld = 0
    baseline_answered = 0
    enhanced_answered = 0

    clarified_questions = 0
    resolved_to_answer = 0

    baseline_ndcgs = []
    baseline_hits = []
    resolved_ndcgs = []
    resolved_hits = []

    for item in test_set:
        question = item["question"]
        expected_ambiguous = item["is_ambiguous"]

        baseline = run_baseline(question)
        detect_only = ask(question, verification_enabled=True, ambiguity_style="rephrase")
        resolved = ask_conversational(
            question,
            verification_enabled=True,
            clarification_response_fn=make_scripted_responder(item.get("clarification_response")),
        )

        detected = detect_only["type"] == "clarification"
        if expected_ambiguous and detected:
            tp += 1
        elif expected_ambiguous and not detected:
            fn += 1
        elif not expected_ambiguous and detected:
            fp += 1
        else:
            tn += 1

        if resolved.get("clarification_asked"):
            clarified_questions += 1
            if resolved["type"] == "answer":
                resolved_to_answer += 1

        # Baseline faithfulness
        if baseline["faithfulness_score"] is not None:
            baseline_answered += 1
            baseline_scores.append(baseline["faithfulness_score"])
            if baseline["faithfulness_score"] < FAITHFULNESS_THRESHOLD:
                baseline_low_delivered += 1

        # Enhanced (resolved) faithfulness: raw draft score, plus what was actually shown
        if resolved["type"] == "answer" and resolved["faithfulness"] is not None:
            enhanced_answered += 1
            draft_score = resolved["faithfulness"]["score"]
            enhanced_scores.append(draft_score)
            if resolved["suppressed"]:
                enhanced_withheld += 1
            elif draft_score < FAITHFULNESS_THRESHOLD:
                enhanced_low_delivered += 1  # should never happen; sanity counter

        source_doc = item.get("source_doc", "")
        baseline_retrieval = _retrieval_scores(question, source_doc, DEFAULT_TOP_K)
        if baseline_retrieval:
            baseline_hits.append(baseline_retrieval["hit_at_k"])
            baseline_ndcgs.append(baseline_retrieval["ndcg_at_k"])

        resolved_retrieval = None
        if resolved["type"] == "answer":
            resolved_query = resolved.get("merged_question") or question
            resolved_rounds = resolved.get("clarification_rounds") or 0
            resolved_top_k = DEFAULT_TOP_K if resolved_rounds == 0 else DEFAULT_TOP_K + 2
            resolved_retrieval = _retrieval_scores(resolved_query, source_doc, resolved_top_k)
            if resolved_retrieval:
                resolved_hits.append(resolved_retrieval["hit_at_k"])
                resolved_ndcgs.append(resolved_retrieval["ndcg_at_k"])

        rows.append({
            "question": question,
            "is_ambiguous": expected_ambiguous,
            "source_doc": source_doc,
            "expected_answer": item.get("expected_answer") or "",
            "baseline_answer": baseline["text"],
            "baseline_faithfulness": baseline["faithfulness_score"],
            "baseline_hit_at_3": None if not baseline_retrieval else baseline_retrieval["hit_at_k"],
            "baseline_ndcg_at_3": None if not baseline_retrieval else baseline_retrieval["ndcg_at_k"],
            "detect_only_type": detect_only["type"],
            "detect_only_text": detect_only["text"],
            "resolved_type": resolved["type"],
            "resolved_rounds": resolved.get("clarification_rounds", 0),
            "resolved_answer": resolved["text"],
            "resolved_draft_faithfulness": (resolved["faithfulness"] or {}).get("score"),
            "resolved_withheld": resolved["suppressed"],
            "resolved_withheld_draft": resolved["original_answer"] or "",
            "resolved_hit_at_3": None if not resolved_retrieval else resolved_retrieval["hit_at_k"],
            "resolved_ndcg_at_3": None if not resolved_retrieval else resolved_retrieval["ndcg_at_k"],
            "grade_baseline": "",
            "grade_resolved": "",
            "grader": "",
            "notes": "",
        })

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"Test set: {test_path}  ({len(test_set)} questions)")
    print(f"Faithfulness threshold for withholding an answer: {FAITHFULNESS_THRESHOLD}\n")

    print("=== Per-question results ===")
    for r in rows:
        print(f"\nQ: {r['question']}")
        print(f"  labelled ambiguous: {r['is_ambiguous']} | detect-only: {r['detect_only_type']} | resolved: {r['resolved_type']} ({r['resolved_rounds']} round(s))")
        print(f"  baseline: {r['baseline_answer'][:110]}")
        print(f"  resolved: {r['resolved_answer'][:110]}")
        if r["resolved_withheld"]:
            print(f"  (resolved draft withheld, faithfulness {fmt(r['resolved_draft_faithfulness'])}): {r['resolved_withheld_draft'][:110]}")

    print("\n=== Ambiguity detection (enhanced, detect-only) ===")
    print(f"  True positives (correctly clarified):  {tp}")
    print(f"  False positives (over-asked):          {fp}")
    print(f"  True negatives (correctly answered):   {tn}")
    print(f"  False negatives (missed ambiguity):    {fn}")
    print(f"  Precision: {fmt(precision)}")
    print(f"  Recall:    {fmt(recall)}")

    print("\n=== Clarification resolution (enhanced, resolved) ===")
    print(f"  Questions that triggered clarification: {clarified_questions}")
    print(f"  ...of which ended in a final answer:     {resolved_to_answer}")

    print("\n=== Faithfulness (raw score of the generated answer, higher = better supported) ===")
    print(f"  Baseline           mean {fmt(mean(baseline_scores))} over {baseline_answered} answered questions")
    print(f"  Enhanced (draft)   mean {fmt(mean(enhanced_scores))} over {enhanced_answered} answered questions")
    print(f"  Answers shown to the user with score below {FAITHFULNESS_THRESHOLD}:")
    print(f"    Baseline: {baseline_low_delivered} of {baseline_answered}")
    print(f"    Enhanced: {enhanced_low_delivered} of {enhanced_answered}  ({enhanced_withheld} draft(s) withheld)")
    print("  Note: the two means cover different question sets/queries where clarification happened,")
    print("  and 'enhanced shows 0' holds by construction. The real cost of withholding is correct")
    print("  answers wrongly withheld, which only manual grading can measure (see CSV).")

    print("\n=== Retrieval ranking (binary relevance vs labelled source_doc) ===")
    print("  A hit is relevant if metadata.source matches the question label.")
    print("  Out-of-scope items (no catalogue source_doc) are excluded.")
    print("  This is document-level NDCG, not clause-level; manual grades still decide correctness.")
    print(f"  Baseline  mean NDCG@{DEFAULT_TOP_K} {fmt(mean(baseline_ndcgs))} | Hit@{DEFAULT_TOP_K} {fmt(mean(baseline_hits))} over {len(baseline_ndcgs)} questions")
    print(f"  Resolved  mean NDCG@{DEFAULT_TOP_K} {fmt(mean(resolved_ndcgs))} | Hit@{DEFAULT_TOP_K} {fmt(mean(resolved_hits))} over {len(resolved_ndcgs)} questions")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-question results written to {csv_path}")
    print("Grade columns are blank: suggested scale is correct / partially correct / incorrect / correct refusal.")


if __name__ == "__main__":
    evaluate()
