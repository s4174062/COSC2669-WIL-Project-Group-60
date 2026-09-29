"""
Rule-based ambiguity detection.

NOT an LLM call: this checks whether a question hits a known
vague pattern that maps to multiple distinct policy pathways, and if so,
returns a clarification question instead of letting the pipeline guess.

Two phrasings per pattern, because the two pipelines behave differently:
  - "direct"   -> for pipeline_conversational.py, which merges the user's
                  answer into the running question automatically. Asks
                  for the missing detail directly, since the system
                  already remembers what was asked.
  - "rephrase" -> for pipeline.py, which has no memory between calls.
                  Asks the user to restate the WHOLE question in one go,
                  since a bare follow-up answer would otherwise be
                  treated as an unrelated new question.

Extend AMBIGUOUS_PATTERNS as the team's rubric
(docs/ambiguity_rubric.md) gets filled in with more real examples.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class AmbiguityPattern:
    trigger: str                   # word/phrase that signals possible ambiguity
    qualifiers: list[str]          # any of these present means it's NOT ambiguous
    clarification_direct: str      # used by the conversational (stateful) pipeline
    clarification_rephrase: str    # used by the stateless pipeline


AMBIGUOUS_PATTERNS = [
    AmbiguityPattern(
        trigger="assessment",
        # Either name the assessment type, or name the pathway they want
        # (in which case the assessment type no longer changes the answer).
        qualifiers=[
            "exam", "assignment", "quiz", "test", "presentation", "report",
            "special consideration", "extension", "equitable", "deferred", "deferral",
        ],
        clarification_direct=(
            "Could you tell me what type of assessment this is — an exam, "
            "an assignment, or something else? The applicable policy "
            "depends on the assessment type."
        ),
        clarification_rephrase=(
            "Could you rephrase your question to tell me what kind of "
            "assessment this is — an exam, an assignment, or something "
            "else? The applicable policy depends on the assessment type."
        ),
    ),
    AmbiguityPattern(
        trigger="more time",
        qualifiers=["extension", "special consideration", "deferred"],
        clarification_direct=(
            "Are you asking about an extension (a short delay, usually "
            "without documentation) or special consideration (for "
            "significant circumstances, usually with documentation)? "
            "They have different application processes."
        ),
        clarification_rephrase=(
            "Could you rephrase your question to specify whether you're "
            "asking about an extension (a short delay, usually without "
            "documentation) or special consideration (for significant "
            "circumstances, usually with documentation)? They have "
            "different application processes."
        ),
    ),
    AmbiguityPattern(
        trigger="extension",
        qualifiers=[
            "sick", "illness", "ill", "unwell", "unforeseen", "unexpected",
            "exam", "how long", "maximum", "outcome", "notify",
            "working day", "due date", "seven", "calendar",
            "documentation", "evidence", "equitable",
        ],
        clarification_direct=(
            "What unforeseen short-term circumstance is preventing you "
            "from completing the work? Extension eligibility depends on "
            "that, and on whether the due date has already passed."
        ),
        clarification_rephrase=(
            "Could you rephrase your question to include the unforeseen "
            "short-term circumstance and whether the due date has already "
            "passed? Extension eligibility depends on those details."
        ),
    ),
    AmbiguityPattern(
        trigger="first semester",
        qualifiers=["exceptional", "exception"],
        clarification_direct=(
            "Leave of absence is not normally available in the first "
            "semester unless you can demonstrate exceptional circumstances. "
            "Do you have exceptional circumstances?"
        ),
        clarification_rephrase=(
            "Could you rephrase your question to say whether you have "
            "exceptional circumstances? Leave of absence is not normally "
            "available in the first semester without them."
        ),
    ),
    AmbiguityPattern(
        trigger="refund",
        qualifiers=["census", "how much", "exact", "approved schedule", "credit"],
        clarification_direct=(
            "Was the withdrawal or leave approved before or after the "
            "census date? Refund eligibility is determined with reference "
            "to the Approved Schedule of Fees and Charges and those dates."
        ),
        clarification_rephrase=(
            "Could you rephrase your question to say whether this was "
            "before or after the census date? Refund eligibility depends "
            "on that timing and the Approved Schedule of Fees and Charges."
        ),
    ),
    AmbiguityPattern(
        trigger="break",
        qualifiers=[
            "leave of absence", "loa", "withdraw", "withdrawal", "drop", "defer",
        ],
        clarification_direct=(
            "Do you mean an approved leave of absence, withdrawing from "
            "a course, or cancelling your program? Those are different "
            "processes with different fee and enrolment effects."
        ),
        clarification_rephrase=(
            "Could you rephrase your question to say whether you mean "
            "leave of absence, course withdrawal, or cancelling your "
            "program? Those processes differ."
        ),
    ),
    AmbiguityPattern(
        trigger="defer",
        qualifiers=["special consideration", "deferred", "admissions", "offer"],
        clarification_direct=(
            "Do you mean a deferred assessment through special consideration, "
            "or deferring an offer before you enrol? Those are different processes."
        ),
        clarification_rephrase=(
            "Could you rephrase your question to say whether you mean a "
            "deferred assessment through special consideration, or deferring "
            "an offer before enrolment? Those processes differ."
        ),
    ),
    AmbiguityPattern(
        trigger="missed",
        qualifiers=[
            "re-enrol", "re-enrolment", "special consideration", "extension",
            "ago", "working day",
        ],
        clarification_direct=(
            "How long ago was the due date, and are you asking about an "
            "extension or special consideration? The applicable pathway "
            "changes once the extension application window has passed."
        ),
        clarification_rephrase=(
            "Could you rephrase your question to say how long ago the due "
            "date was, and whether you mean an extension or special "
            "consideration? The pathway depends on that timing."
        ),
    ),
]


def _mentions(text: str, term: str) -> bool:
    """
    True if `term` appears in `text` as a whole word (a plural is fine).
    Plain substring matching was wrong: "exam" matched "example" and "test"
    matched "latest", which silently marked ambiguous questions as resolved.
    """
    return re.search(rf"\b{re.escape(term)}(?:s|es|zes)?\b", text) is not None


def check_ambiguity(question: str, style: str = "rephrase") -> Optional[str]:
    """
    Returns a clarification question string if the input question matches
    a known ambiguous pattern without a qualifying term present.
    Returns None if the question is considered unambiguous (or matches
    no known pattern at all — absence of a pattern is NOT treated as
    ambiguous, to avoid over-asking).

    style: "direct" (conversational pipeline) or "rephrase" (stateless
    pipeline). Defaults to "rephrase" to match the original stateless
    pipeline.py's default behavior.
    """
    q_lower = question.lower()

    for pattern in AMBIGUOUS_PATTERNS:
        if re.search(rf"\b{re.escape(pattern.trigger)}\b", q_lower):
            has_qualifier = any(_mentions(q_lower, q) for q in pattern.qualifiers)
            if not has_qualifier:
                return pattern.clarification_direct if style == "direct" else pattern.clarification_rephrase

    return None


if __name__ == "__main__":
    test_questions = [
        "I have an assessment in three days and need more time",
        "I have an exam in three days, can I get an extension?",
        "What is the academic integrity policy?",
    ]
    for style in ("rephrase", "direct"):
        print(f"--- style={style} ---")
        for q in test_questions:
            result = check_ambiguity(q, style=style)
            print(f"Q: {q}")
            print(f"  -> {'CLARIFY: ' + result if result else 'proceed to answer'}\n")
