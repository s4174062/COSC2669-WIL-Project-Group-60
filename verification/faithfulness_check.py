"""
Lightweight faithfulness check: does the generated answer contain claims
that aren't traceable back to the retrieved context?

Deliberately NOT a second LLM call judging the first LLM's output (that's
somewhat circular for a small local model). This uses simple sentence-level
word-overlap against the retrieved chunks instead — cheap, deterministic,
and easy to explain/justify in the report. Can be swapped for a proper
NLI/entailment model later if time allows.
"""

import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "this", "that", "these", "those", "it",
    "you", "your", "i", "we", "can", "will", "would", "should", "must",
    "not", "no", "do", "does", "did", "have", "has", "had",
}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def check_faithfulness(answer: str, context_chunks: list[str], overlap_threshold: float = 0.3) -> dict:
    """
    Returns {
        "score": float (0-1, fraction of answer sentences supported),
        "unsupported_sentences": list[str],
    }
    A sentence counts as "supported" if a meaningful fraction of its
    significant words also appear in at least one retrieved chunk.

    A correct "I don't have enough information" refusal makes no claims,
    so it is trivially faithful (score 1.0) rather than flagged as
    unsupported — it's declining to answer, not fabricating one.
    """
    normalised = answer.strip().lower().replace("\u2019", "'")
    if normalised.startswith("i don't have enough information"):
        return {"score": 1.0, "unsupported_sentences": []}

    context_words = _significant_words(" ".join(context_chunks))
    sentences = _split_sentences(answer)

    if not sentences:
        return {"score": 1.0, "unsupported_sentences": []}

    unsupported = []
    for sentence in sentences:
        sentence_words = _significant_words(sentence)
        if not sentence_words:
            continue
        overlap = len(sentence_words & context_words) / len(sentence_words)
        if overlap < overlap_threshold:
            unsupported.append(sentence)

    supported_count = len(sentences) - len(unsupported)
    score = supported_count / len(sentences)

    return {"score": round(score, 3), "unsupported_sentences": unsupported}


if __name__ == "__main__":
    context = [
        "Special Consideration applications must be submitted within two "
        "working days of the assessment date."
    ]
    good_answer = "You must submit your special consideration application within two working days of the assessment."
    bad_answer = "You have thirty days to submit your application, and late fees may apply."

    print("Good answer:", check_faithfulness(good_answer, context))
    print("Bad answer:", check_faithfulness(bad_answer, context))
