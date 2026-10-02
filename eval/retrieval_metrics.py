"""
Retrieval ranking metrics for the labelled test set.

Binary relevance: a retrieved chunk is relevant if metadata.source matches
the question's source_doc. That is document-level, not heading-level, so a
right-policy / wrong-clause hit still counts. Suitable for NDCG@k on this
small four-document KB; not a substitute for manual answer grading.

BLEU/BERTScore are not computed here: clarifications and refusals are
intended outputs and would be scored as failures against expected_answer.
"""

from __future__ import annotations

import math

from config import REQUIRED_SOURCE_IDS


def labelled_source(source_doc: str | None) -> str | None:
    """Return a catalogue source id, or None for out-of-scope / unlabelled items."""
    if not source_doc:
        return None
    key = source_doc.strip()
    return key if key in REQUIRED_SOURCE_IDS else None


def dcg_at_k(gains: list[float], k: int) -> float:
    total = 0.0
    for i, gain in enumerate(gains[:k]):
        total += gain / math.log2(i + 2)
    return total


def ndcg_at_k(relevances: list[float], k: int, relevant_in_collection: bool = True) -> float | None:
    """
    NDCG@k with binary gains. If the query has a labelled source, the ideal
    top-k is k relevant hits (each policy has more than k chunks).
    """
    if k <= 0:
        return None
    dcg = dcg_at_k(relevances, k)
    if relevant_in_collection:
        ideal = [1.0] * k
    else:
        ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(relevances: list[float], k: int) -> float:
    return 1.0 if any(rel > 0 for rel in relevances[:k]) else 0.0


def score_hits(hits: list[dict], expected_source: str, k: int = 3) -> dict:
    sources = [(hit.get("metadata") or {}).get("source") or "" for hit in hits[:k]]
    relevances = [1.0 if source == expected_source else 0.0 for source in sources]
    while len(relevances) < k:
        relevances.append(0.0)
        sources.append("")
    return {
        "sources": sources[:k],
        "hit_at_k": hit_at_k(relevances, k),
        "ndcg_at_k": ndcg_at_k(relevances, k, relevant_in_collection=True),
    }
