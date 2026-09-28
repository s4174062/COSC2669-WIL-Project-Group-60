"""
Shows what the retriever actually returns for a question, so you can tell
whether a wrong or "I don't have enough information" answer is a RETRIEVAL
problem (the right passage never got fetched) or a GENERATION problem (it
was fetched but the model ignored it).

    python eval/inspect_retrieval.py "your question" [top_k] ["phrase the right chunk should contain"]

Run from the project root. top_k defaults to 5 (the pipeline itself uses 3).
The optional phrase is matched case-insensitively against each retrieved
chunk, and the output says at which rank it was found, or that it wasn't.
Distances are cosine distances: smaller means closer to the question.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "retrieval"))
sys.path.append(os.path.join(HERE, "..", "ingestion"))

from retriever import _get_model
from embed import get_collection

PIPELINE_TOP_K = 3


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    expected = sys.argv[3].lower() if len(sys.argv) > 3 else None

    model = _get_model()
    collection = get_collection()
    embedding = model.encode([question]).tolist()
    result = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    print(f"Question: {question}\n")
    found_rank = None
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        hit = bool(expected) and expected in doc.lower()
        if hit and found_rank is None:
            found_rank = rank
        marker = "  <-- contains the expected phrase" if hit else ""
        snippet = " ".join(doc.split())[:200]
        print(f"#{rank}  {meta.get('source', '?')}  (distance {dist:.3f}){marker}")
        print(f"    {snippet}...\n")

    if expected:
        if found_rank is None:
            print(f"Expected phrase \"{expected}\" was NOT in the top {top_k}: this is a retrieval problem.")
        elif found_rank > PIPELINE_TOP_K:
            print(f"Expected phrase found at rank {found_rank}, outside the pipeline's top {PIPELINE_TOP_K}: "
                  "the model never sees it. Retrieval problem.")
        else:
            print(f"Expected phrase found at rank {found_rank}, inside the pipeline's top {PIPELINE_TOP_K}. "
                  "If the answer is still wrong, the problem is generation, not retrieval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
