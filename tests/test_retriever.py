import pytest

from config import REQUIRED_METADATA_FIELDS, EmptyCollectionError
import retriever


COMPLETE_METADATA = {
    "source": "assessment_flexibility",
    "title": "Assessment and Assessment Flexibility Policy",
    "domain": "assessment",
    "url": "https://policies.rmit.edu.au/document/view.php?id=7",
    "section": "Section 3 - Policy",
    "subsection": "Assessment Flexibility",
    "heading": "Extensions",
    "clauses": "34-36",
}


class _FakeModel:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _EmptyCollection:
    def count(self):
        return 0


class _PopulatedCollection:
    def __init__(self, count=5):
        self._count = count
        self.last_n_results = None

    def count(self):
        return self._count

    def query(self, query_embeddings, n_results, include):
        self.last_n_results = n_results
        return {
            "documents": [["clause text"]],
            "metadatas": [[COMPLETE_METADATA]],
            "distances": [[0.42]],
        }


def test_retrieve_hits_empty_collection(monkeypatch):
    monkeypatch.setattr(retriever, "get_collection", lambda: _EmptyCollection())
    with pytest.raises(EmptyCollectionError):
        retriever.retrieve_hits("How long is an extension?")


def test_retrieve_hits_structured_result_and_top_k_cap(monkeypatch):
    collection = _PopulatedCollection(count=2)
    monkeypatch.setattr(retriever, "get_collection", lambda: collection)
    monkeypatch.setattr(retriever, "get_embedding_model", lambda: _FakeModel())

    hits = retriever.retrieve_hits("How long is an extension?", top_k=3)

    assert collection.last_n_results == 2
    assert len(hits) == 1
    assert set(hits[0]) == {"text", "metadata", "distance"}
    assert hits[0]["text"] == "clause text"
    assert hits[0]["distance"] == 0.42
    for field in REQUIRED_METADATA_FIELDS:
        assert field in hits[0]["metadata"]
    assert hits[0]["metadata"]["title"] == COMPLETE_METADATA["title"]
    assert hits[0]["metadata"]["clauses"] == "34-36"


def test_retrieve_uses_retrieve_hits_text_only(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "retrieve_hits",
        lambda query, top_k=3: [
            {"text": "passage A", "metadata": COMPLETE_METADATA, "distance": 0.1},
            {"text": "passage B", "metadata": COMPLETE_METADATA, "distance": 0.2},
        ],
    )
    assert retriever.retrieve("question", top_k=2) == ["passage A", "passage B"]
