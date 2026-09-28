from config import REQUIRED_METADATA_FIELDS
from verify import validate_records
import embed


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


def _record(source, index, text="Clause text"):
    metadata = dict(COMPLETE_METADATA)
    metadata["source"] = source
    return {
        "id": f"{source}_{index}",
        "text": text,
        "metadata": metadata,
    }


def test_validate_records_requires_four_documents_and_metadata():
    sources = [
        {"id": "assessment_flexibility"},
        {"id": "enrolment_procedure"},
        {"id": "refund_of_fees"},
        {"id": "leave_of_absence"},
    ]
    records = [
        _record("assessment_flexibility", 0),
        _record("enrolment_procedure", 0),
        _record("refund_of_fees", 0),
        _record("leave_of_absence", 0),
    ]
    validate_records(records, sources)


def test_validate_records_rejects_duplicate_ids():
    sources = [
        {"id": "assessment_flexibility"},
        {"id": "enrolment_procedure"},
        {"id": "refund_of_fees"},
        {"id": "leave_of_absence"},
    ]
    records = [
        _record("assessment_flexibility", 0),
        _record("assessment_flexibility", 0),
        _record("enrolment_procedure", 0),
        _record("refund_of_fees", 0),
        _record("leave_of_absence", 0),
    ]
    try:
        validate_records(records, sources)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate id to fail")


def test_replace_all_chunks_twice_does_not_duplicate_ids(tmp_path, monkeypatch):
    class FakeModel:
        def encode(self, texts):
            return [[0.05] * 8 for _ in texts]

    monkeypatch.setattr(embed, "DB_PATH", tmp_path / "chroma_db")
    monkeypatch.setattr(embed, "get_embedding_model", lambda: FakeModel())

    records = [
        _record("assessment_flexibility", 0, "First chunk"),
        _record("enrolment_procedure", 0, "Second chunk"),
    ]
    embed.replace_all_chunks(records)
    embed.replace_all_chunks(records)

    stored = embed.get_collection().get(include=["documents"])
    assert stored["ids"] == ["assessment_flexibility_0", "enrolment_procedure_0"]
    assert stored["documents"] == ["First chunk", "Second chunk"]
    assert len(stored["ids"]) == len(set(stored["ids"]))
    for field in REQUIRED_METADATA_FIELDS:
        assert field in COMPLETE_METADATA
