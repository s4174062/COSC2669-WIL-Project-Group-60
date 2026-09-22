from chunk import chunk_clauses


def test_chunk_clauses_groups_same_heading():
    clauses = [
        {
            "clause": "34",
            "section": "Section 3 - Policy",
            "subsection": "Assessment Flexibility",
            "heading": "Extensions",
            "text": "(34) Extensions are available for unforeseen circumstances of a short-term nature.",
        },
        {
            "clause": "35",
            "section": "Section 3 - Policy",
            "subsection": "Assessment Flexibility",
            "heading": "Extensions",
            "text": "(35) Applications are submitted at least one working day before the due date.",
        },
    ]
    chunks = chunk_clauses(clauses, title="Assessment Policy", min_chunk_chars=200)
    assert len(chunks) == 1
    assert chunks[0]["clauses"] == "34-35"
    assert chunks[0]["heading"] == "Extensions"
    assert "Source: Assessment Policy" in chunks[0]["text"]
    assert "(34)" in chunks[0]["text"]
    assert "(35)" in chunks[0]["text"]


def test_chunk_clauses_splits_when_heading_changes():
    clauses = [
        {
            "clause": "35",
            "section": "Section 3 - Policy",
            "subsection": "Assessment Flexibility",
            "heading": "Extensions",
            "text": "(35) Apply before the due date.",
        },
        {
            "clause": "41",
            "section": "Section 3 - Policy",
            "subsection": "Assessment Flexibility",
            "heading": "Special Consideration",
            "text": "(41) Special consideration is available after unexpected circumstances.",
        },
    ]
    chunks = chunk_clauses(clauses, title="Assessment Policy", min_chunk_chars=20)
    assert len(chunks) == 2
    assert chunks[0]["heading"] == "Extensions"
    assert chunks[1]["heading"] == "Special Consideration"
    assert chunks[0]["clauses"] == "35"
    assert chunks[1]["clauses"] == "41"
