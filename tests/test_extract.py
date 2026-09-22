from extract import extract_clauses

SAMPLE_HTML = """
<html>
  <body>
    <div id="sliph-document-content">
      <h1>Section 3 - Policy</h1>
      <h3>Assessment Flexibility</h3>
      <h4>Extensions</h4>
      <p><span class="enumerate">(34) </span> Extensions are available for short-term circumstances.</p>
      <p><span class="enumerate">(35) </span> Applications are submitted at least one working day before the due date.</p>
      <ol>
        <li>Submit to the school</li>
      </ol>
    </div>
  </body>
</html>
"""


def test_extract_numbered_clauses_and_lists():
    clauses = extract_clauses(SAMPLE_HTML)
    assert [item["clause"] for item in clauses] == ["34", "35"]
    assert clauses[0]["section"] == "Section 3 - Policy"
    assert clauses[0]["subsection"] == "Assessment Flexibility"
    assert clauses[0]["heading"] == "Extensions"
    assert "short-term circumstances" in clauses[0]["text"]
    assert "1. Submit to the school" in clauses[1]["text"]


def test_extract_requires_document_content():
    try:
        extract_clauses("<html><body><p>(1) Missing container</p></body></html>")
    except ValueError as exc:
        assert "sliph-document-content" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing document container")
