"""
Split cleaned policy text into chunks for embedding.

Clause-aware chunking keeps RMIT numbered clauses together and attaches
section/heading metadata. Paragraph splitting is kept for ad-hoc tests.
"""

from __future__ import annotations


def chunk_text(text: str, min_chunk_chars: int = 200) -> list[str]:
    """Split text into paragraph-based chunks, merging short ones together."""
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""
    for para in raw_paragraphs:
        buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(buffer) >= min_chunk_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)

    return chunks


def _location(clause: dict) -> tuple[str, str, str]:
    return (
        clause.get("section") or "",
        clause.get("subsection") or "",
        clause.get("heading") or "",
    )


def _format_chunk(group: list[dict], title: str) -> dict:
    section, subsection, heading = _location(group[0])
    parts = [p for p in (section, subsection, heading) if p]
    header = f"Source: {title}"
    if parts:
        header += "\nSection: " + " > ".join(parts)

    body = "\n\n".join(item["text"] for item in group)
    clauses = [item["clause"] for item in group if item.get("clause")]
    if len(clauses) == 1:
        clause_span = clauses[0]
    elif clauses:
        clause_span = f"{clauses[0]}-{clauses[-1]}"
    else:
        clause_span = ""

    return {
        "text": f"{header}\n\n{body}",
        "section": section,
        "subsection": subsection,
        "heading": heading,
        "clauses": clause_span,
    }


def chunk_clauses(
    clauses: list[dict],
    title: str,
    min_chunk_chars: int = 400,
    max_chunk_chars: int = 1400,
) -> list[dict]:
    """Group consecutive clauses from the same heading into retrieval chunks."""
    chunks: list[dict] = []
    buffer: list[dict] = []
    buffer_len = 0
    current_loc: tuple[str, str, str] | None = None

    def flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append(_format_chunk(buffer, title))
            buffer = []
            buffer_len = 0

    for clause in clauses:
        loc = _location(clause)
        text_len = len(clause.get("text") or "")
        would_overflow = buffer and buffer_len + text_len > max_chunk_chars
        if buffer and (loc != current_loc or would_overflow):
            flush()
        buffer.append(clause)
        buffer_len += text_len
        current_loc = loc
        if buffer_len >= min_chunk_chars:
            flush()

    flush()
    return chunks


if __name__ == "__main__":
    sample = (
        "Special Consideration allows students to apply for adjustments "
        "to assessment due to circumstances beyond their control.\n\n"
        "Applications must be submitted within two working days of the "
        "assessment date, via the Special Consideration application form."
    )
    for i, c in enumerate(chunk_text(sample)):
        print(f"--- chunk {i} ---\n{c}\n")
