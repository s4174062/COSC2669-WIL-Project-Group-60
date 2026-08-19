"""
Splits cleaned text into chunks for embedding

Split on paragraph breaks for placeholder / testing, should be 
replaced based on document structure further along.
"""


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


if __name__ == "__main__":
    #test
    sample = (
        "Special Consideration allows students to apply for adjustments "
        "to assessment due to circumstances beyond their control.\n\n"
        "Applications must be submitted within two working days of the "
        "assessment date, via the Special Consideration application form."
    )
    for i, c in enumerate(chunk_text(sample)):
        print(f"--- chunk {i} ---\n{c}\n")
