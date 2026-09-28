"""
Extract numbered clauses and definition rows from RMIT Policy Register HTML.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag


CLAUSE_RE = re.compile(r"^\((\d+)\)\s*")


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.replace("\xa0", " ").strip()


def _element_text(el: Tag) -> str:
    return _clean(el.get_text(" ", strip=True))


def _list_text(ol: Tag) -> str:
    items = []
    for i, li in enumerate(ol.find_all("li", recursive=False), start=1):
        item = _element_text(li)
        if item:
            items.append(f"{i}. {item}")
    return "\n".join(items)


def extract_clauses(html: str) -> list[dict]:
    """Return ordered clause/definition records from a policy HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#sliph-document-content")
    if content is None:
        raise ValueError("Could not find #sliph-document-content in HTML.")

    section = ""
    subsection = ""
    heading = ""
    records: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current and current["text"].strip():
            records.append(current)
        current = None

    for child in content.children:
        if isinstance(child, NavigableString) or not isinstance(child, Tag):
            continue

        classes = child.get("class") or []
        if child.name == "div" and "sliph-document-status" in classes:
            continue
        if child.name == "span" and "top-link" in classes:
            continue
        if child.name == "div" and child.get("id") == "document-top":
            continue

        if child.name in {"h1", "h2"}:
            flush()
            section = _element_text(child)
            subsection = ""
            heading = ""
            continue

        if child.name == "h3":
            flush()
            subsection = _element_text(child)
            heading = ""
            continue

        if child.name == "h4":
            flush()
            heading = _element_text(child)
            continue

        if child.name == "table":
            flush()
            for row in child.find_all("tr"):
                cells = [_element_text(td) for td in row.find_all(["th", "td"])]
                cells = [c for c in cells if c]
                if len(cells) < 2:
                    continue
                term, definition = cells[0], " ".join(cells[1:])
                records.append(
                    {
                        "clause": "",
                        "section": section,
                        "subsection": subsection or "Definitions",
                        "heading": heading or term,
                        "text": f"{term}: {definition}",
                    }
                )
            continue

        if child.name == "ol":
            if current is not None:
                listed = _list_text(child)
                if listed:
                    current["text"] = f"{current['text']}\n{listed}"
            continue

        if child.name == "p":
            text = _element_text(child)
            if not text:
                continue
            match = CLAUSE_RE.match(text)
            if match:
                flush()
                current = {
                    "clause": match.group(1),
                    "section": section,
                    "subsection": subsection,
                    "heading": heading,
                    "text": text,
                }
            elif current is not None:
                current["text"] = f"{current['text']} {text}"

    flush()
    return records


def extract_from_file(path: str | Path) -> list[dict]:
    html = Path(path).read_text(encoding="utf-8")
    return extract_clauses(html)
