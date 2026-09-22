"""
Check processed chunks and the Chroma collection after baseline ingestion.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    PROCESSED_DIR,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_SOURCE_IDS,
    SOURCES_PATH,
)
from embed import get_collection


def load_sources() -> list[dict]:
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def load_processed_records() -> list[dict]:
    records = []
    for source_id in REQUIRED_SOURCE_IDS:
        path = PROCESSED_DIR / f"{source_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("chunks") or [])
    return records


def validate_records(records: list[dict], sources: list[dict]) -> None:
    if not records:
        raise ValueError("No processed chunks found.")

    source_ids = [source["id"] for source in sources]
    if tuple(source_ids) != REQUIRED_SOURCE_IDS:
        raise ValueError(
            f"Expected sources {REQUIRED_SOURCE_IDS}, found {tuple(source_ids)}."
        )

    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    for record in records:
        record_id = record.get("id")
        text = (record.get("text") or "").strip()
        metadata = record.get("metadata") or {}
        if not record_id:
            raise ValueError("A processed chunk is missing an id.")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate chunk id: {record_id}")
        if not text:
            raise ValueError(f"Chunk {record_id} has empty text.")
        missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
        if missing:
            raise ValueError(f"Chunk {record_id} is missing metadata fields: {missing}")
        seen_ids.add(record_id)
        seen_sources.add(metadata["source"])

    missing_docs = set(REQUIRED_SOURCE_IDS) - seen_sources
    if missing_docs:
        raise ValueError(f"Processed chunks are missing documents: {sorted(missing_docs)}")


def validate_collection(records: list[dict]) -> None:
    collection = get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    stored_ids = stored.get("ids") or []
    stored_docs = stored.get("documents") or []
    stored_meta = stored.get("metadatas") or []

    if len(stored_ids) != len(records):
        raise ValueError(
            f"Chroma has {len(stored_ids)} chunks but processed JSON has {len(records)}."
        )
    if len(stored_ids) != len(set(stored_ids)):
        raise ValueError("Chroma collection contains duplicate ids.")

    expected = {record["id"]: record for record in records}
    for chunk_id, document, metadata in zip(stored_ids, stored_docs, stored_meta):
        if chunk_id not in expected:
            raise ValueError(f"Chroma contains unexpected id: {chunk_id}")
        if (document or "").strip() != (expected[chunk_id]["text"] or "").strip():
            raise ValueError(f"Chroma text does not match processed JSON for {chunk_id}.")
        for field in REQUIRED_METADATA_FIELDS:
            if field not in (metadata or {}):
                raise ValueError(f"Chroma chunk {chunk_id} is missing metadata field {field}.")
            if str((metadata or {}).get(field)) != str(expected[chunk_id]["metadata"].get(field)):
                raise ValueError(
                    f"Chroma metadata {field} does not match processed JSON for {chunk_id}."
                )


def verify_ingest(records: list[dict] | None = None) -> dict:
    sources = load_sources()
    processed = records if records is not None else load_processed_records()
    validate_records(processed, sources)
    validate_collection(processed)
    summary = {
        "documents": list(REQUIRED_SOURCE_IDS),
        "chunk_count": len(processed),
        "unique_ids": len({record["id"] for record in processed}),
    }
    print(
        f"Ingest verification passed: {summary['chunk_count']} unique chunks "
        f"across {len(summary['documents'])} documents."
    )
    return summary


if __name__ == "__main__":
    verify_ingest()
