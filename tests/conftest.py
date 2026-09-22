import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "ingestion", ROOT / "retrieval", ROOT / "generation", ROOT / "app"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
