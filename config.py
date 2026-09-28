"""
Shared baseline RAG settings. Paths are resolved from this file so commands
work when run from the repository root.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def add_project_paths() -> None:
    """Make repo, ingestion, retrieval, generation, and app importable."""
    for path in (
        REPO_ROOT,
        REPO_ROOT / "ingestion",
        REPO_ROOT / "retrieval",
        REPO_ROOT / "generation",
        REPO_ROOT / "app",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "llama3"
COLLECTION_NAME = "policy_chunks"
DB_PATH = REPO_ROOT / "chroma_db"
DEFAULT_TOP_K = 3

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SOURCES_PATH = DATA_DIR / "sources.json"
EVAL_QUESTIONS_PATH = REPO_ROOT / "eval" / "test_questions.json"
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"

REQUIRED_SOURCE_IDS = (
    "assessment_flexibility",
    "enrolment_procedure",
    "refund_of_fees",
    "leave_of_absence",
)

REQUIRED_METADATA_FIELDS = (
    "source",
    "title",
    "domain",
    "url",
    "section",
    "subsection",
    "heading",
    "clauses",
)


class EmptyCollectionError(RuntimeError):
    """Chroma has no policy chunks; ingestion has not been run or failed."""


class OllamaNotRunningError(RuntimeError):
    """The local Ollama server is not reachable."""


class OllamaModelMissingError(RuntimeError):
    """The configured generation model is not installed in Ollama."""


EMPTY_COLLECTION_MESSAGE = (
    "The policy vector store is empty. Ingestion does not appear to have been "
    "run. From the repository root, run: python ingestion/ingest.py"
)

OLLAMA_NOT_RUNNING_MESSAGE = (
    "Ollama is not running. Start the Ollama application and retry."
)

OLLAMA_MODEL_MISSING_MESSAGE = (
    f"Ollama model '{GENERATION_MODEL}' is not installed. "
    f"Run: ollama pull {GENERATION_MODEL}"
)

EMPTY_QUESTION_MESSAGE = (
    "Question is empty. Enter a policy question, or type 'quit' to exit."
)

NO_RESULTS_MESSAGE = "I don't have enough information to answer that."
