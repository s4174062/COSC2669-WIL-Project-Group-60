# COSC2669-WIL-Project-Group-60

## Group Members:
- Surin Atik, s4181351
- Kevin La, s4172384
- Chan Yong Park, s4021263
- Riordan Cormick-Cox, s4174062

## Repo structure

```
config.py           # shared models, paths, and settings
data/raw/           # official RMIT Policy Register HTML
data/processed/     # cleaned clause chunks
data/sources.json   # knowledge-base catalogue
ingestion/          # extract, chunk, embed, ingest, verify
retrieval/          # vector store + similarity search
generation/         # prompt + Ollama call
verification/       # missing-info detection + faithfulness check
eval/               # labelled questions + evaluation runner
app/                # baseline CLI and enhanced pipelines
tests/              # unit tests
```

## Knowledge base

Four official RMIT policy documents:

1. Assessment and Assessment Flexibility Policy
2. Enrolment Procedure
3. Refund of Fees Procedure
4. Enrolment Procedure - Leave of Absence

The **baseline** path is retrieve then generate. The **enhanced** path adds missing-information detection and a faithfulness check that can withhold an unsupported draft.

## Setup

You need Python 3.10 or newer and [Ollama](https://ollama.com). Run all commands from the repository root so `chroma_db/` is created in one place (`config.py` resolves that path from the repo, not the current folder).

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ingestion/ingest.py
ollama pull llama3
```

`chroma_db/` is local and git-ignored. Rebuild it with `python ingestion/ingest.py` after changing the four HTML sources.

## Commands

```bash
python -m pytest
python ingestion/ingest.py
python ingestion/verify.py
python retrieval/retriever.py
python app/app.py
python app/pipeline.py
python eval/validate_test_set.py
python eval/evaluate.py
```

Shared settings live in `config.py`: embedding model `all-MiniLM-L6-v2`, generation model `llama3`, collection `policy_chunks`, and `top_k=3`.
