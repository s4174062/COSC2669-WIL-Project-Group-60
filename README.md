# COSC2669-WIL-Project-Group-60

## Group Members:
- Surin Atik, s4181351
- Kevin La, s4172384
- Chan Yong Park, s4021263
- Riordan Cormick-Cox, s4174062

## Repo structure

```
config.py           # shared models, paths, and baseline settings
data/raw/           # official RMIT Policy Register HTML
data/processed/     # cleaned clause chunks
data/sources.json   # knowledge-base catalogue
ingestion/          # extract, chunk, embed, ingest, verify
retrieval/          # vector store + similarity search
generation/         # baseline prompt + Ollama call
eval/               # test questions + baseline evaluation runner
app/                # CLI baseline assistant
tests/              # unit tests for the baseline pipeline
```

## Knowledge base

Four official RMIT policy documents:

1. Assessment and Assessment Flexibility Policy
2. Enrolment Procedure
3. Refund of Fees Procedure (public substitute for enrolment-fees document id=147, which requires staff SSO)
4. Enrolment Procedure - Leave of Absence

This repository currently implements the **baseline** RAG pipeline only: ingest, dense retrieve, generate. It does not add missing-information detection or evidence verification.

## Setup

From the repository root:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ingestion/ingest.py
ollama pull llama3
```

`chroma_db/` is created locally by ingest and is not committed.

## Commands

```bash
python -m pytest
python ingestion/ingest.py
python ingestion/verify.py
python retrieval/retriever.py
python app/app.py
python eval/run_baseline.py
```

Shared settings live in `config.py`: embedding model `all-MiniLM-L6-v2`, generation model `llama3`, collection `policy_chunks`, and `top_k=3`.
