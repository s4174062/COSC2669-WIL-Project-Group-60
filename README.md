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

<<<<<<< HEAD

## Setup

You need Python 3.10 or newer (developed on 3.11) and [Ollama](https://ollama.com).

```bash
# 1. Create and activate an environment
conda create -n policy-rag python=3.11
conda activate policy-rag

# 2. Install Python dependencies
python -m pip install -r requirements.txt

# 3. Get the local model (Ollama must be installed and running)
ollama pull llama3

# 4. Build the vector store from data/raw (run from the project root)
python ingestion/embed.py
```

Step 4 should print one "Added N chunks" line per document, then `Total documents indexed: 4`. The first run also downloads the embedding model (about 90 MB). The vector store lives in `./chroma_db` (git-ignored), so **every person builds their own**. Re-run `embed.py` whenever `data/raw/` changes; it rebuilds the collection from scratch.

> **Run all scripts from the project root.** The vector store path is relative to your current directory, so running `embed.py` and `pipeline.py` from different folders creates two separate `chroma_db` folders and retrieval comes back empty.
=======
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
>>>>>>> 7c2ea26bd8cc73aba4f79aab1c7ee51d967f230c
