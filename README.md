# COSC2669-WIL-Project-Group-60

## Group Members:
- Surin Atik, s4181351
- Kevin La, s4172384
- Chan Yong Park, s4021263
- Riordan Cormick-Cox, s4174062



## Repo structure

```
data/raw/          # official RMIT Policy Register HTML
data/processed/    # cleaned clause chunks
data/sources.json  # knowledge-base catalogue
ingestion/         # extract, chunk, embed, ingest
retrieval/         # vector store + similarity search
generation/        # prompt template + LLM call (Ollama)
eval/              # test question set + evaluation harness
app/               # CLI baseline assistant
```

## Knowledge base

Four official RMIT policy documents:

1. Assessment and Assessment Flexibility Policy
2. Enrolment Procedure
3. Refund of Fees Procedure (public substitute for enrolment-fees document id=147, which requires staff SSO)
4. Enrolment Procedure - Leave of Absence

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ingestion/ingest.py
ollama pull llama3
python app/app.py
```