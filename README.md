# COSC2669-WIL-Project-Group-60

## Group Members:
- Surin Atik, s4181351
- Kevin La, s4172384
- Chan Yong Park, s4021263
- Riordan Cormick-Cox, s4174062



## Repo structure

```
data/raw/          #source policy documents (PDF/HTML/text)
data/processed/     #cleaned chunked text
ingestion/          #convert chunk embed scripts
retrieval/           #vector store + similarity search
generation/          #prompt template + LLM call (Ollama)
eval/                #test question set + evaluation harness
app/                 #simple interface (CLI or Streamlit)
notebooks/           #exploratory / prototyping notebooks
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3
```