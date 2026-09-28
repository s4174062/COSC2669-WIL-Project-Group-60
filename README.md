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