# FMHY-RAG Assistant

A simple Retrieval-Augmented Generation (RAG) assistant for searching FMHY documents using local or cloud models.

---

## Requirements

**System:**
- Python 3.9+
- Ollama (optional, local only): https://ollama.com/download
- Google Gemini API key (required for cloud mode)
- jq, curl, wget (should be preinstalled on Linux/macOS)

**Python packages:**
```bash
pip install faiss-cpu numpy tqdm flask google-generativeai google-api-core
```

---

## Google Gemini API Key (cloud mode, default)

Set your API key as an environment variable:
```bash
export GOOGLE_API_KEY="your_actual_api_key_here"
```

---

## Ollama Models (local)

```bash
ollama pull nomic-embed-text
ollama run artifish/llama3.2-uncensored
```
Make sure Ollama is running before using local scripts.

---

## Setup & Updates

```bash
git clone https://github.com/Felixcegep/FMHY-RAG.git
cd FMHY-RAG
bash update_script/download_docs.sh     # get docs
bash update_script/split_all_docs.sh    # split docs into sections
```

To update:
```bash
bash update_script/download_docs.sh
bash update_script/split_all_docs.sh
python update_rag_local.py      # or update_rag_google.py
```

---

## Build the Index

| Mode          | Command                       |
|---------------|------------------------------|
| Cloud (Gemini)| python update_rag_google.py  |
| Local (Ollama)| python update_rag_local.py   |

---

## Ask a Question (CLI)

| Mode   | Example                                                    |
|--------|------------------------------------------------------------|
| Cloud  | python ask_google.py "Show me where I can watch K-dramas." |
| Local  | python ask_local.py "Show me where I can watch K-dramas."  |

---

## Web App (Flask)

Build the FAISS index and passages.json first, then run:
```bash
python app.py
```
Go to [http://localhost:5000](http://localhost:5000)

---

## Project Structure

```
.
├── app.py
├── ask_local.py
├── ask_google.py
├── update_rag_local.py
├── update_rag_google.py
├── index.faiss
├── passages.json
├── sections/
├── docs/
├── update_script/
│   ├── download_docs.sh
│   └── split_all_docs.sh
└── README.md
```

---

## Example (CLI)

```bash
python ask_google.py "What are the best sites to download audiobooks?"
# Loads index, searches, and shows relevant passages
```

---

## Troubleshooting

| Problem                    | Solution                             |
|----------------------------|--------------------------------------|
| faiss not found            | pip install faiss-cpu                |
| Ollama connection error    | Start Ollama (ollama run ...)        |
| Bad/no search results      | Re-run split_all_docs.sh & update    |
| Flask app not loading      | Make sure index.faiss & passages.json exist |

---

## Source

- https://fmhy.net/
