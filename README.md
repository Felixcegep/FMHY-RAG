# 🧠 FMHY-RAG Assistant

A **RAG** (Retrieval-Augmented Generation) assistant that can run:

- **Locally with Ollama** 🔒  
- **In the cloud with the Google Gemini API** ☁️  
- **Via a Flask web app** 🌐  

It uses **FAISS** for semantic indexing of documents extracted from <https://fmhy.net/>.

---

## ✅ Requirements

### 🧩 System Dependencies

| Component               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Python ≥ 3.9**        | Python interpreter                                                          |
| **Ollama**              | _Optional – for local use only_ → <https://ollama.com/download>            |
| **Google Gemini API key** | _Optional – for cloud usage_                                                |
| **jq, curl, wget**      | Usually preinstalled on Linux/macOS                                        |

### 📦 Python Packages

```bash
pip install faiss-cpu numpy tqdm flask
```

---

## 🔄 Download Ollama Models (Local Version)

```bash
ollama pull nomic-embed-text
ollama run artifish/llama3.2-uncensored
```

> 🚪 Make sure Ollama is running before executing local scripts.

---

## ⚙️ Setup & Updates

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Felixcegep/FMHY-RAG.git
   cd FMHY-RAG
   ```

2. **Download the Markdown source files**:

   ```bash
   bash setup/download_docs.sh
   ```

3. **Split the documents into sections**:

   ```bash
   bash setup/split_all_docs.sh
   ```

4. **(Optional) Update later**:

   ```bash
   bash setup/download_docs.sh     # updates the sources
   bash setup/split_all_docs.sh    # regenerates the sections
   python update_rag_local.py      # or update_rag_google.py
   ```

---

## 🛠️ Build the Index

| Mode                     | Command                        |
|--------------------------|--------------------------------|
| **Local (Ollama)**       | `python update_rag_local.py`   |
| **Cloud (Google Gemini)**| `python update_rag_google.py`  |

---

## ❓ Ask a Question (CLI)

| Mode       | Example                                                             |
|------------|---------------------------------------------------------------------|
| **Local**  | `python ask_local.py "Show me where I can watch Korean dramas."`   |
| **Cloud**  | `python ask_google.py "Show me where I can watch Korean dramas."`  |

---

## 🌐 Run the Web Interface (Flask)

You can also run FMHY-RAG through a simple web app powered by Flask.

1. Make sure the FAISS index and `passages.json` are built.

2. Run the Flask app:

   ```bash
   python app.py
   ```

3. Open your browser and go to:  
   [http://localhost:5000](http://localhost:5000)

> The web app will let you ask questions interactively using the local model.

---

## 📁 Project Structure

```
.
├── app.py                    # Flask web interface
├── ask_local.py              # Query using local embeddings
├── ask_google.py             # Query using Google Gemini embeddings
├── update_rag_local.py       # Builds the FAISS index (local)
├── update_rag_google.py      # Builds the FAISS index (Google)
├── index.faiss               # FAISS index
├── passages.json             # Indexed passages
├── sections/                 # Markdown chunks
├── docs/                     # Raw source documents
├── setup/
│   ├── download_docs.sh
│   └── split_all_docs.sh
└── README.md
```

---

## 💬 Example Usage (CLI)

```bash
$ python ask_local.py "What are the best sites to download audiobooks?"
✅ Loaded index with 2,945 passages
🔍 Searching for: What are the best sites to download audiobooks?
📚 Found 6 relevant passages from 4 sources:
  • Audiobooks_1.md
  • Audiobooks_2.md
  ...
```

---

## 🔧 Troubleshooting

| Issue                            | Solution                                         |
|----------------------------------|--------------------------------------------------|
| `ModuleNotFoundError: faiss`     | Make sure `faiss-cpu` is installed               |
| `ConnectionError` with Ollama    | Make sure Ollama is running (`ollama run`)       |
| Irrelevant or no search results  | Re-run `split_all_docs.sh` and `update_rag_*.py` |
| Flask app not loading            | Check that `index.faiss` and `passages.json` exist and are valid |

---

## 🌐 Helpful Links

- 📚 Source website: [https://fmhy.net](https://fmhy.net/)
