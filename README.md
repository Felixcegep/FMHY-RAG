# 🧠 FMHY-RAG Assistant

A **RAG** (Retrieval-Augmented Generation) assistant that can run:

- **Locally with Ollama** 🔒  
- **In the cloud with the Google Gemini API** ☁️  

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
pip install faiss-cpu numpy tqdm
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

## ❓ Ask a Question

| Mode       | Example                                                             |
|------------|---------------------------------------------------------------------|
| **Local**  | `python ask_local.py "Show me where I can watch Korean dramas."`   |
| **Cloud**  | `python ask_google.py "Show me where I can watch Korean dramas."`  |

---

## 📁 Project Structure

```
.
├── app.py
├── ask_local.py               # Query using local embeddings
├── ask_google.py              # Query using Google Gemini embeddings
├── update_rag_local.py        # Builds the FAISS index (local)
├── update_rag_google.py       # Builds the FAISS index (Google)
├── index.faiss                # FAISS index
├── passages.json              # Indexed passages
├── sections/                  # Markdown chunks
├── docs/                      # Raw source documents
├── setup/
│   ├── download_docs.sh
│   └── split_all_docs.sh
└── README.md
```

---

## 💬 Example Usage

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

---

## 🌐 Helpful Links

- 📚 Source website: [https://fmhy.net](https://fmhy.net/)
