# 🧠 FMHY-RAG Assistant

A **RAG** (Retrieval-Augmented Generation) assistant that can run:

- **Locally with Ollama** 🔒  
- **In the cloud with the Google Gemini API** ☁️ (default)  
- **Via a Flask web app** 🌐  

It uses **FAISS** for semantic indexing of documents extracted from <https://fmhy.net/>.

---

## ✅ Requirements

### 🧩 System Dependencies

| Component                 | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| **Python ≥ 3.9**          | Python interpreter                                                          |
| **Ollama**                | _Optional – for local use only_ → <https://ollama.com/download>             |
| **Google Gemini API key** | _Required for cloud usage (default)_                                        |
| **jq, curl, wget**        | Usually preinstalled on Linux/macOS                                         |

### 📦 Python Packages

```bash
pip install faiss-cpu numpy tqdm flask
```

---

## 🔐 Set Up Google Gemini API Key (Default Mode)

1. **Set API key as an environment variable**

   Open your terminal and run:

   ```bash
   export GOOGLE_API_KEY="your_actual_api_key_here"
   ```

   Then, in your Python scripts, read from this variable:

   ```python
   import os
   import sys

   API_KEY = os.getenv("GOOGLE_API_KEY")
   if not API_KEY:
       print("❌ Please set the GOOGLE_API_KEY environment variable")
       sys.exit(1)
   ```

   > _This way, you don’t hardcode the key in your script, which is safer._

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

| Mode                      | Command                        |
|---------------------------|--------------------------------|
| **Cloud (Google Gemini)** | `python update_rag_google.py`  |
| **Local (Ollama)**        | `python update_rag_local.py`   |

> **Note:** Cloud (Google Gemini) is now the default mode.

---

## ❓ Ask a Question (CLI)

| Mode       | Example                                                              |
|------------|----------------------------------------------------------------------|
| **Cloud**  | `python ask_google.py "Show me where I can watch Korean dramas."`    |
| **Local**  | `python ask_local.py "Show me where I can watch Korean dramas."`     |

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
├── ask_google.py             # Query using Google Gemini embeddings (default)
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
$ python ask_google.py "What are the best sites to download audiobooks?"
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
