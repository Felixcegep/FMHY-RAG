# 🧠 FMHY RAG Assistant

A local **RAG** (Retrieval-Augmented Generation) assistant built with [Ollama](https://ollama.com/) and **FAISS**, designed to answer questions using Markdown content from [https://fmhy.net/](https://fmhy.net/).

---

## ✅ Requirements

### 🧩 System Dependencies

- **Python 3.9+**
- **[Ollama](https://ollama.com/download)** installed and running
- **jq**, **curl**, **wget** (usually preinstalled on Linux/macOS)

### 📦 Python Packages

Install the required Python dependencies:

```bash
pip install faiss-cpu numpy tqdm flask
```

---

## 📅 Download Ollama Models

Before running the app, download the required models:

```bash
ollama pull nomic-embed-text
ollama run artifish/llama3.2-uncensored
```

> 🚪 Keep **Ollama** running while using the application.

---

## ⚙️ Setup & Run

1. **Clone the repository:**

```bash
git clone https://github.com/Felixcegep/FMHY-RAG.git
cd FMHY-RAG
```

2. **Start the web app immediately:**

If you already have `index.faiss` and `passages.json`, you can chat right away:

```bash
python app.py
```

Then go to 👉 [http://localhost:5000](http://localhost:5000) to use the chatbot.

3. **(Optional) Update the knowledge base:**

If you want to update the source content:

- *Linux/macOS:*
```bash
bash pullupdate/download_docs.sh
bash pullupdate/split_all_docs.sh
python update_rag.py
```

- *Windows (PowerShell):*
```powershell
.\pullupdate\download_docs.ps1
.\pullupdate\split_all_docs.ps1
python update_rag.py
```

---

## 📁 Project Structure

```
.
├── app.py                   # Flask app to run the chatbot
├── ask.py                   # CLI script to query manually
├── update_rag.py            # Script to build/update the FAISS index
├── index.faiss              # FAISS index file
├── passages.json            # All embedded chunks stored here
├── sections/                # Markdown files split into sections
├── templates/
│   └── index.html           # Web interface template
├── pullupdate/
│   ├── download_docs.sh     # Bash script to fetch Markdown content
│   ├── download_docs.ps1    # PowerShell equivalent
│   ├── split_all_docs.sh    # Bash script to chunk documents
│   └── split_all_docs.ps1   # PowerShell equivalent
└── docs/                    # Original Markdown documents
```

---

## 💬 Example Usage

```bash
python app.py
```

→ Then ask your questions in the browser UI, like:  
**"Where can I watch Korean dramas?"**

---

## 🔗 Credits

Content based on [https://fmhy.net/](https://fmhy.net/)
