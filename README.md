# FMHY RAG

GitHub: https://github.com/fmhy/FMHY

## What is FMHY?

FMHY: A curated guide to free, open, and private digital content.

This project turns FMHY into a searchable knowledge base powered by Google Gemini and FAISS.


## Requirements

- Python 3.8 or higher  
- Google Gemini API key  
- `pip` (Python package installer)

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <project-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="your_gemini_api_key"
```

How to get your key:

- Go to [Google AI Studio](https://aistudio.google.com/)
- Sign in with your Google account
- Create an API key
- Paste it into `.env`

---

## Usage

### Run the chatbot (CLI)

```bash
python query_faiss.py
```

Starts a conversational interface using the existing vector index (`index.faiss`) and metadata.

### Run the web interface (Flask)

```bash
python app.py
```

Launches a local web interface to interact with the RAG system in your browser.

---

## Update the vector database (optional)

If you've added or changed Markdown files:

```bash
python mkd2vec.py
```

This will re-embed the documents and regenerate the FAISS index.

---

## requirements.txt

```
tqdm
faiss-cpu
numpy
requests
google-generativeai
python-dotenv
Flask
```

---

## Models Used

- `text-embedding-004` — for vector embeddings  
- `gemini-1.5-flash` — low-cost, fast generation  

These models are accessible for free using a default Gemini developer API key.

---

## Credits

- Based on FMHY: https://github.com/fmhy/FMHY  
- Vector RAG implementation by you.

---

## Next Update

The next version will include efficient reindexing, allowing the system to:

- Detect which files have changed  
- Avoid re-embedding unchanged content  
- Rebuild only the necessary parts of the FAISS index  

This will dramatically reduce update time and improve scalability.
