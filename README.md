````markdown
# FMHY RAG

GitHub: https://github.com/fmhy/FMHY

## What is FMHY?

FMHY (Free Media Heck Yeah) is a community-curated resource hub that shares guides, links, and tools for accessing digital content—like movies, TV shows, books, software, and educational material. It emphasizes free, open-source, and privacy-respecting solutions.

In short: **FMHY is a digital survival kit—a well-organized guide to free content, tools, and tips for navigating the internet freely and privately.**

This project turns FMHY into a searchable knowledge base powered by Google Gemini and FAISS.

---

## Features

- Vector-based document search with FAISS  
- Conversational AI interface using Google Gemini  
- Preloaded FMHY knowledge base  
- Simple CLI for querying or updating the database  
- API key managed via environment variables

---

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
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file in the root directory:

```ini
GEMINI_API_KEY="your_gemini_api_key"
```

How to get your key:

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create an API key
4. Paste it into `.env`

---

## Usage

### Run the chatbot

```bash
python query_faiss.py
```

Starts a conversational interface using the existing vector index (`index.faiss`) and metadata.

### Update the vector database (optional)

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

* `text-embedding-004` — for vector embeddings
* `gemini-1.5-flash` — low-cost, fast generation

These models are accessible for free using a default Gemini developer API key.

---

## Credits

Based on FMHY: [https://github.com/fmhy/FMHY](https://github.com/fmhy/FMHY)
Vector RAG implementation by you.

---

## Next Update

The next version will include **efficient reindexing**, allowing the system to:

* Detect which files have changed
* Avoid re-embedding unchanged content
* Rebuild only the necessary parts of the FAISS index

This will dramatically reduce update time and improve scalability.

```

