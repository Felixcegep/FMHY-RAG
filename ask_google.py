import sys
import json
import faiss
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import os

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Please set the GOOGLE_API_KEY environment variable")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# File paths
IDX_FILE = "index.faiss"
PASSAGES_FILE = "passages.json"

# Models
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-1.5-flash"  # Free tier model


def get_embedding(text):
    """Get embedding for search query"""
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        return np.array(result['embedding'], dtype='float32')
    except google_exceptions.ResourceExhausted:
        print("❌ Rate limit hit. Please wait and try again.")
        return None
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None


def generate_answer(question, context):
    """Generate answer using Gemini with optimized prompt for FMHY queries"""
    prompt = f"""You are a helpful assistant specialized in Free Media Heck Yeah (FMHY) resources for streaming and downloading content.

**IMPORTANT LANGUAGE RULES:**
- Sites from "Streaming_Sites.md", "Download_Sites.md", and files without language prefixes are ENGLISH sites by default
- Sites from "French__Franais.md" are French language sites
- Sites from "Russian__.md" are Russian language sites  
- Sites from "Indonesian__Bahasa_Indonesia.md" are Indonesian language sites
- When user asks for English content, prioritize sites from English source files
- When user asks for other languages, use the appropriate language-specific files

**Answer Guidelines:**
- Use ONLY information from the provided CONTEXT
- Focus on actual streaming/download sites over forums or general info
- Include source file names: [Source: filename]
- Include any quality ratings (⭐), warnings, or special notes from context
- Format links as [Site Name](URL) when URLs are available
- Include any registration requirements or special features mentioned
- If asking for English content, treat unlabeled sites as English-compatible

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 1500
            }
        )
        return response.text
    except google_exceptions.ResourceExhausted:
        return "❌ Rate limit hit while generating answer. Please try again in a moment."
    except Exception as e:
        return f"❌ Generation error: {e}"


def search(question, top_k=8):
    """Search for relevant passages and generate answer"""
    if not question.strip():
        return {"answer": "Please provide a question.", "sources": []}

    # Load index and passages
    try:
        index = faiss.read_index(IDX_FILE)
        with open(PASSAGES_FILE, 'r', encoding='utf-8') as f:
            passages = json.load(f)
        print(f"✅ Loaded {len(passages)} passages")
    except FileNotFoundError:
        print("❌ Index files not found. Please run the embedding script first.")
        return {"answer": "Index files not found. Please run the embedding script first.", "sources": []}
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return {"answer": f"Error loading files: {e}", "sources": []}

    # Get query embedding
    print(f"🔍 Searching: {question}")
    query_embedding = get_embedding(question)
    if query_embedding is None:
        return {"answer": "Failed to create query embedding", "sources": []}

    # Search
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    # Get results - handle both 'src' and 'source' field names
    results = []
    sources = set()

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(passages):
            passage = passages[idx]

            # Handle both old format ('src') and new format ('source')
            source_name = passage.get('source') or passage.get('src', 'unknown')

            results.append({
                'text': passage['text'],
                'source': source_name,
                'score': float(score)
            })
            sources.add(source_name)
        elif idx == -1:
            # FAISS returns -1 when fewer results found than requested
            continue
        else:
            print(f"⚠️ Invalid index {idx} from search")

    if not results:
        print("🤷 No relevant information found.")
        return {"answer": "No relevant information found.", "sources": []}

    print(f"📚 Found {len(results)} passages from {len(sources)} sources:")
    for source in sorted(sources):
        print(f"  • {source}")

    # Build context and generate answer
    context = "\n\n".join([
        f"[Source: {r['source']}]\n{r['text']}"
        for r in results
    ])

    print("🤖 Generating answer...")
    answer = generate_answer(question, context)

    print(f"\n💡 Answer:\n{answer}")

    # Return dictionary with answer and sources
    return {
        "answer": answer,
        "sources": list(sources)
    }


def main(question):
    """Main function to be called by Flask app"""
    if not question:
        return {"answer": "Please provide a question.", "sources": []}

    try:
        result = search(question)
        return result
    except Exception as e:
        print(f"❌ Error in main: {e}")
        return {"answer": f"Error processing query: {e}", "sources": []}


if __name__ == "__main__":
    question = input("entrez votre question")
    result = main(question)
    if result:
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {result['sources']}")