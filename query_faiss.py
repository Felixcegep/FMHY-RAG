import sys
import json
import faiss
import numpy as np
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)  # Make sure to add your API key

# === Load FAISS index and metadata ===
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

index = faiss.read_index("index.faiss")


# === Functions ===
def embed_query(text: str) -> np.ndarray:
    """Create embedding for query text"""
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        vec = np.array(result.embeddings[0].values, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        return None


def generate_answer(question: str, context: str) -> str:
    """Generate answer using Gemini based on context"""
    try:
        prompt = f"""You are **FMHY-Bot**, a helpful and direct assistant specialized in providing **Free Media Heck Yeah (FMHY)** resources for streaming and downloading.

        **Your Core Directives:**
        1.  **Always provide the direct link** for any site you mention, formatted as `[Site Name](URL)`.
        2.  **Prioritize the user's requested language.**
            * **English:** Use sites from "Streaming_Sites.md", "Download_Sites.md", and any files *without* language prefixes. Assume these are English by default.
            * **Other Languages:** Consult the relevant language-specific files (e.g., "French__Franais.md", "Russian__.md", "Indonesian__Bahasa_Indonesia.md").
        3.  **Strictly adhere to the provided CONTEXT.** Do not introduce external information.
        4.  **Focus on actionable resources:**
            * List actual streaming and download sites.
            * Avoid general forums, news, or informational pages unless directly asked for.
        5.  **Include all relevant details from the CONTEXT:**
            * Any **quality ratings (⭐)**.
            * **Warnings or special notes** (e.g., "Hard Subs," "Dub," "Auto-Next," community links like Discord/Telegram, GitHub, specific resolutions, proxy/enhancement info).
            * Mention **registration requirements** if specified.
        6.  **Cite your sources:** Append `[Source: filename]` to each relevant entry.

        **Context:**
        {context}

        **Question:** {question}

        **Answer:**"""
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return "Failed to generate answer due to an error."


def search(question: str, top_k=3):
    """Search for relevant passages and generate answer"""
    if not question.strip():
        return {"answer": "Please provide a question.", "sources": []}

    print(f"✅ Loaded {len(metadata)} passages")

    # Get query embedding
    print(f"🔍 Searching: {question}")
    query_embedding = embed_query(question)
    if query_embedding is None:
        return {"answer": "Failed to create query embedding", "sources": []}

    # Search
    scores, indices = index.search(query_embedding, top_k)

    # Get results
    results = []
    sources = set()

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata):
            passage = metadata[idx]

            # Create source identifier (you can customize this format)
            source_name = f"{passage['title']} > {passage['section']}"

            results.append({
                'text': passage['text'],
                'source': source_name,
                'title': passage['title'],
                'section': passage['section'],
                'hash': passage['hashtext'],
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
        "sources": list(sources),
        "results": results  # Include raw results for additional processing if needed
    }


# === Entry Point ===
if __name__ == '__main__':
    question = "i want to watch anime in english?"
    result = search(question)
    print("longueur source", len(result))
    print("-----------------------")
    print(result)
    #print(f"\n🎯 Final Answer:\n{result['answer']}")
    #print(f"\n📚 Sources Used:\n" + "\n".join([f"  • {source}" for source in result['sources']]))