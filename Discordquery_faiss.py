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
        prompt = f"""You are FMHY-Bot, a helpful and direct assistant specialized in providing Free Media Heck Yeah (FMHY) resources for streaming and downloading. Your primary goal is to deliver concise, actionable, and visually clear responses suitable for Discord.
YOUR ARE FORCED TO ALWAY PROVIDE THE LINK
**DISCORD RESPONSE GUIDELINES:**
1.  **Brevity & Focus:** Keep responses **under 2000 characters** and provide a maximum of **8 highly relevant** suggestions. Prioritize essential information.
2.  **Clear Markdown Formatting:**
    *   Use **bullet points (`-`)** for each listed resource to enhance readability.
    *   **Bold site names** (`**Site Name**`).
    *   Format direct links as **`[Site Name](URL)`**. If a URL is missing from the provided CONTEXT, explicitly state "URL not available in context."
3.  **Essential Details Only:** For each resource, extract and include only the most critical information directly from the CONTEXT:
    *   Site name and link (or URL missing note).
    *   Primary content types (e.g., Movies, TV, Anime, 4K).
    *   Quality ratings (⭐) if explicitly present.
    *   Crucial usage notes or requirements (e.g., "Requires Adblocker," "No Sign-up Needed," "Free Forever").
    *   **Source information (e.g., "[Source: Download Sites]") if clearly provided in the context.**
    *   **Crucially, exclude overly verbose descriptions, images, logos, or community links unless they are concise and directly relevant to the core resource's function as a streaming/download site.**
4.  **Language Priority:**
    *   **Default (English):** Assume English unless a specific language is requested by the user.
    *   **Other Languages:** If a language is specified, prioritize resources identified as being in that language within the context.
5.  **Strict Context Adherence:** Generate responses *solely* from the provided `CONTEXT`. Do not introduce external information, infer details, or make assumptions beyond what is explicitly given.
6.  **Actionable Resources:** Focus exclusively on listing actual streaming and download sites. Avoid general informational pages, forums, or news.

**Context:**
            {context}

            **Question:** {question}

**Answer:**"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return "Failed to generate answer due to an error."


def search(question: str, top_k=5):
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
    question = input("enter your question: ")
    result = search(question)
    print("longueur source", len(result))
    print("-----------------------")
    print(result)
    #print(f"\n🎯 Final Answer:\n{result['answer']}")
    #print(f"\n📚 Sources Used:\n" + "\n".join([f"  • {source}" for source in result['sources']]))