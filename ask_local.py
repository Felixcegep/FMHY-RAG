import sys
import json
import faiss
import numpy as np
IDX_FILE = "index.faiss"
PASSAGES_FILE = "passages.json"
# TODO ADD LOCAL
def get_embedding(text, model_name='nomic-embed-text'):
    """
    Get embedding using a local Ollama embedding model.
    Default model: 'nomic-embed-text'.
    """
    try:
        import ollama
        import numpy as np
        response = ollama.embeddings(
            model=model_name,
            prompt=text,
        )
        return np.array(response['embedding'], dtype='float32')
    except Exception as e:
        print(f"❌ Error getting embedding from Ollama: {e}")
        raise

def generate_answer(question, context, model='gemma3:4b'):
    """Generate answer using local Ollama"""
    prompt = f"""You are an assistant specialized in answering questions using only the FreeMediaHeist (FMHY) knowledge base provided as CONTEXT.

**Core Rules:**
1. **Context Only:** Use ONLY information from the provided CONTEXT. If no relevant info exists, respond: "No relevant information found."
2. **Source Citation:** Cite every piece of information with `[Source: Source Name]` from the context snippets.
3. **Include Links:** Use `[Resource Name](URL)` format when URLs are available in the context.
4. **Language:** Default to English resources unless user specifies another language.
5. **Prioritize Unofficial:** List unofficial methods/resources over official ones when both exist.

**Formatting:**
- Use bullet points for multiple items
- Include ⭐ marked resources prominently  
- Include any `Note:` or `Warning:` text from context
- For "watching" queries, focus on streaming/download sites, not forums or repositories

**Example:**
QUESTION: Tell me about free PC game downloads.
ANSWER:
* [CS.RIN.RU](https://cs.rin.ru/forum/) [Source: Gaming] (Note: Account Required)
* [SteamRIP](https://steamrip.com/) [Source: Gaming] (Note: Pre-Installs available)

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'top_p': 0.9}
        )
        return response['message']['content']
    except Exception as e2:
        return f"❌ Model failed: {e2}"

# Load index and passages
try:
    index = faiss.read_index(IDX_FILE)
    with open(PASSAGES_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✅ Loaded index with {len(chunks)} passages")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    sys.exit(1)

def main(QUESTION):
    print(f"🔍 Searching for: {QUESTION}")
    try:
        emb_q = get_embedding(QUESTION)
        emb_q = emb_q.reshape(1, -1)
        faiss.normalize_L2(emb_q)
    except Exception as e:
        print(f"❌ Failed to get query embedding: {e}")
        sys.exit(1)

    D, I = index.search(emb_q, k=6)

    results = []
    sources = set()
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            results.append({
                'text': chunk['text'],
                'source': chunk.get('src', 'unknown'),
                'score': float(score)
            })
            sources.add(chunk.get('src', 'unknown'))

    print(f"📚 Found {len(results)} relevant passages from {len(sources)} sources:")
    for source in sorted(sources):
        print(f"  • {source}")

    context = "\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for r in results])

    print("\n🤖 Generating answer...")
    answer = generate_answer(QUESTION, context)

    return {
        "question": QUESTION,
        "sources": list(sources),
        "context": context,
        "answer": answer
    }