import sys
import json
import faiss
import numpy as np
import ollama

if len(sys.argv) < 4:
    print("Usage: python local_ask.py <INDEX_FILE> <PASSAGES_JSON> <QUESTION>")
    sys.exit(1)

IDX_FILE, PASSAGES_FILE, QUESTION = sys.argv[1], sys.argv[2], sys.argv[3]

def get_embedding(text):
    """Get embedding using local Ollama"""
    try:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return np.array(response["embedding"], dtype='float32')
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        raise

def generate_answer(question, context, model='artifish/llama3.2-uncensored:latest'):
    """Generate answer using local Ollama"""
    prompt = f"""You are a knowledgeable assistant that answers questions using the provided CONTEXT.

INSTRUCTIONS:
- Use ONLY the information found in the CONTEXT below.
- If the answer is not found in the context, say clearly: "No relevant information found."
- Use bullet points or short paragraphs for readability.
- Cite file or section names when possible.
- When there is a ⭐  prioritize and include the link in the answer.
- give 5 website if possible
- alway give the unofficial way prioritize it over official way.
- avoid **Official Websites:**
-be creative
exemple:
    -user (i want a good streaming website)
    -ai (give a good streaming website unofficial)
    -user (i want a good website to download video game )
    -ai (give a good website to download video game unofficial way) 

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'top_p': 0.9}
        )
        return response['message']['content']

    except Exception as e2:
        return f"❌ Both models failed: {e2}"

# Load index and chunks
try:
    index = faiss.read_index(IDX_FILE)
    with open(PASSAGES_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✅ Loaded index with {len(chunks)} passages")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    sys.exit(1)

# Get query embedding
print(f"🔍 Searching for: {QUESTION}")
try:
    emb_q = get_embedding(QUESTION)
    emb_q = emb_q.reshape(1, -1)
    faiss.normalize_L2(emb_q)
except Exception as e:
    print(f"❌ Failed to get query embedding: {e}")
    sys.exit(1)

# Search for top-k results
D, I = index.search(emb_q, k=6)

# Build context
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

# Generate answer
print("\n🤖 Generating answer...")
answer = generate_answer(QUESTION, context)

print("\n💬 Answer:")
print("=" * 50)
print(answer)
print("=" * 50)
