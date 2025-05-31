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

def generate_answer(prompt, model='StormSplits/shira'):
    """Generate answer using local Ollama"""
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,
                'top_p': 0.9
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"❌ Error with {model}: {e}")
        # Fallback to qwen2.5-coder if the first model fails
        try:
            print("🔄 Trying fallback model...")
            response = ollama.chat(
                model='qwen2.5-coder:7b',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e2:
            return f"❌ Both models failed: {e2}"

# Load FAISS index and passages
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

# Search top-k similar passages
D, I = index.search(emb_q, k=6)

# Collect results and show sources
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

# Build context
context = "\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for r in results])

# Prepare prompt
prompt = f"""You are a helpful assistant that answers questions based on provided context.

INSTRUCTIONS:
- Answer using ONLY the information provided in the context below
- If you cannot find relevant information, say so clearly
- Cite the source file names when relevant
- If there is a star ⭐ in front of a link, prioritize it and include the website link
- Be concise but comprehensive
- Provide direct links/URLs when available in the context

CONTEXT:
{context}

QUESTION: {QUESTION}

ANSWER:"""

print("\n🤖 Generating answer...")

# Generate answer
answer = generate_answer(prompt)

print("\n💬 Answer:")
print("=" * 50)
print(answer)
print("=" * 50)