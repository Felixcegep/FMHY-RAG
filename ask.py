import sys
import json
import requests
import faiss
import numpy as np
import ollama  # Add this import

if len(sys.argv) < 4:
    print("Usage: python ask.py <INDEX_FILE> <PASSAGES_JSON> <QUESTION>")
    sys.exit(1)

IDX_FILE, PASSAGES_FILE, QUESTION = sys.argv[1], sys.argv[2], sys.argv[3]
OLLAMA_SERVER = "http://192.168.2.241:11434"

# Load FAISS index and passages
index = faiss.read_index(IDX_FILE)
chunks = json.load(open(PASSAGES_FILE, 'r', encoding='utf-8'))
print(f"✅ Loaded index with {len(chunks)} passages")

# Call Ollama /api/embeddings
embedding_response = requests.post(
    f"{OLLAMA_SERVER}/api/embeddings",
    json={"model": "nomic-embed-text", "prompt": QUESTION}
)
embedding_response.raise_for_status()
embedding_data = embedding_response.json()

if "embedding" not in embedding_data or not embedding_data["embedding"]:
    print("❌ Error: embedding not received or is empty.")
    sys.exit(1)

emb_q = np.array(embedding_data["embedding"], dtype='float32').reshape(1, -1)
faiss.normalize_L2(emb_q)

# Search top-k similar passages
D, I = index.search(emb_q, k=6)
context = "\n\n".join(chunks[i]['text'] for i in I[0])

print(f"🔍 Searching for: {QUESTION}")
sources = set(chunks[i].get('src', 'unknown') for i in I[0])
print(f"📚 Found 6 relevant passages from {len(sources)} sources:")
for source in sorted(sources):
    print(f"  • {source}")

# Prepare prompt for chat model
prompt = f"""You are a helpful assistant.
Answer using only the information below;
cite the file names if relevant.

CONTEXT:
{context}
also if there is a star ⭐ in fromt of the link choose it more often and add the link of the website
QUESTION: {QUESTION}
ANSWER:"""

print("\n🤖 Generating answer...")

# Use ollama library instead of direct HTTP call
try:
    response = ollama.chat(
        model='artifish/llama3.2-uncensored:latest',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    print("\n💬 Answer:")
    print("=" * 50)
    print(response['message']['content'])
    print("=" * 50)

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying alternative connection...")

    # Fallback: try localhost if the IP doesn't work
    try:
        import ollama

        response = ollama.chat(
            model='qwen2.5-coder:7b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        print("\n💬 Answer:")
        print("=" * 50)
        print(response['message']['content'])
        print("=" * 50)
    except Exception as e2:
        print(f"❌ Both attempts failed: {e2}")
        print("Please check if Ollama is running: ollama serve")