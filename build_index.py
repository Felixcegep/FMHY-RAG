import sys, json, pathlib, textwrap
import faiss, numpy as np
from tqdm import tqdm
import ollama


DOC_DIR = "./data"
IDX_OUT = "index.faiss"
TEXT_OUT = "passages.json"

def emb(txt):
    """Generate embeddings using local Ollama"""
    try:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=txt
        )
        return np.array(response["embedding"], dtype='float32')
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        raise


passages = []
vectors = []

# Get all Markdown files
md_files = list(pathlib.Path(DOC_DIR).glob('*.md'))

if not md_files:
    print(f"❌ No markdown files found in {DOC_DIR}")
    sys.exit(1)

print(f"📁 Found {len(md_files)} markdown files")

# Count total chunks for progress bar
total_chunks = 0
for md in md_files:
    contents = md.read_text(encoding='utf-8', errors='ignore')
    total_chunks += len(textwrap.wrap(contents, width=512))

print(f"📊 Processing {total_chunks} total chunks...")

# Progress bar for embedding
with tqdm(total=total_chunks, desc="🔍 Embedding Markdown Chunks", unit="chunk") as pbar:
    for md in md_files:
        contents = md.read_text(encoding='utf-8', errors='ignore')
        chunks = textwrap.wrap(contents, width=512)

        for chunk in chunks:
            if len(chunk.strip()) < 20:  # Skip very short chunks
                pbar.update(1)
                continue

            pid = len(passages)
            passages.append({
                'id': pid,
                'text': chunk.strip(),
                'src': md.name,
                'file_path': str(md)
            })

            try:
                vectors.append(emb(chunk))
            except Exception as e:
                print(f"❌ Failed to embed chunk from {md.name}: {e}")
                passages.pop()  # Remove the passage if embedding fails

            pbar.update(1)

if not vectors:
    print("❌ No embeddings were generated successfully")
    sys.exit(1)

# Create FAISS index
print("🔧 Creating FAISS index...")
mat = np.vstack(vectors)
index = faiss.IndexFlatIP(mat.shape[1])
faiss.normalize_L2(mat)
index.add(mat)

# Save files
faiss.write_index(index, IDX_OUT)
with open(TEXT_OUT, 'w', encoding='utf-8') as f:
    json.dump(passages, f, ensure_ascii=False, indent=2)

print(f'✅ Successfully indexed {len(passages)} chunks → {IDX_OUT}')
print(f'📄 Passages saved to: {TEXT_OUT}')