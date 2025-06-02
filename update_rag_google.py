import sys
import json
import pathlib
import textwrap
import time
import numpy as np
import faiss
from tqdm import tqdm
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import os

# --- Configuration ---

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Please set the GOOGLE_API_KEY environment variable")
    sys.exit(1)


genai.configure(api_key=API_KEY)

# Settings optimized for free tier
DOC_DIR = "./sections"
IDX_OUT = "index.faiss"
TEXT_OUT = "passages.json"
EMBEDDING_MODEL = "models/text-embedding-004"  # Free tier model
BATCH_SIZE = 10  # Smaller batches for free tier
CHUNK_SIZE = 400  # Optimal chunk size
MIN_CHUNK_LENGTH = 50
RETRY_DELAY = 2


def embed_batch(texts):
    """Embed a batch of texts with retry logic"""
    for attempt in range(3):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts
            )
            return [np.array(emb, dtype='float32') for emb in result['embedding']]
        except google_exceptions.ResourceExhausted:
            print(f"Rate limit hit. Waiting {RETRY_DELAY}s... (attempt {attempt + 1}/3)")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Error in batch embedding: {e}")
            return [None] * len(texts)

    print("Failed to embed batch after 3 attempts")
    return [None] * len(texts)


def main():
    # Find markdown files
    md_files = list(pathlib.Path(DOC_DIR).glob('*.md'))
    if not md_files:
        print(f"❌ No .md files found in {DOC_DIR}")
        sys.exit(1)

    print(f"📁 Found {len(md_files)} markdown files")

    # Process files into chunks
    all_chunks = []
    for md_file in tqdm(md_files, desc="📚 Reading files"):
        try:
            content = md_file.read_text(encoding='utf-8', errors='ignore')
            chunks = textwrap.wrap(content, CHUNK_SIZE, break_long_words=False, replace_whitespace=False)

            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) >= MIN_CHUNK_LENGTH:
                    all_chunks.append({
                        'text': chunk,
                        'source': md_file.name,  # Changed from 'src' to 'source'
                        'file_path': str(md_file)
                    })
        except Exception as e:
            print(f"⚠️ Error reading {md_file.name}: {e}")
            continue

    if not all_chunks:
        print("❌ No chunks created")
        sys.exit(1)

    print(f"📝 Created {len(all_chunks)} text chunks")

    # Embed chunks in batches
    embeddings = []
    valid_chunks = []

    print(f"🔍 Creating embeddings in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="✨ Embedding chunks"):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [chunk['text'] for chunk in batch]

        batch_embeddings = embed_batch(texts)

        for chunk, embedding in zip(batch, batch_embeddings):
            if embedding is not None and embedding.size > 0:
                embeddings.append(embedding)
                valid_chunks.append({
                    'id': len(valid_chunks),
                    'text': chunk['text'],
                    'source': chunk['source'],  # Consistent naming
                    'file_path': chunk['file_path']
                })
            else:
                print(f"⚠️ Failed to embed chunk from {chunk['source']}")

        # Small delay between batches for free tier
        if i + BATCH_SIZE < len(all_chunks):
            time.sleep(0.5)

    if not embeddings:
        print("❌ No embeddings were created successfully")
        sys.exit(1)

    print(f"✅ Successfully embedded {len(embeddings)} chunks")

    # Create FAISS index
    print("🔧 Building FAISS index...")
    try:
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Stack embeddings and normalize for cosine similarity
        embedding_matrix = np.vstack(embeddings)
        faiss.normalize_L2(embedding_matrix)
        index.add(embedding_matrix)

        print(f"📊 Index dimension: {dimension}, Total vectors: {index.ntotal}")

    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")
        sys.exit(1)

    # Save files
    try:
        faiss.write_index(index, IDX_OUT)
        print(f"💾 FAISS index saved: {IDX_OUT}")

        with open(TEXT_OUT, 'w', encoding='utf-8') as f:
            json.dump(valid_chunks, f, indent=2, ensure_ascii=False)
        print(f"💾 Passages saved: {TEXT_OUT}")

    except Exception as e:
        print(f"❌ Error saving files: {e}")
        sys.exit(1)

    print(f"\n🎉 Processing complete!")
    print(f"📈 Indexed {len(valid_chunks)} chunks from {len(set(chunk['source'] for chunk in valid_chunks))} files")


if __name__ == "__main__":
    main()