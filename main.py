"""
Google AI Studio + GPU RAG System - Document Indexing
Updated: 2025-05-30 06:02:51 UTC by Felixcegep
Uses GPU for embeddings + Google AI Studio for generation
"""

import os
import chromadb
import glob
import re
import gc
import time
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Iterator

# Configuration
SECTIONS_DIR = "./sections"
CHROMA_DB_PATH = "./chroma_db_gemini"
COLLECTION_NAME = "md_documents_gemini"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 16
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILES_AT_ONCE = 5


class GeminiRAGProcessor:
    """GPU-optimized processor for Gemini API RAG system."""

    def __init__(self):
        self.setup_gpu()
        self.setup_model()

    def setup_gpu(self):
        """Setup GPU optimization for RTX 3060."""
        print("🎮 Setting up GPU for embeddings...")

        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            print(f"✅ GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")

            # Optimize for RTX 3060
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_per_process_memory_fraction(0.8)

        else:
            print("⚠️ GPU not available, using CPU")
            self.device = 'cpu'

    def setup_model(self):
        """Setup embedding model."""
        print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
        print(f"🎯 Device: {self.device}")

        try:
            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=self.device
            )

            if self.device == 'cuda':
                self.model.eval()
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                print(f"📊 GPU memory allocated: {allocated:.2f}GB")

            print("✅ Embedding model ready")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def clean_markdown(self, text: str) -> str:
        """Clean markdown formatting for better processing."""
        # Remove headers but keep structure
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # Convert links to text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove code blocks but keep inline code
        text = re.sub(r'```[\s\S]*?```', '[CODE BLOCK]', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # Clean whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def chunk_text_smart(self, text: str) -> List[str]:
        """Smart chunking optimized for Gemini API context."""
        if len(text) <= CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            if end < len(text):
                # Try to break at paragraph boundaries first
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + 100:  # Ensure chunk isn't too small
                    end = para_break
                else:
                    # Try sentence boundaries
                    sent_break = text.rfind('. ', start, end)
                    if sent_break > start + 100:
                        end = sent_break + 1
                    else:
                        # Try word boundaries
                        word_break = text.rfind(' ', start, end)
                        if word_break > start + 100:
                            end = word_break

            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Skip very small chunks
                chunks.append(chunk)

            start = end - CHUNK_OVERLAP
            if end >= len(text):
                break

        return chunks

    def process_files_with_metadata(self, file_paths: List[str]) -> Iterator[tuple]:
        """Process files with enhanced metadata for Gemini."""
        total_files = len(file_paths)

        for i in range(0, total_files, MAX_FILES_AT_ONCE):
            batch_files = file_paths[i:i + MAX_FILES_AT_ONCE]
            batch_chunks = []
            batch_metadata = []

            batch_num = i // MAX_FILES_AT_ONCE + 1
            total_batches = (total_files - 1) // MAX_FILES_AT_ONCE + 1

            print(f"📂 Processing batch {batch_num}/{total_batches}")

            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.strip():
                        continue

                    # Enhanced processing
                    cleaned = self.clean_markdown(content)
                    chunks = self.chunk_text_smart(cleaned)
                    filename = os.path.basename(file_path)

                    # Extract document structure info
                    headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
                    has_code = bool(re.search(r'```', content))
                    has_links = bool(re.search(r'\[.*?\]\(.*?\)', content))

                    for j, chunk in enumerate(chunks):
                        batch_chunks.append(chunk)
                        batch_metadata.append({
                            'source': filename,
                            'chunk_id': j,
                            'total_chunks': len(chunks),
                            'file_size': len(content),
                            'has_code': has_code,
                            'has_links': has_links,
                            'headers_count': len(headers),
                            'chunk_size': len(chunk),
                            'file_type': 'markdown',
                            'processed_date': '2025-05-30T06:02:51Z'
                        })

                    print(f"  ✅ {filename}: {len(chunks)} chunks")

                except Exception as e:
                    print(f"  ❌ Error processing {file_path}: {e}")

            if batch_chunks:
                yield batch_chunks, batch_metadata

            # GPU cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    def create_embeddings_optimized(self, texts: List[str]) -> List[List[float]]:
        """GPU-optimized embedding creation."""
        print(f"⚡ Creating embeddings for {len(texts)} chunks")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )

            if self.device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                print(f"📊 GPU memory: {allocated:.2f}GB")
                torch.cuda.empty_cache()

            return embeddings.tolist()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ GPU memory issue, using smaller batches...")
                return self._create_embeddings_fallback(texts)
            else:
                raise

    def _create_embeddings_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback for memory issues."""
        all_embeddings = []
        fallback_batch_size = 8

        for i in range(0, len(texts), fallback_batch_size):
            batch = texts[i:i + fallback_batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=fallback_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings.tolist())

            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return all_embeddings


def setup_gemini_database():
    """Setup ChromaDB for Gemini API system."""
    print("🗄️ Setting up ChromaDB for Gemini API...")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Clear existing collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Cleared existing collection")
    except:
        pass

    # Create collection with Gemini-specific metadata
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Markdown documents for Gemini API RAG",
            "llm_provider": "Google AI Studio",
            "embedding_model": EMBEDDING_MODEL,
            "device": "RTX_3060",
            "user": "Felixcegep",
            "created": "2025-05-30T06:02:51Z",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
    )

    return collection


def main():
    """Main processing for Gemini API RAG system."""
    print("🤖 GOOGLE AI STUDIO + GPU RAG CREATION")
    print(f"👤 User: Felixcegep")
    print(f"📅 Date: 2025-05-30 06:02:51 UTC")
    print("=" * 60)

    # System info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
        print(f"🔥 CUDA: {torch.version.cuda}")

    print(f"📊 Configuration:")
    print(f"   • LLM: Google AI Studio (Gemini)")
    print(f"   • Embeddings: {EMBEDDING_MODEL} (GPU)")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Chunk size: {CHUNK_SIZE}")
    print(f"   • Database: {CHROMA_DB_PATH}")
    print("=" * 60)

    # Check files
    if not os.path.exists(SECTIONS_DIR):
        print(f"❌ Create '{SECTIONS_DIR}' directory and add .md files")
        return

    md_files = glob.glob(os.path.join(SECTIONS_DIR, "*.md"))
    if not md_files:
        print(f"❌ No .md files found in '{SECTIONS_DIR}'")
        return

    print(f"📂 Found {len(md_files)} markdown files")

    # Process files
    start_time = time.time()

    try:
        processor = GeminiRAGProcessor()
        collection = setup_gemini_database()

        total_chunks = 0
        chunk_id_counter = 0

        for batch_chunks, batch_metadata in processor.process_files_with_metadata(md_files):
            batch_start = time.time()

            print(f"  ⚡ GPU processing {len(batch_chunks)} chunks...")
            embeddings = processor.create_embeddings_optimized(batch_chunks)

            # Generate IDs
            ids = [f"gemini_chunk_{chunk_id_counter + i}" for i in range(len(batch_chunks))]
            chunk_id_counter += len(batch_chunks)

            # Add to database
            print(f"  💾 Storing in ChromaDB...")
            collection.add(
                embeddings=embeddings,
                documents=batch_chunks,
                metadatas=batch_metadata,
                ids=ids
            )

            batch_time = time.time() - batch_start
            total_chunks += len(batch_chunks)

            print(f"  ✅ Batch completed in {batch_time:.1f}s (Total: {total_chunks} chunks)")

            # Cleanup
            del embeddings, batch_chunks, batch_metadata
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"❌ Processing error: {e}")
        return

    total_time = time.time() - start_time

    print("=" * 60)
    print("🎉 GEMINI RAG DATABASE COMPLETE!")
    print(f"📊 Final Summary:")
    print(f"   • Files processed: {len(md_files)}")
    print(f"   • Total chunks: {total_chunks}")
    print(f"   • Processing time: {total_time:.1f} seconds")
    print(f"   • Speed: {total_chunks / total_time:.1f} chunks/second")
    print(f"   • Database: {CHROMA_DB_PATH}")
    print(f"   • Ready for: Google AI Studio API")
    print("=" * 60)
    print("🎯 Next Steps:")
    print("1. Get Google AI Studio API key from: https://aistudio.google.com/")
    print("2. Set API key: export GOOGLE_API_KEY='your-api-key'")
    print("3. Run: python option1_gemini/query.py")
    print("=" * 60)


if __name__ == "__main__":
    main()