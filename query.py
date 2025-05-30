"""
Google AI Studio + GPU RAG Query System
Updated: 2025-05-30 06:02:51 UTC by Felixcegep
Uses Google AI Studio API with GPU-accelerated embeddings
"""

import chromadb
import google.generativeai as genai
import gc
import time
import torch
import os
from sentence_transformers import SentenceTransformer
from typing import Dict, List

# Configuration
CHROMA_DB_PATH = "./chroma_db_gemini"
COLLECTION_NAME = "md_documents_gemini"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-1.5-flash"  # Fast and cost-effective


class GeminiRAG:
    """Google AI Studio + GPU RAG system."""

    def __init__(self):
        print("🤖 Starting Google AI Studio RAG...")
        self.setup_gpu()
        self.setup_gemini_api()
        self.load_components()
        print("✅ Gemini RAG system ready!")

    def setup_gpu(self):
        """Setup GPU for embeddings."""
        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")

            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_per_process_memory_fraction(0.6)  # Leave room for other processes

        else:
            print("💻 Using CPU for embeddings")
            self.device = 'cpu'

    def setup_gemini_api(self):
        """Setup Google AI Studio API."""
        print("🔑 Setting up Google AI Studio API...")

        # Get API key from environment
        api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            print("❌ Google API key not found!")
            print("Please set your API key:")
            print("1. Get key from: https://aistudio.google.com/")
            print("2. Set environment variable:")
            print("   Windows: set GOOGLE_API_KEY=your-api-key")
            print("   Linux/Mac: export GOOGLE_API_KEY=your-api-key")
            print("3. Or create a .env file with: GOOGLE_API_KEY=your-api-key")

            # Try to get key interactively
            api_key = input("\nOr enter your API key now: ").strip()
            if not api_key:
                raise Exception("Google API key required!")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Test API connection
        try:
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            test_response = self.model.generate_content("Hello")
            print(f"✅ Connected to {GEMINI_MODEL}")

        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            print("Please check your API key and internet connection")
            raise

    def load_components(self):
        """Load database and embedding model."""
        # Load database
        print("📂 Loading Gemini database...")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)

        count = self.collection.count()
        print(f"📚 Found {count} documents")

        if count == 0:
            raise Exception("Database empty! Run option1_gemini/main.py first.")

        # Load embedding model
        print(f"🧠 Loading embeddings: {EMBEDDING_MODEL}")

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            device=self.device
        )

        if self.device == 'cuda':
            self.embedder.eval()
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"📊 GPU memory: {allocated:.2f}GB")

    def search_documents(self, query: str, k: int = 5) -> Dict:
        """Search documents with GPU acceleration."""
        print(f"🔍 Searching: '{query}'")

        start_time = time.time()

        # Create embedding on GPU
        with torch.no_grad():
            query_embedding = self.embedder.encode([query])

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        search_time = time.time() - start_time

        # Cleanup
        del query_embedding
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        if not results['documents'][0]:
            return {
                'documents': [],
                'sources': [],
                'search_time': search_time
            }

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        sources = list(set([meta['source'] for meta in metadatas]))

        # Sort by relevance (lower distance = more relevant)
        doc_data = list(zip(documents, metadatas, distances))
        doc_data.sort(key=lambda x: x[2])  # Sort by distance

        sorted_docs = [item[0] for item in doc_data]
        sorted_meta = [item[1] for item in doc_data]

        print(f"📄 Found {len(documents)} relevant chunks from {len(sources)} files ({search_time:.3f}s)")

        return {
            'documents': sorted_docs,
            'metadatas': sorted_meta,
            'sources': sources,
            'distances': [item[2] for item in doc_data],
            'search_time': search_time
        }

    def generate_answer_gemini(self, query: str, context: str, sources: List[str]) -> tuple:
        """Generate answer using Google AI Studio."""
        # Create optimized prompt for Gemini
        prompt = f"""You are a helpful AI assistant analyzing markdown documents. Based on the provided context, answer the user's question accurately and comprehensively.

CONTEXT FROM DOCUMENTS:
{context[:8000]}  

USER QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, accurate answer based on the context
- Be specific and cite relevant details from the documents
- If the context doesn't fully answer the question, acknowledge this
- Keep your response well-structured and helpful
- Use the information from these source files: {', '.join(sources)}

ANSWER:"""

        try:
            print("🤖 Generating answer with Gemini...")
            start_time = time.time()

            # Generate with Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=1024,
                    stop_sequences=["USER QUESTION:", "CONTEXT FROM DOCUMENTS:"]
                )
            )

            generation_time = time.time() - start_time
            print(f"⚡ Generated in {generation_time:.1f}s")

            return response.text.strip(), generation_time

        except Exception as e:
            print(f"❌ Gemini generation error: {e}")
            return f"Sorry, there was an error generating the response: {e}", 0

    def ask(self, question: str) -> Dict:
        """Main query function."""
        if not question.strip():
            return {'error': 'Please provide a question'}

        total_start = time.time()

        try:
            # Search for relevant documents
            search_results = self.search_documents(question, k=6)

            if not search_results['documents']:
                return {
                    'answer': "❌ No relevant documents found for your question.",
                    'sources': [],
                    'search_time': search_results['search_time'],
                    'generation_time': 0,
                    'total_time': time.time() - total_start,
                    'model_used': GEMINI_MODEL
                }

            # Create rich context from top results
            context_pieces = []
            for i, (doc, meta) in enumerate(zip(search_results['documents'][:4], search_results['metadatas'][:4])):
                source = meta['source']
                context_pieces.append(f"[Source: {source}]\n{doc}")

            context = "\n\n---\n\n".join(context_pieces)

            # Generate answer with Gemini
            answer, generation_time = self.generate_answer_gemini(
                question,
                context,
                search_results['sources']
            )

            total_time = time.time() - total_start

            return {
                'answer': answer,
                'sources': search_results['sources'],
                'search_time': search_results['search_time'],
                'generation_time': generation_time,
                'total_time': total_time,
                'context_length': len(context),
                'num_sources': len(search_results['sources']),
                'model_used': GEMINI_MODEL,
                'relevance_scores': search_results['distances'][:3]
            }

        except Exception as e:
            return {'error': f"Error processing question: {e}"}

    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        stats = {
            'model': GEMINI_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'device': self.device,
            'database_path': CHROMA_DB_PATH
        }

        if self.device == 'cuda':
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024 ** 3,
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            })

        return stats


def display_setup_info():
    """Display setup information."""
    print("🔧 SETUP REQUIREMENTS:")
    print("=" * 40)
    print("1. Google AI Studio API Key:")
    print("   • Get free key: https://aistudio.google.com/")
    print("   • Set environment: GOOGLE_API_KEY=your-key")
    print("")
    print("2. Install dependencies:")
    print("   pip install google-generativeai")
    print("")
    print("3. Run database creation:")
    print("   python option1_gemini/main.py")


def main():
    """Main Gemini RAG query loop."""
    try:
        print("🤖 GOOGLE AI STUDIO + GPU RAG SYSTEM")
        print(f"👤 User: Felixcegep")
        print(f"📅 Date: 2025-05-30 06:02:51 UTC")
        print("=" * 60)

        # Check API key first
        if not os.getenv('GOOGLE_API_KEY'):
            display_setup_info()
            print("\n" + "=" * 60)

        rag = GeminiRAG()

        stats = rag.get_system_stats()

        print("\n" + "=" * 60)
        print("🚀 GEMINI RAG SYSTEM READY")
        print(f"🤖 LLM: {stats['model']}")
        print(f"🧠 Embeddings: {stats['embedding_model']} ({stats['device']})")
        if 'gpu_memory_allocated' in stats:
            print(f"🎮 GPU Memory: {stats['gpu_memory_allocated']:.2f}GB / {stats['gpu_memory_total']:.1f}GB")
        print("=" * 60)
        print("Commands:")
        print("  • Ask questions about your documents")
        print("  • 'stats' - Show system statistics")
        print("  • 'help' - Usage tips")
        print("  • 'exit' - Quit system")
        print("=" * 60)

        while True:
            try:
                question = input(f"\n💬 [Gemini] Question: ").strip()

                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break

                if question.lower() == 'stats':
                    current_stats = rag.get_system_stats()
                    print(f"\n📊 System Statistics:")
                    print(f"LLM Model: {current_stats['model']}")
                    print(f"Embeddings: {current_stats['embedding_model']}")
                    print(f"Device: {current_stats['device']}")
                    if 'gpu_memory_allocated' in current_stats:
                        print(f"GPU Memory: {current_stats['gpu_memory_allocated']:.2f}GB")
                    continue

                if question.lower() == 'help':
                    print("\n📖 Gemini RAG Tips:")
                    print("• Google AI Studio provides fast, high-quality responses")
                    print("• GPU acceleration for embedding search")
                    print("• Supports complex questions and detailed analysis")
                    print("• Type 'stats' to see system performance")
                    print("• Free tier: 15 requests/minute, 1M tokens/day")
                    continue

                if not question:
                    continue

                # Process question
                result = rag.ask(question)

                if 'error' in result:
                    print(f"❌ {result['error']}")
                    continue

                # Display comprehensive results
                print("\n" + "=" * 60)
                print("🤖 GEMINI ANSWER:")
                print("=" * 60)
                print(result['answer'])

                print(f"\n⚡ Performance Metrics:")
                print(f"Search time: {result['search_time']:.3f}s")
                print(f"Generation time: {result['generation_time']:.1f}s")
                print(f"Total time: {result['total_time']:.1f}s")
                print(f"Model: {result['model_used']}")

                print(f"\n📊 Context Analysis:")
                print(f"Sources used: {result['num_sources']} files")
                print(f"Context length: {result['context_length']:,} characters")
                print(f"Top relevance scores: {[f'{score:.3f}' for score in result['relevance_scores'][:3]]}")

                if result['sources']:
                    print(f"\n📚 Source Files:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source}")

                print("=" * 60)

                # GPU cleanup
                if rag.device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        if "API key" in str(e):
            display_setup_info()


if __name__ == "__main__":
    main()