#!/usr/bin/env python3
"""
Query the RAG system using FAISS index and combined chunks.
"""

import json
import numpy as np
from pathlib import Path
from openai import OpenAI
import faiss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.settings import *

class RAGQueryEngine:
    def __init__(self):
        """Initialize the RAG query engine."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.index = None
        self.chunks = None
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index and combined chunks."""
        try:
            # Load FAISS index
            index_path = Path("outputs/faiss_index.bin")
            if not index_path.exists():
                raise FileNotFoundError("FAISS index not found. Run the pipeline first.")
            
            self.index = faiss.read_index(str(index_path))
            logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load combined chunks
            chunks_path = Path("outputs/combined_chunks.json")
            if not chunks_path.exists():
                raise FileNotFoundError("Combined chunks not found. Run the pipeline first.")
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.chunks)} semantic chunks")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text query."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            embedding = response.data[0].embedding
            return np.array(embedding).astype('float32').reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search for semantically similar chunks.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with chunk information and similarity scores
        """
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Get results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'chunk': chunk
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def print_results(self, query: str, results: list):
        """Print search results in a nice format."""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 80)
        
        for result in results:
            chunk = result['chunk']
            print(f"\n📄 Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"📚 Paper: {chunk.get('source_title', 'Unknown')} ({chunk.get('year', 'Unknown')})")
            print(f"📖 Section: {chunk.get('section_header', 'Unknown')}")
            print(f"🎯 Topic: {chunk.get('semantic_topic', 'Unknown')}")
            print(f"📝 Text: {chunk.get('text', '')[:300]}...")
            print("-" * 80)
    
    def interactive_search(self):
        """Run interactive search mode."""
        print("\n🤖 Michael Levin RAG Query Engine")
        print("=" * 50)
        print("Type your questions about Michael Levin's research.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                query = input("❓ Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print(f"\n🔍 Searching for: '{query}'...")
                results = self.search(query, top_k=5)
                
                if results:
                    self.print_results(query, results)
                else:
                    print("❌ No results found.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function."""
    try:
        # Initialize query engine
        engine = RAGQueryEngine()
        
        # Check if we have data
        if not engine.chunks or len(engine.chunks) == 0:
            print("❌ No semantic chunks found. Run the pipeline first.")
            return
        
        # Run interactive search
        engine.interactive_search()
        
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 