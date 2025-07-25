#!/usr/bin/env python3
"""
Streamlit Web App for Michael Levin RAG System
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
import faiss
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import config
import sys
sys.path.append(str(Path(__file__).parent))
from configs.settings import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """Search for semantically similar chunks."""
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

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Michael Levin RAG System",
        page_icon="🧠",
        layout="wide"
    )
    
    # Header
    st.title("🧠 Michael Levin Research Assistant")
    st.markdown("Ask questions about Michael Levin's research on developmental biology, collective intelligence, and bioelectricity.")
    
    # Initialize RAG engine
    try:
        if 'rag_engine' not in st.session_state:
            with st.spinner("Loading RAG system..."):
                st.session_state.rag_engine = RAGQueryEngine()
        
        # Sidebar
        st.sidebar.header("🔧 Settings")
        top_k = st.sidebar.slider("Number of results", 1, 10, 5)
        
        # Show stats
        if st.session_state.rag_engine.chunks:
            st.sidebar.metric("Papers indexed", len(set(chunk.get('source_title', '') for chunk in st.session_state.rag_engine.chunks)))
            st.sidebar.metric("Total chunks", len(st.session_state.rag_engine.chunks))
        
        # Search interface
        st.header("🔍 Search Research Papers")
        
        # Query input
        query = st.text_input(
            "Ask a question about Michael Levin's research:",
            placeholder="e.g., How do developmental systems exhibit collective intelligence?"
        )
        
        # Search button
        if st.button("🔍 Search", type="primary"):
            if query.strip():
                with st.spinner("Searching..."):
                    try:
                        results = st.session_state.rag_engine.search(query, top_k=top_k)
                        
                        if results:
                            st.success(f"Found {len(results)} relevant results!")
                            
                            # Display results
                            for i, result in enumerate(results):
                                chunk = result['chunk']
                                score = result['score']
                                
                                with st.expander(f"📄 Result {i+1} (Score: {score:.4f})", expanded=(i==0)):
                                    col1, col2 = st.columns([1, 3])
                                    
                                    with col1:
                                        st.metric("Similarity", f"{score:.1%}")
                                    
                                    with col2:
                                        st.markdown(f"**Paper:** {chunk.get('source_title', 'Unknown')} ({chunk.get('year', 'Unknown')})")
                                        st.markdown(f"**Section:** {chunk.get('section_header', 'Unknown')}")
                                        st.markdown(f"**Topic:** {chunk.get('semantic_topic', 'Unknown')}")
                                    
                                    st.markdown("**Content:**")
                                    st.markdown(f"> {chunk.get('text', '')}")
                                    
                                    # Show full text in a smaller box
                                    with st.expander("📖 View full text"):
                                        st.text(chunk.get('text', ''))
                        else:
                            st.warning("No results found. Try rephrasing your question.")
                            
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a question to search.")
        
        # Example questions
        st.header("💡 Example Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("How do cells communicate?"):
                st.session_state.example_query = "How do cells communicate?"
                st.rerun()
        
        with col2:
            if st.button("What is bioelectricity?"):
                st.session_state.example_query = "What is bioelectricity?"
                st.rerun()
        
        with col3:
            if st.button("Collective intelligence in biology"):
                st.session_state.example_query = "Collective intelligence in biology"
                st.rerun()
        
        # Handle example query
        if 'example_query' in st.session_state:
            query = st.session_state.example_query
            del st.session_state.example_query
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("Built with Streamlit, OpenAI embeddings, and FAISS vector search.")
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.info("Make sure you've run the pipeline first to create the FAISS index and combined chunks.")

if __name__ == "__main__":
    main() 