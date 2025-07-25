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

def get_conversational_response(query: str, rag_results: list) -> str:
    """Generate a conversational response using RAG results."""
    try:
        # Prepare context from RAG results
        context_parts = []
        for i, result in enumerate(rag_results[:3]):  # Use top 3 results
            chunk = result['chunk']
            context_parts.append(f"Source {i+1} ({chunk.get('source_title', 'Unknown')}): {chunk.get('text', '')}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for conversational response
        prompt = f"""You are Michael Levin, a renowned developmental biologist and researcher. 
You are being interviewed about your research on developmental biology, collective intelligence, and bioelectricity.

Based on the following research context from your papers, provide a conversational answer to the question.
Write as if you're speaking directly to the person asking the question.

Research Context:
{context}

Question: {query}

Please provide a conversational response that:
1. Directly answers the question
2. Draws from the research context provided
3. Sounds like you're speaking naturally
4. Shows your expertise and enthusiasm for the topic
5. Is informative but accessible

Response:"""

        # Generate response using OpenAI
        response = st.session_state.rag_engine.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Michael Levin, a developmental biologist. Respond conversationally and draw from the provided research context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Failed to generate conversational response: {e}")
        return f"Sorry, I couldn't generate a response. Error: {e}"

def rag_page():
    """Pure RAG retrieval page."""
    st.header("🔍 Research Paper Search")
    st.markdown("Search through Michael Levin's research papers to find relevant information.")
    
    # Query input
    query = st.text_input(
        "Ask a question about Michael Levin's research:",
        placeholder="e.g., How do developmental systems exhibit collective intelligence?"
    )
    
    # Search button
    if st.button("🔍 Search Papers", type="primary"):
        if query.strip():
            with st.spinner("Searching..."):
                try:
                    results = st.session_state.rag_engine.search(query, top_k=5)
                    
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

def conversational_page():
    """Conversational interface page."""
    st.header("💬 Chat with Michael Levin")
    st.markdown("Have a conversation with Michael Levin about his research. He'll answer your questions based on his papers.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask Michael Levin a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Michael is thinking..."):
                try:
                    # First, get RAG results
                    rag_results = st.session_state.rag_engine.search(prompt, top_k=5)
                    
                    if rag_results:
                        # Generate conversational response
                        response = get_conversational_response(prompt, rag_results)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Show sources used
                        with st.expander("📚 Sources used"):
                            for i, result in enumerate(rag_results[:3]):
                                chunk = result['chunk']
                                st.markdown(f"**{i+1}.** {chunk.get('source_title', 'Unknown')} ({chunk.get('year', 'Unknown')}) - {chunk.get('section_header', 'Unknown')}")
                    else:
                        response = "I'm sorry, but I couldn't find relevant information in my research papers to answer that question. Could you try rephrasing it or asking about a different aspect of developmental biology, collective intelligence, or bioelectricity?"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Michael Levin RAG System",
        page_icon="🧠",
        layout="wide"
    )
    
    # Header
    st.title("🧠 Michael Levin Research Assistant")
    st.markdown("Explore Michael Levin's research on developmental biology, collective intelligence, and bioelectricity.")
    
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
        
        # Page selection
        st.sidebar.header("📄 Pages")
        page = st.sidebar.radio(
            "Choose a page:",
            ["🔍 Research Search", "💬 Chat with Michael Levin"]
        )
        
        # Display selected page
        if page == "🔍 Research Search":
            rag_page()
        elif page == "💬 Chat with Michael Levin":
            conversational_page()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("Built with Streamlit, OpenAI, and FAISS.")
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.info("Make sure you've run the pipeline first to create the FAISS index and combined chunks.")

if __name__ == "__main__":
    main() 