#!/usr/bin/env python3
"""
Michael Levin Research Assistant - Main App
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from webapp.utils.api_keys import check_api_keys
from webapp.components.rag_engine import RAGQueryEngine

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Michael Levin Research Assistant",
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
        
        # Main content
        st.header("Welcome to the Michael Levin Research Assistant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Research Search
            Search through Michael Levin's research papers to find relevant information.
            """)
            if st.button("Go to Research Search", type="primary"):
                st.session_state.page = "research_search"
                st.rerun()
        
        with col2:
            st.markdown("""
            ### 💬 Chat with Michael Levin
            Have a conversation with Michael Levin about his research.
            """)
            if st.button("Start Chat", type="primary"):
                st.session_state.page = "chat"
                st.rerun()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("Built with Streamlit, OpenAI, and FAISS.")
        
    except ValueError as e:
        # API key error
        st.error(f"Configuration Error: {e}")
        st.info("Please check your `.env` file and ensure your API keys are properly set.")
        
    except FileNotFoundError as e:
        # Missing index files
        st.error(f"Missing Files: {e}")
        st.info("Please run the pipeline first to create the FAISS index and combined chunks.")
        st.code("python scripts/pipelines/paper_pipeline.py")
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.info("Make sure you've run the pipeline first to create the FAISS index and combined chunks.")

if __name__ == "__main__":
    main() 