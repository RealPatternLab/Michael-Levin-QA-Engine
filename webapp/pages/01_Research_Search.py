#!/usr/bin/env python3
"""
Research Search Page
"""

import streamlit as st
from webapp.components.rag_engine import RAGQueryEngine

st.set_page_config(
    page_title="Research Search - Michael Levin RAG System",
    page_icon="🔍",
    layout="wide"
)

def main():
    """Research Search page."""
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

if __name__ == "__main__":
    main() 