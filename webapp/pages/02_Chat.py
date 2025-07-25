#!/usr/bin/env python3
"""
Chat with Michael Levin Page
"""

import streamlit as st
from webapp.components.rag_engine import RAGQueryEngine
from webapp.components.citation_processor import get_conversational_response

st.set_page_config(
    page_title="Chat with Michael Levin - Michael Levin RAG System",
    page_icon="💬",
    layout="wide"
)

def main():
    """Conversational interface page."""
    st.header("💬 Chat with Michael Levin")
    st.markdown("Have a conversation with Michael Levin about his research. He'll answer your questions based on his papers.")
    
    # Add JavaScript for PDF opening
    st.markdown("""
    <script>
    function openPDF(filename) {
        if (filename) {
            // For now, show an alert since we don't have PDFs served
            // In a full implementation, you'd serve the PDF from your server
            alert('PDF functionality coming soon! This would open: ' + filename);
            // window.open('/pdfs/' + filename, '_blank');
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Display HTML content for assistant messages
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
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
                        # Generate conversational response with citations
                        response = get_conversational_response(prompt, rag_results)
                        
                        # Display response with HTML support
                        st.markdown(response, unsafe_allow_html=True)
                        
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

if __name__ == "__main__":
    main() 