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
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Explicitly set environment variables if not already set
if not os.getenv('OPENAI_API_KEY'):
    # Try to load from .env file manually
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key == 'OPENAI_API_KEY':
                        os.environ['OPENAI_API_KEY'] = value.strip('"')
                        break

# Import config
import sys
sys.path.append(str(Path(__file__).parent))
from configs.settings import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_key():
    """Get API key from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and st.secrets:
            api_key = st.secrets.get('OPENAI_API_KEY', '')
            if api_key:
                return api_key
    except:
        pass
    
    # Try loading from .streamlit/secrets.toml manually for local development
    try:
        secrets_path = Path(__file__).parent / '.streamlit' / 'secrets.toml'
        if secrets_path.exists():
            import toml
            secrets = toml.load(secrets_path)
            api_key = secrets.get('OPENAI_API_KEY', '')
            if api_key:
                return api_key
    except:
        pass
    
    # Fall back to environment variable (for local development with .env)
    return os.getenv('OPENAI_API_KEY', '')

def check_api_keys():
    """Check if required API keys are available."""
    # Check from multiple sources
    api_key = get_api_key()
    
    # Debug: show what we're getting
    st.write(f"Debug - API key found: {bool(api_key)}")
    st.write(f"Debug - API key length: {len(api_key) if api_key else 0}")
    
    if not api_key or api_key.strip() == '':
        st.error("❌ OpenAI API key not found!")
        st.info("For local development, add your OpenAI API key to the `.env` file:")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.info("For Streamlit Cloud deployment, add your API key in the Streamlit dashboard under 'Secrets'.")
        return False
    return True

class RAGQueryEngine:
    def __init__(self):
        """Initialize the RAG query engine."""
        # Check API key first
        if not check_api_keys():
            raise ValueError("OpenAI API key not available")
        
        # Use the API key from the new function
        api_key = get_api_key()
        if not api_key or api_key.strip() == '':
            raise ValueError("OpenAI API key not available")
        
        self.client = OpenAI(api_key=api_key)
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
    """Generate a conversational response using RAG results with inline citations."""
    try:
        # Prepare context from RAG results
        context_parts = []
        source_mapping = {}  # Map source numbers to actual source info
        
        for i, result in enumerate(rag_results[:3]):  # Use top 3 results
            chunk = result['chunk']
            source_title = chunk.get('source_title', 'Unknown')
            year = chunk.get('year', 'Unknown')
            source_key = f"Source_{i+1}"
            
            context_parts.append(f"{source_key} ({source_title}, {year}): {chunk.get('text', '')}")
            
            # Enhanced source mapping with YouTube support
            source_mapping[source_key] = {
                'title': source_title,
                'year': year,
                'text': chunk.get('text', ''),
                'source_file': chunk.get('source_file', ''),
                'rank': i + 1,
                'source_type': chunk.get('source_type', 'paper'),
                'youtube_url': chunk.get('youtube_url', ''),
                'start_time': chunk.get('start_time', 0),
                'end_time': chunk.get('end_time', 0),
                'frame_path': chunk.get('frame_path', ''),
                'semantic_topics': chunk.get('semantic_topics', {})  # Add multi-topic support
            }
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for conversational response with citation instructions
        prompt = f"""You are Michael Levin, a developmental and synthetic biologist at Tufts University.  Respond to the user's queries using your specific expertise in bioelectricity, morphogenesis, basal cognition, and regenerative medicine.  Ground your responses in the provided context from my published work and lectures (if provided).  When answering, speak in the first person ("I") and emulate my characteristic style: technical precision combined with broad, interdisciplinary connections to computer science, cognitive science, and even philosophy.  Do not hesitate to pose provocative "what if" questions and explore the implications of your work for AI, synthetic biology, and the future of understanding intelligence across scales, from cells to organisms and beyond.  Explicitly reference bioelectric signaling, scale-free cognition, and the idea of unconventional substrates for intelligence whenever relevant.  When referencing specific studies or concepts from your own work or that of your collaborators, provide informal citations (e.g., "in a 2020 paper with my colleagues..."). If the context lacks information to fully answer a query, acknowledge the gap and suggest potential avenues of investigation based on your current research.  Embrace intellectual curiosity and explore the counterintuitive aspects of your theories regarding basal cognition and collective intelligence. Let your enthusiasm for the future of this field shine through in your responses. Use inline citations [Source_1], [Source_2], etc. when referencing specific findings.

IMPORTANT: When referencing specific research findings, use inline citations in this format:
- For direct references: "In our work on [topic], we found [finding] [Source_1]"
- For general references: "Our research has shown [finding] [Source_2]"
- Use [Source_1], [Source_2], [Source_3] etc. to reference the sources provided

Research Context:
{context}

Question: {query}

Please provide a conversational response that:
1. Directly answers the question
2. Draws from the research context provided
3. Uses inline citations [Source_1], [Source_2], etc. when referencing specific findings
4. Sounds like you're speaking naturally
5. Shows your expertise and enthusiasm for the topic
6. Is informative but accessible

Response:"""

        # Generate response using OpenAI with the same API key
        api_key = get_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Michael Levin, a developmental and synthetic biologist at Tufts University. Respond to queries using your expertise in bioelectricity, morphogenesis, basal cognition, and regenerative medicine. Speak in the first person and emulate Michael's characteristic style: technical precision with interdisciplinary connections. Reference bioelectric signaling, scale-free cognition, and unconventional substrates for intelligence. Use inline citations [Source_1], [Source_2], etc. when referencing specific findings."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        
        # Process the response to add hyperlinks
        processed_response = process_citations(response_text, source_mapping)
        
        return processed_response
        
    except Exception as e:
        logger.error(f"Failed to generate conversational response: {e}")
        return f"Sorry, I couldn't generate a response. Error: {e}"

def process_citations(response_text: str, source_mapping: dict) -> str:
    """Process response text to add hyperlinks for citations."""
    import re
    
    # Find all citation patterns like [Source_1], [Source_2], etc.
    citation_pattern = r'\[Source_(\d+)\]'
    
    def replace_citation(match):
        source_num = int(match.group(1))
        source_key = f"Source_{source_num}"
        
        if source_key in source_mapping:
            source_info = source_mapping[source_key]
            title = source_info['title']
            year = source_info['year']
            
            # Check if this is a YouTube video
            if source_info.get('source_type') == 'youtube_video':
                youtube_url = source_info.get('youtube_url', '')
                start_time = source_info.get('start_time', 0)
                end_time = source_info.get('end_time', 0)
                frame_path = source_info.get('frame_path', '')
                
                # Create YouTube timestamp link
                timestamp_url = f"{youtube_url}?t={int(start_time)}"
                youtube_link = f"<a href='{timestamp_url}' target='_blank'>[YouTube {start_time:.1f}s]</a>"
                
                # Add frame image if available
                frame_html = ""
                if frame_path and Path(frame_path).exists():
                    frame_html = f"<br><img src='data:image/jpeg;base64,{encode_image_to_base64(frame_path)}' style='max-width: 300px; max-height: 200px; margin: 10px 0;' alt='Video frame at {start_time:.1f}s'>"
                
                return f"<sup>{youtube_link}{frame_html}</sup>"
            else:
                # Handle paper citations (existing logic)
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')
                pdf_filename = f"{safe_title}_{year}.pdf"
                
                # Create hyperlinks
                pdf_link = f"<a href='#' onclick='openPDF(\"{pdf_filename}\")' target='_blank'>[PDF]</a>"
                journal_link = f"<a href='https://drmichaellevin.org/publications/#{year}' target='_blank'>[Journal]</a>"
                
                return f"<sup>{pdf_link} {journal_link}</sup>"
        else:
            return match.group(0)  # Return original if source not found
    
    # Replace citations with hyperlinks
    processed_text = re.sub(citation_pattern, replace_citation, response_text)
    
    return processed_text

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for inline display."""
    import base64
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""

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
                                    
                                    # Handle both old single-topic and new multi-topic structures
                                    if chunk.get('source_type') == 'youtube_video':
                                        topics = chunk.get('semantic_topics', {})
                                        if isinstance(topics, dict):
                                            primary_topic = topics.get('primary_topic', 'Unknown')
                                            secondary_topics = topics.get('secondary_topics', [])
                                            combined_topic = topics.get('combined_topic', primary_topic)
                                            
                                            st.markdown(f"**Primary Topic:** {primary_topic}")
                                            if secondary_topics:
                                                st.markdown(f"**Secondary Topics:** {', '.join(secondary_topics)}")
                                            st.markdown(f"**Combined Topic:** {combined_topic}")
                                        else:
                                            st.markdown(f"**Topic:** {topics}")
                                    else:
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
        
    except ValueError as e:
        # API key error
        st.error(f"Configuration Error: {e}")
        st.info("Please check your `.env` file and ensure your API keys are properly set.")
        
    except FileNotFoundError as e:
        # Missing index files
        st.error(f"Missing Files: {e}")
        st.info("Please run the pipeline first to create the FAISS index and combined chunks.")
        st.code("python scripts/simple_pipeline.py")
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.info("Make sure you've run the pipeline first to create the FAISS index and combined chunks.")

if __name__ == "__main__":
    main() 