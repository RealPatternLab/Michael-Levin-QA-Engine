#!/usr/bin/env python3
"""
API Key management utilities for the webapp.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        secrets_path = Path(__file__).parent.parent.parent / '.streamlit' / 'secrets.toml'
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