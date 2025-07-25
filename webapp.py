#!/usr/bin/env python3
"""
Entry point for the Michael Levin Research Assistant webapp.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main app with navigation."""
    # Check which page to show
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Navigation
    if st.session_state.page == "research_search":
        from webapp.pages.research_search import main as research_search
        research_search()
    elif st.session_state.page == "chat":
        from webapp.pages.chat import main as chat
        chat()
    else:
        # Show home page
        from webapp.Home import main as home
        home()

if __name__ == "__main__":
    main() 