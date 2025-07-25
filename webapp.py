#!/usr/bin/env python3
"""
Entry point for the Michael Levin Research Assistant webapp.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import and run the main app
from webapp.Home import main

if __name__ == "__main__":
    main() 