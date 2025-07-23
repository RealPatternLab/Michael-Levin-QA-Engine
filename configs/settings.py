"""
Configuration settings for the PDF processing pipeline.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Input/Output paths
RAW_PAPERS_DIR = INPUTS_DIR / "raw_papers"
EXTRACTED_TEXTS_DIR = OUTPUTS_DIR / "extracted_texts"
METADATA_FILE = OUTPUTS_DIR / "metadata.json"

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

# Processing settings
MAX_TEXT_LENGTH = 2000  # Characters to extract for metadata
OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 500 