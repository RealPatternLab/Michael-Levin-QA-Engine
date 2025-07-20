"""
Configuration file for the Michael Levin QA Engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"

# LLM Configuration
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1

# PDF Processing Configuration
MAX_TEXT_LENGTH = 2000  # Characters to send to LLM 