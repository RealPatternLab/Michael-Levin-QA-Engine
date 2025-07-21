"""
Configuration file for the Michael Levin QA Engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"

# AI Model API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Default AI Model Configuration
DEFAULT_MODEL = "openai"  # Default model provider
DEFAULT_MODEL_NAME = "gpt-3.5-turbo"  # Default model name
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000

# Model-specific configurations with detailed metadata
MODEL_CONFIGS = {
    "openai": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 1000,
        "api_key": OPENAI_API_KEY,
        "description": {
            "max_context_window": 16385,  # tokens
            "multimodal": False,
            "vision_capable": False,
            "cost_per_1k_tokens": 0.002,  # USD
            "speed": "fast",
            "best_for": ["text generation", "chat", "code", "reasoning"],
            "limitations": ["no vision", "limited context"],
            "provider": "OpenAI"
        }
    },
    "openai-gpt4": {
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 1000,
        "api_key": OPENAI_API_KEY,
        "description": {
            "max_context_window": 8192,  # tokens
            "multimodal": False,
            "vision_capable": False,
            "cost_per_1k_tokens": 0.03,  # USD
            "speed": "medium",
            "best_for": ["complex reasoning", "analysis", "creative writing"],
            "limitations": ["no vision", "expensive", "slower"],
            "provider": "OpenAI"
        }
    },
    "openai-gpt4-vision": {
        "model": "gpt-4-vision-preview",
        "temperature": 0.1,
        "max_tokens": 1000,
        "api_key": OPENAI_API_KEY,
        "description": {
            "max_context_window": 128000,  # tokens
            "multimodal": True,
            "vision_capable": True,
            "cost_per_1k_tokens": 0.01,  # USD
            "speed": "medium",
            "best_for": ["image analysis", "document understanding", "visual reasoning"],
            "limitations": ["expensive", "slower than text-only"],
            "provider": "OpenAI"
        }
    },
    "openai-gpt4o": {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 8000,
        "api_key": OPENAI_API_KEY,
        "description": {
            "max_context_window": 128000,  # tokens
            "multimodal": True,
            "vision_capable": True,
            "cost_per_1k_tokens": 0.005,  # USD
            "speed": "fast",
            "best_for": ["multimodal analysis", "document understanding", "visual reasoning"],
            "limitations": ["requires API key"],
            "provider": "OpenAI"
        }
    }
}

# PDF Processing Configuration
MAX_TEXT_LENGTH = 2000

# Legacy configuration (for backward compatibility)
LLM_MODEL = DEFAULT_MODEL_NAME
LLM_TEMPERATURE = DEFAULT_TEMPERATURE 