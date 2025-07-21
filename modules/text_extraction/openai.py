"""
OpenAI implementation of text extraction from PDF images.

This module provides text extraction using OpenAI's multimodal models (GPT-4o).
"""

from pathlib import Path
from typing import List

from .base_extractor import BaseTextExtractor
from ai_models import get_model

class OpenAITextExtractor(BaseTextExtractor):
    """
    OpenAI implementation for text extraction from PDF images.
    
    Uses OpenAI's multimodal models (GPT-4o) to extract text from PDF page images.
    Inherits all common functionality from BaseTextExtractor.
    """
    
    def _initialize_model(self, model_name: str):
        """
        Initialize the OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model to use
            
        Returns:
            Initialized OpenAI model instance
        """
        return get_model(model_name)
    
    def _call_model(self, prompt: str, images: List[Path], max_tokens: int = 8000) -> str:
        """
        Call the OpenAI model with prompt and images.
        
        Args:
            prompt: The prompt to send to the model
            images: List of image paths to send
            max_tokens: Maximum tokens for response
            
        Returns:
            Model response as string
        """
        return self.model.call(prompt, images=images, max_tokens=max_tokens) 