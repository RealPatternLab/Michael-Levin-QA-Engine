"""
Centralized AI model interface for the Levin QA Engine.

This module provides a unified interface for different AI model providers,
allowing easy swapping between OpenAI and other models as they become available.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from .base import BaseModelInterface
from .openai_model import OpenAIModel

# Registry of available models
MODEL_REGISTRY = {
    "openai": OpenAIModel,
    "openai-gpt4": OpenAIModel,
    "openai-gpt4-vision": OpenAIModel,
    "openai-gpt4o": OpenAIModel,
}

def get_model(model_name: str, **kwargs) -> BaseModelInterface:
    """
    Factory function to get a model instance.
    
    Args:
        model_name: Name of the model provider (openai, etc.)
        **kwargs: Additional configuration for the model
        
    Returns:
        Model instance implementing BaseModelInterface
        
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)

def get_available_models() -> List[str]:
    """Get list of available model providers."""
    return list(MODEL_REGISTRY.keys())

# Convenience function for async calls
async def call_model_async(model_name: str, prompt: str, **kwargs) -> str:
    """
    Convenience function for async model calls.
    
    Args:
        model_name: Name of the model provider
        prompt: The prompt to send to the model
        **kwargs: Additional arguments for the model call
        
    Returns:
        Model response as string
    """
    model = get_model(model_name, **kwargs)
    return await model.call_async(prompt, **kwargs)

# Convenience function for sync calls
def call_model(model_name: str, prompt: str, **kwargs) -> str:
    """
    Convenience function for sync model calls.
    
    Args:
        model_name: Name of the model provider
        prompt: The prompt to send to the model
        **kwargs: Additional arguments for the model call
        
    Returns:
        Model response as string
    """
    model = get_model(model_name, **kwargs)
    return model.call(prompt, **kwargs) 