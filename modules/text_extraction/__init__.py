"""
Text extraction module registry and factory functions.

This module provides a unified interface for different LLM text extraction providers,
allowing easy swapping between OpenAI and other providers as they become available.
"""

from typing import Dict, Type
from .base import TextExtractionInterface
from .openai import OpenAITextExtractor

# Registry of available text extraction processors
EXTRACTION_PROCESSOR_REGISTRY = {
    "openai": OpenAITextExtractor,
    "openai-gpt4-vision": OpenAITextExtractor,
}

def get_extraction_processor(processor_name: str, **kwargs) -> TextExtractionInterface:
    """
    Factory function to get a text extraction processor instance.
    
    Args:
        processor_name: Name of the processor to use
        **kwargs: Additional arguments to pass to the processor constructor
        
    Returns:
        Text extraction processor instance implementing TextExtractionInterface
        
    Raises:
        ValueError: If processor is not supported
    """
    if processor_name not in EXTRACTION_PROCESSOR_REGISTRY:
        available = ", ".join(EXTRACTION_PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unsupported text extraction processor: {processor_name}. Available: {available}")
    
    processor_class = EXTRACTION_PROCESSOR_REGISTRY[processor_name]
    return processor_class(processor_name, **kwargs)

# Backward compatibility alias
get_cleaning_processor = get_extraction_processor 