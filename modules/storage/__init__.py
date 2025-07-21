"""
Storage Module

This module provides interfaces for storing and retrieving PDF text extraction results,
following the interface-first design principle for easy provider swapping.
"""

from .base import StorageInterface
from .json_storage import JSONStorage

# Registry of available storage providers
STORAGE_REGISTRY = {
    "json": JSONStorage,
}

def get_storage(provider: str = "json") -> StorageInterface:
    """
    Factory function to get a storage instance.
    
    Args:
        provider: Name of the storage provider
        
    Returns:
        Storage instance implementing StorageInterface
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider not in STORAGE_REGISTRY:
        available = ", ".join(STORAGE_REGISTRY.keys())
        raise ValueError(f"Unknown storage provider '{provider}'. Available: {available}")
    
    storage_class = STORAGE_REGISTRY[provider]
    return storage_class()

def get_available_storage_providers() -> list[str]:
    """Get list of available storage providers."""
    return list(STORAGE_REGISTRY.keys()) 