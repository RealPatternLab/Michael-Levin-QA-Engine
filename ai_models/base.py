"""
Base interface for AI model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio

class BaseModelInterface(ABC):
    """
    Abstract base class for AI model implementations.
    
    All model providers must implement this interface to ensure
    consistent behavior across different backends.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model interface.
        
        Args:
            **kwargs: Model-specific configuration
        """
        self.config = kwargs
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """
        Setup the model with configuration.
        Called during initialization.
        """
        pass
    
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """
        Synchronous call to the model.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            Model response as string
            
        Raises:
            ModelError: If the model call fails
        """
        pass
    
    @abstractmethod
    async def call_async(self, prompt: str, **kwargs) -> str:
        """
        Asynchronous call to the model.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            Model response as string
            
        Raises:
            ModelError: If the model call fails
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that the model is properly configured.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.__class__.__name__,
            "config": self.config,
            "is_async": hasattr(self, 'call_async'),
        }
    
    def __str__(self):
        return f"{self.__class__.__name__}(config={self.config})"

class ModelError(Exception):
    """Exception raised when a model call fails."""
    pass

class ModelConfigError(Exception):
    """Exception raised when model configuration is invalid."""
    pass 