"""
OpenAI model implementation for the centralized AI interface.
"""

import os
from typing import Dict, Any, Optional, List
from openai import OpenAI, AsyncOpenAI
from .base import BaseModelInterface, ModelError, ModelConfigError

class OpenAIModel(BaseModelInterface):
    """
    OpenAI model implementation.
    
    Supports both sync and async calls to OpenAI's chat completions API.
    """
    
    def _setup_model(self):
        """Setup OpenAI client with configuration."""
        # Get API key from config or environment
        api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your-api-key-here':
            raise ModelConfigError("OpenAI API key not configured")
        
        if not api_key.startswith('sk-'):
            raise ModelConfigError("Invalid OpenAI API key format")
        
        # Create sync and async clients
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        
        # Set default model and parameters
        self.default_model = self.config.get('model', 'gpt-3.5-turbo')
        self.default_temperature = self.config.get('temperature', 0.1)
        self.default_max_tokens = self.config.get('max_tokens', 1000)
    
    def call(self, prompt: str, **kwargs) -> str:
        """
        Synchronous call to OpenAI.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments (model, temperature, max_tokens, etc.)
            
        Returns:
            Model response as string
            
        Raises:
            ModelError: If the API call fails
        """
        try:
            # Merge default config with kwargs
            call_config = {
                'model': kwargs.get('model', self.default_model),
                'temperature': kwargs.get('temperature', self.default_temperature),
                'max_tokens': kwargs.get('max_tokens', self.default_max_tokens),
            }
            
            # Handle different prompt formats
            messages = self._format_messages(prompt, kwargs)
            
            response = self.sync_client.chat.completions.create(
                messages=messages,
                **call_config
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise ModelError(f"OpenAI API call failed: {str(e)}")
    
    async def call_async(self, prompt: str, **kwargs) -> str:
        """
        Asynchronous call to OpenAI.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments (model, temperature, max_tokens, etc.)
            
        Returns:
            Model response as string
            
        Raises:
            ModelError: If the API call fails
        """
        try:
            # Merge default config with kwargs
            call_config = {
                'model': kwargs.get('model', self.default_model),
                'temperature': kwargs.get('temperature', self.default_temperature),
                'max_tokens': kwargs.get('max_tokens', self.default_max_tokens),
            }
            
            # Handle different prompt formats
            messages = self._format_messages(prompt, kwargs)
            
            response = await self.async_client.chat.completions.create(
                messages=messages,
                **call_config
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise ModelError(f"OpenAI API call failed: {str(e)}")
    
    def _format_messages(self, prompt: str, kwargs: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format prompt into OpenAI message format.
        
        Args:
            prompt: The prompt text
            kwargs: Additional arguments that may contain system_message
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message if provided
        system_message = kwargs.get('system_message')
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def validate_config(self) -> bool:
        """
        Validate OpenAI configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Test API key with a simple call
            test_response = self.sync_client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False 