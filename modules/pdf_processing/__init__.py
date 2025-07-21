"""
PDF Processing Module

This module provides interfaces for PDF text extraction and image rendering,
following the interface-first design principle for easy provider swapping.
"""

from .base import PDFProcessorInterface
from .pdfplumber_processor import PDFPlumberProcessor
from .pymupdf_processor import PyMuPDFProcessor

# Registry of available PDF processors
PROCESSOR_REGISTRY = {
    "pdfplumber": PDFPlumberProcessor,
    "pymupdf": PyMuPDFProcessor,
}

def get_pdf_processor(provider: str = "pdfplumber") -> PDFProcessorInterface:
    """
    Factory function to get a PDF processor instance.
    
    Args:
        provider: Name of the PDF processor provider
        
    Returns:
        PDF processor instance implementing PDFProcessorInterface
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider not in PROCESSOR_REGISTRY:
        available = ", ".join(PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown PDF processor '{provider}'. Available: {available}")
    
    processor_class = PROCESSOR_REGISTRY[provider]
    return processor_class()

def get_available_processors() -> list[str]:
    """Get list of available PDF processor providers."""
    return list(PROCESSOR_REGISTRY.keys()) 