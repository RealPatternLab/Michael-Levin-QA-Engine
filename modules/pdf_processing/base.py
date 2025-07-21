"""
Base interface for PDF processing modules.

This module defines the interface for PDF processors that render PDF pages to images
for multimodal LLM analysis.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class PageImage:
    """Represents a rendered page image."""
    page_number: int
    image_path: Path
    width: int
    height: int

@dataclass
class ImageRenderingResult:
    """Result of PDF image rendering."""
    pdf_path: Path
    pages: List[PageImage]
    total_pages: int
    output_dir: Path

class PDFProcessorInterface(ABC):
    """
    Abstract interface for PDF processors.
    
    PDF processors render PDF pages to images for multimodal LLM analysis.
    Text extraction is now handled entirely by the LLM.
    """
    
    @abstractmethod
    def render_pages_to_images(
        self,
        pdf_path: Path,
        output_dir: Path,
        image_format: str = "png",
        dpi: int = 150
    ) -> ImageRenderingResult:
        """
        Render PDF pages to images for multimodal analysis.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save rendered images
            image_format: Format for rendered images (png, jpg, etc.)
            dpi: Resolution for rendered images
            
        Returns:
            ImageRenderingResult with page images and metadata
        """
        pass
    
    @abstractmethod
    def get_pdf_metadata(self, pdf_path: Path) -> dict:
        """
        Extract basic metadata from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata (page count, title, etc.)
        """
        pass
    
    @abstractmethod
    def validate_pdf(self, pdf_path: Path) -> bool:
        """
        Validate that the PDF can be processed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF is valid and can be processed
        """
        pass
    
    @abstractmethod
    def get_processor_info(self) -> dict:
        """
        Get information about this processor.
        
        Returns:
            Dictionary with processor information
        """
        pass 