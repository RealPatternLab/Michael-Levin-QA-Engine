"""
PyMuPDF implementation of PDF processor interface.

This module provides PDF image rendering using PyMuPDF (fitz).
"""

import fitz
from pathlib import Path
from typing import List
import logging

from .base import PDFProcessorInterface, ImageRenderingResult, PageImage

logger = logging.getLogger(__name__)

class PyMuPDFProcessor(PDFProcessorInterface):
    """
    PyMuPDF implementation for PDF image rendering.
    
    Uses PyMuPDF to render PDF pages to high-quality images for multimodal analysis.
    """
    
    def __init__(self):
        """Initialize PyMuPDF processor."""
        self.name = "pymupdf"
        self.description = "PyMuPDF-based PDF image renderer"
    
    def render_pages_to_images(
        self,
        pdf_path: Path,
        output_dir: Path,
        image_format: str = "png",
        dpi: int = 150
    ) -> ImageRenderingResult:
        """
        Render PDF pages to images using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save rendered images
            image_format: Format for rendered images (png, jpg, etc.)
            dpi: Resolution for rendered images
            
        Returns:
            ImageRenderingResult with page images and metadata
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        pages = []
        
        try:
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                # Calculate zoom factor for desired DPI
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                
                # Save image
                image_filename = f"page_{page_num + 1:03d}.{image_format}"
                image_path = output_dir / image_filename
                pix.save(str(image_path))
                
                # Create page data
                page_data = PageImage(
                    page_number=page_num + 1,
                    image_path=image_path,
                    width=pix.width,
                    height=pix.height
                )
                pages.append(page_data)
                
                logger.info(f"Rendered page {page_num + 1}/{total_pages}")
                
        finally:
            doc.close()
        
        return ImageRenderingResult(
            pdf_path=pdf_path,
            pages=pages,
            total_pages=total_pages,
            output_dir=output_dir
        )
    
    def get_pdf_metadata(self, pdf_path: Path) -> dict:
        """
        Extract basic metadata from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        
        try:
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            }
            
            return metadata
            
        finally:
            doc.close()
    
    def validate_pdf(self, pdf_path: Path) -> bool:
        """
        Validate that the PDF can be processed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF is valid and can be processed
        """
        try:
            if not pdf_path.exists():
                return False
            
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            
            return page_count > 0
            
        except Exception as e:
            logger.warning(f"PDF validation failed for {pdf_path}: {e}")
            return False
    
    def get_processor_info(self) -> dict:
        """
        Get information about this processor.
        
        Returns:
            Dictionary with processor information
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": ["image_rendering", "metadata_extraction"],
            "supported_formats": ["png", "jpg", "jpeg", "tiff"],
            "max_dpi": 600,
            "library": "PyMuPDF"
        } 