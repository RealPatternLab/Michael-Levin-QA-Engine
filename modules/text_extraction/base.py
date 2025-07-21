"""
Abstract base interface for LLM text extraction from PDF images.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ExtractedTextResult:
    """Data structure for extracted text results."""
    paper_id: str
    extracted_text: str
    extraction_quality: str
    warnings: List[str]
    page_images: List[Path]
    extraction_prompt: str
    llm_response: str
    batch_order: List[int] = None  # Track the order of batches processed

@dataclass
class BatchExtractionResult:
    """Data structure for batch extraction results."""
    batch_id: int
    page_indices: List[int]  # Original page indices in this batch
    extracted_text: str
    extraction_quality: str
    warnings: List[str]
    extraction_prompt: str
    llm_response: str

class TextExtractionInterface(ABC):
    """
    Abstract interface for LLM text extraction from PDF images.
    
    This interface defines the contract for extracting text from PDF page images
    using multimodal LLM analysis.
    """
    
    @abstractmethod
    def extract_text_from_images(
        self,
        paper_id: str,
        page_images: List[Path],
        output_dir: Path,
        max_images_per_batch: int = 5
    ) -> ExtractedTextResult:
        """
        Extract text from PDF page images using multimodal LLM analysis.
        
        Args:
            paper_id: Unique identifier for the paper
            page_images: List of paths to page images (in order)
            output_dir: Directory to save results
            max_images_per_batch: Maximum number of images to send per LLM call
            
        Returns:
            ExtractedTextResult with extracted text and metadata
        """
        pass
    
    @abstractmethod
    def extract_text_from_batch(
        self,
        batch_id: int,
        page_images: List[Path],
        page_indices: List[int]
    ) -> BatchExtractionResult:
        """
        Extract text from a single batch of page images.
        
        Args:
            batch_id: Identifier for this batch
            page_images: List of paths to page images in this batch
            page_indices: Original page indices for ordering
            
        Returns:
            BatchExtractionResult with extracted text for this batch
        """
        pass
    
    @abstractmethod
    def generate_extraction_prompt(
        self,
        page_images: List[Path],
        batch_info: str = ""
    ) -> str:
        """
        Generate a prompt for the LLM to extract text from images.
        
        Args:
            page_images: List of paths to page images
            batch_info: Information about the current batch
            
        Returns:
            Formatted prompt string for the LLM
        """
        pass
    
    @abstractmethod
    def parse_extraction_response(self, response: str) -> str:
        """
        Parse the LLM response to extract the text content.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def assess_extraction_quality(self, extracted_text: str) -> str:
        """
        Assess the quality of extracted text.
        
        Args:
            extracted_text: The extracted text to assess
            
        Returns:
            Quality assessment string
        """
        pass 