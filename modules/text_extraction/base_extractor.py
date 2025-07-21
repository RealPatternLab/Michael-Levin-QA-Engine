"""
Base implementation for LLM text extraction from PDF images.

This module provides a base class with common functionality that can be inherited
by specific LLM implementations (OpenAI, etc.).
"""

import logging
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod

from .base import TextExtractionInterface, ExtractedTextResult, BatchExtractionResult

logger = logging.getLogger(__name__)

class BaseTextExtractor(TextExtractionInterface, ABC):
    """
    Base implementation for text extraction from PDF images.
    
    This class provides common functionality that can be inherited by specific
    LLM implementations. Child classes only need to implement the model-specific
    parts like model initialization and LLM calls.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the base text extractor.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.model = self._initialize_model(model_name)
    
    @property
    def name(self) -> str:
        """Get the name of this extractor."""
        return self.model_name
    
    @property
    def description(self) -> str:
        """Get the description of this extractor."""
        return f"Text extraction using {self.model_name}"
    
    @abstractmethod
    def _initialize_model(self, model_name: str):
        """
        Initialize the specific LLM model.
        
        Args:
            model_name: Name of the model to initialize
            
        Returns:
            Initialized model instance
        """
        pass
    
    @abstractmethod
    def _call_model(self, prompt: str, images: List[Path], max_tokens: int = 8000) -> str:
        """
        Call the specific LLM model with prompt and images.
        
        Args:
            prompt: The prompt to send to the model
            images: List of image paths to send
            max_tokens: Maximum tokens for response
            
        Returns:
            Model response as string
        """
        pass
    
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
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images in batches
        all_warnings = []
        batch_results = []
        batch_order = []
        
        # Split images into batches
        num_batches = (len(page_images) + max_images_per_batch - 1) // max_images_per_batch
        
        logger.info(f"Processing {len(page_images)} pages in {num_batches} batches of max {max_images_per_batch} pages each")
        
        for batch_id in range(num_batches):
            start_idx = batch_id * max_images_per_batch
            end_idx = min(start_idx + max_images_per_batch, len(page_images))
            
            batch_images = page_images[start_idx:end_idx]
            page_indices = list(range(start_idx, end_idx))
            
            logger.info(f"Processing batch {batch_id + 1}/{num_batches} (pages {start_idx + 1}-{end_idx})")
            
            # Extract text from this batch
            batch_result = self.extract_text_from_batch(batch_id, batch_images, page_indices)
            batch_results.append(batch_result)
            batch_order.append(batch_id)
            
            # Store batch result for later ordering
            all_warnings.extend(batch_result.warnings)
        
        # Combine all extracted text in correct order
        combined_text = self._combine_batch_results(batch_results, page_images)
        
        # Assess overall extraction quality
        extraction_quality = self.assess_extraction_quality(combined_text)
        
        # Generate additional warnings about batch processing
        if num_batches > 1:
            all_warnings.append(f"Document processed in {num_batches} batches to stay within token limits")
        
        # Save extracted text
        extracted_text_path = output_dir / f"{paper_id}_extracted.txt"
        extracted_text_path.write_text(combined_text, encoding='utf-8')
        
        return ExtractedTextResult(
            paper_id=paper_id,
            extracted_text=combined_text,
            extraction_quality=extraction_quality,
            warnings=all_warnings,
            page_images=page_images,
            extraction_prompt=f"Multi-batch extraction ({num_batches} batches)",
            llm_response=f"Processed {len(page_images)} pages in {num_batches} batches",
            batch_order=batch_order
        )
    
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
        # Generate extraction prompt for this batch
        batch_info = f" (Batch {batch_id + 1}, pages {page_indices[0] + 1}-{page_indices[-1] + 1})"
        prompt = self.generate_extraction_prompt(page_images, batch_info)
        
        # Call LLM for text extraction
        try:
            response = self._call_model(prompt, page_images, max_tokens=8000)
            extracted_text = self.parse_extraction_response(response)
        except Exception as e:
            logger.error(f"Error calling LLM for batch {batch_id} text extraction: {e}")
            extracted_text = f"Error extracting text from batch {batch_id}: {e}"
            response = f"Error: {e}"
        
        # Assess extraction quality for this batch
        extraction_quality = self.assess_extraction_quality(extracted_text)
        
        # Generate warnings for this batch
        warnings = self._generate_warnings(extracted_text, page_images)
        
        return BatchExtractionResult(
            batch_id=batch_id,
            page_indices=page_indices,
            extracted_text=extracted_text,
            extraction_quality=extraction_quality,
            warnings=warnings,
            extraction_prompt=prompt,
            llm_response=response
        )
    
    def _combine_batch_results(
        self,
        batch_results: List[BatchExtractionResult],
        all_page_images: List[Path]
    ) -> str:
        """
        Combine batch results in the correct page order.
        
        Args:
            batch_results: List of batch extraction results
            all_page_images: All page images for reference
            
        Returns:
            Combined text in correct page order
        """
        # Sort batch results by the first page index in each batch
        sorted_batches = sorted(batch_results, key=lambda x: x.page_indices[0])
        
        combined_text = ""
        
        for i, batch_result in enumerate(sorted_batches):
            if i > 0:
                # Add a page separator between batches
                combined_text += "\n\n" + "="*50 + f" PAGE {batch_result.page_indices[0] + 1} " + "="*50 + "\n\n"
            
            combined_text += batch_result.extracted_text
        
        return combined_text
    
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
        prompt = f"""Extract all text from these PDF pages{batch_info}.

Requirements:
1. Extract ALL text content, including headers, footers, and body text
2. Preserve proper spacing and line breaks
3. Maintain scientific accuracy (formulas, chemical names, etc.)
4. Keep the original structure and flow
5. Remove any obvious noise or artifacts
6. Format as clean, readable text
7. Preserve special characters, superscripts, and subscripts
8. Maintain paragraph structure and formatting
9. If this is a batch of pages, maintain the natural flow between pages

Return the extracted text in a clean, readable format.

Number of page images in this batch: {len(page_images)}
"""
        
        return prompt
    
    def parse_extraction_response(self, response: str) -> str:
        """
        Parse the LLM response to extract the text content.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Extracted text as string
        """
        # Remove any markdown formatting if present
        text = response.strip()
        
        # Remove code block markers if present
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        
        # Remove language identifier if present
        if text.startswith("plaintext"):
            text = text[9:].strip()
        
        return text
    
    def assess_extraction_quality(self, extracted_text: str) -> str:
        """
        Assess the quality of extracted text.
        
        Args:
            extracted_text: The extracted text to assess
            
        Returns:
            Quality assessment string
        """
        if not extracted_text or extracted_text.strip() == "":
            return "poor - no text extracted"
        
        if len(extracted_text.strip()) < 100:
            return "poor - very little text"
        
        if len(extracted_text.strip()) < 500:
            return "fair - limited text content"
        
        # Check for common extraction issues
        issues = []
        if "Error extracting text" in extracted_text:
            issues.append("extraction error")
        
        if len(extracted_text.split('\n')) < 5:
            issues.append("limited structure")
        
        if issues:
            return f"fair - issues: {', '.join(issues)}"
        
        return "good - substantial text extracted"
    
    def _generate_warnings(self, extracted_text: str, page_images: List[Path]) -> List[str]:
        """
        Generate warnings about the extraction process.
        
        Args:
            extracted_text: The extracted text
            page_images: List of page images
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if not extracted_text or extracted_text.strip() == "":
            warnings.append("No text was extracted from the images")
        
        if len(extracted_text.strip()) < 100:
            warnings.append("Very little text was extracted - possible extraction issue")
        
        if "Error extracting text" in extracted_text:
            warnings.append("LLM extraction encountered an error")
        
        if len(page_images) > 1 and len(extracted_text.split('\n')) < 10:
            warnings.append("Limited text structure for multi-page document")
        
        return warnings 