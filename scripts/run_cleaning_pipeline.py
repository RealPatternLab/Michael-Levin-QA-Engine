#!/usr/bin/env python3
"""
Multimodal PDF Text Extraction Pipeline

This script implements a complete pipeline for extracting text from academic PDFs
using multimodal LLMs to analyze PDF page images.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from modules.pdf_processing import get_pdf_processor
from modules.text_extraction import get_extraction_processor
from modules.storage import get_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(output_dir: Path) -> tuple[Path, Path, Path]:
    """
    Set up output directories for the extraction pipeline.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Tuple of (page_images_dir, extracted_texts_dir, results_dir)
    """
    page_images_dir = output_dir / "page_images"
    extracted_texts_dir = output_dir / "extracted_texts"
    results_dir = output_dir / "results"
    
    # Create directories
    for dir_path in [page_images_dir, extracted_texts_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return page_images_dir, extracted_texts_dir, results_dir

def run_extraction_pipeline(
    pdf_path: Path,
    output_dir: Path,
    pdf_processor_name: str = "pymupdf",
    llm_processor_name: str = "openai",
    storage_provider: str = "json",
    force_reprocess: bool = False,
    verbose: bool = False,
    max_images_per_batch: int = 5
) -> bool:
    """
    Run the complete multimodal text extraction pipeline.
    
    Args:
        pdf_path: Path to the PDF file to process
        output_dir: Directory to save results
        pdf_processor_name: Name of PDF processor to use
        llm_processor_name: Name of LLM processor to use
        storage_provider: Name of storage provider to use
        force_reprocess: Whether to reprocess even if already completed
        verbose: Whether to enable verbose logging
        
    Returns:
        True if pipeline completed successfully
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate paper ID from filename
    paper_id = pdf_path.stem
    
    logger.info(f"Starting multimodal text extraction pipeline for {paper_id}")
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"Output: {output_dir}")
    
    # Get processors
    pdf_processor = get_pdf_processor(pdf_processor_name)
    llm_processor = get_extraction_processor(llm_processor_name)
    storage = get_storage(storage_provider)
    
    # Check if already processed
    if not force_reprocess and storage.job_exists(paper_id):
        logger.info(f"Job {paper_id} already exists in storage. Use --force-reprocess to reprocess.")
        return True
    
    try:
        # Setup directories
        page_images_dir, extracted_texts_dir, results_dir = setup_directories(output_dir)
        
        # Step 1: Render PDF pages to images
        logger.info(f"Rendering PDF pages to images...")
        rendering_result = pdf_processor.render_pages_to_images(
            pdf_path=pdf_path,
            output_dir=page_images_dir / paper_id,
            image_format="png",
            dpi=150
        )
        
        logger.info(f"Rendered {rendering_result.total_pages} pages")
        
        # Step 2: Extract text from images using multimodal LLM
        logger.info(f"Extracting text from images using multimodal LLM...")
        page_images = [p.image_path for p in rendering_result.pages]
        
        extraction_result = llm_processor.extract_text_from_images(
            paper_id=paper_id,
            page_images=page_images,
            output_dir=extracted_texts_dir,
            max_images_per_batch=max_images_per_batch
        )
        
        logger.info(f"Text extraction completed. Quality: {extraction_result.extraction_quality}")
        
        # Step 3: Save results
        logger.info(f"Saving results...")
        
        # Save extraction result
        result_file = results_dir / f"{paper_id}_extraction_results.json"
        result_data = {
            "paper_id": extraction_result.paper_id,
            "extraction_quality": extraction_result.extraction_quality,
            "warnings": extraction_result.warnings,
            "page_images": [str(p) for p in extraction_result.page_images],
            "extraction_prompt": extraction_result.extraction_prompt,
            "llm_response": extraction_result.llm_response,
            "total_pages": rendering_result.total_pages,
            "pdf_processor": pdf_processor_name,
            "llm_processor": llm_processor_name
        }
        
        import json
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Save to storage
        storage.save_job(paper_id, result_data)
        
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Extracted text saved to: {extracted_texts_dir / f'{paper_id}_extracted.txt'}")
        logger.info(f"Results saved to: {result_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal PDF Text Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract text from a PDF using default settings
  python scripts/run_cleaning_pipeline.py --pdf data/papers/paper.pdf --output data/extracted

  # Use specific processors and force reprocessing
  python scripts/run_cleaning_pipeline.py --pdf data/papers/paper.pdf --output data/extracted \\
    --pdf-processor pymupdf --llm-processor openai --force-reprocess --verbose
        """
    )
    
    parser.add_argument(
        "--pdf", 
        type=Path, 
        required=True,
        help="Path to PDF file to process"
    )
    
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--pdf-processor",
        choices=["pymupdf", "pdfplumber"],
        default="pymupdf",
        help="PDF processor to use (default: pymupdf)"
    )
    
    parser.add_argument(
        "--llm-processor",
        choices=["openai"],
        default="openai",
        help="LLM processor to use (default: openai)"
    )
    
    parser.add_argument(
        "--storage-provider",
        choices=["json"],
        default="json",
        help="Storage provider to use (default: json)"
    )
    
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing even if job already exists"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=5,
        help="Maximum number of images to send per LLM call (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pdf.exists():
        logger.error(f"PDF file not found: {args.pdf}")
        return 1
    
    # Run pipeline
    success = run_extraction_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output,
        pdf_processor_name=args.pdf_processor,
        llm_processor_name=args.llm_processor,
        storage_provider=args.storage_provider,
        force_reprocess=args.force_reprocess,
        verbose=args.verbose,
        max_images_per_batch=args.max_batch_size
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 