#!/usr/bin/env python3
"""
Test script to process only bioe.2024.0006.pdf
"""

import sys
from pathlib import Path
sys.path.append(str(Path('.')))

from scripts.simple_pipeline import process_semantic_chunking, load_metadata, metadata_lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Process only bioe.2024.0006.pdf"""
    logger.info("Testing semantic chunking for bioe.2024.0006.pdf")
    
    # Load metadata
    metadata = load_metadata()
    
    # Find the PDF file
    pdf_path = Path("inputs/raw_papers/bioe.2024.0006.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    # Process semantic chunking
    success = process_semantic_chunking(pdf_path, metadata)
    
    if success:
        logger.info("✅ Semantic chunking completed successfully")
        
        # Check if metadata was updated
        paper_name = pdf_path.name
        if paper_name in metadata["papers"]:
            steps = metadata["papers"][paper_name].get("steps", {})
            if "semantic_chunking" in steps:
                logger.info("✅ Semantic chunking step found in metadata!")
                logger.info(f"Number of chunks: {steps['semantic_chunking'].get('num_chunks', 'unknown')}")
            else:
                logger.error("❌ Semantic chunking step not found in metadata")
        else:
            logger.error("❌ Paper not found in metadata")
    else:
        logger.error("❌ Semantic chunking failed")

if __name__ == "__main__":
    main() 