#!/usr/bin/env python3
"""Test script for validation function."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.simple_pipeline import validate_extraction_completeness
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_validation():
    """Test the validation function on the bioe file."""
    pdf_path = Path("inputs/raw_papers/bioe.2024.0006.pdf")
    consensus_file = Path("outputs/extracted_texts/bioe.2024.0006/consensus.txt")
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    if not consensus_file.exists():
        logger.error(f"Consensus file not found: {consensus_file}")
        return
    
    # Read consensus text
    with open(consensus_file, 'r', encoding='utf-8') as f:
        consensus_text = f.read()
    
    logger.info(f"Testing validation on {pdf_path.name}")
    logger.info(f"Consensus text length: {len(consensus_text)} characters")
    
    # Run validation
    result = validate_extraction_completeness(pdf_path, consensus_text)
    
    logger.info(f"Validation result: {result}")

if __name__ == "__main__":
    test_validation() 