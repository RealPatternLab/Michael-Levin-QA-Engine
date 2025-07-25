#!/usr/bin/env python3
"""
Paper Processing Pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.processors.text_extraction import TextExtractor
from scripts.processors.semantic_chunking import SemanticChunker
from scripts.processors.embedding_generation import EmbeddingGenerator
from scripts.utils.file_utils import setup_directories, get_pdf_files
from scripts.utils.metadata_utils import load_metadata, save_metadata
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main paper processing pipeline."""
    logger.info("🚀 Starting Paper Processing Pipeline")
    
    # Setup directories
    setup_directories()
    
    # Get PDF files
    pdf_files = get_pdf_files("inputs/papers")
    
    if not pdf_files:
        logger.warning("No PDF files found in inputs/papers/")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Load metadata
    metadata = load_metadata("outputs/papers/metadata.json")
    
    # Initialize processors
    text_extractor = TextExtractor()
    semantic_chunker = SemanticChunker()
    embedding_generator = EmbeddingGenerator()
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Step 1: Text extraction
            if not text_extractor.is_processed(pdf_file, metadata):
                logger.info(f"Step 1: Extracting text from {pdf_file.name}")
                text_extractor.process(pdf_file, metadata)
            
            # Step 2: Semantic chunking
            if not semantic_chunker.is_processed(pdf_file, metadata):
                logger.info(f"Step 2: Creating semantic chunks for {pdf_file.name}")
                semantic_chunker.process(pdf_file, metadata)
            
            # Step 3: Embedding generation
            if not embedding_generator.is_processed(pdf_file, metadata):
                logger.info(f"Step 3: Generating embeddings for {pdf_file.name}")
                embedding_generator.process(pdf_file, metadata)
            
            logger.info(f"✅ Completed processing: {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
            continue
    
    # Step 4: Build combined index
    logger.info("Step 4: Building combined FAISS index")
    embedding_generator.build_combined_index()
    
    logger.info("🎉 Paper processing pipeline completed!")

if __name__ == "__main__":
    main() 