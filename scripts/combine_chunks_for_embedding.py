#!/usr/bin/env python3
"""
Combine all semantic chunks from all papers into a single dataset for embedding.
"""

import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_all_chunks():
    """Combine all semantic chunks from all papers into a single dataset."""
    
    # Find all semantic_chunks.json files
    extracted_dir = Path("outputs/extracted_texts")
    all_chunks = []
    
    if not extracted_dir.exists():
        logger.error("Extracted texts directory not found")
        return []
    
    # Scan all subdirectories for semantic_chunks.json files
    for paper_dir in extracted_dir.iterdir():
        if paper_dir.is_dir():
            chunks_file = paper_dir / "semantic_chunks.json"
            if chunks_file.exists():
                try:
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        paper_chunks = json.load(f)
                    
                    logger.info(f"Loaded {len(paper_chunks)} chunks from {paper_dir.name}")
                    all_chunks.extend(paper_chunks)
                    
                except Exception as e:
                    logger.error(f"Failed to load chunks from {chunks_file}: {e}")
    
    logger.info(f"Total chunks combined: {len(all_chunks)}")
    return all_chunks

def save_combined_chunks(chunks, output_file="combined_semantic_chunks.json"):
    """Save combined chunks to a single file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved {len(chunks)} combined chunks to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save combined chunks: {e}")
        return False

def main():
    """Main function to combine all chunks."""
    logger.info("🔗 Combining all semantic chunks from all papers...")
    
    # Combine all chunks
    all_chunks = combine_all_chunks()
    
    if not all_chunks:
        logger.error("No chunks found to combine")
        return
    
    # Save combined chunks
    success = save_combined_chunks(all_chunks)
    
    if success:
        logger.info("✅ Successfully combined all semantic chunks!")
        logger.info(f"📊 Total chunks available for embedding: {len(all_chunks)}")
        
        # Show some statistics
        papers = set(chunk.get('source_title', 'Unknown') for chunk in all_chunks)
        years = set(chunk.get('year', 'Unknown') for chunk in all_chunks)
        
        logger.info(f"📚 Papers represented: {len(papers)}")
        logger.info(f"📅 Years covered: {sorted(years)}")
        
        # Show a sample chunk
        if all_chunks:
            sample = all_chunks[0]
            logger.info(f"📝 Sample chunk:")
            logger.info(f"  Title: {sample.get('source_title', 'Unknown')}")
            logger.info(f"  Year: {sample.get('year', 'Unknown')}")
            logger.info(f"  Section: {sample.get('section_header', 'Unknown')}")
            logger.info(f"  Topic: {sample.get('semantic_topic', 'Unknown')}")
            logger.info(f"  Text length: {len(sample.get('text', ''))} characters")
    else:
        logger.error("❌ Failed to combine chunks")

if __name__ == "__main__":
    main() 