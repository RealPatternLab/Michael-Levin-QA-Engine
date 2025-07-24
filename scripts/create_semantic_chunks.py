#!/usr/bin/env python3
"""
Create Semantic Chunks for Vector Database

This script analyzes consensus.txt files and extracts semantically meaningful chunks
in JSON format for embedding into a vector database.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.settings import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_metadata() -> Dict[str, Any]:
    """Load existing metadata."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {"papers": {}}

def create_semantic_chunks_prompt() -> str:
    """Create the prompt for Gemini to extract semantic chunks."""
    return """
    Analyze this scientific paper text and extract semantically meaningful chunks for vector database embedding. 

    For each chunk, identify:
    1. A coherent semantic topic or idea
    2. The section where it appears
    3. The page estimate (from section headers with page numbers)
    4. Character position information
    5. The source paper information

    Return ONLY a JSON array of objects with this exact structure:
    [
      {
        "text": "The actual text content of the semantic chunk",
        "source_title": "Full paper title",
        "year": 2023,
        "section_header": "Introduction/Discussion/Results/etc",
        "semantic_topic": "Brief description of the main idea/topic",
        "page_estimate": 5,
        "start_char": 10230,
        "end_char": 10850
      }
    ]

    Guidelines for chunking:
    - Each chunk should represent a complete idea or concept
    - Chunks should be semantically coherent and self-contained
    - Include relevant context but avoid redundancy
    - Respect natural section boundaries when possible
    - Aim for chunks that are meaningful for search and retrieval
    - Include page estimates from section headers like "[PAGE 3] INTRODUCTION"
    - Track character positions for precise location reference

    Paper text to analyze:
    """

def extract_semantic_chunks(consensus_text: str, paper_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract semantic chunks from consensus text using Gemini."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare the prompt with paper metadata
        prompt = create_semantic_chunks_prompt()
        prompt += f"""
        Paper Title: {paper_metadata.get('title', 'Unknown')}
        Authors: {', '.join(paper_metadata.get('authors', []))}
        Year: {paper_metadata.get('year', 'Unknown')}
        Journal: {paper_metadata.get('journal', 'Unknown')}
        
        Consensus Text:
        {consensus_text}
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                # Parse JSON response
                response_text = response.text.strip()
                
                # Handle markdown code blocks
                if response_text.startswith('```'):
                    lines = response_text.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]
                    if lines and lines[-1].startswith('```'):
                        lines = lines[:-1]
                    response_text = '\n'.join(lines).strip()
                
                chunks = json.loads(response_text)
                
                # Validate chunks structure
                if not isinstance(chunks, list):
                    logger.error("Response is not a list")
                    return []
                
                # Validate each chunk has required fields
                valid_chunks = []
                for chunk in chunks:
                    required_fields = ["text", "source_title", "year", "section_header", "semantic_topic", "page_estimate", "start_char", "end_char"]
                    if all(field in chunk for field in required_fields):
                        valid_chunks.append(chunk)
                    else:
                        logger.warning(f"Skipping chunk with missing fields: {chunk}")
                
                logger.info(f"✅ Successfully extracted {len(valid_chunks)} semantic chunks")
                return valid_chunks
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response.text[:500]}...")
                return []
        else:
            logger.error("Empty response from Gemini")
            return []
        
    except Exception as e:
        logger.error(f"Failed to extract semantic chunks: {e}")
        return []

def process_paper_chunks(pdf_name: str, metadata: Dict[str, Any]) -> bool:
    """Process semantic chunks for a single paper."""
    try:
        logger.info(f"Processing semantic chunks for: {pdf_name}")
        
        # Get paper metadata
        paper_metadata = metadata["papers"].get(pdf_name, {})
        steps = paper_metadata.get("steps", {})
        text_extraction = steps.get("text_extraction", {})
        
        if not text_extraction.get("completed", False):
            logger.warning(f"Skipping {pdf_name} - text extraction not completed")
            return False
        
        # Get consensus file path
        consensus_file = EXTRACTED_TEXTS_DIR / f"{Path(pdf_name).stem}.txt"
        if not consensus_file.exists():
            logger.warning(f"Skipping {pdf_name} - consensus file not found: {consensus_file}")
            return False
        
        # Read consensus text
        with open(consensus_file, 'r', encoding='utf-8') as f:
            consensus_text = f.read()
        
        if not consensus_text.strip():
            logger.warning(f"Skipping {pdf_name} - consensus file is empty")
            return False
        
        logger.info(f"Loaded consensus text: {len(consensus_text)} characters")
        
        # Extract semantic chunks
        chunks = extract_semantic_chunks(consensus_text, paper_metadata.get("metadata", {}))
        
        if not chunks:
            logger.error(f"Failed to extract chunks for {pdf_name}")
            return False
        
        # Save chunks to JSON file
        chunks_file = EXTRACTED_TEXTS_DIR / f"{Path(pdf_name).stem}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        logger.info(f"✅ Successfully saved {len(chunks)} chunks to {chunks_file}")
        
        # Update metadata
        metadata["papers"][pdf_name]["steps"]["semantic_chunks"] = {
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            "chunks_file": str(chunks_file),
            "num_chunks": len(chunks)
        }
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process chunks for {pdf_name}: {e}")
        return False

def save_metadata(metadata: Dict[str, Any]):
    """Save metadata to file."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main function to create semantic chunks for all papers."""
    logger.info("🚀 Starting semantic chunk creation for all papers")
    
    # Load metadata
    metadata = load_metadata()
    
    if not metadata.get("papers"):
        logger.info("No papers found in metadata")
        return
    
    # Get all papers with completed text extraction
    processed_papers = []
    for pdf_name, paper_data in metadata["papers"].items():
        steps = paper_data.get("steps", {})
        if steps.get("text_extraction", {}).get("completed", False):
            processed_papers.append(pdf_name)
    
    logger.info(f"Found {len(processed_papers)} papers with completed text extraction")
    
    if not processed_papers:
        logger.info("No papers with completed text extraction found")
        return
    
    # Process each paper
    successful = 0
    failed = 0
    
    for pdf_name in processed_papers:
        if process_paper_chunks(pdf_name, metadata):
            successful += 1
        else:
            failed += 1
    
    # Save updated metadata
    save_metadata(metadata)
    
    logger.info(f"🎉 Semantic chunk creation completed!")
    logger.info(f"✅ Successfully processed: {successful} papers")
    logger.info(f"❌ Failed to process: {failed} papers")

if __name__ == "__main__":
    main() 