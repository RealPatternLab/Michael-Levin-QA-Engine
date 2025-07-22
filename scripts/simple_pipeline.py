#!/usr/bin/env python3
"""
Simple PDF Processing Pipeline

This script does exactly what you need:
1. Check for new PDFs in inputs/raw_papers directory
2. Extract metadata using OpenAI (for renaming)
3. Rename PDFs with descriptive names
4. Extract text using Google Document AI
5. Save everything in outputs/ directory
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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

def get_pdf_files() -> list[Path]:
    """Get all PDF files from inputs/raw_papers directory."""
    if not RAW_PAPERS_DIR.exists():
        logger.error(f"Input directory not found: {RAW_PAPERS_DIR}")
        return []
    
    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files

def load_metadata() -> Dict[str, Any]:
    """Load existing metadata or create new."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {"papers": {}, "processed_files": []}

def save_metadata(metadata: Dict[str, Any]):
    """Save metadata to file."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
        return None

def extract_metadata_with_openai(text: str, filename: str) -> Optional[Dict[str, Any]]:
    """Extract metadata using OpenAI."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""
        Extract metadata from this academic paper. Return ONLY a JSON object with these fields:
        - title: The full paper title
        - authors: List of author names (first author first)
        - year: Publication year (4-digit year)
        - journal: Journal name (abbreviated)
        - is_levin_paper: Boolean indicating if Michael Levin is an author

        Paper filename: {filename}
        Paper text (first {MAX_TEXT_LENGTH} chars): {text[:MAX_TEXT_LENGTH]}

        Return ONLY the JSON object, no markdown formatting.
        """
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts metadata from academic papers. Return only raw JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=OPENAI_MAX_TOKENS
        )
        
        # Parse JSON response
        response_text = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            response_text = '\n'.join(lines).strip()
        
        metadata = json.loads(response_text)
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract metadata with OpenAI: {e}")
        return None

def extract_text_with_document_ai(pdf_path: Path) -> Optional[str]:
    """Extract text using Google Document AI."""
    try:
        from google.cloud import documentai_v1 as documentai
        
        if not GOOGLE_DOCUMENT_AI_PROJECT_ID or not GOOGLE_DOCUMENT_AI_PROCESSOR_ID:
            logger.error("Google Document AI credentials not configured")
            return None
        
        # Initialize client
        client = documentai.DocumentProcessorServiceClient()
        processor_name = f"projects/{GOOGLE_DOCUMENT_AI_PROJECT_ID}/locations/{GOOGLE_DOCUMENT_AI_LOCATION}/processors/{GOOGLE_DOCUMENT_AI_PROCESSOR_ID}"
        
        # Read PDF
        with open(pdf_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()
        
        # Create document
        document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Process document
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=document
        )
        
        result = client.process_document(request=request)
        return result.document.text
        
    except Exception as e:
        logger.error(f"Failed to extract text with Document AI: {e}")
        return None

def generate_filename(metadata: Dict[str, Any]) -> str:
    """Generate a clean filename from metadata."""
    title = metadata.get('title', 'Unknown')
    authors = metadata.get('authors', [])
    year = metadata.get('year', 'unknown')
    
    # Clean title
    import re
    clean_title = re.sub(r'[^\w\s-]', '', title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip())
    clean_title = clean_title[:50]
    
    # Get first author's last name
    first_author = authors[0] if authors else "Unknown"
    author_last_name = first_author.split()[-1] if first_author else "Unknown"
    
    return f"levin_{clean_title}_{year}_{author_last_name}.pdf"

def process_pdf(pdf_path: Path, metadata: Dict[str, Any]) -> bool:
    """Process a single PDF file."""
    try:
        logger.info(f"Processing: {pdf_path.name}")
        
        # Step 1: Extract text using Document AI
        logger.info("Extracting text with Google Document AI...")
        extracted_text = extract_text_with_document_ai(pdf_path)
        
        if not extracted_text:
            logger.error("Failed to extract text")
            return False
        
        # Step 2: Generate new filename
        new_filename = generate_filename(metadata)
        new_path = pdf_path.parent / new_filename
        
        # Step 3: Rename file
        pdf_path.rename(new_path)
        logger.info(f"Renamed to: {new_filename}")
        
        # Step 4: Save extracted text
        EXTRACTED_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
        
        text_file = EXTRACTED_TEXTS_DIR / f"{new_path.stem}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        logger.info(f"Saved text to: {text_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")
        return False

def main():
    """Main pipeline function."""
    logger.info("🚀 Starting simple PDF processing pipeline")
    
    # Get PDF files
    pdf_files = get_pdf_files()
    if not pdf_files:
        logger.info("No PDF files found")
        return
    
    # Load existing metadata
    metadata = load_metadata()
    processed_files = set(metadata.get("processed_files", []))
    
    # Process each PDF
    for pdf_path in pdf_files:
        if pdf_path.name in processed_files:
            logger.info(f"Skipping {pdf_path.name} - already processed")
            continue
        
        logger.info(f"Processing {pdf_path.name}")
        
        # Extract text for metadata extraction
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"Failed to extract text from {pdf_path.name}")
            continue
        
        # Extract metadata with OpenAI
        paper_metadata = extract_metadata_with_openai(text, pdf_path.name)
        if not paper_metadata:
            logger.error(f"Failed to extract metadata from {pdf_path.name}")
            continue
        
        # Process the PDF (rename and extract text)
        if process_pdf(pdf_path, paper_metadata):
            # Update metadata
            metadata["processed_files"].append(pdf_path.name)
            metadata["papers"][pdf_path.name] = {
                "metadata": paper_metadata,
                "processed_at": datetime.now().isoformat()
            }
            save_metadata(metadata)
            
            logger.info(f"✅ Successfully processed {pdf_path.name}")
        else:
            logger.error(f"❌ Failed to process {pdf_path.name}")
    
    logger.info("🎉 Pipeline completed!")

if __name__ == "__main__":
    main() 