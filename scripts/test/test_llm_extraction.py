#!/usr/bin/env python3
"""
Test script to verify LLM extraction works with one PDF.
This is for testing the LLM extraction before running the full metadata extraction.
"""

import os
import json
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import MAX_TEXT_LENGTH, DEFAULT_MODEL
from pypdf import PdfReader
from ai_models import get_model

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path.name}: {e}")
        return None

def test_llm_extraction(pdf_path):
    """Test LLM extraction on a single PDF."""
    
    print(f"🔍 Testing LLM extraction on: {pdf_path.name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("❌ Could not extract text from PDF")
        return
    
    print(f"📝 Extracted {len(text)} characters of text")
    
    # Create prompt
    prompt = f"""
    Extract the following information from this academic paper text. Return ONLY a raw JSON object (no markdown formatting, no code blocks) with these fields:
    - title: The full paper title
    - authors: List of author names (first author first)
    - year: Publication year (4-digit year)
    - journal: Journal name (abbreviated)
    - topic: Primary research topic (choose from: bioelectric, self_boundary, morphogenetic, collective_intelligence, ml_hypothesis, regeneration, development, cancer, computation, neuroevolution, general)
    - is_levin_paper: Boolean indicating if Michael Levin is an author or if the paper is about his work

    Paper filename: {pdf_path.name}
    
    Paper text (first {MAX_TEXT_LENGTH} characters):
    {text[:MAX_TEXT_LENGTH]}
    
    IMPORTANT: Return ONLY the raw JSON object, no markdown formatting, no ```json``` code blocks, just the JSON.
    
    Example response format:
    {{
        "title": "Example Paper Title",
        "authors": ["Author One", "Author Two"],
        "year": 2024,
        "journal": "Example Journal",
        "topic": "bioelectric",
        "is_levin_paper": true
    }}
    """
    
    try:
        # Get model using the centralized interface
        model = get_model(DEFAULT_MODEL)
        
        print(f"🤖 Sending to {DEFAULT_MODEL.upper()}...")
        
        # Call the model with system message
        response = model.call(
            prompt,
            system_message="You are a helpful assistant that extracts metadata from academic papers. Return only raw JSON without any markdown formatting or code blocks.",
            max_tokens=500
        )
        
        # Parse the JSON response
        metadata_text = response.strip()
        
        # Handle markdown code blocks (fallback)
        if metadata_text.startswith('```'):
            # Remove markdown code block markers
            lines = metadata_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]  # Remove opening ```
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]  # Remove closing ```
            metadata_text = '\n'.join(lines).strip()
        
        print(f"📄 Raw LLM response:")
        print(metadata_text)
        
        metadata = json.loads(metadata_text)
        
        print(f"\n✅ Successfully extracted metadata:")
        print(f"   Title: {metadata.get('title', 'Unknown')}")
        print(f"   Authors: {', '.join(metadata.get('authors', []))}")
        print(f"   Year: {metadata.get('year', 'Unknown')}")
        print(f"   Journal: {metadata.get('journal', 'Unknown')}")
        print(f"   Topic: {metadata.get('topic', 'Unknown')}")
        print(f"   Is Levin paper: {metadata.get('is_levin_paper', False)}")
        
        return metadata
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response: {e}")
        print(f"Raw response: {metadata_text}")
        return None
    except Exception as e:
        print(f"❌ Error with LLM extraction: {e}")
        return None

def main():
    """Test LLM extraction on the first PDF found."""
    from config import RAW_PAPERS_DIR
    
    if not RAW_PAPERS_DIR.exists():
        print("❌ data/raw_papers directory not found")
        return
    
    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in data/raw_papers")
        return
    
    # Test with the first PDF
    test_pdf = pdf_files[0]
    test_llm_extraction(test_pdf)
    
    print(f"\n💡 Next steps:")
    print(f"1. If the extraction looks good, run: python scripts/llm_extract_metadata.py")
    print(f"2. Then rename files: python scripts/rename_from_database.py")

if __name__ == "__main__":
    main() 