#!/usr/bin/env python3
"""
Extract metadata from PDFs using LLM and update papers.json database.
This script does NOT rename files - that's handled by a separate script.
Uses async/await for concurrent LLM processing.
"""

import os
import json
import sys
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_TEXT_LENGTH, RAW_PAPERS_DIR
from pypdf import PdfReader
from openai import OpenAI, AsyncOpenAI

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

async def extract_metadata_with_llm_async(text, filename, client):
    """Use LLM to extract metadata from PDF text (async version)."""
    
    prompt = f"""
    Extract the following information from this academic paper text. Return ONLY a raw JSON object (no markdown formatting, no code blocks) with these fields:
    - title: The full paper title
    - authors: List of author names (first author first)
    - year: Publication year (4-digit year)
    - journal: Journal name (abbreviated)
    - topic: Primary research topic (choose from: bioelectric, self_boundary, morphogenetic, collective_intelligence, ml_hypothesis, regeneration, development, cancer, computation, neuroevolution, general)
    - is_levin_paper: Boolean indicating if Michael Levin is an author or if the paper is about his work

    Paper filename: {filename}
    
    Paper text:
    {text[:MAX_TEXT_LENGTH]}  # First {MAX_TEXT_LENGTH} characters for context
    
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
        # Check if API key is set
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-api-key-here':
            print("❌ OpenAI API key not configured")
            print("   Run: python scripts/test_api_key.py")
            return None
        
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts metadata from academic papers. Return only raw JSON without any markdown formatting or code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE
        )
        
        # Parse the JSON response
        metadata_text = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks (fallback)
        if metadata_text.startswith('```'):
            # Remove markdown code block markers
            lines = metadata_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]  # Remove opening ```
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]  # Remove closing ```
            metadata_text = '\n'.join(lines).strip()
        
        metadata = json.loads(metadata_text)
        return metadata
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response for {filename}: {e}")
        print(f"Raw response: {metadata_text}")
        return None
    except Exception as e:
        print(f"❌ Error with LLM extraction for {filename}: {e}")
        return None

def extract_metadata_with_llm(text, filename):
    """Use LLM to extract metadata from PDF text (sync version for backward compatibility)."""
    
    prompt = f"""
    Extract the following information from this academic paper text. Return ONLY a raw JSON object (no markdown formatting, no code blocks) with these fields:
    - title: The full paper title
    - authors: List of author names (first author first)
    - year: Publication year (4-digit year)
    - journal: Journal name (abbreviated)
    - topic: Primary research topic (choose from: bioelectric, self_boundary, morphogenetic, collective_intelligence, ml_hypothesis, regeneration, development, cancer, computation, neuroevolution, general)
    - is_levin_paper: Boolean indicating if Michael Levin is an author or if the paper is about his work

    Paper filename: {filename}
    
    Paper text:
    {text[:MAX_TEXT_LENGTH]}  # First {MAX_TEXT_LENGTH} characters for context
    
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
        # Check if API key is set
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-api-key-here':
            print("❌ OpenAI API key not configured")
            print("   Run: python scripts/test_api_key.py")
            return None
        
        # Create OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts metadata from academic papers. Return only raw JSON without any markdown formatting or code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE
        )
        
        # Parse the JSON response
        metadata_text = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks (fallback)
        if metadata_text.startswith('```'):
            # Remove markdown code block markers
            lines = metadata_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]  # Remove opening ```
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]  # Remove closing ```
            metadata_text = '\n'.join(lines).strip()
        
        metadata = json.loads(metadata_text)
        return metadata
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response: {e}")
        print(f"Raw response: {metadata_text}")
        return None
    except Exception as e:
        print(f"❌ Error with LLM extraction: {e}")
        return None

def load_paper_database():
    """Load existing paper database or create new one."""
    db_path = Path("data/papers.json")
    
    if db_path.exists():
        try:
            with open(db_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("⚠️  Existing papers.json is corrupted or empty, creating new database")
    
    # Create new database structure
    return {
        "metadata": {
            "created_date": "2024-01-15T00:00:00",
            "version": "1.0",
            "description": "Paper database for Michael Levin QA Engine"
        },
        "papers": []
    }

def save_paper_database(papers_db):
    """Save paper database to JSON file."""
    db_path = Path("data/papers.json")
    db_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(db_path, 'w') as f:
            json.dump(papers_db, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving database: {e}")
        return False

def generate_paper_id(metadata):
    """Generate a unique ID for the paper."""
    title = metadata.get('title', 'Unknown')
    year = metadata.get('year', 'unknown')
    
    # Create ID from title and year
    title_words = re.findall(r'\b\w+\b', title.lower())
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    key_words = [word for word in title_words if word not in common_words and len(word) > 3]
    identifier = '_'.join(key_words[:2])[:10]
    
    return f"{identifier}_{year}"

async def process_single_pdf(pdf_file, papers_db, client, semaphore):
    """Process a single PDF file asynchronously."""
    async with semaphore:  # Limit concurrent requests
        print(f"\n📄 Processing: {pdf_file.name}")
        
        # Check if already in database
        existing_paper = None
        for paper in papers_db['papers']:
            if paper['filename'] == pdf_file.name:
                existing_paper = paper
                break
        
        if existing_paper:
            print(f"   ⚠️  Already in database, skipping: {pdf_file.name}")
            return None
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        if not text:
            print(f"   ❌ Failed to extract text from {pdf_file.name}")
            return None
        
        # Extract metadata with LLM
        metadata = await extract_metadata_with_llm_async(text, pdf_file.name, client)
        if not metadata:
            print(f"   ❌ Failed to extract metadata from {pdf_file.name}")
            return None
        
        # Create paper record
        paper_record = {
            "id": generate_paper_id(metadata),
            "title": metadata.get('title', 'Unknown Title'),
            "authors": metadata.get('authors', []),
            "year": metadata.get('year', 'unknown'),
            "journal": metadata.get('journal', 'unknown'),
            "topic": metadata.get('topic', 'general'),
            "is_levin_paper": metadata.get('is_levin_paper', False),
            "filename": pdf_file.name,
            "download_url": "",  # Will be filled manually
            "source": "llm_extraction",
            "download_date": "2024-01-15",
            "file_size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 1),
            "levin_contribution": "lead_author" if metadata.get('is_levin_paper', False) else "unknown",
            "doi": ""  # Will be filled manually
        }
        
        # Add to database immediately
        papers_db['papers'].append(paper_record)
        
        # Save database after each successful extraction
        try:
            save_paper_database(papers_db)
            print(f"   ✅ Saved to database: {paper_record['id']}")
        except Exception as e:
            print(f"   ❌ Failed to save database: {e}")
            # Remove the paper record if save failed
            papers_db['papers'].pop()
            return None
        
        print(f"   Title: {metadata.get('title', 'Unknown')}")
        print(f"   Authors: {', '.join(metadata.get('authors', []))}")
        print(f"   Year: {metadata.get('year', 'Unknown')}")
        print(f"   Journal: {metadata.get('journal', 'Unknown')}")
        print(f"   Topic: {metadata.get('topic', 'Unknown')}")
        print(f"   Is Levin paper: {metadata.get('is_levin_paper', False)}")
        
        return paper_record

async def extract_and_update_database_async():
    """Extract metadata from PDFs and update database using async processing."""
    
    if not RAW_PAPERS_DIR.exists():
        print("❌ data/raw_papers directory not found")
        return
    
    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in data/raw_papers")
        return
    
    # Load existing database
    papers_db = load_paper_database()
    
    print(f"🔍 Extracting metadata from {len(pdf_files)} PDF files (async processing):")
    
    # Check if API key is set
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-api-key-here':
        print("❌ OpenAI API key not configured")
        print("   Run: python scripts/test_api_key.py")
        return
    
    # Create async OpenAI client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Create semaphore to limit concurrent requests (avoid rate limits)
    semaphore = asyncio.Semaphore(3)  # Process 3 PDFs concurrently
    
    # Process PDFs concurrently
    tasks = []
    for pdf_file in pdf_files:
        task = process_single_pdf(pdf_file, papers_db, client, semaphore)
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful extractions
    successful_extractions = [r for r in results if r is not None and not isinstance(r, Exception)]
    
    if successful_extractions:
        print(f"\n📊 Successfully processed {len(successful_extractions)} new papers")
        print(f"💾 Database saved after each extraction - no work lost!")
        print(f"⚡ Async processing completed!")
    else:
        print("\n✅ No new papers to add to database")

def extract_and_update_database():
    """Extract metadata from PDFs and update database (sync version for backward compatibility)."""
    
    if not RAW_PAPERS_DIR.exists():
        print("❌ data/raw_papers directory not found")
        return
    
    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in data/raw_papers")
        return
    
    # Load existing database
    papers_db = load_paper_database()
    existing_filenames = {paper['filename'] for paper in papers_db['papers']}
    
    print("🔍 Extracting metadata from PDF files:")
    
    new_papers = []
    updated_papers = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 Processing ({i}/{len(pdf_files)}): {pdf_file.name}")
        
        # Check if already in database
        existing_paper = None
        for paper in papers_db['papers']:
            if paper['filename'] == pdf_file.name:
                existing_paper = paper
                break
        
        if existing_paper:
            print(f"   ⚠️  Already in database, skipping: {pdf_file.name}")
            continue
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        if not text:
            print(f"   ❌ Failed to extract text from {pdf_file.name}")
            continue
        
        # Extract metadata with LLM
        metadata = extract_metadata_with_llm(text, pdf_file.name)
        if not metadata:
            print(f"   ❌ Failed to extract metadata from {pdf_file.name}")
            continue
        
        # Create paper record
        paper_record = {
            "id": generate_paper_id(metadata),
            "title": metadata.get('title', 'Unknown Title'),
            "authors": metadata.get('authors', []),
            "year": metadata.get('year', 'unknown'),
            "journal": metadata.get('journal', 'unknown'),
            "topic": metadata.get('topic', 'general'),
            "is_levin_paper": metadata.get('is_levin_paper', False),
            "filename": pdf_file.name,
            "download_url": "",  # Will be filled manually
            "source": "llm_extraction",
            "download_date": "2024-01-15",
            "file_size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 1),
            "levin_contribution": "lead_author" if metadata.get('is_levin_paper', False) else "unknown",
            "doi": ""  # Will be filled manually
        }
        
        # Add to database immediately
        papers_db['papers'].append(paper_record)
        new_papers.append(paper_record)
        
        # Save database after each successful extraction
        try:
            save_paper_database(papers_db)
            print(f"   ✅ Saved to database: {paper_record['id']}")
        except Exception as e:
            print(f"   ❌ Failed to save database: {e}")
            # Remove the paper record if save failed
            papers_db['papers'].pop()
            new_papers.pop()
            continue
        
        print(f"   Title: {metadata.get('title', 'Unknown')}")
        print(f"   Authors: {', '.join(metadata.get('authors', []))}")
        print(f"   Year: {metadata.get('year', 'Unknown')}")
        print(f"   Journal: {metadata.get('journal', 'Unknown')}")
        print(f"   Topic: {metadata.get('topic', 'Unknown')}")
        print(f"   Is Levin paper: {metadata.get('is_levin_paper', False)}")
    
    if new_papers:
        print(f"\n📊 Successfully processed {len(new_papers)} new papers")
        print(f"💾 Database saved after each extraction - no work lost!")
    else:
        print("\n✅ No new papers to add to database")

async def main_async():
    """Main async function to extract metadata and update database."""
    await extract_and_update_database_async()

def main():
    """Main function to extract metadata and update database (uses async version)."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 