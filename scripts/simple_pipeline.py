#!/usr/bin/env python3
"""
Simple PDF Processing Pipeline

This script does exactly what you need:
1. Check for new PDFs in inputs/raw_papers directory
2. Extract metadata using OpenAI (for renaming)
3. Extract text using Gemini 1.5 Pro (direct PDF processing)
4. Save everything in outputs/ directory
5. Track processing steps in metadata
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

# Thread lock for metadata operations
metadata_lock = threading.Lock()

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
    with metadata_lock:
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {"papers": {}}

def save_metadata(metadata: Dict[str, Any]):
    """Save metadata to file."""
    with metadata_lock:
        METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using pypdf for metadata extraction."""
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

def extract_text_with_gemini(pdf_path: Path) -> Optional[str]:
    """Extract text directly from PDF using Gemini 1.5 Pro with multiple parallel passes for consensus extraction."""
    try:
        import google.generativeai as genai
        from pypdf import PdfReader, PdfWriter
        import io
        import time
        import asyncio
        import concurrent.futures
        from typing import List
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Read PDF to get page count
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        logger.info(f"PDF has {total_pages} pages, processing with Gemini 1.5 Pro using parallel multi-pass consensus extraction")
        
        # Process in 40-page chunks with 1-page overlap
        chunk_size = 40
        overlap = 1
        max_retries = 3
        num_passes = 5
        
        # Create subfolder for this PDF
        pdf_stem = pdf_path.stem
        pdf_output_dir = EXTRACTED_TEXTS_DIR / pdf_stem
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        def process_single_pass(pass_num: int) -> str:
            """Process a single extraction pass."""
            logger.info(f"Starting extraction pass {pass_num + 1}/{num_passes}")
            
            # Create a fresh reader for this pass to avoid thread safety issues
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            pass_text = []
            
            for chunk_start in range(0, total_pages, chunk_size - overlap):
                chunk_end = min(chunk_start + chunk_size, total_pages)
                chunk_num = len(pass_text) + 1
                total_chunks = (total_pages + chunk_size - overlap - 1) // (chunk_size - overlap)
                
                logger.info(f"Processing chunk {chunk_num}/{total_chunks}: pages {chunk_start + 1}-{chunk_end} (pass {pass_num + 1})")
                
                # Create a new PDF with just this chunk of pages
                writer = PdfWriter()
                
                for page_num in range(chunk_start, chunk_end):
                    writer.add_page(reader.pages[page_num])
                
                # Write chunk to memory
                chunk_pdf = io.BytesIO()
                writer.write(chunk_pdf)
                chunk_pdf.seek(0)
                pdf_content = chunk_pdf.getvalue()
                
                # Validate PDF chunk was created successfully
                if len(pdf_content) == 0:
                    logger.error(f"Failed to create PDF chunk for pages {chunk_start + 1}-{chunk_end}")
                    return None
                
                # Prepare prompt for Gemini
                prompt = f"""
                Extract all text content from this scientific PDF, from the very beginning to the very end, including any introductory material preceding the main title. Prioritize capturing all content that appears to be part of the scientific paper itself.

                Specifically:

                * **Include:** Absolutely all text from the beginning of the PDF to the end, including but not limited to the Introduction, Methods, Results, Discussion, Conclusion, and any other sections containing core scientific findings or arguments. This includes any content appearing *before* the main title or abstract. Preserve section headings and paragraph structure. Pay special attention to introductory sections and ensure no paragraphs are missed, regardless of their title (e.g., "WHAT DOES IT FEEL LIKE TO BE A PANCREAS?").

                * **Exclude:** Headers, footers, page numbers, and publication metadata (e.g., journal name, publication date). References/bibliography sections, figure captions, table captions, and the content of figures and tables themselves should also be excluded.

                * **Prioritize:** Complete capture of all textual content within the scientific paper, from start to finish. Accuracy and completeness of the entire text extraction are paramount. Do not cut off content mid-section.

                * **Do Not:** Summarize, paraphrase, add commentary, or include any explanatory text of your own. Only extract the verbatim text from the PDF.

                This is chunk {chunk_num} of {total_chunks} (pass {pass_num + 1} of {num_passes}) covering pages {chunk_start + 1}-{chunk_end} of {total_pages}.

                Return the extracted text as a single, continuous block of text.

                PDF content:
                """
                
                # Retry logic for each chunk
                chunk_text = None
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Attempt {attempt + 1}/{max_retries} for chunk {chunk_num} (pass {pass_num + 1})")
                        
                        # Send PDF chunk to Gemini
                        content_parts = [prompt]
                        content_parts.append({
                            "mime_type": "application/pdf",
                            "data": pdf_content
                        })
                        
                        response = model.generate_content(content_parts)
                        
                        if response.text and len(response.text.strip()) > 100:  # Minimum content validation
                            chunk_text = response.text.strip()
                            logger.info(f"✅ Successfully processed chunk {chunk_num}: pages {chunk_start + 1}-{chunk_end} ({len(chunk_text)} characters)")
                            break
                        else:
                            logger.warning(f"Attempt {attempt + 1}: Empty or too short response for chunk {chunk_num} ({len(response.text) if response.text else 0} characters)")
                            if attempt < max_retries - 1:
                                time.sleep(2)  # Wait before retry
                            else:
                                last_error = "Empty or insufficient response from Gemini"
                    
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1}: Error processing chunk {chunk_num}: {e}")
                        last_error = str(e)
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                
                # If all retries failed for this chunk, fail the entire process
                if chunk_text is None:
                    logger.error(f"❌ Failed to process chunk {chunk_num} after {max_retries} attempts. Last error: {last_error}")
                    logger.error(f"❌ Cannot proceed with incomplete text extraction. Failed at pages {chunk_start + 1}-{chunk_end}")
                    return None
                
                pass_text.append(chunk_text)
            
            # Combine chunks for this pass
            pass_combined = "\n\n".join(pass_text)
            
            # Save this pass to file
            pass_file = pdf_output_dir / f"pass_{pass_num + 1}.txt"
            with open(pass_file, 'w', encoding='utf-8') as f:
                f.write(pass_combined)
            
            logger.info(f"✅ Completed pass {pass_num + 1}: {len(pass_combined)} characters (saved to {pass_file})")
            return pass_combined
        
        # Run all passes in parallel
        logger.info(f"Starting {num_passes} parallel extraction passes...")
        with ThreadPoolExecutor(max_workers=num_passes) as executor:
            # Submit all passes
            future_to_pass = {executor.submit(process_single_pass, pass_num): pass_num for pass_num in range(num_passes)}
            
            # Collect results
            all_extractions = []
            failed_passes = []
            for future in as_completed(future_to_pass):
                pass_num = future_to_pass[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_extractions.append(result)
                        logger.info(f"✅ Pass {pass_num + 1} completed successfully")
                    else:
                        logger.error(f"❌ Pass {pass_num + 1} failed")
                        failed_passes.append(pass_num + 1)
                except Exception as e:
                    logger.error(f"❌ Pass {pass_num + 1} failed with exception: {e}")
                    failed_passes.append(pass_num + 1)
        
        # Check if we have enough successful passes to proceed
        if len(all_extractions) < 1:
            logger.error(f"❌ No passes completed successfully. Cannot proceed with extraction.")
            return None
        
        if len(all_extractions) < 2:
            logger.warning(f"⚠️ Only {len(all_extractions)} pass completed successfully. Using single pass extraction.")
            # For single pass, just return the content directly
            consensus_text = all_extractions[0]
        else:
            if failed_passes:
                logger.warning(f"⚠️ {len(failed_passes)} passes failed: {failed_passes}. Proceeding with {len(all_extractions)} successful passes.")
            
            # Create consensus extraction from successful passes
            logger.info(f"Creating consensus extraction from {len(all_extractions)} successful passes...")
            consensus_text = create_consensus_extraction_from_files(pdf_output_dir, len(all_extractions))
        
        # Save consensus result
        consensus_file = pdf_output_dir / "consensus.txt"
        with open(consensus_file, 'w', encoding='utf-8') as f:
            f.write(consensus_text)
        logger.info(f"✅ Consensus extraction saved to: {consensus_file}")
        
        # Final validation - ensure we have substantial content
        if len(consensus_text.strip()) < 1000:
            logger.error(f"❌ Final text is too short ({len(consensus_text)} characters). Extraction may have failed.")
            return None
        
        return consensus_text
        
    except Exception as e:
        logger.error(f"Failed to extract text with Gemini: {e}")
        return None

def create_consensus_extraction_from_files(pdf_output_dir: Path, num_passes: int) -> str:
    """Create a consensus extraction from saved pass files."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Read all pass files
        extractions = []
        for pass_num in range(num_passes):
            pass_file = pdf_output_dir / f"pass_{pass_num + 1}.txt"
            if pass_file.exists():
                with open(pass_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    extractions.append(content)
                    logger.info(f"Loaded pass {pass_num + 1}: {len(content)} characters")
            else:
                logger.warning(f"Pass file not found: {pass_file}")
                extractions.append("")  # Empty extraction for missing pass
        
        if not extractions or all(len(ext) == 0 for ext in extractions):
            logger.error("No valid extractions found")
            return ""
        
        # Prepare all extractions for comparison
        extractions_text = "\n\n".join([f"EXTRACTION {i+1}:\n{ext}" for i, ext in enumerate(extractions)])
        
        prompt = f"""
        Consolidate the following {num_passes} text extractions from a scientific PDF into a single, high-quality extraction. Each extraction is delimited by ```. The final output must contain the full, combined text of the extractions, prioritizing completeness, accuracy, and a coherent structure. Include all content present in any of the extractions. Remove redundant information while preserving unique details. If discrepancies exist, choose the most complete and accurate version. Maintain the original document's logical flow, including section headers, figures, tables, and references. Do not include any commentary or descriptions about your selection process or the source of the content. Output only the combined, consolidated text of the scientific PDF.

        ```
        {extractions_text}
        ```
        """
        
        response = model.generate_content(prompt)
        
        if response.text and len(response.text.strip()) > max(len(ext) for ext in extractions) * 0.3:
            logger.info(f"✅ Successfully created consensus extraction: {len(response.text)} characters")
            return response.text.strip()
        else:
            logger.warning("Consensus creation failed, returning longest extraction")
            return max(extractions, key=len)
        
    except Exception as e:
        logger.error(f"Failed to create consensus extraction: {e}")
        return max(extractions, key=len) if extractions else ""

def remove_overlapping_content(text: str, num_chunks: int) -> str:
    """Remove overlapping content from combined text using Gemini."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        I have extracted text from a scientific paper using {num_chunks} overlapping chunks. 
        The chunks overlap by 1 page to ensure no content is missed.
        
        Your task is to read through the entire text and remove any duplicate/overlapping content.
        Look for:
        1. Repeated paragraphs or sentences
        2. Duplicate section headers
        3. Overlapping text at chunk boundaries
        4. Any other repeated content
        
        IMPORTANT:
        - Keep all unique content
        - Remove only the duplicate/overlapping parts
        - Maintain the logical flow and structure
        - Preserve all section headers (but remove duplicates)
        - Keep the text coherent and readable
        
        Return the cleaned text with overlapping content removed.
        
        Text to process:
        {text}
        """
        
        response = model.generate_content(prompt)
        
        if response.text and len(response.text.strip()) > len(text) * 0.5:  # Should be at least 50% of original
            logger.info(f"✅ Successfully removed overlapping content: {len(response.text)} characters")
            return response.text.strip()
        else:
            logger.warning("Post-processing failed, returning original text")
            return text
        
    except Exception as e:
        logger.error(f"Failed to remove overlapping content: {e}")
        return text

def generate_descriptive_name(metadata: Dict[str, Any]) -> str:
    """Generate a descriptive name from metadata (for tracking, not renaming files)."""
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
    
    return f"levin_{clean_title}_{year}_{author_last_name}"

def process_metadata_extraction(pdf_path: Path, metadata: Dict[str, Any]) -> bool:
    """Process metadata extraction step."""
    try:
        logger.info(f"Processing metadata extraction for: {pdf_path.name}")
        
        # Extract text for metadata extraction
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"Failed to extract text from {pdf_path.name}")
            return False
        
        logger.info(f"✅ Successfully extracted text from {pdf_path.name} ({len(text)} characters)")
        
        # Extract metadata with OpenAI
        paper_metadata = extract_metadata_with_openai(text, pdf_path.name)
        if not paper_metadata:
            logger.error(f"Failed to extract metadata from {pdf_path.name}")
            return False
        
        logger.info(f"✅ Successfully extracted metadata: {paper_metadata.get('title', 'Unknown title')}")
        
        # Generate descriptive name
        descriptive_name = generate_descriptive_name(paper_metadata)
        
        # Update metadata
        with metadata_lock:
            metadata["papers"][pdf_path.name] = {
                "original_filename": pdf_path.name,
                "descriptive_name": descriptive_name,
                "metadata": paper_metadata,
                "steps": {
                    "metadata_extraction": {
                        "completed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        
        save_metadata(metadata)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process metadata extraction for {pdf_path.name}: {e}")
        return False

def process_text_extraction(pdf_path: Path, metadata: Dict[str, Any]) -> bool:
    """Process text extraction step."""
    try:
        logger.info(f"Processing text extraction for: {pdf_path.name}")
        
        # Extract text using Gemini 1.5 Pro
        logger.info("Extracting text with Gemini 1.5 Pro...")
        extracted_text = extract_text_with_gemini(pdf_path)
        
        if not extracted_text:
            logger.error("Failed to extract text")
            return False
        
        # Save extracted text (using original filename) in main directory
        EXTRACTED_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
        text_file = EXTRACTED_TEXTS_DIR / f"{pdf_path.stem}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        logger.info(f"Saved extracted text to: {text_file}")
        
        # Update metadata
        with metadata_lock:
            if pdf_path.name not in metadata["papers"]:
                metadata["papers"][pdf_path.name] = {
                    "original_filename": pdf_path.name,
                    "steps": {}
                }
        
        # Validate extraction completeness
        logger.info("Validating extraction completeness...")
        validation_result = validate_extraction_completeness(pdf_path, extracted_text)
        
        with metadata_lock:
            metadata["papers"][pdf_path.name]["steps"]["text_extraction"] = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "text_file": str(text_file),
                "detailed_passes_dir": str(EXTRACTED_TEXTS_DIR / pdf_path.stem),
                "validation": validation_result
            }
        
        save_metadata(metadata)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process text extraction for {pdf_path.name}: {e}")
        return False

def validate_extraction_completeness(pdf_path: Path, consensus_text: str) -> Dict[str, Any]:
    """Validate that the consensus text faithfully represents the original PDF content."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Extract text from PDF for comparison
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            logger.error(f"Failed to extract text from PDF for validation: {pdf_path}")
            return {"faithfully_transcribed": False, "missing_segments": ["Failed to extract PDF text for comparison"]}
        
        # Prepare prompt for validation
        prompt = f"""
        Compare the provided PDF with the consensus text extraction and determine if any important content from the PDF is missing from the consensus text. Focus on scientific content such as section headers, key findings, conclusions, data descriptions, results, and substantial paragraphs. Ignore minor formatting differences, page numbers, headers, and footers.

        IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:

        If all content is present:
        {{
          "faithfully_transcribed": True,
          "missing_segments": []
        }}

        If content is missing:
        {{
          "faithfully_transcribed": False,
          "missing_segments": [
            "The section 'WHAT DOES IT FEEL LIKE TO BE A PANCREAS?' appears in the PDF but is missing from the consensus text.",
            "The discussion of voltage-gated ion channels in section 3.2 appears to be incomplete in the consensus."
          ]
        }}

        Where:
        - "faithfully_transcribed": Boolean (True/False) indicating whether the consensus text appears to contain all important content from the PDF
        - "missing_segments": Array of strings. If faithfully_transcribed is True, this should be an empty array []. If faithfully_transcribed is False, this should contain specific descriptions of missing content.

        PDF content:
        {pdf_text}

        Consensus text:
        {consensus_text}

        Return ONLY the JSON object, no other text or explanations.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            logger.info(f"Validation response: {response.text[:200]}...")  # Debug log
            try:
                # Parse JSON response
                import json
                response_text = response.text.strip()
                
                # Handle markdown code blocks
                if response_text.startswith('```'):
                    lines = response_text.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]
                    if lines and lines[-1].startswith('```'):
                        lines = lines[:-1]
                    response_text = '\n'.join(lines).strip()
                
                validation_result = json.loads(response_text)
                
                if validation_result.get("faithfully_transcribed", False):
                    logger.info("✅ Validation passed: Consensus text appears to faithfully represent the PDF")
                else:
                    missing_segments = validation_result.get("missing_segments", [])
                    logger.warning(f"⚠️ Validation failed: Found {len(missing_segments)} missing segments")
                    for segment in missing_segments:
                        logger.warning(f"  - {segment}")
                
                return validation_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse validation JSON response: {e}")
                return {"faithfully_transcribed": False, "missing_segments": ["Failed to parse validation response"]}
        else:
            logger.error("Empty response from validation")
            return {"faithfully_transcribed": False, "missing_segments": ["Empty validation response"]}
        
    except Exception as e:
        logger.error(f"Failed to validate extraction completeness: {e}")
        return {"faithfully_transcribed": False, "missing_segments": [f"Validation error: {str(e)}"]}

def reset_file_processing(pdf_filename: str):
    """Reset processing state for a specific file to allow reprocessing."""
    metadata = load_metadata()
    
    with metadata_lock:
        if pdf_filename in metadata["papers"]:
            # Remove the file from metadata to allow reprocessing
            del metadata["papers"][pdf_filename]
            save_metadata(metadata)
            logger.info(f"Reset processing state for {pdf_filename}")
        else:
            logger.info(f"File {pdf_filename} not found in metadata")

def process_pdf_file(pdf_path: Path, metadata: Dict[str, Any]) -> bool:
    """Process a single PDF file - both metadata and text extraction."""
    logger.info(f"Processing {pdf_path.name}")
    
    # Check if file exists in metadata
    file_metadata = metadata["papers"].get(pdf_path.name, {})
    steps = file_metadata.get("steps", {})
    
    # Step 1: Metadata extraction
    if not steps.get("metadata_extraction", {}).get("completed", False):
        logger.info(f"Step 1: Extracting metadata for {pdf_path.name}")
        if process_metadata_extraction(pdf_path, metadata):
            logger.info(f"✅ Completed metadata extraction for {pdf_path.name}")
        else:
            logger.error(f"❌ Failed metadata extraction for {pdf_path.name}")
            return False
    else:
        logger.info(f"Skipping metadata extraction for {pdf_path.name} - already completed")
    
    # Step 2: Text extraction
    if not steps.get("text_extraction", {}).get("completed", False):
        logger.info(f"Step 2: Extracting text for {pdf_path.name}")
        if process_text_extraction(pdf_path, metadata):
            logger.info(f"✅ Completed text extraction for {pdf_path.name}")
        else:
            logger.error(f"❌ Failed text extraction for {pdf_path.name}")
            return False
    else:
        logger.info(f"Skipping text extraction for {pdf_path.name} - already completed")
    
    return True

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
    
    # Process all PDFs in parallel
    logger.info(f"Processing {len(pdf_files)} PDF files in parallel...")
    with ThreadPoolExecutor(max_workers=min(len(pdf_files), 5)) as executor: # Limit concurrent workers
        future_to_pdf = {executor.submit(process_pdf_file, pdf_path, metadata): pdf_path for pdf_path in pdf_files}
        
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"✅ Completed processing for {pdf_path.name}")
                else:
                    logger.error(f"❌ Failed processing for {pdf_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed processing for {pdf_path.name} with exception: {e}")
    
    logger.info("🎉 Pipeline completed!")

if __name__ == "__main__":
    main() 