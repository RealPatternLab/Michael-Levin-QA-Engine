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
from typing import Dict, Any, Optional, List
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
    """Save metadata to file with timeout protection."""
    import threading
    import time
    
    def save_with_timeout():
        """Save metadata with timeout protection."""
        try:
            METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a temporary file first
            temp_file = METADATA_FILE.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Atomic move to final location
            temp_file.replace(METADATA_FILE)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False
    
    # Run save operation with timeout
    result = [None]
    error = [None]
    
    def save_thread():
        try:
            result[0] = save_with_timeout()
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=save_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout=30)  # 30 second timeout
    
    if thread.is_alive():
        logger.error("❌ save_metadata timed out after 30 seconds")
        return False
    
    if error[0]:
        logger.error(f"❌ save_metadata failed: {error[0]}")
        return False
    
    return result[0]

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

                * **Section Headers with Page Numbers:** For each major section header (Introduction, Methods, Results, Discussion, Conclusion, etc.), include the page number where that section begins. Format section headers as "[PAGE X] SECTION TITLE" where X is the page number. For example: "[PAGE 3] INTRODUCTION" or "[PAGE 7] RESULTS AND DISCUSSION".

                * **Exclude:** Headers, footers, page numbers, and publication metadata (e.g., journal name, publication date). References/bibliography sections, figure captions, table captions, and the content of figures and tables themselves should also be excluded.

                * **Prioritize:** Complete capture of all textual content within the scientific paper, from start to finish. Accuracy and completeness of the entire text extraction are paramount. Do not cut off content mid-section.

                * **Do Not:** Summarize, paraphrase, add commentary, or include any explanatory text of your own. Only extract the verbatim text from the PDF.

                This is chunk {chunk_num} of {total_chunks} (pass {pass_num + 1} of {num_passes}) covering pages {chunk_start + 1}-{chunk_end} of {total_pages}.

                Return the extracted text as a single, continuous block of text with section headers marked with their page numbers.

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
        Consolidate the following {num_passes} text extractions from a scientific PDF into a single, high-quality extraction. Each extraction is delimited by ```. The final output must contain the full, combined text of the extractions, prioritizing completeness, accuracy, and a coherent structure. Preserve all page numbers associated with section headers. If any extraction contains section headers with page numbers (e.g., "[PAGE 3] INTRODUCTION"), make sure to keep these page numbers in the final consensus. Include all content present in any of the extractions. Remove redundant information while preserving unique details. If discrepancies exist, choose the most complete and accurate version. Maintain the original document's logical flow, including section headers, figures, tables, and references. Do not include any commentary or descriptions about your selection process or the source of the content. Output only the combined, consolidated text of the scientific PDF.

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
        
        save_success = save_metadata(metadata)
        if not save_success:
            logger.error(f"Failed to save metadata for {pdf_path.name}")
            return False
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
        
        save_success = save_metadata(metadata)
        if not save_success:
            logger.error(f"Failed to save metadata for {pdf_path.name}")
            return False
        return True
        
    except Exception as e:
        logger.error(f"Failed to process text extraction for {pdf_path.name}: {e}")
        return False

def validate_extraction_completeness(pdf_path: Path, consensus_text: str) -> Dict[str, Any]:
    """Validate that the consensus text faithfully represents the PDF content."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Read the original PDF to compare
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return {"faithfully_transcribed": False, "missing_segments": ["Failed to read original PDF"]}
        
        prompt = f"""
        Compare the extracted consensus text against the original PDF text to determine if the extraction is complete and faithful.

        Original PDF text (first 2000 chars):
        {pdf_text[:2000]}

        Extracted consensus text (first 2000 chars):
        {consensus_text[:2000]}

        Analyze if the consensus text faithfully represents the PDF content. Look for:
        1. Missing sections or paragraphs
        2. Incomplete sentences or thoughts
        3. Missing figures, tables, or references
        4. Structural differences

        Return ONLY a JSON object with this exact structure:
        {{
            "faithfully_transcribed": true/false,
            "missing_segments": ["list of missing sections or issues found"]
        }}

        CRITICAL: Use True/False (capitalized) for the boolean value.
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
                
                validation_result = json.loads(response_text)
                
                # Handle both "true"/"false" and "True"/"False"
                if isinstance(validation_result.get("faithfully_transcribed"), str):
                    validation_result["faithfully_transcribed"] = validation_result["faithfully_transcribed"].lower() == "true"
                
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

def create_semantic_chunks_prompt() -> str:
    """Create the prompt for Gemini to extract semantic chunks."""
    return """
    Extract semantically meaningful chunks from the provided scientific paper text for embedding into a vector database. Semantic chunking, in this context, means dividing the paper into logical units of information that represent self-contained, coherent ideas or topics. Each chunk should be focused on a single concept, finding, or argument.

    Guidelines for good chunks:

    * **Self-contained:** A chunk should make sense on its own without requiring excessive context from surrounding text.
    * **Coherent:** The information within a chunk should relate to a single, identifiable topic.
    * **Specific:** Avoid overly broad or generic chunks. Aim for specificity and detail.
    * **Optimal size:** Chunks should ideally be between 100-300 words, although shorter or longer chunks are acceptable if they represent a complete semantic unit.

    Section headers (e.g., "Introduction," "Methods," "Results," "Discussion") should be identified and included in the output JSON. If page numbers are associated with section headers (e.g., "[PAGE 3] INTRODUCTION"), extract the page number and include it as "page_estimate" in the JSON. Calculate character positions (start_char and end_char) relative to the beginning of the entire input text.

    CRITICAL: You must return ONLY a valid JSON array. The JSON must be properly formatted with:
    - All strings properly quoted and escaped
    - No trailing commas
    - Properly closed brackets and braces
    - No unterminated strings

    The output should be a JSON array of objects, strictly adhering to this format:

    ```json
    [
      {
        "text": "The actual text content of the semantic chunk",
        "source_title": "Full paper title",
        "year": 2023,
        "section_header": "Introduction/Discussion/Results/etc",
        "semantic_topic": "Brief description of the main idea/topic (max 20 words)",
        "page_estimate": 5
      }
    ]
    ```

    Examples:

    **Good Chunk:** "Our results demonstrate a statistically significant correlation between variable X and variable Y (p < 0.05). This finding supports the hypothesis that X influences Y through mechanism Z. Further research is needed to explore the specific pathways involved." (semantic_topic: "Correlation between X and Y and its implications")

    **Bad Chunk:** "Introduction. In this study, we investigated... Methods. We used a randomized controlled trial... Results. The results are presented in Table 1." (This combines multiple unrelated topics and sections.)

    Challenges:

    * **Overlapping content:** Minimize redundancy between chunks. If a concept is discussed across multiple sections, try to synthesize the information into a single, comprehensive chunk or create distinct chunks with clear semantic distinctions.
    * **Incomplete ideas:** If an idea spans multiple paragraphs, ensure the entire idea is captured within a single chunk. Do not split a coherent thought across multiple chunks.

    Ensure the chunks are suitable for vector database search and retrieval. This means each chunk should represent a distinct, searchable concept that can be effectively retrieved based on its semantic content. Provide the `source_title` and `year` information using the values I will supply separately. I will provide the paper title and year outside of the main paper text.

    IMPORTANT: Return ONLY the JSON array, no additional text, explanations, or markdown formatting.

    Paper text to analyze:
    """

def extract_semantic_chunks(consensus_text: str, paper_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract semantic chunks from consensus text using Gemini."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            import google.generativeai as genai
            import time
            import threading
            
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
            
            logger.info(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})...")
            start_time = time.time()
            
            # Simple timeout mechanism
            response = None
            api_error = None
            
            def call_gemini():
                nonlocal response, api_error
                try:
                    response = model.generate_content(prompt)
                except Exception as e:
                    api_error = e
            
            # Run API call in thread with timeout
            thread = threading.Thread(target=call_gemini)
            thread.daemon = True
            thread.start()
            thread.join(timeout=120)  # 2 minute timeout
            
            if thread.is_alive():
                logger.error(f"Gemini API call timed out after 120 seconds (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(2)
                    continue
                return []
            
            if api_error:
                logger.error(f"Gemini API call failed (attempt {attempt + 1}): {api_error}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(2)
                    continue
                return []
            
            logger.info(f"Gemini API call completed in {time.time() - start_time:.1f}s")
            
            if response and response.text:
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
                    
                    # Try to fix common JSON issues
                    response_text = fix_json_response(response_text)
                    
                    chunks = json.loads(response_text)
                    
                    # Validate chunks structure
                    if not isinstance(chunks, list):
                        logger.error("Response is not a list")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                            continue
                        return []
                    
                    # Validate each chunk has required fields
                    valid_chunks = []
                    for chunk in chunks:
                        required_fields = ["text", "source_title", "year", "section_header", "semantic_topic", "page_estimate"]
                        if all(field in chunk for field in required_fields):
                            valid_chunks.append(chunk)
                        else:
                            logger.warning(f"Skipping chunk with missing fields: {chunk}")
                    
                    if valid_chunks:
                        logger.info(f"✅ Successfully extracted {len(valid_chunks)} semantic chunks")
                        return valid_chunks
                    else:
                        logger.error("No valid chunks found after validation")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                            continue
                        return []
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    logger.error(f"Response text: {response.text[:500]}...")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                        continue
                    return []
            else:
                logger.error("Empty response from Gemini")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                return []
            
        except Exception as e:
            logger.error(f"Failed to extract semantic chunks (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2)
                continue
            return []
    
    logger.error(f"Failed to extract semantic chunks after {max_retries} attempts")
    return []

def fix_json_response(response_text: str) -> str:
    """Fix common JSON formatting issues in Gemini responses."""
    import re
    
    # Remove any trailing commas before closing brackets/braces
    response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
    
    # Fix unterminated strings by finding the last complete quote
    lines = response_text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Count quotes in the line
        quote_count = line.count('"')
        if quote_count % 2 != 0:  # Odd number of quotes means unterminated string
            # Find the last complete quote pair and truncate after it
            last_complete_quote = line.rfind('",')
            if last_complete_quote != -1:
                line = line[:last_complete_quote + 2]
            else:
                # If no complete quote pair, try to close the string
                last_quote = line.rfind('"')
                if last_quote != -1:
                    line = line[:last_quote + 1] + '",'
        
        fixed_lines.append(line)
    
    response_text = '\n'.join(fixed_lines)
    
    # Ensure the JSON array is properly closed
    if not response_text.strip().endswith(']'):
        # Find the last complete object and close the array
        last_complete_object = response_text.rfind('}')
        if last_complete_object != -1:
            response_text = response_text[:last_complete_object + 1] + '\n]'
    
    return response_text

def process_semantic_chunking(pdf_path: Path, metadata: Dict[str, Any]) -> bool:
    """Process semantic chunking for a single PDF file."""
    try:
        pdf_name = pdf_path.name
        logger.info(f"Processing semantic chunking for: {pdf_name}")
        
        # Check if consensus text exists - look in extracted_texts subdirectory
        pdf_output_dir = Path("outputs/extracted_texts") / pdf_path.stem
        consensus_file = pdf_output_dir / "consensus.txt"
        
        if not consensus_file.exists():
            logger.error(f"Consensus text not found for {pdf_name}")
            return False
        
        # Read consensus text
        with open(consensus_file, 'r', encoding='utf-8') as f:
            consensus_text = f.read()
        
        if not consensus_text.strip():
            logger.error(f"Empty consensus text for {pdf_name}")
            return False
        
        # Get paper metadata
        paper_metadata = metadata["papers"].get(pdf_name, {})
        if not paper_metadata:
            logger.error(f"No metadata found for {pdf_name}")
            return False
        
        # Extract semantic chunks
        chunks = extract_semantic_chunks(consensus_text, paper_metadata.get("metadata", {}))
        
        if not chunks:
            logger.error(f"Failed to extract semantic chunks for {pdf_name}")
            return False
        
        # Save chunks to JSON file in the same directory as consensus.txt
        chunks_file = pdf_output_dir / "semantic_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(chunks)} semantic chunks to {chunks_file}")
        
        # Update metadata with more robust error handling
        logger.info(f"Updating metadata for {pdf_name}...")
        try:
            with metadata_lock:
                logger.info(f"Inside metadata_lock for {pdf_name}")
                
                # Ensure the paper exists in metadata
                if pdf_name not in metadata["papers"]:
                    logger.error(f"Paper {pdf_name} not found in metadata during update")
                    return False
                
                # Ensure steps dict exists
                if "steps" not in metadata["papers"][pdf_name]:
                    logger.info(f"Creating steps dict for {pdf_name}")
                    metadata["papers"][pdf_name]["steps"] = {}
                
                # Add semantic chunking step
                logger.info(f"Adding semantic_chunking step for {pdf_name}")
                metadata["papers"][pdf_name]["steps"]["semantic_chunking"] = {
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "num_chunks": len(chunks),
                    "chunks_file": str(chunks_file)
                }
                
                # Save metadata with explicit error handling
                logger.info(f"Saving metadata for {pdf_name}...")
                try:
                    save_success = save_metadata(metadata)
                    if save_success:
                        logger.info(f"✅ Metadata saved successfully for {pdf_name}")
                    else:
                        logger.error(f"❌ Failed to save metadata for {pdf_name}")
                        # Try to save a backup
                        try:
                            backup_file = Path("outputs/metadata_backup.json")
                            with open(backup_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            logger.info(f"✅ Created backup at {backup_file}")
                        except Exception as backup_error:
                            logger.error(f"❌ Failed to create backup: {backup_error}")
                        return False
                except Exception as save_error:
                    logger.error(f"❌ Failed to save metadata for {pdf_name}: {save_error}")
                    # Try to save a backup
                    try:
                        backup_file = Path("outputs/metadata_backup.json")
                        with open(backup_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        logger.info(f"✅ Created backup at {backup_file}")
                    except Exception as backup_error:
                        logger.error(f"❌ Failed to create backup: {backup_error}")
                    return False
                
        except Exception as lock_error:
            logger.error(f"❌ Error in metadata_lock block for {pdf_name}: {lock_error}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process semantic chunking for {pdf_path.name}: {e}")
        return False

def process_vector_embedding(pdf_path: Path, metadata: Dict[str, Any]) -> bool:
    """Process vector embedding step for a single PDF file."""
    try:
        pdf_name = pdf_path.name
        logger.info(f"Processing vector embedding for: {pdf_name}")
        
        # Check if semantic chunks exist
        pdf_output_dir = Path("outputs/extracted_texts") / pdf_path.stem
        chunks_file = pdf_output_dir / "semantic_chunks.json"
        
        if not chunks_file.exists():
            logger.error(f"Semantic chunks not found for {pdf_name}")
            return False
        
        # Load semantic chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            logger.error(f"No semantic chunks found for {pdf_name}")
            return False
        
        logger.info(f"Found {len(chunks)} semantic chunks for {pdf_name}")
        
        # Create embeddings for all chunks
        embeddings = []
        chunk_texts = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            if chunk_text.strip():
                chunk_texts.append(chunk_text)
        
        if not chunk_texts:
            logger.error(f"No valid text chunks found for {pdf_name}")
            return False
        
        # Create embeddings using OpenAI
        logger.info(f"Creating embeddings for {len(chunk_texts)} chunks...")
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Batch process embeddings (OpenAI allows up to 2048 inputs per request)
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i + batch_size]
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(chunk_texts) + batch_size - 1)//batch_size}")
                
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-large"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"✅ Successfully created {len(all_embeddings)} embeddings")
            
            # Save embeddings to file
            embeddings_file = pdf_output_dir / "embeddings.json"
            embeddings_data = {
                "chunks": chunks,
                "embeddings": all_embeddings,
                "model": "text-embedding-3-large",
                "timestamp": datetime.now().isoformat()
            }
            
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved embeddings to {embeddings_file}")
            
            # Update metadata
            logger.info(f"Updating metadata for {pdf_name}...")
            try:
                with metadata_lock:
                    logger.info(f"Inside metadata_lock for {pdf_name}")
                    
                    # Ensure the paper exists in metadata
                    if pdf_name not in metadata["papers"]:
                        logger.error(f"Paper {pdf_name} not found in metadata during update")
                        return False
                    
                    # Ensure steps dict exists
                    if "steps" not in metadata["papers"][pdf_name]:
                        logger.info(f"Creating steps dict for {pdf_name}")
                        metadata["papers"][pdf_name]["steps"] = {}
                    
                    # Add vector embedding step
                    logger.info(f"Adding vector_embedding step for {pdf_name}")
                    metadata["papers"][pdf_name]["steps"]["vector_embedding"] = {
                        "completed": True,
                        "timestamp": datetime.now().isoformat(),
                        "num_embeddings": len(all_embeddings),
                        "model": "text-embedding-3-large",
                        "embeddings_file": str(embeddings_file)
                    }
                    
                    # Save metadata with explicit error handling
                    logger.info(f"Saving metadata for {pdf_name}...")
                    try:
                        save_success = save_metadata(metadata)
                        if save_success:
                            logger.info(f"✅ Metadata saved successfully for {pdf_name}")
                        else:
                            logger.error(f"❌ Failed to save metadata for {pdf_name}")
                            # Try to save a backup
                            try:
                                backup_file = Path("outputs/metadata_backup.json")
                                with open(backup_file, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                logger.info(f"✅ Created backup at {backup_file}")
                            except Exception as backup_error:
                                logger.error(f"❌ Failed to create backup: {backup_error}")
                            return False
                    except Exception as save_error:
                        logger.error(f"❌ Failed to save metadata for {pdf_name}: {save_error}")
                        # Try to save a backup
                        try:
                            backup_file = Path("outputs/metadata_backup.json")
                            with open(backup_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            logger.info(f"✅ Created backup at {backup_file}")
                        except Exception as backup_error:
                            logger.error(f"❌ Failed to create backup: {backup_error}")
                        return False
                    
            except Exception as lock_error:
                logger.error(f"❌ Error in metadata_lock block for {pdf_name}: {lock_error}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embeddings for {pdf_name}: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to process vector embedding for {pdf_path.name}: {e}")
        return False

def build_combined_faiss_index():
    """Build a combined FAISS index from all embeddings."""
    try:
        import faiss
        import numpy as np
        
        logger.info("🔍 Building combined FAISS index from all embeddings...")
        
        # Find all embeddings files
        extracted_dir = Path("outputs/extracted_texts")
        all_embeddings = []
        all_chunks = []
        
        if not extracted_dir.exists():
            logger.error("Extracted texts directory not found")
            return False
        
        # Load all embeddings
        for paper_dir in extracted_dir.iterdir():
            if paper_dir.is_dir():
                embeddings_file = paper_dir / "embeddings.json"
                if embeddings_file.exists():
                    try:
                        with open(embeddings_file, 'r', encoding='utf-8') as f:
                            embeddings_data = json.load(f)
                        
                        chunks = embeddings_data.get('chunks', [])
                        embeddings = embeddings_data.get('embeddings', [])
                        
                        if chunks and embeddings and len(chunks) == len(embeddings):
                            all_chunks.extend(chunks)
                            all_embeddings.extend(embeddings)
                            logger.info(f"Loaded {len(embeddings)} embeddings from {paper_dir.name}")
                        else:
                            logger.warning(f"Skipping {paper_dir.name} - mismatched chunks/embeddings")
                            
                    except Exception as e:
                        logger.error(f"Failed to load embeddings from {embeddings_file}: {e}")
        
        if not all_embeddings:
            logger.error("No embeddings found to build index")
            return False
        
        logger.info(f"Total embeddings to index: {len(all_embeddings)}")
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Build FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings_array)
        
        # Save index and metadata
        index_file = Path("outputs/faiss_index.bin")
        chunks_file = Path("outputs/combined_chunks.json")
        
        faiss.write_index(index, str(index_file))
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Built FAISS index with {len(all_embeddings)} embeddings")
        logger.info(f"✅ Saved index to {index_file}")
        logger.info(f"✅ Saved combined chunks to {chunks_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        return False

def append_to_faiss_index(new_chunks, new_embeddings):
    """Append new data to existing index."""
    # Load existing index
    existing_index = faiss.read_index("outputs/faiss_index.bin")
    
    # Add new embeddings
    new_embeddings_array = np.array(new_embeddings).astype('float32')
    faiss.normalize_L2(new_embeddings_array)
    existing_index.add(new_embeddings_array)
    
    # Save updated index
    faiss.write_index(existing_index, "outputs/faiss_index.bin")
    
    # Append to combined_chunks.json
    with open("outputs/combined_chunks.json", "r") as f:
        existing_chunks = json.load(f)
    existing_chunks.extend(new_chunks)
    # Save updated chunks

def process_faiss_index_building(metadata: Dict[str, Any]) -> bool:
    """Process FAISS index building step."""
    try:
        logger.info("Processing FAISS index building...")
        
        # Check if all papers have embeddings
        papers_with_embeddings = 0
        total_papers = 0
        
        for paper_name, paper_data in metadata.get("papers", {}).items():
            total_papers += 1
            steps = paper_data.get("steps", {})
            if steps.get("vector_embedding", {}).get("completed", False):
                papers_with_embeddings += 1
        
        if papers_with_embeddings == 0:
            logger.error("No papers have embeddings yet. Run vector embedding step first.")
            return False
        
        if papers_with_embeddings < total_papers:
            logger.warning(f"Only {papers_with_embeddings}/{total_papers} papers have embeddings")
            logger.info("Building index with available embeddings...")
        
        # Build the index
        success = build_combined_faiss_index()
        
        if success:
            # Update metadata
            with metadata_lock:
                if "global_steps" not in metadata:
                    metadata["global_steps"] = {}
                
                metadata["global_steps"]["faiss_index"] = {
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "papers_with_embeddings": papers_with_embeddings,
                    "total_papers": total_papers
                }
                
                save_success = save_metadata(metadata)
                if save_success:
                    logger.info("✅ FAISS index building completed and metadata updated")
                else:
                    logger.error("❌ Failed to save metadata after FAISS index building")
                    return False
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to process FAISS index building: {e}")
        return False

def reset_file_processing(pdf_filename: str):
    """Reset processing state for a specific file to allow reprocessing."""
    metadata = load_metadata()
    
    with metadata_lock:
        if pdf_filename in metadata["papers"]:
            # Remove the file from metadata to allow reprocessing
            del metadata["papers"][pdf_filename]
            save_success = save_metadata(metadata)
            if save_success:
                logger.info(f"Reset processing state for {pdf_filename}")
            else:
                logger.error(f"Failed to save metadata when resetting {pdf_filename}")
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
    
    # Step 3: Semantic chunking
    if not steps.get("semantic_chunking", {}).get("completed", False):
        logger.info(f"Step 3: Extracting semantic chunks for {pdf_path.name}")
        if process_semantic_chunking(pdf_path, metadata):
            logger.info(f"✅ Completed semantic chunking for {pdf_path.name}")
        else:
            logger.error(f"❌ Failed semantic chunking for {pdf_path.name}")
            return False
    else:
        logger.info(f"Skipping semantic chunking for {pdf_path.name} - already completed")
    
    # Step 4: Vector embedding
    if not steps.get("vector_embedding", {}).get("completed", False):
        logger.info(f"Step 4: Embedding semantic chunks for {pdf_path.name}")
        if process_vector_embedding(pdf_path, metadata):
            logger.info(f"✅ Completed vector embedding for {pdf_path.name}")
        else:
            logger.error(f"❌ Failed vector embedding for {pdf_path.name}")
            return False
    else:
        logger.info(f"Skipping vector embedding for {pdf_path.name} - already completed")
    
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
    
    # Step 5: Build combined FAISS index
    logger.info("Step 5: Building combined FAISS index...")
    if process_faiss_index_building(metadata):
        logger.info("✅ Completed FAISS index building")
    else:
        logger.error("❌ Failed FAISS index building")
    
    logger.info("🎉 Pipeline completed!")

if __name__ == "__main__":
    main() 