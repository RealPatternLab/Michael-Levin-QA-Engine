#!/usr/bin/env python3
"""
Reprocess Consensus Files

This script goes through all processed papers and reprocesses their consensus.txt files
using the updated consensus prompt that preserves page numbers in section headers.
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

def load_metadata() -> Dict[str, Any]:
    """Load existing metadata."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {"papers": {}}

def create_consensus_extraction_from_files(pdf_output_dir: Path, num_passes: int) -> str:
    """Create a consensus extraction from saved pass files using updated prompt."""
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

def reprocess_paper_consensus(pdf_name: str, metadata: Dict[str, Any]) -> bool:
    """Reprocess consensus for a single paper."""
    try:
        logger.info(f"Reprocessing consensus for: {pdf_name}")
        
        # Get the detailed passes directory
        paper_metadata = metadata["papers"].get(pdf_name, {})
        steps = paper_metadata.get("steps", {})
        text_extraction = steps.get("text_extraction", {})
        
        if not text_extraction.get("completed", False):
            logger.warning(f"Skipping {pdf_name} - text extraction not completed")
            return False
        
        detailed_passes_dir = text_extraction.get("detailed_passes_dir")
        if not detailed_passes_dir:
            logger.warning(f"Skipping {pdf_name} - no detailed passes directory found")
            return False
        
        pdf_output_dir = Path(detailed_passes_dir)
        if not pdf_output_dir.exists():
            logger.warning(f"Skipping {pdf_name} - detailed passes directory does not exist: {pdf_output_dir}")
            return False
        
        # Count available pass files
        pass_files = list(pdf_output_dir.glob("pass_*.txt"))
        num_passes = len(pass_files)
        
        if num_passes == 0:
            logger.warning(f"Skipping {pdf_name} - no pass files found in {pdf_output_dir}")
            return False
        
        logger.info(f"Found {num_passes} pass files for {pdf_name}")
        
        # Create new consensus
        consensus_text = create_consensus_extraction_from_files(pdf_output_dir, num_passes)
        
        if not consensus_text:
            logger.error(f"Failed to create consensus for {pdf_name}")
            return False
        
        # Save new consensus
        consensus_file = pdf_output_dir / "consensus.txt"
        with open(consensus_file, 'w', encoding='utf-8') as f:
            f.write(consensus_text)
        
        logger.info(f"✅ Successfully reprocessed consensus for {pdf_name}: {len(consensus_text)} characters")
        
        # Update metadata with new timestamp
        metadata["papers"][pdf_name]["steps"]["text_extraction"]["consensus_reprocessed"] = {
            "timestamp": datetime.now().isoformat(),
            "num_passes_used": num_passes
        }
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to reprocess consensus for {pdf_name}: {e}")
        return False

def save_metadata(metadata: Dict[str, Any]):
    """Save metadata to file."""
    with metadata_lock:
        METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    """Main function to reprocess all consensus files."""
    logger.info("🚀 Starting consensus reprocessing for all papers")
    
    # Load metadata
    metadata = load_metadata()
    
    if not metadata.get("papers"):
        logger.info("No papers found in metadata")
        return
    
    # Get all processed papers (excluding Watson-Levin which was already reprocessed)
    processed_papers = []
    for pdf_name, paper_data in metadata["papers"].items():
        steps = paper_data.get("steps", {})
        if steps.get("text_extraction", {}).get("completed", False):
            # Skip Watson-Levin paper as it was already reprocessed
            if pdf_name == "watson-levin-2023-the-collective-intelligence-of-evolution-and-development.pdf":
                logger.info(f"Skipping {pdf_name} - already reprocessed with updated prompts")
                continue
            processed_papers.append(pdf_name)
    
    logger.info(f"Found {len(processed_papers)} papers with completed text extraction (excluding Watson-Levin)")
    
    if not processed_papers:
        logger.info("No papers with completed text extraction found")
        return
    
    # Reprocess all papers in parallel
    logger.info(f"Processing {len(processed_papers)} papers in parallel...")
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=min(len(processed_papers), 5)) as executor: # Limit concurrent workers
        future_to_pdf = {executor.submit(reprocess_paper_consensus, pdf_name, metadata): pdf_name for pdf_name in processed_papers}
        
        for future in as_completed(future_to_pdf):
            pdf_name = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"✅ Successfully reprocessed consensus for {pdf_name}")
                    successful += 1
                else:
                    logger.error(f"❌ Failed to reprocess consensus for {pdf_name}")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ Failed to reprocess consensus for {pdf_name} with exception: {e}")
                failed += 1
    
    # Save updated metadata
    save_metadata(metadata)
    
    logger.info(f"🎉 Consensus reprocessing completed!")
    logger.info(f"✅ Successfully reprocessed: {successful} papers")
    logger.info(f"❌ Failed to reprocess: {failed} papers")

if __name__ == "__main__":
    main() 