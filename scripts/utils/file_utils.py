#!/usr/bin/env python3
"""
File utilities for the pipeline.
"""

import os
from pathlib import Path
from typing import List

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "inputs/papers",
        "inputs/videos", 
        "inputs/interviews",
        "inputs/datasets",
        "outputs/papers/extracted_texts",
        "outputs/papers/semantic_chunks",
        "outputs/papers/embeddings",
        "outputs/videos",
        "outputs/interviews", 
        "outputs/combined"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_pdf_files(input_dir: str) -> List[Path]:
    """Get all PDF files from the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        return []
    
    pdf_files = list(input_path.glob("*.pdf"))
    return sorted(pdf_files)

def get_output_dir(pdf_file: Path, output_type: str) -> Path:
    """Get the output directory for a specific PDF and output type."""
    # Create a safe filename from the PDF name
    safe_name = pdf_file.stem.replace(" ", "_")
    return Path(f"outputs/papers/{output_type}/{safe_name}")

def ensure_output_dir(output_dir: Path) -> Path:
    """Ensure the output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir 