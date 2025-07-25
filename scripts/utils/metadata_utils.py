#!/usr/bin/env python3
"""
Metadata utilities for the pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from file."""
    metadata_file = Path(metadata_path)
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    # Return default metadata structure
    return {
        "papers": {},
        "global_steps": {},
        "last_updated": datetime.now().isoformat()
    }

def save_metadata(metadata: Dict[str, Any], metadata_path: str) -> bool:
    """Save metadata to file with error handling."""
    try:
        metadata_file = Path(metadata_path)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update timestamp
        metadata["last_updated"] = datetime.now().isoformat()
        
        # Write to temporary file first
        temp_file = metadata_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Atomic move
        temp_file.replace(metadata_file)
        return True
        
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def update_paper_metadata(metadata: Dict[str, Any], pdf_name: str, step: str, status: Dict[str, Any]):
    """Update metadata for a specific paper and step."""
    if "papers" not in metadata:
        metadata["papers"] = {}
    
    if pdf_name not in metadata["papers"]:
        metadata["papers"][pdf_name] = {}
    
    if "steps" not in metadata["papers"][pdf_name]:
        metadata["papers"][pdf_name]["steps"] = {}
    
    metadata["papers"][pdf_name]["steps"][step] = status

def is_step_completed(metadata: Dict[str, Any], pdf_name: str, step: str) -> bool:
    """Check if a processing step is completed for a paper."""
    try:
        return metadata["papers"][pdf_name]["steps"][step]["completed"]
    except (KeyError, TypeError):
        return False 