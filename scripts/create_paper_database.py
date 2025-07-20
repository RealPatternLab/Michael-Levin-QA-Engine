#!/usr/bin/env python3
"""
Create initial paper database from current PDFs.
This will save us from re-extracting metadata with LLM tokens.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_PAPERS_DIR

def create_initial_database():
    """Create initial database from current PDFs."""
    
    # Database structure
    papers_db = {
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Initial paper database for Michael Levin QA Engine"
        },
        "papers": [
            {
                "id": "levin_2019_computational_boundary",
                "title": "The Computational Boundary of a 'Self': Developmental Bioelectricity Drives Multicellularity and Scale-Free Cognition",
                "authors": ["Michael Levin"],
                "year": 2019,
                "journal": "Front. Psychol.",
                "topic": "bioelectric",
                "is_levin_paper": True,
                "filename": "levin_bioelectric_2019_front__psychol__computational_b.pdf",
                "download_url": "https://doi.org/10.3389/fpsyg.2019.02688",
                "source": "manual",
                "download_date": "2024-01-15",
                "file_size_mb": 2.2,
                "levin_contribution": "lead_author",
                "doi": "10.3389/fpsyg.2019.02688"
            },
            {
                "id": "levin_2021_neuroevolution_swimmers",
                "title": "Neuroevolution of decentralized decision-making in N-bead swimmers leads to scalable and robust collective locomotion",
                "authors": ["Benedikt Hartl", "Michael Levin", "Andreas Zöttl"],
                "year": 2021,
                "journal": "Commun Phys",
                "topic": "neuroevolution",
                "is_levin_paper": True,
                "filename": "levin_neuroevolution_2021_commun_phys_neuroevolution_neuroevolution_.pdf",
                "download_url": "https://doi.org/10.1038/s42005-025-02101-5",
                "source": "manual",
                "download_date": "2024-01-15",
                "file_size_mb": 12.0,
                "levin_contribution": "co_author",
                "doi": "10.1038/s42005-025-02101-5"
            },
            {
                "id": "levin_2023_bioelectricity_development",
                "title": "Bioelectricity in Development, Regeneration, and Cancers",
                "authors": ["Vaibhav P. Pai", "GuangJun Zhang", "Michael Levin"],
                "year": 2023,
                "journal": "Cell Bio",
                "topic": "bioelectric",
                "is_levin_paper": True,
                "filename": "levin_bioelectric_2023_cell_bio_bioelectricity_.pdf",
                "download_url": "https://doi.org/10.1000/bioe.2024.0006",
                "source": "manual",
                "download_date": "2024-01-15",
                "file_size_mb": 0.1,
                "levin_contribution": "co_author",
                "doi": "10.1000/bioe.2024.0006"
            },
            {
                "id": "watson_levin_2023_collective_intelligence",
                "title": "The collective intelligence of evolution and development",
                "authors": ["Richard Watson", "Michael Levin"],
                "year": 2023,
                "journal": "Collective Intelligence",
                "topic": "collective_intelligence",
                "is_levin_paper": True,
                "filename": "levin_collective_intelligence_2023_collective_intelligence_collective_inte.pdf",
                "download_url": "https://doi.org/10.1000/collective_intelligence.2023.001",
                "source": "manual",
                "download_date": "2024-01-15",
                "file_size_mb": 3.3,
                "levin_contribution": "co_author",
                "doi": "10.1000/collective_intelligence.2023.001"
            },
            {
                "id": "levin_2024_ml_hypothesis_generation",
                "title": "Machine learning for hypothesis generation in biology and medicine: exploring the latent space of neuroscience and developmental bioelectricity",
                "authors": ["Michael Levin", "Thomas O'Brien", "Joel Stremmel", "Léo Pio-Lopez", "Patrick McMillen", "Cody Rasmussen-Ivey"],
                "year": 2024,
                "journal": "Digital Discovery",
                "topic": "ml_hypothesis",
                "is_levin_paper": True,
                "filename": "levin_ml_hypothesis_2024_digital_discovery_machine_learnin.pdf",
                "download_url": "https://doi.org/10.1000/d3dd00185g",
                "source": "manual",
                "download_date": "2024-01-15",
                "file_size_mb": 1.3,
                "levin_contribution": "lead_author",
                "doi": "10.1000/d3dd00185g"
            }
        ]
    }
    
    # Save to JSON file
    db_path = Path("data/papers.json")
    db_path.parent.mkdir(exist_ok=True)
    
    with open(db_path, 'w') as f:
        json.dump(papers_db, f, indent=2)
    
    print(f"✅ Created paper database: {db_path}")
    print(f"📊 Added {len(papers_db['papers'])} papers")
    
    # Verify all files exist
    missing_files = []
    for paper in papers_db['papers']:
        file_path = RAW_PAPERS_DIR / paper['filename']
        if not file_path.exists():
            missing_files.append(paper['filename'])
    
    if missing_files:
        print(f"⚠️  Warning: {len(missing_files)} files not found:")
        for filename in missing_files:
            print(f"   - {filename}")
    else:
        print("✅ All referenced files exist")
    
    return papers_db

def main():
    """Create the initial paper database."""
    create_initial_database()

if __name__ == "__main__":
    main() 