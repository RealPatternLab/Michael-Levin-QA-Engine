#!/usr/bin/env python3
"""
Rename PDF files using database metadata.
This script reads from papers.json and renames files accordingly.
"""

import os
import json
import shutil
import re
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_PAPERS_DIR

def load_paper_database():
    """Load the paper database."""
    db_path = Path("data/papers.json")
    
    if not db_path.exists():
        print("❌ Paper database not found.")
        print("   Run: python scripts/create_paper_database.py")
        print("   Or: python scripts/llm_extract_metadata.py")
        return None
    
    with open(db_path, 'r') as f:
        return json.load(f)

def generate_filename_from_database(paper_record):
    """Generate filename from database metadata."""
    
    # Extract components
    title = paper_record.get('title', 'Unknown Title')
    authors = paper_record.get('authors', [])
    year = paper_record.get('year', 'unknown')
    journal = paper_record.get('journal', 'unknown')
    topic = paper_record.get('topic', 'general')
    is_levin = paper_record.get('is_levin_paper', False)
    
    # Generate identifier from title
    title_words = re.findall(r'\b\w+\b', title.lower())
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    key_words = [word for word in title_words if word not in common_words and len(word) > 3]
    identifier = '_'.join(key_words[:2])[:15]
    
    # Clean journal name
    journal_clean = re.sub(r'[^\w]', '_', journal.lower())
    
    # Generate filename
    if is_levin:
        new_filename = f"levin_{topic}_{year}_{journal_clean}_{identifier}.pdf"
    else:
        # Use first author's last name
        if authors:
            first_author = authors[0]
            last_name = first_author.split()[-1].lower()
            last_name = re.sub(r'[^\w]', '', last_name)
        else:
            last_name = 'unknown'
        
        new_filename = f"{last_name}_{topic}_{year}_{journal_clean}_{identifier}.pdf"
    
    return new_filename

def rename_from_database():
    """Rename PDF files using database metadata."""
    
    # Load database
    papers_db = load_paper_database()
    if not papers_db:
        return
    
    print("🔍 Using database to rename PDF files:")
    
    rename_mapping = {}
    missing_files = []
    processed_files = []
    
    for paper in papers_db['papers']:
        current_filename = paper['filename']
        current_path = RAW_PAPERS_DIR / current_filename
        
        if not current_path.exists():
            missing_files.append(current_filename)
            continue
        
        # Generate new filename from database
        new_filename = generate_filename_from_database(paper)
        
        # Only rename if different
        if new_filename != current_filename:
            rename_mapping[current_filename] = new_filename
            
            print(f"\n📄 Processing: {current_filename}")
            print(f"   Title: {paper.get('title', 'Unknown')}")
            print(f"   Authors: {', '.join(paper.get('authors', []))}")
            print(f"   Year: {paper.get('year', 'Unknown')}")
            print(f"   Journal: {paper.get('journal', 'Unknown')}")
            print(f"   Topic: {paper.get('topic', 'Unknown')}")
            print(f"   Is Levin paper: {paper.get('is_levin_paper', False)}")
            print(f"   → {new_filename}")
        else:
            print(f"✅ {current_filename} (already correctly named)")
            processed_files.append(current_filename)
    
    # Report missing files
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} files in database but not found on disk:")
        for filename in missing_files:
            print(f"   - {filename}")
    
    if not rename_mapping:
        print(f"\n✅ All {len(processed_files)} files are already correctly named!")
        return
    
    # Show proposed changes
    print(f"\n🔄 Ready to rename {len(rename_mapping)} files:")
    for old_name, new_name in rename_mapping.items():
        print(f"   {old_name} → {new_name}")
    
    # Perform renaming
    print("\n✅ Renaming files...")
    success_count = 0
    error_count = 0
    
    for old_name, new_name in rename_mapping.items():
        old_path = RAW_PAPERS_DIR / old_name
        new_path = RAW_PAPERS_DIR / new_name
        
        if old_path.exists():
            try:
                shutil.move(str(old_path), str(new_path))
                print(f"✅ {old_name} → {new_name}")
                success_count += 1
                
                # Update database with new filename
                for paper in papers_db['papers']:
                    if paper['filename'] == old_name:
                        paper['filename'] = new_name
                        break
                        
            except Exception as e:
                print(f"❌ Error renaming {old_name}: {e}")
                error_count += 1
        else:
            print(f"⚠️  File not found: {old_name}")
            error_count += 1
    
    # Save updated database with new filenames
    if success_count > 0:
        db_path = Path("data/papers.json")
        with open(db_path, 'w') as f:
            json.dump(papers_db, f, indent=2)
        print(f"✅ Updated database with new filenames")
    
    print(f"\n📊 Renaming complete: {success_count} successful, {error_count} errors")

def main():
    """Main function to execute database-based renaming."""
    rename_from_database()

if __name__ == "__main__":
    main() 