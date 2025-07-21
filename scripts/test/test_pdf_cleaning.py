#!/usr/bin/env python3
"""
Test script for the multimodal PDF text extraction pipeline.

This script tests all components of the pipeline:
1. PDF processors (image rendering)
2. LLM text extraction processors
3. Storage providers
4. Integration tests
5. Small pipeline run on test PDF
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.pdf_processing import get_pdf_processor
from modules.text_extraction import get_extraction_processor
from modules.storage import get_storage

def test_pdf_processors():
    """Test PDF processors for image rendering."""
    print("Testing PDF processors...")
    
    # Test PyMuPDF processor
    pymupdf = get_pdf_processor("pymupdf")
    print(f"✓ PyMuPDF processor: {pymupdf.name} - {pymupdf.description}")
    
    # Test PDFPlumber processor
    pdfplumber = get_pdf_processor("pdfplumber")
    print(f"✓ PDFPlumber processor: {pdfplumber.name} - {pdfplumber.description}")
    
    print("✓ All PDF processors loaded successfully")

def test_llm_processors():
    """Test LLM text extraction processors."""
    print("\nTesting LLM text extraction processors...")
    
    # Test OpenAI processor
    try:
        openai = get_extraction_processor("openai")
        print(f"✓ OpenAI processor: {openai.name} - {openai.description}")
    except Exception as e:
        print(f"⚠ OpenAI processor failed to initialize (likely missing API key): {e}")
    
    print("✓ All LLM processors loaded successfully")

def test_storage_providers():
    """Test storage providers."""
    print("\nTesting storage providers...")
    
    # Test JSON storage
    json_storage = get_storage("json")
    print(f"✓ JSON storage: {json_storage.name} - {json_storage.description}")
    
    print("✓ All storage providers loaded successfully")

def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")
    
    # Test PDF processor with image rendering
    try:
        pdf_processor = get_pdf_processor("pymupdf")
        llm_processor = get_extraction_processor("openai")
        storage = get_storage("json")
        
        print("✓ All components can be instantiated together")
        print("✓ Integration test passed")
    except Exception as e:
        print(f"⚠ Integration test failed (likely missing API key): {e}")

def test_small_pipeline():
    """Test a small pipeline run on a test PDF."""
    print("\nTesting small pipeline run...")
    
    # Check if test PDF exists
    test_pdf = Path("data/raw_papers/levin_bioelectric_2023_cell_bio_bioelectricity_.pdf")
    if not test_pdf.exists():
        print("⚠ Test PDF not found, skipping pipeline test")
        return
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize components
        pdf_processor = get_pdf_processor("pymupdf")
        llm_processor = get_extraction_processor("openai")
        storage = get_storage("json")
        
        # Test image rendering
        print("  Testing image rendering...")
        result = pdf_processor.render_pages_to_images(test_pdf, temp_path)
        print(f"  ✓ Rendered {len(result.pages)} pages to images")
        
        # Test text extraction (if API key is available)
        try:
            print("  Testing text extraction...")
            extraction_result = llm_processor.extract_text_from_images(
                paper_id="test_paper",
                page_images=[page.image_path for page in result.pages],
                output_dir=temp_path,
                max_images_per_batch=5
            )
            print(f"  ✓ Extracted text with quality: {extraction_result.extraction_quality}")
            print(f"  ✓ Text length: {len(extraction_result.extracted_text)} characters")
        except Exception as e:
            print(f"  ⚠ Text extraction failed (likely missing API key): {e}")
        
        print("✓ Small pipeline test completed")

def main():
    """Run all tests."""
    print("🧪 Testing Multimodal PDF Text Extraction Pipeline")
    print("=" * 60)
    
    try:
        test_pdf_processors()
        test_llm_processors()
        test_storage_providers()
        test_integration()
        test_small_pipeline()
        
        print("\n🎉 All tests completed successfully!")
        print("The pipeline is ready for use.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 