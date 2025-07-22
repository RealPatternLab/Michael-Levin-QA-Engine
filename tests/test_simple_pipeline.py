"""
Tests for the simple PDF processing pipeline.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any

# Import the functions we want to test
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.simple_pipeline import (
    get_pdf_files,
    load_metadata,
    save_metadata,
    extract_text_from_pdf,
    extract_metadata_with_openai,
    extract_text_with_document_ai,
    generate_filename,
    process_pdf
)


class TestGetPdfFiles:
    """Test PDF file discovery."""
    
    def test_get_pdf_files_success(self, tmp_path):
        """Test finding PDF files in directory."""
        # Create test directory structure
        raw_papers_dir = tmp_path / "inputs" / "raw_papers"
        raw_papers_dir.mkdir(parents=True)
        
        # Create test PDF files
        (raw_papers_dir / "test1.pdf").touch()
        (raw_papers_dir / "test2.pdf").touch()
        (raw_papers_dir / "not_a_pdf.txt").touch()
        
        with patch('scripts.simple_pipeline.RAW_PAPERS_DIR', raw_papers_dir):
            result = get_pdf_files()
            
        assert len(result) == 2
        assert any("test1.pdf" in str(f) for f in result)
        assert any("test2.pdf" in str(f) for f in result)
    
    def test_get_pdf_files_empty_directory(self, tmp_path):
        """Test behavior when directory is empty."""
        raw_papers_dir = tmp_path / "inputs" / "raw_papers"
        raw_papers_dir.mkdir(parents=True)
        
        with patch('scripts.simple_pipeline.RAW_PAPERS_DIR', raw_papers_dir):
            result = get_pdf_files()
            
        assert result == []
    
    def test_get_pdf_files_directory_not_exists(self, tmp_path):
        """Test behavior when directory doesn't exist."""
        non_existent_dir = tmp_path / "nonexistent"
        
        with patch('scripts.simple_pipeline.RAW_PAPERS_DIR', non_existent_dir):
            result = get_pdf_files()
            
        assert result == []


class TestLoadMetadata:
    """Test metadata loading functionality."""
    
    def test_load_metadata_existing_file(self, tmp_path):
        """Test loading existing metadata file."""
        metadata_file = tmp_path / "metadata.json"
        test_data = {"papers": {"test.pdf": {"title": "Test"}}, "processed_files": ["test.pdf"]}
        
        with patch('scripts.simple_pipeline.METADATA_FILE', metadata_file):
            with open(metadata_file, 'w') as f:
                json.dump(test_data, f)
            
            result = load_metadata()
            
        assert result == test_data
    
    def test_load_metadata_new_file(self, tmp_path):
        """Test creating new metadata when file doesn't exist."""
        metadata_file = tmp_path / "metadata.json"
        
        with patch('scripts.simple_pipeline.METADATA_FILE', metadata_file):
            result = load_metadata()
            
        assert result == {"papers": {}, "processed_files": []}
    
    def test_load_metadata_corrupted_file(self, tmp_path):
        """Test handling corrupted JSON file."""
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text("invalid json")
        
        with patch('scripts.simple_pipeline.METADATA_FILE', metadata_file):
            result = load_metadata()
            
        assert result == {"papers": {}, "processed_files": []}


class TestSaveMetadata:
    """Test metadata saving functionality."""
    
    def test_save_metadata_success(self, tmp_path):
        """Test successfully saving metadata."""
        metadata_file = tmp_path / "metadata.json"
        test_data = {"papers": {"test.pdf": {"title": "Test"}}, "processed_files": ["test.pdf"]}
        
        with patch('scripts.simple_pipeline.METADATA_FILE', metadata_file):
            save_metadata(test_data)
            
        assert metadata_file.exists()
        with open(metadata_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == test_data


class TestExtractTextFromPdf:
    """Test PDF text extraction."""
    
    def test_extract_text_from_pdf_success(self, tmp_path):
        """Test successful text extraction."""
        pdf_path = tmp_path / "test.pdf"
        
        # Mock PdfReader
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_reader.pages = [mock_page]
        
        with patch('scripts.simple_pipeline.PdfReader', return_value=mock_reader):
            result = extract_text_from_pdf(pdf_path)
            
        assert result == "Test PDF content\n"
    
    def test_extract_text_from_pdf_failure(self, tmp_path):
        """Test text extraction failure."""
        pdf_path = tmp_path / "test.pdf"
        
        with patch('scripts.simple_pipeline.PdfReader', side_effect=Exception("PDF Error")):
            result = extract_text_from_pdf(pdf_path)
            
        assert result is None


class TestExtractMetadataWithOpenai:
    """Test OpenAI metadata extraction."""
    
    def test_extract_metadata_with_openai_success(self):
        """Test successful metadata extraction."""
        mock_response = Mock()
        mock_response.choices[0].message.content = '{"title": "Test Paper", "authors": ["John Doe"], "year": 2024, "journal": "Nature", "is_levin_paper": false}'
        
        with patch('scripts.simple_pipeline.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            result = extract_metadata_with_openai("test text", "test.pdf")
            
        assert result["title"] == "Test Paper"
        assert result["authors"] == ["John Doe"]
        assert result["year"] == 2024
        assert result["journal"] == "Nature"
        assert result["is_levin_paper"] is False
    
    def test_extract_metadata_with_openai_markdown_response(self):
        """Test handling markdown code blocks in response."""
        mock_response = Mock()
        mock_response.choices[0].message.content = '```json\n{"title": "Test Paper"}\n```'
        
        with patch('scripts.simple_pipeline.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            result = extract_metadata_with_openai("test text", "test.pdf")
            
        assert result["title"] == "Test Paper"
    
    def test_extract_metadata_with_openai_failure(self):
        """Test metadata extraction failure."""
        with patch('scripts.simple_pipeline.OpenAI', side_effect=Exception("API Error")):
            result = extract_metadata_with_openai("test text", "test.pdf")
            
        assert result is None
    
    def test_extract_metadata_with_openai_invalid_json(self):
        """Test handling invalid JSON response."""
        mock_response = Mock()
        mock_response.choices[0].message.content = 'invalid json'
        
        with patch('scripts.simple_pipeline.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            result = extract_metadata_with_openai("test text", "test.pdf")
            
        assert result is None


class TestExtractTextWithDocumentAi:
    """Test Google Document AI text extraction."""
    
    def test_extract_text_with_document_ai_success(self, tmp_path):
        """Test successful text extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")
        
        mock_document = Mock()
        mock_document.text = "Extracted text content"
        
        mock_result = Mock()
        mock_result.document = mock_document
        
        with patch('scripts.simple_pipeline.documentai') as mock_documentai:
            mock_client = Mock()
            mock_client.process_document.return_value = mock_result
            mock_documentai.DocumentProcessorServiceClient.return_value = mock_client
            
            # Mock environment variables
            with patch.dict('os.environ', {
                'GOOGLE_DOCUMENT_AI_PROJECT_ID': 'test-project',
                'GOOGLE_DOCUMENT_AI_PROCESSOR_ID': 'test-processor'
            }):
                result = extract_text_with_document_ai(pdf_path)
                
        assert result == "Extracted text content"
    
    def test_extract_text_with_document_ai_missing_credentials(self):
        """Test behavior when credentials are missing."""
        with patch.dict('os.environ', {}, clear=True):
            result = extract_text_with_document_ai(Path("test.pdf"))
            
        assert result is None
    
    def test_extract_text_with_document_ai_failure(self, tmp_path):
        """Test text extraction failure."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")
        
        with patch('scripts.simple_pipeline.documentai', side_effect=Exception("API Error")):
            with patch.dict('os.environ', {
                'GOOGLE_DOCUMENT_AI_PROJECT_ID': 'test-project',
                'GOOGLE_DOCUMENT_AI_PROCESSOR_ID': 'test-processor'
            }):
                result = extract_text_with_document_ai(pdf_path)
                
        assert result is None


class TestGenerateFilename:
    """Test filename generation."""
    
    def test_generate_filename_complete_metadata(self):
        """Test filename generation with complete metadata."""
        metadata = {
            "title": "Machine Learning for Hypothesis Generation",
            "authors": ["John Doe", "Jane Smith"],
            "year": 2024
        }
        
        result = generate_filename(metadata)
        
        assert result.startswith("levin_Machine_Learning_for_Hypothesis_Generation_2024_Doe.pdf")
    
    def test_generate_filename_missing_metadata(self):
        """Test filename generation with missing metadata."""
        metadata = {
            "title": "Test Paper",
            "authors": [],
            "year": None
        }
        
        result = generate_filename(metadata)
        
        assert result.startswith("levin_Test_Paper_unknown_Unknown.pdf")
    
    def test_generate_filename_special_characters(self):
        """Test filename generation with special characters in title."""
        metadata = {
            "title": "Paper with (brackets) & symbols!",
            "authors": ["John Doe"],
            "year": 2024
        }
        
        result = generate_filename(metadata)
        
        assert result.startswith("levin_Paper_with_brackets_symbols_2024_Doe.pdf")
    
    def test_generate_filename_long_title(self):
        """Test filename generation with very long title."""
        long_title = "A" * 100  # Very long title
        metadata = {
            "title": long_title,
            "authors": ["John Doe"],
            "year": 2024
        }
        
        result = generate_filename(metadata)
        
        # Should be truncated to 50 characters
        assert len(result.split('_')[1]) <= 50


class TestProcessPdf:
    """Test PDF processing workflow."""
    
    def test_process_pdf_success(self, tmp_path):
        """Test successful PDF processing."""
        # Setup test files
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")
        
        metadata = {
            "title": "Test Paper",
            "authors": ["John Doe"],
            "year": 2024
        }
        
        # Mock Document AI
        mock_document = Mock()
        mock_document.text = "Extracted text content"
        mock_result = Mock()
        mock_result.document = mock_document
        
        with patch('scripts.simple_pipeline.extract_text_with_document_ai', return_value="Extracted text content"):
            with patch('scripts.simple_pipeline.EXTRACTED_TEXTS_DIR', tmp_path / "outputs" / "extracted_texts"):
                result = process_pdf(pdf_path, metadata)
                
        assert result is True
        # Check that file was renamed
        new_files = list(tmp_path.glob("*.pdf"))
        assert len(new_files) == 1
        assert "levin_Test_Paper_2024_Doe.pdf" in str(new_files[0])
    
    def test_process_pdf_document_ai_failure(self, tmp_path):
        """Test PDF processing when Document AI fails."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")
        
        metadata = {
            "title": "Test Paper",
            "authors": ["John Doe"],
            "year": 2024
        }
        
        with patch('scripts.simple_pipeline.extract_text_with_document_ai', return_value=None):
            result = process_pdf(pdf_path, metadata)
            
        assert result is False
    
    def test_process_pdf_exception(self, tmp_path):
        """Test PDF processing when exception occurs."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")
        
        metadata = {
            "title": "Test Paper",
            "authors": ["John Doe"],
            "year": 2024
        }
        
        with patch('scripts.simple_pipeline.extract_text_with_document_ai', side_effect=Exception("Processing error")):
            result = process_pdf(pdf_path, metadata)
            
        assert result is False


# Fixtures for common test data
@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "title": "Test Research Paper",
        "authors": ["Michael Levin", "John Doe"],
        "year": 2024,
        "journal": "Nature",
        "is_levin_paper": True
    }


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return "This is a test PDF with some content for extraction."


# Integration tests
class TestIntegration:
    """Integration tests for the pipeline."""
    
    def test_full_pipeline_workflow(self, tmp_path, sample_metadata):
        """Test the complete pipeline workflow."""
        # This would be a more comprehensive integration test
        # that tests the entire workflow end-to-end
        pass 