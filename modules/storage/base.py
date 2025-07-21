"""
Abstract base interface for storage providers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ExtractionJob:
    """Data structure for a PDF text extraction job."""
    paper_id: str
    pdf_path: Path
    page_images: List[Path]
    extraction_prompt: str
    llm_response: str
    extracted_text: str
    extraction_quality: str
    warnings: List[str]
    extracted_text_path: Optional[Path] = None
    status: str = "pending"  # pending, processing, completed, failed
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class StorageInterface(ABC):
    """
    Abstract interface for storing and retrieving PDF text extraction results.
    
    This interface allows easy swapping between different storage backends
    (JSON, SQLite, cloud storage, etc.) while maintaining a consistent API.
    """
    
    @abstractmethod
    def save_job(self, paper_id: str, job_data: Dict[str, Any]) -> bool:
        """
        Save an extraction job to storage.
        
        Args:
            paper_id: Unique identifier for the paper
            job_data: Dictionary containing job data
            
        Returns:
            True if saved successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_job(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an extraction job by paper ID.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Job data dictionary if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all extraction jobs.
        
        Returns:
            List of all job data dictionaries
        """
        pass
    
    @abstractmethod
    def update_job(self, paper_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an extraction job with new data.
        
        Args:
            paper_id: Unique identifier for the paper
            updates: Dictionary of fields to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_job(self, paper_id: str) -> bool:
        """
        Delete an extraction job.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def job_exists(self, paper_id: str) -> bool:
        """
        Check if an extraction job exists.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            True if job exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_jobs_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all jobs with a specific status.
        
        Args:
            status: Status to filter by (pending, processing, completed, failed)
            
        Returns:
            List of job data dictionaries with the specified status
        """
        pass 