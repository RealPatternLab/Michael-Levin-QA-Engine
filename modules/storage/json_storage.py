"""
JSON implementation of storage interface.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .base import StorageInterface

logger = logging.getLogger(__name__)

class JSONStorage(StorageInterface):
    """
    JSON file-based storage for PDF text extraction results.
    
    Stores extraction jobs in a JSON file for simple, human-readable persistence.
    """
    
    def __init__(self, storage_file: str = "data/jobs_metadata/extraction_jobs.json"):
        """
        Initialize JSON storage.
        
        Args:
            storage_file: Path to the JSON file for storage
        """
        self.storage_file = Path(storage_file)
        self._ensure_storage_file()
    
    @property
    def name(self) -> str:
        """Get the name of this storage provider."""
        return "json"
    
    @property
    def description(self) -> str:
        """Get the description of this storage provider."""
        return "JSON file-based storage for extraction results"
    
    def _ensure_storage_file(self):
        """Ensure the storage file exists with proper structure."""
        if not self.storage_file.exists():
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_data({"jobs": {}})
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load {self.storage_file}, creating new storage")
            return {"jobs": {}}
    
    def _save_data(self, data: Dict[str, Any]):
        """Save data to JSON file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save data to {self.storage_file}: {e}")
            raise
    
    def save_job(self, paper_id: str, job_data: Dict[str, Any]) -> bool:
        """
        Save an extraction job to JSON storage.
        
        Args:
            paper_id: Unique identifier for the paper
            job_data: Dictionary containing job data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = self._load_data()
            
            # Add metadata
            job_data["paper_id"] = paper_id
            job_data["status"] = job_data.get("status", "completed")
            job_data["created_at"] = job_data.get("created_at", datetime.now().isoformat())
            job_data["updated_at"] = datetime.now().isoformat()
            
            data["jobs"][paper_id] = job_data
            self._save_data(data)
            
            logger.info(f"Saved extraction job for {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save extraction job for {paper_id}: {e}")
            return False
    
    def get_job(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an extraction job by paper ID.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Job data dictionary if found, None otherwise
        """
        try:
            data = self._load_data()
            return data["jobs"].get(paper_id)
        except Exception as e:
            logger.error(f"Failed to get extraction job for {paper_id}: {e}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all extraction jobs.
        
        Returns:
            List of all job data dictionaries
        """
        try:
            data = self._load_data()
            return list(data["jobs"].values())
        except Exception as e:
            logger.error(f"Failed to list extraction jobs: {e}")
            return []
    
    def update_job(self, paper_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an extraction job with new data.
        
        Args:
            paper_id: Unique identifier for the paper
            updates: Dictionary of fields to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            data = self._load_data()
            
            if paper_id not in data["jobs"]:
                logger.warning(f"Job {paper_id} not found for update")
                return False
            
            # Update existing job data
            data["jobs"][paper_id].update(updates)
            data["jobs"][paper_id]["updated_at"] = datetime.now().isoformat()
            
            self._save_data(data)
            logger.info(f"Updated extraction job for {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update extraction job for {paper_id}: {e}")
            return False
    
    def delete_job(self, paper_id: str) -> bool:
        """
        Delete an extraction job.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            data = self._load_data()
            
            if paper_id in data["jobs"]:
                del data["jobs"][paper_id]
                self._save_data(data)
                logger.info(f"Deleted extraction job for {paper_id}")
                return True
            else:
                logger.warning(f"Job {paper_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete extraction job for {paper_id}: {e}")
            return False
    
    def job_exists(self, paper_id: str) -> bool:
        """
        Check if an extraction job exists.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            True if job exists, False otherwise
        """
        try:
            data = self._load_data()
            return paper_id in data["jobs"]
        except Exception as e:
            logger.error(f"Failed to check if job exists for {paper_id}: {e}")
            return False
    
    def get_jobs_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all jobs with a specific status.
        
        Args:
            status: Status to filter by (pending, processing, completed, failed)
            
        Returns:
            List of job data dictionaries with the specified status
        """
        try:
            data = self._load_data()
            return [
                job for job in data["jobs"].values() 
                if job.get("status") == status
            ]
        except Exception as e:
            logger.error(f"Failed to get jobs by status {status}: {e}")
            return [] 