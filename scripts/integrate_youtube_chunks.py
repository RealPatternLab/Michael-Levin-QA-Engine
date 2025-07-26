#!/usr/bin/env python3
"""
Integrate YouTube Video Chunks with Existing FAISS Index

This script:
1. Loads existing FAISS index and combined chunks
2. Loads YouTube video chunks from metadata files
3. Embeds YouTube chunks using OpenAI
4. Adds YouTube chunks to FAISS index
5. Updates combined chunks with YouTube data
6. Saves updated index and chunks
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI
import faiss
from datetime import datetime

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

class YouTubeIntegrator:
    def __init__(self):
        """Initialize the YouTube integrator."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.index = None
        self.chunks = []
        
    def load_existing_data(self):
        """Load existing FAISS index and combined chunks."""
        try:
            # Load FAISS index
            index_path = Path("outputs/faiss_index.bin")
            if not index_path.exists():
                raise FileNotFoundError("FAISS index not found. Run the pipeline first.")
            
            self.index = faiss.read_index(str(index_path))
            logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load combined chunks
            chunks_path = Path("outputs/combined_chunks.json")
            if not chunks_path.exists():
                raise FileNotFoundError("Combined chunks not found. Run the pipeline first.")
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.chunks)} existing chunks")
            
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
            raise
    
    def load_youtube_chunks(self) -> List[Dict[str, Any]]:
        """Load YouTube chunks from metadata files."""
        youtube_chunks = []
        
        try:
            # Load YouTube metadata to check integration status
            youtube_metadata = {}
            if YOUTUBE_METADATA_FILE.exists():
                with open(YOUTUBE_METADATA_FILE, 'r') as f:
                    youtube_metadata = json.load(f)
            
            # Find all YouTube metadata files
            metadata_files = list(YOUTUBE_TRANSCRIPTS_DIR.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                video_info = metadata['video_info']
                youtube_url = video_info['youtube_url']
                video_id = self._get_video_id(youtube_url)
                
                # Check if already integrated
                if video_id in youtube_metadata.get("videos", {}):
                    if "faiss_integration" in youtube_metadata["videos"][video_id].get("processed_steps", []):
                        logger.info(f"Video already integrated into FAISS, skipping: {youtube_url}")
                        continue
                
                chunks = metadata['chunks']
                
                for chunk in chunks:
                    # Create chunk in the same format as paper chunks
                    youtube_chunk = {
                        'text': chunk['text'],
                        'source_title': video_info['video_title'],
                        'year': video_info['upload_date'][:4] if video_info['upload_date'] else 'Unknown',
                        'source_type': 'youtube_video',
                        'youtube_url': chunk['youtube_url'],
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'semantic_topics': chunk['semantic_topics'],  # Updated to handle multi-topic structure
                        'frame_path': chunk['frame_path'],
                        'source_file': str(metadata_file),
                        'section_header': f"Video Segment ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)",
                        'paper_metadata': {
                            'title': video_info['video_title'],
                            'authors': [video_info['channel']],
                            'year': video_info['upload_date'][:4] if video_info['upload_date'] else 'Unknown',
                            'url': video_info['youtube_url'],
                            'type': 'youtube_video'
                        }
                    }
                    youtube_chunks.append(youtube_chunk)
            
            logger.info(f"✅ Loaded {len(youtube_chunks)} YouTube chunks from {len(metadata_files)} videos")
            return youtube_chunks
            
        except Exception as e:
            logger.error(f"Failed to load YouTube chunks: {e}")
            return []
    
    def _get_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL."""
        import re
        # Handle various YouTube URL formats
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        # Fallback: use URL as ID
        return youtube_url
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            embedding = response.data[0].embedding
            return np.array(embedding).astype('float32').reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def embed_youtube_chunks(self, youtube_chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Embed YouTube chunks using OpenAI."""
        embeddings = []
        
        for i, chunk in enumerate(youtube_chunks):
            logger.info(f"Embedding YouTube chunk {i+1}/{len(youtube_chunks)}")
            
            # Create text for embedding (focus on semantic topics rather than full transcript)
            topics = chunk['semantic_topics']
            primary_topic = topics.get('primary_topic', '')
            secondary_topics = ' '.join(topics.get('secondary_topics', []))
            combined_topic = topics.get('combined_topic', primary_topic)
            interdisciplinary_theme = topics.get('interdisciplinary_theme', '')
            
            # Embed semantic topics + video context, not the full transcript text
            embed_text = f"{chunk['source_title']} {primary_topic} {secondary_topics} {combined_topic} {interdisciplinary_theme}"
            
            embedding = self.get_embedding(embed_text)
            embeddings.append(embedding)
        
        return np.vstack(embeddings)
    
    def add_to_faiss_index(self, embeddings: np.ndarray):
        """Add embeddings to FAISS index."""
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            logger.info(f"✅ Added {len(embeddings)} YouTube embeddings to FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to add to FAISS index: {e}")
            raise
    
    def save_updated_data(self, youtube_chunks: List[Dict[str, Any]]):
        """Save updated FAISS index and combined chunks."""
        try:
            # Save updated FAISS index
            index_path = Path("outputs/faiss_index.bin")
            faiss.write_index(self.index, str(index_path))
            logger.info(f"✅ Saved updated FAISS index with {self.index.ntotal} vectors")
            
            # Add YouTube chunks to combined chunks
            self.chunks.extend(youtube_chunks)
            
            # Save updated combined chunks
            chunks_path = Path("outputs/combined_chunks.json")
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved updated combined chunks with {len(self.chunks)} total chunks")
            
            # Mark videos as integrated in YouTube metadata
            self._mark_videos_integrated(youtube_chunks)
            
        except Exception as e:
            logger.error(f"Failed to save updated data: {e}")
            raise
    
    def _mark_videos_integrated(self, youtube_chunks: List[Dict[str, Any]]):
        """Mark videos as integrated in YouTube metadata."""
        try:
            if not YOUTUBE_METADATA_FILE.exists():
                logger.warning("YouTube metadata file not found, skipping integration marking")
                return
            
            with open(YOUTUBE_METADATA_FILE, 'r') as f:
                youtube_metadata = json.load(f)
            
            # Track which videos were integrated
            integrated_videos = set()
            for chunk in youtube_chunks:
                youtube_url = chunk.get('youtube_url', '')
                if youtube_url:
                    video_id = self._get_video_id(youtube_url)
                    integrated_videos.add(video_id)
            
            # Mark videos as integrated
            for video_id in integrated_videos:
                if video_id in youtube_metadata.get("videos", {}):
                    if "faiss_integration" not in youtube_metadata["videos"][video_id].get("processed_steps", []):
                        youtube_metadata["videos"][video_id]["processed_steps"].append("faiss_integration")
                        youtube_metadata["videos"][video_id]["step_data"]["faiss_integration"] = {
                            "integrated_at": datetime.now().isoformat(),
                            "chunk_count": len([c for c in youtube_chunks if c.get('youtube_url', '').startswith(f"https://www.youtube.com/watch?v={video_id}")])
                        }
                        youtube_metadata["videos"][video_id]["last_updated"] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(YOUTUBE_METADATA_FILE, 'w') as f:
                json.dump(youtube_metadata, f, indent=2)
            
            logger.info(f"✅ Marked {len(integrated_videos)} videos as integrated")
            
        except Exception as e:
            logger.error(f"Failed to mark videos as integrated: {e}")
    
    def integrate_youtube_data(self):
        """Main integration function."""
        try:
            # Load existing data
            self.load_existing_data()
            
            # Load YouTube chunks
            youtube_chunks = self.load_youtube_chunks()
            
            if not youtube_chunks:
                logger.warning("No YouTube chunks found. Run youtube_pipeline.py first.")
                return
            
            # Embed YouTube chunks
            embeddings = self.embed_youtube_chunks(youtube_chunks)
            
            # Add to FAISS index
            self.add_to_faiss_index(embeddings)
            
            # Save updated data
            self.save_updated_data(youtube_chunks)
            
            logger.info("✅ Successfully integrated YouTube data!")
            
        except Exception as e:
            logger.error(f"Failed to integrate YouTube data: {e}")
            raise

def main():
    """Main function."""
    integrator = YouTubeIntegrator()
    integrator.integrate_youtube_data()

if __name__ == "__main__":
    main() 