#!/usr/bin/env python3
"""
YouTube Video Processing Pipeline

This script processes YouTube videos for the Michael Levin RAG system:
1. Download YouTube videos (presentations and interviews)
2. Generate transcripts with timestamps using AssemblyAI
3. Extract frames at regular intervals
4. Create semantic chunks with timestamps
5. Embed chunks into vector database
6. Link responses to specific video timestamps
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import threading
import cv2
import numpy as np
from dataclasses import dataclass
import yt_dlp
import requests
import time
from openai import OpenAI
import faiss

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

@dataclass
class VideoSegment:
    """Represents a segment of video with transcript and metadata."""
    start_time: float
    end_time: float
    text: str
    semantic_topics: Dict[str, Any]  # Changed from semantic_topic to semantic_topics
    frame_path: Optional[str] = None
    youtube_url: Optional[str] = None
    video_title: Optional[str] = None

class YouTubeProcessor:
    def __init__(self):
        """Initialize the YouTube processor."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.assemblyai_api_key = ASSEMBLYAI_API_KEY
        self.metadata = self.load_youtube_metadata()
        
        # Create directories
        YOUTUBE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        YOUTUBE_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        YOUTUBE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_youtube_metadata(self) -> Dict[str, Any]:
        """Load existing YouTube metadata or create new."""
        if YOUTUBE_METADATA_FILE.exists():
            try:
                with open(YOUTUBE_METADATA_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"videos": {}}
    
    def save_youtube_metadata(self):
        """Save YouTube metadata to file."""
        try:
            YOUTUBE_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(YOUTUBE_METADATA_FILE, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save YouTube metadata: {e}")
    
    def get_video_id(self, youtube_url: str) -> str:
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
    
    def is_video_processed(self, youtube_url: str, step: str) -> bool:
        """Check if a video has been processed through a specific step."""
        video_id = self.get_video_id(youtube_url)
        if video_id in self.metadata["videos"]:
            return step in self.metadata["videos"][video_id].get("processed_steps", [])
        return False
    
    def mark_video_processed(self, youtube_url: str, step: str, data: Dict[str, Any] = None):
        """Mark a video as processed through a specific step."""
        video_id = self.get_video_id(youtube_url)
        
        if video_id not in self.metadata["videos"]:
            self.metadata["videos"][video_id] = {
                "youtube_url": youtube_url,
                "processed_steps": [],
                "step_data": {},
                "last_updated": datetime.now().isoformat()
            }
        
        if step not in self.metadata["videos"][video_id]["processed_steps"]:
            self.metadata["videos"][video_id]["processed_steps"].append(step)
        
        if data:
            self.metadata["videos"][video_id]["step_data"][step] = data
        
        self.metadata["videos"][video_id]["last_updated"] = datetime.now().isoformat()
        self.save_youtube_metadata()
    
    def download_video(self, youtube_url: str) -> Optional[Dict[str, Any]]:
        """Download YouTube video and return metadata."""
        try:
            # Check if already downloaded
            if self.is_video_processed(youtube_url, "download"):
                logger.info(f"Video already downloaded, skipping: {youtube_url}")
                video_id = self.get_video_id(youtube_url)
                download_data = self.metadata["videos"][video_id]["step_data"].get("download", {})
                if download_data.get("video_path") and Path(download_data["video_path"]).exists():
                    return download_data
                else:
                    logger.warning(f"Video marked as downloaded but file not found, re-downloading: {youtube_url}")
            
            # Configure yt-dlp
            ydl_opts = {
                'format': 'best[height<=720]',  # Limit to 720p for processing
                'outtmpl': str(YOUTUBE_VIDEOS_DIR / '%(title)s.%(ext)s'),
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignoreerrors': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(youtube_url, download=False)
                
                # Check duration
                if info.get('duration', 0) > MAX_VIDEO_DURATION:
                    logger.warning(f"Video too long ({info['duration']}s), skipping: {youtube_url}")
                    return None
                
                # Download video
                ydl.download([youtube_url])
                
                # Find downloaded file
                video_title = info.get('title', 'Unknown')
                safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')
                
                video_files = list(YOUTUBE_VIDEOS_DIR.glob(f"{safe_title}.*"))
                if not video_files:
                    logger.error(f"Could not find downloaded video for: {youtube_url}")
                    return None
                
                video_path = video_files[0]
                
                video_metadata = {
                    'youtube_url': youtube_url,
                    'video_title': video_title,
                    'video_path': str(video_path),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', ''),
                    'channel': info.get('channel', ''),
                    'view_count': info.get('view_count', 0),
                    'processed_at': datetime.now().isoformat()
                }
                
                # Mark as downloaded
                self.mark_video_processed(youtube_url, "download", video_metadata)
                
                return video_metadata
                
        except Exception as e:
            logger.error(f"Failed to download video {youtube_url}: {e}")
            return None
    
    def upload_to_assemblyai(self, video_path: str) -> Optional[str]:
        """Upload video to AssemblyAI and return upload URL."""
        try:
            logger.info(f"Uploading video to AssemblyAI: {video_path}")
            
            headers = {
                "authorization": self.assemblyai_api_key,
                "content-type": "application/json"
            }
            
            # Get upload URL
            upload_url_response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers
            )
            upload_url = upload_url_response.json()["upload_url"]
            
            # Upload the video file
            with open(video_path, "rb") as f:
                response = requests.put(upload_url, data=f)
            
            if response.status_code == 200:
                logger.info("✅ Video uploaded to AssemblyAI successfully")
                return upload_url
            else:
                logger.error(f"Failed to upload video: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload video to AssemblyAI: {e}")
            return None
    
    def transcribe_with_assemblyai(self, upload_url: str, youtube_url: str) -> Optional[List[Dict[str, Any]]]:
        """Transcribe video using AssemblyAI with timestamps."""
        try:
            # Check if already transcribed
            if self.is_video_processed(youtube_url, "transcription"):
                logger.info(f"Video already transcribed, skipping: {youtube_url}")
                video_id = self.get_video_id(youtube_url)
                transcription_data = self.metadata["videos"][video_id]["step_data"].get("transcription", {})
                if transcription_data.get("segments"):
                    return transcription_data["segments"]
                else:
                    logger.warning(f"Video marked as transcribed but segments not found, re-transcribing: {youtube_url}")
            
            logger.info("Starting AssemblyAI transcription...")
            
            headers = {
                "authorization": self.assemblyai_api_key,
                "content-type": "application/json"
            }
            
            # Request transcription
            transcript_request = {
                "audio_url": upload_url,
                "word_boost": ["bioelectricity", "morphogenesis", "regeneration", "collective intelligence", "basal cognition"],
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True
            }
            
            transcript_response = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                json=transcript_request,
                headers=headers
            )
            
            if transcript_response.status_code != 200:
                logger.error(f"Failed to start transcription: {transcript_response.text}")
                return None
            
            transcript_id = transcript_response.json()["id"]
            logger.info(f"Transcription started with ID: {transcript_id}")
            
            # Poll for completion
            while True:
                polling_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers
                )
                
                if polling_response.status_code != 200:
                    logger.error(f"Failed to get transcription status: {polling_response.text}")
                    return None
                
                polling_result = polling_response.json()
                
                if polling_result["status"] == "completed":
                    logger.info("✅ Transcription completed")
                    segments = self._process_assemblyai_result(polling_result)
                    
                    # Mark as transcribed
                    self.mark_video_processed(youtube_url, "transcription", {
                        "transcript_id": transcript_id,
                        "segments": segments,
                        "processed_at": datetime.now().isoformat()
                    })
                    
                    return segments
                elif polling_result["status"] == "error":
                    logger.error(f"Transcription failed: {polling_result}")
                    return None
                else:
                    logger.info(f"Transcription status: {polling_result['status']}")
                    time.sleep(3)
                    
        except Exception as e:
            logger.error(f"Failed to transcribe with AssemblyAI: {e}")
            return None
    
    def _process_assemblyai_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process AssemblyAI transcription result into segments."""
        segments = []
        
        try:
            # Process utterances (speaker segments)
            for utterance in result.get("utterances", []):
                segments.append({
                    'start': utterance['start'] / 1000.0,  # Convert to seconds
                    'end': utterance['end'] / 1000.0,
                    'text': utterance['text'],
                    'speaker': utterance.get('speaker', 'A')
                })
            
            # If no utterances, fall back to words
            if not segments:
                words = result.get("words", [])
                current_segment = []
                current_start = 0
                
                for word in words:
                    if not current_segment:
                        current_start = word['start'] / 1000.0
                    
                    current_segment.append(word['text'])
                    
                    # Create segment every ~10 seconds or at punctuation
                    if (word['end'] / 1000.0 - current_start > 10) or word['text'].endswith(('.', '!', '?')):
                        segments.append({
                            'start': current_start,
                            'end': word['end'] / 1000.0,
                            'text': ' '.join(current_segment),
                            'speaker': 'A'
                        })
                        current_segment = []
                        current_start = word['end'] / 1000.0
                
                # Add final segment
                if current_segment:
                    segments.append({
                        'start': current_start,
                        'end': words[-1]['end'] / 1000.0 if words else current_start,
                        'text': ' '.join(current_segment),
                        'speaker': 'A'
                    })
            
            logger.info(f"Processed {len(segments)} segments from AssemblyAI")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to process AssemblyAI result: {e}")
            return []
    
    def extract_frames(self, video_path: str, youtube_url: str, interval: int = FRAME_EXTRACTION_INTERVAL) -> Dict[float, str]:
        """Extract frames from video at regular intervals."""
        try:
            # Check if already extracted
            if self.is_video_processed(youtube_url, "frame_extraction"):
                logger.info(f"Frames already extracted, skipping: {youtube_url}")
                video_id = self.get_video_id(youtube_url)
                frame_data = self.metadata["videos"][video_id]["step_data"].get("frame_extraction", {})
                if frame_data.get("frames"):
                    return frame_data["frames"]
                else:
                    logger.warning(f"Video marked as frame-extracted but frames not found, re-extracting: {youtube_url}")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            frames = {}
            frame_interval = int(fps * interval)
            
            for frame_num in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_num / fps
                    frame_filename = f"frame_{timestamp:.1f}.jpg"
                    frame_path = YOUTUBE_FRAMES_DIR / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    frames[timestamp] = str(frame_path)
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # Mark as frame-extracted
            self.mark_video_processed(youtube_url, "frame_extraction", {
                "frames": frames,
                "frame_count": len(frames),
                "processed_at": datetime.now().isoformat()
            })
            
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return {}
    
    def create_semantic_chunks(self, segments: List[Dict[str, Any]], video_metadata: Dict[str, Any]) -> List[VideoSegment]:
        """Create semantic chunks from transcript segments."""
        try:
            youtube_url = video_metadata['youtube_url']
            
            # Check if already chunked
            if self.is_video_processed(youtube_url, "semantic_chunking"):
                logger.info(f"Semantic chunks already created, skipping: {youtube_url}")
                video_id = self.get_video_id(youtube_url)
                chunk_data = self.metadata["videos"][video_id]["step_data"].get("semantic_chunking", {})
                if chunk_data.get("chunks"):
                    # Convert back to VideoSegment objects
                    chunks = []
                    for chunk_dict in chunk_data["chunks"]:
                        chunks.append(VideoSegment(
                            start_time=chunk_dict['start_time'],
                            end_time=chunk_dict['end_time'],
                            text=chunk_dict['text'],
                            semantic_topics=chunk_dict['semantic_topics'],
                            youtube_url=chunk_dict['youtube_url'],
                            video_title=chunk_dict['video_title']
                        ))
                    return chunks
                else:
                    logger.warning(f"Video marked as chunked but chunks not found, re-chunking: {youtube_url}")
            
            # Combine segments into larger chunks for semantic analysis
            chunk_size = 300  # seconds
            chunks = []
            
            current_chunk = []
            current_start = 0
            
            for segment in segments:
                if segment['end'] - current_start > chunk_size and current_chunk:
                    # Process current chunk
                    chunk_text = " ".join([s['text'] for s in current_chunk])
                    semantic_topics = self._extract_semantic_topics(chunk_text, video_metadata)
                    
                    chunks.append(VideoSegment(
                        start_time=current_start,
                        end_time=current_chunk[-1]['end'],
                        text=chunk_text,
                        semantic_topics=semantic_topics,
                        youtube_url=video_metadata['youtube_url'],
                        video_title=video_metadata['video_title']
                    ))
                    
                    # Start new chunk
                    current_chunk = [segment]
                    current_start = segment['start']
                else:
                    current_chunk.append(segment)
            
            # Process final chunk
            if current_chunk:
                chunk_text = " ".join([s['text'] for s in current_chunk])
                semantic_topics = self._extract_semantic_topics(chunk_text, video_metadata)
                
                chunks.append(VideoSegment(
                    start_time=current_start,
                    end_time=current_chunk[-1]['end'],
                    text=chunk_text,
                    semantic_topics=semantic_topics,
                    youtube_url=video_metadata['youtube_url'],
                    video_title=video_metadata['video_title']
                ))
            
            logger.info(f"Created {len(chunks)} semantic chunks")
            
            # Mark as chunked
            chunk_data = [
                {
                    'start_time': chunk.start_time,
                    'end_time': chunk.end_time,
                    'text': chunk.text,
                    'semantic_topics': chunk.semantic_topics,
                    'youtube_url': chunk.youtube_url,
                    'video_title': chunk.video_title
                }
                for chunk in chunks
            ]
            
            self.mark_video_processed(youtube_url, "semantic_chunking", {
                "chunks": chunk_data,
                "chunk_count": len(chunks),
                "processed_at": datetime.now().isoformat()
            })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to create semantic chunks: {e}")
            return []
    
    def _extract_semantic_topics(self, text: str, video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic topics from chunk text using OpenAI with multi-topic support."""
        try:
            prompt = f"""Analyze this transcript segment from a Michael Levin video and identify the main topics discussed.

Video: {video_metadata.get('video_title', 'Unknown')}
Text: {text[:800]}...

Michael Levin's work is highly interdisciplinary and often covers multiple concepts simultaneously. Identify up to 3 main topics, ranked by importance.

Return as JSON:
{{
    "primary_topic": "main concept (2-4 words)",
    "secondary_topics": ["concept2", "concept3"],
    "combined_topic": "natural combination of all topics (4-6 words)",
    "interdisciplinary_theme": "broader theme connecting the topics"
}}

Focus on Levin's key areas: bioelectricity, morphogenesis, collective intelligence, basal cognition, regeneration, synthetic biology, scale-free cognition, unconventional substrates for intelligence."""

            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                import json
                topics = json.loads(response_text)
                
                # Validate and provide defaults
                topics = {
                    'primary_topic': topics.get('primary_topic', 'General Discussion'),
                    'secondary_topics': topics.get('secondary_topics', []),
                    'combined_topic': topics.get('combined_topic', topics.get('primary_topic', 'General Discussion')),
                    'interdisciplinary_theme': topics.get('interdisciplinary_theme', 'Interdisciplinary Research')
                }
                
                return topics
                
            except json.JSONDecodeError:
                # Fallback to simple topic extraction
                logger.warning(f"Failed to parse JSON response, using fallback: {response_text}")
                return {
                    'primary_topic': response_text.strip(),
                    'secondary_topics': [],
                    'combined_topic': response_text.strip(),
                    'interdisciplinary_theme': 'Interdisciplinary Research'
                }
            
        except Exception as e:
            logger.error(f"Failed to extract semantic topics: {e}")
            return {
                'primary_topic': 'General Discussion',
                'secondary_topics': [],
                'combined_topic': 'General Discussion',
                'interdisciplinary_theme': 'Interdisciplinary Research'
            }
    
    def add_frames_to_chunks(self, chunks: List[VideoSegment], frames: Dict[float, str]) -> List[VideoSegment]:
        """Add frame paths to chunks based on timing."""
        for chunk in chunks:
            # Find the closest frame to the middle of the chunk
            chunk_middle = (chunk.start_time + chunk.end_time) / 2
            closest_frame_time = min(frames.keys(), key=lambda x: abs(x - chunk_middle))
            
            if abs(closest_frame_time - chunk_middle) <= FRAME_EXTRACTION_INTERVAL:
                chunk.frame_path = frames[closest_frame_time]
        
        return chunks
    
    def save_video_metadata(self, video_metadata: Dict[str, Any], chunks: List[VideoSegment]):
        """Save video metadata and chunks to JSON."""
        try:
            # Create metadata structure
            metadata = {
                'video_info': video_metadata,
                'chunks': [
                    {
                        'start_time': chunk.start_time,
                        'end_time': chunk.end_time,
                        'text': chunk.text,
                        'semantic_topics': chunk.semantic_topics,
                        'frame_path': chunk.frame_path,
                        'youtube_url': chunk.youtube_url,
                        'video_title': chunk.video_title
                    }
                    for chunk in chunks
                ],
                'processed_at': datetime.now().isoformat()
            }
            
            # Save to file
            safe_title = "".join(c for c in video_metadata['video_title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            metadata_path = YOUTUBE_TRANSCRIPTS_DIR / f"{safe_title}_metadata.json"
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save video metadata: {e}")
    
    def process_video(self, youtube_url: str) -> bool:
        """Process a single YouTube video end-to-end."""
        try:
            logger.info(f"Processing video: {youtube_url}")
            
            # Check if already fully processed
            if self.is_video_processed(youtube_url, "completed"):
                logger.info(f"Video already fully processed, skipping: {youtube_url}")
                return True
            
            # Download video
            video_metadata = self.download_video(youtube_url)
            if not video_metadata:
                return False
            
            # Upload to AssemblyAI
            upload_url = self.upload_to_assemblyai(video_metadata['video_path'])
            if not upload_url:
                return False
            
            # Transcribe video
            segments = self.transcribe_with_assemblyai(upload_url, youtube_url)
            if not segments:
                return False
            
            # Extract frames
            frames = self.extract_frames(video_metadata['video_path'], youtube_url)
            
            # Create semantic chunks
            chunks = self.create_semantic_chunks(segments, video_metadata)
            
            # Add frames to chunks
            chunks = self.add_frames_to_chunks(chunks, frames)
            
            # Save metadata
            self.save_video_metadata(video_metadata, chunks)
            
            # Mark as completed
            self.mark_video_processed(youtube_url, "completed", {
                "processed_at": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "total_frames": len(frames)
            })
            
            logger.info(f"✅ Successfully processed video: {video_metadata['video_title']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process video {youtube_url}: {e}")
            return False

def main():
    """Main function to process YouTube videos."""
    processor = YouTubeProcessor()
    
    # Example YouTube URLs (you can add more)
    youtube_urls = [
        # Add your Michael Levin video URLs here
        # "https://www.youtube.com/watch?v=example1",
        # "https://www.youtube.com/watch?v=example2",
    ]
    
    if not youtube_urls:
        logger.info("No YouTube URLs provided. Add URLs to the youtube_urls list in main().")
        return
    
    for url in youtube_urls:
        processor.process_video(url)

if __name__ == "__main__":
    main() 