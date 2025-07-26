# YouTube Video Processing Pipeline

This guide explains how to add YouTube videos to the Michael Levin RAG system.

## Overview

The YouTube pipeline processes Michael Levin's videos (presentations and interviews) to:
1. Download videos from YouTube
2. Generate transcripts with timestamps using AssemblyAI
3. Extract frames at regular intervals
4. Create semantic chunks with timestamps
5. Embed chunks into the vector database
6. Link responses to specific video timestamps with frame previews

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required for video processing)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### 3. Configure API Keys

Add your API keys to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

**Get AssemblyAI API Key:**
1. Sign up at https://www.assemblyai.com/
2. Get your API key from the dashboard
3. AssemblyAI offers 5 hours of free transcription per month

## Usage

### Step 1: Process YouTube Videos

1. Edit `scripts/youtube_pipeline.py` and add your YouTube URLs:

```python
youtube_urls = [
    "https://www.youtube.com/watch?v=example1",
    "https://www.youtube.com/watch?v=example2",
    # Add more URLs here
]
```

2. Run the YouTube processing pipeline:

```bash
python scripts/youtube_pipeline.py
```

This will:
- Download videos to `inputs/youtube_videos/`
- Upload videos to AssemblyAI for transcription
- Generate transcripts with timestamps
- Extract frames to `outputs/youtube_frames/`
- Save metadata to `outputs/youtube_transcripts/`

### Step 2: Integrate with Existing Index

Run the integration script to add YouTube chunks to the FAISS index:

```bash
python scripts/integrate_youtube_chunks.py
```

This will:
- Load existing FAISS index and chunks
- Embed YouTube chunks using OpenAI
- Add YouTube chunks to the index
- Update combined chunks with YouTube data

### Step 3: Test the App

Run the Streamlit app to test YouTube integration:

```bash
streamlit run app.py
```

## Features

### Video Citations
When the app references YouTube content, it will:
- Link directly to the specific timestamp in the video
- Display the video frame from that timestamp
- Show the video title and duration

### Frame Extraction
- Frames are extracted every 10 seconds (configurable)
- Frames are matched to semantic chunks
- Frames are displayed inline with citations

### Semantic Chunking
- Videos are chunked into 5-minute segments
- Each chunk gets multiple semantic topics using OpenAI:
  - **Primary Topic**: Main concept discussed
  - **Secondary Topics**: Additional concepts covered
  - **Combined Topic**: Natural combination of all topics
  - **Interdisciplinary Theme**: Broader connecting theme
- Chunks maintain timestamp information
- Optimized for Michael Levin's interdisciplinary work

### AssemblyAI Transcription
- High-quality transcription with speaker detection
- Word-level timestamps for precise linking
- Automatic punctuation and formatting
- Word boosting for scientific terms

## Configuration

Edit `configs/settings.py` to customize:

```python
# YouTube processing settings
TRANSCRIPTION_SERVICE = "assemblyai"  # "assemblyai" or "whisper"
FRAME_EXTRACTION_INTERVAL = 10  # Extract frame every N seconds
MAX_VIDEO_DURATION = 7200  # 2 hours max for processing
```

## Directory Structure

```
inputs/
├── youtube_videos/          # Downloaded videos
outputs/
├── youtube_transcripts/     # Video metadata and chunks
├── youtube_frames/         # Extracted video frames
├── youtube_metadata.json   # Processing status tracking
├── faiss_index.bin         # Updated vector index
└── combined_chunks.json    # Updated chunks with YouTube data
```

## Processing Steps and Metadata Tracking

The pipeline tracks each processing step to avoid reprocessing:

### Processing Steps:
1. **download**: Video downloaded from YouTube
2. **transcription**: AssemblyAI transcription completed
3. **frame_extraction**: Frames extracted from video
4. **semantic_chunking**: Chunks created with multi-topic analysis
5. **completed**: All processing steps finished
6. **faiss_integration**: Chunks added to vector database

### Metadata Structure:
```json
{
  "videos": {
    "VIDEO_ID": {
      "youtube_url": "https://www.youtube.com/watch?v=...",
      "processed_steps": ["download", "transcription", "frame_extraction", "semantic_chunking", "completed"],
      "step_data": {
        "download": { "video_path": "...", "duration": 3600 },
        "transcription": { "transcript_id": "...", "segments": [...] },
        "frame_extraction": { "frames": {...}, "frame_count": 360 },
        "semantic_chunking": { "chunks": [...], "chunk_count": 12 },
        "completed": { "processed_at": "...", "total_chunks": 12, "total_frames": 360 },
        "faiss_integration": { "integrated_at": "...", "chunk_count": 12 }
      },
      "last_updated": "2024-01-01T12:00:00"
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg using the instructions above
2. **AssemblyAI API key**: Make sure your API key is in the `.env` file
3. **Video too long**: Videos over 2 hours are skipped (configurable)
4. **API rate limits**: AssemblyAI has rate limits, but the script handles them
5. **Upload failures**: Check your internet connection and AssemblyAI service status

### Performance Tips

- AssemblyAI is much faster than Whisper (no GPU required)
- Videos are processed in the cloud, so local performance doesn't matter
- AssemblyAI offers 5 hours of free transcription per month
- Consider processing videos in batches to manage API usage

## Example Output

When a user asks about bioelectricity, the app might respond:

"In our work on bioelectric signaling, we found that cells can communicate through electrical signals [Source_1]"

Where [Source_1] links to a specific timestamp in a YouTube video with a frame preview showing Michael Levin discussing that topic.

## Adding New Videos

To add new videos:

1. Add YouTube URLs to `scripts/youtube_pipeline.py`
2. Run `python scripts/youtube_pipeline.py`
3. Run `python scripts/integrate_youtube_chunks.py`
4. Restart the Streamlit app

The new videos will be automatically integrated into the knowledge base.

## AssemblyAI Benefits

- **No GPU required**: Works perfectly on MacBook Air
- **High accuracy**: Better than Whisper for scientific content
- **Speaker detection**: Identifies different speakers in interviews
- **Word boosting**: Prioritizes scientific terms like "bioelectricity"
- **Fast processing**: Cloud-based, so no local computational load
- **Free tier**: 5 hours of transcription per month 