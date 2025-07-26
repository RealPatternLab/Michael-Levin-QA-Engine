# Michael Levin QA Engine

A simple, streamlined pipeline for processing Michael Levin's research papers and YouTube videos.

## What It Does

### Paper Processing:
1. **Finds PDFs** in `inputs/raw_papers/`
2. **Extracts metadata** using OpenAI (title, authors, year, etc.)
3. **Renames PDFs** with descriptive names
4. **Extracts text** using Gemini 1.5 Pro (direct PDF processing)
5. **Saves everything** in `outputs/` directory

### YouTube Video Processing:
1. **Downloads videos** from YouTube using yt-dlp
2. **Transcribes videos** using AssemblyAI with timestamps
3. **Extracts frames** at regular intervals for visual context
4. **Creates semantic chunks** with multi-topic analysis
5. **Integrates with existing** paper-based knowledge base

## Quick Start

### 1. Setup Environment

Add these to your `.env` file:

```bash
# OpenAI (for metadata extraction and embeddings)
OPENAI_API_KEY=your_openai_api_key

# Google (for Gemini 1.5 Pro)
GOOGLE_API_KEY=your_google_api_key

# AssemblyAI (for video transcription)
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
```

### 2. Install Dependencies

```bash
# Install dependencies with uv
uv sync

# Install test dependencies (optional)
uv sync --extra test
```

### 3. Install FFmpeg (Required for video processing)

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

### 4. Add Content

**For Papers:**
```bash
mkdir -p inputs/raw_papers
cp your_paper.pdf inputs/raw_papers/
```

**For YouTube Videos:**
Edit `inputs/youtube_videos/youtube_urls.txt` and add YouTube URLs:
```
# Add your Michael Levin video URLs below:
https://www.youtube.com/watch?v=your_video_id_1
https://www.youtube.com/watch?v=your_video_id_2
```

### 5. Run Pipeline

**For Papers:**
```bash
uv run python scripts/simple_pipeline.py
```

**For YouTube Videos:**
```bash
uv run python scripts/youtube_pipeline.py
uv run python scripts/integrate_youtube_chunks.py
```

### 6. Start the App

```bash
uv run streamlit run app.py
```

## What You Get

- **Renamed PDFs**: `levin_Descriptive_Title_2024_Author.pdf`
- **Extracted text**: `outputs/extracted_texts/levin_Descriptive_Title_2024_Author.txt`
- **Metadata tracking**: `outputs/metadata.json`

## Project Structure

```
levin-qa-engine/
├── inputs/
│   ├── raw_papers/          # Put PDFs here
│   └── youtube_videos/      # Downloaded videos (auto-created)
├── outputs/
│   ├── extracted_texts/      # Extracted text files
│   ├── youtube_transcripts/  # Video metadata and chunks
│   ├── youtube_frames/      # Extracted video frames
│   ├── youtube_metadata.json # Processing status tracking
│   ├── faiss_index.bin      # Vector database
│   ├── combined_chunks.json # Combined paper + video chunks
│   └── metadata.json        # Paper processing metadata
├── scripts/
│   ├── simple_pipeline.py    # Main paper pipeline script
│   ├── youtube_pipeline.py   # YouTube processing script
│   └── integrate_youtube_chunks.py # Integration script
├── configs/
│   └── settings.py           # Configuration settings
└── .env                      # API keys
```

## How It Works

1. **Smart Detection**: Only processes new PDFs (skips already processed ones)
2. **Metadata Extraction**: Uses OpenAI to extract paper title, authors, year, etc.
3. **File Renaming**: Creates descriptive filenames like `levin_Machine_Learning_2024_Levin.pdf`
4. **Text Extraction**: Uses Gemini 1.5 Pro for direct PDF processing in 10-page chunks
5. **Simple Storage**: Everything saved in plain text/JSON files

## Example Output

**Input**: `paper.pdf`
**Output**: 
- `levin_Machine_Learning_for_Hypothesis_Generation_2024_Levin.pdf`
- `outputs/extracted_texts/levin_Machine_Learning_for_Hypothesis_Generation_2024_Levin.txt`

## Development

```bash
# Install dependencies
uv sync

# Install test dependencies
uv sync --extra test

# Run paper pipeline
uv run python scripts/simple_pipeline.py

# Run YouTube pipeline
uv run python scripts/youtube_pipeline.py
uv run python scripts/integrate_youtube_chunks.py

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=scripts --cov-report=html

# Check results
ls outputs/extracted_texts/
ls outputs/youtube_transcripts/
cat outputs/metadata.json
cat outputs/youtube_metadata.json
```

## Why This Approach

- ✅ **Simple**: One script, clear workflow
- ✅ **Reliable**: Uses proven APIs (OpenAI + Gemini 1.5 Pro)
- ✅ **Fast**: No complex abstractions or unnecessary processing
- ✅ **Iterative**: Easy to modify and extend
- ✅ **Organized**: Clean file structure and naming
- ✅ **Tested**: Comprehensive pytest coverage
- ✅ **Modern**: Uses `uv` for fast dependency management
- ✅ **Direct**: Gemini 1.5 Pro processes PDFs directly without intermediate steps

## Next Steps

This creates the foundation. From here you can:
- Add vector embeddings
- Build a RAG system
- Create a query interface
- Scale to more papers

But first, let's get the basic pipeline working perfectly! 🚀

## Future Development Ideas

### Multi-Topic Analysis for Papers
Currently, YouTube videos use multi-topic analysis while papers use single-topic analysis. This distinction was made because:

- **Papers**: Structured, focused sections with clear single purposes
- **Videos**: Conversational, interdisciplinary discussions covering multiple concepts

**Potential Enhancement**: Consider implementing conditional multi-topic analysis for papers based on section type:
- **Multi-topic sections**: Discussion, Introduction, Abstract (interdisciplinary content)
- **Single-topic sections**: Methods, Results (focused content)

This could improve retrieval accuracy for papers that bridge multiple concepts while maintaining simplicity for focused sections.

### Additional Future Enhancements
- **Enhanced metadata extraction**: Better author identification and affiliation parsing
- **Citation analysis**: Extract and link referenced papers
- **Figure and table extraction**: Include captions and descriptions
- **Temporal analysis**: Track how concepts evolve over time in Levin's work

