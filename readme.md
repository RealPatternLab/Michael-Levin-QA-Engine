# Michael Levin QA Engine

A simple, streamlined pipeline for processing Michael Levin's research papers.

## What It Does

1. **Finds PDFs** in `inputs/raw_papers/`
2. **Extracts metadata** using OpenAI (title, authors, year, etc.)
3. **Renames PDFs** with descriptive names
4. **Extracts text** using Google Document AI
5. **Saves everything** in `outputs/` directory

## Quick Start

### 1. Setup Environment

Add these to your `.env` file:

```bash
# OpenAI (for metadata extraction)
OPENAI_API_KEY=your_openai_api_key

# Google Document AI (for text extraction)
GOOGLE_DOCUMENT_AI_PROJECT_ID=your_project_id
GOOGLE_DOCUMENT_AI_LOCATION=us
GOOGLE_DOCUMENT_AI_PROCESSOR_ID=your_processor_id
```

### 2. Add PDFs

Put your PDF files in `inputs/raw_papers/`:

```bash
mkdir -p inputs/raw_papers
cp your_paper.pdf inputs/raw_papers/
```

### 3. Run Pipeline

```bash
uv run python scripts/simple_pipeline.py
```

## What You Get

- **Renamed PDFs**: `levin_Descriptive_Title_2024_Author.pdf`
- **Extracted text**: `outputs/extracted_texts/levin_Descriptive_Title_2024_Author.txt`
- **Metadata tracking**: `outputs/metadata.json`

## Project Structure

```
levin-qa-engine/
├── inputs/
│   └── raw_papers/          # Put PDFs here
├── outputs/
│   ├── extracted_texts/      # Extracted text files
│   └── metadata.json         # Processing metadata
├── scripts/
│   └── simple_pipeline.py    # Main pipeline script
├── configs/
│   └── settings.py           # Configuration settings
└── .env                      # API keys
```

## How It Works

1. **Smart Detection**: Only processes new PDFs (skips already processed ones)
2. **Metadata Extraction**: Uses OpenAI to extract paper title, authors, year, etc.
3. **File Renaming**: Creates descriptive filenames like `levin_Machine_Learning_2024_Levin.pdf`
4. **Text Extraction**: Uses Google Document AI for high-quality text extraction
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

# Run pipeline
uv run python scripts/simple_pipeline.py

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=scripts --cov-report=html

# Check results
ls outputs/extracted_texts/
cat outputs/metadata.json
```

## Why This Approach

- ✅ **Simple**: One script, clear workflow
- ✅ **Reliable**: Uses proven APIs (OpenAI + Google Document AI)
- ✅ **Fast**: No complex abstractions or unnecessary processing
- ✅ **Iterative**: Easy to modify and extend
- ✅ **Organized**: Clean file structure and naming
- ✅ **Tested**: Comprehensive pytest coverage
- ✅ **Modern**: Uses `uv` for fast dependency management

## Next Steps

This creates the foundation. From here you can:
- Add vector embeddings
- Build a RAG system
- Create a query interface
- Scale to more papers

But first, let's get the basic pipeline working perfectly! 🚀

