# Repository Refactor Plan

## Current Issues
1. **Unclear input/output mapping**: `outputs/extracted_texts/` doesn't indicate source
2. **Mixed concerns**: Streamlit app mixed with processing logic
3. **No clear separation** between different input types
4. **Hard to scale** for multiple input sources

## Proposed New Structure

```
michael-levin-qa-engine/
├── inputs/
│   ├── papers/                    # Research papers (PDFs)
│   ├── videos/                    # Video content (future)
│   ├── interviews/                # Interview transcripts (future)
│   └── datasets/                  # Other datasets (future)
├── outputs/
│   ├── papers/                    # Processed paper outputs
│   │   ├── extracted_texts/
│   │   ├── semantic_chunks/
│   │   ├── embeddings/
│   │   └── metadata.json
│   ├── videos/                    # Processed video outputs (future)
│   ├── interviews/                # Processed interview outputs (future)
│   └── combined/                  # Combined outputs for RAG
│       ├── faiss_index.bin
│       ├── combined_chunks.json
│       └── metadata.json
├── webapp/                        # Streamlit application
│   ├── pages/                     # Multi-page app structure
│   │   ├── 01_Research_Search.py
│   │   └── 02_Chat.py
│   ├── components/                # Reusable components
│   │   ├── rag_engine.py
│   │   └── citation_processor.py
│   ├── utils/                     # Webapp utilities
│   │   └── api_keys.py
│   └── Home.py                    # Main app entry point
├── scripts/
│   ├── pipelines/                 # Processing pipelines
│   │   ├── paper_pipeline.py
│   │   ├── video_pipeline.py      # Future
│   │   └── interview_pipeline.py  # Future
│   ├── processors/                # Individual processors
│   │   ├── text_extraction.py
│   │   ├── semantic_chunking.py
│   │   └── embedding_generation.py
│   └── utils/                     # Shared utilities
│       ├── file_utils.py
│       └── metadata_utils.py
├── configs/
│   ├── settings.py
│   └── pipelines/                 # Pipeline configurations
│       ├── paper_config.py
│       └── webapp_config.py
└── tests/
    ├── test_pipelines/
    ├── test_processors/
    └── test_webapp/
```

## Benefits

1. **Clear Input/Output Mapping**: Each input type has its own output directory
2. **Modular Design**: Separate concerns (pipelines, processors, webapp)
3. **Scalable**: Easy to add new input types
4. **Organized Webapp**: Proper Streamlit multi-page structure
5. **Reusable Components**: Shared utilities and components

## Migration Steps

1. Create new directory structure
2. Move existing files to new locations
3. Update import paths
4. Refactor Streamlit app into multi-page structure
5. Update pipeline scripts to use new structure
6. Test everything works
7. Update documentation

## File Naming Convention

- Input files: `{type}_{descriptive_name}_{year}_{authors}.{ext}`
- Output directories: `outputs/{type}/`
- Processing steps: `{step}_{timestamp}.{ext}` 