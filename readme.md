# levin-qa-engine

This project builds a Retrieval-Augmented Generation (RAG) system to simulate interacting with the ideas of researcher **Michael Levin**. Using a small set of manually collected research papers, we construct a local pipeline that can answer questions grounded in Levin's published work.

## 🎯 Goals

- Understand Michael Levin's core ideas through direct interaction
- Deepen ML engineering skills in:
  - RAG pipelines
  - Embeddings & vector databases
  - LangChain or LlamaIndex
  - Modular ML pipelines
- Build a foundation for later automation via agentic systems

## ✅ Phase 1 Scope

- Manually download 3–5 PDFs
- Extract and clean text
- Chunk text into semantic blocks
- Generate vector embeddings
- Store in FAISS or ChromaDB
- Query with LangChain RAG pipeline
- Deliver answers via CLI or Streamlit app

## 🧱 Stack

- Python 3.11+
- **UV** for package management
- **Ruff** for code quality and formatting
- FAISS or ChromaDB
- LangChain or LlamaIndex
- OpenAI or SentenceTransformers
- Streamlit (optional)
- GCP (planned in later phase)

## 📁 Project Structure

```
levin-qa-engine/
├── data/
│   ├── raw_papers/         # Store PDF files
│   └── processed_chunks/   # Optional cache of chunked text
├── scripts/
│   ├── extract_text.py     # PDF to text
│   ├── chunk_text.py       # Chunking logic
│   ├── embed_chunks.py     # Generate + store embeddings
│   └── run_query.py        # Main CLI app
├── rag/
│   ├── vector_store.py     # FAISS or ChromaDB abstraction
│   └── query_engine.py     # Retrieval + LLM prompt logic
├── notebooks/
│   └── exploration.ipynb   # Optional experiments
├── pyproject.toml          # UV dependencies
├── .env                    # API keys (OpenAI, etc.)
└── README.md
```

## 🚀 Development Setup

### Prerequisites
- Python 3.11+
- UV package manager
- Git

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd levin-qa-engine

# Switch to dev branch
git checkout dev

# Install dependencies (as needed)
uv add <package-name>

# Run scripts
uv run python scripts/script.py

# Lint and format code
uv run ruff check
uv run ruff format
```

## 🎯 Learning & Scaling Strategy

### Phase 1: Local Learning (Current Focus)
- **Goal**: Master RAG fundamentals hands-on
- **Environment**: Local development with UV
- **Tools**: FAISS/ChromaDB, LangChain, local embeddings
- **Outcome**: Working prototype with deep understanding

### Phase 2: GCP Cloud Migration (Future)
- **Goal**: Scale for production and advanced features
- **Environment**: Google Cloud Platform
- **Services**: Cloud Run, Cloud Storage, Vertex AI
- **Benefits**: Auto-scaling, managed services, cost optimization

## 📝 Development Rules

- **MUST use UV** for all Python package management
- **MUST use Ruff** for all linting and formatting
- Add dependencies incrementally as needed
- Follow PEP 8 standards (enforced by Ruff)
- Use type hints where appropriate
- Write docstrings for all functions and classes

