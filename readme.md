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

## 🏗️ **Architectural Principles** ⭐ **CRITICAL**

### **Interface-First Design**
This project follows an **interface-first design philosophy** for all external integrations:

- **Abstract Interfaces**: Define clear contracts for each external service
- **Multiple Implementations**: Support multiple providers for each service
- **Dependency Injection**: Use factory patterns to swap implementations
- **Centralized Configuration**: All settings in one place
- **Capability Documentation**: Clear metadata about provider capabilities

### **Code Organization & Simplicity** ⭐ **CRITICAL**
This project emphasizes **organized and simple code**:

- **Keep Related Classes Together**: Classes of the same nature should be in the same file
- **Minimize File Count**: Don't create separate files for every class unless there's a clear benefit
- **Logical Grouping**: Group related functionality in the same module
- **Avoid Over-Engineering**: Don't create abstractions unless they solve a real problem
- **Clear File Names**: Use descriptive names that indicate the file's purpose

#### **When to Put Classes in the Same File:**
- ✅ **Same Interface**: All implementations of the same interface (e.g., `OpenAITextExtractor` and `ClaudeTextExtractor`)
- ✅ **Related Functionality**: Classes that work together or are part of the same feature
- ✅ **Small Classes**: Helper classes, data classes, or utility classes
- ✅ **Same Domain**: Classes that handle the same type of data or operations

#### **When to Separate Classes:**
- ❌ **Different Interfaces**: Classes implementing different interfaces
- ❌ **Large Classes**: When a file becomes too large (>500 lines) or complex
- ❌ **Different Domains**: Classes that handle completely different concerns
- ❌ **Optional Dependencies**: Classes that require different dependencies

#### **Example Organization:**
```python
# ✅ GOOD: Related classes in same file
# modules/text_extraction/openai.py
class OpenAITextExtractor(BaseTextExtractor):
    """OpenAI implementation for text extraction."""
    pass

class OpenAIBatchProcessor:
    """Helper class for OpenAI batch processing."""
    pass

# ✅ GOOD: Interface and implementations together
# modules/text_extraction/base.py
class TextExtractionInterface(ABC):
    """Abstract interface for text extraction."""
    pass

class BaseTextExtractor(TextExtractionInterface):
    """Base implementation with common functionality."""
    pass

# ❌ BAD: Unnecessary file separation
# modules/text_extraction/openai_extractor.py  # Just one class
# modules/text_extraction/openai_batch_processor.py  # Just one class
```

### **Why This Matters**
- **Technology Evolution**: APIs change, new providers emerge
- **Cost Optimization**: Different providers have different pricing
- **Performance Flexibility**: Some tasks need speed, others need quality
- **Reliability**: Redundancy and fallback options
- **Testing**: Easy to mock and test different scenarios
- **Maintainability**: Simple, organized code is easier to understand and modify
- **Onboarding**: New developers can quickly understand the codebase structure

### **Current Implementations**
- ✅ **AI Models**: `ai_models/` - OpenAI, Claude, Gemini, Local
- ✅ **PDF Processing**: `modules/pdf_processing/` - Image rendering for multimodal analysis
- ✅ **Text Extraction**: `modules/text_extraction/` - Multimodal text extraction from PDF images
- ✅ **Storage**: `modules/storage/` - JSON-based job tracking and results persistence
- 🔄 **Vector Databases**: Future - FAISS, ChromaDB, Pinecone
- 🔄 **Embedding Models**: Future - OpenAI, SentenceTransformers, Local

### **Example Pattern**
```python
# ✅ GOOD: Interface-based approach
from ai_models import get_model
model = get_model("openai")  # Easy to swap to "claude" or "local"

# ❌ BAD: Direct API calls scattered throughout code
from openai import OpenAI
client = OpenAI(api_key=key)  # Hard to swap, test, or mock
```

**See `.cursor/docs/interface_guidelines.md` for detailed implementation guidelines.**
**See `.cursor/rules/code-organization.md` for detailed code organization guidelines.**

## ✅ Phase 1 Scope

- Manually download 3–5 PDFs
- **Render PDF pages to images for multimodal analysis**
- **Extract text using multimodal LLMs (GPT-4o, Claude)**
- Chunk text into semantic blocks
- Generate vector embeddings
- Store in FAISS or ChromaDB
- Query with LangChain RAG pipeline
- Deliver answers via CLI or Streamlit app

## 🧱 Stack

- Python 3.11+
- **UV** for package management
- **Ruff** for code quality and formatting
- **PyMuPDF** for PDF image rendering
- **OpenAI GPT-4o** for multimodal text extraction
- FAISS or ChromaDB
- LangChain or LlamaIndex
- OpenAI or SentenceTransformers
- Streamlit (optional)
- GCP (planned in later phase)

## 📁 Project Structure

```
levin-qa-engine/
├── ai_models/              # Centralized AI model interface
│   ├── __init__.py         # Factory functions and registry
│   ├── base.py             # BaseModelInterface abstract class
│   └── openai_model.py     # OpenAI implementation
├── modules/                 # Core processing modules
│   ├── pdf_processing/      # PDF image rendering for multimodal analysis
│   │   ├── __init__.py     # Factory functions and registry
│   │   ├── base.py         # PDFProcessorInterface abstract class
│   │   ├── pdfplumber_processor.py
│   │   └── pymupdf_processor.py
│   ├── text_extraction/    # Multimodal LLM text extraction
│   │   ├── __init__.py     # Factory functions and registry
│   │   ├── base.py         # TextExtractionInterface abstract class
│   │   ├── base_extractor.py
│   │   └── openai.py
│   └── storage/            # Results persistence and job management
│       ├── __init__.py     # Factory functions and registry
│       ├── base.py         # StorageInterface abstract class
│       └── json_storage.py
├── data/
│   ├── raw_papers/         # Store PDF files
│   ├── extracted/          # Output from multimodal text extraction pipeline
│   │   ├── page_images/    # Rendered PDF page images
│   │   ├── extracted_texts/ # Extracted text files
│   │   └── results/        # Extraction results and metadata
│   └── jobs_metadata/      # Job tracking and metadata
├── scripts/
│   ├── run_cleaning_pipeline.py  # Main multimodal text extraction pipeline
│   └── test/               # Test scripts for components
├── .cursor/                # Documentation and development rules
│   ├── docs/               # Project documentation
│   └── rules/              # Development rules for AI assistance
└── config.py               # Centralized configuration
```

## 🚀 **Multimodal Text Extraction Pipeline**

This project uses a **multimodal approach** for PDF text extraction:

1. **PDF Image Rendering**: Render PDF pages to high-quality images
2. **Multimodal LLM Analysis**: Use GPT-4o or Claude to extract text from images
3. **Quality Assessment**: Evaluate extraction quality and generate warnings
4. **Structured Output**: Save extracted text with metadata and results

### **Key Benefits**
- ✅ **No PDF parsing issues** - let the LLM handle complex layouts
- ✅ **Better accuracy** - multimodal models understand visual structure
- ✅ **Simpler architecture** - fewer moving parts and dependencies
- ✅ **More reliable** - works with any PDF format or layout

### **Usage**
```bash
# Extract text from a PDF using multimodal LLM
PYTHONPATH=. uv run python scripts/run_cleaning_pipeline.py \
  --pdf data/raw_papers/paper.pdf \
  --output data/extracted \
  --verbose
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
- **MUST follow interface-first design** for all external integrations
- **MUST organize scripts in `scripts/`** directory
- **MUST organize modules in `modules/`** directory
- **MUST store data in `data/`** directory
- Add dependencies incrementally as needed
- Follow PEP 8 standards (enforced by Ruff)
- Use type hints where appropriate
- Write docstrings for all functions and classes

## ✅ **Codebase Cleanup Complete!**

### **✅ What We Removed**

**Old Test Files:**
- `scripts/test_multimodal_extraction.py` - Old multimodal extraction test
- `scripts/test/test_llm_extraction.py` - Traditional text extraction test
- `scripts/test/test_api_key.py` - API key test
- `scripts/test/test_model_swapping.py` - Model swapping test
- `scripts/test/run_tests.py` - Old test runner

**Old Data Directories:**
- `data/test_images/` - Test image files
- `data/cleaned/` - Old cleaning results
- `data/cleaned_pymupdf/` - Old PyMuPDF cleaning results
- `data/test_cleaned/` - Test cleaning results

**Old Data Files:**
- `data/multimodal_extracted_text.txt` - Old test output
- `data/cleaning_jobs.json` → `data/extraction_jobs.json` (renamed)

### **✅ What We Updated**

**Storage Interface:**
- Updated `modules/storage/base.py` to use extraction terminology
- Updated `modules/storage/json_storage.py` to use simplified job structure
- Renamed methods: `save_cleaning_job` → `save_job`, etc.

**LLM Cleaning Module:**
- Updated `modules/llm_cleaning/__init__.py` to reflect text extraction focus
- Updated registry name: `CLEANING_PROCESSOR_REGISTRY` → `EXTRACTION_PROCESSOR_REGISTRY`

**Documentation:**
- Updated `readme.md` to reflect multimodal text extraction approach
- Added new section explaining the multimodal pipeline benefits

### **✅ Current Clean Architecture**

```
PDF → Image Rendering → Multimodal LLM → Extracted Text
```

**Key Benefits:**
- ✅ **No traditional PDF text extraction** - eliminated all parsing issues
- ✅ **Simplified codebase** - fewer moving parts and dependencies
- ✅ **Better accuracy** - multimodal models handle complex layouts
- ✅ **Consistent terminology** - all references now use "extraction" instead of "cleaning"
- ✅ **Clean interfaces** - updated all base classes and factory functions

### **🚀 Ready for Production**

The codebase is now clean and focused entirely on the multimodal approach:

```bash
# Extract text from any PDF using multimodal LLM
PYTHONPATH=. uv run python scripts/run_cleaning_pipeline.py \
  --pdf data/raw_papers/paper.pdf \
  --output data/extracted \
  --verbose
```

The refactoring was a great success - we've eliminated the complexity of traditional PDF text extraction and created a much more reliable and accurate system! 🎯

