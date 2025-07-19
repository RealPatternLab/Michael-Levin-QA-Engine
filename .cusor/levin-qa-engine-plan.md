# 🧠 Project Plan: Digital Twin of Michael Levin (Phase 1: Manual RAG)

## ✅ Goal

Build a local, working RAG system that can answer questions using a small set of Michael Levin's research papers. This should use:

- Python  
- **UV** for package management
- **Ruff** for code quality and formatting
- A vector database (FAISS or ChromaDB)  
- LangChain or LlamaIndex  
- A CLI or basic Streamlit UI  
- A modular structure for expansion  

**Learning Strategy**: Start locally for hands-on learning of RAG fundamentals, then migrate to GCP cloud for production scaling.

---

## 📁 Repo Structure (Cursor-Ready)

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

---

## 🧩 Step-by-Step Plan

### Step 0: Environment Setup

- Initialize project with UV: `uv init`
- Add dependencies with UV:

```bash
uv add pypdf langchain faiss-cpu openai sentence-transformers chromadb streamlit ruff
```

- Store API keys (e.g., OpenAI) in a `.env` file
- Use `uv run ruff check` and `uv run ruff format` for code quality

---

### Step 1: Manual Data Collection

- Download 3–5 papers from:
  - [Levin Lab Website](https://drmichaellevin.org/publications/)
  - [Tufts Faculty Page](https://facultyprofiles.tufts.edu/michael-levin-1/publications)
- Save PDFs in `data/raw_papers/`

---

### Step 2: Extract & Clean Text

- Use `pypdf` or `pdfminer.six` to extract text from each PDF  
- Clean up headers, footers, and page numbers manually if needed  
- Save cleaned `.txt` files to `data/processed_chunks/`

---

### Step 3: Chunk the Text

- Use `RecursiveCharacterTextSplitter` from LangChain  
- Recommended: ~500 token chunks with ~50 token overlap  
- Store each chunk as a dictionary:  
  ```json
  {"source": "filename", "chunk": "chunk_text"}
  ```

---

### Step 4: Generate Embeddings

- Use either:
  - `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) for local
  - `OpenAI embeddings` for API-based
- Store embeddings in a FAISS or ChromaDB index  
- Tag each with metadata (source filename, chunk ID, etc.)

---

### Step 5: Build RAG Query Pipeline

1. Embed the user's query  
2. Retrieve top-k similar chunks from the vector store  
3. Construct the prompt:

```
Context:
[chunk1]
[chunk2]
...

Question:
[user question]

Answer as Michael Levin would, based only on this context.
```

4. Call LLM (e.g., GPT-3.5/4 or local HuggingFace model)  
5. Return result

---

### Step 6: Interface

Start simple:

- Option 1: `run_query.py` CLI tool  
- Option 2: `streamlit_app.py` for a basic local web UI

---

### Step 7: Test + Tune

Try real questions like:
- "What does Michael Levin say about bioelectric fields?"
- "How does he define morphogenetic intelligence?"

Evaluate for:
- Faithfulness to source
- Style consistency
- Grounding (no hallucinations)

---

## 🔁 Future Milestones (Don't Do Yet)

| Milestone                        | Outcome                                  |
|----------------------------------|-------------------------------------------|
| **Local Learning Phase**         | Master RAG fundamentals locally          |
| **GCP Migration**                | Deploy to Cloud Run for production scaling |
| Automate PDF collection          | Use scraping or arXiv API                |
| Add YouTube transcripts          | Use `youtube-transcript-api` + Whisper  |
| Build Agent to Recreate Pipeline| Use LangChain Agents or AutoGen         |
| Add TTS or video                 | Explore multi-modal output               |
| Fine-tuning                      | Style-tune LLM on Levin's corpus        |

---

## 🎯 Learning & Scaling Strategy

### Phase 1: Local Learning (Current Focus)
- **Goal**: Master RAG fundamentals hands-on
- **Environment**: Local development with Python venv
- **Tools**: FAISS/ChromaDB, LangChain, local embeddings
- **Outcome**: Working prototype with deep understanding

### Phase 2: GCP Cloud Migration (Future)
- **Goal**: Scale for production and advanced features
- **Environment**: Google Cloud Platform
- **Services**: Cloud Run, Cloud Storage, Vertex AI
- **Benefits**: Auto-scaling, managed services, cost optimization

---

## 🧠 Improvements Inspired by Grok & Gemini

| Insight                          | Credit  | How I Incorporated It                               |
|----------------------------------|---------|------------------------------------------------------|
| Emphasis on cleaning PDF text    | Grok    | Structured chunk cleaning step                       |
| LLM prompt templating            | Gemini  | Added example format to prompt                       |
| Documenting rationale for tools  | Both    | Highlighted why FAISS, why LangChain                 |
| Evaluation methods               | Grok    | Suggested comparison to Levin's real interviews      |
| Tool modularity (future agents)  | Gemini  | Created `rag/` module layout for reuse and scaling   |
| Metadata tagging for source      | Both    | All vectors tagged with chunk ID + filename          |
