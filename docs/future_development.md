# Future Development Notes

## 📊 Paper Database & Metadata Storage

### Current State
- PDFs are renamed using LLM extraction (costs tokens)
- No persistent storage of metadata
- No tracking of paper sources/links

### Proposed Improvements

#### 1. Paper Database Schema
Create a structured database (SQLite/JSON) to store:

```json
{
  "papers": [
    {
      "id": "unique_identifier",
      "title": "Paper Title",
      "authors": ["Author 1", "Author 2"],
      "year": 2024,
      "journal": "Journal Name",
      "topic": "bioelectric",
      "is_levin_paper": true,
      "filename": "levin_topic_year_journal_identifier.pdf",
      "download_url": "https://doi.org/...",
      "source": "unpaywall|manual|automated",
      "download_date": "2024-01-15",
      "file_size_mb": 2.3,
      "pages": 15,
      "doi": "10.1000/...",
      "abstract": "Paper abstract...",
      "keywords": ["keyword1", "keyword2"],
      "levin_contribution": "lead_author|co_author|subject_matter"
    }
  ]
}
```

#### 2. Benefits of Database Approach
- **Cost Savings**: No need to re-extract metadata with LLM
- **Reproducibility**: Track exact sources and download dates
- **Automation**: Enable automated paper discovery and download
- **Analytics**: Track research trends, collaboration networks
- **Backup**: Metadata survives file corruption/movement

#### 3. Implementation Strategy

##### Phase 1: Manual Database Creation
```python
# scripts/create_paper_database.py
def create_initial_database():
    """Create initial database from current PDFs"""
    # Extract metadata from existing PDFs
    # Store in SQLite/JSON database
    # Include download tracking info
```

##### Phase 2: Automated Paper Discovery
```python
# scripts/discover_papers.py
def discover_new_papers():
    """Automatically find new Levin papers"""
    # Use Unpaywall API
    # Search academic databases
    # Track new publications
```

##### Phase 3: Automated Download Pipeline
```python
# scripts/auto_download_papers.py
def download_new_papers():
    """Download newly discovered papers"""
    # Check if paper already exists
    # Download if new
    # Update database
    # Rename using stored metadata
```

#### 4. Database Location Options
- **SQLite**: `data/papers.db` (simple, portable)
- **JSON**: `data/papers.json` (human-readable)
- **PostgreSQL**: For production scaling

#### 5. Metadata Sources
- **DOI**: Primary identifier for papers
- **Unpaywall API**: Free access to paper metadata
- **CrossRef API**: Additional metadata
- **arXiv API**: For preprints
- **PubMed API**: For biomedical papers

#### 6. File Naming Strategy
Instead of LLM extraction, use database metadata:
```python
def generate_filename_from_database(paper_record):
    """Generate filename from database metadata"""
    author_last = paper_record['authors'][0].split()[-1].lower()
    topic = paper_record['topic']
    year = paper_record['year']
    journal = clean_journal_name(paper_record['journal'])
    identifier = generate_identifier(paper_record['title'])
    
    return f"{author_last}_{topic}_{year}_{journal}_{identifier}.pdf"
```

## 🔄 Automation Pipeline

### Paper Discovery Workflow
1. **Monitor**: Check for new Levin papers weekly
2. **Discover**: Use APIs to find new publications
3. **Download**: Automatically download new papers
4. **Process**: Extract text, generate embeddings
5. **Update**: Add to vector database
6. **Notify**: Alert when new papers are added

### Cost Optimization
- **Batch Processing**: Process multiple papers at once
- **Caching**: Store LLM responses for reuse
- **Selective Processing**: Only use LLM for new papers
- **Local Models**: Use local LLMs for simple tasks

## 📈 Scaling Considerations

### Local Development
- SQLite database
- Manual paper discovery
- Local vector storage

### Cloud Migration
- PostgreSQL database
- Automated discovery pipeline
- Cloud vector storage (Pinecone, Weaviate)
- Scheduled jobs for updates

### Production Features
- Web interface for paper management
- Email notifications for new papers
- Paper recommendation system
- Collaboration tracking
- Citation analysis

## 🛠️ Technical Debt

### Current Issues to Address
1. **Hardcoded paths**: Use config file for all paths
2. **Error handling**: Add retry logic for API calls
3. **Logging**: Add proper logging throughout
4. **Testing**: Add unit tests for core functions
5. **Documentation**: Add docstrings and examples

### Code Quality Improvements
1. **Type hints**: Add throughout codebase
2. **Configuration**: Centralize all settings
3. **Error recovery**: Handle edge cases gracefully
4. **Performance**: Optimize for large paper collections
5. **Security**: Secure API key handling

## 🎯 Next Steps Priority

1. **High Priority**
   - Create paper database schema
   - Implement metadata storage
   - Add download tracking

2. **Medium Priority**
   - Automated paper discovery
   - Batch processing pipeline
   - Error handling improvements

3. **Low Priority**
   - Web interface
   - Advanced analytics
   - Collaboration features 