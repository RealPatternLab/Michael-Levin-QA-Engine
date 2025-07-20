# Automation Strategy for Paper Collection

## 🎯 Future Automation Goals

Once our RAG pipeline is working locally, we'll want to automate:
1. **Paper discovery** - Find new Levin papers automatically
2. **PDF acquisition** - Download papers using Unpaywall
3. **Text processing** - Extract and chunk text automatically
4. **Vector updates** - Update embeddings with new content

## 🔍 Unpaywall Integration Options

### Option 1: Unpaywall API (Recommended)
- **API Endpoint**: `https://api.unpaywall.org/v2/{doi}?email=your@email.com`
- **Rate Limits**: 100,000 requests per day
- **Response**: JSON with PDF URLs and metadata
- **Cost**: Free for academic use

### Option 2: Browser Automation
- **Tool**: Selenium or Playwright
- **Process**: Navigate to publisher pages, trigger Unpaywall extension
- **Complexity**: Higher, requires browser automation
- **Reliability**: Lower, depends on UI changes

### Option 3: Hybrid Approach
- **Primary**: Use Unpaywall API for known DOIs
- **Fallback**: Browser automation for edge cases
- **Manual**: Human intervention for complex cases

## 📊 Unpaywall API Integration

### API Usage Example
```python
import requests

def get_paper_pdf_url(doi, email):
    """Get free PDF URL from Unpaywall API"""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Check for best PDF URL
        if data.get('best_oa_location'):
            return data['best_oa_location']['url_for_pdf']
        elif data.get('first_oa_location'):
            return data['first_oa_location']['url_for_pdf']
    
    return None
```

### Required Information
- **DOI**: Digital Object Identifier for each paper
- **Email**: Required for API access (academic email preferred)
- **Rate limiting**: Respect 100k requests/day limit

## 🤖 Agent-Based Automation

### LangChain Agent Strategy
```python
from langchain.agents import Tool, AgentExecutor
from langchain.tools import BaseTool

class UnpaywallTool(BaseTool):
    name = "unpaywall_pdf_finder"
    description = "Find free PDF URLs for academic papers using Unpaywall API"
    
    def _run(self, doi: str) -> str:
        # Implementation using Unpaywall API
        pass

# Agent can use this tool to find PDFs
tools = [UnpaywallTool()]
agent = AgentExecutor.from_agent_and_tools(llm, tools, verbose=True)
```

### Browser Automation with Selenium
```python
from selenium import webdriver
from selenium.webdriver.common.by import By

def find_pdf_with_browser(doi):
    """Use browser automation to find PDFs"""
    driver = webdriver.Chrome()
    
    # Navigate to publisher page
    driver.get(f"https://doi.org/{doi}")
    
    # Look for Unpaywall green tab
    green_tab = driver.find_element(By.CLASS_NAME, "unpaywall-tab")
    if green_tab:
        pdf_url = green_tab.get_attribute("href")
        return pdf_url
    
    driver.quit()
    return None
```

## 📋 Automation Pipeline Design

### Phase 1: Paper Discovery
1. **Monitor sources**:
   - Levin Lab website RSS/updates
   - PubMed alerts for "Levin M[author]"
   - Google Scholar alerts
   - arXiv preprints

2. **Extract metadata**:
   - Title, authors, DOI, journal
   - Publication date
   - Abstract (for relevance filtering)

### Phase 2: PDF Acquisition
1. **Check Unpaywall API** for each DOI
2. **Download PDF** if available
3. **Apply naming convention** automatically
4. **Store in `data/raw_papers/`**

### Phase 3: Text Processing
1. **Extract text** from new PDFs
2. **Clean and chunk** text
3. **Generate embeddings** for new chunks
4. **Update vector store**

### Phase 4: RAG Updates
1. **Reindex** with new content
2. **Test queries** for quality
3. **Monitor performance** improvements

## 🔧 Implementation Plan

### Immediate (Current Phase)
- [ ] Focus on manual collection for learning
- [ ] Build working RAG pipeline
- [ ] Document requirements for automation

### Short Term (Next Phase)
- [ ] Research Unpaywall API integration
- [ ] Build paper discovery monitoring
- [ ] Create automated PDF downloader
- [ ] Test with small batch of papers

### Long Term (Production)
- [ ] Full automation pipeline
- [ ] Agent-based paper discovery
- [ ] Continuous learning system
- [ ] Cloud deployment with auto-scaling

## 🛠️ Technical Requirements

### API Integration
- **Unpaywall API**: For PDF discovery
- **PubMed API**: For paper metadata
- **ArXiv API**: For preprints
- **Rate limiting**: Respect all API limits

### Browser Automation
- **Selenium/Playwright**: For complex cases
- **Headless mode**: For server deployment
- **Proxy rotation**: For rate limiting
- **Error handling**: Robust failure recovery

### Data Management
- **DOI tracking**: Database of processed papers
- **Version control**: Track paper updates
- **Quality metrics**: Monitor PDF quality
- **Backup strategy**: Preserve downloaded content

## 🎯 Success Metrics

### Automation Success
- **Coverage**: % of Levin papers successfully downloaded
- **Speed**: Time from discovery to RAG integration
- **Quality**: PDF text extraction success rate
- **Cost**: API usage and computational resources

### RAG Performance
- **Query accuracy**: Improvement with new papers
- **Response quality**: More comprehensive answers
- **Coverage breadth**: More topics covered
- **Temporal relevance**: Up-to-date information

## 📝 Notes for Implementation

### Legal Considerations
- **Respect terms of service** for all APIs
- **Academic use only** for Unpaywall
- **Rate limiting** compliance
- **Data retention** policies

### Technical Challenges
- **PDF format variations** across publishers
- **Text extraction quality** differences
- **DOI resolution** reliability
- **Network stability** for downloads

### Future Enhancements
- **Multi-author support** beyond Levin
- **Cross-referencing** with related papers
- **Citation network** analysis
- **Trend detection** in research topics 