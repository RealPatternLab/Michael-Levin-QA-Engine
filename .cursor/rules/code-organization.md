# Code Organization Rules

## 🎯 **Core Principle: Keep It Simple and Organized**

This project emphasizes **organized and simple code** over complex abstractions. The goal is to create a codebase that is easy to understand, maintain, and extend.

## 📁 **File Organization Rules**

### **When to Put Classes in the Same File**

✅ **DO** put classes in the same file when they are:

1. **Same Interface**: All implementations of the same interface
   ```python
   # modules/text_extraction/openai.py
   class OpenAITextExtractor(BaseTextExtractor):
       """OpenAI implementation for text extraction."""
       pass
   
   class ClaudeTextExtractor(BaseTextExtractor):
       """Claude implementation for text extraction."""
       pass
   ```

2. **Related Functionality**: Classes that work together or are part of the same feature
   ```python
   # modules/storage/json_storage.py
   class JSONStorage(StorageInterface):
       """JSON file-based storage implementation."""
       pass
   
   class JSONStorageHelper:
       """Helper utilities for JSON storage operations."""
       pass
   ```

3. **Small Classes**: Helper classes, data classes, or utility classes
   ```python
   # modules/text_extraction/base.py
   @dataclass
   class ExtractedTextResult:
       """Data structure for extracted text results."""
       pass
   
   class TextExtractionInterface(ABC):
       """Abstract interface for text extraction."""
       pass
   
   class BaseTextExtractor(TextExtractionInterface):
       """Base implementation with common functionality."""
       pass
   ```

4. **Same Domain**: Classes that handle the same type of data or operations
   ```python
   # modules/pdf_processing/pymupdf_processor.py
   class PyMuPDFProcessor(PDFProcessorInterface):
       """PyMuPDF-based PDF processor."""
       pass
   
   class PyMuPDFHelper:
       """Helper utilities for PyMuPDF operations."""
       pass
   ```

### **When to Separate Classes**

❌ **DON'T** put classes in the same file when they are:

1. **Different Interfaces**: Classes implementing different interfaces
   ```python
   # ❌ BAD: Different interfaces in same file
   class PDFProcessorInterface(ABC):
       pass
   
   class StorageInterface(ABC):
       pass
   
   # ✅ GOOD: Separate files
   # modules/pdf_processing/base.py
   class PDFProcessorInterface(ABC):
       pass
   
   # modules/storage/base.py
   class StorageInterface(ABC):
       pass
   ```

2. **Large Classes**: When a file becomes too large (>500 lines) or complex
   ```python
   # ❌ BAD: Too many classes in one file
   class OpenAITextExtractor(BaseTextExtractor):
       # 200 lines of code
       pass
   
   class ClaudeTextExtractor(BaseTextExtractor):
       # 200 lines of code
       pass
   
   class GeminiTextExtractor(BaseTextExtractor):
       # 200 lines of code
       pass
   
   # ✅ GOOD: Separate files for large implementations
   # modules/text_extraction/openai.py
   class OpenAITextExtractor(BaseTextExtractor):
       pass
   
   # modules/text_extraction/claude.py
   class ClaudeTextExtractor(BaseTextExtractor):
       pass
   ```

3. **Different Domains**: Classes that handle completely different concerns
   ```python
   # ❌ BAD: Different domains in same file
   class PDFProcessor:
       """PDF processing functionality."""
       pass
   
   class EmailSender:
       """Email sending functionality."""
       pass
   
   # ✅ GOOD: Separate files for different domains
   # modules/pdf_processing/processor.py
   class PDFProcessor:
       pass
   
   # modules/notifications/email.py
   class EmailSender:
       pass
   ```

4. **Optional Dependencies**: Classes that require different dependencies
   ```python
   # ❌ BAD: Different dependencies in same file
   class OpenAITextExtractor:
       # Requires openai package
       pass
   
   class LocalTextExtractor:
       # Requires torch package
       pass
   
   # ✅ GOOD: Separate files for different dependencies
   # modules/text_extraction/openai.py
   class OpenAITextExtractor:
       pass
   
   # modules/text_extraction/local.py
   class LocalTextExtractor:
       pass
   ```

## 🏗️ **Module Structure Guidelines**

### **Interface-First Design**
- Put interfaces and base classes in `base.py` files
- Keep implementations in separate files when they're large or have different dependencies
- Use factory patterns in `__init__.py` files

### **Example Module Structure**
```
modules/text_extraction/
├── __init__.py          # Factory functions and registry
├── base.py              # Interface and base implementation
├── openai.py            # OpenAI implementation (with related helpers)
├── claude.py            # Claude implementation (with related helpers)
└── local.py             # Local implementation (with related helpers)
```

### **File Naming Conventions**
- Use descriptive names that indicate the file's purpose
- Use lowercase with underscores for file names
- Use PascalCase for class names
- Use descriptive names for modules

```python
# ✅ GOOD: Clear, descriptive names
modules/text_extraction/openai.py
modules/pdf_processing/pymupdf_processor.py
modules/storage/json_storage.py

# ❌ BAD: Unclear or generic names
modules/extraction/openai.py
modules/processing/processor.py
modules/storage/storage.py
```

## 🔧 **Implementation Guidelines**

### **Keep It Simple**
- Don't create abstractions unless they solve a real problem
- Prefer composition over inheritance when possible
- Use clear, descriptive names for classes and methods
- Write docstrings for all classes and methods

### **Avoid Over-Engineering**
```python
# ❌ BAD: Over-engineered
class AbstractTextExtractionStrategy(ABC):
    @abstractmethod
    def extract_text(self, images: List[Path]) -> str:
        pass

class OpenAIStrategy(AbstractTextExtractionStrategy):
    def extract_text(self, images: List[Path]) -> str:
        # Implementation
        pass

class StrategyFactory:
    def create_strategy(self, strategy_type: str) -> AbstractTextExtractionStrategy:
        # Factory implementation
        pass

# ✅ GOOD: Simple and direct
class OpenAITextExtractor(BaseTextExtractor):
    def extract_text_from_images(self, page_images: List[Path]) -> str:
        # Direct implementation
        pass
```

### **Use Logical Grouping**
- Group related functionality in the same module
- Keep related classes in the same file
- Use clear module boundaries

## 📋 **Code Review Checklist**

When reviewing code organization, ask:

- [ ] Are related classes in the same file?
- [ ] Are unrelated classes separated?
- [ ] Are file names descriptive and clear?
- [ ] Is the module structure logical?
- [ ] Are there unnecessary abstractions?
- [ ] Is the code easy to understand?
- [ ] Are dependencies clearly separated?
- [ ] Is the file size reasonable (<500 lines)?

## 🎯 **Summary**

**Remember**: The goal is **simple, organized, and maintainable code**. Don't over-engineer, and keep related functionality together. When in doubt, prefer simplicity over complexity. 