# Development Rules

## 🐍 Python Development Standards

### Package Management
- **MUST use UV** for all Python package management
- No pip, conda, or other package managers allowed
- Use `uv add` for adding dependencies
- Use `uv run` for running scripts
- Use `uv sync` for installing dependencies

### Code Quality & Formatting
- **MUST use Ruff** for all linting and formatting
- No black, flake8, isort, or other linters/formatters allowed
- Use `uv run ruff check` for linting
- Use `uv run ruff format` for formatting
- Use `uv run ruff check --fix` for auto-fixing issues

### Code Organization & Simplicity
- **MUST keep related classes together** in the same file
- **MUST minimize file count** - don't create separate files for every class
- **MUST use logical grouping** for related functionality
- **MUST avoid over-engineering** - don't create abstractions unless they solve a real problem
- **MUST use clear, descriptive file names** that indicate the file's purpose

#### **When to Put Classes in the Same File:**
- ✅ **Same Interface**: All implementations of the same interface
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

### Project Setup
- Use `uv init` to initialize new Python projects
- Use `uv add` to add dependencies to pyproject.toml
- Use `uv sync` to install dependencies from pyproject.toml
- Use `uv run` to execute scripts in the virtual environment

### Code Style
- Follow PEP 8 standards (enforced by Ruff)
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Git Workflow
- Commit frequently with descriptive messages
- Use conventional commit format when possible
- Keep commits atomic and focused

### Environment Management
- Use UV's built-in virtual environment management
- No manual venv creation needed
- Store dependencies in pyproject.toml
- Use .env files for environment variables (not in version control)

## 📋 Step-by-Step Development Process

### Before Each Step
- **Explain the context**: Why are we doing this step? How does it fit into the overall project goals?
- **Reference the plan**: Which step from our planning document are we about to execute?
- **Set expectations**: What should the outcome look like?

### During Each Step
- **Add dependencies incrementally**: Only add packages when we actually need them
- **Document decisions**: Why did we choose this approach over alternatives?
- **Test as we go**: Verify each component works before moving to the next
- **Keep it simple**: Don't over-engineer solutions

### After Each Step
- **Check off completion**: Mark the step as complete in our planning document
- **Commit progress**: Make a git commit with descriptive message
- **Update documentation**: Reflect any changes to the project structure or approach
- **Plan next step**: What's the logical next step based on what we just accomplished?

### Progress Tracking
- Keep the planning document updated with completion status
- Use checkboxes (✅) to mark completed steps
- Add notes about any deviations from the original plan
- Document lessons learned and insights gained

## 🚀 Quick Commands Reference

```bash
# Initialize new project
uv init

# Add dependencies
uv add langchain faiss-cpu openai sentence-transformers

# Install dependencies
uv sync

# Run scripts
uv run python script.py

# Lint code
uv run ruff check

# Format code
uv run ruff format

# Auto-fix issues
uv run ruff check --fix
``` 