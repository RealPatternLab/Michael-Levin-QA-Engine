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