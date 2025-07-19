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

## 📋 Step-by-Step Development Process

### Before Each Step
- **Explain the context**: Why are we doing this step? How does it fit into the overall project goals?
- **Reference the plan**: Which step from our planning document are we about to execute?
- **Set expectations**: What should the outcome look like?

### During Each Step
- **Add dependencies incrementally**: Only add packages when we actually need them
- **Document decisions**: Why did we choose this approach over alternatives?
- **Test as we go**: Verify each component works before moving to the next

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