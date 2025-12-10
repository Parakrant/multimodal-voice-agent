# Contributing to Multi-Modal Voice Agent Pipeline

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include logs and error messages**
- **Specify your environment** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** and ensure the code follows our style guidelines
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Write clear commit messages**
6. **Submit a pull request**

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/yourusername/multimodal-voice-agent.git
cd multimodal-voice-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use type hints where possible

### Example:

```python
async def example_function(param1: str, param2: int) -> Dict[str, Any]:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Dictionary containing the result
    """
    # Implementation
    pass
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests after the first line

Example:
```
Add sentiment analysis tool

- Implement keyword-based sentiment detection
- Add positive/negative word counting
- Return sentiment score between 0 and 1

Fixes #123
```

## Testing

Before submitting a pull request:

1. Run the server and verify it starts without errors:
```bash
python main_enhanced.py
```

2. Test the web client:
```bash
open client_enhanced.html
```

3. Run the benchmark:
```bash
python benchmark.py --max-concurrency 5
```

4. Check for common issues:
- No syntax errors
- All endpoints respond correctly
- WebSocket connections work
- No API key leaks in code

## Adding New Tools

To add a new LLM tool:

1. Define the tool schema in `LLM_TOOLS`:
```python
{
    "type": "function",
    "function": {
        "name": "my_new_tool",
        "description": "Clear description",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"]
        }
    }
}
```

2. Implement the tool function:
```python
async def my_new_tool(param1: str) -> Dict[str, Any]:
    """Tool implementation"""
    return {"success": True, "result": ...}
```

3. Register in `TOOL_REGISTRY`:
```python
TOOL_REGISTRY["my_new_tool"] = my_new_tool
```

4. Update documentation

## Documentation

When adding new features:

- Update README.md
- Add examples to QUICKSTART.md
- Update API documentation if adding endpoints
- Include inline code comments for complex logic

## Code Review Process

1. All submissions require review
2. Reviewers will check:
   - Code quality and style
   - Test coverage
   - Documentation updates
   - Performance implications
   - Security considerations

## Questions?

Feel free to ask questions by:
- Opening an issue
- Starting a discussion
- Reaching out to maintainers

Thank you for contributing! 🚀
