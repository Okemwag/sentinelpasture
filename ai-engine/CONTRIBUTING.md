# Contributing to AI Engine

Thank you for your interest in contributing to the AI Engine project!

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   make install-dev
   ```

## Code Style

We follow these style guidelines:

- **PEP 8** for Python code style
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **Type hints** for all public functions
- **Docstrings** for all modules, classes, and functions

### Running Code Formatters

```bash
make format
```

### Running Linters

```bash
make lint
```

## Testing

We use pytest for testing. All new features should include tests.

### Running Tests

```bash
# Run all tests with coverage
make test

# Run tests quickly without coverage
make test-fast
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures for common setup
- Use `@pytest.mark.asyncio` for async tests

Example:

```python
import pytest

@pytest.mark.asyncio
async def test_my_feature():
    result = await my_async_function()
    assert result is not None
```

## Type Checking

We use mypy for static type checking:

```bash
make type-check
```

## Documentation

- Update docstrings for any changed functions
- Update README.md if adding new features
- Build docs locally to verify:
  ```bash
  make docs
  ```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run all checks:
   ```bash
   make all
   ```
4. Commit with clear, descriptive messages
5. Push and create a pull request
6. Ensure all CI checks pass

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Example:
```
Add meta-learning capability for rapid adaptation

- Implement MAML algorithm
- Add tests for few-shot learning
- Update documentation

Fixes #123
```

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Questions?

Feel free to open an issue for any questions or concerns.
