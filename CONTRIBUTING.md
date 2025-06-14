# Contributing to Template

Thank you for your interest in contributing to this project! This guide will help you
get started with contributing to our Python template project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [AI-Assisted Development](#ai-assisted-development)

## 🤝 Code of Conduct

This project follows a simple code of conduct: be respectful, inclusive, and
constructive in all interactions. We welcome contributions from developers of all
backgrounds and experience levels.

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git
- A GitHub account

### Development Setup

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/your-username/template.git
   cd template
   ```

2. **Install uv (if not already installed):**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**

   ```bash
   uv sync --all-extras
   ```

4. **Install pre-commit hooks:**

   ```bash
   uv run pre-commit install
   ```

5. **Verify the setup:**

   ```bash
   uv run pytest
   uv run ruff check .
   uv run pyright modules
   ```

## 🛠️ Making Contributions

### Types of Contributions

We welcome several types of contributions:

- 🐛 **Bug fixes**: Fix issues or unexpected behavior
- ✨ **New features**: Add new functionality
- 📚 **Documentation**: Improve docs, examples, or guides
- 🧪 **Tests**: Add or improve test coverage
- ⚡ **Performance**: Optimize existing code
- 🧹 **Refactoring**: Improve code structure without changing functionality

### Finding Something to Work On

1. Check our [Issues](https://github.com/okotaku/template/issues) for open bugs and
   feature requests
2. Look for issues labeled `good first issue` if you're new to the project
3. Check our [Discussions](https://github.com/okotaku/template/discussions) for ideas
4. Propose your own improvements

### Before You Start

1. Check if an issue already exists for your contribution
2. If not, create an issue to discuss your proposed changes
3. Wait for feedback before starting major work
4. Assign yourself to the issue to avoid duplicate work

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow our coding standards
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
uv run pytest

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run pyright modules

# Run pre-commit checks
uv run pre-commit run --all-files
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

**Commit Message Convention:**

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `perf:` for performance improvements
- `chore:` for maintenance tasks

### 5. Push and Create a Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub using our
[PR template](.github/PULL_REQUEST_TEMPLATE.md).

## 📝 Coding Standards

### Python Code Style

We use several tools to maintain code quality:

- **[Ruff](https://github.com/astral-sh/ruff)**: For linting and formatting
- **[pyright](https://github.com/microsoft/pyright)**: For fast static type checking
- **[pre-commit](https://pre-commit.com/)**: For automated quality checks

### Key Guidelines

1. **Type Hints**: All functions should have proper type hints

   ```python
   def process_data(data: list[str], count: int = 10) -> dict[str, int]:
       ...
   ```

2. **Docstrings**: Use Google-style docstrings

   ```python
   def example_function(param1: str, param2: int) -> bool:
       """Brief description of the function.

       Args:
           param1: Description of param1.
           param2: Description of param2.

       Returns:
           Description of return value.

       Raises:
           ValueError: Description of when this error is raised.
       """
   ```

3. **Error Handling**: Use specific exception types and meaningful error messages

   ```python
   if not data:
       raise ValueError("Data cannot be empty")
   ```

4. **Variable Names**: Use descriptive names

   ```python
   # Good
   user_count = len(users)

   # Bad
   n = len(users)
   ```

## 🧪 Testing

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure: `modules/foo.py` → `tests/test_foo.py`
- Use descriptive test names that describe what is being tested

```python
def test_process_data_with_valid_input():
    """Test that process_data works correctly with valid input."""
    result = process_data(["a", "b", "c"], count=2)
    assert result == {"a": 1, "b": 1}
```

### Test Coverage

- Aim for >90% test coverage
- Write both unit tests and integration tests
- Test edge cases and error conditions

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=modules

# Run specific test file
uv run pytest tests/test_specific.py

# Run with verbose output
uv run pytest -v
```

## 📚 Documentation

### Code Documentation

- Write clear docstrings for all public functions and classes
- Include examples in docstrings when helpful
- Keep inline comments focused on *why*, not *what*

### Project Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for changes affecting AI development workflows
- Add examples for new features

## 🤖 AI-Assisted Development

This project is optimized for AI-assisted development with Claude Code.

### Using Claude Code

- Reference [CLAUDE.md](CLAUDE.md) for comprehensive project context
- Include AI assistance information in PR descriptions
- Ensure AI-generated code is reviewed and tested
- Follow the established development patterns and workflows

### Best Practices for AI Development

1. **Review AI suggestions**: Always review and understand AI-generated code
2. **Test thoroughly**: AI-generated code still needs comprehensive testing
3. **Document AI usage**: Note when AI assistance was used in your PR
4. **Maintain quality**: AI assistance doesn't replace code quality standards
5. **Use project context**: Leverage the CLAUDE.md file for better AI understanding

## 🏷️ Release Process

Releases are managed by maintainers:

1. Version bumps follow [Semantic Versioning](https://semver.org/)
2. Releases are tagged with `v*.*.*` format
3. GitHub Actions automatically builds and publishes releases
4. Changelog is generated from commit messages

## ❓ Getting Help

- **Questions**: Use
  [GitHub Discussions](https://github.com/okotaku/template/discussions)
- **Bugs**: Create an [Issue](https://github.com/okotaku/template/issues)
- **Chat**: Join our community discussions

## 🙏 Recognition

Contributors are recognized in:

- README.md contributors section
- Release notes for significant contributions
- Project documentation where appropriate

______________________________________________________________________

Thank you for contributing to our project! Every contribution, no matter how small,
helps make this project better for everyone. 🎉
