# Claude Code Configuration

This file contains configuration and context for Claude Code to better understand and work with this project.

## Project Overview

This is a modern Python template project optimized for AI-assisted development with Claude Code:

- **Package Management**: Uses `uv` for fast, reliable dependency management
- **Code Quality**: Configured with `ruff` for linting/formatting and `pyright` for type checking
- **Testing**: Uses `pytest` with coverage reporting
- **CI/CD**: GitHub Actions with automated testing and quality checks
- **Pre-commit**: Automated code quality checks before commits
- **AI-First**: Fully configured for Claude Code development workflows
- **Simple Structure**: Clean, minimal codebase for easy understanding

## Development Workflow

### Environment Setup

```bash
# Install dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

### Common Commands

```bash
# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run pyright modules

# Run pre-commit checks
uv run pre-commit run --all-files
```

### Testing

- Test files are located in the `tests/` directory
- Use `pytest` for testing framework
- Coverage reports are generated automatically

### Code Quality

- **Linting**: `ruff` with comprehensive rule set
- **Formatting**: `ruff format` with consistent style
- **Type Checking**: `mypy` with strict configuration
- **Pre-commit**: Automated checks on commit

## Project Structure

```
├── modules/                 # Main package directory
│   ├── __init__.py
│   ├── configs/            # Configuration files
│   ├── tools/              # Utility tools
│   ├── entry_point.py      # CLI entry point
│   └── version.py          # Version information
├── tests/                  # Test files
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
├── .pre-commit-config.yaml # Pre-commit configuration
└── README.md               # Project documentation
```

## Key Files to Understand

- `pyproject.toml`: Main project configuration, dependencies, and tool settings
- `modules/__init__.py`: Main package with hello() function
- `modules/version.py`: Version management
- `tests/test_basic.py`: Basic functionality tests
- `.pre-commit-config.yaml`: Code quality automation setup
- `.github/workflows/build.yml`: CI/CD pipeline configuration

## Dependencies

### Core Dependencies

- No core dependencies by default - add as needed

### Development Dependencies

- `pytest`: Testing framework
- `coverage`: Code coverage reporting
- `ruff`: Fast Python linter and formatter
- `pyright`: Fast static type checker
- `pre-commit`: Git hooks framework

## Best Practices for This Project

1. **Type Hints**: All new code should include proper type hints
2. **Testing**: Write tests for new functionality in the `tests/` directory
3. **Documentation**: Update docstrings and documentation for new features
4. **Code Quality**: Run pre-commit hooks before committing
5. **Dependencies**: Use `uv add` to add new dependencies
6. **Simplicity**: Keep the codebase clean and minimal

## Common Issues and Solutions

### Dependency Management

- Use `uv sync` to install/update dependencies
- Use `uv add package-name` to add new dependencies
- Lock file (`uv.lock`) should be committed to version control

### Code Quality Issues

- Run `uv run ruff check --fix .` to auto-fix linting issues
- Run `uv run ruff format .` to format code
- Check `uv run pyright modules` for type issues

### Testing Issues

- Ensure test files follow the pattern `test_*.py` or `*_test.py`
- Use `uv run pytest -v` for verbose test output
- Check coverage with `uv run pytest --cov=modules`

This project follows modern Python development practices and is specifically optimized for productivity with Claude Code, Anthropic's official AI development CLI.
