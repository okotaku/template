# Claude Code Configuration

This file contains configuration and context for Claude Code to better understand and
work with this project.

## Project Overview

This is a modern Python template project optimized for AI-assisted development with
Claude Code:

- **Package Management**: Uses `uv` (0.5.28+) for fast, reliable dependency management
- **Code Quality**:
  - `ruff` (0.8.0+) for lightning-fast linting and formatting
  - `pyright` (1.1.390+) for strict type checking
  - Comprehensive pre-commit hooks with auto-fixing
- **Testing**:
  - `pytest` (8.3+) with coverage reporting
  - Parallel test execution with `pytest-xdist`
  - Type-annotated test suite
- **CI/CD**:
  - Modern GitHub Actions with reusable workflows
  - Multi-platform testing (Linux, macOS, Windows)
  - Security scanning with Trivy and Bandit
  - Automated dependency updates with Renovate
- **Type Safety**:
  - Full type annotations with Python 3.12+ features
  - Strict pyright configuration
  - Runtime type checking support
- **Developer Experience**:
  - EditorConfig for consistent formatting
  - Semantic PR validation
  - Auto-labeling for PRs
  - Performance benchmarking support

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
uv run pytest -n auto  # Run tests in parallel
uv run pytest --cov    # Run with coverage

# Run linting and formatting
uv run ruff check . --fix
uv run ruff format .

# Run type checking
uv run pyright modules
uv run pyright modules tests  # Type check everything

# Run pre-commit checks
uv run pre-commit run --all-files

# CLI usage
uv run template --help
uv run template --name Alice
uv run template --version
```

### Testing

- Test files are located in the `tests/` directory
- Use `pytest` for testing framework
- Coverage reports are generated automatically

### Code Quality

- **Linting**: `ruff` with comprehensive rule set (ALL rules enabled)
- **Formatting**: `ruff format` with consistent style and docstring formatting
- **Type Checking**: `pyright` with strict configuration
- **Pre-commit**: 15+ hooks for comprehensive code quality
- **Security**: Integrated Trivy, Bandit, and TruffleHog scanning

## Project Structure

```text
├── modules/                 # Main package directory
│   ├── __init__.py         # Package initialization with exports
│   ├── cli.py              # CLI entry point
│   ├── version.py          # Version information
│   └── py.typed            # PEP 561 type marker
├── tests/                  # Test files
│   ├── test_basic.py       # Core functionality tests
│   └── test_cli.py         # CLI tests
├── docs/                   # Documentation
├── .github/                # GitHub configuration
│   ├── workflows/          # CI/CD workflows
│   ├── actions/            # Reusable actions
│   ├── labeler.yml         # PR auto-labeling
│   └── semantic.yml        # Semantic PR config
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
├── .pre-commit-config.yaml # Pre-commit hooks
├── .editorconfig           # Editor configuration
├── .markdownlint.json      # Markdown linting rules
├── renovate.json           # Dependency update config
└── README.md               # Project documentation
```

## Key Files to Understand

- `pyproject.toml`: Main project configuration with modern tool settings
- `modules/__init__.py`: Main package with typed exports
- `modules/cli.py`: Command-line interface implementation
- `modules/version.py`: Advanced version management with metadata
- `tests/test_basic.py`: Comprehensive test suite with type annotations
- `tests/test_cli.py`: CLI functionality tests
- `.pre-commit-config.yaml`: 15+ hooks for code quality
- `.github/workflows/build.yml`: Advanced CI/CD pipeline
- `renovate.json`: Automated dependency management

## Dependencies

### Core Dependencies

- No core dependencies by default - add as needed

### Development Dependencies

- `pytest` (8.3+): Modern testing framework
- `pytest-cov` (6.0+): Coverage plugin for pytest
- `pytest-xdist` (3.6+): Parallel test execution
- `pytest-timeout` (2.3+): Test timeout management
- `coverage` (7.6+): Code coverage reporting
- `ruff` (0.8.0+): Lightning-fast linter and formatter
- `pyright` (1.1.390+): Microsoft's static type checker
- `pre-commit` (4.0+): Git hooks framework
- `types-*`: Type stubs for better type checking

## Best Practices for This Project

1. **Type Hints**:

   - Use Python 3.12+ syntax (`str | None` instead of `Optional[str]`)
   - Add `from __future__ import annotations` for better forward compatibility
   - Use `TYPE_CHECKING` blocks for import optimization

2. **Testing**:

   - Write comprehensive tests with type annotations
   - Use pytest fixtures and parametrization
   - Aim for >80% code coverage
   - Mark tests with appropriate markers (@pytest.mark.unit, etc.)

3. **Documentation**:

   - Use Google-style docstrings
   - Include Examples sections in docstrings
   - Keep README and CLAUDE.md updated

4. **Code Quality**:

   - Run `uv run pre-commit run --all-files` before committing
   - Fix all ruff and pyright warnings
   - Use semantic commit messages (feat:, fix:, docs:, etc.)

5. **Dependencies**:

   - Use `uv add` for runtime dependencies
   - Use `uv add --dev` for development dependencies
   - Let Renovate handle updates automatically

6. **Security**:

   - Never commit secrets or API keys
   - Review security scan results in CI
   - Keep dependencies up to date

## Common Issues and Solutions

### Dependency Management

- Use `uv sync` to install/update dependencies
- Use `uv add package-name` to add new dependencies
- Lock file (`uv.lock`) should be committed to version control

### Code Quality Issues

- Run `uv run ruff check --fix .` to auto-fix linting issues
- Run `uv run ruff format .` to format code
- Check `uv run pyright modules tests` for type issues
- Use `uv run pre-commit run --all-files` to run all checks
- For specific hooks: `uv run pre-commit run <hook-id> --all-files`

### Testing Issues

- Ensure test files follow the pattern `test_*.py` or `*_test.py`
- Use `uv run pytest -v` for verbose test output
- Check coverage with `uv run pytest --cov=modules`

## Advanced Features

### Performance Benchmarking

Create benchmarks in `tests/benchmarks/` and run with:

```bash
uv run pytest tests/benchmarks/ --benchmark-only
```

### Security Scanning

Security scans run automatically in CI, but you can run locally:

```bash
# Install trivy first
trivy fs .

# Run bandit
uv add --dev bandit
uv run bandit -r modules/
```

### Multi-Platform Testing

The CI pipeline tests on Ubuntu, macOS, and Windows automatically. To test locally on
different Python versions:

```bash
uv run --python 3.13 pytest
```

This project follows modern Python development practices and is specifically optimized
for productivity with Claude Code, Anthropic's official AI development CLI. It
incorporates the latest tools and best practices as of 2024, providing a solid
foundation for any Python project.
