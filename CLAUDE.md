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

7. **Documentation Updates**:

   - When developing new features, update CLAUDE.md and README.md as needed
   - Document new dependencies, commands, or workflows
   - Keep project structure section current with any new directories or files
   - Update best practices section if introducing new patterns or conventions

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

## AI-Assisted Development with Claude Code

This template is designed from the ground up for AI-assisted development, specifically
optimized for Claude Code.

### Why AI-First Development?

Modern software development is increasingly enhanced by AI tools that can:

- **Generate code faster**: Write boilerplate, implement algorithms, create tests
- **Improve code quality**: Suggest optimizations, catch bugs, enforce patterns
- **Accelerate learning**: Explain complex code, suggest best practices
- **Boost productivity**: Handle repetitive tasks, refactor code, write documentation

### Claude Code Integration

[Claude Code](https://claude.ai/code) is Anthropic's official CLI tool designed for
AI-assisted development.

#### Why Claude Code?

- **🧠 Context Aware**: Claude Code understands your entire project structure and
  codebase context
- **🚀 Modern Tools**: Native integration with modern Python tools like uv, ruff, and
  pyright
- **🔧 Workflow Optimized**: Designed for complete development workflows, not just code
  generation
- **📚 Project Understanding**: Uses CLAUDE.md for comprehensive project context and best
  practices

### Claude Code Setup Guide

#### Installation

1. **Prerequisites**

   - A Claude account (sign up at [claude.ai](https://claude.ai))
   - Command line access
   - This template project

2. **Install Claude Code** Follow the official installation instructions at
   [claude.ai/code](https://claude.ai/code).

3. **Authentication**

   ```bash
   claude-code auth login
   ```

#### Project Setup

1. **Initialize Claude Code**

   ```bash
   claude-code init
   ```

2. **Test the Setup**

   ```bash
   claude-code "Explain the project structure"
   ```

### AI Development Workflows

#### 1. Code Generation

**Example**: Generate a new feature with Claude Code

```bash
claude-code "Add a data validation module with type checking"
```

Template's structure makes it easy for AI to:

- Understand existing patterns
- Follow project conventions
- Generate consistent code
- Add appropriate tests

#### 2. Code Review and Optimization

**Example**: Review and improve existing code

```bash
claude-code "Review the data processing module for performance optimizations"
```

Template provides context through:

- Type hints for better understanding
- Clear module organization
- Comprehensive testing patterns
- Performance benchmarking setup

#### 3. Testing and Quality Assurance

**Example**: Generate comprehensive tests

```bash
claude-code "Create comprehensive tests for the new validation module"
```

Template's testing setup includes:

- pytest configuration
- Coverage reporting
- Type checking integration
- Pre-commit hooks

#### 4. Documentation Generation

**Example**: Create documentation for new features

```bash
claude-code "Create documentation for the validation module with usage examples"
```

### Best Practices for AI Development

#### 1. Provide Clear Context

**Good prompts include**:

- Specific requirements
- Expected behavior
- Error handling needs
- Performance considerations

**Example**:

```text
Create a data validator that:
- Accepts pandas DataFrames
- Validates column types and ranges
- Returns detailed error messages
- Handles missing values gracefully
- Includes comprehensive type hints
```

#### 2. Iterate and Refine

**Development cycle**:

1. Generate initial implementation
2. Run tests and quality checks
3. Refine based on feedback
4. Add edge cases and documentation
5. Optimize performance

#### 3. Leverage Project Patterns

**Reference existing code**:

```text
Follow the pattern used in modules/cli.py for:
- Argument parsing
- Configuration handling
- Logging setup
- Error handling
```

#### 4. Use Quality Gates

**Always run**:

```bash
# Code quality checks
uv run ruff check .
uv run pyright modules

# Tests
uv run pytest

# Pre-commit hooks
uv run pre-commit run --all-files
```

### Advanced AI Workflows

#### Multi-file Refactoring

Claude Code can handle complex refactoring across multiple files:

```bash
# Refactor module structure
claude-code "Refactor the data processing pipeline to use dependency injection"
```

#### Performance Optimization

Get AI assistance for performance improvements:

```bash
# Optimize computational bottlenecks
claude-code "Optimize the training loop for better GPU utilization"
```

#### Architecture Design

Use AI for high-level architectural decisions:

```bash
# Design new features
claude-code "Design a plugin system for custom data loaders"
```

### Integration Tips

#### 1. Keep CLAUDE.md Updated

Update the CLAUDE.md file when you:

- Add new modules or features
- Change development workflows
- Update dependencies or tools
- Implement new patterns

#### 2. Use Descriptive Commit Messages

Help AI understand your project evolution:

```bash
git commit -m "feat: add data validation with custom error types

- Implements ValidationError hierarchy
- Adds DataFrame schema validation
- Includes comprehensive test suite
- Updates documentation with examples"
```

#### 3. Maintain Clean Project Structure

Keep your project AI-friendly:

- Clear module organization
- Consistent naming conventions
- Comprehensive type hints
- Good separation of concerns

### Common Workflows

#### Feature Development Workflow

1. **Plan the Feature**

   ```bash
   claude-code "Design a plugin system for custom data loaders"
   ```

2. **Generate Implementation**

   ```bash
   claude-code "Implement the plugin system designed earlier"
   ```

3. **Create Tests**

   ```bash
   claude-code "Create comprehensive tests for the plugin system"
   ```

4. **Add Documentation**

   ```bash
   claude-code "Document the plugin system with examples"
   ```

5. **Review and Optimize**

   ```bash
   claude-code "Review the plugin system implementation for improvements"
   ```

#### Debugging Workflow

1. **Understand the Problem**

   ```bash
   claude-code "Help me understand this error: [paste error message]"
   ```

2. **Analyze Context**

   ```bash
   claude-code "What could cause this issue in modules/cli.py?"
   ```

3. **Generate Solutions**

   ```bash
   claude-code "Fix the configuration loading error"
   ```

4. **Test the Fix**

   ```bash
   claude-code "Create a test to verify the fix works correctly"
   ```

### Integration with Template Tools

#### With uv

Claude Code understands Template's uv configuration:

```bash
# Add dependencies
claude-code "Add pandas as a dependency using uv"

# Update dependencies
claude-code "Update the development dependencies"
```

#### With ruff

Use Claude Code to fix linting issues:

```bash
# Fix ruff issues
claude-code "Fix all ruff linting errors in modules/"

# Optimize imports
claude-code "Optimize imports according to ruff configuration"
```

#### With pyright

Get help with type issues:

```bash
# Fix type errors
claude-code "Resolve pyright type errors in modules/"

# Add type hints
claude-code "Add comprehensive type hints to modules/version.py"
```

### Troubleshooting AI Development

#### Common Issues

##### AI generates incompatible code

- Ensure CLAUDE.md is up to date
- Provide more specific context
- Reference existing patterns

##### Quality checks fail

- Run checks before and after AI assistance
- Use incremental development
- Leverage pre-commit hooks

##### Tests break after AI changes

- Generate tests alongside code
- Use TDD approach with AI
- Validate behavior preservation

#### Getting Better Results

1. **Be specific**: Provide detailed requirements and constraints
2. **Show examples**: Reference existing code patterns
3. **Iterate**: Use AI assistance in small, manageable chunks
4. **Validate**: Always test and review AI-generated code
5. **Learn**: Understand the patterns AI suggests
