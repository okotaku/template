# Quick Start

Get up and running with Template in just a few minutes.

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

<!-- start-install -->

### Install uv

If you don't have uv installed yet:

::::{tab-set}

:::{tab-item} macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
:::

:::{tab-item} Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
:::

:::{tab-item} pip
```bash
pip install uv
```
:::

::::

### Install Template

::::{tab-set}

:::{tab-item} From GitHub
```bash
uv add git+https://github.com/okotaku/template.git
```
:::

:::{tab-item} Development Installation
```bash
git clone https://github.com/okotaku/template.git
cd template
uv sync --all-extras
```
:::

::::

<!-- end-install -->

## First Steps

### 1. Verify Installation

```bash
uv run python -c "import modules; print('Template installed successfully!')"
```

### 2. Run Tests

```bash
uv run pytest
```

### 3. Check Code Quality

```bash
uv run ruff check .
uv run pyright modules
```

### 4. Set Up Pre-commit (Development)

If you're contributing or modifying the code:

```bash
uv run pre-commit install
```

## Basic Usage

### Command Line Interface

Template provides a CLI for common operations:

```bash
# Show help
uv run modules --help

# Run a basic command
uv run modules your-command
```

### Python API

Use Template in your Python code:

```python
from modules import your_module

# Basic usage example
result = your_module.process_data(["example", "data"])
print(result)
```

## Development Workflow

### Setting Up for Development

1. **Clone and install**:
   ```bash
   git clone https://github.com/okotaku/template.git
   cd template
   uv sync --all-extras
   ```

2. **Install pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

3. **Make your changes** and run quality checks:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run pyright modules
   uv run pytest
   ```

### Using with Claude Code

This template is optimized for [Claude Code](https://claude.ai/code). The project includes:

- **CLAUDE.md**: Comprehensive project context for AI assistance
- **Modern tooling**: uv, ruff, pyright for fast AI-assisted development
- **Clear structure**: Well-organized codebase for better AI understanding

See the [Claude Code Guide](claude-code-guide.md) for detailed instructions.

## Next Steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 📖 User Guide
:link: user-guide/index
:link-type: doc

Learn about configuration, development patterns, and best practices.
:::

:::{grid-item-card} 🤖 AI Development
:link: ai-development
:link-type: doc

Discover how to leverage AI tools for enhanced productivity.
:::

:::{grid-item-card} 🛠️ API Reference
:link: api/index
:link-type: doc

Explore the complete API documentation.
:::

:::{grid-item-card} 🙌 Contributing
:link: contributing
:link-type: doc

Learn how to contribute to the project.
:::

::::

## Troubleshooting

### Common Issues

**Import errors**: Make sure you've installed Template with `uv sync --all-extras`.

**Permission errors**: On Windows, you might need to run commands in an elevated prompt.

**uv not found**: Ensure uv is installed and added to your PATH.

### Getting Help

- 📖 Check the [User Guide](user-guide/index.md)
- 🐛 Report issues on [GitHub](https://github.com/okotaku/template/issues)
- 💬 Join discussions on [GitHub Discussions](https://github.com/okotaku/template/discussions)
- 🤖 See [Claude Code Guide](claude-code-guide.md) for AI development help