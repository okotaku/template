# Modern Python Template

[![CI/CD](https://github.com/okotaku/template/actions/workflows/build.yml/badge.svg)](https://github.com/okotaku/template/actions/workflows/build.yml)
[![Documentation](https://img.shields.io/badge/docs-README-blue)](README.md)
[![License](https://img.shields.io/github/license/okotaku/template.svg)](https://github.com/okotaku/template/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-0.5.28%2B-green)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Claude Code Ready](https://img.shields.io/badge/Claude%20Code-Ready-brightgreen)](CLAUDE.md)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Renovate](https://img.shields.io/badge/renovate-enabled-brightgreen?logo=renovatebot)](https://github.com/renovatebot/renovate)

[📘 Documentation](README.md) |
[🐛 Report Bug](https://github.com/okotaku/template/issues/new?labels=bug) |
[✨ Request Feature](https://github.com/okotaku/template/issues/new?labels=enhancement) |
[🤖 Claude Code Setup](CLAUDE.md)

## 🚀 Features

- **🎯 Modern Python**: Built for Python 3.12+ with latest language features
- **📦 uv Package Manager**: Lightning-fast dependency management
- **🔧 Developer Tools**: Pre-configured with Ruff, Pyright, and pre-commit
- **🤖 AI-First Development**: Optimized for Claude Code workflows
- **✅ Comprehensive Testing**: Pytest with coverage, parallel execution, and
  benchmarking
- **🔒 Security First**: Integrated security scanning with Trivy and Bandit
- **🔄 CI/CD Pipeline**: Advanced GitHub Actions with reusable workflows
- **📝 Type Safety**: Full type annotations with strict checking
- **🎨 Code Quality**: 15+ pre-commit hooks for consistent code
- **🔄 Auto Updates**: Renovate bot for dependency management

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Development](#-development)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## 🏃 Quick Start

```bash
# Clone the repository
git clone https://github.com/okotaku/template.git
cd template

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras

# Run the CLI
uv run template --name "Your Name"

# Run tests
uv run pytest
```

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installing uv

<details>
<summary>macOS and Linux</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

<details>
<summary>Windows</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</details>

<details>
<summary>Using pip</summary>

```bash
pip install uv
```

</details>

### Setting up the project

1. **Clone the repository**

   ```bash
   git clone https://github.com/okotaku/template.git
   cd template
   ```

2. **Install dependencies**

   ```bash
   uv sync --all-extras
   ```

3. **Install pre-commit hooks**

   ```bash
   uv run pre-commit install
   ```

4. **Verify installation**

   ```bash
   uv run template --version
   ```

## 📖 Usage

### Command Line Interface

The template provides a simple CLI for demonstration:

```bash
# Show help
uv run template --help

# Basic greeting
uv run template

# Personalized greeting
uv run template --name Alice

# Verbose output
uv run template --verbose
```

### As a Python Package

```python
import modules

# Get version
print(modules.__version__)

# Use functions
print(modules.hello())
print(modules.greet("World"))
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run in parallel
uv run pytest -n auto

# Run specific test file
uv run pytest tests/test_basic.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Run specific hooks
uv run ruff check . --fix
uv run ruff format .
uv run pyright modules

# Check types
uv run pyright modules tests
```

### Additional Resources

```bash
# View changelog
cat CHANGELOG.md

# View comprehensive development guide
cat CLAUDE.md

# View project structure
ls -la
```

## 📁 Project Structure

```text
template/
├── modules/                # Source code
│   ├── __init__.py        # Package exports
│   ├── cli.py             # CLI implementation
│   ├── version.py         # Version management
│   └── py.typed           # Type marker file
├── tests/                 # Test suite
│   ├── test_basic.py      # Basic tests
│   └── test_cli.py        # CLI tests
├── .github/               # GitHub configuration
│   ├── workflows/         # CI/CD pipelines
│   ├── actions/           # Reusable actions
│   └── *.yml             # Various configs
├── pyproject.toml         # Project metadata
├── uv.lock               # Locked dependencies
├── .pre-commit-config.yaml # Pre-commit hooks
├── CLAUDE.md             # Claude Code config
└── README.md             # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for
details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Run tests and linting (`uv run pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feat/amazing-feature`)
7. Open a Pull Request

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for
details.

## 🙏 Acknowledgments

- Built with [uv](https://github.com/astral-sh/uv) for fast package management
- Linted with [Ruff](https://github.com/astral-sh/ruff) for lightning-fast code quality
- Type-checked with [Pyright](https://github.com/microsoft/pyright) for robust type
  safety
- Tested with [Pytest](https://pytest.org/) for comprehensive test coverage
- Automated with [GitHub Actions](https://github.com/features/actions) for CI/CD
- Optimized for [Claude Code](https://github.com/anthropics/claude-engineer) development

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/okotaku/template?style=social)
![GitHub forks](https://img.shields.io/github/forks/okotaku/template?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/okotaku/template?style=social)

______________________________________________________________________

<p align="center">
  Made with ❤️ by <a href="https://github.com/okotaku">okotaku</a>
</p>

<p align="center">
  <a href="#modern-python-template">⬆ Back to top</a>
</p>
