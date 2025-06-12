# template

[![build](https://github.com/okotaku/template/actions/workflows/build.yml/badge.svg)](https://github.com/okotaku/template/actions/workflows/build.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://template.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/okotaku/template.svg)](https://github.com/okotaku/template/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/okotaku/template.svg)](https://github.com/okotaku/template/issues)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked with pyright](https://img.shields.io/badge/type%20checked-pyright-blue)](https://github.com/microsoft/pyright)
[![Claude Code Ready](https://img.shields.io/badge/Claude%20Code-Ready-brightgreen)](CLAUDE.md)

[📘 Documentation](https://template0.readthedocs.io/en/latest/) |
[🤔 Reporting Issues](https://github.com/okotaku/template/issues/new/choose) |
[🤖 Claude Code Setup](CLAUDE.md)

## 📄 Table of Contents

- [template](#template)
  - [📄 Table of Contents](#-table-of-contents)
  - [📖 Introduction 🔝](#-introduction-)
  - [🛠️ Installation 🔝](#%EF%B8%8F-installation-)
  - [👨‍🏫 Get Started 🔝](#-get-started-)
  - [📘 Documentation 🔝](#-documentation-)
  - [🤖 AI Development Tools 🔝](#-ai-development-tools-)
  - [🙌 Contributing 🔝](#-contributing-)
  - [🎫 License 🔝](#-license-)
  - [🖊️ Citation 🔝](#%EF%B8%8F-citation-)
  - [Acknowledgement](#acknowledgement)

## 📖 Introduction [🔝](#-table-of-contents)

Template is a modern Python project template optimized for AI-assisted development with Claude Code.

## 🛠️ Installation [🔝](#-table-of-contents)

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Install uv

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Install template

#### From GitHub (recommended)

```bash
uv add git+https://github.com/okotaku/template.git
```

#### Development installation

```bash
git clone https://github.com/okotaku/template.git
cd template
uv sync --all-extras
```

#### Docker installation

```bash
# Development environment
docker-compose up template

# Jupyter Lab environment
docker-compose up template-jupyter

# Documentation development
docker-compose up template-docs
```

## 👨‍🏫 Get Started [🔝](#-table-of-contents)

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/okotaku/template.git
   cd template
   ```

2. Install dependencies:

   ```bash
   uv sync --all-extras
   ```

3. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

4. Run tests:

   ```bash
   uv run pytest
   ```

5. Run linting:

   ```bash
   uv run ruff check .
   uv run pyright modules
   ```

### Usage

```python
import modules

# Basic usage
print(modules.hello())  # "Hello from Template!"

# Check version
print(modules.__version__)
```

## 📘 Documentation [🔝](#-table-of-contents)

For detailed user guides and advanced guides, please refer to our [Documentation](https://template0.readthedocs.io/en/latest/):

- [Get Started](https://template0.readthedocs.io/en/latest/get_started.html) for get started.

## 🤖 AI Development Tools [🔝](#-table-of-contents)

This template is optimized for AI-powered development with Claude Code, Anthropic's official CLI tool.

### Claude Code Integration

This template is fully configured for [Claude Code](https://claude.ai/code):

- Comprehensive project understanding through CLAUDE.md
- Automated development workflows with uv integration
- Context-aware code generation and refactoring
- Integrated testing and quality assurance
- Modern Python development best practices
- Pre-configured for optimal AI assistance

See the [Claude Code Setup](CLAUDE.md) for detailed configuration and usage instructions.

## 🙌 Contributing [🔝](#-table-of-contents)

We welcome contributions from the community! Please read our [Contributing Guide](CONTRIBUTING.md) for detailed information on how to contribute to this project.

### Quick Start for Contributors

1. Fork the repository
2. Install dependencies: `uv sync --all-extras`
3. Install pre-commit: `uv run pre-commit install`
4. Make your changes and add tests
5. Run quality checks: `uv run pre-commit run --all-files`
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🎫 License [🔝](#-table-of-contents)

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation [🔝](#-table-of-contents)

If Modules is helpful to your research, please cite it as below.

```
@misc{modules2023,
    title = {{Modules}: XXX},
    author = {{Modules Contributors}},
    howpublished = {\url{https://github.com/okotaku/template}},
    year = {2023}
}
```

## Acknowledgement

This repo borrows the architecture design and part of the code from [mmengine](https://github.com/open-mmlab/mmengine).

Also, please check the following openmmlab projects and the corresponding Documentation.

- [OpenMMLab](https://openmmlab.com/)
- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.

```
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```
