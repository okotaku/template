# Template Documentation

Welcome to the documentation for **Template**, a modern Python project template optimized for AI-assisted development with Claude Code.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 🚀 Quick Start
:link: quickstart
:link-type: doc

Get up and running with Template in minutes. Install dependencies, set up your environment, and start developing.
:::

:::{grid-item-card} 📖 User Guide
:link: user-guide/index
:link-type: doc

Learn how to use Template effectively. Covers installation, configuration, and common workflows.
:::

:::{grid-item-card} 🤖 AI Development
:link: ai-development
:link-type: doc

Discover how to leverage Claude Code for AI-assisted development with this template.
:::

:::{grid-item-card} 🛠️ API Reference
:link: api/index
:link-type: doc

Detailed documentation of all modules, classes, and functions in the Template project.
:::

::::

## What is Template?

Template is a modern Python project template that provides:

- **🚀 Fast Development**: Uses `uv` for lightning-fast dependency management
- **🤖 AI-First**: Optimized for Claude Code and AI-assisted development
- **🔧 Modern Tools**: Pre-configured with the latest Python development tools
- **📦 Best Practices**: Follows modern Python packaging and development standards
- **🧪 Quality Assurance**: Comprehensive testing and code quality setup

## Key Features

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} 
:class-header: text-center
**Package Management**
^^^
Uses `uv` for fast, reliable dependency management with lock files for reproducible builds.
:::

:::{grid-item-card}
:class-header: text-center
**Code Quality**
^^^
Pre-configured with `ruff`, `pyright`, and `pre-commit` for consistent code quality and formatting.
:::

:::{grid-item-card}
:class-header: text-center
**Testing**
^^^
Complete testing setup with `pytest`, coverage reporting, and GitHub Actions CI/CD.
:::

:::{grid-item-card}
:class-header: text-center
**Documentation**
^^^
Modern Sphinx documentation with MyST markdown support and automatic API generation.
:::

:::{grid-item-card}
:class-header: text-center
**AI Integration**
^^^
Fully configured for Claude Code with comprehensive project context and workflows.
:::

:::{grid-item-card}
:class-header: text-center
**Type Safety**
^^^
Strict pyright configuration and comprehensive type hints for better code reliability.
:::

::::

## Getting Started

```{include} quickstart.md
:start-after: <!-- start-install -->
:end-before: <!-- end-install -->
```

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
installation
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/index
```

```{toctree}
:maxdepth: 2
:caption: AI Development

ai-development
claude-code-guide
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
changelog
contributing
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`