# API Reference

This section contains the complete API documentation for Template.

## Overview

The Template API is organized into several main modules:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 📦 modules
Core package containing the main functionality
:::

:::{grid-item-card} 🔧 modules.tools
Utility tools and command-line interfaces
:::

:::{grid-item-card} ⚙️ modules.configs
Configuration management and settings
:::

:::{grid-item-card} 🚀 modules.entry_point
Main CLI entry point and command handling
:::

::::

## Auto-Generated API Documentation

The complete API documentation is automatically generated from the source code using Sphinx AutoAPI.

```{eval-rst}
.. autoapi-nested:: modules
   :prune-completely:
```

## Package Structure

### modules

The main package containing core functionality.

### modules.tools

Utility tools and scripts:
- `train.py`: Training utilities
- `copy_cfg.py`: Configuration copying utilities  
- `list_cfg.py`: Configuration listing utilities

### modules.configs

Configuration management:
- Configuration loading and validation
- Settings management
- Environment-specific configurations

### modules.entry_point

Command-line interface:
- Main CLI entry point
- Command parsing and routing
- Help and documentation

## Usage Examples

### Basic Usage

```python
import modules

# Basic module usage
result = modules.some_function(data)
```

### CLI Usage

```bash
# Show help
uv run modules --help

# Run specific commands
uv run modules command --option value
```

### Configuration

```python
from modules.configs import load_config

# Load configuration
config = load_config("path/to/config.yaml")
```

## Type Annotations

All public APIs include comprehensive type annotations for better IDE support and type safety:

```python
from typing import Dict, List, Optional
from modules.tools import process_data

def example_function(
    data: List[str], 
    options: Optional[Dict[str, str]] = None
) -> Dict[str, int]:
    return process_data(data, options)
```

## Error Handling

Template uses consistent error handling patterns:

```python
from modules.exceptions import TemplateError

try:
    result = modules.risky_operation()
except TemplateError as e:
    print(f"Template error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Contributing to the API

When contributing to Template:

1. **Add type hints** to all public functions
2. **Write comprehensive docstrings** using Google style
3. **Include usage examples** in docstrings
4. **Handle errors appropriately** with custom exceptions
5. **Update documentation** when adding new features

For more details, see the [Contributing Guide](../contributing.md).

## API Stability

Template follows [Semantic Versioning](https://semver.org/):

- **Major versions** (1.0.0 → 2.0.0): Breaking API changes
- **Minor versions** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch versions** (1.0.0 → 1.0.1): Bug fixes, backward compatible

## Getting Help

For API-related questions:

- 📖 Check the auto-generated documentation below
- 🐛 Report API issues on [GitHub](https://github.com/okotaku/template/issues)
- 💬 Ask questions in [discussions](https://github.com/okotaku/template/discussions)
- 🤖 Use [Claude Code](../claude-code-guide.md) for development assistance