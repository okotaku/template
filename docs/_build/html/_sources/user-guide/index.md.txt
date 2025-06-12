# User Guide

Welcome to the Template User Guide.

## Overview

Template is a modern Python project template designed for AI-assisted development with Claude Code. It provides:

- **Fast Development Environment**: Using `uv` for dependency management
- **Modern Tooling**: Pre-configured with ruff, pyright, pytest, and more
- **AI Integration**: Optimized for Claude Code development workflows
- **Simple Structure**: Clean, minimal codebase

## Getting Started

1. **[Installation](../installation.md)**: Install Template and its dependencies
2. **[Quick Start](../quickstart.md)**: Get up and running in minutes
3. **[Claude Code Guide](../claude-code-guide.md)**: Set up AI-assisted development

## Key Concepts

### Project Structure

Template follows a clean, minimal Python project structure:

```
template/
├── modules/              # Main package
│   ├── __init__.py      # Package with hello() function
│   └── version.py       # Version information
├── tests/               # Test files
│   ├── __init__.py
│   └── test_basic.py    # Basic tests
├── docs/                # Documentation
├── pyproject.toml       # Project configuration
├── uv.lock             # Dependency lock file
├── CLAUDE.md           # AI development context
└── README.md           # Project overview
```

### Development Philosophy

Template is built around several key principles:

1. **AI-First**: Designed for AI-assisted development workflows
2. **Modern Tools**: Uses the latest Python tooling and best practices
3. **Type Safety**: Comprehensive type hints and pyright integration
4. **Quality**: Automated code quality checks and testing
5. **Simplicity**: Clear structure and straightforward workflows

### Tool Integration

Template integrates several modern Python tools:

- **[uv](https://docs.astral.sh/uv/)**: Fast package manager and virtual environment tool
- **[ruff](https://docs.astral.sh/ruff/)**: Extremely fast Python linter and formatter
- **[pyright](https://pyright.readthedocs.io/)**: Static type checker
- **[pytest](https://pytest.org/)**: Testing framework
- **[pre-commit](https://pre-commit.com/)**: Git hooks for code quality

## Common Workflows

### Adding a New Feature

1. **Plan the feature** with AI assistance
2. **Generate initial implementation**
3. **Add comprehensive tests**
4. **Update documentation**
5. **Run quality checks**

### Fixing a Bug

1. **Reproduce the issue**
2. **Write a failing test**
3. **Implement the fix**
4. **Verify the test passes**
5. **Run full test suite**

### Refactoring Code

1. **Ensure good test coverage**
2. **Plan the refactoring approach**
3. **Make incremental changes**
4. **Verify tests still pass**
5. **Update documentation**

## Best Practices

### Code Quality

- **Use type hints**: Add type annotations to all functions and methods
- **Write docstrings**: Document public APIs with Google-style docstrings
- **Follow conventions**: Use consistent naming and code style
- **Test thoroughly**: Aim for high test coverage

### Development Workflow

- **Use AI assistance**: Leverage Claude Code for faster development
- **Commit frequently**: Make small, focused commits
- **Run checks early**: Use pre-commit hooks to catch issues
- **Document changes**: Update relevant documentation

### Project Management

- **Keep dependencies updated**: Use automated dependency updates
- **Monitor code quality**: Track metrics over time
- **Maintain documentation**: Keep docs in sync with code
- **Use semantic versioning**: Follow semver for releases

## Advanced Topics

### Custom Configurations

Learn how to customize Template for specific use cases:

- **Environment-specific settings**
- **Custom tool configurations**
- **Integration with other tools**
- **CI/CD customizations**

### Performance Optimization

Techniques for optimizing Template-based projects:

- **Profiling and benchmarking**
- **Memory usage optimization**
- **Parallel processing**
- **Caching strategies**

### Deployment

Best practices for deploying Template-based applications:

- **Docker containerization**
- **Cloud deployment**
- **Environment management**
- **Monitoring and logging**

## Getting Help

If you need help:

- 📖 Check the relevant sections in this guide
- 🐛 Search [existing issues](https://github.com/okotaku/template/issues)
- 💬 Ask in [discussions](https://github.com/okotaku/template/discussions)
- 🤖 Use [Claude Code](../claude-code-guide.md) for development questions
- 📝 Create a [new issue](https://github.com/okotaku/template/issues/new/choose)

## Contributing

Want to improve Template? Check out our [Contributing Guide](../contributing.md) to learn how to:

- Report bugs and suggest features
- Contribute code improvements
- Write and improve documentation
- Help with testing and quality assurance

## Next Steps

Choose your path based on what you want to do:

- **New to Template?** → Start with the [Quick Start Guide](../quickstart.md)
- **Setting up development?** → Read the [Development Guide](development.md)
- **Having issues?** → Check [Troubleshooting](troubleshooting.md)
- **Using AI tools?** → See the [Claude Code Guide](../claude-code-guide.md)