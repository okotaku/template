# Claude Code Setup Guide

This guide walks you through setting up and using Claude Code with the Template project.

## What is Claude Code?

[Claude Code](https://claude.ai/code) is Anthropic's official CLI tool that brings Claude's AI capabilities directly to your development environment. It's designed for:

- **Code generation and editing**
- **Project understanding and navigation** 
- **Testing and debugging assistance**
- **Documentation and refactoring**

## Installation

### Prerequisites

- A Claude account (sign up at [claude.ai](https://claude.ai))
- Command line access
- Template project (see [Installation Guide](installation.md))

### Install Claude Code

Follow the official installation instructions at [claude.ai/code](https://claude.ai/code).

### Authentication

After installation, authenticate with your Claude account:

```bash
claude-code auth login
```

## Project Setup

### 1. Initialize Claude Code

In your Template project directory:

```bash
claude-code init
```

This creates the necessary configuration files for Claude Code to understand your project.

### 2. Verify CLAUDE.md

Template includes a pre-configured `CLAUDE.md` file that provides comprehensive project context. Verify it's present and up-to-date:

```bash
cat CLAUDE.md
```

The file should contain:
- Project overview and structure
- Development workflows
- Tool configurations
- Best practices
- Common commands

### 3. Test the Setup

Verify Claude Code can understand your project:

```bash
claude-code "Explain the project structure"
```

## Basic Usage

### Getting Help

```bash
# General help
claude-code --help

# Ask about the project
claude-code "What does this project do?"

# Get development guidance
claude-code "How do I add a new feature?"
```

### Code Operations

#### Generate Code
```bash
# Generate a new module
claude-code "Create a data validation module with type hints"

# Add a new function
claude-code "Add a function to process CSV files in modules/tools/"
```

#### Review and Improve Code
```bash
# Review existing code
claude-code "Review modules/entry_point.py for improvements"

# Optimize performance
claude-code "Optimize the data processing in modules/tools/train.py"
```

#### Fix Issues
```bash
# Debug errors
claude-code "Fix the import error in tests/test_models/"

# Resolve type issues
claude-code "Fix pyright errors in modules/"
```

### Testing Operations

#### Generate Tests
```bash
# Create tests for a module
claude-code "Generate comprehensive tests for modules/tools/list_cfg.py"

# Add edge case tests
claude-code "Add edge case tests for the data validation module"
```

#### Debug Test Failures
```bash
# Understand test failures
claude-code "Why is test_train.py failing?"

# Fix broken tests
claude-code "Fix the failing tests in tests/test_datasets/"
```

### Documentation Operations

#### Generate Documentation
```bash
# Document a function
claude-code "Add comprehensive docstrings to modules/tools/train.py"

# Create usage examples
claude-code "Create usage examples for the CLI interface"
```

#### Update Documentation
```bash
# Update README for new features
claude-code "Update README.md to include the new validation module"

# Update API docs
claude-code "Update the API documentation for recent changes"
```

## Advanced Workflows

### Feature Development Workflow

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

### Refactoring Workflow

1. **Analyze Current Code**
   ```bash
   claude-code "Analyze modules/ for refactoring opportunities"
   ```

2. **Plan Refactoring**
   ```bash
   claude-code "Create a refactoring plan for better separation of concerns"
   ```

3. **Execute Refactoring**
   ```bash
   claude-code "Refactor modules/tools/ following the planned approach"
   ```

4. **Update Tests**
   ```bash
   claude-code "Update tests to match the refactored code structure"
   ```

5. **Verify Quality**
   ```bash
   uv run ruff check .
   uv run pyright modules
   uv run pytest
   ```

### Debugging Workflow

1. **Understand the Problem**
   ```bash
   claude-code "Help me understand this error: [paste error message]"
   ```

2. **Analyze Context**
   ```bash
   claude-code "What could cause this issue in modules/entry_point.py?"
   ```

3. **Generate Solutions**
   ```bash
   claude-code "Fix the configuration loading error"
   ```

4. **Test the Fix**
   ```bash
   claude-code "Create a test to verify the fix works correctly"
   ```

## Integration with Template Tools

### With uv

Claude Code understands Template's uv configuration:

```bash
# Add dependencies
claude-code "Add pandas as a dependency using uv"

# Update dependencies
claude-code "Update the development dependencies"
```

### With ruff

Use Claude Code to fix linting issues:

```bash
# Fix ruff issues
claude-code "Fix all ruff linting errors in modules/"

# Optimize imports
claude-code "Optimize imports according to ruff configuration"
```

### With pyright

Get help with type issues:

```bash
# Fix type errors
claude-code "Resolve pyright type errors in modules/tools/"

# Add type hints
claude-code "Add comprehensive type hints to modules/version.py"
```

### With pytest

Improve your tests:

```bash
# Fix failing tests
claude-code "Debug and fix the failing test in tests/test_tools/"

# Improve test coverage
claude-code "Add tests to improve coverage for modules/configs/"
```

## Best Practices

### 1. Provide Clear Context

**Good prompts**:
- "Create a data validator following the pattern in modules/tools/train.py"
- "Fix the type error in line 45 of modules/entry_point.py"
- "Add comprehensive tests for the CSV processing functionality"

**Avoid vague prompts**:
- "Fix this code"
- "Make it better"
- "Add some tests"

### 2. Reference Project Patterns

Always reference existing patterns:
```bash
# Good
claude-code "Add error handling following the pattern in modules/tools/copy_cfg.py"

# Better
claude-code "Add configuration validation using the same error handling pattern as modules/tools/copy_cfg.py, with custom ValidationError types"
```

### 3. Validate AI Suggestions

Always run quality checks after AI assistance:

```bash
# After any code changes
uv run ruff check .
uv run ruff format .
uv run pyright modules
uv run pytest

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### 4. Iterate and Refine

Use AI assistance iteratively:

1. Start with a basic implementation
2. Add error handling and edge cases
3. Optimize for performance
4. Add comprehensive tests
5. Generate documentation

### 5. Keep CLAUDE.md Updated

When you add new features or change workflows:

```bash
claude-code "Update CLAUDE.md to include the new plugin system"
```

## Troubleshooting

### Common Issues

**Claude Code doesn't understand the project**
- Ensure CLAUDE.md is present and up-to-date
- Run `claude-code init` in the project root
- Check that you're in the correct directory

**Generated code doesn't follow project patterns**
- Reference specific files when asking for help
- Update CLAUDE.md with new patterns
- Provide more specific context in prompts

**Quality checks fail after AI assistance**
- Run checks incrementally during development
- Use smaller, focused changes
- Always validate before committing

**AI suggestions are inconsistent**
- Provide consistent context across sessions
- Reference the same patterns and conventions
- Keep project documentation updated

### Getting Better Results

1. **Start with project understanding**
   ```bash
   claude-code "Explain how this project is organized"
   ```

2. **Reference existing code**
   ```bash
   claude-code "Follow the pattern in modules/tools/train.py"
   ```

3. **Be specific about requirements**
   ```bash
   claude-code "Add type hints, error handling, and docstrings"
   ```

4. **Validate incrementally**
   ```bash
   # After each AI suggestion
   uv run ruff check . && uv run pyright modules
   ```

## Advanced Configuration

### Custom Prompts

Create reusable prompts for common tasks:

```bash
# Create a new feature following project conventions
alias new-feature="claude-code 'Create a new feature following Template project conventions with type hints, tests, and documentation'"

# Code review with project standards
alias review-code="claude-code 'Review this code for Template project standards including type safety, error handling, and documentation'"
```

### Integration with Git

Use Claude Code for git-related tasks:

```bash
# Generate commit messages
claude-code "Generate a commit message for these changes"

# Create PR descriptions
claude-code "Create a PR description for the new validation feature"
```

## Next Steps

- 🚀 Try the [development workflows](user-guide/development.md)
- 🧪 Learn about [testing with AI](user-guide/testing.md)
- 📖 Explore [advanced AI development patterns](ai-development.md)
- 🤝 See [contributing with AI assistance](contributing.md)

## Resources

- [Claude Code Documentation](https://claude.ai/code)
- [Template Project Structure](user-guide/index.md)
- [Development Best Practices](user-guide/development.md)
- [Troubleshooting Guide](user-guide/troubleshooting.md)