# Installation Guide

This guide covers different ways to install Template and its dependencies.

## Requirements

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, or Linux
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended)

## Installing uv

Template uses `uv` for fast and reliable package management. If you don't have it installed:

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

:::{tab-item} Homebrew (macOS)
```bash
brew install uv
```
:::

:::{tab-item} pip
```bash
pip install uv
```
:::

::::

Verify the installation:

```bash
uv --version
```

## Installing Template

### Option 1: User Installation

Install Template as a user package:

```bash
uv add git+https://github.com/okotaku/template.git
```

This installs Template in your current Python environment.

### Option 2: Development Installation

For development or if you want to modify Template:

```bash
# Clone the repository
git clone https://github.com/okotaku/template.git
cd template

# Install with all development dependencies
uv sync --all-extras
```

### Option 3: Specific Version

Install a specific version or branch:

```bash
# Install specific tag
uv add git+https://github.com/okotaku/template.git@v0.1.0

# Install specific branch
uv add git+https://github.com/okotaku/template.git@main
```

## Dependency Groups

Template organizes dependencies into groups for different use cases:

### Core Dependencies

Installed by default:
- `torch`: Deep learning framework
- `torchvision`: Computer vision utilities  
- `mmengine`: Core engine framework

### Development Dependencies

Install with `--extra dev`:
```bash
uv sync --extra dev
```

Includes:
- `pytest`: Testing framework
- `coverage`: Code coverage
- `ruff`: Linting and formatting
- `pyright`: Type checking
- `pre-commit`: Git hooks

### Documentation Dependencies

Install with `--extra docs`:
```bash
uv sync --extra docs
```

Includes:
- `sphinx`: Documentation generator
- `furo`: Modern Sphinx theme
- `myst-parser`: Markdown support

### All Dependencies

Install everything:
```bash
uv sync --all-extras
```

## Optional: PyTorch with CUDA

For GPU support, install PyTorch with CUDA:

```bash
# CUDA 12.1 (recommended)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Verification

Verify your installation:

```bash
# Check Template installation
uv run python -c "import modules; print('Template installed successfully!')"

# Check PyTorch installation
uv run python -c "import torch; print(f'PyTorch {torch.__version__} installed')"

# Check CUDA availability (if installed)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Environment Management

### Virtual Environments

uv automatically manages virtual environments, but you can create them explicitly:

```bash
# Create a new environment
uv venv template-env

# Activate (Unix)
source template-env/bin/activate

# Activate (Windows)
template-env\Scripts\activate

# Install Template in the environment
uv pip install git+https://github.com/okotaku/template.git
```

### Project Environments

For project-specific environments:

```bash
# Create project with Template
mkdir my-project
cd my-project
uv init
uv add git+https://github.com/okotaku/template.git
```

## Docker Installation

Use the provided Dockerfile for containerized environments:

```bash
# Build the image
docker build -t template .

# Run a container
docker run -it template bash
```

The Docker image includes:
- Python 3.13
- uv package manager
- All Template dependencies
- Development tools

## Troubleshooting

### Common Issues

**uv command not found**
- Ensure uv is installed and in your PATH
- Restart your terminal after installation
- On Windows, you might need to restart your entire system

**Permission errors**
- On Linux/macOS, avoid using `sudo` with uv
- On Windows, run as administrator if needed
- Check file permissions in your project directory

**Network issues**
- If behind a corporate firewall, configure proxy settings
- Use `--trusted-host` for internal PyPI mirrors
- Check your internet connection

**Import errors**
- Verify Template is installed: `uv list | grep modules`
- Check your Python path: `uv run python -c "import sys; print(sys.path)"`
- Ensure you're in the correct environment

### Getting Help

If you encounter issues:

1. Check the [troubleshooting section](user-guide/troubleshooting.md)
2. Search [existing issues](https://github.com/okotaku/template/issues)
3. Create a [new issue](https://github.com/okotaku/template/issues/new/choose)
4. Ask in [discussions](https://github.com/okotaku/template/discussions)

### Performance Tips

- Use `uv` instead of `pip` for faster installation
- Enable dependency caching: `export UV_CACHE_DIR=~/.uv-cache`
- Use `--no-dev` flag to skip development dependencies in production
- Consider using `--frozen` for reproducible builds

## Next Steps

After installation:

1. 📖 Read the [Quick Start Guide](quickstart.md)
2. 🤖 Set up [Claude Code integration](claude-code-guide.md)
3. 🛠️ Explore the [User Guide](user-guide/index.md)
4. 🧪 Run the test suite: `uv run pytest`