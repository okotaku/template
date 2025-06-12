"""Template - A modern Python project template."""

from .version import __version__

__all__ = ["__version__"]


def hello() -> str:
    """Return a greeting message."""
    return "Hello from Template!"
