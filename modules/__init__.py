"""Template - A modern Python project template optimized for AI-assisted development.

This package provides a clean, minimal starting point for Python projects with:
- Modern tooling (uv, ruff, pyright)
- Comprehensive CI/CD pipelines
- Type safety and code quality automation
- AI-first development with Claude Code
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .version import __version__, __version_tuple__

if TYPE_CHECKING:
    from typing import Final

__all__: Final[list[str]] = [
    "__version__",
    "__version_tuple__",
    "greet",
    "hello",
]


def hello() -> str:
    """Return a greeting message.

    Returns:
        A friendly greeting string.

    Examples:
        >>> hello()
        'Hello from Template!'
    """
    return "Hello from Template!"


def greet(name: str | None = None) -> str:
    """Return a personalized greeting message.

    Args:
        name: The name to greet. If None, returns a generic greeting.

    Returns:
        A personalized or generic greeting string.

    Examples:
        >>> greet()
        'Hello there!'
        >>> greet("Alice")
        'Hello Alice!'
    """
    if name is None:
        return "Hello there!"
    return f"Hello {name}!"
