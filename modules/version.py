"""Version information for the template package.

This module provides version information using importlib.metadata when available,
falling back to a hardcoded version for development environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__: Final[str] = version("template")
    except PackageNotFoundError:
        # Package is not installed, use development version
        __version__ = "0.2.0.dev0"
except ImportError:
    # Python < 3.8 compatibility (though we require 3.12+)
    __version__ = "0.2.0"


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Args:
        version_str: Version string like "1.2.3" or "1.2.3.dev0"

    Returns:
        Tuple of version components as integers.
    """
    # Remove any dev/alpha/beta suffixes
    base_version = version_str.split(".dev")[0].split("a")[0].split("b")[0].split("rc")[0]
    return tuple(int(x) for x in base_version.split("."))


__version_tuple__: Final[tuple[int, ...]] = _parse_version(__version__)

# Useful version checks
__version_info__ = {
    "major": __version_tuple__[0] if __version_tuple__ else 0,
    "minor": __version_tuple__[1] if len(__version_tuple__) > 1 else 0,
    "patch": __version_tuple__[2] if len(__version_tuple__) > 2 else 0,
    "release": "dev" not in __version__,
}

__all__ = ["__version__", "__version_tuple__", "__version_info__"]