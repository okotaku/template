"""Version information."""

try:
    from importlib.metadata import version

    __version__ = version("template")
except ImportError:
    __version__ = "0.1.0"
