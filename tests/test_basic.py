"""Basic tests for the template package."""

import modules


def test_version() -> None:
    """Test that version is accessible."""
    assert hasattr(modules, "__version__")  # noqa: S101
    assert isinstance(modules.__version__, str)  # noqa: S101


def test_hello() -> None:
    """Test the hello function."""
    result = modules.hello()
    assert result == "Hello from Template!"  # noqa: S101
    assert isinstance(result, str)  # noqa: S101
