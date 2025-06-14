"""Basic tests for the template package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import modules
from modules import __version__, __version_tuple__, greet, hello
from modules.version import __version_info__

if TYPE_CHECKING:
    from collections.abc import Callable


class TestVersion:
    """Test version-related functionality."""

    def test_version_exists(self) -> None:
        """Test that version is accessible."""
        assert hasattr(modules, "__version__")
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_tuple(self) -> None:
        """Test version tuple parsing."""
        assert hasattr(modules, "__version_tuple__")
        assert isinstance(__version_tuple__, tuple)
        assert all(isinstance(x, int) for x in __version_tuple__)
        assert len(__version_tuple__) >= 2  # At least major.minor

    def test_version_info(self) -> None:
        """Test version info dictionary."""
        assert isinstance(__version_info__, dict)
        assert "major" in __version_info__
        assert "minor" in __version_info__
        assert "patch" in __version_info__
        assert "release" in __version_info__
        assert isinstance(__version_info__["major"], int)
        assert isinstance(__version_info__["minor"], int)
        assert isinstance(__version_info__["patch"], int)
        assert isinstance(__version_info__["release"], bool)


class TestGreetings:
    """Test greeting functions."""

    def test_hello(self) -> None:
        """Test the hello function."""
        result = hello()
        assert result == "Hello from Template!"
        assert isinstance(result, str)

    def test_greet_without_name(self) -> None:
        """Test greet function without a name."""
        result = greet()
        assert result == "Hello there!"
        assert isinstance(result, str)

    def test_greet_with_name(self) -> None:
        """Test greet function with a name."""
        result = greet("Alice")
        assert result == "Hello Alice!"
        assert isinstance(result, str)

    def test_greet_with_none(self) -> None:
        """Test greet function with explicit None."""
        result = greet(None)
        assert result == "Hello there!"
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("", "Hello !"),
            ("Alice", "Hello Alice!"),
            ("Bob", "Hello Bob!"),
            ("世界", "Hello 世界!"),
            ("123", "Hello 123!"),
            ("Alice Smith", "Hello Alice Smith!"),
        ],
    )
    def test_greet_various_names(self, name: str, expected: str) -> None:
        """Test greet function with various names."""
        assert greet(name) == expected


class TestModuleExports:
    """Test module exports and public API."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        assert hasattr(modules, "__all__")
        expected_exports = {"__version__", "__version_tuple__", "hello", "greet"}
        actual_exports = set(modules.__all__)
        assert expected_exports == actual_exports

    def test_all_exports_exist(self) -> None:
        """Test that all exports in __all__ actually exist."""
        for name in modules.__all__:
            assert hasattr(modules, name), f"Export {name} not found in module"

    def test_exported_items_are_callable_or_str(self) -> None:
        """Test that exported items have expected types."""
        type_checks: dict[str, type | tuple[type, ...]] = {
            "__version__": str,
            "__version_tuple__": tuple,
            "hello": Callable,
            "greet": Callable,
        }
        for name, expected_type in type_checks.items():
            item = getattr(modules, name)
            if expected_type is Callable:
                assert callable(item), f"{name} should be callable"
            else:
                assert isinstance(
                    item, expected_type
                ), f"{name} should be {expected_type}"


@pytest.mark.unit
class TestTypeAnnotations:
    """Test that functions have proper type annotations."""

    def test_hello_annotations(self) -> None:
        """Test hello function annotations."""
        annotations = hello.__annotations__
        assert "return" in annotations
        assert annotations["return"] == str

    def test_greet_annotations(self) -> None:
        """Test greet function annotations."""
        annotations = greet.__annotations__
        assert "name" in annotations
        assert "return" in annotations
        assert annotations["return"] == str
        # Check that name accepts str | None
        assert str(annotations["name"]) in ["str | None", "typing.Union[str, None]"]
