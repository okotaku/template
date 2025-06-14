"""Tests for the CLI module."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from modules import __version__
from modules.cli import create_parser, main

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


class TestCLIParser:
    """Test the argument parser."""

    def test_create_parser(self) -> None:
        """Test parser creation."""
        parser = create_parser()
        assert parser.prog == "template"
        assert parser.description == "A modern Python project template"

    def test_parser_version(self) -> None:
        """Test version argument."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_parser_name(self) -> None:
        """Test name argument."""
        parser = create_parser()
        args = parser.parse_args(["--name", "Alice"])
        assert args.name == "Alice"

    def test_parser_name_short(self) -> None:
        """Test short name argument."""
        parser = create_parser()
        args = parser.parse_args(["-n", "Bob"])
        assert args.name == "Bob"

    def test_parser_verbose(self) -> None:
        """Test verbose argument."""
        parser = create_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

        args = parser.parse_args(["-v"])
        assert args.verbose is True

        args = parser.parse_args([])
        assert args.verbose is False


class TestCLIMain:
    """Test the main CLI function."""

    def test_main_no_args(self, capsys: CaptureFixture[str]) -> None:
        """Test main with no arguments."""
        exit_code = main([])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert captured.out == "Hello from Template!\n"
        assert captured.err == ""

    def test_main_with_name(self, capsys: CaptureFixture[str]) -> None:
        """Test main with name argument."""
        exit_code = main(["--name", "Alice"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert captured.out == "Hello Alice!\n"
        assert captured.err == ""

    def test_main_with_name_short(self, capsys: CaptureFixture[str]) -> None:
        """Test main with short name argument."""
        exit_code = main(["-n", "Bob"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert captured.out == "Hello Bob!\n"
        assert captured.err == ""

    def test_main_verbose(self, capsys: CaptureFixture[str]) -> None:
        """Test main with verbose flag."""
        exit_code = main(["--verbose"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Hello from Template!" in captured.out
        assert f"Running template version {__version__}" in captured.out
        assert captured.err == ""

    def test_main_name_and_verbose(self, capsys: CaptureFixture[str]) -> None:
        """Test main with both name and verbose."""
        exit_code = main(["--name", "Charlie", "--verbose"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Hello Charlie!" in captured.out
        assert f"Running template version {__version__}" in captured.out
        assert captured.err == ""

    def test_main_keyboard_interrupt(
        self,
        monkeypatch: MonkeyPatch,
        capsys: CaptureFixture[str],
    ) -> None:
        """Test handling of keyboard interrupt."""

        # Mock hello to raise KeyboardInterrupt instead of mocking print
        def mock_hello() -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("modules.cli.hello", mock_hello)
        exit_code = main([])
        assert exit_code == 130
        captured = capsys.readouterr()
        assert "Interrupted by user" in captured.err

    def test_main_exception(
        self,
        monkeypatch: MonkeyPatch,
        capsys: CaptureFixture[str],
    ) -> None:
        """Test handling of general exceptions."""

        def mock_hello() -> str:
            raise RuntimeError("Test error")

        monkeypatch.setattr("modules.cli.hello", mock_hello)
        exit_code = main([])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err

    def test_main_with_sys_argv(
        self,
        monkeypatch: MonkeyPatch,
        capsys: CaptureFixture[str],
    ) -> None:
        """Test main using sys.argv."""
        monkeypatch.setattr(sys, "argv", ["template", "--name", "David"])
        exit_code = main()
        assert exit_code == 0
        captured = capsys.readouterr()
        assert captured.out == "Hello David!\n"


@pytest.mark.parametrize(
    ("args", "expected_output"),
    [
        ([], "Hello from Template!"),
        (["--name", "Eve"], "Hello Eve!"),
        (["-n", "Frank"], "Hello Frank!"),
        (["--name", ""], "Hello from Template!"),  # Empty name is treated as None
        (["--name", "123"], "Hello 123!"),
        (["--name", "Alice Smith"], "Hello Alice Smith!"),
    ],
)
def test_main_various_inputs(
    args: list[str],
    expected_output: str,
    capsys: CaptureFixture[str],
) -> None:
    """Test main with various input combinations."""
    exit_code = main(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert expected_output in captured.out
