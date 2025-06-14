"""Command-line interface for the template package."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, NoReturn

from . import __version__, greet, hello

if TYPE_CHECKING:
    from collections.abc import Sequence


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="template",
        description="A modern Python project template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  template              # Print a greeting
  template --name Alice # Print a personalized greeting
  template --version    # Show version information
        """.strip(),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show program's version number and exit",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="name to greet",
        metavar="NAME",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable verbose output",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI application.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        # Treat empty string as None
        name = args.name if args.name else None
        message = greet(name) if name is not None else hello()

        print(message)

        if args.verbose:
            print(f"\nRunning template version {__version__}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run() -> NoReturn:
    """Entry point for the CLI application."""
    sys.exit(main())


if __name__ == "__main__":
    run()
