"""Command-line interface for the Probability Distribution Visualizer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run_streamlit(extra_args: list[str] | None = None) -> int:
    """Launch the Streamlit web app."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(repo_root, "web", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd)


def _run_tests(extra_args: list[str] | None = None) -> int:
    """Run the test suite."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probviz",
        description="Probability Distribution Visualizer CLI",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    p_app = sub.add_parser("app", help="Launch the Streamlit web app")
    p_app.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to Streamlit")

    p_test = sub.add_parser("test", help="Run the test suite")
    p_test.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to pytest")

    sub.add_parser("version", help="Print the package version")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "app":
        return _run_streamlit(args.extra)
    if args.command == "test":
        return _run_tests(args.extra)
    if args.command == "version":
        from src import __version__

        print(__version__)
        return 0
    # Default: launch the app (preserves old `probviz` behaviour of opening the UI).
    return _run_streamlit()


if __name__ == "__main__":
    raise SystemExit(main())
