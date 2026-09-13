"""Command-line interface for the Probability Distribution Visualizer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _find_app_path() -> str | None:
    """Locate web/app.py from either source checkout or pip install."""
    # Try installed package first: web is a top-level package.
    try:
        import web  # type: ignore[import-untyped]

        cand = os.path.join(os.path.dirname(web.__file__), "app.py")
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    # Fall back to source checkout layout: src/cli.py -> repo_root/web/app.py
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(repo_root, "web", "app.py")
    if os.path.isfile(cand):
        return cand
    # Last attempt: cwd/web/app.py
    cand = os.path.join(os.getcwd(), "web", "app.py")
    if os.path.isfile(cand):
        return cand
    return None


def _find_tests_dir() -> str | None:
    """Locate tests/ from either source checkout or pip install."""
    candidates: list[str] = []
    # Source checkout: repo_root/tests
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(repo_root, "tests"))
    except Exception:
        pass
    candidates.append(os.path.join(os.getcwd(), "tests"))
    # Installed tests package (if shipped)
    try:
        import tests  # type: ignore[import-untyped]

        candidates.append(os.path.dirname(tests.__file__))  # type: ignore[attr-defined]
    except Exception:
        pass
    for cand in candidates:
        if os.path.isdir(cand):
            # must contain at least one test file
            try:
                if any(f.startswith("test_") for f in os.listdir(cand)):
                    return cand
            except OSError:
                continue
    return None


def _run_streamlit(extra_args: list[str] | None = None) -> int:
    """Launch the Streamlit web app."""
    app_path = _find_app_path()
    if app_path is None:
        print(
            "error: could not locate web/app.py. "
            "If installed via pip, ensure the 'web' package is present; "
            "otherwise run from the repository checkout.",
            file=sys.stderr,
        )
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd)


def _run_tests(extra_args: list[str] | None = None) -> int:
    """Run the test suite."""
    # Check pytest is available before trying to discover tests/
    try:
        import pytest  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        print(
            "error: pytest is not installed. Install test extras with:\n"
            "  pip install 'probviz[test]'  # or: pip install pytest",
            file=sys.stderr,
        )
        return 1
    tests_dir = _find_tests_dir()
    if tests_dir is None:
        print(
            "error: could not locate tests/ directory. "
            "If installed via pip, the tests are not shipped in this build; "
            "clone https://github.com/sanskarpan/probviz and run from source, "
            "or install with tests: pip install 'probviz[test]' and run pytest directly.",
            file=sys.stderr,
        )
        return 1
    cmd = [sys.executable, "-m", "pytest", tests_dir, "-q"]
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
    # Use parse_known_args so that flags like --server.headless (forwarded to
    # streamlit via REMAINDER) are not rejected as "unrecognized arguments".
    # REMAINDER already captures most cases, but parse_known_args is a safety
    # net for invocations where argparse would otherwise error before reaching
    # the subparser.
    args, unknown = parser.parse_known_args(argv)

    # If unknown remains and we have a subcommand that accepts REMAINDER, merge
    # it into extra so users can do `probviz app --server.port 8501`.
    if args.command in ("app", "test") and unknown:
        # args.extra is a list (may be empty); extend with unknown.
        extra = list(getattr(args, "extra", []) or [])
        extra.extend(unknown)
        args.extra = extra  # type: ignore[attr-defined]
    elif unknown:
        parser.print_usage(sys.stderr)
        print(f"probviz: error: unrecognized arguments: {' '.join(unknown)}", file=sys.stderr)
        return 2

    if args.command == "app":
        return _run_streamlit(args.extra)  # type: ignore[arg-type]
    if args.command == "test":
        return _run_tests(args.extra)  # type: ignore[arg-type]
    if args.command == "version":
        from src import __version__

        print(__version__)
        return 0
    # Default: launch the app (preserves old `probviz` behaviour of opening the UI).
    return _run_streamlit()


if __name__ == "__main__":
    raise SystemExit(main())
