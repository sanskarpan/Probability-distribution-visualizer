"""Tests for the `probviz` CLI (src/cli.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import build_parser, main


def test_parser_commands():
    parser = build_parser()
    assert parser.parse_args(["app"]).command == "app"
    assert parser.parse_args(["test"]).command == "test"
    assert parser.parse_args(["version"]).command == "version"


def test_version(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out  # non-empty version string
    assert out[0].isdigit()
