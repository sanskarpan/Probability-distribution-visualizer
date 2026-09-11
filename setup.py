"""Backward-compatible setup shim.

The canonical build metadata lives in ``pyproject.toml`` (PEP 517/621).
This file exists only so ``pip install -e .`` keeps working on old
tooling that invokes ``setup.py`` directly.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
