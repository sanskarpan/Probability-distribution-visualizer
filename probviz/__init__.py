"""Public namespace for the probviz distribution."""

import sys
from importlib import import_module

from src import __version__

_MODULES = (
    "distributions",
    "fitting",
    "monte_carlo",
    "statistical_tests",
    "utils",
    "visualizers",
)

for _module in _MODULES:
    sys.modules[f"{__name__}.{_module}"] = import_module(f"src.{_module}")

__all__ = ["__version__"]
