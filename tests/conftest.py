"""Pytest configuration: ensure the ``src`` layout is importable.

Tests historically used per-file ``sys.path.insert`` hacks. This conftest
provides the same path setup centrally so new tests can use plain
``from src.... import ...`` imports.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
