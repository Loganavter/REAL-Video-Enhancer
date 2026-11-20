"""Make the suite runnable from any directory: the repo root, the backend package, and the harness.

The tests import three roots: ``harness.*`` (from the repository root), ``backend.src.*`` (from the
root) and ``src.*`` (the backend package itself, i.e. ``backend/`` on the path). Before this file
existed each invocation carried its own ``PYTHONPATH``, which is why the same suite was green in one
shell and an ImportError in another.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _path in (_ROOT, os.path.join(_ROOT, "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
