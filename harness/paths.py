"""Package-anchored paths: the evidence corpus lives in ``harness/docs/``.

While the corpus sat at the repository root every default was cwd-relative, so the CLI only ran from
the checkout root. Anchoring on this file makes ``python -m harness.cli`` independent of where it is
started. A locator that still begins with ``docs/`` is read as ``harness/docs/`` - records taken before
the move keep the provenance string they were written with, and it still resolves.
"""
from __future__ import annotations

from pathlib import Path

HARNESS = Path(__file__).resolve().parent
DOCS = HARNESS / "docs"
ROOT = HARNESS.parent


def doc(rel: str) -> str:
    """Absolute path of a file inside ``harness/docs/``."""
    return str(DOCS / rel)


def resolve(path: str) -> str:
    """Resolve a locator from a record, a verdict or the command line.

    Absolute paths pass through; ``docs/...`` means ``harness/docs/...``; anything else is tried
    against the repository root first and left as-is when it does not exist there.
    """
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if path.startswith("docs/"):
        return str(DOCS / path[len("docs/"):])
    candidate = ROOT / path
    return str(candidate) if candidate.exists() else path
