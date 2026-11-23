"""Package-anchored paths: the evidence corpus lives in ``harness/docs/``.

While the corpus sat at the repository root every default was cwd-relative, so the CLI only ran from
the checkout root. Anchoring on this file makes ``python -m harness.cli`` independent of where it is
started. A locator that still begins with ``docs/`` is read as ``harness/docs/`` - records taken before
the move keep the provenance string they were written with, and it still resolves.

The plans, research, measurements, sessions and notes left this repository for the framework's
knowledge plane (``SLI-framework/projects/rve/docs/``), so a locator that begins with ``projects/``
is resolved against that checkout when it is present and left untouched when it is not.
"""
from __future__ import annotations

import os
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
DOCS = HARNESS / "docs"
ROOT = HARNESS.parent

#: The framework checkout that holds the knowledge plane; ``RVE_FRAMEWORK_ROOT`` overrides the
#: sibling layout this machine uses (``~/Загрузки/projects/SLI-framework``).
FRAMEWORK = Path(os.environ.get("RVE_FRAMEWORK_ROOT") or ROOT.parent / "SLI-framework")


def doc(rel: str) -> str:
    """Absolute path of a file inside ``harness/docs/``."""
    return str(DOCS / rel)


def resolve(path: str) -> str:
    """Resolve a locator from a record, a verdict or the command line.

    Absolute paths pass through; ``docs/...`` means ``harness/docs/...``; ``projects/...`` means a
    document in the framework's knowledge plane (resolved against the checkout when it exists);
    anything else is tried against the repository root first and left as-is when it does not exist
    there.
    """
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if path.startswith("docs/"):
        return str(DOCS / path[len("docs/"):])
    if path.startswith("projects/"):
        plane = FRAMEWORK / path
        return str(plane) if plane.exists() else path
    candidate = ROOT / path
    return str(candidate) if candidate.exists() else path
