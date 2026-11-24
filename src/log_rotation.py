"""Keep the last N runs of the app log instead of only one.

Why this exists (2026-11-25): `log.txt` used to be truncated at import, and the first fix kept exactly ONE previous run
(`log.prev.txt`). Two harness runs 33 s apart then rotated the only backend seek breakdown of a session out of existence
and a measurement landed on air. The seek profile needs the run that just happened AND the one before it, because a
rebuild is often what made it misbehave.

Pure filesystem, no imports from the app: the module is called once at `src/Util.py` import time and tested on its own.
"""

from __future__ import annotations

import os
import pathlib


def rotate(path: str | pathlib.Path, *, generations: int = 2) -> list[pathlib.Path]:
    """Shift `path` into `path+'.prev'`, `path+'.prev2'`, ... and leave `path` missing for the caller to recreate.

    Returns the files it moved, newest first. A missing or empty `path` moves nothing: an empty log (a run that died
    before its first line) is not worth a generation.
    """
    source = pathlib.Path(path)
    keep = max(1, int(generations))
    moved: list[pathlib.Path] = []
    try:
        if not source.exists() or source.stat().st_size == 0:
            return moved
    except OSError:
        return moved
    # Oldest first: the last generation falls off the end.
    for index in range(keep - 1, 0, -1):
        older = source.with_name(f"{source.name}.prev{'' if index == 1 else index}")
        newer = source.with_name(f"{source.name}.prev{'' if index == 0 else index + 1}")
        if older.exists():
            try:
                os.replace(older, newer)
                moved.append(newer)
            except OSError:
                pass
    previous = source.with_name(f"{source.name}.prev")
    try:
        os.replace(source, previous)
        moved.append(previous)
    except OSError:
        pass
    return moved
