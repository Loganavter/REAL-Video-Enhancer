"""The two rules around reading a video's metadata, kept free of Qt so they can be tested anywhere.

Both rules came from one traceback (owner, 2026-11-22):

    SettingsTab.updateFFMpegCommand -> VideoLoader.getData -> total_frames / fps -> ZeroDivisionError

* the settings page re-probes the input on every encoder change, and the field can hold the batch sentinel
  (`{ MULTIPLE_FILES }`), an empty string or a folder - the backend answers none of those, so `fps` came back as `0.0`;
* `getData()` divided by that `fps` without asking whether it was there.

Python-only on purpose: `src/constants.py` imports PySide6, so anything reachable from there cannot be exercised by the
portable harness interpreter.
"""

from __future__ import annotations

import os


def is_probeable_input(path: str) -> bool:
    """True when `path` is something the backend can be asked about.

    A URL counts (the backend fetches it and `loadVideo` accepts one); the batch sentinel, an empty field, a bare
    separator and a folder do not.
    """
    text = (path or "").strip()
    if len(text) < 2 or "{" in text or "}" in text:
        return False
    if text.startswith(("http://", "https://")):
        return True
    return os.path.isfile(text)


def duration_seconds(total_frames: int, fps: float) -> float:
    """`frames / fps`, or ``0.0`` when the probe reported no frame rate.

    Zero fps means "the backend could not tell us", not "an infinitely short video": the division raised
    ZeroDivisionError through the settings page and, on the single-file path, through `loadVideo`.
    """
    try:
        frames = int(total_frames)
        rate = float(fps)
    except (TypeError, ValueError):
        return 0.0
    if not rate:
        return 0.0
    return frames / rate
