"""The log keeps two runs, not one - the evidence the seek profile needs survives a rebuild.

Why this exists: `log.txt` used to be truncated at import (the owner's held 147 bytes while `preview-log.txt` had three
hours of history), then it kept exactly one previous run - and two harness runs 33 s apart rotated the only backend seek
breakdown of a session into oblivion. A measurement that lands on air is not a measurement.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.log_rotation import rotate  # noqa: E402


def _write(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_two_generations_survive_two_runs(tmp_path):
    log = tmp_path / "log.txt"
    _write(log, "run 1\n")
    rotate(log, generations=2)
    assert not log.exists(), "the caller recreates it"
    _write(log, "run 2\n")
    rotate(log, generations=2)
    _write(log, "run 3\n")

    assert log.read_text(encoding="utf-8") == "run 3\n"
    assert (tmp_path / "log.txt.prev").read_text(encoding="utf-8") == "run 2\n"
    assert (tmp_path / "log.txt.prev2").read_text(encoding="utf-8") == "run 1\n"


def test_the_third_run_drops_the_oldest(tmp_path):
    log = tmp_path / "log.txt"
    for text in ("run 1\n", "run 2\n", "run 3\n"):
        _write(log, text)
        rotate(log, generations=2)
        log.unlink(missing_ok=True)
    _write(log, "run 4\n")

    assert (tmp_path / "log.txt.prev").read_text(encoding="utf-8") == "run 3\n"
    assert (tmp_path / "log.txt.prev2").read_text(encoding="utf-8") == "run 2\n"
    assert not (tmp_path / "log.txt.prev3").exists(), "two generations is the setting"


def test_an_empty_or_missing_log_is_not_worth_a_generation(tmp_path):
    log = tmp_path / "log.txt"
    assert rotate(log, generations=2) == [], "a missing log moves nothing"
    _write(log, "")
    assert rotate(log, generations=2) == [], "a run that never wrote a line must not push the evidence out"
    assert not (tmp_path / "log.txt.prev").exists()
