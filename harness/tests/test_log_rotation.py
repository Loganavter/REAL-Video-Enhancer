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


def test_the_terminal_log_rotates_once_per_run_not_once_per_job(tmp_path, monkeypatch):
    """A render job constructs its own `TerminalLog`; four jobs used to rotate the first job's stream out of existence.

    The module's own docstring promises two RUNS, the same as `log.txt`. Rotation is per process, so a job appends to
    the run's file and the previous run survives as `.prev` (telemetry audit 2026-11-27, 5.9).
    """
    from src.ui import terminal_log as module

    monkeypatch.setattr(module, "_ROTATED", set())
    path = tmp_path / "logs-terminal.txt"
    _write(path, "job 1\n")

    first = module.TerminalLog(path, generations=2)
    first.write("job 1 more\n")
    first.close()

    # That first open is the RUN boundary: whatever was in the file is the previous run and goes to `.prev`.
    assert (tmp_path / "logs-terminal.txt.prev").read_text(encoding="utf-8") == "job 1\n"

    second = module.TerminalLog(path, generations=2)
    second.write("job 2\n")
    second.close()

    text = path.read_text(encoding="utf-8")
    assert "job 1 more" in text, "the first job's stream must survive the second job's construction"
    assert "job 2" in text
    assert (tmp_path / "logs-terminal.txt.prev").read_text(encoding="utf-8") == "job 1\n", \
        "a second job in the same run must not rotate again"
