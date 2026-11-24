"""The backend's terminal stream is kept in its own file, verbatim, with the previous runs rotated.

Why (owner, 2026-11-25: "логично тогда иметь условно файл logs-terminal.txt"): the backend's records go to the Logs
tab, not to `log.txt` - a deliberate split, but it left no file to read after the window closed, and several
diagnoses in this session had to be reconstructed from telemetry alone.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.terminal_log import TerminalLog  # noqa: E402


def test_the_stream_is_mirrored_verbatim_with_one_header(tmp_path):
    path = tmp_path / "logs-terminal.txt"
    writer = TerminalLog(path, generations=2)
    writer.write("BACKEND: seek: displayed the frame at 8.105s (frame pts 8.136s)\n")
    writer.write("   spaces kept inside\n")
    writer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# backend stream "), "one header per run, so two streams are never mistaken"
    assert lines[1] == "BACKEND: seek: displayed the frame at 8.105s (frame pts 8.136s)"
    assert lines[2] == "   spaces kept inside", "only the trailing newline is trimmed - the reader wants what was printed"
    assert sum(1 for line in lines if line.startswith("# backend stream ")) == 1


def test_an_empty_run_does_not_push_the_previous_stream_out(tmp_path):
    path = tmp_path / "logs-terminal.txt"
    first = TerminalLog(path, generations=2)
    first.write("run 1 line")
    first.close()

    unused = TerminalLog(path, generations=2)      # opened, nothing written (a render that never started)
    unused.close()

    assert "run 1 line" in (tmp_path / "logs-terminal.txt.prev").read_text(encoding="utf-8") or \
        "run 1 line" in path.read_text(encoding="utf-8"), "the evidence is still on disk after an empty open"


def test_two_runs_are_kept(tmp_path):
    path = tmp_path / "logs-terminal.txt"
    for index in (1, 2):
        writer = TerminalLog(path, generations=2)
        writer.write(f"run {index} line")
        writer.close()

    assert "run 2 line" in path.read_text(encoding="utf-8")
    assert "run 1 line" in (tmp_path / "logs-terminal.txt.prev").read_text(encoding="utf-8")
