"""The backend's own terminal stream, kept in a file next to the app's log.

Why this exists (owner, 2026-11-25: "логично тогда иметь условно файл logs-terminal.txt"): the backend's records are
deliberately NOT copied into `log.txt` - they go to the Logs tab through `ProcessTab`'s readout loop, and only a
`render failed` summary is re-logged. That was a reasonable split (the pane is a consumer of the stream, not of the
app's file), but it left a hole the other way: once the window closed, the backend's `seek:`, `display:` and engine
lines were gone, and several diagnoses in this session had to be reconstructed from the pane's telemetry alone.

This file mirrors the stream verbatim. It is raw on purpose: the tab's filter decides what a PERSON reads, while a
forensic reader wants the line the backend actually printed. Rotation keeps two runs, the same as `log.txt`, for the
same reason: the run that misbehaved is usually the previous one.
"""

from __future__ import annotations

import datetime
import os
import pathlib

from ..log_rotation import rotate

#: Beside the app's `log.txt` (the process CWD is the app root for every entry point this tree has).
FILE_NAME = "logs-terminal.txt"

#: Paths already rotated by THIS process. A `TerminalLog` is constructed once per render job, and rotating on every
#: construction moved the previous job's stream into `.prev2` and then out of the two generations entirely - four jobs
#: in one run destroyed the first job's evidence (telemetry audit 2026-11-27, 5.9). Rotation is about RUNS, which is
#: what this module's docstring promises ("keeps two runs, the same as log.txt"); jobs append to the run's file.
_ROTATED: set[str] = set()

#: Env override, like the pane's telemetry file.
ENV_NAME = "RVE_TERMINAL_LOG_FILE"


def log_path() -> pathlib.Path:
    explicit = str(os.environ.get(ENV_NAME) or "").strip()
    return pathlib.Path(explicit) if explicit else pathlib.Path(os.getcwd()) / FILE_NAME


class TerminalLog:
    """Append-only mirror of one render's backend stream; rotates the previous runs out of the way on open."""

    def __init__(self, path: str | pathlib.Path | None = None, *, generations: int = 2) -> None:
        self.path = pathlib.Path(path) if path is not None else log_path()
        self._handle = None
        self._wrote_header = False
        key = str(self.path.resolve())
        if key not in _ROTATED:
            try:
                rotate(self.path, generations=generations)
            except Exception:
                pass                                # a read-only tree still gets the stream on stderr
            _ROTATED.add(key)
        try:
            self._handle = open(self.path, "a", encoding="utf-8")
        except OSError as exc:
            print(f"terminal log unavailable ({exc}); the backend stream stays on the tab only", flush=True)
            self._handle = None

    def _header(self) -> None:
        if self._wrote_header or self._handle is None:
            return
        self._wrote_header = True
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        self._handle.write(f"# backend stream {stamp} app_pid={os.getpid()}\n")
        self._handle.flush()

    def write(self, line: str) -> None:
        """One line as the backend printed it (trailing whitespace trimmed, no other editing)."""
        if self._handle is None:
            return
        self._header()
        try:
            self._handle.write(str(line).rstrip() + "\n")
            self._handle.flush()
        except OSError:
            self._handle = None

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
