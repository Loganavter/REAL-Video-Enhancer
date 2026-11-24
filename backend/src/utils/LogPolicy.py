"""The Logs block's policy: two levels, a bounded progress cadence, and once-per-job phases.

Pure Python (stdlib only, no Qt, no torch), imported by both processes:

* the app - ``src/Util.py`` loads it next to the backend it launches;
* the backend - ``backend/src/utils/Util.py``, the render loop, the int8 body and the
  TensorRT handler.

Levels
------
``info`` (default)
    Lifecycle, phases, per-job summaries. What a person reads: process start, device/model
    init, engine build/load, reader/writer init, first frame, total and ms/frame.
``debug``
    Per-frame / per-call detail: the int8 tail and phase breakdowns, the per-frame profile
    parts, the host-timeline style rows. Off unless ``RVE_LOG_DEBUG=1``
    (``true``/``yes``/``on`` also count). Nothing is deleted - the same call prints again
    with the switch on.

The switch is read on every call rather than cached at import: the backend child inherits
the parent's environment, and a test can flip it without reloading the module.

Progress cadence
----------------
One line per **second of wall clock** (``DEFAULT_PROGRESS_INTERVAL_S``), not one per
percent. Frame rate in this app spans ~3 fps (a 4x model on a long clip) to ~1000 fps (a
short clip on a fast card), so a percent cadence is 300 lines/s at one end and looks
stalled at the other; seconds bound the block at ~1 line/s whatever the render is doing.
"""

from __future__ import annotations

import os
import time

LEVEL_DEBUG = "debug"
LEVEL_INFO = "info"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"

#: The one switch that turns per-frame detail back on. Documented in the module docstring.
DEBUG_ENV = "RVE_LOG_DEBUG"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Progress cadence, in seconds. See the module docstring for why seconds and not percent.
DEFAULT_PROGRESS_INTERVAL_S = 1.0


def debug_enabled() -> bool:
    """Is per-frame / per-call detail wanted in this process?"""
    return os.environ.get(DEBUG_ENV, "").strip().lower() in _TRUTHY


def level_enabled(level: str) -> bool:
    """Would a line at ``level`` be emitted? ``info`` and above always are."""
    return str(level or LEVEL_INFO).strip().lower() != LEVEL_DEBUG or debug_enabled()


class ProgressLimiter:
    """At most one progress line per ``interval_s`` seconds of wall clock.

    State is per instance, so one instance per job gives a fresh window: a new render's
    first progress line is never swallowed because the previous job printed recently.
    """

    def __init__(self, interval_s: float = DEFAULT_PROGRESS_INTERVAL_S, clock=time.monotonic):
        self.interval_s = float(interval_s)
        self._clock = clock
        self._last: float | None = None

    def reset(self) -> None:
        """Start a new window now (the next ``allow()`` returns True)."""
        self._last = None

    def allow(self, now: float | None = None, force: bool = False) -> bool:
        """May a progress line be emitted? ``force`` emits and restarts the window."""
        moment = self._clock() if now is None else float(now)
        if force or self._last is None or (moment - self._last) >= self.interval_s:
            self._last = moment
            return True
        return False

    def maybe(self, emit, now: float | None = None) -> bool:
        """Call ``emit()`` only if the window allows it; return whether it ran."""
        if not self.allow(now):
            return False
        emit()
        return True


class JobPhases:
    """Once-per-job phase lines: the first call per key wins.

    One instance per job (``RenderVideo`` owns one). ``first()`` returns True the first
    time a key is asked for and False afterwards, so a phase line cannot be repeated by a
    loop that happens to pass the mark twice.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def first(self, key: str) -> bool:
        key = str(key)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def reset(self) -> None:
        self._seen.clear()

    def seen(self) -> tuple[str, ...]:
        return tuple(sorted(self._seen))
