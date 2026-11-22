"""What the Logs tab shows: the backend's output, minus the telemetry.

The backend prints three kinds of lines, and only one of them is a log:

* **events** - a phase started, a model loaded, a render finished, plus warnings and errors. This is what a person
  reads, and it is what this filter keeps.
* **telemetry** - ``host timeline: upload 0.04 ... model-call 0.21 ...`` once per frame, ``fp16 engine GPU: 0.68
  ms/frame (last 8)`` every eight, and ``BACKEND: reuse: 0.0% of 250 frames`` every 250. Measured on a 6 s clip: 106
  of the 106 lines the old pane showed were 40 events and 66 telemetry rows (owner, 2026-11-22: "каждый кадр
  рапортует об отрисовке кадра"). The numbers are still recorded in the run report and in `log.txt`.
* **the live progress line** - ``FPS: 29 Current Frame: 120 ETA: 0:03:19``, reprinted twice a second. The process page
  already shows FPS, ETA and the progress bar from the same line, so the log copy is pure duplication.

The filter is a pure function of the line stream (no Qt, no state outside the instance), so the rules are asserted
directly in `harness/tests/test_render_log.py` against lines captured from a real backend run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Levels a line can carry; the host maps them to theme colours.
LEVEL_INFO = "info"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"
LEVEL_PHASE = "phase"


@dataclass(frozen=True)
class LogLine:
    """One line the Logs tab will show, with the level the host colours it by."""

    text: str
    level: str = LEVEL_INFO


#: The live progress line, reprinted twice a second while a render runs.
_PROGRESS = re.compile(r"FPS:\s*[\d.]+.*Current Frame:\s*\d+.*ETA:")
#: Per-frame / per-chunk instrumentation the backend prints from the render loop.
_PER_FRAME = (
    re.compile(r"^host timeline:"),
    re.compile(r"^(fp16|int8|int8 body) engine GPU:"),
)
#: Periodic telemetry: a reuse counter printed every 250 frames.
_PERIODIC = (re.compile(r"^BACKEND: reuse: .* of \d+ frames"),)
#: Multi-line dumps that are one long command each.
_DUMPS = (
    re.compile(r"^BACKEND: FFMPEG (READ|WRITE) COMMAND:"),
    re.compile(r"^BACKEND: CLI Arguments:"),
)
#: Setup chatter an operator cannot act on, printed once per device/stream (five times per render in the measured
#: sample). Kept the first time it appears, because "which device" is worth one line.
_CHATTER = re.compile(
    r"^(BACKEND: )?(Handling device:|Handling precision:|Using Device:|Initializing stream for device"
    r"|Getting Input Video Properties)"
)
#: Debug leftovers that shipped.
_DEBUG = re.compile(r"^BACKEND: DEBUG")

_ERROR = re.compile(r"(Traceback|Exception|\bERROR\b|\bCRITICAL\b|\bfailed\b|\bFailed\b)")
_WARN = re.compile(r"(\bWARN(ING)?\b|\bunavailable\b|\bskipped\b|\bforced\b|\bfalling back\b)")
_PHASE = re.compile(
    r"(^BACKEND: Rendering$|^BACKEND: startup: first frame|^BACKEND: render loop:|^BACKEND: profile over"
    r"|^BACKEND: Building|^BACKEND: Loading|engine built in)"
)


@dataclass
class RenderLogFilter:
    """Accumulate the backend's stdout into the user-facing log."""

    #: Set while a multi-line dump's continuation lines must be swallowed (the CLI argument list).
    _skip_next: bool = False
    #: Chatter already shown once, so the repeats are dropped.
    _seen_chatter: set[str] = field(default_factory=set)
    #: How many lines each rule removed - the number the harness and the close-out report quote.
    dropped: dict[str, int] = field(default_factory=dict)

    def _drop(self, reason: str) -> list[LogLine]:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        return []

    def feed(self, raw: str) -> list[LogLine]:
        """One backend line in, the lines the Logs tab keeps out (usually none or one)."""
        text = str(raw).strip()
        if self._skip_next:
            self._skip_next = False
            if text:
                return self._drop("dump")
        if not text or text.rstrip(":") in ("", "BACKEND"):
            # The backend prints a bare `BACKEND:` separator before some blocks; on its own it is an empty row.
            return self._drop("blank")
        if _PROGRESS.search(text):
            return self._drop("progress")
        if any(pattern.search(text) for pattern in _PER_FRAME):
            return self._drop("per-frame")
        if any(pattern.search(text) for pattern in _PERIODIC):
            return self._drop("periodic")
        if _DEBUG.search(text):
            return self._drop("debug")
        for pattern in _DUMPS:
            if pattern.search(text):
                if pattern is _DUMPS[1]:
                    self._skip_next = True
                return self._drop("dump")
        if _CHATTER.search(text):
            if text in self._seen_chatter:
                return self._drop("repeat")
            self._seen_chatter.add(text)
        return [LogLine(text, self._level(text))]

    def feed_all(self, lines) -> list[LogLine]:
        kept: list[LogLine] = []
        for line in lines:
            kept.extend(self.feed(line))
        return kept

    @staticmethod
    def _level(text: str) -> str:
        if _ERROR.search(text):
            return LEVEL_ERROR
        if _WARN.search(text):
            return LEVEL_WARN
        if _PHASE.search(text):
            return LEVEL_PHASE
        return LEVEL_INFO

    def hidden_total(self) -> int:
        return sum(self.dropped.values())

    def hidden_summary(self) -> LogLine | None:
        """One closing line naming what was hidden - the log never quietly loses output.

        "" (nothing hidden) returns ``None``. The counts are per class, so a reader can tell whether the backend was
        merely chatty or whether a whole instrument was switched on.
        """
        if not self.dropped:
            return None
        parts = ", ".join(f"{count} {reason}" for reason, count in sorted(self.dropped.items()))
        return LogLine(f"— {self.hidden_total()} telemetry lines hidden ({parts})", LEVEL_INFO)
