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

Three more rules make the pane read like a status report rather than a profiler dump (owner, 2026-11-28: "логи вообще
не для человека пишутся ... лучше бы там условно раз в секунду писали сколько к/с обработано за это время и ETA
оставшееся ... нету выделения действительно важных этапов"):

* :func:`phase_header` turns a phase line into a rule with the phase in it - one bold accent row a reader's eye lands
  on without a second line of whitespace;
* :class:`LiveStatus` is the curated once-a-second line the page's tick appends (``1.9 s/frame · 0.53 fps · ETA
  0:12:10 · frame 120/493``), built from the numbers the tick already carries. The backend's own ``FPS: ... ETA:``
  line stays filtered, so the log never states the same numbers twice;
* :class:`BuildHistory` and :func:`annotate_build_line` answer "how long until the engine is built" honestly: the
  duration is unknown before it happens, so the start line names the median of the builds this machine actually
  timed in that GPU/precision/model slot, and says so plainly when there is no history.

The filter is a pure function of the line stream (no Qt, no state outside the instance), so the rules are asserted
directly in `harness/tests/test_render_log.py` and `harness/tests/test_log_live_status.py` against lines captured from
a real backend run.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
from dataclasses import dataclass, field
from typing import Sequence

#: Levels a line can carry; the host maps them to theme colours.
LEVEL_INFO = "info"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"
LEVEL_PHASE = "phase"
#: The curated once-a-second progress line (:class:`LiveStatus`); painted quiet - it is ambient, not an event.
LEVEL_STATUS = "status"


@dataclass(frozen=True)
class LogLine:
    """One line the Logs tab will show, with the level the host colours it by."""

    text: str
    level: str = LEVEL_INFO


#: The live progress line, reprinted twice a second while a render runs.
_PROGRESS = re.compile(r"FPS:\s*[\d.]+.*Current Frame:\s*\d+.*ETA:")
#: Per-frame / per-chunk instrumentation the backend prints from the render loop.
#:
#: `int8 host: 16 launches cost 0.73 ms of HOST ...; whole hop loop 0.92 ms` repeats several times per frame on the
#: int8 path (the prefix widens with each body - `int8 body host:` today, another body tomorrow), and
#: `BACKEND: display: drew source frame 391 named 13.055s (pts 13.055s)` prints once per displayed frame. Both are
#: measurements, not events: the run report and `logs-terminal.txt` keep them.
_PER_FRAME = (
    re.compile(r"^host timeline:"),
    re.compile(r"^(fp16|int8|int8 body) engine GPU:"),
    re.compile(r"^(?:[A-Za-z0-9_.]+\s+)*host:\s+\d+ launches cost\s+[\d.]+\s+ms of HOST"),
    re.compile(r"^(?:BACKEND:\s+)?display:\s+drew source frame\s+\d+\s+named\s+[\d.]+s"),
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
    #: How many lines each rule removed - the number the harness and the close-out report quote, PER REASON: "66
    #: per-frame" and "1 periodic" are different findings about the backend, and one total would hide which.
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
        # A line that names a CAUSE is read even when it happens to carry a telemetry shape: a profiler line that
        # says `failed`/`WARN` is the one line a reader must not lose (owner's rule: the filter hides numbers, never
        # reasons). Only the telemetry rules below are bypassed; the dumps and DEBUG rules keep their own lines.
        if not (_ERROR.search(text) or _WARN.search(text)):
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
        level = self._level(text)
        if level == LEVEL_PHASE:
            return [LogLine(phase_header(text), LEVEL_PHASE)]
        return [LogLine(text, level)]

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


# -- milestones: a phase line becomes a rule the eye can land on ----------------------------------------------------


#: Width of a phase header row, in characters. Long enough to read as a divider, short enough to stay one row at the
#: pane's usual width; a narrower pane word-wraps it like any other log row, so the width is a preference, not a rule.
PHASE_HEADER_WIDTH = 64


def phase_header(text: str, *, width: int = PHASE_HEADER_WIDTH) -> str:
    """A phase as one rule row: ``── BACKEND: Rendering ──────────────────────────────────────────``.

    The owner's first sketch was a bare ``======`` line above the phase, then "хотя наверное лучше было бы по-другому
    придумать". A separate rule row costs a line that says nothing (and a phase cluster - Building, engine built,
    Building - turns into six rows), so the rule carries the phase itself: the same single ConsoleLine the filter
    would have emitted, painted bold/accent by its LEVEL_PHASE, with the horizontal rule extending past it. A phase
    longer than the width is not truncated - the row just ends with the closing rule.
    """
    text = str(text).strip()
    body = f"── {text} "
    if len(body) + 2 <= width:
        body += "─" * (width - len(body) - 2)
    return body + "──"


# -- the curated once-a-second line -------------------------------------------------------------------------------


def _rate_text(fps: float) -> list[str]:
    """The speed half of the status line: fps, and seconds per frame only where the inverse is readable.

    29 fps is `29.3 fps`; `0.03 s/frame` beside it is noise. Below 10 fps the inverse is the number a person wants
    (a 1.9 s/frame engine is why the render takes hours), so it leads.
    """
    if fps <= 0:
        return []
    seconds = 1.0 / fps
    if fps < 10:
        return [f"{seconds:.1f} s/frame", f"{fps:.2f} fps"]
    return [f"{fps:.1f} fps"]


def status_text(*, fps: float = 0.0, eta: str = "", frame: int = 0, total: int = 0) -> str:
    """The one line a person needs while a render runs, or "" when the tick has nothing to say yet.

    Shape: ``1.9 s/frame · 0.53 fps · ETA 0:12:10 · frame 120/493``. Every number comes from the tick's own state
    (`fps`, `eta`, `currentFrame`/`max_value`); nothing here re-reads the backend or parses its prose. The ``·``
    separator keeps the four fields scannable without the punctuation noise of ``|`` or brackets.
    """
    fps = float(fps or 0.0)
    frame = max(0, int(frame or 0))
    total = max(0, int(total or 0))
    if frame <= 0 and fps <= 0:
        return ""                                   # nothing has rendered yet: an empty status is noise
    parts = _rate_text(fps)
    eta = str(eta or "").strip()
    if eta:
        parts.append(f"ETA {eta}")
    if total:
        parts.append(f"frame {min(frame, total)}/{total}")
    elif frame:
        parts.append(f"frame {frame}")
    return " · ".join(parts)


@dataclass
class LiveStatus:
    """At most one status line per interval - the cadence, with the clock passed in so a test never sleeps.

    `maybe_line` is the whole contract: it answers ``None`` while the interval has not elapsed, and a
    :class:`LogLine` (LEVEL_STATUS) carrying the tick's numbers once it has. The page replaces the previous status
    line instead of appending a new one, so a 30-minute render leaves ONE status row in the pane, not 1800.
    """

    interval: float = 1.0
    _last: float | None = None

    def reset(self) -> None:
        """A new job: its first tick may write at once (the old job's second is not this job's second)."""
        self._last = None

    def maybe_line(self, now: float, *, fps: float = 0.0, eta: str = "", frame: int = 0,
                   total: int = 0) -> LogLine | None:
        if self._last is not None and (now - self._last) < self.interval:
            return None
        text = status_text(fps=fps, eta=eta, frame=frame, total=total)
        if not text:
            return None
        self._last = float(now)
        return LogLine(text, LEVEL_STATUS)


# -- engine builds: the estimate is a recorded median, never a guess ---------------------------------------------

#: Beside `settings.txt` (the app's settings owner names the file's directory; machine timings do not belong in the
#: key,value settings file itself - they are per GPU and would churn every write).
BUILD_HISTORY_FILE = "engine-build-times.json"
#: Durations kept per GPU/precision/model slot, and slots kept overall. Five is enough for a median that ignores one
#: bad build (a cold driver, a contended GPU) and small enough that the file stays a few hundred bytes.
BUILD_HISTORY_CAP = 5
BUILD_HISTORY_SLOTS = 40

_BUILD_START = re.compile(
    r"^(?:BACKEND:\s+)?Building TensorRT engine (?P<engine>.+?)\.(?:\s+This may take a while)?\.*$")
_BUILD_DONE = re.compile(r"^(?:BACKEND:\s+)?TensorRT engine built in (?P<seconds>[\d.]+) seconds")
#: The cache HIT: `Loading TensorRT engine from <engine>.engine.` / `Loading raw TensorRT plan from X.plan.`
_BUILD_CACHED = re.compile(r"^(?:BACKEND:\s+)?Loading (?:raw )?TensorRT (?:engine|plan) ")
_MODEL_IN_ENGINE = re.compile(r"^(?P<model>.+?\.(?:pth|pt|onnx|safetensors))(?:_|$)")
_PRECISION_IN_ENGINE = re.compile(r"_(?P<precision>fp16|fp32|bf16|fp8)(?:_|$)")
_GPU_IN_ENGINE = re.compile(r"_(?P<gpu>NVIDIA[^_]+?)_trt-")


def engine_build_history_path(settings_file: str | os.PathLike | None) -> pathlib.Path:
    """Where the build times live: next to `settings.txt`, or the working directory when no settings path is known."""
    if settings_file:
        return pathlib.Path(settings_file).with_name(BUILD_HISTORY_FILE)
    return pathlib.Path(os.getcwd()) / BUILD_HISTORY_FILE


def build_slot(engine_name: str, *, gpu: str = "") -> "BuildSlot":
    """The slot a build belongs to: GPU + precision + model, and the short label a person reads.

    The engine file name already carries all three (`dncnn_color_blind.pth_fp16_NVIDIA GeForce RTX 4060 Ti_trt-...`),
    which is why the build can be timed at all: the start line prints that name and the finished line does not print
    it again. The device line's name is preferred when the page has it - the engine name is a file name, and a
    device name a file system rewrote is not the GPU the estimate belongs to.
    """
    name = str(engine_name or "").strip()
    model_match = _MODEL_IN_ENGINE.match(name)
    precision_match = _PRECISION_IN_ENGINE.search(name)
    model = model_match.group("model") if model_match else ""
    precision = precision_match.group("precision") if precision_match else ""
    device = str(gpu or "").strip()
    if not device:
        gpu_match = _GPU_IN_ENGINE.search(name)
        device = gpu_match.group("gpu").strip() if gpu_match else "unknown GPU"
    return BuildSlot(gpu=device, precision=precision, model=model or name)


@dataclass(frozen=True)
class BuildSlot:
    """One GPU/precision/model slot: its history key and the words a build line shows for it."""

    gpu: str
    precision: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.gpu or 'unknown GPU'}|{self.precision or 'unknown precision'}|{self.model or 'unknown model'}"

    @property
    def label(self) -> str:
        return " ".join(part for part in (self.model, self.precision) if part) or "engine"


def build_key(engine_name: str, *, gpu: str = "") -> str:
    """:meth:`BuildSlot.key` for a bare engine name (the history's dictionary key)."""
    return build_slot(engine_name, gpu=gpu).key


def format_duration(seconds: float) -> str:
    """``48 s`` / ``4 min 10 s`` / ``1 h 03 min`` - the shape the build hint and its estimate both read."""
    total = int(round(max(0.0, float(seconds))))
    if total < 60:
        return f"{total} s"
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {rest:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def estimate_suffix(durations: Sequence[float]) -> str:
    """What to append to ``Building TensorRT engine``: the median of the recorded runs, or the honest first-build."""
    runs = [float(value) for value in durations if value is not None]
    if not runs:
        return " (first build on this GPU - no estimate yet)"
    plural = "build" if len(runs) == 1 else "builds"
    return f" (last {len(runs)} {plural}: ~{format_duration(statistics.median(runs))})"


#: TensorRT's builder progress monitor (`IProgressMonitor` + `ProgressMonitorPhase`, 8.5+). The builder runs in the
#: BACKEND process, so this app process is not where the fraction lives - the probe answers "can a fraction be read
#: HERE", and a producer on the other side of the pipe (a printed fraction) is what would make tier 2 live.
TRT_MODULE = "tensorrt"


def monitor_available(module=None) -> bool:
    """Does this process expose TensorRT's build progress monitor? A feature detection, never a requirement.

    Here it answers False: the GUI imports no TensorRT (`src/` has no `import tensorrt`) and the rve backend builds the
    engine in its own process. That is exactly why the estimate is a recorded median instead of a promise - the
    fraction exists only where the builder runs. Anything that DOES hold the module (a test double, a future
    in-process build) passes it in and gets the truth about that module.
    """
    if module is None:
        import importlib.util

        try:
            if importlib.util.find_spec(TRT_MODULE) is None:
                return False
            module = importlib.import_module(TRT_MODULE)
        except Exception:
            return False
    return hasattr(module, "IProgressMonitor") and hasattr(module, "ProgressMonitorPhase")


def progress_eta(elapsed: float, fraction: float | None) -> float | None:
    """Seconds left of a build reported ``fraction`` complete: ``elapsed / p * (1 - p)``.

    ``None`` when the fraction cannot answer (absent, unreadable, or not started): a 0 % build has no rate yet, and
    dividing by zero is how an estimate becomes a lie.
    """
    if fraction is None:
        return None
    try:
        done = float(fraction)
        spent = float(elapsed or 0.0)
    except (TypeError, ValueError):
        return None
    if done <= 0.0:
        return None
    return max(0.0, spent / done * (1.0 - done))


def build_hint(*, cached: bool = False, fraction: float | None = None, elapsed: float | None = None,
               durations: Sequence[float] = ()) -> str:
    """The suffix a build line carries, in the order that can answer best first.

    1. **cache hit** - the engine file exists, the run loads it: there is no build to wait for;
    2. **builder progress** - TensorRT reported ``fraction``: percent built and the remaining time from the elapsed
       clock (only reachable by a caller that can read the monitor, see :func:`monitor_available`);
    3. **recorded history** - the median of this machine's earlier builds in that slot, or the honest first-build.
    """
    if cached:
        return " (loading cached engine - no build to wait for)"
    remaining = progress_eta(elapsed or 0.0, fraction)
    if remaining is not None:
        return f" ({int(round(100.0 * float(fraction)))}% built, ~{format_duration(remaining)} left)"
    return estimate_suffix(durations)


@dataclass
class BuildHistory:
    """The build durations this machine actually measured, per GPU/precision/model slot.

    Persisted as a small json beside `settings.txt` (see :func:`engine_build_history_path`) because a build is the one
    part of a render whose duration is unknown before it happens and stable after it: the second build of the same
    model on the same GPU is the first one's duration. An unreadable or absent file is an EMPTY history, never an
    error - an estimate is allowed to be missing, a render is not allowed to fail over one.
    """

    path: pathlib.Path | None = None
    cap: int = BUILD_HISTORY_CAP
    slots: int = BUILD_HISTORY_SLOTS
    history: dict[str, list[float]] = field(default_factory=dict)
    #: Slots whose start line was seen and whose finished line has not arrived yet. Sequential by construction
    #: (one build at a time), a list so a nested build cannot lose its parent.
    _pending: list[tuple[str, float]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | os.PathLike | None, *, cap: int = BUILD_HISTORY_CAP,
             slots: int = BUILD_HISTORY_SLOTS) -> "BuildHistory":
        store = cls(path=pathlib.Path(path) if path else None, cap=cap, slots=slots)
        try:
            with open(store.path, encoding="utf-8") as handle:
                document = json.load(handle)
            raw = document.get("builds") if isinstance(document, dict) else None
            if isinstance(raw, dict):
                for key, values in raw.items():
                    if isinstance(values, list):
                        clean = [float(value) for value in values if isinstance(value, (int, float))]
                        if clean:
                            store.history[str(key)] = clean[-cap:]
        except (OSError, ValueError, TypeError):
            pass                                    # no file yet / a hand-edited one: the estimate starts empty
        return store

    def durations(self, key: str) -> list[float]:
        return list(self.history.get(str(key), ()))

    def median(self, key: str) -> float | None:
        runs = self.durations(key)
        return statistics.median(runs) if runs else None

    def note_start(self, key: str, now: float) -> None:
        """A build just started - remembered so the finished line (which names no engine) closes the right slot."""
        self._pending.append((str(key), float(now)))

    def note_finished(self, now: float, seconds: float | None = None) -> float | None:
        """Record the duration of the build that started last; the backend's own number wins over the clock.

        The backend prints ``TensorRT engine built in 12.34 seconds`` - measured by the builder itself, so it does not
        include the GUI's read latency. The injected clock is the fallback for a line without a number, and the reason
        a test can drive this without sleeping.
        """
        if not self._pending:
            return None
        key, started = self._pending.pop()
        duration = float(seconds) if seconds is not None else max(0.0, float(now) - started)
        runs = self.history.setdefault(key, [])
        runs.append(round(duration, 2))
        del runs[:-self.cap]
        if len(self.history) > self.slots:
            for stale in list(self.history)[:-self.slots]:
                if stale != key:
                    self.history.pop(stale, None)
        self.save()
        return duration

    def save(self) -> None:
        """Write the history atomically; a read-only tree keeps it in memory for this run."""
        if self.path is None:
            return
        document = {"version": 1, "builds": self.history}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            temporary = f"{self.path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=1, sort_keys=True)
            os.replace(temporary, self.path)
        except OSError:
            pass                                    # an estimate is a convenience; a read-only tree still renders


def annotate_build_line(text: str, history: "BuildHistory | None", *, now: float, gpu: str = "",
                        fraction: float | None = None, elapsed: float | None = None) -> str:
    """The line to feed the filter: a build START gains the estimate, a build FINISH is recorded on the way through.

    The engine file name is a machine key (`..._trt-10.8.0.43_torch_tensorrt-2.9.0_opt-5.engine`) and the tab is for a
    person, so the start line keeps the model and the precision and drops the rest - the full name is still in
    `logs-terminal.txt`. A `Loading TensorRT engine from ...` line is a CACHE HIT: the engine already exists, nothing
    is built, and the line says so instead of inventing an ETA. ``fraction`` is the builder's own progress when a
    caller can read it (see :func:`monitor_available`); without it the recorded median answers.

    Returns every other line unchanged. ``history`` may be None (a cell without a writable settings directory), in
    which case the line is still decorated with the honest "no estimate yet".
    """
    raw = str(text)
    prefix = "BACKEND: " if raw.startswith("BACKEND: ") else ""
    body = raw[len(prefix):]
    start = _BUILD_START.match(body)
    if start is not None:
        slot = build_slot(start.group("engine"), gpu=gpu)
        runs: list[float] = []
        if history is not None:
            history.note_start(slot.key, now)
            runs = history.durations(slot.key)
        hint = build_hint(fraction=fraction, elapsed=elapsed, durations=runs)
        return f"{prefix}Building TensorRT engine {slot.label}{hint}"
    if _BUILD_CACHED.match(body):
        return raw + build_hint(cached=True)
    done = _BUILD_DONE.match(body)
    if done is not None and history is not None:
        try:
            seconds = float(done.group("seconds"))
        except (TypeError, ValueError):
            seconds = None
        history.note_finished(now, seconds=seconds)
    return raw
