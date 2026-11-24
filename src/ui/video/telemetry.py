"""Numbers for the preview pane, so a report can carry a measurement instead of a guess.

Why this module exists: every question about this pane used to be answered by argument - "is the upload the cost",
"does the seek pay a decode", "is the picture stale or is the paint cadence off". Each has a number, and the pane
never wrote one down. This is the one place that counts them and writes them as greppable lines.

Levels (`RVE_PREVIEW_LOG`):

* unset / `0` - **events only**: one line per action (a press, a seek, a mode change, an RHI re-init). Cheap enough
  to leave on; `log(..., level="info")` keeps them in `log.txt` beside the rest of the app's record.
* `1` - events **+ one aggregate line per second**: frame/paint/upload counts, per-stage milliseconds, the interval
  between paints, drops, and the seek turnaround.
* `2` - everything above **+ one line per frame**. Use it for a short session only; at 24-60 fps it is a lot of text.

The lines go to the pane's OWN file - `preview-log.txt` next to the app's `log.txt` (or `RVE_PREVIEW_LOG_FILE`) -
one line per record, flushed as it is written, with a header per process so two sessions cannot be mistaken for one.
That file is the thing to paste into a report; the app's `log.txt` stays clean.

The lines are `key=value` so they can be pasted or parsed with `awk`:

    preview: stats frames=24 paints=24 up=24 up_ms=0.01/1.34 pass_ms=0.40/0.98 gui_ms=0.62/0.98 int_ms=41.2/44.0/90.1
    preview: stats … drop=0 seek_ms=38.4/… window_s=1.0

`up_ms` is the enqueue call only; `pass_ms` is the pass (resource update + draws), i.e. where the copy actually happens.

(`a/b` is median/max, `a/b/c` is median/p95/max.)

Nothing here imports Qt or the app: a test drives it with a fake writer and a fake clock, and the file writer is a
plain append. The module never touches the app log, so it cannot be silenced by the app's log policy and cannot
flood it either.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

#: `RVE_PREVIEW_LOG`: 0 events only (default), 1 + a per-second aggregate, 2 + a line per frame.
ENV_NAME = "RVE_PREVIEW_LOG"

#: Where the lines go. Default: `preview-log.txt` beside the app's own `log.txt` (the process CWD, which is the app
#: root for every entry point this tree has).
FILE_ENV_NAME = "RVE_PREVIEW_LOG_FILE"
FILE_NAME = "preview-log.txt"

#: How often the aggregate line may be written, seconds.
STATS_PERIOD_S = 1.0

_writer: Callable[[str, str], None] | None = None
_clock: Callable[[], float] = time.monotonic
_open_file = None
_opened_path: str | None = None


def level() -> int:
    """The current verbosity (see the module docstring); anything unparseable means events only."""
    import os

    raw = str(os.environ.get(ENV_NAME) or "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, min(2, value))


def set_writer(writer: Callable[[str, str], None] | None) -> None:
    """Route lines somewhere else (tests). ``None`` restores the file."""
    global _writer
    _writer = writer


def set_clock(clock: Callable[[], float] | None) -> None:
    """Replace the monotonic clock (tests). ``None`` restores ``time.monotonic``."""
    global _clock
    _clock = clock or time.monotonic


def log_path() -> str:
    """The file the lines are appended to (``RVE_PREVIEW_LOG_FILE`` wins; default beside ``log.txt``)."""
    import os

    explicit = str(os.environ.get(FILE_ENV_NAME) or "").strip()
    return explicit or os.path.join(os.getcwd(), FILE_NAME)


def _handle():
    """The append handle, opened on first use; a directory that cannot be opened falls back to the console."""
    global _open_file, _opened_path
    path = log_path()
    if _open_file is not None and _opened_path == path:
        return _open_file
    try:
        _open_file = open(path, "a", encoding="utf-8")
        _opened_path = path
        import datetime

        _open_file.write(
            f"# preview telemetry session {datetime.datetime.now().isoformat(timespec='seconds')} "
            f"level={level()} pid={__import__('os').getpid()}\n")
        _open_file.flush()
    except OSError as exc:  # a read-only tree still gets the numbers, on the console
        print(f"preview: log file unavailable ({exc}); writing to stdout", flush=True)
        _open_file = None
    return _open_file


def _write(line: str, which: str) -> None:
    """One record: to the injected writer (tests), else the pane's file, else the console. Never the app log."""
    if _writer is not None:
        _writer(line, which)
        return
    handle = _handle()
    if handle is None:
        print(line, flush=True)
        return
    handle.write(line + "\n")
    handle.flush()


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def event(kind: str, **fields) -> None:
    """One line per action: always written (``level="info"``), so a session leaves a trace of what was pressed."""
    parts = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    _write(f"preview: {kind} {parts}".rstrip(), "info")


# --- the aggregate ---------------------------------------------------------------------------------
_frames = 0
_paints = 0
_uploads = 0
_bytes = 0
_gui: list[float] = []
_upload: list[float] = []
_intervals: list[float] = []
_pass: list[float] = []
_drops = 0
_last_paint: float | None = None
_last_stats = 0.0
_started = False
#: What the pane is painting: `("frame", n)` from a decoder's own number, `("seq", n)` from the display channel's
#: sequence. A pair on purpose - the two numbers mean different things and must not be printed as if they were one.
_shown: tuple[str, int] | None = None
_seek_ms: list[float] = []
#: the last written snapshot, for a control verb or a test to read without waiting for the next second
_last_snapshot: dict[str, object] = {}


def note_frame(*, gui_ms: float | None = None, pts: float | None = None, index: int | None = None,
               seq: int | None = None) -> None:
    """A picture was handed to the pane: `index` is a decoder's frame number, `seq` the display channel's sequence."""
    global _frames, _shown
    _frames += 1
    if index is not None:
        _shown = ("frame", int(index))
    elif seq is not None:
        _shown = ("seq", int(seq))
    if gui_ms is not None:
        _gui.append(float(gui_ms))
    if level() >= 2:
        event("frame", index=index, seq=seq, pts=None if pts is None else f"{pts:.3f}", gui_ms=_fmt(gui_ms))


def note_upload(*, upload_ms: float | None = None, size_bytes: int | None = None) -> None:
    """The GPU got a new picture (one upload per new frame, never per paint)."""
    global _uploads, _bytes
    _uploads += 1
    if upload_ms is not None:
        _upload.append(float(upload_ms))
    if size_bytes:
        _bytes += int(size_bytes)


def note_paint(*, interval_ms: float | None = None, pass_ms: float | None = None) -> None:
    """One completed render pass, optionally with its own cost (resource update + draw + endPass).

    `pass_ms` is the number that answers "does the GPU hand-off cost anything": `note_upload` can only time the
    enqueue call (`QRhiResourceUpdateBatch.uploadTexture`, microseconds), while the copy itself happens inside the
    pass - an owner's log showed `up_ms=0.01` for a 0.5 MB frame, i.e. a cost that had simply moved.
    """
    global _paints, _last_paint
    _paints += 1
    if pass_ms is not None:
        _pass.append(float(pass_ms))
    now = _clock()
    if _last_paint is not None:
        delta = (now - _last_paint) * 1000.0
        # The interval a reader cares about is the pane's own cadence, not the gap after an idle hour.
        if 0.0 < delta < 1000.0:
            _intervals.append(interval_ms if interval_ms is not None else delta)
    _last_paint = now


def note_drops(count: int) -> None:
    """Frames the queue threw away (oldest for the live stream, late for the playback grid)."""
    global _drops
    if count:
        _drops += int(count)


#: A press-to-picture turnaround below this is not a seek that happened - it is a frame that was already on its way
#: when the press landed (observed: 0.1 ms, closing with the display frame that was mid-flight). Counting it would
#: claim the seek cost nothing.
MIN_SEEK_MS = 1.0


def note_seek(turnaround_ms: float, *, target: int | None = None,
              landed: tuple[str, int] | None = None, target_s: float | None = None,
              landed_time: float | None = None, target_gen: int | None = None,
              landed_gen: int | None = None) -> None:
    """Press -> first new picture, in milliseconds, naming the frame that was asked for and the one the PANE shows.

    `target` is the frame the press pointed at (`fraction x the strip's total`). `landed` is the pane's own identity for
    the picture it paints - `("frame", n)` where the producer numbers frames, `("seq", n)` where the only real identity
    is the display channel's sequence. It is deliberately NOT the renderer's frame counter: that is a different fact
    and it produced a wrong reading once (owner, 2026-11-24: the log claimed the picture was already the requested
    frame, which the counter could not prove).
    """
    turnaround_ms = float(turnaround_ms)
    fields: dict[str, object] = {"target_frame": target}
    if target_s is not None:
        fields["target_s"] = f"{float(target_s):.3f}"
    if landed is not None:
        kind, number = landed
        fields["landed_frame" if kind == "frame" else "landed_seq"] = int(number)
    if landed_time is not None:
        # The media time of the picture on screen, and how far it is from where the press pointed. THIS is the number
        # that answers "did the seek go where it was told" - a sequence number cannot (owner, 2026-11-24).
        fields["landed_time"] = f"{float(landed_time):.3f}"
        if target_s is not None:
            fields["delta_s"] = f"{float(landed_time) - float(target_s):+.3f}"
    if target_gen is not None:
        fields["target_gen"] = int(target_gen)
    if landed_gen is not None:
        # The pair that says whether this CLOSE is legitimate: the pane asked for generation N and the picture answers
        # M. `landed_gen` absent means the picture is a streaming frame - which never closes a seek any more.
        fields["landed_gen"] = int(landed_gen)
    if turnaround_ms < MIN_SEEK_MS:
        event("seek instant", turnaround_ms=f"{turnaround_ms:.2f}", **fields)
        return
    _seek_ms.append(turnaround_ms)
    event("seek", turnaround_ms=f"{turnaround_ms:.1f}", **fields)


def _summary(samples: list[float]) -> str:
    if not samples:
        return "-"
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return f"{statistics.median(ordered):.2f}/{p95:.2f}/{ordered[-1]:.2f}"


def _shown_text() -> str:
    if _shown is None:
        return "-"
    kind, number = _shown
    return f"f{number}" if kind == "frame" else f"seq{number}"


def snapshot() -> dict[str, object]:
    """The counters since the last flush, as data (a control verb, a test)."""
    return {
        "frames": _frames,
        "paints": _paints,
        "uploads": _uploads,
        "bytes": _bytes,
        "drops": _drops,
        "gui_ms": list(_gui),
        "upload_ms": list(_upload),
        "interval_ms": list(_intervals),
        "pass_ms": list(_pass),
        "seek_ms": list(_seek_ms),
        "shown": _shown,
    }


def _reset(*, session: bool = False) -> None:
    """Drop the window's counters. ``session=True`` also forgets the clock, which is what a test starts from.

    The two are separate on purpose: a flush resets the counters and keeps the window clock running, while a new
    session (a test, a restarted recorder) forgets both - otherwise the next flush after a reset would treat itself as
    the session's first and skip a window.
    """
    global _frames, _paints, _uploads, _bytes, _drops, _last_snapshot, _started, _last_stats, _shown
    _frames = _paints = _uploads = _bytes = _drops = 0
    if session:
        # `shown` is a property of the PANE (what it is painting right now), not a per-window counter: clearing it on
        # every flush made an idle second report `shown=-` while the picture was still on screen (measured 2026-11-24).
        _shown = None
        _started = False
        _last_stats = 0.0
    _gui.clear()
    _upload.clear()
    _intervals.clear()
    _pass.clear()
    _seek_ms.clear()
    _last_snapshot = {}


def session_start(**fields) -> None:
    """One line once per started pane: where the file is and what the pane was at that moment."""
    _write(f"preview: session file={log_path()} " + " ".join(f"{k}={v}" for k, v in fields.items()),
           "info")


def maybe_stats(now: float | None = None, *, force: bool = False) -> bool:
    """Write the aggregate line when a second has passed (or on demand); True when it wrote one."""
    global _last_stats
    now = _clock() if now is None else now
    global _started
    if not _started:
        # The clock is monotonic, i.e. arbitrary: a window measured from zero reports the machine's uptime as the
        # session length (observed: window_s=86132.5 on the first line). The first call starts the clock instead.
        _started = True
        _last_stats = now
        if not force:
            return False
    if not force:
        if level() < 1 or now - _last_stats < STATS_PERIOD_S:
            return False
    minutes = now - _last_stats
    _last_stats = now
    line = (
        f"preview: stats frames={_frames} paints={_paints} up={_uploads} "
        f"up_ms={_summary(_upload)} pass_ms={_summary(_pass)} gui_ms={_summary(_gui)} "
        f"int_ms={_summary(_intervals)} drop={_drops} shown={_shown_text()} "
        f"seek_ms={_summary(_seek_ms)} "
        f"mb={_bytes / 1e6:.1f} window_s={minutes:.1f}"
    )
    _last_snapshot = snapshot()
    _write(line, "info")
    _reset()
    return True
