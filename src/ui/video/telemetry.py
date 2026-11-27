"""Numbers for the preview pane, so a report can carry a measurement instead of a guess.

Why this module exists: every question about this pane used to be answered by argument - "is the upload the cost",
"does the seek pay a decode", "is the picture stale or is the paint cadence off". Each has a number, and the pane
never wrote one down. This is the one place that counts them and writes them as greppable lines.

Levels (`RVE_PREVIEW_LOG`):

* unset / `0` - **events only**: one line per action (a press, a seek, a mode change, an RHI re-init). Cheap enough
  to leave on; `log(..., level="info")` keeps them in `log.txt` beside the rest of the app's record.
* `1` - events **+ one aggregate line per second**: frame/paint/upload counts, per-stage milliseconds, the interval
  between paints, drops, and the seek turnaround - plus ONE `snapshot` line per second carrying the preview state's own
  numbers (`kind, duration, produced, played, playhead, finished, ticket.open, epoch`), the same keys the control
  channel's preview dump serves. The state line is `note_snapshot`; the counters are `maybe_stats`.
* `2` - everything above **+ one line per frame**. Use it for a short session only; at 24-60 fps it is a lot of text.

A `draw` line is a **DELTA record** at levels 0/1: the pane writes one only when the picture or the geometry changes, so
its absence means "no change", NOT "no paint". At level 2 it is written per paint. The line says which it is
(`record=delta` / `record=paint`); count paints with `stats paints=`, never by counting `draw` lines.

The lines go to the pane's OWN file - `preview-log.txt` next to the app's `log.txt` (or `RVE_PREVIEW_LOG_FILE`) -
one line per record, flushed as it is written, with a header per process so two sessions cannot be mistaken for one.
That file is the thing to paste into a report; the app's `log.txt` stays clean.

The lines are `key=value` tokens separated by spaces. A value that carries a space, a quote or a backslash is
double-quoted and escaped (`reason="queue-decode-failed: boom happened"`), and a newline in a value is written as the
two characters `\n`, so one record is always one line. `telemetry.parse(line)` is the reader that survives that
(space-splitting and `dict(part.split("=", 1))` do not: they turn the reason above into `queue-decode-failed:`):

    preview: stats frames=24 paints=24 up=24 up_ms=0.01/1.34 pass_ms=0.40/0.98 gui_ms=0.62/0.98 int_ms=41.2/44.0/90.1
    preview: stats … seek_ms=38.4/… window_s=1.0

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

#: How often the preview STATE's aggregate (`note_snapshot`) may be written, seconds - its own window, separate from
#: the frame counters: one `snapshot` line per second regardless of which counter has a sample.
SNAPSHOT_PERIOD_S = 1.0

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


def _field(key: str, value) -> str:
    """One `key=value` token, quoted when the value could be mistaken for a separator.

    A bare value must not carry whitespace, a quote or a backslash; anything else is double-quoted with `\\` and `"`
    escaped and control characters written as `\\n`/`\\r`/`\\t`, so a value can never split a record or inject a second
    one. `parse` is the matching reader.
    """
    text = value if isinstance(value, str) else str(value)
    if text and not any(character.isspace() or character in '"\\' for character in text):
        return f"{key}={text}"
    escaped = (text.replace("\\", "\\\\").replace('"', '\\"')
               .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return f'{key}="{escaped}"'


def parse(line: str) -> dict[str, str]:
    """Read ONE record back into its fields (`preview:` lines only; anything else answers `{}`).

    The counterpart of `_field`'s quoting: `shlex` keeps a quoted value whole, so
    `preview: seek refused press=1 reason="queue-decode-failed: boom happened"` answers
    `{"_event": "seek refused", "press": "1", "reason": "queue-decode-failed: boom happened"}`.
    `_event` is the record's kind (which may be two words) - the other keys are the line's own fields. A value carrying
    a newline comes back with the escape (`\\n`) it was written as: a record is one line by construction.
    """
    import shlex

    text = line.strip()
    if not text.startswith("preview: "):
        return {}
    try:
        tokens = shlex.split(text[len("preview: "):])
    except ValueError:          # an unbalanced quote is a malformed record, not a crash for the reader
        return {}
    first = next((index for index, token in enumerate(tokens) if "=" in token), len(tokens))
    fields: dict[str, str] = {"_event": " ".join(tokens[:first])}
    for token in tokens[first:]:
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def wall() -> str:
    """Wall clock, seconds with milliseconds, for joining this file with the backend's.

    `monotonic` cannot cross a process boundary; both processes are on one host, so `time.time()` can. That join is
    what turns `turnaround_ms` (a sum) into its parts: press -> backend picked the request -> decoded -> published ->
    painted. Nothing here assumes the two clocks tick together beyond being the same system clock.
    """
    return f"{time.time():.3f}"


def event(kind: str, **fields) -> None:
    """One line per action: always written (``level="info"``), so a session leaves a trace of what was pressed.

    `kind == "draw"` also carries `record=`: `delta` when the writer only calls it on a change (levels 0/1) and `paint`
    at level 2, where it is one line per paint. A reader that counts draw lines as paints is counting changes.
    """
    parts = [_field(key, value) for key, value in fields.items() if value is not None]
    if kind == "draw":
        parts.append(_field("record", "paint" if level() >= 2 else "delta"))
    _write(f"preview: {kind} {' '.join(parts)}".rstrip(), "info")


# --- the aggregate ---------------------------------------------------------------------------------
_frames = 0
_paints = 0
#: Compare-mode paints, and how many of them showed the two halves at DIFFERENT moments (the flicker's number).
_pair_total = 0
_pair_off = 0
#: Paints whose halves were within a frame and a half but not within half a frame: the timestamps' own drift, not a
#: mismatch the eye would see.
_pair_strict = 0
#: Compare-mode paints that showed ONE half (`CommittedPair.lone_half`). They are not in `_pair_off`'s numerator and
#: used to be missing from its denominator too, so a pane that painted the result alone every second read as perfectly
#: paired (telemetry audit 2026-11-27).
_pair_lone = 0
#: Of the sub-frame paints: which side of the result the source was on. Alternating values = a wobble, one value = offset.
_pair_ahead = 0
_pair_behind = 0
_uploads = 0
_bytes = 0
_gui: list[float] = []
_upload: list[float] = []
_intervals: list[float] = []
_pass: list[float] = []
_last_paint: float | None = None
_last_stats = 0.0
_started = False
#: What the pane is painting: `("frame", n)` from a decoder's own number, `("seq", n)` from the display channel's
#: sequence. A pair on purpose - the two numbers mean different things and must not be printed as if they were one.
_shown: tuple[str, int] | None = None
_seek_ms: list[float] = []
#: the last written snapshot, for a control verb or a test to read without waiting for the next second
_last_snapshot: dict[str, object] = {}
#: The state aggregate's own window (`note_snapshot`): its first call starts it, so a monotonic clock is never printed.
_snapshot_started = False
_last_snapshot_at = 0.0


def note_frame(*, gui_ms: float | None = None, pts: float | None = None, index: int | None = None,
               seq: int | None = None, media_time: float | None = None,
               generation: int | None = None, cleared: bool = False) -> None:
    """A picture was handed to the pane: `index` is a decoder's frame number, `seq` the display channel's sequence.

    `media_time` and `generation` are the picture's OWN names, straight from the header that carried it (or from the
    tail frame's PTS). They are printed for every picture, with or without a seek: "which frame was drawn" must be
    answerable from the log alone (owner, 2026-11-24 - the line used to say only `frame seq=98`).

    `cleared=True` is the OTHER case: the pane was told to be empty. It does not count as a picture and it forgets the
    name - `frames` used to increment and `shown=` kept naming the last picture after a clear, so an empty pane read as
    a pane showing a frame (telemetry audit 2026-11-27, 5.3).
    """
    global _frames, _shown
    if cleared:
        _shown = None
        if gui_ms is not None:
            _gui.append(float(gui_ms))
        if level() >= 1:
            event("frame cleared", gen=generation)
        return
    _frames += 1
    if index is not None:
        _shown = ("frame", int(index))
    elif seq is not None:
        _shown = ("seq", int(seq))
    time_text = None
    if media_time is not None:
        time_text = f"{float(media_time):.3f}"
    elif pts is not None:
        time_text = f"{float(pts):.3f}"
    if gui_ms is not None:
        _gui.append(float(gui_ms))
    if level() >= 2:
        event("frame", index=index, seq=seq, t=time_text, gen=generation, gui_ms=_fmt(gui_ms))


def note_upload(*, upload_ms: float | None = None, size_bytes: int | None = None) -> None:
    """The GPU got a new picture (one upload per new frame, never per paint)."""
    global _uploads, _bytes
    _uploads += 1
    if upload_ms is not None:
        _upload.append(float(upload_ms))
    if size_bytes:
        _bytes += int(size_bytes)


def note_pair(*, delta_frames: float | None = None, lone_half: str | None = None) -> None:
    """One compare-mode paint: how far apart the two halves were, in frames - or that there was one half.

    Level-independent on purpose: a per-paint log line costs more than the thing it measures (at 24 fps the level 2 run
    distorted the very pairing it was counting), while this counter rides the one line per second that level 1 already
    writes. `pair_off` counts paints whose halves were more than half a frame apart - the flicker, as a number.

    `delta_frames` must be computed in the PANE's cadence (seconds / the pane's own frame period), never with a
    hard-coded rate: multiplied by 24 a 60 fps clip read a two-frame desync as 0 (telemetry audit 2026-11-27).
    `lone_half` is a paint that showed one half (`"result"` or `"source"`); it enters the denominator and a counter of
    its own, because treating it as a perfect pair is how a pane painting alone looked healthy.
    """
    global _pair_total, _pair_off, _pair_strict, _pair_ahead, _pair_behind, _pair_lone
    _pair_total += 1
    if lone_half:
        _pair_lone += 1
        return
    if delta_frames is None:
        raise ValueError("note_pair needs delta_frames or lone_half")

    if abs(float(delta_frames)) > 1.5:
        # "Off" is what the EYE would call a different moment: more than a frame and a half (the gate's own tolerance).
        _pair_off += 1
    elif abs(float(delta_frames)) > 0.5:
        _pair_strict += 1
        # Which way the half is off. The owner's report - "то на кадр впереди, то на кадр позади" - is this pair of
        # counters alternating; one of them alone growing is a constant offset, not a wobble.
        if float(delta_frames) > 0.0:
            _pair_ahead += 1
        else:
            _pair_behind += 1


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


#: A press-to-picture turnaround below this is not a seek that happened - it is a frame that was already on its way
#: when the press landed (observed: 0.1 ms, closing with the display frame that was mid-flight). Counting it would
#: claim the seek cost nothing.
MIN_SEEK_MS = 1.0


def note_seek(turnaround_ms: float, *, target: int | None = None,
              landed: tuple[str, int] | None = None, target_s: float | None = None,
              landed_time: float | None = None, target_gen: int | None = None,
              landed_gen: int | None = None, press: int | None = None,
              ts: str | None = None) -> None:
    """Press -> first new picture, in milliseconds, naming the frame that was asked for and the one the PANE shows.

    `target` is the frame the press pointed at (`fraction x the strip's total`). `landed` is the pane's own identity for
    the picture it paints - `("frame", n)` where the producer numbers frames, `("seq", n)` where the only real identity
    is the display channel's sequence. It is deliberately NOT the renderer's frame counter: that is a different fact
    and it produced a wrong reading once (owner, 2026-11-24: the log claimed the picture was already the requested
    frame, which the counter could not prove).
    """
    turnaround_ms = float(turnaround_ms)
    fields: dict[str, object] = {"target_frame": target, "press": press}
    if ts is not None:
        fields["ts"] = ts
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
        "pair_total": _pair_total,
        "pair_off": _pair_off,
        "pair_lone": _pair_lone,
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
    global _frames, _paints, _uploads, _bytes, _last_snapshot, _started, _last_stats, _shown
    global _pair_total, _pair_off, _pair_strict, _pair_ahead, _pair_behind, _pair_lone
    global _snapshot_started, _last_snapshot_at
    _frames = _paints = _uploads = _bytes =     _pair_total = _pair_off = 0
    _pair_strict = _pair_ahead = _pair_behind = _pair_lone = 0
    if session:
        # `shown` is a property of the PANE (what it is painting right now), not a per-window counter: clearing it on
        # every flush made an idle second report `shown=-` while the picture was still on screen (measured 2026-11-24).
        _shown = None
        _started = False
        _last_stats = 0.0
        _snapshot_started = False
        _last_snapshot_at = 0.0
    _gui.clear()
    _upload.clear()
    _intervals.clear()
    _pass.clear()
    _seek_ms.clear()
    _last_snapshot = {}


def session_start(**fields) -> None:
    """One line once per started pane: where the file is and what the pane was at that moment."""
    values = {"file": log_path(), **fields}
    _write("preview: session " + " ".join(_field(key, value) for key, value in values.items()), "info")


def note_snapshot(**fields) -> bool:
    """The preview STATE as ONE aggregate line per second (plan P5); True when it wrote one.

    The pane's file had no per-second reading of the state itself: the scrubber's lines are EVENTS (a press, a refusal)
    and every derived number lived only in a widget, so "the band says 20.6 s while the picture is at 8.9 s" could only
    be argued from a screenshot. This line carries the same keys the debug dump serves (`kind, duration, produced,
    played, playhead, finished, ticket.open, epoch`), so the file and the control channel describe one pane one way.

    Written at level >= 1 and at most once per `SNAPSHOT_PERIOD_S`; the level-0 file keeps the event lines the owner
    reads (a press, a seek) and gains nothing. The first call only starts the window (a monotonic clock is arbitrary),
    and a boolean prints as `1`/`0`, an unknown value as `-`, a float with three decimals.
    """
    global _snapshot_started, _last_snapshot_at
    now = _clock()
    if not _snapshot_started:
        _snapshot_started = True
        _last_snapshot_at = now
        return False
    if level() < 1 or now - _last_snapshot_at < SNAPSHOT_PERIOD_S:
        return False
    _last_snapshot_at = now
    parts = []
    for key, value in fields.items():
        if value is None:
            text = "-"
        elif isinstance(value, bool):
            text = "1" if value else "0"
        elif isinstance(value, float):
            text = f"{value:.3f}"
        else:
            text = str(value)
        parts.append(_field(key, text))
    _write("preview: snapshot " + " ".join(parts), "info")
    return True


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
        f"int_ms={_summary(_intervals)} shown={_shown_text()} "
        f"pair_off={_pair_off}/{_pair_total} pair_lone={_pair_lone} pair_sub={_pair_strict} "
        f"pair_ahead={_pair_ahead} pair_behind={_pair_behind} "
        f"seek_ms={_summary(_seek_ms)} "
        f"mb={_bytes / 1e6:.1f} window_s={minutes:.1f}"
    )
    _last_snapshot = snapshot()
    _write(line, "info")
    _reset()
    return True
