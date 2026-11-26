"""The preview's numbers, each derived in exactly ONE place.

Four pure functions: how long the clip is, how many seconds of it the upscaler has produced, how far a press may
reach, and the `f done / total` readout. No Qt, no widget, no host object: every one of them is a function of values
the page already holds, so the numbers can be checked with no window and no render.

Why they live outside `ProcessTab`. The same quantity used to be computed in two places and the copies drifted; the
owner's 2026-11-26 screenshot is the case that named the class - an export at frame 67 of 493 with a fully blue strip,
because the pane reported the whole clip as produced for a `DISPLAY_FEED` while the render stood at 14 %. One
derivation per number, one caller per derivation, is the guard that cannot drift apart.

The reach and the produced band are deliberately DIFFERENT functions even though both are "seconds": the band is what
the upscaler has written, the reach is what a press may ask the host to render on demand. Collapsing them is the
`produced`-as-the-whole-clip defect with a longer name.

`PreviewBackendKind` is the only import that is not the standard library: `preview.state` is Qt-free by construction
(Enum + frozen dataclass), so this module imports on an interpreter without PySide6.
"""

from __future__ import annotations

from .state import PreviewBackendKind

__all__ = ["clip_duration", "produced_seconds", "reach_seconds", "frame_ratio"]


def clip_duration(options, decoder=None, queue_duration: float = 0.0) -> float:
    """The clip's length in seconds: the queue item's own length, else the opened file's, else frames / fps.

    `decoder` is whatever holds the OPENED file's duration (the decoder's `opened` signal is where the page's copy of
    this number comes from), so an object with a `duration` reading is enough. `queue_duration` is the queue pane's
    length, passed by the page while the queue's `QUEUE_FILE` owner is the pane's clock; 0.0 otherwise, which keeps
    the preview and export numbers exactly.

    Returns 0.0 when nothing knows the length: the caller then has no axis to place a marker on.
    """
    if float(queue_duration or 0.0) > 0.0:
        return float(queue_duration)
    if decoder is not None:
        opened = float(getattr(decoder, "duration", 0.0) or 0.0)
        if opened > 0.0:
            return opened
    frames = float(getattr(options, "videoFrameCount", 0) or 0)
    fps = float(getattr(options, "videoFps", 0) or 0)
    if frames > 0.0 and fps > 0.0:
        return frames / fps
    return 0.0


def produced_seconds(kind: PreviewBackendKind, *, fraction: float, duration: float,
                     rendering: bool) -> float:
    """What the strip's BLUE means: seconds the upscaler has produced.

    A finished file is fully produced - and so is a queue item, which is a finished file the queue owns (its whole
    length exists the moment it opens). Everything else is a live render and the band is its progress - including an
    EXPORT whose pane follows the display feed: that pane used to report the whole clip as produced, so a render at
    14 % showed the whole bar blue and the grey "not upscaled yet" half never appeared (owner, 2026-11-26, screenshot
    of an export at frame 67/493 with a fully blue strip). The feed can seek anywhere on demand, but that is the
    CEILING (`reach_seconds`), not the produced share.
    """
    if duration <= 0.0:
        return 0.0
    if kind in (PreviewBackendKind.FINISHED_FILE, PreviewBackendKind.QUEUE_FILE):
        return duration
    if kind is PreviewBackendKind.DISPLAY_FEED and not rendering:
        # The render is over and the feed is showing the finished file: there is nothing left to produce.
        return duration
    return max(0.0, min(1.0, float(fraction))) * duration


def reach_seconds(kind: PreviewBackendKind, *, duration: float, produced: float,
                  any_frame: bool) -> float:
    """The furthest content time a press may ask for.

    A finished file (and a queue item, which is one the queue owns) and a feed reach the whole clip. So does a
    preview whose block is usable (`any_frame`): the host renders ANY timeline frame on demand, past the produced
    band. A growing file's tail has only what the renderer has written, so it reaches the produced edge - and
    `produced` is that edge, in seconds, from `produced_seconds` (or the tail's own flushed reading), never a second
    definition of the band.

    0.0 when the length is unknown: there is nothing honest to seek to.
    """
    if duration <= 0.0:
        return 0.0
    if kind in (PreviewBackendKind.FINISHED_FILE, PreviewBackendKind.QUEUE_FILE,
                PreviewBackendKind.DISPLAY_FEED):
        return float(duration)
    if any_frame:
        return float(duration)
    return max(0.0, float(produced))


def frame_ratio(source_frames: int, frames_done: int) -> str:
    """`f <source frames processed> / <source frames>`.

    BOTH numbers are the source packet set (`frames_done`/`source_frames`). The writer's CFR count is a separate fact
    (`output_frames`) and is never a denominator here: subtracting the two populations is exactly the `493/495` report
    of 2026-11-25. `frames_done` is clamped into `[0, source_frames]` - a tick can sample a counter that is one packet
    past the probe's total at the very end, and "f 494 / 493" would read as a second truth.
    """
    total = max(0, int(source_frames))
    done = max(0, min(total, int(frames_done))) if total else max(0, int(frames_done))
    if total <= 0:
        return "f \u2014"
    return f"f {done} / {total}"
