"""The preview player's own frame path: frames, a PTS clock, a file decoder, a tail reader.

The split is deliberate. ``frames`` and ``clock`` are pure Python, because the queue policy and the
clock are the two things whose bugs read as "the preview stutters" and whose fixes have to be
testable without a window or a decoder on the interpreter. ``decode`` and ``tail`` need PySide6 for
their signals (and PyAV at call time only), so they are imported on first use: importing the package
stays possible on an interpreter without Qt, which is where the pure-half tests run.

One surface consumes all four: `VideoPreviewSurface` (``src/ui/widgets/video_preview.py``) pushes
decoder output into a `FrameQueue`, asks a `PlaybackClock` what is due, and uploads the frame the
queue hands back. Nothing in here touches a widget.
"""

from .clock import PlaybackClock
from .frames import FORMATS, PLANAR_FORMATS, FrameQueue, FrameRef, frame_pts, pack_frame, pack_planes

__all__ = [
    "FORMATS",
    "PLANAR_FORMATS",
    "FrameQueue",
    "FrameRef",
    "PlaybackClock",
    "VideoDecoder",
    "TailReader",
    "frame_pts",
    "pack_frame",
    "pack_planes",
]


def __getattr__(name: str):
    """The Qt half, on first use (see the module docstring)."""
    if name == "VideoDecoder":
        from .decode import VideoDecoder

        return VideoDecoder
    if name == "TailReader":
        from .tail import TailReader

        return TailReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
