"""Segment bounds for the video read: pure arithmetic, importable where cv2 and torch are not."""

from __future__ import annotations


def segment_args(start_time, end_time) -> list[str]:
    """The ffmpeg arguments that bound a read to ``[start_time, end_time]``.

    ``FFmpegRead`` accepted both times since it was written and never built them into its command, so the reader
    streamed to EOF: a "3 second" test render of a 25-minute episode measured 9238 frames / 385 s (row
    ``rve-start-end-time-ignored-2026-11-21``), which made every ad-hoc segment measurement cost minutes of GPU and
    describe a longer population than it claimed.

    Both are INPUT options and both go BEFORE ``-i``: ``-ss`` is a keyframe seek refined to the exact timestamp by
    the decoder (frame-accurate since ffmpeg 2.1) and ``-t`` bounds how much of the input is read, so the seek does
    not decode what precedes it. ``end_time`` alone starts at 0, ``start_time`` alone still runs to the end of the
    file, and the measured frame count matches round((end - start) * fps) on a synthetic 25 fps clip.
    """
    args: list[str] = []
    start = 0.0
    if start_time not in (None, ""):
        try:
            start = max(0.0, float(start_time))
        except (TypeError, ValueError):
            start = 0.0
    if start > 0.0:
        args += ["-ss", f"{start:.6f}"]
    if end_time not in (None, ""):
        try:
            end = float(end_time)
        except (TypeError, ValueError):
            end = None
        if end is not None:
            args += ["-t", f"{max(0.0, end - start):.6f}"]
    return args


def audio_trim_filter(start_time, end_time) -> str:
    """``atrim`` for the segment, so a bounded video does not carry audio past its last frame.

    The writer trimmed the START only (`atrim=start=S`), which was invisible while the reader ignored the bound:
    the file was always the whole source. With the read bounded to ``[start, end]`` an end-less trim leaves minutes
    of audio after the last video frame.
    """
    start = 0.0
    if start_time not in (None, ""):
        try:
            start = max(0.0, float(start_time))
        except (TypeError, ValueError):
            start = 0.0
    end = None
    if end_time not in (None, ""):
        try:
            end = float(end_time)
        except (TypeError, ValueError):
            end = None
    if end is not None and end > start:
        return f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS"
    return f"atrim=start={start:.6f},asetpts=PTS-STARTPTS"
