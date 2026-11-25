"""Exact source-frame counting for a container: video packets, not `duration x fps`.

Why this module exists: a container may have a PTS gap, so `duration x fps` (and OpenCV's
`CAP_PROP_FRAME_COUNT`) counts CFR SLOTS, not frames. Measured on the owner's clip (2026-11-25): 493 video packets with
one gap at 20.479 -> 20.604 s, `duration x fps = 495`. The renderer's progress is source frames (493) while the old
probe's total was the slot estimate (495), so the UI read `493/495` — "two frames lost" that never existed. The writer
rebuilds CFR (495 packets in the output, two of them repeats): that is a SECOND population and must never be subtracted
from the source count.

The count is a demux pass (`-c copy -f null`, no decode), the same order of cost as the reader's own pass; for a segment
seek the count includes the keyframe pre-roll, exactly like the reader's `-ss`.
"""

from __future__ import annotations

import re
import subprocess

_FRAME = re.compile(r"frame=\s*(\d+)")


def count_video_packets(ffmpeg_path: str, input_file: str, start_time=None, end_time=None) -> int | None:
    """The number of video packets ffmpeg demuxes from `input_file`, or None when it cannot read it.

    `start_time`/`end_time` bound the read the way the reader's `segment_args` do: `-ss` before the input (keyframe
    seek refined to the timestamp), `-t` after it. A copy read never decodes, so a 20 s clip costs milliseconds and a
    feature-length file the demux pass only.
    """
    command = [str(ffmpeg_path), "-hide_banner", "-nostdin"]
    start = 0.0
    if start_time not in (None, ""):
        try:
            start = max(0.0, float(start_time))
        except (TypeError, ValueError):
            start = 0.0
    if start > 0.0:
        command += ["-ss", f"{start:.6f}"]
    command += ["-i", str(input_file), "-map", "0:v:0", "-c", "copy"]
    if end_time not in (None, ""):
        try:
            end = float(end_time)
            if end > start:
                command += ["-t", f"{end - start:.6f}"]
        except (TypeError, ValueError):
            pass
    command += ["-f", "null", "-"]
    try:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        text = process.stderr.read()
    except Exception:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    matches = _FRAME.findall(text or "")
    return int(matches[-1]) if matches else None
