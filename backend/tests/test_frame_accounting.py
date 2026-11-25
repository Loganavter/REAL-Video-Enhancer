"""The frame statistics count source packets, not CFR slots (owner, 2026-11-25: "493/495").

A container can carry a PTS gap: the owner's clip has 493 video packets with one gap at 20.479 -> 20.604 s, while
`duration x fps` (and OpenCV's `CAP_PROP_FRAME_COUNT`) says 495 CFR slots. The renderer's progress is source frames; the
old probe's total was the slot estimate, so the UI read `493/495` as two lost frames. The writer separately rebuilds CFR
(495 output frames, two repeats) — a second population that must never be subtracted from the source count.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from backend.src.utils.MediaCount import count_video_packets

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FFMPEG = os.path.join(ROOT, "bin", "ffmpeg")

pytestmark = pytest.mark.skipif(not os.path.exists(FFMPEG), reason="the bundled ffmpeg is not present")


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-v", "error", *args], check=True)


def _make_clip(path: str, frames: int = 10, fps: int = 10) -> None:
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=64x64:rate={fps}:duration={frames / fps}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", path)


def test_the_packet_count_is_exact_and_independent_of_the_cfr_slots(tmp_path):
    source = str(tmp_path / "clip.mkv")
    gap = str(tmp_path / "gap.mkv")
    _make_clip(source, frames=10)
    # Drop one frame but keep the other frames' own PTS: the container now has 9 packets across a 10-slot timeline,
    # the same shape as the owner's 493 packets / 495 slots.
    _ffmpeg("-i", source, "-vf", "select='not(eq(n,5))'", "-fps_mode", "passthrough",
            "-c:v", "libx264", "-y", gap)

    assert count_video_packets(FFMPEG, source) == 10
    assert count_video_packets(FFMPEG, gap) == 9, "the exact source count follows the packets, not the slot estimate"
    assert count_video_packets(FFMPEG, gap, start_time=0.0, end_time=1.0) == 9

    # The two populations are different facts: the packet count is 9, the CFR slot count (duration x fps) is 10.
    # The writer would emit 10 output frames (one repeat); no UI may print 9/10 as "one frame lost".
    banner = subprocess.run([FFMPEG, "-hide_banner", "-i", gap], capture_output=True, text=True).stderr
    assert "10 fps" in banner and "Duration: 00:00:01.00" in banner


def test_a_whole_file_count_is_none_when_ffmpeg_cannot_read_it(tmp_path):
    assert count_video_packets(FFMPEG, str(tmp_path / "missing.mkv")) is None
