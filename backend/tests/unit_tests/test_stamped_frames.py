"""The reader can name every frame with its SOURCE time - the one thing a counter cannot do.

Why (measured 2026-11-25, owner's clip): the export's pipe is CFR-converted by default and yields **495 frames from a
493-frame container**; the two inserted frames exist in no source frame, so `frames_rendered / fps` names the picture
after the first gap with the wrong frame's time, permanently. With `stamp_pts` the reader switches the pipe to
`-fps_mode passthrough`, adds `showinfo`, and pairs one `pts_time` line per chunk BY COUNT (stderr runs ahead of the
pipe, never behind), so `Frame.pts` is the container's own timestamp. Measured on the same clip: 493 frames, monotonic,
and the gap preserved (20.479 -> 20.604).
"""

from __future__ import annotations

import pathlib
import queue
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from src.FFmpegBuffers import FFmpegRead  # noqa: E402


def _reader(**kwargs) -> FFmpegRead:
    """A reader that never starts a process: the command and the pairing are what these pins are about."""
    reader = object.__new__(FFmpegRead)
    reader.ffmpeg_path = "ffmpeg"
    reader.inputFile = "clip.mkv"
    reader.width = 4
    reader.height = 4
    reader.start_time = 0.0
    reader.end_time = None
    reader.borderX = 0
    reader.borderY = 0
    reader.hdr_mode = False
    reader.high_bitdepth = False
    reader.yuv420pMOD = False
    reader.color_space = None
    reader.color_range = None
    reader.input_pixel_format = "rgb24"
    reader.stamp_pts = bool(kwargs.get("stamp_pts", False))
    reader._pts_queue = queue.Queue() if reader.stamp_pts else None
    return reader


def test_stamped_mode_asks_for_passthrough_and_showinfo():
    """Both halves are needed: passthrough keeps the container's frame count, showinfo names each of them."""
    stamped = _reader(stamp_pts=True).command()
    assert stamped[-3:] == ["-fps_mode", "passthrough", "-"], "CFR would insert frames no source contains"
    assert any("showinfo" in part for part in stamped), "the timestamp has to be printed per frame"
    plain = _reader(stamp_pts=False).command()
    assert not any("showinfo" in part for part in plain)
    assert "-fps_mode" not in plain, "the default reader keeps the export's CFR semantics"


def test_the_timestamp_lines_parse_and_pair_in_order():
    reader = _reader(stamp_pts=True)
    for value in ("0.000000", "0.041708", "0.083417"):
        reader._pts_queue.put(float(value))
    assert [reader._take_pts() for _ in range(3)] == [0.0, 0.041708, 0.083417]


def test_the_stream_ending_is_not_a_timestamp():
    """The drain thread ends the queue with None; a caller must get None, never a fabricated time."""
    reader = _reader(stamp_pts=True)
    reader._pts_queue.put(None)
    assert reader._take_pts() is None


def test_an_unstamped_reader_never_claims_a_time():
    """"The reader could not name it" is a real answer: the renderer then falls back to its own counter."""
    assert _reader(stamp_pts=False)._take_pts() is None


def test_showinfo_lines_are_matched_by_their_own_grammar():
    """`showinfo` prints `pts_time:` (and `pts_time:` can be negative at the head of a trimmed file)."""
    match = FFmpegRead._PTS_TIME.search("[Parsed_showinfo_3 @ 0x55] n:0 pts:0 pts_time:0.041708 pos:...")
    assert match is not None and match.group(1) == "0.041708"
    negative = FFmpegRead._PTS_TIME.search("n:0 pts:-2 pts_time:-0.083417 pos:...")
    assert negative is not None and negative.group(1) == "-0.083417"
