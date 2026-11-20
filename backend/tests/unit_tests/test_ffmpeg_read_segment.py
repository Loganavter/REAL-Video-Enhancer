"""The segment bound on the video read: one flag pair used to be accepted and silently ignored.

Row ``rve-start-end-time-ignored-2026-11-21``: ``--start_time 0 --end_time 8`` on a 25-minute episode rendered
9238 frames (385 s) instead of 192, because ``FFmpegRead`` stored both times and never built them into its ffmpeg
command. That made every ad-hoc segment measurement cost minutes of GPU and describe a longer population than it
claimed.

Two layers, matching the two things that can break:
* the argument arithmetic, pure and importable without cv2/torch (a verification cell has neither);
* the read itself, on a clip this test generates with the bundled ffmpeg, counting frames - the acceptance says a
  bounded read must return round((end - start) * fps) frames.
"""

import os
import subprocess
import sys

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.utils.Segment import audio_trim_filter, segment_args  # noqa: E402

FFMPEG = os.path.join(os.path.dirname(BACKEND_DIR), "bin", "ffmpeg")


@pytest.mark.parametrize("start,end,expected", [
    (None, None, []),                                   # no bound at all: the old behaviour, deliberately
    (0, 8, ["-t", "8.000000"]),                         # end only: starts at 0
    (1, None, ["-ss", "1.000000"]),                     # start only: runs to the end of the file
    (1, 2.5, ["-ss", "1.000000", "-t", "1.500000"]),    # a segment: duration, not the end timestamp
    ("", 0, ["-t", "0.000000"]),                        # empty strings from an older settings/caller path
    (2, 1, ["-ss", "2.000000", "-t", "0.000000"]),      # degenerate: never negative
])
def test_segment_args(start, end, expected):
    assert segment_args(start, end) == expected


def _make_clip(path, seconds=4, fps=25, size="160x120"):
    """A synthetic clip with the bundled ffmpeg: deterministic, tiny, and no repo fixture to keep in sync."""
    result = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not generate a test clip: {result.stderr.strip()[:200]}")


def _count_frames(reader):
    """Drain the reader and count complete frames."""
    width, height = reader.width, reader.height
    chunk = reader.inputFrameChunkSize
    count = 0
    while True:
        data = reader.readProcess.stdout.read(chunk)
        if len(data) < chunk:
            break
        count += 1
        if count > 10_000:                              # a bound that does not work must fail, not hang
            break
    reader.readProcess.stdout.close()
    reader.readProcess.wait(timeout=30)
    return count


@pytest.fixture()
def clip(tmp_path):
    if not os.path.exists(FFMPEG):
        pytest.skip("no bundled ffmpeg")
    path = str(tmp_path / "testsrc.mkv")
    _make_clip(path)
    return path


def _reader(clip, start, end):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torch")
    from src.FFmpegBuffers import FFmpegRead
    return FFmpegRead(
        inputFile=clip, width=160, height=120, start_time=start, end_time=end,
        borderX=0, borderY=0, hdr_mode=False, backend="pytorch", device="cuda", gpu_id=0,
        dtype="float16", color_space=None, color_primaries=None, color_transfer=None,
        color_range=None, input_pixel_format="yuv420p", ffmpeg_path=FFMPEG)


def test_a_bounded_read_returns_the_segment_only(clip):
    """The acceptance's own number: 1.0-2.5 s at 25 fps is 37-38 frames, not the clip's 100."""
    whole = _count_frames(_reader(clip, 0, None))
    middle = _count_frames(_reader(clip, 1.0, 2.5))
    assert 95 <= whole <= 101, f"the unbounded read returned {whole} frames for a 4 s / 25 fps clip"
    assert 34 <= middle <= 40, f"the bounded read returned {middle} frames, expected ~38 for 1.5 s at 25 fps"


def test_the_bound_survives_the_reader_that_used_to_ignore_it(clip):
    """A reader built with both times must put them in its command - the regression this row is about."""
    reader = _reader(clip, 1.0, 2.5)
    try:
        command = reader.command()
        assert "-ss" in command and "1.000000" in command
        assert "-t" in command and "1.500000" in command
        # both are INPUT options: -ss seeks, -t bounds how much of the input is read. Placed before -i so the
        # seek does not decode what precedes it, which is what makes a segment test cheap.
        assert command.index("-ss") < command.index("-i")
        assert command.index("-t") < command.index("-i")
    finally:
        reader.readProcess.stdout.close()
        reader.readProcess.kill()
        reader.readProcess.wait(timeout=30)


@pytest.mark.parametrize("start,end,expected", [
    (0, 8, "atrim=start=0.000000:end=8.000000,asetpts=PTS-STARTPTS"),
    (120, 123, "atrim=start=120.000000:end=123.000000,asetpts=PTS-STARTPTS"),
    (None, None, "atrim=start=0.000000,asetpts=PTS-STARTPTS"),          # unbounded: the old filter
    (120, None, "atrim=start=120.000000,asetpts=PTS-STARTPTS"),         # start only: play to the end
    (2, 1, "atrim=start=2.000000,asetpts=PTS-STARTPTS"),                # degenerate: never an empty range
])
def test_audio_trim_matches_the_video_bound(start, end, expected):
    """A bounded video must not carry audio past its last frame."""
    assert audio_trim_filter(start, end) == expected


def test_a_mid_file_segment_renders_about_three_seconds(clip):
    """`-t` is a time bound, so the count lands within a frame of round((end-start)*fps).

    Measured on a 10 s run of a 25-minute episode: 120-123 s gives 73 frames at 23.976 fps where the arithmetic says
    72 - one frame at the segment edge, not minutes of the whole file.
    """
    count = _count_frames(_reader(clip, 1.0, 2.0))
    assert 24 <= count <= 27, f"{count} frames for 1 s at 25 fps"
