"""The resident preview decoder: the frames it returns are the ones a per-request reader returns.

`RenderVideo._seek_preview` served every seek by opening an `FFmpegRead` at the requested time, reading
one frame and closing it - one ffmpeg spawn (start-up, demux seek, decoder init) per request, and a scrub
that arrives faster than a spawn turns into a chain of processes. `utils/SeekReader.py` keeps one decoder
alive and walks it forward. Two things make that safe and worth it, and both are measured here on a real
generated clip, not argued:

* **Identity.** Every timestamp is decoded twice - through `FFmpegRead` opened at that timestamp (the path
  that is being replaced) and through one `SeekReader` - and the two frames must be *the same bytes*, not
  the same size. Forward walks, backward respawns, off-grid timestamps and repeated requests are covered,
  and the frames must actually differ from timestamp to timestamp (otherwise identity is vacuous).
* **Cost.** Per-request vs resident milliseconds for the same 10 timestamps in one process, and the
  numbers the module's `forward_limit_seconds` rests on: a spawn costs about as much as walking a
  fraction of a second of frames, so beyond that a re-open is the cheaper answer.

The clip is generated here, inside the clone's `scratch/`, with the ffmpeg on PATH: this interpreter has
no torch and no GPU, and none of the above needs either.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import statistics
import subprocess
import time

import pytest

from src.FFmpegBuffers import FFmpegRead
from src.utils.SeekReader import SeekReader, probe_frame_grid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "scratch")
CLIP = os.path.join(SCRATCH, "seek-clip.mp4")
EVIDENCE = os.path.join(SCRATCH, "seek-reader-comparison.txt")

WIDTH, HEIGHT, FPS, DURATION = 160, 120, 10.0, 6.0
FRAME_PERIOD = 1.0 / FPS
#: 60 frames, keyframe every 12: a walk crosses GOPs, so a keyframe dependency would show up as a
#: different picture rather than a timing blip.
FRAMES = int(FPS * DURATION)


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        pytest.skip("ffmpeg is not installed")
    return binary


def _make_clip(name: str, width: int, height: int, fps: float, duration: float,
               preset: str = "medium", gop: int = 12) -> str:
    """Generate `scratch/<name>` once: a real encoded clip, so `-ss` and the decoder are the real ones."""
    ffmpeg = _ffmpeg()
    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, name)
    if not os.path.exists(path):
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps:g}:duration={duration:g}",
             "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-g", str(gop),
             "-crf", "18" if preset == "medium" else "23", path],
            check=True,
        )
    assert os.path.getsize(path) > 0
    return path


@pytest.fixture(scope="session")
def clip():
    """A 160x120, 10 fps, 6 s h264 clip - generated once inside the clone."""
    return _make_clip("seek-clip.mp4", WIDTH, HEIGHT, FPS, DURATION)


def _reader_at(path: str, seconds: float, width: int = WIDTH, height: int = HEIGHT) -> FFmpegRead:
    """The one-shot reader the seek path used before this module - the reference to reproduce."""
    return FFmpegRead(
        inputFile=path,
        width=width,
        height=height,
        start_time=float(seconds),
        end_time=None,
        borderX=0,
        borderY=0,
        hdr_mode=False,
        high_bitdepth=False,
        backend="pytorch",
        device="cpu",
        gpu_id=0,
        dtype="float16",
        color_space=None,
        color_primaries=None,
        color_transfer=None,
        color_range=None,
        input_pixel_format=None,
        ffmpeg_path=_ffmpeg(),
    )


def _per_request_frame(path: str, seconds: float, width: int = WIDTH,
                       height: int = HEIGHT) -> bytes | None:
    reader = _reader_at(path, seconds, width, height)
    try:
        return reader.read_frame()
    finally:
        reader.close()


def _resident(path: str, start_time: float = 0.0, width: int = WIDTH,
              height: int = HEIGHT, **kwargs) -> SeekReader:
    return SeekReader(
        path, width=width, height=height, start_time=start_time,
        end_time=None, borderX=0, borderY=0, hdr_mode=False, high_bitdepth=False,
        backend="pytorch", device="cpu", gpu_id=0, dtype="float16",
        color_space=None, color_primaries=None, color_transfer=None, color_range=None,
        input_pixel_format=None, ffmpeg_path=_ffmpeg(), **kwargs,
    )


def _md5(chunk: bytes | None) -> str | None:
    return None if chunk is None else hashlib.md5(chunk).hexdigest()


def _assert_identical(path: str, seconds: float, chunk: bytes | None) -> str | None:
    expected = _per_request_frame(path, seconds)
    assert _md5(chunk) == _md5(expected), (
        f"frame at {seconds:.3f}s differs between the per-request reader and the resident one "
        f"({_md5(expected)} vs {_md5(chunk)})")
    return _md5(chunk)


# -- identity ----------------------------------------------------------------------------------------


def test_a_forward_walk_returns_the_per_request_frames(clip):
    """Ascending requests are served by reading and skipping; each frame must be byte-identical."""
    timestamps = [0.0, 0.1, 0.3, 0.7, 1.0, 1.4, 2.0, 2.9, 3.5]
    reader = _resident(clip, start_time=timestamps[0])
    try:
        digests = [_assert_identical(clip, t, reader.frame_at(t)) for t in timestamps]
    finally:
        reader.close()
    assert all(digests), "every timestamp inside the clip has a frame"
    assert len(set(digests)) > 5, "the frames differ from each other, so identity is not vacuous"
    assert reader.spawns == 1, f"one forward walk is one spawn, got {reader.spawns}"
    assert reader.frames_skipped >= 20, "the walk really skipped frames, it did not respawn"


def test_a_backward_seek_respawns_and_still_matches(clip):
    """A decoder cannot read backwards: the request behind the position re-opens at the target."""
    reader = _resident(clip, start_time=3.0)
    try:
        forward = _assert_identical(clip, 3.0, reader.frame_at(3.0))
        spawns_after_first = reader.spawns
        backward = _assert_identical(clip, 0.9, reader.frame_at(0.9))
        assert reader.spawns == spawns_after_first + 1, "a backward move is exactly one re-open"
        _assert_identical(clip, 0.5, reader.frame_at(0.5))
        _assert_identical(clip, 2.4, reader.frame_at(2.4))
    finally:
        reader.close()
    assert forward and backward and forward != backward


def test_off_grid_timestamps_pick_the_frame_ffmpeg_picks(clip):
    """A UI scrub asks for arbitrary times, not frame boundaries: same rule as `-ss` (first frame >= t)."""
    timestamps = [1.05, 1.949, 2.999, 0.02, 4.333, 4.34]
    reader = _resident(clip, start_time=1.0)
    try:
        _assert_identical(clip, 1.0, reader.frame_at(1.0))
        for seconds in timestamps:
            _assert_identical(clip, seconds, reader.frame_at(seconds))
    finally:
        reader.close()


def test_a_repeated_timestamp_returns_the_same_frame(clip):
    reader = _resident(clip, start_time=1.0)
    try:
        first = reader.frame_at(1.0)
        again = reader.frame_at(1.0)
        later = reader.frame_at(1.0)
    finally:
        reader.close()
    assert _md5(first) == _md5(again) == _md5(later)


def test_past_the_end_returns_none(clip):
    reader = _resident(clip, start_time=1.0)
    try:
        assert reader.frame_at(DURATION + 0.5) is None
        assert reader.frame_at(10_000.0) is None
        # ... and the reader still answers a request inside the clip afterwards
        assert _assert_identical(clip, 2.0, reader.frame_at(2.0))
    finally:
        reader.close()


def test_an_unreadable_source_returns_none_instead_of_raising(clip):
    reader = SeekReader(
        os.path.join(SCRATCH, "no-such-file.mp4"), width=WIDTH, height=HEIGHT, start_time=0.0,
        end_time=None, borderX=0, borderY=0, hdr_mode=False, high_bitdepth=False,
        backend="pytorch", device="cpu", gpu_id=0, dtype="float16",
        color_space=None, color_primaries=None, color_transfer=None, color_range=None,
        input_pixel_format=None, ffmpeg_path=_ffmpeg(),
    )
    try:
        assert reader.frame_at(1.0) is None
        assert reader.frame_at("not a time") is None
        assert reader.frame_at(float("nan")) is None
    finally:
        reader.close()
    assert reader.frame_at(1.0) is None, "a closed reader answers None instead of opening a decoder"


# -- the frame grid ----------------------------------------------------------------------------------


def test_the_grid_probe_reads_a_constant_rate(clip):
    first, pitch = probe_frame_grid(clip, _ffmpeg(), frames=13, near=0.0)
    assert abs(pitch - FRAME_PERIOD) < 1e-6, f"pitch {pitch} should be one frame of a 10 fps clip"
    assert abs(first) < FRAME_PERIOD, f"probe starts on the first frame, got {first}"


def test_the_grid_probe_refuses_a_missing_file():
    assert probe_frame_grid(os.path.join(SCRATCH, "no-such-file.mp4"), _ffmpeg()) is None


# -- cost --------------------------------------------------------------------------------------------


def test_the_resident_reader_is_cheaper_than_a_spawn_per_seek(clip):
    """10 timestamps, one process, both paths - the comparison `DONE.md` and the module quote."""
    timestamps = [round(0.3 + 0.42 * index, 3) for index in range(10)]

    per_request: list[float] = []
    for seconds in timestamps:
        started = time.perf_counter()
        chunk = _per_request_frame(clip, seconds)
        per_request.append((time.perf_counter() - started) * 1000.0)
        assert chunk is not None, f"{seconds}s is inside the clip"

    # Resident: the object is built as the render builds it (start_time = the first request), so the
    # first of the ten includes the grid probe and the first spawn.
    resident = _resident(clip, start_time=timestamps[0])
    resident_times: list[float] = []
    try:
        for seconds in timestamps:
            started = time.perf_counter()
            chunk = resident.frame_at(seconds)
            resident_times.append((time.perf_counter() - started) * 1000.0)
            assert chunk is not None
        stats = resident.stats()
        spawns = resident.spawns
        skipped = resident.frames_skipped
    finally:
        resident.close()

    warm = resident_times[1:]
    median_per_request = statistics.median(per_request)
    median_resident = statistics.median(resident_times)
    median_warm = statistics.median(warm)
    steps = int(round(0.42 * FPS))
    # What the walk costs per skipped frame, and therefore how many frames a spawn is worth: the
    # module's forward_limit_seconds is one spawn's worth of walking.
    walk_ms_per_frame = median_warm / steps if steps else 0.0
    break_even_frames = median_per_request / walk_ms_per_frame if walk_ms_per_frame else float("inf")

    lines = [
        "SeekReader vs one FFmpegRead spawn per request",
        f"clip: {os.path.basename(clip)}, {WIDTH}x{HEIGHT}, {FPS:g} fps, {DURATION:g} s, rgb24 read",
        f"timestamps (s): {', '.join(f'{t:.2f}' for t in timestamps)}",
        f"per request (ms): {', '.join(f'{v:.1f}' for v in per_request)}",
        f"resident (ms):    {', '.join(f'{v:.1f}' for v in resident_times)}",
        f"median per request: {median_per_request:.1f} ms",
        f"median resident (all 10, first includes the spawn+probe): {median_resident:.1f} ms",
        f"median resident (9 warm): {median_warm:.1f} ms",
        f"walk cost per skipped frame: {walk_ms_per_frame:.2f} ms "
        f"(a {steps}-frame, {0.42:.2f}s step)",
        f"one spawn is worth about {break_even_frames:.1f} frames "
        f"({break_even_frames / FPS:.2f} s of walking)",
        f"decoder spawns over the 10 requests: {spawns}; frames skipped: {skipped}",
        f"stats: {stats}",
    ]
    evidence = "\n".join(lines)
    os.makedirs(SCRATCH, exist_ok=True)
    with open(EVIDENCE, "w", encoding="utf-8") as handle:
        handle.write(evidence + "\n")
    print("\n" + evidence)

    assert spawns == 1, f"ten ascending requests are one spawn, got {spawns}"
    assert median_warm < median_per_request, (
        f"a resident walk ({median_warm:.1f} ms) must beat a spawn per request "
        f"({median_per_request:.1f} ms) on this clip")


def test_the_forward_limit_reopens_instead_of_walking_forever(clip):
    """A jump further than the walk is worth pays one spawn - the same frame, not the same cost."""
    reader = _resident(clip, start_time=0.2, forward_limit_seconds=0.5)
    try:
        assert _assert_identical(clip, 0.2, reader.frame_at(0.2))
        spawns = reader.spawns
        # 2.5 s ahead: 25 frames of walking against a 5-frame limit -> re-open.
        assert _assert_identical(clip, 2.7, reader.frame_at(2.7))
        assert reader.spawns == spawns + 1
        # Within the limit: no new spawn.
        spawns = reader.spawns
        assert _assert_identical(clip, 3.1, reader.frame_at(3.1))
        assert reader.spawns == spawns
    finally:
        reader.close()


def test_the_break_even_shrinks_as_the_frame_grows():
    """The default `forward_limit_seconds` is a cost decision, so measure it where frames are big.

    A walk pays the pipe for every skipped frame; a spawn pays it once (plus its start-up). That is why
    the break-even is 42 s at a 160x120 read and about a second at 1080p/2160p - the numbers behind the
    module's default, reproduced here so the constant is not folklore. The assertions are only the
    ordering (a warm walk must beat a spawn) - the absolute milliseconds belong to this machine.
    """
    scale = [(1920, 1080, 24.0, 4.0, "1080p"), (3840, 2160, 24.0, 2.0, "2160p")]
    step = 0.25      # a scrub step: 6 frames of walking at 24 fps
    lines = ["break-even of one spawn against a forward walk (rgb24 read)"]
    for width, height, fps, duration, label in scale:
        path = _make_clip(f"seek-scale-{label}.mp4", width, height, fps, duration,
                          preset="ultrafast", gop=int(fps))
        spawns_ms: list[float] = []
        for index in range(5):
            seconds = 0.2 + index * 0.31
            started = time.perf_counter()
            assert _per_request_frame(path, seconds, width, height) is not None
            spawns_ms.append((time.perf_counter() - started) * 1000.0)
        reader = _resident(path, start_time=0.2, width=width, height=height)
        walks_ms: list[float] = []
        try:
            for index in range(7):
                started = time.perf_counter()
                assert reader.frame_at(0.2 + index * step) is not None
                walks_ms.append((time.perf_counter() - started) * 1000.0)
        finally:
            reader.close()
        spawn = statistics.median(spawns_ms)
        walk = statistics.median(walks_ms[1:])
        per_frame = walk / (step * fps)
        break_even = spawn / per_frame
        lines.append(
            f"{label} {width}x{height}@{fps:g}: spawn {spawn:.1f} ms | walk per seek {walk:.1f} ms "
            f"({per_frame:.2f} ms/frame) | break-even {break_even:.0f} frames "
            f"({break_even / fps:.2f} s)")
        assert walk < spawn, f"{label}: a warm walk ({walk:.1f} ms) must beat a spawn ({spawn:.1f} ms)"
        assert per_frame > 0
    evidence = "\n".join(lines)
    with open(os.path.join(SCRATCH, "seek-reader-scale.txt"), "w", encoding="utf-8") as handle:
        handle.write(evidence + "\n")
    print("\n" + evidence)


def test_a_probe_started_early_is_what_the_first_press_uses(monkeypatch):
    """The grid probe is a whole extra ffmpeg decoding 13 frames; it must not run inside the first press.

    Measured 2026-11-25: the probe costs ~133 ms at 1440x1080 and ~273 ms at 2160p, and it used to run inline in
    whichever request arrived first - the first seek of every render cost hundreds of milliseconds more than the rest.
    `GridProbe` starts it where the render already spends seconds (engine load) and `_ensure_grid` takes the answer.
    """
    from src.utils import SeekReader as SR

    calls: list[tuple] = []

    def fake_probe(input_file, ffmpeg_path="ffmpeg", frames=SR._PROBE_FRAMES, near=0.0):
        calls.append((input_file, near))
        return (0.5, 0.041666)

    monkeypatch.setattr(SR, "probe_frame_grid", fake_probe)
    probe = SR.GridProbe("clip.mkv", "ffmpeg", near=0.0)
    assert probe.value(timeout=5.0) == (0.5, 0.041666)
    assert len(calls) == 1, "one probe, once"

    # A reader handed that handle must NOT probe again, and must see the same grid.
    inline: list[int] = []
    monkeypatch.setattr(SR, "probe_frame_grid", lambda *a, **k: inline.append(1))
    reader = SR.SeekReader("clip.mkv", width=4, height=4, grid_probe=probe)
    assert reader._ensure_grid() == (0.5, 0.041666)
    assert inline == [], "the inline probe must not run when a probe already answered"


def test_a_probe_that_never_answers_falls_back_to_one_spawn_per_seek(monkeypatch):
    """A dead probe is not a hang: `value()` times out to None and the reader keeps today's behaviour."""
    from src.utils import SeekReader as SR

    class Dead:
        def value(self, timeout=None):
            return None

    monkeypatch.setattr(SR, "probe_frame_grid", lambda *a, **k: (_ for _ in ()).throw(AssertionError("inline ran")))
    reader = SR.SeekReader("clip.mkv", width=4, height=4, grid_probe=Dead())
    assert reader._ensure_grid() is None, "no grid: the seek re-opens the source, which is the documented fallback"


def test_the_grid_is_fitted_not_a_median_and_the_frame_names_itself():
    """The container quantises PTS to milliseconds; a median of adjacent deltas inherits that error, a fit does not.

    Measured 2026-11-25 on the owner's clip: 13 probed frames of a 23.976 source read back as a 42.000 ms pitch against
    a true 41.708. Over a 20 s clip that is ~4 frames of drift, and it is why the pane could not name the frame it
    decoded - the request was published as the picture's time instead (`delta_s` structurally zero).
    """
    from src.utils.SeekReader import SeekReader, fit_grid

    truth = 0.041708                    # 23.976 fps
    quantised = [round(offset * truth * 1000.0) / 1000.0 for offset in range(13)]
    median_delta = round(truth * 1000.0) / 1000.0          # what the old rule produced
    first, pitch = fit_grid(quantised)

    assert abs(pitch - truth) < 0.0002, f"fit pitch {pitch:.6f} vs true {truth}"
    assert abs(pitch - truth) < abs(median_delta - truth), "the fit must beat the median it replaced"
    for offset in range(13):
        assert abs(quantised[offset] - (first + offset * pitch)) <= 0.0005

    # A reader with that grid names the frame it just read: `_read` advances the index, so the returned frame is the
    # one behind it.
    class Chunk:
        def read_frame(self):
            return b"frame"

    reader = SeekReader("clip.mkv", width=4, height=4)
    reader._grid = (first, pitch)
    reader._reader = Chunk()
    reader._next_index = 10
    reader._read()
    assert reader.last_pts == pytest.approx(first + 10 * pitch, abs=1e-6)
