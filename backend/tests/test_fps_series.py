"""The run's speed series: bounded, monotonic, and a WINDOW rate that can show a slowdown.

Measured need (owner, 2026-11-28): the Properties window printed the per-phase AVERAGES
(`phases_ms_per_frame`), so a stall, a ramp or a section where the pipeline fell apart averaged into one plausible
number. The report now carries `fps_series` - sparse `(frame, seconds)` samples from the render loop, turned into
`frames`/`seconds`/`fps` lists here - and the window paints it as a curve.

What this file pins, on a synthetic profile (no torch, no render, no clock):

* a long run produces at most ``MAX_POINTS`` points, however many frames it rendered;
* frames and seconds are monotonic and the three lists are the same length;
* every point after the first is a WINDOW rate (``delta_frames / delta_seconds``), so the second half of a run that
  slowed down reads slow instead of being averaged away by the first half;
* an empty run yields NO series (``None``), not a division by zero and not an empty curve to chart.
"""

from __future__ import annotations

import pytest

from backend.src.utils.FpsSeries import MAX_POINTS, fps_series, sample_stride


def _profile(frames: int, *, seconds_per_frame: float = 1.0 / 30.0, slow_from: float | None = None,
             slow_seconds_per_frame: float = 1.0 / 3.0) -> dict:
    """A synthetic profile shaped exactly like the render loop's: samples every `sample_stride(frames)` frames.

    `slow_from` is the share of the run after which the pipeline falls to `slow_seconds_per_frame` per frame - the
    shape the averages cannot show and the curve must.
    """
    stride = sample_stride(frames)
    samples = []
    elapsed = 0.0
    for frame in range(stride, frames + 1, stride):
        elapsed += stride * seconds_per_frame
        if slow_from is not None and frame >= frames * slow_from:
            elapsed += stride * (slow_seconds_per_frame - seconds_per_frame)
        samples.append((frame, elapsed))
    samples.append((frames, elapsed))   # the loop's own final point
    return {"frames": frames, "fps_samples": samples}


def test_a_long_run_is_bounded_and_monotonic():
    frames = 24_000
    series = fps_series(_profile(frames))

    assert series is not None
    assert len(series["frames"]) <= MAX_POINTS, (
        f"{frames} frames produced {len(series['frames'])} points: the chart is bounded by construction, not by luck"
    )
    assert len(series["frames"]) == len(series["seconds"]) == len(series["fps"])
    assert series["frames"][0] == 0 and series["seconds"][0] == pytest.approx(0.0) and series["fps"][0] == pytest.approx(0.0)
    assert all(b > a for a, b in zip(series["frames"], series["frames"][1:])), "frame numbers must never go backwards"
    assert all(b >= a for a, b in zip(series["seconds"], series["seconds"][1:])), "the clock must never go backwards"
    assert series["frames"][-1] == frames, "the curve must end where the run ended, not on the last stride boundary"


def test_every_point_is_the_rate_over_its_own_window():
    series = fps_series(_profile(6_000, seconds_per_frame=1.0 / 30.0))
    frames = series["frames"]
    seconds = series["seconds"]
    rates = series["fps"]

    assert len(frames) > 3
    for index in range(1, len(frames)):
        delta_frames = frames[index] - frames[index - 1]
        delta_seconds = seconds[index] - seconds[index - 1]
        expected = round(delta_frames / delta_seconds, 3) if delta_seconds > 0 else 0.0
        assert rates[index] == expected, (
            f"point {index} says {rates[index]} fps but its own window is {delta_frames}/{delta_seconds}"
        )
    assert rates[1] == pytest.approx(30.0, abs=0.5), "a steady 30 fps run must read 30 in its windows"


def test_a_slow_second_half_reads_slow_in_the_windows_while_the_average_hides_it():
    frames = 12_000
    series = fps_series(_profile(frames, slow_from=0.5))

    rates = series["fps"]
    middle = len(rates) // 2
    early = [rate for rate in rates[1:middle] if rate > 0]
    late = [rate for rate in rates[middle:] if rate > 0]
    assert early and late
    assert sum(early) / len(early) == pytest.approx(30.0, abs=1.0)
    assert sum(late) / len(late) == pytest.approx(3.0, abs=1.0), (
        "the second half ran at 3 fps; a global average would read ~16 and the curve would lie"
    )


def test_an_empty_run_has_no_series():
    assert fps_series({}) is None
    assert fps_series({"frames": 0}) is None
    assert fps_series({"frames": 0, "fps_samples": []}) is None
    # Only the synthetic origin: one point is not a curve, and a window needs a span.
    assert fps_series({"frames": 1, "fps_samples": [(0, 0.0)]}) is None


def test_the_stride_divides_a_whole_run_into_about_two_hundred_samples():
    assert sample_stride(24_000) == 100
    assert sample_stride(239) == 1, "a short run samples every frame and is under the bound anyway"
    assert sample_stride(0) == 1, "a missing estimate must not ask for a zero stride"
    assert sample_stride(None) == 1


# --- the report's own wiring -------------------------------------------------------------------------
# `RenderVideo` imports torch at module scope and this interpreter has none; the code under test never touches it, so
# a bare module stub is enough to import the real class (the same trick `harness/tests/test_render_status_truth.py`
# uses). The stubs are removed again so a later test that genuinely needs torch still skips instead of importing a fake.

_STUBBED = []
for _name in ("torch", "cv2"):
    try:
        __import__(_name)
    except ImportError:
        import sys
        import types

        sys.modules[_name] = types.ModuleType(_name)
        _STUBBED.append(_name)

_RENDER_VIDEO = pytest.importorskip("backend.src.RenderVideo")

if _STUBBED:
    import sys

    for _name in _STUBBED:
        sys.modules.pop(_name, None)
    sys.modules.pop("backend.src.RenderVideo", None)


def _report(profile: dict) -> dict:
    """`RenderVideo._run_report` on a bare instance: only the profile and the settings it reads are set."""
    render = object.__new__(_RENDER_VIDEO.Render)
    render.upscaleModel = "4x-UltraSharp.pth"
    render.precision = "fp16"
    render.backend = "tensorrt"
    render.temporalReuse = None
    render.temporalReuseCorrection = False
    render.autoTuner = None
    render.qualityPreset = "best"
    render.int8FitPreset = "int8"
    render._profile = profile
    return render._run_report()


def test_the_run_report_stores_the_curve_and_leaves_it_out_when_the_run_stored_nothing():
    profile = _profile(24_000)
    report = _report(profile)
    assert report["fps_series"] == fps_series(profile), (
        "the report must carry exactly the bounded window series the module computes"
    )
    assert "fps_series" not in _report({"frames": 0}), (
        "an empty run omits the key so the window says why instead of charting an empty curve"
    )
