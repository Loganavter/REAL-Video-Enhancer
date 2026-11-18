"""The int8 probe's two jobs are separable, and the shipped path takes only one of them.

Why this is pinned: the probe used to certify the picture before the first frame, and that verdict both
moved with the configuration it certified (the pair set is not a function of the clip alone) and was
overwritten by every forced `int8Fit` - which is what the Fast preset sets. So the certification is now
opt-in, the calibration stays mandatory, and a log line must not claim the fp16 engine runs while the
render is running the body.

These tests hold the seam, not the arithmetic: the numbers that matter are measured on renders
(`docs/wsp/verified/`), because that is what a viewer sees.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_calibration_only_result_quotes_no_percentage():
    probe_mod = pytest.importorskip("src.utils.Int8Probe")
    result = probe_mod.Int8ProbeResult(
        True, float("nan"), float("nan"), float("nan"), 0, 0.1, "calibration only",
        frames_used=24, certified=False)
    text = result.stats()
    assert "calibration only" in text
    assert "24 frames" in text
    # a withheld number must not reappear as "nan%" - that is how a withdrawn figure gets quoted again
    assert "nan" not in text.lower()
    assert "composition" not in text


def test_certifying_result_still_prints_the_verdict():
    probe_mod = pytest.importorskip("src.utils.Int8Probe")
    result = probe_mod.Int8ProbeResult(False, 0.082, 0.266, 25, 7, 0.1, "over the bar")
    text = result.stats()
    assert "composition" in text and "fp16" in text


def test_certification_is_opt_in_in_the_probe_signature():
    probe_mod = pytest.importorskip("src.utils.Int8Probe")
    assert inspect.signature(probe_mod.probe).parameters["certify"].default is True


def test_the_body_serves_only_the_whole_frame_path():
    """With a gate created the render is tiled and the body is not wired in; that is the only case in
    which the 'fp16 engine runs' message is true."""
    rv = pytest.importorskip("src.RenderVideo")
    assert rv.Render._int8_body_serves_this_render(SimpleNamespace(temporalReuse=None)) is True
    assert rv.Render._int8_body_serves_this_render(SimpleNamespace(temporalReuse=object())) is False
