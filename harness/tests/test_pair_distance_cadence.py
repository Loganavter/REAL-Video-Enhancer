"""`pair_off` must be in the pane's frames, not in a hard-coded 24.

Telemetry audit (2026-11-27, 5.1): `delta_frames = (source - result) * 24.0` in the pane's `render()` ignored the
clip's own rate. On a 60 fps clip a two-frame desync read 0.8 (and 0 once rounded), and `note_pair` counted only paints
that HAD both halves - so the number quoted in every live proof could not see the failure it exists for.

The conversion now lives in one function, and the pane's cadence (`_result_frame_seconds`, learned from the result
stream) is the unit. This file pins it at three rates and pins that the lone-half case is counted, not ignored.
"""

from __future__ import annotations

import pytest

from src.ui.video import telemetry
from src.ui.widgets.video_preview import pair_distance_frames


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_two_frames_apart_reads_as_two_frames_at_any_rate(fps):
    period = 1.0 / fps
    assert pair_distance_frames(2 * period, 0.0, period) == pytest.approx(2.0)
    assert pair_distance_frames(0.0, 4 * period, period) == pytest.approx(-4.0)


def test_the_same_picture_is_zero_at_any_rate():
    for fps in (24.0, 60.0):
        assert pair_distance_frames(7.25, 7.25, 1.0 / fps) == 0.0


def test_a_zero_period_cannot_divide_by_zero():
    assert pair_distance_frames(1.0, 0.0, 0.0) == pytest.approx(1000.0)


def test_a_two_frame_gap_at_60fps_is_off_by_the_counters_own_bar():
    """0.8 under the old 24 fps constant; 2.0 now, which crosses the gate's 1.5-frame `pair_off` bar."""
    telemetry._reset(session=True)
    telemetry.note_pair(delta_frames=pair_distance_frames(2 / 60.0, 0.0, 1 / 60.0))
    assert telemetry.snapshot()["pair_off"] == 1
