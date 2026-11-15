"""Tests for AutoGateTuner: the threshold is chosen from measured (LR, HR) pairs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.TemporalReuse import AutoGateTuner  # noqa: E402


def _feed(tuner, pairs):
    for lr, hr in pairs:
        tuner.add(lr, hr)


def test_busy_content_keeps_the_seed():
    """If nothing qualifies under the error budget, do not drag the threshold to the minimum."""
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=1.0)
    _feed(t, [(60.0 + i, 200.0) for i in range(10)])
    assert t.ready
    assert t.chosen == 5.0


def test_clean_content_takes_the_widest_threshold_that_fits():
    """If even the widest candidate keeps the HR error inside the budget, reuse as much as
    possible: the threshold is a cap on error, not a fixed quality setting."""
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=1.0)
    _feed(t, [(0.2 + 0.01 * i, 0.3) for i in range(10)])
    assert t.ready
    assert t.chosen == max(t.candidates)


def test_threshold_grows_with_grain():
    """A noisy source needs a wider threshold to reach the same error budget."""
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=1.0)
    # deltas are 1..10 luma levels, HR amplification ~0.2 levels per LR level
    _feed(t, [(float(v), 0.2 * v) for v in range(1, 11)])
    assert t.ready
    assert t.chosen >= 4.0


def test_no_auto_veto():
    """The full-res veto is never enabled automatically (it kills grainy film)."""
    t = AutoGateTuner(seed=5.0, min_pairs=4)
    _feed(t, [(0.1, 0.2)] * 6)
    assert t.ready
    assert t.veto is None


def test_waits_for_enough_pairs():
    t = AutoGateTuner(seed=5.0, min_pairs=8)
    _feed(t, [(1.0, 0.5)] * 7)
    assert not t.ready
    _feed(t, [(1.0, 0.5)])
    assert t.ready


def test_stats_are_readable():
    t = AutoGateTuner(seed=5.0, min_pairs=4)
    _feed(t, [(1.0, 0.2)] * 4)
    text = t.stats()
    assert "threshold" in text and "veto" in text


def test_clamped_to_the_configured_range():
    t = AutoGateTuner(seed=5.0, min_pairs=4, budget=100.0, clamp=(2.0, 6.0))
    _feed(t, [(float(v), 0.0) for v in range(1, 20)])
    assert t.ready
    assert t.chosen <= 6.0
