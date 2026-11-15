"""Tests for AutoGateTuner: the threshold is chosen from measured (LR, HR) pairs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import TemporalReuse
from src.utils.TemporalReuse import AutoGateTuner  # noqa: E402


def _feed(tuner, pairs):
    for lr, hr in pairs:
        tuner.add(lr, hr)


def test_busy_content_falls_back_to_the_strictest_candidate():
    """If nothing qualifies under the error budget, reuse as little as the range allows.

    Keeping the configured seed here was the old behaviour, and it is what let a grainy film
    run at 5.0/255: the tuner had just proven that no threshold in its range holds the error
    inside the budget, and then used one anyway. That produced frames at 34.7 dB against the
    ungated render (measured on the film clip, chain 4).
    """
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=1.0)
    _feed(t, [(1.0, 200.0) for _ in range(10)])       # calm frames, absurd HR error
    assert t.ready
    assert t.chosen == min(t.candidates)


def test_chain_depth_is_what_the_tuner_has_to_score():
    """A single-step delta understates a reuse: the gate can serve a 4-frame-old reference.

    This is the rule the film clip needed - at threshold 5.0 with chain 4 its worst frames
    were 34.7 dB against the ungated render while every one-step delta looked calm.
    """
    np = pytest.importorskip("numpy")
    calm = np.zeros((4, 4))
    drifted = np.full((4, 4), 3.0)

    # the gate's reference is the frame `depth` steps back, not "any of the last `depth`"
    assert TemporalReuse.worst_chain_delta(calm, [calm, drifted], 1) == pytest.approx(3.0)
    # a shallow chain never sees the old drift at all
    assert TemporalReuse.worst_chain_delta(calm, [calm, calm], 1) == pytest.approx(0.0)
    assert TemporalReuse.worst_chain_delta(calm, [calm, calm, drifted], 3) == pytest.approx(3.0)
    assert TemporalReuse.worst_chain_delta(calm, [], 4) == float("inf")


def test_clean_content_takes_the_widest_threshold_that_fits():
    """If even the widest candidate keeps the HR error inside the budget, reuse as much as
    possible: the threshold is a cap on error, not a fixed quality setting."""
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=1.0)
    _feed(t, [(0.2 + 0.01 * i, 0.3) for i in range(10)])
    assert t.ready
    # the widest candidate the clamp allows: 12 and 4 measured the same only while the tile
    # artifact dominated; with the tile path off it is 0.517% against 0.045%
    assert t.chosen == t.clamp[1] == 4.0


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


def test_best_preset_refuses_to_reuse_when_nothing_is_certified():
    """`fallback="off"` is the quality-first answer: no reuse beats the least-bad reuse."""
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=1.0, fallback="off")
    _feed(t, [(60.0 + i, 200.0) for i in range(10)])
    assert t.ready                 # calibration is over
    assert t.chosen is None        # ...and the verdict is "do not reuse"
    assert "reuse is OFF" in t.stats()


def test_fast_preset_still_reuses_the_strictest_threshold():
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=1.0, fallback="strictest")
    _feed(t, [(1.0, 200.0) for _ in range(10)])
    assert t.ready
    assert t.chosen == min(t.candidates)


def test_an_unknown_fallback_is_refused():
    with pytest.raises(ValueError):
        AutoGateTuner(fallback="maybe")


def test_chain_depth_is_certified_separately_from_the_threshold():
    """A clip whose 4-frame drift is too large can still be certified two frames deep.

    This is the anime case: the one-step error is ~1.6 levels and the four-step error ~9.3,
    so the useful answer is "reuse, but never from a reference more than N frames old",
    not "throw the gate away" and not "reuse four frames deep and live with it".
    """
    t = AutoGateTuner(min_pairs=6, budget=2.0, chain_budget=12.0)
    for index in range(6):
        # depth 1 and 2 calm, depth 3 and 4 drift past even the chain budget
        t.add(1.0, (1.5, 1.8, 16.0, 20.0))
    assert t.ready
    assert t.chosen_chain == 2
    assert t.chosen == t.clamp[1]


def test_the_clamp_still_bounds_the_choice():
    """The clamp is a bound, not a fix: thresholds 12 and 4 measured the same on the clip
    (0.288% against 0.287% of pixels above 8 levels), so it is here to keep the number inside a
    physically meaningful range, and the test only pins that it does."""
    t = AutoGateTuner(min_pairs=4, budget=50.0, clamp=(2.0, 4.0))
    for _ in range(4):
        t.add(1.0, 0.5)
    assert t.ready
    assert t.chosen == 4.0


def test_a_scalar_delta_is_treated_as_depth_one():
    t = AutoGateTuner(min_pairs=4, budget=2.0)
    for _ in range(4):
        t.add(1.0, 1.2)
    assert t.ready
    assert t.chosen_chain == 1
    assert t.chosen is not None


def test_no_depth_certifies_means_no_reuse_under_best():
    t = AutoGateTuner(min_pairs=4, budget=2.0, chain_budget=12.0, fallback="off")
    for _ in range(4):
        t.add(1.0, (20.0, 20.0, 20.0, 20.0))     # over the budget at every depth
    assert t.ready
    assert t.chosen is None


def test_a_grainy_source_refuses_at_any_threshold():
    """Above the measured noise floor, reuse is refused rather than tuned.

    Three real Blu-rays measured at the current settings: noise floor 0.95 -> 0.005% of pixels
    above 8 levels, 1.32 -> 0.097%, 2.27 -> 0.470% (i.e. five times the acceptance bar, and the
    same at threshold 4 and at threshold 2). So the floor is the signal, not the threshold.
    """
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=2.0, fallback="off",
                      noise_floor_refuse=2.0)
    _feed(t, [(3.0 + 0.1 * i, 0.5) for i in range(10)])   # calm frames sit above the floor
    assert t.ready
    assert t.chosen is None
    assert "above the" in t.stats()


def test_a_clean_source_is_not_refused_by_the_floor():
    t = AutoGateTuner(seed=5.0, min_pairs=8, budget=2.0, noise_floor_refuse=2.0)
    _feed(t, [(0.2 + 0.01 * i, 0.4) for i in range(10)])
    assert t.ready
    assert t.chosen is not None
