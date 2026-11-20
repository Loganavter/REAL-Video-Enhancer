"""Tests for ``backend/src/utils/TemporalReuse.py`` - the reference-frame gate.

Three layers, because the cell that has to verify this ships without torch (the bundled
interpreter lives outside the worker jail):

* **decision** - no torch, no numpy: ``scaled_block``, ``block_grid``, ``luma_weights`` and
  ``decide_reference`` (chain limit semantics, scene cut, "no reference yet");
* **numpy** - the real state machine (``ReferenceFrameGate.observe``: last-kept-frame
  reference, chain counter, scene cut, pixel veto, counters) driven with numpy block grids
  and luma computed the way the torch path computes them (integral-image block means with
  ``ceil`` semantics, matching ``F.avg_pool2d(..., ceil_mode=True)``). This is where the
  reuse rates, the "max over block means" property and the threshold calibration are
  measured;
* **torch** - the real class end to end (``luma``/``blocks``/``is_duplicate``): the pooling
  is cross-checked against the numpy reference implementation above, fp16 input, device
  locality and the 1080p cost. Skipped when torch is not installed
  (``pytest.importorskip``).

Levels are 8-bit luma levels (0..255) throughout, the unit the thresholds use; frames are
``(H, W, 3)`` in [0, 1] like the renderer hands them over.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.utils import TemporalReuse as TR  # noqa: E402

BLOCK = 32                      # the block side the statistic is defined at 1080p
LEVEL = 1.0 / 255.0             # one 8-bit luma level, in [0, 1] frame units


# --------------------------------------------------------------------------------------
# numpy reference implementation of the statistic (test-side framing)
# --------------------------------------------------------------------------------------
def np_luma(frame):
    """3-tap weighted luma in [0, 1] - mirrors ``ReferenceFrameGate.luma``."""
    np = pytest.importorskip("numpy")
    weights = np.array(TR.luma_weights((1, 2, 1)), dtype=np.float64)
    return np.asarray(frame, dtype=np.float64) @ weights


def np_blocks(frame, block: int = BLOCK):
    """``(H, W, 3)`` in [0, 1] -> ``(1, 1, ceil(H/b), ceil(W/b))`` in 8-bit levels.

    Integral-image block means with ``ceil`` clipping: the tail block averages only the
    pixels that exist, exactly like ``F.avg_pool2d(x, block, ceil_mode=True)``.
    """
    np = pytest.importorskip("numpy")
    luma = np_luma(frame)
    height, width = luma.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = luma.cumsum(axis=0).cumsum(axis=1)
    y0 = np.arange(0, height, block)
    y1 = np.minimum(y0 + block, height)
    x0 = np.arange(0, width, block)
    x1 = np.minimum(x0 + block, width)
    total = (integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)]
             - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)])
    count = np.outer(y1 - y0, x1 - x0)
    return (total / count * 255.0)[None, None]


def np_luma_batch(frame):
    """``(1, 1, H, W)`` luma in [0, 1] - the array ``observe(luma=...)`` expects."""
    np = pytest.importorskip("numpy")
    return np_luma(frame)[None, None]


def feed(gate, frame, block: int = BLOCK):
    """One ``is_duplicate`` step, numpy-side: the same two inputs the class feeds itself."""
    return gate.observe(np_blocks(frame, block), np_luma_batch(frame))


def np_frame(value: float, height: int = 128, width: int = 128):
    """Flat cel at ``value`` (0..1) as an ``(H, W, 3)`` frame."""
    np = pytest.importorskip("numpy")
    return np.full((height, width, 3), float(value), dtype=np.float64)


def np_cel(seed: int, height: int = 128, width: int = 128, grain: float = 0.0):
    """A deterministic "drawn cel": low-frequency content plus optional per-pixel grain."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
    base = 0.25 + 0.5 * (0.6 * ys + 0.4 * xs) + 0.05 * np.sin(6.0 * xs + seed)
    frame = np.stack([base, base, base], axis=-1)
    if grain:
        frame = frame + grain * LEVEL * rng.integers(-1, 2, size=frame.shape)
    return np.clip(frame, 0.0, 1.0)


def strict_gate(**kwargs):
    """A gate pinned to the 32 px block the statistic is defined at (small test frames)."""
    kwargs.setdefault("block", BLOCK)
    kwargs.setdefault("pixel_veto", None)
    return TR.ReferenceFrameGate(**kwargs)


# --------------------------------------------------------------------------------------
# decision layer (pure python)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("height,block", [
    (1080, 32),      # the calibration reference
    (2160, 64),      # 4K: the block doubles
    (720, 21),
    (480, 14),
    (8, 2),          # floor: never smaller than 2 px
    (1, 2),
])
def test_scaled_block_tracks_the_frame_height(height, block):
    assert TR.scaled_block(height) == block


@pytest.mark.parametrize("height,width,block,expected", [
    (1080, 1440, 32, (34, 45)),   # the production case: 1080 = 33*32 + 24, 1440 = 45*32
    (64, 96, 32, (2, 3)),
    (32, 32, 32, (1, 1)),
    (37, 53, 32, (2, 2)),         # both axes partial, smaller than 2 blocks
    (1, 1, 32, (1, 1)),
])
def test_block_grid_rounds_up(height, width, block, expected):
    assert TR.block_grid(height, width, block) == expected


def test_block_grid_rejects_degenerate_input():
    with pytest.raises(ValueError):
        TR.block_grid(0, 10, 32)
    with pytest.raises(ValueError):
        TR.block_grid(10, 10, 0)


def test_luma_weights_are_normalized():
    assert TR.luma_weights((1, 2, 1)) == pytest.approx((0.25, 0.5, 0.25))
    assert TR.luma_weights((0.299, 0.587, 0.114)) == pytest.approx(
        (0.299, 0.587, 0.114), abs=1e-12)
    assert sum(TR.luma_weights((3, 5, 2))) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        TR.luma_weights((0, 0, 0))


@pytest.mark.parametrize("delta,chain,max_chain,cut,expected", [
    (0.0, 0, 0, False, True),      # identical frame, unbounded chain
    (4.9, 500, 0, False, True),    # max_chain=0 never caps the chain
    (4.9, 0, 4, False, True),
    (4.9, 3, 4, False, True),
    (4.9, 4, 4, False, False),     # chain reached the cap -> must render
    (5.1, 0, 0, False, False),     # moved more than the threshold
    (0.0, 0, 0, True, False),      # scene cut always forces a render
    (float("inf"), 0, 0, False, False),   # no reference yet
])
def test_decide_reference(delta, chain, max_chain, cut, expected):
    assert TR.decide_reference(delta, chain, 5.0, max_chain, cut) is expected


# --------------------------------------------------------------------------------------
# numpy layer: the state machine
# --------------------------------------------------------------------------------------
def test_defaults_match_the_anime_scripter_calibration():
    gate = TR.ReferenceFrameGate()
    assert gate.block is None                       # None = scaled per frame
    assert gate.block_for(1080) == 32
    assert gate.block_for(2160) == 64
    assert (gate.threshold, gate.max_chain, gate.pixel_veto) == (5.0, 4, 40.0)
    assert gate.luma_taps == (1.0, 2.0, 1.0)
    assert gate.scene_threshold == 32.0


def test_constructor_validates():
    with pytest.raises(ValueError):
        TR.ReferenceFrameGate(block=0)
    with pytest.raises(ValueError):
        TR.ReferenceFrameGate(max_chain=-1)
    with pytest.raises(ValueError):
        TR.ReferenceFrameGate(pixel_veto=-1.0)
    TR.ReferenceFrameGate(pixel_veto=None)          # veto off is legal


def test_first_frame_is_never_a_duplicate():
    gate = strict_gate()
    assert gate.can_reuse is False
    assert feed(gate, np_frame(0.5)) is False
    assert gate.can_reuse is True
    assert (gate.processed, gate.reused, gate.chain) == (1, 0, 0)


def test_held_cel_reuses_against_the_reference_unbounded():
    """30 frames: one cel change, then 29 identical ones -> 29 reuses with max_chain=0."""
    gate = strict_gate(max_chain=0)
    cel = np_cel(seed=1)
    decisions = [feed(gate, cel) for _ in range(30)]

    assert decisions[0] is False
    assert all(decisions[1:])
    assert gate.reused == 29
    assert gate.processed == 30
    assert gate.reused / gate.processed == pytest.approx(29 / 30)


def test_held_cel_chain_cap_limits_each_run_to_two():
    """Same 30 frames with max_chain=2: at most 2 consecutive reuses, 20 reuses in total."""
    gate = strict_gate(max_chain=2)
    cel = np_cel(seed=2)
    decisions = [feed(gate, cel) for _ in range(30)]

    assert decisions[0] is False
    assert gate.reused == 20
    longest = current = 0
    for flag in decisions:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    assert longest == 2


def test_reference_is_the_last_kept_frame_not_the_previous_one():
    """A 0.1-level-per-frame drift stays reusable against the kept frame for 3 frames.

    Compared with the *previous* frame every step would be 0.1 levels and stay a duplicate
    forever; against the kept frame the delta grows until the threshold is reached and the
    reference is refreshed. That is the property which lets a held cel be reused
    indefinitely while a fade is still caught.
    """
    gate = strict_gate(max_chain=0, threshold=0.35)
    deltas = []
    decisions = []
    reference = None                            # the test tracks what was kept, too
    for index in range(10):
        frame = np_frame(0.4 + index * 0.1 * LEVEL)
        blocks = np_blocks(frame)
        deltas.append(float("inf") if reference is None
                      else float(abs(blocks - reference).max()))
        reuse = feed(gate, frame)
        decisions.append(reuse)
        if not reuse:
            reference = blocks

    assert decisions == [False, True, True, True, False, True, True, True, False, True]
    assert deltas[0] == float("inf")            # nothing to compare against yet
    assert deltas[1:] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1],
                                       abs=0.002)
    assert gate.reused == 7


def test_one_level_fade_is_capped_by_the_chain_at_the_default_threshold():
    """1 level/frame at the steered defaults (threshold 5, max_chain 4).

    The two limits answer different questions: the block statistic alone lets the fade drift
    until it exceeds 5.0 levels, the chain cap stops it at 4. With ``max_chain=0`` the fade is
    re-rendered only once it has drifted past the threshold - which is why the production
    default carries a chain cap. (The exact boundary is float-sensitive: a level is 1/255 of
    a frame unit, so a 5-level delta lands a few ULP either side of 5.0.)
    """
    levels = [100.0 + index for index in range(30)]
    frames = [np_frame(level * LEVEL) for level in levels]

    def run(gate):
        kept_level = 0.0
        worst_drift = 0.0
        decisions = []
        for level, frame in zip(levels, frames):
            reuse = feed(gate, frame)
            decisions.append(reuse)
            if reuse:
                worst_drift = max(worst_drift, level - kept_level)
            else:
                kept_level = level
        return worst_drift, decisions

    capped_drift, decisions = run(strict_gate(max_chain=4))      # threshold 5.0, the default
    assert decisions.count(True) == 24                           # 29 candidates, 4 of every 5
    assert capped_drift <= 4.0                                   # bounded by max_chain
    longest = current = 0
    for flag in decisions:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    assert longest == 4                                          # the chain, not the threshold

    unbounded_drift, unbounded = run(strict_gate(max_chain=0))
    print(f"\n1 level/frame fade: chain-capped drift {capped_drift:.0f} levels, "
          f"threshold-only drift {unbounded_drift:.0f} levels")
    assert unbounded.count(True) == 24                           # the threshold, not the chain
    assert unbounded_drift >= 4.0                                # reaches the chain's bound...
    assert unbounded_drift <= 6.0                                # ...and is stopped by 5.0


def test_block_statistic_catches_motion_a_global_mean_hides():
    """One changed 32 px block in a 512x512 frame: 40 levels locally, ~0.16 globally."""
    gate = strict_gate()
    base = np_frame(0.5, height=512, width=512)
    assert feed(gate, base) is False

    changed = base.copy()
    changed[64:96, 160:192] += 40.0 * LEVEL     # exactly one 32x32 block
    assert np_blocks(changed).shape == np_blocks(base).shape

    global_before = float(np_luma(base).mean() * 255.0)
    global_after = float(np_luma(changed).mean() * 255.0)
    assert abs(global_after - global_before) < 0.35            # a global mean would skip it
    assert float(abs(np_blocks(changed) - np_blocks(base)).max()) == pytest.approx(
        40.0, abs=0.05)                                        # the block max does not

    assert feed(gate, changed) is False
    assert gate.reused == 0


def test_moved_rectangle_is_caught_by_the_block_statistic():
    """A 40x40 px rectangle moving across a 1080p-sized cel (the steered acceptance case)."""
    np = pytest.importorskip("numpy")
    gate = TR.ReferenceFrameGate()                     # defaults: 32 px blocks at 1080p
    base = np_cel(seed=21, height=1080, width=960)

    before = base.copy()
    before[400:440, 200:240] += 40.0 * LEVEL
    after = base.copy()
    after[400:440, 600:640] += 40.0 * LEVEL

    assert feed(gate, before, block=gate.block_for(1080)) is False
    delta = float(abs(np_blocks(after, block=32) - np_blocks(before, block=32)).max())
    assert delta > gate.threshold                      # caught by the block mean alone
    assert feed(gate, after, block=gate.block_for(1080)) is False
    assert gate.reused == 0
    assert gate.vetoes == 0                            # the block statistic did the work


def test_pixel_veto_rescues_a_mouth_flap():
    """A 6x6 px flapping patch: 2 levels for the block mean, 60 for the pixel veto.

    This is the case the block mean dilutes - a mouth flap, a blink, a small effect - and the
    reason TheAnimeScripter re-checks would-be drops with a full-resolution max.
    """
    np = pytest.importorskip("numpy")
    mouth_open = np_frame(0.5, height=256, width=256)
    mouth_closed = mouth_open.copy()
    mouth_closed[100:106, 100:106] += 60.0 * LEVEL     # 36 px inside one 32 px block

    delta = float(abs(np_blocks(mouth_closed) - np_blocks(mouth_open)).max())
    assert delta <= TR.REFERENCE_THRESHOLD             # the block mean alone would skip it
    assert delta == pytest.approx(60.0 * 36.0 / 1024.0, abs=0.05)   # 36 px in a 32x32 block

    vetoed = strict_gate(pixel_veto=40.0)
    assert feed(vetoed, mouth_open) is False
    assert feed(vetoed, mouth_closed) is False         # rescued by the veto
    assert vetoed.vetoes == 1
    assert vetoed.reused == 0

    blind = strict_gate(pixel_veto=None)               # veto off: the flap is swallowed
    assert feed(blind, mouth_open) is False
    assert feed(blind, mouth_closed) is True
    assert blind.vetoes == 0


def test_full_frame_change_is_caught():
    gate = strict_gate()
    assert feed(gate, np_frame(0.5)) is False
    assert feed(gate, np_frame(0.5 + 6 * LEVEL)) is False       # +6 levels
    assert gate.reused == 0


def test_dither_and_grain_do_not_trip_the_statistic():
    """The statistic is the max over BLOCK means: +-2 levels of per-pixel grain averages out.

    This is the reason for pooling instead of taking a per-pixel max: encoded video never
    has bit-identical frames, and a per-pixel statistic would call every frame different.
    """
    gate = strict_gate(max_chain=0, threshold=0.35, pixel_veto=None)
    cel = np_cel(seed=3)
    noisy = np_cel(seed=3, grain=2.0)           # same drawing, different dither pattern

    assert float(abs(np_blocks(noisy) - np_blocks(cel)).max()) < 0.35
    assert feed(gate, cel) is False
    assert feed(gate, noisy) is True


def test_scene_cut_resets_even_when_the_block_delta_is_small():
    """A 40-level global jump: 40 < threshold 100, but the cut rule (32 levels) fires."""
    gate = strict_gate(threshold=100.0, max_chain=0)
    assert feed(gate, np_frame(20 * LEVEL)) is False
    assert feed(gate, np_frame(30 * LEVEL)) is True            # 10 levels: no cut
    assert gate.chain == 1
    assert gate.scene_cuts == 0

    assert feed(gate, np_frame(70 * LEVEL)) is False           # +40 -> cut
    assert gate.scene_cuts == 1
    assert gate.chain == 0
    assert gate.reused == 1


def test_reset_forgets_the_reference_and_keeps_counters():
    gate = strict_gate()
    cel = np_cel(seed=4)
    feed(gate, cel)
    assert feed(gate, cel) is True
    gate.reset()

    assert gate.can_reuse is False
    assert gate.chain == 0
    assert gate.resets == 1
    assert gate.reused == 1
    assert feed(gate, cel) is False


def test_resolution_change_is_treated_as_a_cut():
    gate = strict_gate()
    assert feed(gate, np_frame(0.5, height=64, width=64)) is False
    assert feed(gate, np_frame(0.5, height=96, width=96)) is False
    assert gate.scene_cuts == 1


@pytest.mark.parametrize("height,width", [(37, 53), (75, 101), (63, 64), (1080, 1440)])
def test_odd_sizes(height, width):
    """Odd frames work, and a change confined to the partial tail block is still caught."""
    gate = strict_gate()
    base = np_frame(0.5, height=height, width=width)
    assert feed(gate, base) is False
    assert feed(gate, base) is True

    blocks = np_blocks(base)
    assert blocks.shape == (1, 1, *TR.block_grid(height, width, BLOCK))

    changed = base.copy()
    changed[-BLOCK:, -BLOCK:, :] += 60.0 * LEVEL    # the last (partial) block, all of it
    assert feed(gate, changed) is False


def test_skip_rate_on_a_synthetic_anime_sequence():
    """3 cels, each held for 10 frames: one change followed by 9 holds.

    30 frames total, 3 renders, 27 reuses => 90% reuse with max_chain=0. The default gate
    (max_chain=4) copies 4 of the 9 holds per cel (R R R R M R R R R M) => 8 per cel, 24
    overall (80%): the cap costs a fifth of the reuse but bounds both the freeze and any
    drift the reference can accumulate.
    """
    unbounded = strict_gate(max_chain=0)
    capped = strict_gate(max_chain=4)                    # the steered default
    reuses = capped_reuses = 0
    for cel_index in range(3):
        cel = np_cel(seed=100 + cel_index)
        for _hold in range(10):
            reuses += bool(feed(unbounded, cel))
            capped_reuses += bool(feed(capped, cel))

    assert reuses == 27
    assert reuses / 30 == pytest.approx(0.9)
    assert capped_reuses == 24
    assert capped_reuses / 30 == pytest.approx(0.8)


def test_threshold_calibration_keeps_more_of_a_noisy_cel_at_5_than_at_0_35():
    """Reuse rate on three synthetic sequences at the steered threshold vs the old one.

    Both gates carry the production pixel veto and ``max_chain=0``, so only the threshold
    differs. The numbers are printed for DONE.md: the supervisor compares them against the
    60-80% keepage measured on real anime.
    """
    def rate(threshold, frames):
        """Fresh gate per sequence: a reference left over from the previous one would hide
        exactly the difference being measured."""
        gate = strict_gate(threshold=threshold, max_chain=0, pixel_veto=40.0)
        return sum(bool(feed(gate, frame)) for frame in frames) / (len(frames) - 1)

    np = pytest.importorskip("numpy")
    base = np_cel(seed=31)
    rng = np.random.default_rng(99)
    noisy_cel = [np.clip(base + 5.0 * LEVEL * rng.integers(-1, 2, size=base.shape), 0.0, 1.0)
                 for _ in range(20)]                                  # one cel, grain +-5
    fade = [np_frame((100.0 + index) * LEVEL) for index in range(20)]

    # line-art-like texture, panned 2 px per frame: block means are nearly shift invariant,
    # the per-pixel max is not - this is what the pixel veto is for.
    ys = np.linspace(0.0, 1.0, 128)[:, None]
    xs = np.linspace(0.0, 1.0, 128)[None, :]
    ink = 0.35 + 0.4 * ((np.sin(38.0 * xs) * np.cos(34.0 * ys)) > 0.2)
    ink = np.stack([ink, ink, ink], axis=-1)
    pan = [np.roll(ink, shift=2 * index, axis=1) for index in range(20)]

    # the price of a looser threshold: a 5x5 px, 20-level patch (a mouth line) sitting in the
    # cel. It is a real change (0.49 block-mean levels, under the 40-level pixel veto), so at
    # 5.0 it is dropped - this is the shape of TheAnimeScripter's 2.0-2.2% dropped changes.
    flap = []
    for index in range(20):
        frame = base.copy()
        if index % 2:
            frame[40:45, 40:45] += 20.0 * LEVEL   # 25 px inside one 32 px block
        flap.append(frame)

    rates = {
        "held cel, grain +-5": (rate(0.35, noisy_cel), rate(5.0, noisy_cel)),
        "held cel + 5x5 20L flap": (rate(0.35, flap), rate(5.0, flap)),
        "1 level/frame fade": (rate(0.35, fade), rate(5.0, fade)),
        "2 px/frame pan": (rate(0.35, pan), rate(5.0, pan)),
    }
    print("\nreuse rate at threshold 0.35 vs 5.0 (max_chain=0, pixel veto 40):")
    for name, (strict, loose) in rates.items():
        print(f"  {name:24s} {strict * 100:5.1f}% -> {loose * 100:5.1f}%")

    for strict, loose in rates.values():
        assert loose >= strict                # never stricter at the looser threshold
    assert rates["held cel, grain +-5"][1] == pytest.approx(1.0)   # 5.0 keeps the whole cel
    assert rates["2 px/frame pan"][1] == pytest.approx(0.0)        # the veto keeps the pan out
    assert rates["held cel + 5x5 20L flap"][0] < 0.10             # 0.35 catches the flap
    assert rates["held cel + 5x5 20L flap"][1] == pytest.approx(1.0)   # 5.0 drops it


# --------------------------------------------------------------------------------------
# torch layer: the real class (skipped where torch is not installed)
# --------------------------------------------------------------------------------------
def _torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("height,width", [(64, 96), (75, 101), (32, 32), (1080, 1440)])
def test_torch_blocks_match_the_numpy_reference(height, width):
    torch = _torch()
    np = pytest.importorskip("numpy")
    frame = np_cel(seed=11, height=height, width=width)

    tensor = torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32))
    gate = strict_gate(block=BLOCK)
    blocks = gate.blocks(tensor)

    assert tuple(blocks.shape) == (1, 1, *TR.block_grid(height, width, BLOCK))
    assert blocks.device == tensor.device      # no hidden .cpu() in the pooling
    np.testing.assert_allclose(blocks.cpu().numpy(), np_blocks(frame), atol=1e-3)


def test_torch_luma_is_weighted_and_full_resolution():
    torch = _torch()
    np = pytest.importorskip("numpy")
    frame = np_cel(seed=14, height=48, width=64)
    frame[..., 0] = np.clip(frame[..., 0] + 0.10, 0.0, 1.0)   # channels differ, weights matter
    frame[..., 2] = np.clip(frame[..., 2] - 0.07, 0.0, 1.0)
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32))

    luma = TR.ReferenceFrameGate().luma(tensor)
    assert tuple(luma.shape) == (1, 1, 48, 64)
    assert luma.dtype == torch.float32
    np.testing.assert_allclose(luma.cpu().numpy()[0, 0], np_luma(frame), atol=1e-5)


def test_torch_held_cel_counts_match_the_numpy_layer():
    torch = _torch()
    np = pytest.importorskip("numpy")
    frame = np_cel(seed=12, height=128, width=128)
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32))

    gate = TR.ReferenceFrameGate(block=BLOCK, max_chain=0, pixel_veto=None)
    assert [gate.is_duplicate(tensor) for _ in range(30)].count(True) == 29

    capped = TR.ReferenceFrameGate(block=BLOCK, max_chain=2, pixel_veto=None)
    assert [capped.is_duplicate(tensor) for _ in range(30)].count(True) == 20


def test_torch_pixel_veto_rescues_a_blink_with_defaults():
    """End to end with the production defaults: 6x6 px patch, 60 levels, 4K-height blocks."""
    torch = _torch()
    np = pytest.importorskip("numpy")
    frame = np_frame(0.5, height=256, width=256)
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32))
    moved = tensor.clone()
    moved[:, :, 100:106, 100:106] += 60.0 / 255.0

    gate = TR.ReferenceFrameGate(block=BLOCK)          # threshold 5.0, veto 40.0
    assert gate.is_duplicate(tensor) is False
    assert gate.is_duplicate(tensor) is True           # identical frame
    assert gate.is_duplicate(moved) is False           # vetoed despite the small block mean
    assert gate.vetoes == 1


def test_torch_fp16_input_is_decided_like_fp32():
    torch = _torch()
    np = pytest.importorskip("numpy")
    frame = np_cel(seed=13, height=96, width=128)
    base = torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32))
    half = base.half()

    halves = strict_gate(block=BLOCK)
    singles = strict_gate(block=BLOCK)
    assert halves.blocks(half).dtype == torch.float32   # pooled in fp32, not fp16
    assert halves.is_duplicate(half) is False
    assert singles.is_duplicate(base) is False
    assert halves.is_duplicate(half) is True
    assert singles.is_duplicate(base) is True

    moved = base.clone()
    moved[:, :, 32:64, 32:64] += 40.0 / 255.0
    assert halves.is_duplicate(moved.half()) is False
    assert singles.is_duplicate(moved) is False


def test_torch_delta_matches_the_numpy_statistic():
    """``delta`` is the worst block mean of the frame against the kept reference."""
    torch = _torch()
    np = pytest.importorskip("numpy")
    base = np_frame(0.5, height=512, width=512)
    changed = base.copy()
    changed[64:96, 160:192] += 40.0 * LEVEL

    def to_tensor(frame):
        return torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32))

    gate = strict_gate(block=BLOCK)
    assert gate.delta(to_tensor(changed)) == float("inf")      # no reference yet
    gate.is_duplicate(to_tensor(base))
    assert gate.delta(to_tensor(changed)) == pytest.approx(40.0, abs=0.01)
    assert gate.delta(to_tensor(base)) == pytest.approx(0.0, abs=1e-6)

    resized = np_frame(0.5, height=256, width=256)
    assert gate.delta(to_tensor(resized)) == float("inf")      # shape changed


def test_torch_rejects_wrong_shape():
    torch = _torch()
    gate = TR.ReferenceFrameGate()
    with pytest.raises(ValueError):
        gate.blocks(torch.zeros(3, 64, 64))
    with pytest.raises(ValueError):
        gate.blocks(torch.zeros(2, 3, 64, 64))
    with pytest.raises(ValueError):
        gate.blocks(torch.zeros(1, 4, 64, 64))      # 4 channels, 3 luma taps


def test_torch_1080p_cost():
    """The budget is ~1 ms per frame at 1080p on the GPU that renders it.

    Measured here on the CPU of the verification cell (no CUDA in the jail) as an upper
    bound: the cheap path is one memory-bound pooling pass and the pixel veto is one more
    pass, so the same work on a GPU is roughly an order of magnitude cheaper. The numbers
    are printed for the record either way.
    """
    torch = _torch()
    frame = torch.rand(1, 3, 1080, 1440)

    def measure(gate, calls: int = 10) -> float:
        for _ in range(3):
            gate.is_duplicate(frame)                # warm up + allocate
        samples = []
        for _ in range(calls):
            start = time.perf_counter()
            gate.is_duplicate(frame)
            samples.append((time.perf_counter() - start) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    cheap = measure(TR.ReferenceFrameGate(pixel_veto=None))   # 32 px blocks only
    vetoed = measure(TR.ReferenceFrameGate())                 # + full-res max delta
    print(f"\n1080p reference gate (CPU): block-only {cheap:.3f} ms, "
          f"with pixel veto {vetoed:.3f} ms")
    assert cheap < 10.0
    assert vetoed < 10.0


def test_a_scene_cut_is_reported_so_callers_can_reset_their_histories():
    """The gate drops its reference at a cut; callers keeping their own history must know.

    The auto tuner keeps the last `max_chain` output grids to score depth-2/3/4 reuses. Without a
    reset at the cut it scores reusing across a shot change - something the gate never does - and
    one busy opening is enough to pin the chain to 1 for the whole title (measured on three real
    Blu-rays; Chainsaw Man reads 0.2 at depth 1 and 16-35 at depths 2-4 on calm frames).
    """
    torch = _torch()
    gate = TR.ReferenceFrameGate(threshold=4.0, max_chain=4)
    dark = torch.zeros(1, 3, 1080, 1920)
    bright = torch.ones(1, 3, 1080, 1920)
    gate.is_duplicate(dark)
    assert gate.last_scene_cut is False           # no reference yet, nothing to cut from
    gate.is_duplicate(dark)
    assert gate.last_scene_cut is False
    gate.is_duplicate(bright)                     # full-range jump: a cut
    assert gate.last_scene_cut is True
    gate.is_duplicate(bright)
    assert gate.last_scene_cut is False           # the new reference is now stable


# --------------------------------------------------------------------------------------
# torch layer: the coarse statistic (decide on the drawing, not on the scan's registration)
# --------------------------------------------------------------------------------------
def _line_art(torch, height=270, width=360, seed=0):
    """A deterministic line-art frame: strokes on a mid field, i.e. the content a worst-block rule must see."""
    generator = torch.Generator().manual_seed(seed)
    frame = torch.full((1, 3, height, width), 0.66)
    for _ in range(40):
        y = int(torch.randint(4, height - 6, (1,), generator=generator).item())
        x = int(torch.randint(4, width - 8, (1,), generator=generator).item())
        frame[:, :, y:y + 2, x:x + max(3, width // 40)] = 0.10
    return frame


def _shift_torch(torch, frame, dx, dy):
    """Content moved by exactly (dx, dy) pixels, replicate borders - what a scan's wobble does."""
    height, width = int(frame.shape[-2]), int(frame.shape[-1])
    ys = torch.arange(height, dtype=torch.float32) - float(dy)
    xs = torch.arange(width, dtype=torch.float32) - float(dx)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack(((grid_x / (width - 1)) * 2 - 1, (grid_y / (height - 1)) * 2 - 1), dim=-1)[None]
    return torch.nn.functional.grid_sample(frame, grid, mode="bilinear", padding_mode="border",
                                           align_corners=True)


def test_coarse_warp_moves_content_in_the_direction_it_is_given():
    """The sign of the geometry, pinned on a single bright dot.

    Both warps in this feature are one ``grid_sample`` call, and ``grid_sample`` reads the INPUT at the grid's
    coordinates - so an offset of +d moves the CONTENT by -d. The first version got this backwards in the
    statistic and in the emission, and the measurement that caught it was exactly this: a dot moved by +3 and
    then read where it is.
    """
    torch = _torch()
    frame = torch.zeros(1, 1, 32, 48)
    frame[0, 0, 10, 20] = 1.0
    gate = TR.ReferenceFrameGate(statistic="coarse", pixel_veto=None)
    for dx, dy in ((3.0, 5.0), (-4.0, 0.0), (0.0, -2.0)):
        moved = gate._shift_tensor(frame, dx, dy)
        y, x = divmod(int(moved.flatten().argmax()), 48)
        assert (x, y) == (20 + int(dx), 10 + int(dy)), f"content moved by ({dx},{dy}) landed at ({x},{y})"


def test_coarse_corrector_align_moves_the_base_by_the_shift():
    torch = _torch()
    from src.utils.TemporalCorrection import ReferenceFrameCorrector
    lr = torch.zeros(1, 1, 32, 48)
    lr[0, 0, 10, 20] = 1.0
    corrector = ReferenceFrameCorrector()
    corrector.seed(lr, lr)
    assert corrector.align(3.0, 5.0) is True
    y, x = divmod(int(corrector._lr.flatten().argmax()), 48)
    assert (x, y) == (23, 15)
    assert corrector.aligned == 1


def test_coarse_registration_reads_a_known_shift_with_the_right_sign():
    """A synthetic shift must come back with its sign and within a fraction of a pixel of itself."""
    torch = _torch()
    frame = _line_art(torch, 216, 288)
    gate = TR.ReferenceFrameGate(statistic="coarse", pixel_veto=None)
    reference = gate._lowpass(torch.nn.functional.avg_pool2d(gate.luma(frame), 2),
                              gate.coarse_sigma / 2.0)
    for truth in ((2.0, 3.0), (-2.0, -3.0)):
        moved = _shift_torch(torch, frame, *truth)
        current = gate._lowpass(torch.nn.functional.avg_pool2d(gate.luma(moved), 2),
                                gate.coarse_sigma / 2.0)
        dx, dy, response, trusted = gate._register(reference, current)
        assert trusted and response > 0.2
        # the estimator is biased low on a broad peak (measured 0.37 px read as 0.21); the sign is not
        assert dx == pytest.approx(truth[0], abs=0.6), f"{truth} read as ({dx:.2f},{dy:.2f})"
        assert dy == pytest.approx(truth[1], abs=0.6), f"{truth} read as ({dx:.2f},{dy:.2f})"


def test_coarse_statistic_accepts_a_sub_pixel_wobble_the_pixel_statistic_refuses():
    """The whole point, on one frame pair: 0.4 px of scan wobble is not a changed cel."""
    torch = _torch()
    frame = _line_art(torch, 216, 288)
    moved = _shift_torch(torch, frame, 0.4, -0.25)

    pixel_gate = TR.ReferenceFrameGate(statistic="mean_abs", threshold=4.0, pixel_veto=None)
    pixel_gate.is_duplicate(frame)
    pixel_reuse = pixel_gate.is_duplicate(moved)
    pixel_delta = pixel_gate.last_delta

    coarse_gate = TR.ReferenceFrameGate(statistic="coarse", threshold=4.0, pixel_veto=None)
    coarse_gate.is_duplicate(frame)
    coarse_reuse = coarse_gate.is_duplicate(moved)
    coarse_delta = coarse_gate.last_delta

    print(f"\n0.4 px wobble: pixel statistic {pixel_delta:.2f} (reuse {pixel_reuse}), "
          f"coarse {coarse_delta:.2f} (reuse {coarse_reuse})")
    assert not pixel_reuse and pixel_delta > 4.0
    assert coarse_reuse and coarse_delta <= 4.0
    assert coarse_delta < pixel_delta / 2


def test_coarse_statistic_still_catches_a_change_that_matters():
    """A statistic that accepts everything is not a statistic: a 12-level area change must be a miss."""
    torch = _torch()
    frame = _line_art(torch, 216, 288)
    changed = frame.clone()
    changed[:, :, 60:140, 80:200] = (changed[:, :, 60:140, 80:200] + 12.0 / 255.0).clamp(0, 1)
    gate = TR.ReferenceFrameGate(statistic="coarse", threshold=4.0, pixel_veto=None)
    gate.is_duplicate(frame)
    assert gate.is_duplicate(changed) is False
    assert gate.last_delta >= 10.0


def test_coarse_identity_reads_exactly_zero_and_registers_no_shift():
    torch = _torch()
    frame = _line_art(torch, 216, 288)
    gate = TR.ReferenceFrameGate(statistic="coarse", threshold=0.001, pixel_veto=None)
    gate.is_duplicate(frame)
    assert gate.is_duplicate(frame) is True
    assert gate.last_delta == pytest.approx(0.0, abs=1e-9)
    assert gate.last_shift == (pytest.approx(0.0, abs=1e-6), pytest.approx(0.0, abs=1e-6))


def test_coarse_registration_is_refused_on_a_flat_frame_instead_of_inventing_half_the_picture():
    """A uniform frame has no correlation peak; the estimator answered with half the picture (120 px)."""
    torch = _torch()
    flat = torch.full((1, 3, 216, 288), 0.5)
    gate = TR.ReferenceFrameGate(statistic="coarse", threshold=4.0, pixel_veto=None)
    gate.is_duplicate(flat)
    assert gate.is_duplicate(flat) is True
    assert gate.registration_refusals >= 1
    assert gate.last_shift == (0.0, 0.0)
    assert gate.last_shift_step == (0.0, 0.0)


def test_coarse_shift_step_is_the_increment_for_the_emitter():
    """A reuse must move the picture by the increment against the previous frame, not against the reference."""
    torch = _torch()
    frame = _line_art(torch, 216, 288)
    gate = TR.ReferenceFrameGate(statistic="coarse", threshold=12.0, max_chain=4, pixel_veto=None)
    gate.is_duplicate(frame)                               # miss: the reference
    gate.is_duplicate(_shift_torch(torch, frame, 1.0, 0.0))
    first = gate.last_shift_step
    gate.is_duplicate(_shift_torch(torch, frame, 2.0, 0.0))
    second = gate.last_shift_step
    assert first[0] == pytest.approx(1.0, abs=0.6)
    # the second step is one more pixel on top of the first, so the increment is ~1 not ~2
    assert second[0] == pytest.approx(1.0, abs=0.7)
    assert gate.last_shift[0] == pytest.approx(2.0, abs=0.7)
