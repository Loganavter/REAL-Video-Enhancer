"""The int8 body's coverage rule: the hop COUNT is data, the width is the kernel's constexpr.

Why this is pinned: the constructor used to require exactly seventeen Conv2d and seventeen PReLU, i.e. the
sixteen-hop shape of the model the lane was built on. Everything else in the body is driven by the model -
the launch loop, the halo buffers, the requant chain - so that check was a constant standing in for a
measurement, and it refused models of the same architecture with a different depth (the SuperUltraCompact
family has nine inner convs) that need no new kernel at all (row rve-hop-count-general-2026-11-20).

The tests use a synthetic body: no spandrel, no weights, no CUDA. They hold the rule, not the arithmetic -
whether a nine-hop body is faster and still correct is a render question.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
nn = torch.nn

from src.pytorch.Int8Body import Int8Body  # noqa: E402


class _Body(nn.Module):
    """The shape `Int8Body` reads: `body` = head conv, PReLU, n x (conv, PReLU), tail conv; plus an upsampler.

    That is SRVGGNetCompact's layout, and the point of the synthetic version is that the rule under test is
    structural - it never looks at weights.
    """

    def __init__(self, n_inner: int, feat: int = 64, tail: int = 12, scale: int = 2):
        super().__init__()
        layers = [nn.Conv2d(3, feat, 3, padding=1), nn.PReLU(feat)]
        for _ in range(n_inner):
            layers += [nn.Conv2d(feat, feat, 3, padding=1), nn.PReLU(feat)]
        layers.append(nn.Conv2d(feat, tail, 3, padding=1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(scale)
        self.upscale = scale


@pytest.mark.parametrize("n_inner", [9, 16, 17])
def test_any_hop_count_at_the_tuned_width_is_accepted(n_inner):
    """Nine hops (SuperUltraCompact) and seventeen (the Archivist denoisers) are the same kernel, twice."""
    body = Int8Body(_Body(n_inner))
    assert len(body.inner) == n_inner, "the launch loop must follow the model, not a constant"
    assert body.feat == 64


def test_a_single_hop_body_is_refused_with_a_reason():
    """With one inner conv the first and the last hop are the same launch, and the zero-halo scheme assumes
    the last hop writes the PLAIN buffer the tail reads. Two is the floor."""
    with pytest.raises(ValueError, match="n>=2"):
        Int8Body(_Body(1))


def test_a_native_width_is_read_from_the_model():
    """The width is the kernel's `C` constexpr, so a 48-channel model is a recompile, not a rewrite - the
    constructor reads it from the model, and pinning a different one must say why it cannot."""
    body = Int8Body(_Body(16, feat=48, tail=3, scale=1))
    assert body.feat == 48, "the width must come from the model, not from a default"
    with pytest.raises(ValueError, match="asked for 64"):
        Int8Body(_Body(16, feat=48, tail=3, scale=1), feat=64)


def test_a_width_that_is_not_a_power_of_two_is_padded_not_refused():
    """`tl.arange` needs a power of two and the int8 dot needs K a multiple of 16, so 24 and 48 are padded up to
    32 and 64 rather than refused - the padding channels carry zero and the tail reads only `cout`."""
    body = Int8Body(_Body(16, feat=24, tail=3, scale=1))
    assert body.feat == 24 and body.cpad == 32
    body = Int8Body(_Body(16, feat=48, tail=3, scale=1))
    assert body.feat == 48 and body.cpad == 64
    body = Int8Body(_Body(16, feat=32, tail=3, scale=1))
    assert body.cpad == 32, "a power of two must pay nothing"


def test_supports_follows_the_same_rule():
    """`supports` is what the render asks before building: it must agree with the constructor, or the log
    would say "supported" and the build would raise."""
    assert Int8Body.supports(_Body(9))
    assert Int8Body.supports(_Body(16))
    assert Int8Body.supports(_Body(16, feat=48)), "a native width is supported; it is just not tuned"
    assert Int8Body.supports(_Body(16, feat=24)), "24 is padded to 32, not refused"
    assert not Int8Body.supports(_Body(16, feat=4)), "below 8 channels there is nothing to tile"
    assert not Int8Body.supports(_Body(1))


# ======================================================================================
# the probe's loader reads the same shape from the weights
# ======================================================================================


def _state(feat: int, inner: int, scale: int) -> dict:
    """A Compact state dict with the shapes the loader reads (values are irrelevant)."""
    out = {"body.0.weight": torch.zeros(feat, 3, 3, 3), "body.0.bias": torch.zeros(feat)}
    for i in range(inner):
        out[f"body.{2 + 2 * i}.weight"] = torch.zeros(feat, feat, 3, 3)
    out[f"body.{2 + 2 * inner}.weight"] = torch.zeros(3 * scale * scale, feat, 3, 3)
    return out


@pytest.mark.parametrize("feat,inner,scale", [(64, 16, 2), (64, 8, 2), (48, 16, 1), (32, 16, 1)])
def test_the_probe_reads_width_depth_and_scale_from_the_weights(feat, inner, scale):
    from src.utils.Int8Probe import compact_shape

    assert compact_shape(_state(feat, inner, scale)) == (feat, inner, scale)


# ======================================================================================
# padding a non-power-of-two width
#
# `tl.arange` needs a power of two and the int8 dot needs K a multiple of 16, so 48 is padded to 64 and 24 to 32
# with zero weights and neutral scales. Two bugs lived in that seam and both are pinned here, on CPU: the int8
# tail multiplied its weight by the last hop's per-channel scale, which is cpad long after padding, and the fused
# head wrote a feat-wide row into a cpad-wide matrix. Neither fails in a test that only constructs the body.
# ======================================================================================


def _built_body(n_inner: int, feat: int, tail: int = 3, scale: int = 1):
    """A body built on CPU with synthetic grids - enough to exercise every padding site."""
    body = Int8Body(_Body(n_inner, feat=feat, tail=tail, scale=scale))
    ones = torch.ones(feat)
    body._grids = (ones, [torch.ones(feat) for _ in range(n_inner)])
    body._launches = None
    body._build()
    return body


@pytest.mark.parametrize("feat,cpad", [(48, 64), (24, 32), (32, 32), (64, 64)])
def test_the_padded_widths_reach_every_kernel_site(feat, cpad):
    body = _built_body(16, feat)
    assert body.cpad == cpad
    w = body._launches[0]["w"]
    assert w.shape == (9, cpad, cpad), "the hop's taps must be cpad x cpad"
    assert float(w[:, feat:, :].abs().sum()) == 0.0, "padding rows must be zero"
    assert float(w[:, :, feat:].abs().sum()) == 0.0, "padding columns must be zero"
    for key, fill in (("vec", 1.0), ("os", 1.0), ("slope", 1.0), ("bias", 0.0)):
        v = body._launches[0][key]
        assert v.numel() == cpad, f"{key} must be cpad long"
        if cpad != feat:
            assert float(v[feat:].sum()) == fill * (cpad - feat), f"{key} padding must be neutral"
    assert body._head_w32.shape == (32, cpad), "the fused head's matrix is (K=32, cpad)"
    assert float(body._head_w32[:, feat:].abs().sum()) == 0.0
    assert body._tail_taps.shape == (9, cpad, 16), "the int8 tail reads cpad channels and pads cout to 16"
    assert float(body._tail_taps[:, feat:, :].abs().sum()) == 0.0


def test_a_padded_width_needs_the_two_int8_boundaries():
    """The fp16 head-quantise and tail-dequant passes are not padded, so a padded width requires the fused head
    and the int8 tail - and says so rather than writing a feat-wide buffer the hops will read as cpad."""
    class _Wide(_Body):
        def __init__(self):
            super().__init__(16, feat=48, tail=48, scale=2)   # a tail wider than 16 channels: no int8 tail

    with pytest.raises(ValueError, match="needs the int8 tail"):
        Int8Body(_Wide())


# ======================================================================================
# the start-up pick's rule
#
# The pick builds the body at several clamp values and keeps the best by the app's own gate on HELD-OUT frames.
# The rule has one guard - the winner must be twice as good as the default - because on three frames the gap
# between adjacent K values is often a single pixel and an unguarded pick would chase noise.
# ======================================================================================


def test_the_pick_keeps_a_clear_winner():
    assert Int8Body.choose_grid_k({1: 0.012, 2: 0.0002, 3: 0.002, 4: 0.004}) == 2


def test_the_pick_leaves_a_marginal_winner_alone():
    """Epsilon's real numbers: K=3 is 1.4x better than K=2 - below the guard, so K=2 stays."""
    assert Int8Body.choose_grid_k({1: 1.09, 2: 0.0962, 3: 0.0665, 4: 0.0998}) == 2


def test_the_pick_takes_a_doubling_win_at_another_k():
    """realesr-animevideov3's real numbers: K=3 is 7x better than K=2, so the pick leaves the default."""
    assert Int8Body.choose_grid_k({1: 0.1246, 2: 0.0504, 3: 0.0071, 4: 0.0446}) == 3


def test_the_pick_keeps_the_default_when_the_best_other_k_only_ties_it():
    """Garfield's real numbers: K=4 reads 0.0029 against K=2's 0.0030 - a tie, not a win, and the 300x win was
    against the per-tensor grid, not against K=2. A tie must not move the default."""
    assert Int8Body.choose_grid_k({1: 0.24, 2: 0.0030, 3: 0.0157, 4: 0.0029}) == 2


def test_the_pick_is_stable_when_every_candidate_is_clean():
    """All-clean models read 0.0000% at every K; the default must win ties rather than drift."""
    assert Int8Body.choose_grid_k({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}) == 2


def test_a_non_contiguous_frame_is_made_contiguous_before_the_head_kernel():
    """The head kernel indexes the frame as `c*H*W + y*W + x`, so a view reads as garbage codes - and a view is
    what `torch.from_numpy(img).permute(2, 0, 1)` produces, which is how Int8Probe builds its calibration frames.
    The copy is invisible when the caller is already contiguous, which is the case on the render path."""
    body = Int8Body(_Body(16))
    x = torch.zeros(1, 3, 8, 8, dtype=torch.float16).permute(0, 1, 3, 2)   # a non-contiguous view
    assert not x.is_contiguous()
    # the guard is at the top of __call__, before any kernel: exercise it without CUDA by checking the source
    import inspect

    src = inspect.getsource(Int8Body.__call__)
    assert "x = x.contiguous()" in src, "the body must not hand a view to the head kernel"
    assert "is_contiguous" in src


# ======================================================================================
# the calibration cache
#
# The grids are 24 vectors of 64 floats that depend on (model, clip) and nothing else, and measuring them means
# running the fp16 model over every calibration frame - ~4.8 s for 24 frames in eager. A hit removes all of it.
# The key is a fingerprint of the frames plus the model's digest, so a different clip or model misses by
# construction; a cache that silently served another clip's grids would be a stale-number source.
# ======================================================================================


def _tiny_frames(n: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(1, 3, 16, 16, generator=g) for _ in range(n)]


def test_the_cache_round_trips_and_skips_the_measurement(tmp_path, monkeypatch):
    monkeypatch.setenv("RVE_INT8_CALIB_CACHE", str(tmp_path))
    frames = _tiny_frames()
    model = _Body(16)                      # ONE model: the cache keys on its weights, so two random inits differ
    first = Int8Body(model)
    first.calibrate(frames)
    assert first._grids is not None

    second = Int8Body(model)
    calls = []
    monkeypatch.setattr(Int8Body, "_measure", lambda self, x: calls.append(1) or (_ for _ in ()).throw(
        AssertionError("a cache hit must not measure")))
    second.calibrate(frames)
    assert not calls, "the hit still ran the model"
    assert torch.allclose(first._grids[0], second._grids[0]), "the cached grids must be the measured ones"
    # the cache stores CPU tensors; a grid left on the wrong device fails only at the first kernel launch, which
    # is how this was found (certify raised a device mismatch on the first cached run)
    assert second._grids[0].device == first._grids[0].device, "the hit must land on the body's device"


def test_a_different_clip_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("RVE_INT8_CALIB_CACHE", str(tmp_path))
    model = _Body(16)
    Int8Body(model).calibrate(_tiny_frames(seed=1))
    other = Int8Body(model)
    other.calibrate(_tiny_frames(seed=2))       # different frames -> a real measurement, not a stale hit
    assert other._grids is not None


def test_the_cache_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("RVE_INT8_CALIB_CACHE", "0")
    Int8Body(_Body(16)).calibrate(_tiny_frames())
    assert not list(tmp_path.glob("*.pt")), "RVE_INT8_CALIB_CACHE=0 must not write anything"


def test_a_clean_default_skips_the_sweep():
    """When K=2's own share is already small the sweep cannot change the decision: the ladder's largest gain is
    7.1x, so 0.01% x 7.1 = 0.071% - inside the 0.1% bar, which is the point. Below it the switch only improves a
    number that is already orders inside; above it the sweep can buy real margin AT the bar. The threshold was
    0.002% until 2026-11-20, when the whole ladder was read: 14 of 22 models sit at or below 0.002% and 5 more
    between 0.003% and 0.017%, so 0.002% paid for the sweep on models that gain 3-4x on a quantity 6-30x inside
    the bar. It also has to absorb the 3x gap between the pick's six-frame score and the ladder's three-frame one
    (realesr: 0.0165% against 0.0504%), which is why it is not simply the ladder's near-bar number."""
    assert Int8Body.skip_sweep(0.0002), "Futsuu's K=2 is 0.0002%: nothing to win"
    assert Int8Body.skip_sweep(0.0000), "an already-clean model must not be swept"
    assert Int8Body.skip_sweep(0.0030), "Garfield gains at most 1.05x: not worth a build"
    assert not Int8Body.skip_sweep(0.0504), "realesr's K=2 is 0.0504%: the sweep is what finds K=3"
    assert not Int8Body.skip_sweep(0.0166), "Ani4K's K=2 is 0.0166%: same"
