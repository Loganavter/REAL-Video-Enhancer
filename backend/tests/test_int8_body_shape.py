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
