"""The reuse correction, unit-tested on tiny tensors: zero change means zero correction, and the
low-frequency part is passed at unit gain.

The numbers the shipped correction is held to are in docs/wsp/verified/reuse-decision-statistic.txt; these tests
only pin the algebra, because a correction that silently adds the wrong term is indistinguishable from a good one
until a frame is rendered.
"""
import pytest

torch = pytest.importorskip("torch")

from backend.src.utils.TemporalCorrection import ReferenceFrameCorrector


def pair(h=16, w=16, value=0.5, dtype=torch.float32):
    lr = torch.full((1, 3, h, w), value, dtype=dtype)
    hr = torch.full((1, 3, 2 * h, 2 * w), value, dtype=dtype)
    return hr, lr


def test_no_correction_without_a_seeded_pair():
    c = ReferenceFrameCorrector()
    _, lr = pair()
    assert c.correct(lr) is None
    assert c.refused == 1


def test_unchanged_frame_is_not_modified():
    c = ReferenceFrameCorrector()
    hr, lr = pair()
    c.seed(hr, lr)
    out = c.correct(lr.clone())
    assert out is not None
    assert torch.allclose(out, hr, atol=1e-6)


def test_a_flat_shift_is_passed_at_unit_gain_at_least():
    """A constant change is entirely low-frequency: the correction must move the frame by the same amount
    (the flat mask may add the grain gain on top, so the check is a range, not an equality)."""
    c = ReferenceFrameCorrector()
    hr, lr = pair(value=0.50)
    c.seed(hr, lr)
    shifted = lr + 0.10
    out = c.correct(shifted)
    delta = float((out - hr).mean())
    assert 0.095 <= delta <= 0.14, delta
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_the_base_is_the_previous_emitted_frame_not_the_seed():
    """Recursion: the second correction must build on the first output, not on the cached model frame."""
    c = ReferenceFrameCorrector()
    hr, lr = pair(value=0.30)
    c.seed(hr, lr)
    first = c.correct(lr + 0.10)
    second = c.correct(lr + 0.20)
    assert float(second.mean()) > float(first.mean()) + 0.05


def test_shape_change_refuses_instead_of_crashing():
    c = ReferenceFrameCorrector()
    hr, lr = pair()
    c.seed(hr, lr)
    assert c.correct(torch.zeros(1, 3, 8, 8)) is None
    assert c.refused == 1


def test_the_mask_only_adds_gain_where_the_frame_is_flat_and_small_delta():
    """A high-contrast edge and a large delta must both fall back to unit gain."""
    c = ReferenceFrameCorrector(alpha=2.0)
    hr = torch.full((1, 3, 16, 16), 0.5)
    lr = torch.zeros(1, 3, 8, 8)
    lr[..., ::2] = 1.0                       # strong gradient everywhere
    c.seed(hr, lr)
    shifted = lr + 0.02                      # small delta, but the gradient veto fires
    out = c.correct(shifted)
    delta = float((out - hr).mean())
    # without the mask a 0.02 step would read up to 0.04; with it, ~0.02
    assert delta < 0.03, delta
