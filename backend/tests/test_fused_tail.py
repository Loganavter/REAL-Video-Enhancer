"""Tests for ``backend/src/pytorch/FusedTail.py`` (fuse an x4 compact tail down to x2).

Three layers, because the cell that has to verify this ships without torch (the
bundled interpreter with torch lives outside the worker jail):

* **taps / structure (numpy, no torch)** - tap normalisation, the delta at
  ``target_scale == source_scale``, the box kernel, and the derived ``down``
  kernel against an independent dense-matrix collapse.  These pin the index
  algebra without needing a network;
* **algebra (numpy, no torch)** - the fused tail against a *reference
  implementation written from the definition* (`_ref_tail`: conv_last ->
  pixel_shuffle -> nearest skip -> dense separable downsample).  The reference
  never mentions channel folding or convolutions of convolutions, so agreement
  here is evidence, not a tautology.  Measured: interior agreement ~1e-15,
  differences confined to the 2-pixel border ring where the reference clamps the
  *HR* sample index and the fused convolution replicate-pads in *LR* space;
* **torch path, replayed on numpy (no torch)** - ``apply_tail`` itself, called
  with a stand-in whose ``conv2d``/``pad``/``pixel_shuffle`` carry *torch's*
  signatures over the numpy helpers: same block-diagonal/grouped ``conv_b``,
  same border per convolution, and a parity assert against ``apply_tail_numpy``.
  This layer exists because the cell that verifies the module has no torch: a
  ``padding_mode=`` kwarg on ``F.conv2d`` and a dense weight behind ``groups=C``
  both pass every numpy test and die on the first real frame;
* **torch (skipped when torch is missing)** - the real thing: a spandrel
  ``Compact``/``SRVGGNetCompact``, ``fuse_tail`` on it, ``apply_tail`` against
  ``net(x)`` downsampled by the numpy reference, plus the identity
  (``target_scale == source_scale``) and error paths.

The border numbers are printed as well as asserted: ``pytest -s`` shows them.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys

import numpy as np
import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.pytorch import FusedTail as FT  # noqa: E402

TOL = 1.0 / 255.0          # acceptance tolerance from the task spec
SEED = 1234


# ======================================================================================
# reference implementation - written from the definition, not from the derivation
# ======================================================================================
def _ref_cubic(t: np.ndarray, a: float = FT.BICUBIC_A) -> np.ndarray:
    t = np.abs(np.asarray(t, dtype=np.float64))
    return np.where(
        t <= 1.0,
        (a + 2.0) * t**3 - (a + 3.0) * t**2 + 1.0,
        np.where(t < 2.0, a * t**3 - 5.0 * a * t**2 + 8.0 * a * t - 4.0 * a, 0.0),
    )


def _ref_support(kernel: str, d: int) -> float:
    return (2.0 if kernel == "bicubic" else 0.5) * d


def _ref_resize_matrix(n_out: int, n_in: int, d: int, kernel: str = "bicubic",
                       border: str = "clamp", a: float = FT.BICUBIC_A) -> np.ndarray:
    """Dense 1D matrix of a ``1/d`` resize, ``align_corners=False``.

    Row ``p`` samples the source coordinate ``u = (p + 0.5) * d - 0.5`` with the
    (optionally antialiased) kernel ``kern((r - u)/d)`` normalised to sum 1.
    ``border="clamp"`` folds out-of-range taps onto the edge sample (what
    ``F.interpolate`` does); ``border="renorm"`` drops them and renormalises
    (what PIL/cv2-style resamplers do).  Built here from that definition alone -
    no convolution folding, no sub-pixel bookkeeping.
    """
    support = _ref_support(kernel, d)
    half = int(math.ceil(support))
    M = np.zeros((n_out, n_in), dtype=np.float64)
    for p in range(n_out):
        u = (p + 0.5) * d - 0.5
        origin = int(math.floor(u))
        pairs = []
        for j in range(origin - half, origin + half + 1):
            rel = j - u
            if abs(rel) > support + 1e-9:
                continue
            w = (float(_ref_cubic(np.array([rel / d]), a)[0]) if kernel == "bicubic"
                 else float(abs(rel / d) <= 0.5 + 1e-9))
            if w == 0.0:
                continue
            pairs.append((j, w))
        total = sum(w for _, w in pairs)
        assert total != 0.0
        pairs = [(j, w / total) for j, w in pairs]
        if border == "clamp":
            for j, w in pairs:
                M[p, min(max(j, 0), n_in - 1)] += w
        else:
            keep = [(j, w) for j, w in pairs if 0 <= j < n_in]
            total = sum(w for _, w in keep)
            for j, w in keep:
                M[p, j] += w / total
    return M


def _ref_conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
                pad: int = 1, pad_mode: str = "constant") -> np.ndarray:
    """Stride-1 cross-correlation (torch semantics), ``w`` shaped ``(out, in, kh, kw)``."""
    if pad:
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                    mode="edge" if pad_mode == "replicate" else "constant")
    else:
        xp = x
    kh, kw = w.shape[2], w.shape[3]
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))
    out = np.einsum("nchwuv,ocuv->nohw", win, w, optimize=True)
    if b is not None:
        out = out + b.reshape(1, -1, 1, 1)
    return out


def _ref_pixel_shuffle(x: np.ndarray, r: int) -> np.ndarray:
    n, c, h, w = x.shape
    return (x.reshape(n, c // r**2, r, r, h, w)
             .transpose(0, 1, 4, 2, 5, 3)
             .reshape(n, c // r**2, h * r, w * r))


def _ref_nearest(x: np.ndarray, r: int) -> np.ndarray:
    return np.repeat(np.repeat(x, r, axis=2), r, axis=3)


def _ref_downscale(hr: np.ndarray, d: int, kernel: str = "bicubic",
                   border: str = "clamp") -> np.ndarray:
    Mh = _ref_resize_matrix(hr.shape[2] // d, hr.shape[2], d, kernel, border)
    Mw = _ref_resize_matrix(hr.shape[3] // d, hr.shape[3], d, kernel, border)
    return np.einsum("pr,ncrs,qs->ncpq", Mh, hr, Mw)


def _ref_tail(conv_w, conv_b, features, x, source_scale, target_scale,
              kernel="bicubic", border="clamp"):
    """``downscale(conv_last(features).pixel_shuffle(s) + nearest(x, s))``."""
    d = source_scale // target_scale
    z = _ref_conv2d(features, conv_w, conv_b, pad=conv_w.shape[2] // 2)
    hr = _ref_pixel_shuffle(z, source_scale) + _ref_nearest(x, source_scale)
    return _ref_downscale(hr, d, kernel, border=border)


def _interior_mask(n_out: int, n_in: int, d: int, kernel: str = "bicubic") -> np.ndarray:
    """Output positions whose whole tap window lies inside the frame."""
    support = _ref_support(kernel, d)
    half = int(math.ceil(support))
    mask = np.zeros(n_out, dtype=bool)
    for p in range(n_out):
        u = (p + 0.5) * d - 0.5
        mask[p] = (math.floor(u) - half >= 0) and (math.floor(u) + half <= n_in - 1)
    return mask


def _split_error(got, ref, d, kernel="bicubic"):
    """(interior max abs error, border max abs error) split by the tap mask."""
    err = np.abs(got - ref)
    mi = _interior_mask(got.shape[2], ref.shape[2] * d, d, kernel)
    mj = _interior_mask(got.shape[3], ref.shape[3] * d, d, kernel)
    interior = err[:, :, mi][:, :, :, mj].max()
    border = err.copy()
    border[:, :, mi, :] = 0.0
    border[:, :, :, mj] = 0.0
    return float(interior), float(border.max())


def _report(name, interior, border):
    print(f"{name}: interior {interior:.3e} ({interior * 255:.5f}/255), "
          f"border {border:.3e} ({border * 255:.3f}/255)")
    return interior, border


def _random_case(rng, source_scale=4, target_scale=2, num_feat=16, size=16,
                 weight_scale=0.05):
    conv_w = rng.standard_normal((3 * source_scale**2, num_feat, 3, 3)) * weight_scale
    conv_b = rng.standard_normal(3 * source_scale**2) * weight_scale
    x = rng.random((1, 3, size, size))
    features = rng.random((1, num_feat, size, size))
    return conv_w, conv_b, features, x


# ======================================================================================
# taps / structure (numpy only)
# ======================================================================================
def test_gather_weights_are_normalised_per_phase():
    for source_scale, target_scale in ((4, 2), (4, 1), (2, 1), (4, 4), (3, 1)):
        g, radius = FT.gather_weights(source_scale, target_scale)
        per_phase = g.sum(axis=(1, 2))
        assert per_phase.shape == (target_scale,)
        assert np.allclose(per_phase, 1.0, atol=1e-12), (source_scale, target_scale, per_phase)
        assert radius >= 0


def test_three_by_three_support_for_four_to_two():
    """4x4 -> 2x2 needs only LR +-1, so conv_b is 3x3 (the documented contract)."""
    _, radius = FT.gather_weights(4, 2)
    assert radius == 1
    # 4x4 -> 1x1 spans two LR pixels in the downscale, so it needs 5x5
    _, radius_one = FT.gather_weights(4, 1)
    assert radius_one == 2


def test_identity_scale_taps_are_a_delta():
    g, radius = FT.gather_weights(4, 4)
    assert radius == 0
    assert np.allclose(g.sum(axis=2), np.eye(4), atol=0.0)   # one sub-pixel per phase
    assert np.allclose(g, np.eye(4)[:, :, None], atol=0.0)


def test_area_taps_are_the_uniform_box():
    g, radius = FT.gather_weights(4, 2, kernel="area")
    # the box has width d = 2 HR samples, which is *within* one LR pixel, so the
    # area fold needs no neighbour at all: 1x1 convolutions.
    assert radius == 0
    for phase in range(2):
        assert int(np.count_nonzero(g[phase])) == 2          # the two HR rows of the block
        assert np.allclose(g[phase][g[phase] != 0], 0.5)
    assert np.allclose(g.sum(axis=(1, 2)), 1.0)


def test_down_kernel_matches_the_reference_subpixel_mass():
    """`down` = the taps collapsed over sub-pixels, from the dense reference matrices."""
    s, t = 4, 2
    d = s // t
    # 4 LR pixels wide, 8 output rows; rows t*2 + phase sit in the interior, so the
    # reference's taps for that row group into exactly the three LR offsets -1, 0, +1
    M = _ref_resize_matrix(t * 4, s * 4, d)
    mass = FT.subpixel_mass(FT.gather_weights(s, t)[0])
    for phase in range(t):
        row = M[t * 2 + phase]
        groups = [row[j * s:(j + 1) * s].sum() for j in (1, 2, 3)]   # LR offsets -1, 0, +1
        assert np.allclose(groups, mass[phase], atol=1e-12), (phase, groups, mass[phase])


def test_derived_kernels_are_sparse_and_block_diagonal():
    s, t = 4, 2
    fused = FT.derive_kernels(np.random.default_rng(0).standard_normal((3 * s * s, 8, 3, 3)),
                              np.zeros(3 * s * s), s, t)
    assert fused["conv_b"].shape == (3 * t * t, 3 * s * s, 3, 3)
    assert fused["down"].shape == (3 * t * t, 3, 3, 3)
    # each of the 4 target sub-pixels only reads its own channel group
    for o in range(3 * t * t):
        c = o // (t * t)
        used = np.abs(fused["conv_b"][o]).sum(axis=(1, 2)) > 0
        expected = np.zeros(3 * s * s, dtype=bool)
        expected[c * s * s:(c + 1) * s * s] = True
        assert np.array_equal(used, expected), o
    counts = FT.nonzeros(fused)
    g, _ = FT.gather_weights(s, t)
    mass = FT.subpixel_mass(g)
    # per output channel the kernel is an outer product, so the non-zeros multiply
    nz_v = int(np.count_nonzero(g[0]))
    nz_m = int(np.count_nonzero(mass[0]))
    assert counts["conv_b"] == 3 * t * t * nz_v * nz_v
    assert counts["down"] == 3 * t * t * nz_m * nz_m
    assert counts["conv_b"] < fused["conv_b"].size


def test_conv2d_helper_is_cross_correlation():
    """torch's conv2d is cross-correlation: a tap at (0, 0) shifts the result down-right.

    This pins the one convention the torch path could silently get wrong, so the
    numpy mirror and ``F.conv2d`` are talking about the same operator.
    """
    x = np.zeros((1, 1, 5, 5))
    x[0, 0, 1, 1] = 1.0
    k = np.zeros((1, 1, 3, 3))
    k[0, 0, 0, 0] = 1.0                      # top-left tap
    out = FT._conv2d_numpy(x, k, None, pad=1, pad_mode="constant")
    assert out[0, 0, 2, 2] == 1.0 and out.sum() == 1.0


def test_conv_b_is_block_diagonal_enough_for_a_grouped_conv():
    """apply_tail runs conv_b with groups=C: prove no tap crosses a colour block."""
    fused = FT.derive_kernels(np.random.default_rng(5).standard_normal((48, 8, 3, 3)),
                              np.zeros(48), 4, 2)
    FT._assert_block_diagonal(fused["conv_b"], 3, 4, 2)      # must not raise
    broken = fused["conv_b"].copy()
    broken[0, 16] = 1.0                                      # a tap outside colour 0
    with pytest.raises(RuntimeError, match="block diagonal"):
        FT._assert_block_diagonal(broken, 3, 4, 2)


def test_pixel_shuffle_helper_matches_the_torch_convention():
    """ps[c, s*h + i, s*w + j] == z[c*s^2 + i*s + j, h, w] - quoted in the module docstring."""
    rng = np.random.default_rng(3)
    z = rng.standard_normal((1, 3 * 16, 5, 7))
    out = FT._pixel_shuffle_numpy(z, 4)
    for c in range(3):
        for i in range(4):
            for j in range(4):
                assert np.allclose(out[0, c, i::4, j::4], z[0, c * 16 + i * 4 + j])


# ======================================================================================
# algebra: fused tail vs the reference (numpy only)
# ======================================================================================
def test_conv_b_reproduces_pixel_shuffle_then_downscale():
    """The pixel-shuffle/downscale fold alone, isolated from conv_last."""
    rng = np.random.default_rng(SEED)
    s, t, d = 4, 2, 2
    fused = FT.derive_kernels(np.zeros((3 * s * s, 4, 3, 3)), 0.1 * np.ones(3 * s * s), s, t)
    z = rng.standard_normal((1, 3 * s * s, 12, 12))
    got = _ref_pixel_shuffle(
        FT._conv2d_numpy(z, fused["conv_b"], None, pad=1, pad_mode="replicate"), t)
    ref = _ref_downscale(_ref_pixel_shuffle(z, s), d)
    interior, border = _report("conv_b only", *_split_error(got, ref, d))
    assert interior <= TOL
    assert got.shape == ref.shape == (1, 3, 24, 24)


def test_full_tail_matches_the_reference_in_the_interior():
    rng = np.random.default_rng(SEED)
    s, t = 4, 2
    conv_w, conv_b, features, x = _random_case(rng)
    fused = FT.derive_kernels(conv_w, conv_b, s, t)
    got = FT.apply_tail_numpy(fused, features, x)
    ref = _ref_tail(conv_w, conv_b, features, x, s, t)
    interior, border = _report("4x4 -> 2x2", *_split_error(got, ref, s // t))
    assert got.shape == ref.shape == (1, 3, 32, 32)
    assert interior <= TOL        # measured ~1e-15: the fold is exact, not approximate
    assert border >= 0.0


def test_error_is_confined_to_the_border_ring():
    """Outside the 2-pixel tap ring the fused tail is exact for *every* case tried."""
    rng = np.random.default_rng(SEED + 1)
    s, t = 4, 2
    conv_w, conv_b, features, x = _random_case(rng)
    fused = FT.derive_kernels(conv_w, conv_b, s, t)
    got = FT.apply_tail_numpy(fused, features, x)
    ref = _ref_tail(conv_w, conv_b, features, x, s, t)
    err = np.abs(got - ref)
    half = int(math.ceil(_ref_support("bicubic", s // t)))
    assert err[:, :, half:-half, half:-half].max() <= 1e-12
    assert err[:, :, half:-half, half:-half].max() < err.max()


def test_reported_border_error_bounds():
    """Records the number the supervisor should expect at the frame edge.

    On white-noise *features* (worst case: neighbouring sub-pixel channels are
    uncorrelated) the ring error measures ~7e-2; on spatially smooth features,
    what a real network produces, ~4.6e-2.  The absolute scale tracks the weight
    magnitude (these are synthetic weights), the point of the measurement is that
    the deviation lives on the ring and nowhere else.  Both are printed; only the
    ordering and a sanity bound are asserted.
    """
    rng = np.random.default_rng(SEED + 2)
    s, t = 4, 2
    conv_w, conv_b, features, x = _random_case(rng)
    fused = FT.derive_kernels(conv_w, conv_b, s, t)
    noisy = _split_error(FT.apply_tail_numpy(fused, features, x),
                         _ref_tail(conv_w, conv_b, features, x, s, t), s // t)[1]

    smooth_features = np.empty_like(features)
    yy, xx = np.mgrid[0:features.shape[2], 0:features.shape[3]]
    wave = np.sin(xx / 5.0) * np.cos(yy / 6.0)
    for c in range(features.shape[1]):
        smooth_features[0, c] = 0.5 + 0.4 * wave * (1.0 + 0.01 * c)
    smooth_x = 0.5 + 0.4 * wave
    smooth_x = np.repeat(smooth_x[None, None], 3, axis=1)
    smooth = _split_error(FT.apply_tail_numpy(fused, smooth_features, smooth_x),
                          _ref_tail(conv_w, conv_b, smooth_features, smooth_x, s, t),
                          s // t)[1]
    _report("border, noise features", 0.0, noisy)
    _report("border, smooth features", 0.0, smooth)
    assert smooth < noisy
    assert noisy < 1.0


def test_area_kernel_reproduces_the_box_mean():
    rng = np.random.default_rng(SEED + 3)
    s, t = 4, 2
    conv_w, conv_b, features, x = _random_case(rng)
    fused = FT.derive_kernels(conv_w, conv_b, s, t, kernel="area")
    got = FT.apply_tail_numpy(fused, features, x)
    z = _ref_conv2d(features, conv_w, conv_b, pad=1)
    hr = _ref_pixel_shuffle(z, s) + _ref_nearest(x, s)
    box = hr.reshape(hr.shape[0], hr.shape[1], hr.shape[2] // 2, 2,
                     hr.shape[3] // 2, 2).mean(axis=(3, 5))
    interior, border = _report("area", *_split_error(got, box, s // t, "area"))
    assert interior <= TOL


def test_identity_target_reproduces_the_plain_tail():
    """target_scale == source_scale must be a no-op (delta taps, 1x1 convs)."""
    rng = np.random.default_rng(SEED + 4)
    s = 4
    conv_w, conv_b, features, x = _random_case(rng, source_scale=s, target_scale=s)
    fused = FT.derive_kernels(conv_w, conv_b, s, s)
    assert fused["radius"] == 0
    got = FT.apply_tail_numpy(fused, features, x)
    ref = (_ref_pixel_shuffle(_ref_conv2d(features, conv_w, conv_b, pad=1), s)
           + _ref_nearest(x, s))
    assert np.allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("source_scale,target_scale", [(4, 1), (2, 1)])
def test_larger_factors_match_the_reference(source_scale, target_scale):
    """4x4 -> 1x1 and 2x2 -> 1x1: 5x5 kernels, same derivation."""
    rng = np.random.default_rng(SEED + 5)
    conv_w, conv_b, features, x = _random_case(
        rng, source_scale=source_scale, target_scale=target_scale, num_feat=12, size=10)
    fused = FT.derive_kernels(conv_w, conv_b, source_scale, target_scale,
                              num_in_ch=3)
    assert fused["radius"] == 2
    got = FT.apply_tail_numpy(fused, features, x)
    ref = _ref_tail(conv_w, conv_b, features, x, source_scale, target_scale)
    interior, border = _report(f"{source_scale}x -> {target_scale}x",
                               *_split_error(got, ref, source_scale // target_scale))
    assert interior <= TOL
    assert got.shape[2:] == ref.shape[2:]


def test_skip_has_unit_dc_gain():
    """A flat frame and a zero conv_last: the fused skip must return the same flat frame."""
    s, t = 4, 2
    fused = FT.derive_kernels(np.zeros((3 * s * s, 4, 3, 3)), np.zeros(3 * s * s), s, t)
    x = np.full((1, 3, 8, 8), 0.25)
    got = FT.apply_tail_numpy(fused, np.zeros((1, 4, 8, 8)), x)
    assert np.allclose(got, 0.25, atol=1e-12)


def test_no_skip_returns_none_down():
    fused = FT.derive_kernels(np.zeros((48, 4, 3, 3)), None, 4, 2, skip=False)
    assert fused["down"] is None
    got = FT.apply_tail_numpy(fused, np.zeros((1, 4, 6, 6)))
    assert got.shape == (1, 3, 12, 12)


def test_tail_params_counts_the_work():
    fused = FT.derive_kernels(np.zeros((48, 64, 3, 3)), None, 4, 2)
    params = FT.tail_params(fused)
    g, _ = FT.gather_weights(4, 2)
    nz_v = int(np.count_nonzero(g[0]))
    assert params["conv_a_macs"] == 64 * 48 * 9
    assert params["conv_b_macs"] == 12 * 48 * 9
    assert params["conv_b_macs_sparse"] == 12 * nz_v * nz_v   # 4 of 9 taps survive
    assert params["conv_b_macs_grouped"] == 12 * 16 * 9       # groups=3: a third of the work
    assert params["down_macs"] == 12 * 3 * 9


# ======================================================================================
# error paths
# ======================================================================================
def test_value_error_when_target_does_not_divide_source():
    with pytest.raises(ValueError, match="does not divide"):
        FT.gather_weights(4, 3)
    with pytest.raises(ValueError, match="does not divide"):
        FT.derive_kernels(np.zeros((48, 4, 3, 3)), None, 4, 3)


def test_value_error_for_upscale_and_bad_shapes():
    with pytest.raises(ValueError, match="upscale"):
        FT.derive_kernels(np.zeros((48, 4, 3, 3)), None, 2, 4)
    with pytest.raises(ValueError, match="not a multiple"):
        FT.derive_kernels(np.zeros((47, 4, 3, 3)), None, 4, 2)
    with pytest.raises(ValueError, match="4D"):
        FT.derive_kernels(np.zeros((48, 4)), None, 4, 2)
    with pytest.raises(ValueError, match="bias"):
        FT.derive_kernels(np.zeros((48, 4, 3, 3)), np.zeros(12), 4, 2)
    with pytest.raises(ValueError, match="equal in/out"):
        FT.derive_kernels(np.zeros((48, 4, 3, 3)), None, 4, 2, num_in_ch=1)


def test_value_error_message_names_the_two_scales():
    with pytest.raises(ValueError) as excinfo:
        FT.derive_kernels(np.zeros((48, 4, 3, 3)), None, 4, 3)
    assert "4" in str(excinfo.value) and "3" in str(excinfo.value)


# ======================================================================================
# the torch path itself, without torch: torch's *signatures* over the numpy helpers
# ======================================================================================
class _NumpyFunctional:
    """``torch.nn.functional``'s surface, implemented with the numpy helpers.

    The signatures are torch's, which is the point: ``conv2d`` takes
    ``(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`` and
    has **no** ``padding_mode`` - that keyword belongs to ``nn.Conv2d``.  So a
    ``padding_mode=`` call raises ``TypeError`` here exactly as it does in torch,
    and the border has to be built with ``pad``.  Recording every call lets the
    test assert *which* border each convolution gets.
    """

    def __init__(self):
        self.pads: list[tuple[int, int, int, int, str]] = []   # replicate/constant

    def pad(self, x, pad, mode="constant", value=None):
        left, right, top, bottom = pad
        self.pads.append((left, right, top, bottom, mode))
        return np.pad(np.asarray(x), ((0, 0), (0, 0), (top, bottom), (left, right)),
                      mode="edge" if mode == "replicate" else "constant")

    def conv2d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        w, x = np.asarray(weight), np.asarray(x)
        if groups == 1:
            return FT._conv2d_numpy(x, w, bias, pad=padding, pad_mode="constant")
        # torch's grouped layout: weight is (out, in // groups, kh, kw), so
        # weight.shape[1] * groups must equal the input channel count.
        assert w.shape[1] * groups == x.shape[1], (w.shape, x.shape, groups)
        self.grouped_calls = getattr(self, "grouped_calls", []) + [w.shape[1]]
        og, ig = w.shape[0] // groups, w.shape[1]
        return np.concatenate([
            FT._conv2d_numpy(x[:, g * ig:(g + 1) * ig], w[g * og:(g + 1) * og],
                             None if bias is None else np.asarray(bias)[g * og:(g + 1) * og],
                             pad=padding, pad_mode="constant")
            for g in range(groups)
        ], axis=1)

    def pixel_shuffle(self, x, r):
        return FT._pixel_shuffle_numpy(np.asarray(x), r)


class _NumpyTorch:
    """Stand-in for the ``torch`` module: ``inference_mode`` + ``pixel_shuffle``."""

    def __init__(self):
        self.F = _NumpyFunctional()

    def inference_mode(self):
        return contextlib.nullcontext()

    def pixel_shuffle(self, x, r):
        return self.F.pixel_shuffle(x, r)


class _FakeTensor(np.ndarray):
    """Just enough of a ``torch.Tensor`` for ``apply_tail``: ``.clamp(lo, hi)``."""

    def clamp(self, lo, hi):
        return _tensor(np.clip(np.asarray(self), lo, hi))


def _tensor(x) -> _FakeTensor:
    """Wrap a numpy array as the stand-in tensor (``ndarray.view`` needs this form)."""
    return np.asarray(x).view(_FakeTensor)


def _features_body(num_feat: int = 16):
    """Token pre-tail module: widens the 3 model input channels to ``num_feat``."""

    def body(v):
        v = np.asarray(v)
        return np.tile(v, (1, -(-num_feat // v.shape[1]), 1, 1))[:, :num_feat]

    return body


def test_apply_tail_pads_with_the_functional_api(monkeypatch):
    """Pin the bug that the cell could not see: ``F.conv2d`` has no ``padding_mode``.

    ``conv_a`` is the model's own ``conv_last`` and keeps its zero padding;
    ``conv_b`` and ``down`` replicate-pad in LR space (the module docstring's
    border convention).  A ``padding_mode=`` kwarg would raise ``TypeError``
    against these signatures, so this fails loudly if the fold ever tries it.
    """
    rng = np.random.default_rng(SEED + 6)
    fused = FT.derive_kernels(*_random_case(rng)[:2], 4, 2)
    fused["body"] = (_features_body(),)                  # a token pre-tail module
    fake = _NumpyTorch()
    monkeypatch.setattr(FT, "_torch", lambda: (fake, fake.F))
    with pytest.raises(TypeError):
        fake.F.conv2d(np.zeros((1, 1, 3, 3)), np.zeros((1, 1, 3, 3)),
                      padding=1, padding_mode="replicate")   # torch's own behaviour
    FT.apply_tail(fused, _tensor(rng.random((1, 3, 8, 8))))
    assert fake.F.pads == [(1, 1, 1, 1, "replicate"), (1, 1, 1, 1, "replicate")], \
        fake.F.pads


def test_apply_tail_matches_the_numpy_mirror_through_the_torch_path(monkeypatch):
    """The torch call sequence, replayed on numpy helpers, must equal ``apply_tail_numpy``.

    Same filter order, same border, same grouped ``conv_b``, same pixel shuffle -
    so the two implementations can only agree if the torch path's conventions
    are the ones the numpy algebra was proven with.
    """
    rng = np.random.default_rng(SEED + 7)
    conv_w, conv_b, features, x = _random_case(rng)
    fused = FT.derive_kernels(conv_w, conv_b, 4, 2)
    body = (_features_body(),)
    fused["body"] = body
    fused["clamp_input"] = True
    fake = _NumpyTorch()
    monkeypatch.setattr(FT, "_torch", lambda: (fake, fake.F))

    got = FT.apply_tail(fused, _tensor(x))
    xc = np.clip(x, 0.0, 1.0)
    want = FT.apply_tail_numpy(fused, body[0](xc), xc)
    assert got.shape == want.shape == (1, 3, 32, 32)   # 16 LR px, target scale 2
    assert np.abs(got - want).max() <= 1e-12


def test_apply_tail_identity_runs_through_the_torch_path(monkeypatch):
    """``target_scale == source_scale``: radius 0, so no padding at all and a 1x1 conv_b."""
    rng = np.random.default_rng(SEED + 8)
    conv_w, conv_b, features, x = _random_case(rng, source_scale=4, target_scale=4)
    fused = FT.derive_kernels(conv_w, conv_b, 4, 4)
    assert fused["radius"] == 0
    body = (_features_body(),)
    fused["body"] = body
    fake = _NumpyTorch()
    monkeypatch.setattr(FT, "_torch", lambda: (fake, fake.F))
    got = FT.apply_tail(fused, _tensor(x))
    assert fake.F.pads == []                     # 1x1 kernels need no border
    want = FT.apply_tail_numpy(fused, body[0](x), x)
    assert np.abs(got - want).max() <= 1e-12


# ======================================================================================
# torch (skipped in the worker cell - no torch there)
# ======================================================================================
def _compact_class():
    torch = pytest.importorskip("torch")
    try:
        from src.pytorch.spandrel.architectures.Compact.__arch.SRVGG import (
            SRVGGNetCompact,
        )
    except Exception as e:  # pragma: no cover - only when spandrel's import breaks
        pytest.skip(f"spandrel Compact unavailable: {e}")
    return torch, SRVGGNetCompact


def _small_compact(seed=7, num_feat=8, upscale=4):
    torch, SRVGGNetCompact = _compact_class()
    torch.manual_seed(seed)
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat,
                          num_conv=1, upscale=upscale, act_type="relu").eval()
    return torch, net


def _numpy_reference_for_net(torch, net, x, target_scale, kernel="bicubic"):
    """net(x) at native scale, downsampled by the numpy reference."""
    with torch.inference_mode():
        hr = net(x).detach().float().cpu().numpy()
    d = net.upscale // target_scale
    return _ref_downscale(hr, d, kernel)


def test_torch_fused_tail_matches_the_downscaled_x4_output():
    torch, net = _small_compact()
    x = torch.rand(1, 3, 12, 12)
    fused = FT.fuse_tail(net, 4, 2)
    assert tuple(fused["conv_b"].shape) == (12, 48, 3, 3)
    with torch.inference_mode():
        got = FT.apply_tail(fused, x).float().cpu().numpy()
    ref = _numpy_reference_for_net(torch, net, x, 2)
    interior, border = _report("torch compact 4x->2x", *_split_error(got, ref, 2))
    assert got.shape == ref.shape == (1, 3, 24, 24)
    assert interior <= TOL


def test_torch_identity_target_matches_the_model_output():
    torch, net = _small_compact(seed=11)
    x = torch.rand(1, 3, 10, 10)
    fused = FT.fuse_tail(net, 4, 4)
    with torch.inference_mode():
        got = FT.apply_tail(fused, x)
        ref = net(x)
    assert torch.allclose(got, ref, atol=1e-6)


def test_torch_apply_tail_matches_the_numpy_mirror():
    torch, net = _small_compact(seed=13)
    x = torch.rand(1, 3, 9, 9)
    fused = FT.fuse_tail(net, 4, 2)
    with torch.inference_mode():
        got = FT.apply_tail(fused, x).float().cpu().numpy()
    features = x.clamp(0, 1)
    for m in net.body[:-1]:
        features = m(features)
    np_fused = FT.derive_kernels(
        net.body[-1].weight.detach().float().cpu().numpy(),
        net.body[-1].bias.detach().float().cpu().numpy(), 4, 2)
    ref = FT.apply_tail_numpy(np_fused, features.detach().numpy(),
                              x.clamp(0, 1).detach().numpy())
    assert np.abs(got - ref).max() <= 1e-5


def test_torch_accepts_a_descriptor_like_wrapper():
    """The app hands over ``UpscaleModelWrapper``/``ImageModelDescriptor``, not the net."""
    torch, net = _small_compact(seed=17)

    class Shim:                      # mimics ImageModelDescriptor.model
        def __init__(self, m):
            self.model = m

    fused = FT.fuse_tail(Shim(net), 4, 2)
    with torch.inference_mode():
        out = FT.apply_tail(fused, torch.rand(1, 3, 8, 8))
    assert tuple(out.shape) == (1, 3, 16, 16)


def test_torch_native_scale_mismatch_raises():
    torch, net = _small_compact(seed=19)
    with pytest.raises(ValueError, match="native scale"):
        FT.fuse_tail(net, 2, 1)


def test_torch_target_must_divide_source():
    torch, net = _small_compact(seed=23)
    with pytest.raises(ValueError, match="does not divide"):
        FT.fuse_tail(net, 4, 3)
