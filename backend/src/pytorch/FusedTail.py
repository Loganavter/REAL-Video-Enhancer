"""FusedTail - collapse a compact model's upscale tail so an x4 model can emit x2
(or xN) directly, instead of rendering at x4 and downscaling afterwards.

Why this exists
---------------
Every Real-ESRGAN style "compact" model (spandrel ``Compact`` / ``SRVGGNetCompact``)
ends with::

    z   = conv_last(y)                      # y = body features (1, 64, H, W) at LR res
    hr  = pixel_shuffle(z, s) + up_s(x)     # (1, 3, s*H, s*W),  s = native scale
    out = downscale(hr, s / t)              # only when the user asked for t < s

For an x4 model asked to produce a 2x video, the app currently renders the full
4320p frame and then runs a cv2 resize back down (:func:`resize_image_bytes`).
That is 4x the output pixels for every stage downstream (frame tensor, uint8
conversion, ffmpeg encode, the resize pass itself).  This module rewrites the
tail so the network composes at the *target* resolution:

    out = pixel_shuffle(down(x) + conv_b(conv_a(y)), t)

``conv_a`` is the model's own last convolution (unchanged weights), ``conv_b`` is
derived from it plus the downscale taps, ``down`` is the same taps collapsed over
the residual ``nearest`` skip.  No retraining: the tail is linear in ``y``, and
pixel_shuffle + a fixed separable downscale is a fixed linear map, so the whole
composition folds into two 3x3 convolutions.

Derivation
----------
Write ``s`` for the native scale, ``t`` for the target scale, ``d = s / t`` (an
integer, else we raise), ``C = num_out_ch``.

1. ``pixel_shuffle`` is a pure channel permutation on the HR grid
   (PyTorch's definition, quoted so the index algebra below is auditable)::

       ps[c, s*h + i, s*w + j] = z[c*s^2 + i*s + j, h, w]

   i.e. sub-pixel ``(i, j)`` of LR pixel ``(h, w)`` lives in HR row ``s*h + i``.

2. The downscale is separable and *linear*.  With ``align_corners=False`` the
   output pixel ``p`` of a ``1/d`` resize samples HR coordinate::

       u(p) = (p + 0.5) * d - 0.5

   so for the output pixel ``p = t*h' + i'`` (``i' = p mod t`` is the target
   sub-pixel phase, ``h'`` its LR row)::

       u = s*h' + u0(i'),      u0(i') = (i' + 0.5)*d - 0.5

   and the tap of HR row ``r = s*(h' + a) + i`` weighs
   ``w_v(i, a) = kern((r - u)/d)`` - one HR sample, one tap, so nothing is lost
   in the collapse::

       Gv[i'][i][a] = kern((s*a + i - u0(i')) / d)

   ``kern`` is the Keys cubic (``a = -0.75``, the coefficient torch's bicubic and
   PIL use) truncated at ``|x| <= 2``, or the box kernel for ``kernel="area"``
   (exactly what ``cv2.INTER_AREA`` does for an integer factor).  Taps are
   normalised per phase so that ``sum_{i,a} Gv[i'][i][a] = 1``; that keeps a
   flat region flat (DC gain 1) and makes the composition independent of the
   ``1/d`` gain convention.

   Because ``|s*a + i - u0| <= 2*d`` bounds ``a``, the LR support is 3x3 whenever
   ``2*d <= s`` (true for 4->2: ``a`` in {-1, 0, 1}) and 5x5 for larger factors
   (4->1, 2->1) - ``radius`` is returned so the caller knows the kernel size.

3. Applying the 2D downscale to ``ps + up_s(x)`` gives, per output channel
   ``o = c*t^2 + i'*t + j'`` (the layout ``pixel_shuffle(t)`` expects back)::

       out[o, h', w'] = sum_{i,j,a,b} Gv[i'][i][a] * Gh[j'][j][b] * z[c*s^2+i*s+j, h'+a, w'+b]
                      + sum_{a,b} (sum_i Gv[i'][i][a]) * (sum_j Gh[j'][j][b]) * x[c, h'+a, w'+b]

   The first term is ``conv_b`` applied to ``z`` and the second is ``down``
   applied to the model input: the residual skip is separable through the two
   axes, so summing the sub-pixel dimension collapses ``up_s`` to one tap per
   LR offset.  Both convolutions are plain 3x3 (5x5 for the larger factors),
   stride 1, so ``conv_b(conv_a(y))`` reproduces ``downscale(pixel_shuffle(conv_last(y)))``
   exactly - the 3x3 of ``conv_last`` is kept verbatim as ``conv_a``, so no
   kernel factorisation loss is involved.

Exactness and borders
---------------------
Interior pixels are exact (up to floating point): the two paths sum the same
taps of the same values.  At the frame border the two do *not* agree, because
they clamp different things: the reference clamps the *HR* sample index, so a
tap that falls outside the frame lands on the nearest *sub-pixel* (a different
channel of the edge LR pixel), while ``conv_b`` replicate-pads in *LR* space and
therefore reads the same sub-pixel of the edge LR pixel.  ``down`` is unaffected
(the residual skip has no sub-pixel dimension), and the deviation is confined to
the ring where a tap window leaves the frame - 2 output pixels for 4->2.  It is
measured and printed by ``backend/tests/test_fused_tail.py``; on white-noise
features it is ~4e-2, on smooth features less.

Both conventions are defensible (``F.interpolate`` clamps, ``cv2.INTER_AREA``
reflects, so no downscale has a "true" edge), which is why the fused tail does
not try to imitate either.  If an exact match to the clamped reference is ever
wanted, pad the LR feature map before ``conv_b`` with a *sub-pixel-permuted* copy
of the edge row/column (row offset ``a < 0`` must carry sub-pixel ``i = 0`` of LR
row 0, since every HR row above the frame clamps to HR row 0; columns likewise,
corners by composing the two permutations).  That makes the fold exact
everywhere for ~4 extra rows/columns of work.

Cost
----
``conv_last`` runs at LR resolution in both paths, so this is not a FLOP win in
the tail itself; the win is that the pipeline never materialises an ``s x``
frame.  See ``DONE.md`` for the 1080p / 4x native / 2x output numbers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Keys cubic coefficient used by torch's bicubic resampler and by PIL.
BICUBIC_A = -0.75

#: Kernel support in units of the downscale factor ``d`` (source pixels).
#: ``area`` is a box of width exactly ``d`` (support 1/2 in output units), i.e. the
#: disjoint d x d block mean that ``cv2.INTER_AREA`` computes for an integer factor.
KERNEL_SUPPORT = {"bicubic": 2.0, "area": 0.5}

DEFAULT_KERNEL = "bicubic"

_TAP_EPS = 1e-12


# --------------------------------------------------------------------------------------
# 1D taps
# --------------------------------------------------------------------------------------
def _keys_cubic(t: np.ndarray, a: float) -> np.ndarray:
    """Keys cubic interpolation kernel, ``a`` = -0.75 reproduces torch/PIL."""
    t = np.abs(np.asarray(t, dtype=np.float64))
    t2 = t * t
    t3 = t2 * t
    near = (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0
    far = a * t3 - 5.0 * a * t2 + 8.0 * a * t - 4.0 * a
    return np.where(t <= 1.0, near, np.where(t < 2.0, far, 0.0))


def _kernel(name: str, t: np.ndarray, a: float) -> np.ndarray:
    if name == "bicubic":
        return _keys_cubic(t, a)
    if name == "area":  # box of width d: |r - u| <= d/2, the d x d block mean
        return (np.abs(np.asarray(t, dtype=np.float64)) <= 0.5 + 1e-9).astype(np.float64)
    raise ValueError(f"unknown downscale kernel {name!r}; expected 'bicubic' or 'area'")


def gather_weights(
    source_scale: int,
    target_scale: int,
    kernel: str = DEFAULT_KERNEL,
    a: float = BICUBIC_A,
) -> tuple[np.ndarray, int]:
    """Per-axis gather weights of one integer downscale step, in LR-relative units.

    Returns ``(g, radius)`` where ``g`` has shape ``(target_scale, source_scale,
    2*radius + 1)`` and::

        g[i', i, a + radius] = weight of the sample at LR offset ``a``
                               sub-pixel ``i``, for target phase ``i'``

    Each phase is normalised so ``g[i'].sum() == 1``.  ``radius == 0`` means no
    neighbour LR pixel is needed: the delta at ``target_scale == source_scale``,
    and the ``d x d`` box of ``kernel="area"`` (which never straddles an LR pixel).
    """
    s, t = int(source_scale), int(target_scale)
    if s < 1 or t < 1:
        raise ValueError(f"scales must be >= 1, got source={source_scale} target={target_scale}")
    if s % t:
        raise ValueError(
            f"target_scale {t} does not divide source_scale {s}: cannot fuse exactly"
        )
    d = s // t
    support = KERNEL_SUPPORT[kernel] * d
    half = int(math.ceil(support))

    raw: list[dict[tuple[int, int], float]] = []
    radius = 0
    for phase in range(t):
        u = (phase + 0.5) * d - 0.5
        origin = int(math.floor(u))
        acc: dict[tuple[int, int], float] = {}
        total = 0.0
        for j in range(origin - half, origin + half + 1):
            rel = j - u
            if abs(rel) > support + 1e-9:
                continue
            w = float(_kernel(kernel, np.array([rel / d]), a)[0])
            if abs(w) <= _TAP_EPS:
                continue
            off, sub = divmod(j, s)  # HR offset j -> LR offset `off`, sub-pixel `sub`
            assert 0 <= sub < s
            acc[(sub, off)] = acc.get((sub, off), 0.0) + w
            total += w
            radius = max(radius, abs(off))
        if total == 0.0:
            raise RuntimeError(f"degenerate tap set for phase {phase}")  # pragma: no cover
        raw.append({k: v / total for k, v in acc.items()})

    g = np.zeros((t, s, 2 * radius + 1), dtype=np.float64)
    for phase, taps in enumerate(raw):
        for (sub, off), w in taps.items():
            g[phase, sub, off + radius] = w
    return g, radius


def subpixel_mass(g: np.ndarray) -> np.ndarray:
    """Sum the sub-pixel dimension: ``down`` weights, shape ``(target_scale, k)``."""
    return g.sum(axis=1)


# --------------------------------------------------------------------------------------
# kernel assembly (backend agnostic, numpy)
# --------------------------------------------------------------------------------------
def derive_kernels(
    conv_weight: Any,
    conv_bias: Any = None,
    source_scale: int = 4,
    target_scale: int = 2,
    *,
    num_in_ch: int | None = None,
    kernel: str = DEFAULT_KERNEL,
    a: float = BICUBIC_A,
    skip: bool = True,
) -> dict[str, Any]:
    """Derive the fused tail weights from a compact model's last convolution.

    ``conv_weight`` has shape ``(num_out_ch * source_scale**2, num_feat, k, k)``
    (``conv_last`` of an ``SRVGGNetCompact``); ``conv_bias`` is its bias or
    ``None``.  Returns a dict of numpy arrays (float64):

    ``conv_a``       ``(num_feat -> num_out_ch*s^2, kh, kw)`` - identical to ``conv_last``
    ``conv_a_bias``  ``(num_out_ch*s^2,)`` or ``None``
    ``conv_b``       ``(num_out_ch*t^2, num_out_ch*s^2, 2r+1, 2r+1)``
    ``down``         ``(num_out_ch*t^2, num_in_ch, 2r+1, 2r+1)`` or ``None``
    plus ``source_scale``, ``target_scale``, ``factor`` (``s // t``), ``radius`` and
    ``kernel``.  ``conv_b``/``down`` are indexed ``[out_ch, in_ch]`` like every
    other torch convolution here, so they can be handed straight to
    :func:`apply_tail_numpy` or ``F.conv2d``.

    ``skip=False`` drops the nearest-upsampled residual (a pure SR tail) and
    returns ``down=None``.
    """
    w = np.asarray(conv_weight, dtype=np.float64)
    if w.ndim != 4:
        raise ValueError(f"conv_weight must be 4D (out,in,kh,kw), got shape {w.shape}")
    s, t = int(source_scale), int(target_scale)
    if t > s:
        raise ValueError(f"target_scale {t} > source_scale {s}: fusing an upscale is pointless")
    if s % t:
        raise ValueError(f"target_scale {t} does not divide source_scale {s}: cannot fuse exactly")

    n_out_ch_total = w.shape[0]
    if n_out_ch_total % (s * s):
        raise ValueError(
            f"conv_weight has {n_out_ch_total} output channels, not a multiple of "
            f"source_scale**2 = {s * s}: not a compact tail"
        )
    num_out_ch = n_out_ch_total // (s * s)
    num_feat = w.shape[1]
    if num_in_ch is None:
        num_in_ch = num_out_ch
    if skip and int(num_in_ch) != num_out_ch:
        raise ValueError(
            f"nearest residual needs equal in/out channels, got in={num_in_ch} out={num_out_ch}"
        )

    bias = None if conv_bias is None else np.asarray(conv_bias, dtype=np.float64).reshape(-1)
    if bias is not None and bias.shape[0] != n_out_ch_total:
        raise ValueError(f"bias has {bias.shape[0]} entries, conv has {n_out_ch_total} rows")

    g, radius = gather_weights(s, t, kernel=kernel, a=a)
    k = 2 * radius + 1

    conv_b = np.zeros((num_out_ch * t * t, n_out_ch_total, k, k), dtype=np.float64)
    for c in range(num_out_ch):
        for i2 in range(t):
            for j2 in range(t):
                o = c * t * t + i2 * t + j2
                for i in range(s):
                    for j in range(s):
                        m = c * s * s + i * s + j
                        conv_b[o, m] = np.outer(g[i2, i], g[j2, j])

    if skip:
        mass = subpixel_mass(g)
        down = np.zeros((num_out_ch * t * t, num_in_ch, k, k), dtype=np.float64)
        for c in range(num_out_ch):
            for i2 in range(t):
                for j2 in range(t):
                    down[c * t * t + i2 * t + j2, c] = np.outer(mass[i2], mass[j2])
    else:
        down = None

    return {
        "conv_a": w,
        "conv_a_bias": bias,
        "conv_b": conv_b,
        "down": down,
        "source_scale": s,
        "target_scale": t,
        "factor": s // t,
        "radius": radius,
        "kernel": kernel,
        "num_feat": num_feat,
        "num_out_ch": num_out_ch,
    }


def nonzeros(fused: dict[str, Any]) -> dict[str, int]:
    """Report how sparse the derived kernels are (``conv_b`` is block diagonal)."""
    out = {}
    for key in ("conv_b", "down"):
        arr = fused.get(key)
        out[key] = 0 if arr is None else int(np.count_nonzero(arr))
    return out


# --------------------------------------------------------------------------------------
# numpy reference implementation (CPU-testable, no torch)
# --------------------------------------------------------------------------------------
def _conv2d_numpy(x: np.ndarray, w: np.ndarray, bias: np.ndarray | None, pad: int,
                  pad_mode: str) -> np.ndarray:
    """Stride-1 cross-correlation, torch semantics, ``w`` shaped ``(out, in, kh, kw)``."""
    if pad:
        if pad_mode == "replicate":
            xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
        else:
            xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    else:
        xp = x
    kh, kw = w.shape[2], w.shape[3]
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))
    out = np.einsum("nchwuv,ocuv->nohw", win, w, optimize=True)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


def _pixel_shuffle_numpy(x: np.ndarray, r: int) -> np.ndarray:
    """PyTorch ``PixelShuffle``: ``out[c, h*r+i, w*r+j] == x[c*r^2 + i*r + j, h, w]``."""
    n, c, h, w = x.shape
    if c % (r * r):
        raise ValueError(f"{c} channels cannot be pixel-shuffled by {r}")
    out = x.reshape(n, c // (r * r), r, r, h, w)
    out = out.transpose(0, 1, 4, 2, 5, 3)
    return out.reshape(n, c // (r * r), h * r, w * r)


def apply_tail_numpy(fused: dict[str, Any], y: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    """numpy mirror of :func:`apply_tail`.

    ``y`` is the body feature map ``(1, num_feat, H, W)``; ``x`` the model input
    ``(1, num_in_ch, H, W)`` needed by the residual skip (already clamped, see
    :func:`apply_tail`).  Uses whichever (numpy or torch) arrays ``fused`` holds
    for the weights - convert them to numpy first for the pure-numpy path.
    """
    conv_a = _as_numpy(fused["conv_a"])
    conv_a_bias = fused.get("conv_a_bias")
    conv_a_bias = None if conv_a_bias is None else _as_numpy(conv_a_bias)
    conv_b = _as_numpy(fused["conv_b"])
    down = fused.get("down")
    down = None if down is None else _as_numpy(down)
    radius = int(fused["radius"])
    t = int(fused["target_scale"])

    y = np.asarray(y, dtype=np.float64)
    z = _conv2d_numpy(y, conv_a, conv_a_bias, pad=conv_a.shape[2] // 2, pad_mode="constant")
    acc = _conv2d_numpy(z, conv_b, None, pad=radius, pad_mode="replicate")
    if down is not None:
        if x is None:
            raise ValueError("fused tail has a residual skip: pass the (clamped) model input x")
        acc = acc + _conv2d_numpy(np.asarray(x, dtype=np.float64), down, None,
                                  pad=radius, pad_mode="replicate")
    return _pixel_shuffle_numpy(acc, t)


def _as_numpy(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "detach"):  # torch tensor
        return a.detach().float().cpu().numpy()
    return np.asarray(a)


# --------------------------------------------------------------------------------------
# torch API
# --------------------------------------------------------------------------------------
def _torch():
    try:
        import torch  # noqa: F401
        import torch.nn.functional as F  # noqa: F401
    except ImportError as e:  # pragma: no cover - the cell ships without torch
        raise RuntimeError(
            "FusedTail.fuse_tail/apply_tail need torch; the numpy path "
            "(derive_kernels/apply_tail_numpy) does not"
        ) from e
    import torch
    import torch.nn.functional as F

    return torch, F


def _unwrap_model(model: Any):
    """Peel spandrel descriptors / the app's wrapper down to the nn.Module."""
    torch, _ = _torch()
    net = model
    for _ in range(4):
        if isinstance(net, torch.nn.Module) and not hasattr(net, "get_model"):
            break
        getter = getattr(net, "get_model", None)
        if callable(getter):
            net = getter()
            continue
        inner = getattr(net, "model", None)
        if isinstance(inner, torch.nn.Module):
            net = inner
            continue
        raise ValueError(f"cannot find an nn.Module inside {type(model).__name__}")
    return net


def _find_tail(net: Any):
    """Locate ``(pixel_shuffle_module, conv_last)`` inside a compact model."""
    torch, _ = _torch()
    modules = list(net.named_modules())
    ps_index = None
    ps = None
    for idx, (_, m) in enumerate(modules):
        if isinstance(m, torch.nn.PixelShuffle):
            ps_index, ps = idx, m
    if ps is None:
        raise ValueError(f"{type(net).__name__} has no PixelShuffle: not a compact tail")
    conv = None
    for _, m in modules[:ps_index]:
        if isinstance(m, torch.nn.Conv2d):
            conv = m
    if conv is None:
        raise ValueError(f"no Conv2d before the PixelShuffle in {type(net).__name__}")
    return ps, conv


def fuse_tail(
    model: Any,
    source_scale: int,
    target_scale: int,
    *,
    kernel: str = DEFAULT_KERNEL,
    a: float = BICUBIC_A,
    skip: bool = True,
    clamp_input: bool = True,
) -> dict[str, Any]:
    """Return an equivalent-but-smaller tail for ``model``.

    Returns a dict with ``conv_a`` (``(num_feat -> num_out_ch*s^2, k, k)``),
    ``conv_b``, ``down`` and the bookkeeping keys of :func:`derive_kernels`, all
    as torch tensors on the model's device/dtype, plus ``model``/``body``/
    ``clamp_input`` so :func:`apply_tail` can run the shared prefix.

    Raises ``ValueError`` if ``target_scale`` does not divide ``source_scale``,
    if the model's native scale differs from ``source_scale``, or if the model
    has no compact tail.
    """
    torch, _ = _torch()
    net = _unwrap_model(model)
    ps, conv_last = _find_tail(net)
    native = int(ps.upscale_factor)
    if native != int(source_scale):
        raise ValueError(
            f"model native scale is {native}, not the given source_scale {source_scale}"
        )

    w = conv_last.weight.detach()
    bias = None if conv_last.bias is None else conv_last.bias.detach()
    body = getattr(net, "body", None)
    num_in_ch = conv_last.in_channels
    if body is not None and len(body) > 0:
        first = body[0]
        if isinstance(first, torch.nn.Conv2d):
            num_in_ch = first.in_channels

    np_fused = derive_kernels(
        w.detach().float().cpu().numpy(),
        None if bias is None else bias.detach().float().cpu().numpy(),
        source_scale,
        target_scale,
        num_in_ch=num_in_ch,
        kernel=kernel,
        a=a,
        skip=skip,
    )

    def to_t(arr):
        return None if arr is None else torch.as_tensor(
            arr, dtype=w.dtype, device=w.device
        ).contiguous()

    _assert_block_diagonal(np_fused["conv_b"], np_fused["num_out_ch"], source_scale,
                           target_scale)
    fused: dict[str, Any] = dict(np_fused)
    fused["conv_a"] = conv_last.weight
    fused["conv_a_bias"] = conv_last.bias
    fused["conv_b"] = to_t(np_fused["conv_b"])
    fused["down"] = to_t(np_fused["down"])
    fused["model"] = net
    fused["body"] = tuple(_body_modules(net, conv_last))
    fused["clamp_input"] = bool(clamp_input)
    return fused


def _assert_block_diagonal(conv_b: np.ndarray, num_out_ch: int, source_scale: int,
                           target_scale: int) -> None:
    """``apply_tail`` folds ``conv_b`` as a grouped convolution - prove the blocks are clean."""
    if conv_b.shape[0] != num_out_ch * target_scale**2:
        raise RuntimeError(
            f"conv_b has {conv_b.shape[0]} output channels, expected "
            f"{num_out_ch}*{target_scale}^2"
        )
    out_block = np.arange(conv_b.shape[0]) // (target_scale**2)
    in_block = np.arange(conv_b.shape[1]) // (source_scale**2)
    off_block = conv_b[out_block[:, None] != in_block[None, :]]
    if off_block.size and np.any(off_block != 0.0):  # pragma: no cover - construction
        raise RuntimeError(
            "derived conv_b is not block diagonal over output colours: the grouped "
            "convolution path in apply_tail would be wrong"
        )


def _body_modules(net: Any, conv_last: Any) -> list:
    """Modules that run between the model input and ``conv_last`` (compact layout)."""
    body = getattr(net, "body", None)
    if body is None:
        raise ValueError(
            f"{type(net).__name__} exposes no `.body`: cannot run the pre-tail prefix"
        )
    mods = list(body)
    while mods and mods[-1] is conv_last:
        mods.pop()
    if not mods:
        raise ValueError("the model body is empty before conv_last")
    return mods


def _tail_conv(F, x, weight, bias, radius: int, groups: int = 1,
               pad_mode: str = "replicate"):
    """Stride-1 convolution with the same border convention as :func:`_conv2d_numpy`.

    ``pad_mode="replicate"`` is what ``conv_b``/``down`` need (the fold reads the
    same sub-pixel of the edge LR pixel), ``"constant"`` is the model's own
    zero padding for ``conv_a``.  ``F.conv2d`` only knows zero padding -
    ``padding_mode=`` is a keyword of ``nn.Conv2d``, not of the functional API -
    so the replicate border is materialised with :func:`torch.nn.functional.pad`
    and the convolution then runs unpadded.
    """
    kwargs = {"groups": groups} if groups > 1 else {}
    if radius <= 0:
        return F.conv2d(x, weight, bias, **kwargs)
    if pad_mode == "replicate":
        x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
        return F.conv2d(x, weight, bias, **kwargs)
    if pad_mode != "constant":
        raise ValueError(f"unknown pad_mode {pad_mode!r}; expected 'replicate' or 'constant'")
    return F.conv2d(x, weight, bias, padding=radius, **kwargs)


def _grouped_conv_b(fused: dict[str, Any], conv_b: Any, num_groups: int,
                    source_scale: int, target_scale: int):
    """Slice the block-diagonal ``conv_b`` into a ``groups=num_groups`` conv weight.

    ``F.conv2d`` with ``groups=C`` wants ``(C*t^2, s^2, kh, kw)`` - the *per-group*
    input channels, not the full ``C*s^2`` - so the dense tensor is cut along the
    colour diagonal: group ``c`` keeps rows ``[c*t^2, (c+1)*t^2)`` and columns
    ``[c*s^2, (c+1)*s^2)``.  (Passing the dense ``(C*t^2, C*s^2, kh, kw)`` with
    ``groups=C`` is rejected: torch checks ``weight.size(1) * groups == in``.)

    The result is cached on ``fused`` - it only depends on the weights, so
    rebuilding it for every frame would be pure overhead.
    """
    cached = fused.get("conv_b_grouped")
    if cached is not None:
        return cached
    t2, s2 = int(target_scale) ** 2, int(source_scale) ** 2
    blocks = [conv_b[g * t2:(g + 1) * t2, g * s2:(g + 1) * s2] for g in range(num_groups)]
    if hasattr(conv_b, "detach"):                      # torch tensor
        torch, _ = _torch()
        grouped = torch.cat(blocks, dim=0).contiguous()
    else:
        grouped = np.concatenate(blocks, axis=0)
    fused["conv_b_grouped"] = grouped
    return grouped


def apply_tail(fused: dict[str, Any], x: Any) -> Any:
    """Run the fused tail: ``x`` is the model input, output is ``target_scale`` larger.

    The pre-tail body is shared with the loaded model (``fused['body']``), so the
    only extra work versus the original tail is ``conv_b`` at LR resolution plus
    the ``down`` skip convolution - and everything after this point runs at
    ``target_scale`` instead of ``source_scale``.

    ``conv_b`` is block diagonal by construction (output sub-pixel ``(i', j')`` of
    colour ``c`` only reads the ``s^2`` sub-pixels of ``c``), so it runs as a
    ``groups=num_out_ch`` convolution over the colour-diagonal slices (see
    :func:`_grouped_conv_b`): three times fewer multiply-adds than the dense
    tensor and bit-comparable to it.
    """
    torch, F = _torch()
    radius = int(fused["radius"])
    t = int(fused["target_scale"])
    groups = int(fused.get("num_out_ch", 1))
    with torch.inference_mode():
        xc = x.clamp(0.0, 1.0) if fused.get("clamp_input", True) else x
        y = xc
        for m in fused["body"]:
            y = m(y)
        # conv_a is the model's own conv_last: zero padding, never replicate
        z = _tail_conv(F, y, fused["conv_a"], fused.get("conv_a_bias"),
                       conv_a_pad(fused), pad_mode="constant")
        conv_b = fused["conv_b"]
        num_groups = groups
        if (num_groups > 1
                and conv_b.shape[0] == num_groups * t * t
                and conv_b.shape[1] == num_groups * int(fused["source_scale"]) ** 2):
            weight_b = _grouped_conv_b(fused, conv_b, num_groups,
                                       int(fused["source_scale"]), t)
            groups_used = num_groups
        else:                                          # not block diagonal: dense conv
            weight_b = conv_b
            groups_used = 1
        acc = _tail_conv(F, z, weight_b, None, radius, groups_used)
        if fused.get("down") is not None:
            acc = acc + _tail_conv(F, xc, fused["down"], None, radius)
        return F.pixel_shuffle(acc, t)


def conv_a_pad(fused: dict[str, Any]) -> int:
    """Zero padding radius of ``conv_a`` (the model's own conv_last padding)."""
    kh = int(fused["conv_a"].shape[2])
    return kh // 2


def tail_params(fused: dict[str, Any]) -> dict[str, int]:
    """MACs per LR pixel of the fused tail, and the derived kernels' nonzero count."""
    k = int(fused["conv_b"].shape[2])
    r = int(fused["conv_b"].shape[1])          # conv_b input channels (= C*s^2)
    o = int(fused["conv_b"].shape[0])          # conv_b output channels (= C*t^2)
    conv_a = fused["conv_a"]
    mac_a = int(conv_a.shape[1]) * int(conv_a.shape[0]) * int(conv_a.shape[2]) * int(conv_a.shape[3])
    down = fused.get("down")
    mac_b_nnz, mac_down_nnz = 0, 0
    conv_b = fused["conv_b"]
    if hasattr(conv_b, "detach"):
        mac_b_nnz = int(torch_count_nonzero(conv_b))
    else:
        mac_b_nnz = int(np.count_nonzero(conv_b))
    if down is not None:
        if hasattr(down, "detach"):
            mac_down_nnz = int(torch_count_nonzero(down))
        else:
            mac_down_nnz = int(np.count_nonzero(down))
    groups = int(fused.get("num_out_ch", 1)) or 1
    grouped = (o * r * k * k) // groups if o % groups == 0 and r % groups == 0 else o * r * k * k
    return {
        "conv_a_macs": mac_a,
        "conv_b_macs": o * r * k * k,
        "conv_b_macs_grouped": grouped,
        "conv_b_macs_sparse": mac_b_nnz,
        "down_macs": 0 if down is None else int(
            down.shape[0] * down.shape[1] * down.shape[2] * down.shape[3]
        ),
        "down_macs_sparse": mac_down_nnz,
    }


def torch_count_nonzero(t) -> int:
    return int((t != 0).sum().item())


__all__ = [
    "BICUBIC_A",
    "KERNEL_SUPPORT",
    "DEFAULT_KERNEL",
    "derive_kernels",
    "gather_weights",
    "subpixel_mass",
    "nonzeros",
    "fuse_tail",
    "apply_tail",
    "apply_tail_numpy",
    "tail_params",
]
