"""The correction that makes a reused frame usable: frequency split, recursive base.

The gate decides *whether* a frame can be served from the cache; without a correction the output is the reference's
picture verbatim, which loses the parts of the frame that changed (37 of 130 reuses over the 0.1% bar on the 1040 s
calibration cut). The measured fix is algebraic, not geometric:

    r      = LR_t - LR_{t-1}
    r_lf   = box5(r)                       the tone / displacement part, added at gain 1
    r_hf   = r - r_lf                      the grain part, added at gain alpha on flat static blocks
    out    = out_{t-1} + nearest_s(r_lf + alpha_g * r_hf)

Recursive (`out_{t-1}` is the previous EMITTED frame, not the cached model output) so a chain of reuses does not
jump k frames back. Measured in `docs/wsp/verified/reuse-decision-statistic.txt`: per channel, alpha 1.35 with the
mask, 0/130 over the bar at texture 1.001 on the calibration clip and 3/124 at 1.021 on the held-out one, for
2.428 ms per reused frame on this GPU (`harness/scripts/reuse_correction_cost.py`).

The scale `s` is read from the seeded pair, not assumed. 2x was the measured configuration, but 4x is the common
model scale, and a corrector that only accepted `hr == 2*lr` refused exactly the configuration most users run: the
gate would reuse a frame and emit the cached picture verbatim, which is the damage this file exists to remove.
Nothing in the frequency split depends on the scale until the residual is upsampled, so only that step uses `s`;
`s == 2` must stay bit-identical to the numbers above.

Not covered: non-integer scales (1.5x) and anamorphic pairs whose horizontal and vertical factors differ are
refused, so those renders keep the verbatim fallback. The algebra at 3x/4x is pinned by the unit tests only; no
render has been measured at those scales.

Two alternatives are measured and refused, so they are not implemented here. Warping the cached frame by a motion
field: the codec's own vectors reference a picture that is not the one the gate holds, and a DIS flow from the held
frame fails on the same frames (`docs/wsp/verified/reuse-mv-reference.txt`). Clamping the result into the local
colour range: it does not move the failures and costs 35x on frames that pass
(`docs/wsp/PRIOR-ART-reuse-detail-2026-11-20.md`).

The failure this correction does NOT fix, and which is why the gate refuses dense moving line art on its own:
doubled line art at 171-189 levels, 8 of 8 reuses on a segment where the picture moves everywhere
(`docs/wsp/verified/reuse-motion-content.txt`).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ReferenceFrameCorrector:
    """Stateful: keeps the previous emitted frame and its LR frame, both on the GPU.

    `seed()` is called after every frame that ran the model; `correct()` after every frame the gate reuses. Both
    take the tensors the upscyler already has, so no extra host round trip is involved.
    """

    def __init__(self, alpha: float = 1.35, blur: int = 5,
                 mask_tau: float = 3.0, mask_grad: float = 5.0) -> None:
        self.alpha = float(alpha)
        self.blur = int(blur) if int(blur) % 2 == 1 else int(blur) + 1
        # thresholds are given in 8-bit levels and converted to the tensor's 0..1 scale
        self.mask_tau = float(mask_tau) / 255.0
        self.mask_grad = float(mask_grad) / 255.0
        self._lr = None
        self._hr = None
        self.applied = 0
        self.refused = 0

    # -- state ---------------------------------------------------------------------------
    def reset(self) -> None:
        self._lr = None
        self._hr = None

    def seed(self, hr_tensor: torch.Tensor, lr_tensor: torch.Tensor) -> None:
        """After a frame that ran the model: this pair becomes the next reuse's base."""
        self._hr = hr_tensor
        self._lr = lr_tensor

    # -- correction ----------------------------------------------------------------------
    @torch.inference_mode()
    def correct(self, lr_tensor: torch.Tensor) -> torch.Tensor | None:
        """Return the corrected frame, or None when no correction can be made (caller emits verbatim).

        The upscale factor comes from the pair's shapes, so 2x, 3x and 4x models all get the correction; a
        non-integer or anamorphic pair cannot be corrected by a single nearest upsample and is refused.
        """
        if self._lr is None or self._hr is None:
            self.refused += 1
            return None
        if tuple(lr_tensor.shape) != tuple(self._lr.shape):
            # resolution change, a seek, or a different input path: the pair is not comparable
            self.refused += 1
            return None
        h, w = lr_tensor.shape[-2], lr_tensor.shape[-1]
        if h <= 0 or w <= 0:
            self.refused += 1
            return None
        s_h = self._hr.shape[-2] // h
        s_w = self._hr.shape[-1] // w
        # same factor on both axes (an anamorphic pair needs two different upsamplers, not this one),
        # integer (the floor division must reproduce the shape exactly), and at least 2x
        if s_h != s_w or s_h < 2 or self._hr.shape[-2] != s_h * h or self._hr.shape[-1] != s_w * w:
            self.refused += 1
            return None
        r = lr_tensor - self._lr
        pad = self.blur // 2
        lf = F.avg_pool2d(r, self.blur, stride=1, padding=pad)
        hf = r - lf
        loc = F.avg_pool2d(r.abs(), self.blur, stride=1, padding=pad).mean(dim=1, keepdim=True)
        grey = lr_tensor.mean(dim=1, keepdim=True)
        grad = (grey.diff(dim=2, prepend=grey[:, :, :1]).abs() +
                grey.diff(dim=3, prepend=grey[:, :, :, :1]).abs())
        keep = ((loc < self.mask_tau) & (grad < self.mask_grad)).to(r.dtype)
        corr = lf + (1.0 + keep * (self.alpha - 1.0)) * hf
        up = F.interpolate(corr, scale_factor=s_h, mode="nearest")
        out = (self._hr + up).clamp_(0.0, 1.0)
        self._hr = out
        self._lr = lr_tensor
        self.applied += 1
        return out
