"""The Compact (Real-ESRGAN / SRVGGNetCompact) block of the engine, in one place.

What belongs here and why it is one unit: the int8 body is only correct for a Compact architecture
(64 body channels, 18 convs), its kernels are written for that body, and the fused tail folds the
tail of the *same* architecture so an x4 model can emit x2 without rendering 4320p. The vendored
model definition stays in `spandrel/architectures/Compact` - it is third-party code with its own
LICENSE and it moves with spandrel - and is re-exported here so callers have one import.

    from .compact import Int8Body, first_frame_sanity, CompactArch

The old paths (`pytorch.Int8Body`, `pytorch.Int8Kernels`, `pytorch.FusedTail`) are kept as
one-line re-export shims: `src.pytorch.Int8Body`, `scripts`, the probe and the tests import them by
name and there is no reason to break a working import to prove a directory moved.
"""
from __future__ import annotations

from .arch import CompactArch, SRVGGNetCompact
from .fused_tail import (derive_kernels, gather_weights, nonzeros, subpixel_mass)
from .int8_body import FIRST_FRAME_SANITY as first_frame_sanity
from .int8_body import Int8Body, calib_crops

__all__ = [
    "CompactArch",
    "SRVGGNetCompact",
    "Int8Body",
    "calib_crops",
    "first_frame_sanity",
    "gather_weights",
    "subpixel_mass",
    "derive_kernels",
    "nonzeros",
]
