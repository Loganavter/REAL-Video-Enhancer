"""The interpolation factor is a COUNT of frames, not a rate.

Backend audit 2026-11-27 (5.13): `checkArguments` rejected only `< 0` and `== 1`, so a fraction slipped between the
integers. For `0 < f < 1` the ceiled count produced no interpolated frame while `outputFPS = fps * f` slowed the whole
file down; for a fractional `f > 1` on ncnn the cache never advanced, because the timestep was computed from the ceiled
count (`InterpolateNCNN.py:195`) while `max_timestep` came from the raw fraction (`RenderVideo.py:266`), so every output
frame was a blend toward the first. A refusal at the door beats a silently wrong file.
"""

from __future__ import annotations


def whole_factor(value: float) -> int:
    """`value` as an int when it is a whole number; `ValueError` naming the nearest whole factors otherwise.

    The factor multiplies the frame rate AND divides the interpolator's timestep, so it has to be a frame count. The
    message names the two integers a caller can choose, because "must be an integer" alone leaves them guessing.
    """
    number = float(value)
    if number != int(number):
        lower = int(number)
        raise ValueError(
            f"interpolate_factor must be a whole number of frames (2, 3, 4...), got {number:g}: it multiplies the "
            f"frame rate and divides the interpolator's timestep, so a fraction cannot be honoured. "
            f"Use {max(1, lower)} or {lower + 1}."
        )
    return int(number)
