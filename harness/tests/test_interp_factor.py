"""A fractional `--interpolate_factor` must be refused, not rounded into a wrong file.

Backend audit 2026-11-27, 5.13: the factor multiplies the frame rate (`outputFPS = fps * f`) and divides the
interpolator's timestep (`max_timestep = (f - 1) / f` against a count-derived timestep), so a fraction has no meaning.
The old check rejected only `< 0` and `== 1`; the two consequences were a slow-motion export (`0 < f < 1`) and, on ncnn,
every frame blended toward the first (cache never advanced).
"""

from __future__ import annotations

import pytest

from src.utils.InterpFactor import whole_factor


@pytest.mark.parametrize("value,expected", [(2, 2), (2.0, 2), (4, 4), (1.0, 1), (8.0, 8)])
def test_a_whole_factor_is_accepted(value, expected):
    assert whole_factor(value) == expected


@pytest.mark.parametrize("value,neighbours", [(2.5, "2 or 3"), (1.5, "1 or 2"), (0.5, "1 or 1"), (3.25, "3 or 4")])
def test_a_fraction_is_refused_and_names_its_neighbours(value, neighbours):
    with pytest.raises(ValueError) as raised:
        whole_factor(value)
    message = str(raised.value)
    assert "whole number" in message
    assert neighbours.split(" or ")[0] in message and neighbours.split(" or ")[1] in message
