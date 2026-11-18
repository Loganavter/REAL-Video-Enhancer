"""The Best/Fast switch is the int8 switch: one rule, pinned.

The window's combo box used to feed the frame-reuse gate's fallback behaviour. The gate is off by default
since 2026-11-18, so the switch was re-pointed at the body's precision - best runs the fp16 engine, fast
runs the int8 body, and the tooltip says the picture may move either way because that is what was measured.
These tests pin the mapping and the precedence, because a silent swap here would change what every render
computes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# RenderVideo pulls torch/tensorrt; the resolver itself is plain python, so import it lazily and skip when
# the interpreter cannot load the module at all (a bare CPU python without torch).
RV = pytest.importorskip("src.RenderVideo")


def test_best_runs_the_fp16_engine():
    assert RV.resolve_int8_fit(None, "best") == "fp16"


def test_fast_runs_the_int8_body():
    assert RV.resolve_int8_fit(None, "fast") == "int8"


def test_an_explicit_fit_wins_over_the_switch():
    """Every measurement run passes --int8_fit explicitly; the preset must not overwrite it."""
    assert RV.resolve_int8_fit("fp16", "fast") == "fp16"
    assert RV.resolve_int8_fit("int8", "best") == "int8"
    assert RV.resolve_int8_fit("auto", "best") == "auto"


def test_unknown_values_are_refused_loudly():
    with pytest.raises(ValueError):
        RV.resolve_int8_fit(None, "medium")
    with pytest.raises(ValueError):
        RV.resolve_int8_fit("int4", "best")
