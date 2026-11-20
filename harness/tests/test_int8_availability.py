"""The Fast preset's availability rule: int8 tensor cores, i.e. compute capability 7.5 and up.

The body is Triton kernels that accumulate in int32 (`tl.dot(..., out_dtype=tl.int32)`), and Triton has no integer
tensor-core instruction to lower that to below Turing. So on an older card Fast is not a slow path, it is an absent
one - and the interface hides it rather than offering a switch that cannot do anything.

Unknown capability deliberately does not hide it: on a machine where neither nvidia-smi nor the backend's torch can
answer, the probe still decides at render time, and hiding a working configuration is worse than offering one that
may fall back to float16.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

pytest.importorskip("PySide6")
pytest.importorskip("requests")

from src.ui.GPUDetect import INT8_MIN_CAPABILITY, int8_decision  # noqa: E402


def test_turing_ampere_ada_and_hopper_are_supported():
    for capability in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0)):
        available, reason = int8_decision("Nvidia", capability)
        assert available, (capability, reason)


def test_volta_and_older_do_not_have_the_instruction():
    for capability in ((7, 0), (6, 1), (5, 2)):
        available, reason = int8_decision("Nvidia", capability)
        assert not available
        assert "tensor cores" in reason


def test_a_non_nvidia_card_has_no_cuda_path_at_all():
    available, reason = int8_decision("AMD", None)
    assert not available
    assert "AMD" in reason


def test_unknown_capability_keeps_the_option():
    available, reason = int8_decision("Nvidia", None)
    assert available
    assert "unknown" in reason


def test_the_threshold_is_the_one_the_kernels_need():
    # Turing is the first architecture with int8 tensor cores; the kernels are written for that instruction.
    assert INT8_MIN_CAPABILITY == (7, 5)
