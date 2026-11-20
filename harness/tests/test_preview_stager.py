"""The preview stager: one thumbnail in flight, handed over on the next frame.

The synchronous path could not hold a per-frame maximum on this card (0.5 ms median,
1.1-2.2 ms tail, and the tail did not follow the byte count), so the loop enqueues and
publishes what the previous frame left in pinned memory. The properties that matter:

* `submit` never blocks and never overwrites a thumbnail still in flight;
* `take` returns nothing until the device copy completed, then exactly the payload;
* the payload is the frame that was submitted, not a later one.

CUDA-only: on a CPU device the copy is synchronous and the ordering is trivial, so there
is nothing here worth asserting.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from src.utils.Frame import Frame  # noqa: E402
from src.utils.PreviewChannel import FMT_YUV420  # noqa: E402
from src.utils.PreviewStager import PreviewStager  # noqa: E402


def payload_frame(value: int, width: int = 8, height: int = 6) -> Frame:
    n = width * height
    c = (height // 2) * (width // 2)
    frame = Frame(backend="pytorch", width=width, height=height, device="cuda", gpu_id=0,
                  hdr_mode=False, dtype="float32")
    frame._tensor = torch.full((n + 2 * c,), value, dtype=torch.uint8, device="cuda")
    return frame


def drain(stager, timeout=2.0):
    """Return the payload once the device copy lands (bounded, so a hang is a failure)."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = stager.take()
        if got is not None:
            return got
        torch.cuda.synchronize()
        time.sleep(0.002)
    raise AssertionError("the staged thumbnail never arrived")


def test_take_is_empty_until_the_copy_lands():
    stager = PreviewStager(8, 6)
    assert not stager.busy()
    assert stager.take() is None, "nothing was submitted yet"
    assert stager.submit(payload_frame(128))
    assert stager.busy(), "one in flight, so the next submit must be refused"
    assert not stager.submit(payload_frame(200)), "a second thumbnail would overwrite the first"


def test_the_payload_belongs_to_the_submitted_frame():
    stager = PreviewStager(8, 6)
    stager.submit(payload_frame(128))
    payload, fmt, high_bitdepth = drain(stager)
    assert fmt == FMT_YUV420 and high_bitdepth is False
    assert len(payload) == 72, "8x6 Y plus 4x3 U and V"
    levels = np.frombuffer(payload, np.uint8)
    assert levels.min() == 128 and levels.max() == 128


def test_a_finished_thumbnail_frees_the_slot_for_the_next_frame():
    stager = PreviewStager(8, 6)
    stager.submit(payload_frame(128))
    drain(stager)
    assert not stager.busy()
    assert stager.submit(payload_frame(200))
    payload, _fmt, _flags = drain(stager)
    assert np.frombuffer(payload, np.uint8).min() == 200


def test_a_thumbnail_much_smaller_than_the_frame_keeps_the_output_offsets():
    """Regression: the output offsets were the *source* Y plane's, so every 4K frame wrote
    past the buffer and the int8 preview silently produced nothing. 8x6 could not catch it
    (source size == thumbnail size); this uses a frame 4x wider than the thumbnail."""
    frame = payload_frame(128, width=64, height=48)
    stager = PreviewStager(16, 12)
    assert stager.submit(frame)
    payload, fmt, _flags = drain(stager)
    assert fmt == FMT_YUV420
    assert len(payload) == 16 * 12 + 2 * 8 * 6, "Y at 16x12, chroma at 8x6"
    levels = np.frombuffer(payload, np.uint8)
    assert levels.min() == 128 and levels.max() == 128
