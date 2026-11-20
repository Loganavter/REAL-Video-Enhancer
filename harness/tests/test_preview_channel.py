"""The preview channel's contract: header, two slots, sequence, and the `wanted` byte.

The GUI allocates the block and the backend fills it, in two processes that never share
an import - so the layout is duplicated on purpose (`backend/src/utils/PreviewChannel.py`
and `src/ui/QTcustom.py`) and this test pins the half the backend owns. The failure worth
naming is the old channel's: one slot, no sequence, no lock, so a reader could assemble a
"frame" out of two different renders.

No torch here, so this file runs under both interpreters.
"""
from __future__ import annotations

import os
import struct
import sys

import pytest
from multiprocessing import shared_memory

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.utils.PreviewChannel import (  # noqa: E402
    FMT_RGB888,
    HEADER,
    OFF_WANTED,
    PreviewChannel,
)


@pytest.fixture()
def block():
    name = f"/rve_preview_test_{os.getpid()}"
    width, height = 6, 4
    shm = shared_memory.SharedMemory(name=name, create=True,
                                     size=HEADER.size + 2 * width * height * 3)
    try:
        yield name, width, height, shm
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def test_publish_writes_the_header_and_clears_wanted(block):
    name, width, height, shm = block
    channel = PreviewChannel(name, box=(width, height))
    assert channel.attached
    shm.buf[OFF_WANTED] = 1
    assert channel.wanted()
    pixels = bytes(range(72))
    assert channel.publish(pixels, width, height)
    seq, w, h, fmt, wanted, flags, reserved = HEADER.unpack_from(shm.buf, 0)
    assert (w, h, fmt) == (width, height, FMT_RGB888)
    assert seq == 1 and wanted == 0, "publishing consumes the request"
    assert reserved == 0 and flags == 0


def test_two_publishes_land_in_different_slots_and_advance_the_sequence(block):
    name, width, height, shm = block
    channel = PreviewChannel(name, box=(width, height))
    slot = width * height * 3
    channel.publish(b"\x01" * slot, width, height)
    channel.publish(b"\x02" * slot, width, height)
    seq = HEADER.unpack_from(shm.buf, 0)[0]
    assert seq == 2
    newest = HEADER.size + (seq & 1) * slot
    previous = HEADER.size + ((seq - 1) & 1) * slot
    assert bytes(shm.buf[newest:newest + slot]) == b"\x02" * slot
    assert bytes(shm.buf[previous:previous + slot]) == b"\x01" * slot


def test_a_wrong_sized_payload_is_refused_loudly(block):
    name, width, height, _shm = block
    channel = PreviewChannel(name, box=(width, height))
    with pytest.raises(ValueError):
        channel.publish(b"\x00" * 5, width, height)


def test_a_block_that_never_appears_costs_nothing():
    channel = PreviewChannel(f"/rve_preview_absent_{os.getpid()}", wait_seconds=0.0)
    assert not channel.attached
    assert not channel.wanted()
    assert channel.publish(b"\x00" * 72, 6, 4) is False


def test_no_id_is_the_headless_case():
    channel = PreviewChannel(None)
    assert not channel.attached and not channel.wanted()
    assert channel.publish(b"\x00" * 72, 6, 4) is False
