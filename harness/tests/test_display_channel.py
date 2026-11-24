"""The display channel's byte contract, offscreen and without a GPU.

The pane reads these bytes with its own copy of the layout (`src/ui/QTcustom.py`), so every offset here
is a promise between two processes that never see each other's code: a field at the wrong offset does
not raise, it draws a torn or colour-shifted picture. These tests pin the offsets, the two-slot sequence
flip, the "nobody is watching" refusal and a real shared-memory round trip, plus the 16-byte seek block
whose newest generation the render loop acts on.

No torch, no CUDA, no ffmpeg: the channel and the request block are plain memory, and the stager is only
asked the questions that must answer without a device.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
for _path in (_ROOT, os.path.join(_ROOT, "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.utils import DisplayChannel as DC  # noqa: E402
from src.utils.DisplayStager import DisplayStager, fit_box  # noqa: E402
from src.utils.shm import attach as shm_attach  # noqa: E402

_counter = 0


def _name(slug: str) -> str:
    """A fresh segment name: the pid keeps parallel runs out of each other's blocks."""
    global _counter
    _counter += 1
    return f"/rve-display-test-{slug}-{os.getpid()}-{_counter}"


def _channel(box=(4, 4), slug="ch") -> DC.DisplayChannel:
    channel = DC.DisplayChannel(_name(slug), box=box)
    assert channel.attached, "a channel with a name creates its block when the GUI has not"
    assert channel.created
    return channel


def _ask(channel: DC.DisplayChannel) -> None:
    """What the pane does when it starts watching."""
    channel._shm.buf[DC.OFFSET_WANTED] = 1


def _frame(seed: int, width: int = 2, height: int = 2) -> bytes:
    return bytes(((seed + i) & 0xFF) for i in range(width * height * 3))


# --- header ----------------------------------------------------------------------------------------


def test_the_header_is_the_literal_struct_and_wanted_sits_at_offset_nine():
    assert DC.HEADER.format == "<IHHBBBI d"
    # the literal struct: 4 + 2 + 2 + 1 + 1 + 1 + 4 + 8 - the `I` is the reserved word, the `d` the picture's time
    assert DC.HEADER.size == DC.HEADER_SIZE == 23
    assert (DC.OFFSET_SEQ, DC.OFFSET_WIDTH, DC.OFFSET_HEIGHT, DC.OFFSET_FMT,
            DC.OFFSET_WANTED, DC.OFFSET_FLAGS, DC.OFFSET_RESERVED) == (0, 4, 6, 8, 9, 10, 11)
    assert DC.OFFSET_TIME == 15
    packed = DC.HEADER.pack(7, 640, 360, DC.FMT_RGB888, 3, 0, 0, 3.5)
    assert len(packed) == 23
    # the pane's poll is a single byte at 9; every field BEFORE it must stay where it is
    assert packed[DC.OFFSET_WANTED] == 3
    assert DC.HEADER.unpack(packed) == (7, 640, 360, 1, 3, 0, 0, 3.5)
    # the slots start where the struct ends, never at a hard-coded offset
    assert DC.HEADER_SIZE == DC.OFFSET_TIME + 8


# --- the writer's side -----------------------------------------------------------------------------


def test_publish_refuses_when_the_pane_is_not_asking():
    channel = _channel()
    try:
        assert channel.wanted() is False
        assert channel.publish(_frame(0), 2, 2) is False
        assert channel.seq == 0, "a refused publish must not advance the sequence"
    finally:
        channel.close()


def test_two_slots_and_the_sequence_flips():
    channel = _channel(box=(2, 2))
    try:
        _ask(channel)
        first = _frame(0)
        second = _frame(100)
        assert channel.publish(first, 2, 2) is True
        assert channel.seq == 1
        slot_of_first = DC.HEADER_SIZE
        assert bytes(channel._shm.buf[slot_of_first:slot_of_first + 12]) == first
        assert channel.publish(second, 2, 2) is True
        assert channel.seq == 2
        # a second slot, after the first one, and the first one is left as it was
        slot_of_second = DC.HEADER_SIZE + channel.slot_size
        assert bytes(channel._shm.buf[slot_of_second:slot_of_second + 12]) == second
        assert bytes(channel._shm.buf[slot_of_first:slot_of_first + 12]) == first
        # the newest complete slot is (seq - 1) & 1, and it holds the frame just published
        assert (channel.seq - 1) & 1 == 1
    finally:
        channel.close()


def test_a_payload_larger_than_the_slot_is_refused_not_truncated():
    channel = _channel(box=(2, 2))
    try:
        _ask(channel)
        assert channel.slot_size == 12
        assert channel.publish(_frame(0, 4, 4), 4, 4) is False
        assert channel.seq == 0
    finally:
        channel.close()


def test_a_payload_whose_length_disagrees_with_its_geometry_is_refused():
    channel = _channel(box=(2, 2))
    try:
        _ask(channel)
        assert channel.publish(b"\x00" * 11, 2, 2) is False
        assert channel.seq == 0
    finally:
        channel.close()


def test_a_publish_can_name_the_picture_it_sends():
    """The seek path knows the media time of the frame it decoded; that is what makes the landing checkable.

    A sequence number proves a NEW publication, not WHICH picture (owner, 2026-11-24).
    """
    channel = _channel(box=(2, 2))
    try:
        _ask(channel)
        assert channel.publish(_frame(0, 2, 2), 2, 2, time_s=3.5) is True
        assert channel.published_time() == 3.5
        assert channel.publish(_frame(1, 2, 2), 2, 2) is True
        assert channel.published_time() is None, "a picture the writer cannot name must not report a time"
    finally:
        channel.close()


# --- the reader's side, through a real segment -----------------------------------------------------


def test_a_full_publish_read_round_trip_through_shared_memory():
    name = _name("roundtrip")
    channel = DC.DisplayChannel(name, box=(4, 4))
    assert channel.attached and channel.created
    reader = None
    try:
        _ask(channel)
        payload = _frame(7, 3, 2)          # 3x2 is inside a 4x4 box, i.e. an aspect-preserving fit
        assert channel.publish(payload, 3, 2) is True

        reader = shm_attach(name)
        seq, width, height, fmt, wanted, flags, reserved, stamp = DC.HEADER.unpack_from(reader.buf, 0)
        assert (width, height, fmt) == (3, 2, DC.FMT_RGB888)
        assert wanted == 1, "the writer must leave the pane's own byte alone"
        assert (flags, reserved) == (0, 0)
        assert stamp == DC.TIME_UNSET, "a publish without a time must say so, not claim second zero"
        slot = (seq - 1) & 1
        start = DC.HEADER_SIZE + slot * ((len(reader.buf) - DC.HEADER_SIZE) // 2)
        assert bytes(reader.buf[start:start + 18]) == payload
    finally:
        if reader is not None:
            reader.close()
        channel.close()


# --- the seek request ------------------------------------------------------------------------------


def test_the_seek_block_is_sixteen_bytes_of_double_then_generation():
    assert DC.SEEK_STRUCT.format == "<dI"
    assert DC.SEEK_STRUCT.size == 12
    assert DC.SEEK_BLOCK_SIZE == 16
    assert struct.calcsize("<dI") == 12


def test_a_newer_generation_is_distinguishable_and_newest_wins():
    state = DC.SeekState(_name("seek"))
    try:
        assert state.attached and state.created
        assert state.pending() is None, "the generation seen at attach is the baseline"
        DC.write_seek(state, 30.25, 1)
        assert state.pending() == 30.25
        assert state.pending() is None, "one request is acted on once"
        # newest wins: the second request overwrites the first, there is no queue
        DC.write_seek(state, 5.0, 2)
        DC.write_seek(state, 44.75, 3)
        assert state.pending() == 44.75
        assert state.last_generation == 3
    finally:
        state.close()


def test_write_seek_never_touches_the_reserved_tail():
    state = DC.SeekState(_name("tail"))
    try:
        DC.write_seek(state, 1.0, 1)
        assert bytes(state._shm.buf[12:16]) == b"\x00\x00\x00\x00"
    finally:
        state.close()


# --- the stager, without a device ------------------------------------------------------------------


def test_the_stager_imports_without_torch_and_answers_the_empty_cases():
    stager = DisplayStager(64, 36)
    assert stager.busy() is False
    assert stager.take() is None
    assert stager.submit(None) is False, "no frame means no copy, not an exception"


@pytest.mark.parametrize("source,box,expected", [
    ((1920, 1080), (640, 360), (640, 360)),      # 16:9 into a 16:9 box: exactly the box
    ((1440, 1080), (640, 360), (480, 360)),      # 4:3 fitted by height, aspect kept
    ((100, 100), (640, 360), (360, 360)),        # square fitted by the short side
    ((1, 1000), (640, 360), (1, 360)),           # degenerate width never collapses to zero
])
def test_fit_box_keeps_the_aspect_and_never_exceeds_the_box(source, box, expected):
    assert fit_box(source[0], source[1], box[0], box[1]) == expected


# --- the CLI token the pane sends ----------------------------------------------------------------


def test_the_display_size_token_is_parsed_or_refused():
    # A child interpreter with `backend/` as cwd, the way the app launches the backend: at the repo root
    # `src` is a different package, so the same import must be proved where it really runs.
    script = (
        "import argparse, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('b', 'rve-backend.py')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert module.parse_size('1280x720') == (1280, 720)\n"
        "assert module.parse_size('640X360') == (640, 360)\n"
        "assert module.parse_size(None) is None\n"
        "for bad in ('1280', '0x720', '1280x720x2', 'bad'):\n"
        "    try:\n"
        "        module.parse_size(bad)\n"
        "    except argparse.ArgumentTypeError:\n"
        "        continue\n"
        "    raise AssertionError('accepted ' + bad)\n"
        "print('parse ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=os.path.join(_ROOT, "backend"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-1500:]
    assert "parse ok" in result.stdout
