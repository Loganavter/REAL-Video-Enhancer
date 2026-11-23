"""The queue policy: capacity, latest-wins, and which frame is "due".

These are the two behaviours an operator experiences as "the preview stutters" or "the preview is a
second behind the render". Both are decidable without a decoder, a window or a GPU, so they are
pinned here with plain numbers - the half of the player that must not regress when the surface is
being rewritten around it.

`frames.py` imports neither Qt nor PyAV, which is what lets this file run on the repo's bare
interpreter (no PyAV there).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.ui.video.frames import FrameQueue, FrameRef, frame_pts, pack_frame


def ref(pts: float, index: int | None = None) -> FrameRef:
    return FrameRef(
        buffer=b"\x00" * 4,
        width=2,
        height=2,
        format="rgb24",
        stride=6,
        pts=pts,
        index=int(pts * 100) if index is None else index,
    )


def pts_of(frame: FrameRef | None) -> float | None:
    return None if frame is None else frame.pts


# -- capacity and drop policy ----------------------------------------------------------------------

def test_a_full_queue_drops_its_oldest_frame_and_says_how_many():
    queue = FrameQueue(4)
    for i in range(4):
        assert queue.push(ref(i / 10)) == 0, "four frames fit in a four-deep queue"
    assert len(queue) == 4

    assert queue.push(ref(0.4)) == 1, "the fifth frame costs exactly one drop"
    assert len(queue) == 4

    # Latest wins: the oldest frame (0.0) is gone, so the newest frame due at 0.25 is 0.2, and the
    # maxlen of 4 means only 0.2/0.3/0.4 were ever in the queue.
    assert pts_of(queue.pop_due(0.25)) == pytest.approx(0.2)
    assert len(queue) == 2
    assert pts_of(queue.pop_due(0.5)) == pytest.approx(0.4)


def test_a_single_slot_queue_always_keeps_the_newest_frame():
    queue = FrameQueue(1)
    assert queue.push(ref(0.0)) == 0
    assert queue.push(ref(0.1)) == 1
    assert queue.pop_due(1.0).pts == pytest.approx(0.1)


def test_a_queue_needs_room_for_at_least_one_frame():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            FrameQueue(bad)


# -- which frame is due ----------------------------------------------------------------------------

def test_pop_due_returns_the_newest_due_frame_and_drops_the_older_due_ones():
    queue = FrameQueue(4)
    for i in range(3):
        queue.push(ref(i / 25))

    due = queue.pop_due(0.085)

    assert due is not None and due.pts == pytest.approx(0.08), "newest frame at or before now"
    assert len(queue) == 0, "0.0 and 0.04 are late now and must not be replayed later"


def test_pop_due_leaves_frames_that_are_still_in_the_future():
    queue = FrameQueue(4)
    queue.push(ref(0.0))
    queue.push(ref(0.5))
    queue.push(ref(0.6))

    assert pts_of(queue.pop_due(0.1)) == pytest.approx(0.0)
    assert len(queue) == 2, "the two future frames stay queued"
    assert pts_of(queue.pop_due(0.7)) == pytest.approx(0.6)


def test_pop_due_is_none_when_nothing_has_been_reached_yet():
    queue = FrameQueue(4)
    queue.push(ref(1.0))
    assert queue.pop_due(0.9) is None, "a future frame is not a frame to draw"
    assert len(queue) == 1, "and it is not consumed by asking"


def test_pop_due_on_an_empty_queue_is_none_not_an_exception():
    assert FrameQueue(4).pop_due(12.0) is None


def test_clear_forgets_everything():
    queue = FrameQueue(4)
    queue.push(ref(0.0))
    queue.push(ref(0.1))
    queue.clear()
    assert len(queue) == 0
    assert queue.pop_due(1.0) is None


def test_a_frame_ref_is_immutable():
    frame = ref(0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.pts = 1.0


# -- packing a decoded frame -----------------------------------------------------------------------

class _Plane:
    """The bit of a numpy array `pack_frame` uses on the rgb24 path."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


class _StubFrame:
    """The duck-typed surface of a decoded PyAV frame (no PyAV on this interpreter)."""

    def __init__(self, width: int, height: int, time, payload: bytes) -> None:
        self.width, self.height, self.time = width, height, time
        self._payload = payload
        self.formats: list[str] = []

    def to_ndarray(self, format: str):  # noqa: A002 - PyAV's keyword
        self.formats.append(format)
        return _Plane(self._payload)


def test_rgb24_packing_keeps_the_geometry_stride_and_pts():
    pixels = bytes([1, 2, 3, 4, 5, 6])  # 3x1 rgb24
    frame = _StubFrame(3, 1, 1.25, pixels)

    packed = pack_frame(frame, index=31)

    assert packed.format == "rgb24"
    assert (packed.width, packed.height, packed.stride) == (3, 1, 9)
    assert packed.buffer == pixels
    assert packed.pts == pytest.approx(1.25)
    assert packed.index == 31
    assert frame.formats == ["rgb24"], "the decoder is asked for RGB, not left to its own format"


def test_a_frame_without_a_timestamp_reports_zero_rather_than_crashing():
    assert frame_pts(_StubFrame(1, 1, None, b"\x00\x00\x00")) == 0.0


def test_an_explicit_pts_overrides_a_missing_timestamp():
    frame = _StubFrame(1, 1, None, b"\x00\x00\x00")
    assert pack_frame(frame, index=7, pts=0.4).pts == pytest.approx(0.4)


def test_yuv420p_packing_stacks_luma_then_both_chroma_planes():
    numpy = pytest.importorskip("numpy")
    width, height = 4, 4
    chroma_w, chroma_h = width // 2, height // 2

    plane = numpy.zeros((height + 2 * chroma_h, width), dtype=numpy.uint8)
    plane[:height, :width] = 10                                      # Y
    plane[height:height + chroma_h, :chroma_w] = 20                   # U
    plane[height + chroma_h:, :chroma_w] = 30                         # V

    class _YuvFrame(_StubFrame):
        def to_ndarray(self, format: str):  # noqa: A002
            self.formats.append(format)
            return plane

    packed = pack_frame(_YuvFrame(width, height, 0.5, b""), index=2, pixel_format="yuv420p")

    assert packed.format == "yuv420p"
    assert packed.stride == width, "plane 0's stride is the luma row length"
    assert len(packed.buffer) == width * height + 2 * chroma_w * chroma_h
    assert packed.buffer[:width * height] == bytes([10]) * (width * height)
    assert packed.buffer[width * height:width * height + chroma_w * chroma_h] == \
        bytes([20]) * (chroma_w * chroma_h)
    assert packed.buffer[width * height + chroma_w * chroma_h:] == \
        bytes([30]) * (chroma_w * chroma_h)


def test_an_unknown_format_is_refused_loudly():
    with pytest.raises(ValueError):
        pack_frame(_StubFrame(1, 1, 0.0, b""), index=0, pixel_format="nv12")
