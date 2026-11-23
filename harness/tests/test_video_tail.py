"""The tail reader: a file that grows while it is being read.

This is the live-preview path (plan-rve-video-preview-2026-11-23, "Full resolution on both sides"):
the renderer writes a preview file, the GUI follows it. The three things that must never break are
decidable without a decoder, so the test drives a fake container over real bytes in a real file:

* EOF is a wait - no error, no busy retry, no work when the size has not changed;
* a partially written trailing record is never emitted, and the same record becomes exactly one
  frame once it is complete;
* resuming after a dead demuxer re-reads from the last published frame and repeats nothing.

The fake container mimics the one behaviour that matters from libavformat: after it has returned the
end of the file once, it stays dead until it is reopened (its EOF flag is sticky), which is why the
reader has to reopen and seek back. `tail.py` takes the container from a factory for exactly this.

Files live under the clone's `scratch/` - this cell may not write outside its workdir.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import types

import pytest

pytest.importorskip("PySide6")

from src.ui.video.tail import TailReader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "scratch")

FRAME_BYTES = 12      # a 2x2 rgb24 "frame" in the fake container
FPS = 10.0


@pytest.fixture(autouse=True)
def _qt_app(qapp):
    """Signals need an application object; the session fixture hands out the one QApplication."""
    return qapp


@pytest.fixture
def scratch_dir():
    os.makedirs(SCRATCH, exist_ok=True)
    path = tempfile.mkdtemp(prefix="tail-", dir=SCRATCH)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _Plane:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


class _FakeFrame:
    width = height = 2

    def __init__(self, payload: bytes, pts: float) -> None:
        self.time = pts
        self._payload = payload

    def to_ndarray(self, format: str):  # noqa: A002 - PyAV's keyword
        assert format == "rgb24"
        return _Plane(self._payload)


class _FakePacket:
    def __init__(self, index: int, payload: bytes) -> None:
        self._index = index
        self._payload = payload

    def decode(self):
        return [_FakeFrame(self._payload, self._index / FPS)]


class _CodecContext:
    thread_count = 0

    def flush_buffers(self) -> None:
        pass


class _FakeStream:
    average_rate = FPS
    codec_context = _CodecContext()


class _FakeContainer:
    """Fixed-size records in a file; EOF is sticky until the container is reopened."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.streams = types.SimpleNamespace(video=[_FakeStream()])
        self.offset = 0
        self.seeks: list[tuple[int, bool]] = []
        self.closed = False
        self._dead = False

    def demux(self, stream):
        if self._dead:
            return iter(())
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        count = len(data) // FRAME_BYTES
        first = self.offset // FRAME_BYTES
        packets = [
            _FakePacket(first + i, data[i * FRAME_BYTES:(i + 1) * FRAME_BYTES])
            for i in range(count)
        ]
        self.offset += count * FRAME_BYTES
        if not packets:
            self._dead = True          # sticky, like libavformat's EOF flag
        return self._snapshot(packets)

    def _snapshot(self, packets):
        """Hand out the snapshot, then stay at EOF until this container is reopened."""
        for packet in packets:
            yield packet
        self._dead = True

    def seek(self, offset: int, backward: bool = True) -> None:
        self.seeks.append((offset, backward))
        written = os.path.getsize(self.path) // FRAME_BYTES
        index = min(int(round(offset / 1_000_000 * FPS)), written)
        self.offset = max(0, index) * FRAME_BYTES

    def close(self) -> None:
        self.closed = True


class _Harness:
    """The reader plus everything a test needs to observe it."""

    def __init__(self, path: str, factory=None, **kwargs) -> None:
        self.path = path
        self.frames = []
        self.sizes = []
        self.errors = []
        self.containers: list[_FakeContainer] = []
        self.reader = TailReader(container_factory=factory or self._factory, **kwargs)
        self.reader.frame_ready.connect(self.frames.append)
        self.reader.grew.connect(self.sizes.append)
        self.reader.error.connect(self.errors.append)

    def _factory(self, path: str) -> _FakeContainer:
        container = _FakeContainer(path)
        self.containers.append(container)
        return container

    def append(self, count: int, start: int = 0) -> None:
        with open(self.path, "ab") as handle:
            for i in range(start, start + count):
                handle.write(bytes([(i + 1) % 256]) * FRAME_BYTES)

    def append_bytes(self, data: bytes) -> None:
        with open(self.path, "ab") as handle:
            handle.write(data)


def _emptied(directory: str, name: str = "preview.mp4") -> str:
    path = os.path.join(directory, name)
    open(path, "wb").close()
    return path


def test_eof_is_a_wait_and_growth_is_the_next_picture(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path)
    assert harness.reader.open(path) is True

    assert harness.reader.poll() == 0, "an empty file is not an error and has no frames"
    assert harness.errors == []

    harness.append(2)
    assert harness.reader.poll() == 2
    assert [round(frame.pts, 3) for frame in harness.frames] == [0.0, 0.1]
    assert harness.sizes == [2 * FRAME_BYTES], "growth is reported with the new size"

    assert harness.reader.poll() == 0, "no growth, no work"
    assert len(harness.frames) == 2

    harness.append(1, start=2)
    assert harness.reader.poll() == 1
    assert round(harness.frames[-1].pts, 3) == 0.2
    assert harness.sizes == [2 * FRAME_BYTES, 3 * FRAME_BYTES]
    assert harness.errors == []


def test_a_partial_trailing_frame_is_not_emitted_and_completes_later(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path)
    harness.reader.open(path)

    harness.append(1)
    harness.append_bytes(b"\xaa" * (FRAME_BYTES // 2))   # half of frame 1 is on disk

    assert harness.reader.poll() == 1, "only the complete record is a frame"
    assert [round(frame.pts, 3) for frame in harness.frames] == [0.0]

    harness.append_bytes(b"\xaa" * (FRAME_BYTES - FRAME_BYTES // 2))
    assert harness.reader.poll() == 1, "the completed record is emitted exactly once"
    assert [round(frame.pts, 3) for frame in harness.frames] == [0.0, 0.1]
    assert harness.errors == [], "a half-written fragment is the normal state of a growing file"


def test_resuming_after_a_dead_demuxer_does_not_repeat_frames(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path)
    harness.reader.open(path)

    harness.append(3)
    assert harness.reader.poll() == 3
    assert [round(frame.pts, 3) for frame in harness.frames] == [0.0, 0.1, 0.2]

    # The fake container is dead after EOF (like libavformat); the reader must reopen and seek back
    # instead of believing the demuxer's second "end of file".
    harness.append(2, start=3)
    assert harness.reader.poll() == 2, "the reader resumed through a fresh container"
    assert len(harness.containers) == 2
    assert harness.containers[0].closed is True, "the dead container is released, not leaked"
    offset, backward = harness.containers[1].seeks[-1]
    assert backward is True, "resume seeks backwards: it may not look past what was written"
    assert offset <= round(0.2 * 1_000_000)

    assert [round(frame.pts, 3) for frame in harness.frames] == [0.0, 0.1, 0.2, 0.3, 0.4], \
        "each frame reaches the surface once"


def test_a_replaced_file_starts_over(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path)
    harness.reader.open(path)

    harness.append(3)
    harness.reader.poll()
    with open(path, "wb") as handle:          # a new render at the same path
        handle.write(b"\x77" * (2 * FRAME_BYTES))

    assert harness.reader.poll() == 2
    assert [round(frame.pts, 3) for frame in harness.frames] == [0.0, 0.1, 0.2, 0.0, 0.1], \
        "the new file's frames are numbered from its own start"


def test_a_container_that_never_opens_is_reported_once_after_the_grace_window(scratch_dir):
    path = _emptied(scratch_dir)

    def refuse(_path):
        raise ValueError("no header yet")

    harness = _Harness(path, factory=refuse, open_grace_polls=2)
    harness.reader.open(path)

    harness.append_bytes(b"\x00")
    assert harness.reader.poll() == 0
    assert harness.errors == [], "the first unreadable poll is the header still being written"

    harness.append_bytes(b"\x00")
    assert harness.reader.poll() == 0
    assert len(harness.errors) == 1 and "still not readable" in harness.errors[0]

    harness.append_bytes(b"\x00")
    assert harness.reader.poll() == 0
    assert len(harness.errors) == 1, "a repeat at every growth would be noise, not a report"


def test_a_missing_file_is_refused_at_open(scratch_dir):
    harness = _Harness(os.path.join(scratch_dir, "not-there.mp4"))
    assert harness.reader.open(os.path.join(scratch_dir, "not-there.mp4")) is False
    assert len(harness.errors) == 1 and "no such file" in harness.errors[0]
    assert harness.reader.poll() == 0, "a reader with no file does nothing"


def test_close_releases_the_container_and_stops_polling(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path)
    harness.reader.open(path)
    harness.append(1)
    harness.reader.poll()

    harness.reader.close()

    assert harness.containers[0].closed is True
    assert harness.reader.poll() == 0
    assert len(harness.frames) == 1


def test_poll_without_growth_does_not_open_a_second_container(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path)
    harness.reader.open(path)
    harness.append(1)
    assert harness.reader.poll() == 1

    for _ in range(3):
        assert harness.reader.poll() == 0

    assert len(harness.containers) == 1, "an idle reader is not an I/O loop"
