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
import time
import types

import pytest

pytest.importorskip("PySide6")

from src.ui.video.tail import LiveTailThread, TailReader

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


def test_a_missing_file_is_a_wait_not_an_error(scratch_dir):
    """The renderer creates its output when the encoder starts, so a reader that begins at "render just launched"
    sees nothing for a moment. `False` means "try again", and the log stays quiet - measured against a real preview
    render, where the old behaviour printed ten "no such file" lines before the first fragment."""
    harness = _Harness(os.path.join(scratch_dir, "not-there.mp4"))
    assert harness.reader.open(os.path.join(scratch_dir, "not-there.mp4")) is False
    assert harness.errors == [], "a missing file is not an error the caller has to handle"
    assert harness.reader.poll() == 0, "a reader with no file does nothing"
    # and it becomes usable as soon as the writer creates the file
    path = os.path.join(scratch_dir, "appears.mp4")
    harness.append(0) if False else None
    open(path, "wb").write(b"")
    assert harness.reader.open(path) is True


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


# -- seeking while the file is still being written -----------------------------------------------
#
# The owner drags the strip mid-render. Before `TailReader.seek` existed the reader only ever stood at
# the written edge, so every emitted frame put the picture back there ("на каждом новом кадре ставим
# воспроизведение в конец"). A seek must show its target on the very next poll - the tail polls every
# 120 ms - and following must then continue from the target, once per frame.


def test_a_backward_seek_shows_the_target_on_the_next_poll(scratch_dir):
    path = _emptied(scratch_dir)
    # One frame per poll: "visible within one poll" is then exactly one emitted frame.
    harness = _Harness(path, max_frames_per_poll=1)
    harness.reader.open(path)
    harness.append(10)                         # 0.0 .. 0.9 s, the produced edge is 0.9
    for _ in range(10):
        assert harness.reader.poll() == 1
    assert [round(frame.pts, 3) for frame in harness.frames] == [round(i / FPS, 3) for i in range(10)]

    landed = harness.reader.seek(0.3)
    assert round(landed, 3) == 0.3, "the seek reports where it landed"

    assert harness.reader.poll() == 1, "the next poll has work even though the file did not grow"
    assert round(harness.frames[10].pts, 3) == 0.3, \
        "the poll after a seek shows the TARGET, not the produced edge (0.9)"
    assert harness.errors == [], "a seek inside written data is not a failure"


def test_following_after_a_seek_emits_each_frame_once_in_order(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path, max_frames_per_poll=100)
    harness.reader.open(path)
    harness.append(10)                         # 0.0 .. 0.9
    assert harness.reader.poll() == 10

    harness.reader.seek(0.3)
    assert harness.reader.poll() == 7
    assert [round(frame.pts, 3) for frame in harness.frames[10:]] == \
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], \
        "the seek replays from the target to the produced edge, in order"

    harness.append(2, start=10)                # the renderer writes two more frames
    assert harness.reader.poll() == 2
    assert [round(frame.pts, 3) for frame in harness.frames[17:]] == [1.0, 1.1], \
        "frames that arrive later are emitted exactly once, in order"
    assert harness.reader.poll() == 0
    assert harness.reader.poll() == 0
    assert [round(frame.pts, 3) for frame in harness.frames[17:]] == [1.0, 1.1], \
        "an idle poll repeats nothing - the playhead does not jump back to the edge"

    harness.append(1, start=12)
    assert harness.reader.poll() == 1
    assert round(harness.frames[-1].pts, 3) == 1.2, "the playhead advances with the new frames"


def test_seeking_to_the_produced_edge_is_follow_live(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path, max_frames_per_poll=100)
    harness.reader.open(path)
    harness.append(4)                          # 0.0 .. 0.3
    assert harness.reader.poll() == 4

    edge = harness.reader.produced_seconds
    assert round(edge, 3) == 0.3, "the produced edge is the last frame the surface has seen"

    assert round(harness.reader.seek(edge), 3) == 0.3
    assert harness.reader.poll() == 1, "the edge frame is drawn once, like any frame that just arrived"
    assert round(harness.frames[-1].pts, 3) == 0.3

    harness.append(1, start=4)
    assert harness.reader.poll() == 1, "follow-live continues after a seek to the edge"
    assert round(harness.frames[-1].pts, 3) == 0.4
    assert harness.reader.poll() == 0


def test_a_seek_past_the_produced_edge_is_clamped_to_it(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path, max_frames_per_poll=100)
    harness.reader.open(path)
    harness.append(3)                          # 0.0 .. 0.2
    assert harness.reader.poll() == 3

    landed = harness.reader.seek(60.0)
    assert round(landed, 3) == 0.2, "nothing exists in front of the produced edge"
    assert harness.reader.poll() == 1
    assert round(harness.frames[-1].pts, 3) == 0.2

    offset, backward = harness.containers[-1].seeks[-1]
    assert backward is True, "the seek may not look past what was written"
    assert offset <= round(0.2 * 1_000_000)
    assert [round(frame.pts, 3) for frame in harness.frames[3:]] == [0.2], "no frame beyond the edge"
    assert harness.errors == []


def test_a_seek_with_nothing_produced_lands_at_the_start(scratch_dir):
    path = _emptied(scratch_dir)
    harness = _Harness(path, max_frames_per_poll=100)
    harness.reader.open(path)

    assert harness.reader.seek(5.0) == 0.0, "with no frame written there is only one place to stand"
    harness.append(2)
    assert harness.reader.poll() == 2, "and the tail follows as if nothing had been asked"
    assert [round(frame.pts, 3) for frame in harness.frames] == [0.0, 0.1]


def test_the_live_tail_thread_polls_without_a_caller_clock_and_takes_a_seek(qapp, scratch_dir):
    """The tail must run on its own thread: a QTimer built in the render's worker never fires there.

    Measured 2026-11-23: the pane showed the 640x360 thumbnail for a whole 12 s preview because the reader's clock
    was a `QTimer` on a `threading.Thread` (no Qt event loop), and the strip's signals were connected on that same
    thread - so even a working reader would have had its seek queued into nothing. This drives the shipped loop the
    way ProcessTab does: start it, deliver its signals through the test's event loop, and hand it a seek from
    outside.
    """
    path = _emptied(scratch_dir)
    with open(path, "ab") as handle:
        for index in range(4):
            handle.write(bytes([index + 1]) * FRAME_BYTES)

    containers: list[_FakeContainer] = []

    def factory(target: str) -> _FakeContainer:
        container = _FakeContainer(target)
        containers.append(container)
        return container

    thread = LiveTailThread(path, interval_ms=20, max_frames_per_poll=100, paused=False,
                            container_factory=factory)
    frames = []
    errors = []
    thread.frame_ready.connect(frames.append)
    thread.error.connect(errors.append)
    thread.start()
    try:
        deadline = time.time() + 5
        while len(frames) < 4 and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert [round(frame.pts, 3) for frame in frames] == [0.0, 0.1, 0.2, 0.3], \
            "the loop reads the written part with no clock from its owner"
        assert thread.published_frames == 4
        assert round(thread.produced_seconds, 3) == 0.3

        # A seek is asked for from another thread: it must be applied between polls, not raced into the reader.
        thread.seek(0.1)
        deadline = time.time() + 5
        while len(frames) < 7 and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert [round(frame.pts, 3) for frame in frames[4:]] == [0.1, 0.2, 0.3], \
            "the replay starts at the target and skips nothing behind it"
        assert errors == []
    finally:
        thread.stop_and_wait()
    assert thread.isRunning() is False, "the loop stops when its owner says so"
    assert containers and containers[-1].closed is True, "the demuxer is released, not leaked"


def _wait_for(qapp, predicate, seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_paused_tail_reads_the_first_frame_then_holds_the_picture(qapp, scratch_dir):
    """The preview stands paused by default: one frame on screen, and then no polling at all.

    "Из начально на ней и стояло" (owner, 2026-11-23) - but paused on a black pane would be paused on nothing, so the
    loop reads until the first frame exists. After that the renderer may write as much as it likes: the picture, the
    playhead and the reader's position do not move until the transport is pressed.
    """
    path = _emptied(scratch_dir)
    with open(path, "ab") as handle:
        handle.write(b"\x01" * (2 * FRAME_BYTES))

    def factory(target: str) -> _FakeContainer:
        return _FakeContainer(target)

    thread = LiveTailThread(path, interval_ms=20, max_frames_per_poll=100, container_factory=factory)
    frames = []
    thread.frame_ready.connect(frames.append)
    assert thread.paused is True, "a preview starts paused"
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 2), "the first frame reaches the pane"
        with open(path, "ab") as handle:            # the renderer keeps going
            handle.write(b"\x02" * (3 * FRAME_BYTES))
        time.sleep(0.6)
        qapp.processEvents()
        assert len(frames) == 2, f"a paused preview must not advance (got {len(frames)} frames)"
        assert round(thread.produced_seconds, 3) == 0.1, "the produced edge does not advance while held"

        thread.set_paused(False)
        assert _wait_for(qapp, lambda: len(frames) >= 5), "pressing play reads the backlog that was written"
        assert [round(frame.pts, 3) for frame in frames[2:5]] == [0.2, 0.3, 0.4]
        thread.set_paused(True)
        held = len(frames)
        with open(path, "ab") as handle:
            handle.write(b"\x03" * (2 * FRAME_BYTES))
        time.sleep(0.5)
        qapp.processEvents()
        assert len(frames) == held, "pausing again holds the picture where it stood"
    finally:
        thread.stop_and_wait()


def test_a_seek_while_paused_shows_its_target_and_holds_it(qapp, scratch_dir):
    """A strip press must work on a held preview: it reads once, shows the target, and stands there."""
    path = _emptied(scratch_dir)
    with open(path, "ab") as handle:
        handle.write(b"\x05" * (4 * FRAME_BYTES))

    def factory(target: str) -> _FakeContainer:
        return _FakeContainer(target)

    thread = LiveTailThread(path, interval_ms=20, max_frames_per_poll=100, container_factory=factory)
    frames = []
    thread.frame_ready.connect(frames.append)
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 4)
        time.sleep(0.3)
        qapp.processEvents()
        assert len(frames) == 4, "held after the first poll"

        thread.seek(0.2)
        assert _wait_for(qapp, lambda: len(frames) > 4), "the seek is applied while paused"
        assert [round(frame.pts, 3) for frame in frames[4:]] == [0.2], \
            "a held seek publishes its TARGET only - no backlog behind it (see test_preview_seek_held.py)"
        time.sleep(0.4)
        qapp.processEvents()
        assert len(frames) == 5, "and the picture stands on the target afterwards, not at the edge"
    finally:
        thread.stop_and_wait()



def test_a_reader_built_without_a_factory_has_the_real_demuxer(qapp):
    """The default factory is what every live preview uses; a test that always injects one cannot see it disappear.

    This is not hypothetical: deleting an unused helper next to it took `_default_container_factory` with it, and the
    live pane went black while the whole suite stayed green - every test here injects a fake container (measured
    2026-11-23 on a 1440x1080 preview render).
    """
    reader = TailReader(pixel_format="rgb24")
    assert callable(reader._container_factory)
    assert reader._container_factory.__name__ == "_default_container_factory"
    thread = LiveTailThread("/never/created.mkv", interval_ms=20)
    assert thread is not None, "the thread must be constructible with the default reader arguments"
