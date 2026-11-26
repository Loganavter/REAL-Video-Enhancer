"""A strip seek on a HELD preview lands on its target and stays there; the LMB compare follows the picture.

The owner's report (2026-11-24): "I can pause and the seeker stops, but any attempt to move it puts me at the current
end of the upscale. And an LMB click either shows a couple of frames of desync or shows me the frame that is currently
in the upscale."

Two defects, both decidable offscreen with the fake container `test_video_tail.py` already drives:

* **A held seek published a backlog.** `LiveTailThread.run` applies a pending seek and then - because its
  `elif self._paused ... continue` is skipped for exactly that iteration - falls through to `reader.poll()`, which
  emits up to `max_frames_per_poll` (default 4) frames: the target AND the three behind it. The pane's playhead is set
  per emitted frame, so a press on 30 % left the picture and the marker about four frames past the target.
* **The LMB compare followed whatever the marker said.** `VideoPreviewSurface.mousePressEvent` asks for the source at
  `self._progress[2]`, and the pane publishes that value from the frame it was just given (`ProcessTab`
  `_on_preview_frame`: `set_frame(image)` then `set_progress(playhead=frame.pts)`). So the two ARE equal by
  construction - the defect was what the pane was given. Test (d) drives the real slot and proves the equality holds
  for the frame that is actually on screen, which is exactly what a held seek now lands on.

What is pinned, in the owner's terms:

(a) a held seek emits EXACTLY ONE frame, at the target;
(b) a second held seek emits one frame again (the reader really holds, it does not resume in bulk);
(c) a seek while PLAYING still emits a run of frames from the target, each once (the 2026-11-23 fix stays);
(d) the time the LMB press puts on `originalRequested` equals the pts of the frame the pane was last given.

Round 2 (2026-11-24, the second live run) pins the part the first pass got wrong. A press past the published
edge used to be answered by falling back to that edge, and on a HELD pane the edge is stale - the reader stopped
polling at the pause - so the picture landed where the pane last stood, or behind it:

(e) a press with no frame at it publishes NOTHING, and the next reachable press still lands exactly one frame;
(f) a press past the edge is judged on FRESH information: the muxer may have flushed between the pause and the
    press, and the frame written after the pause must be found;
(g) over a run of held presses the playhead never decreases and is never re-published onto a stale edge, while the
    deepest frame the file holds stays reachable.

Files live under the clone's `scratch/` - this cell may not write outside its workdir.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

# The app root must win over the backend for `ProcessTab` (both are `src` namespace portions; the suite's conftest puts
# the backend first, which makes `src.constants` the backend's module - the house pattern of `test_seek_ceiling.py`).
_ROOT = str(Path(__file__).resolve().parents[2])
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

try:  # pragma: no cover - depends on the interpreter
    import cpuinfo  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda *args, **kwargs: {}
    sys.modules["cpuinfo"] = _cpuinfo
try:  # pragma: no cover
    import pypresence  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _presence = types.ModuleType("pypresence")
    _presence.Presence = object
    sys.modules["pypresence"] = _presence

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.ProcessTab import ProcessTab  # noqa: E402
from src.ui.video.tail import LiveTailThread  # noqa: E402
from src.ui.widgets.video_preview import VideoPreviewSurface  # noqa: E402

sys.path[:] = _ORIGINAL_SYS_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "scratch")

FRAME_BYTES = 12      # a 2x2 rgb24 "frame" in the fake container
FPS = 10.0
INTERVAL_MS = 20
PANE_SIZE = (800, 600)


# -- the fake container (the shape `test_video_tail.py` uses) ------------------------------------------


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


@pytest.fixture(autouse=True)
def _qt_app(qapp):
    return qapp


@pytest.fixture
def scratch_dir():
    os.makedirs(SCRATCH, exist_ok=True)
    path = tempfile.mkdtemp(prefix="seek-held-", dir=SCRATCH)
    try:
        yield path
    finally:
        import shutil

        shutil.rmtree(path, ignore_errors=True)


def _written(directory: str, count: int) -> str:
    path = os.path.join(directory, "preview.mp4")
    with open(path, "wb") as handle:
        for index in range(count):
            handle.write(bytes([(index + 1) % 256]) * FRAME_BYTES)
    return path


def _append(path: str, count: int, start: int) -> None:
    with open(path, "ab") as handle:
        for index in range(start, start + count):
            handle.write(bytes([(index + 1) % 256]) * FRAME_BYTES)


def _spawn(path: str, *, paused: bool) -> LiveTailThread:
    """The shipped loop with its DEFAULT per-poll cap (4): the overshoot is the default, not a test setting."""
    thread = LiveTailThread(path, interval_ms=INTERVAL_MS, paused=paused,
                            container_factory=_FakeContainer)
    return thread


def _wait_for(qapp, predicate, seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _settle(qapp, seconds: float = 0.4) -> None:
    """Let the loop run several of its own intervals, so "nothing more arrived" is a real observation."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def _pts(frames) -> list[float]:
    return [round(frame.pts, 3) for frame in frames]


# -- (a) and (b): a held seek is ONE frame, twice in a row ---------------------------------------------


def test_a_held_seek_publishes_exactly_the_target_frame(qapp, scratch_dir):
    """Paused, a strip press shows its target and nothing else: no backlog, no four-frame overshoot."""
    path = _written(scratch_dir, 10)                 # 0.0 .. 0.9 s
    thread = _spawn(path, paused=True)
    frames = []
    thread.frame_ready.connect(frames.append)
    thread.start()
    try:
        assert thread.paused is True
        assert _wait_for(qapp, lambda: len(frames) >= 4), "the first poll paints the opening frames"
        _settle(qapp)
        held = len(frames)
        assert _pts(frames) == [0.0, 0.1, 0.2, 0.3], f"the held pane stands on a small band (got {_pts(frames)})"

        thread.seek(0.5)
        assert _wait_for(qapp, lambda: len(frames) > held), "the seek is applied while paused"
        _settle(qapp)
        assert _pts(frames[held:]) == [0.5], (
            f"a held seek publishes ONLY its target, got {_pts(frames[held:])} "
            f"(the produced edge is {round(thread.produced_seconds, 3)})")
        assert thread.published_frames == held + 1, "one frame published, not a burst"
        assert round(thread.produced_seconds, 3) == 0.5, (
            "the reader's own edge is the highest frame it published; the seek moved the playhead onto it")
        assert round(frames[-1].pts, 3) == 0.5, "and the frame on the surface is the target"
    finally:
        thread.stop_and_wait()


def test_a_held_pane_advances_the_produced_band_without_moving_the_playhead(qapp):
    """The band is the RENDERER's progress (`_refresh_held_progress`), not the tail's: a held pane's marker stays.

    This is the half of a hold that must keep working: the white part of the strip grows while the render runs, and
    the playhead/marker does not creep with it (a marker that drifts to the band's edge reads as "the seeker is at
    the current end of the upscale" even when the picture never moved).
    """
    pane = _pane()
    tab = _live_tab(pane, SimpleNamespace())
    tab._preview_paused = True
    image = QImage(2, 2, QImage.Format.Format_RGB888)
    image.fill(QColor("#204060"))
    pane.set_frame(image)
    pane.set_progress(rendered=4.0, total=100.0, playhead=0.4, duration=1.0)

    tab.currentFrame = 60
    tab.max_value = 100
    tab._refresh_held_progress()

    rendered, total, playhead, duration = pane._progress
    assert (rendered, total) == (60.0, 100.0), "the produced band follows the renderer"
    assert playhead == pytest.approx(0.4), "the playhead does not move with it"
    assert duration == pytest.approx(1.0), "and the axis it is drawn on does not shrink either"


def test_a_second_held_seek_publishes_one_frame_again(qapp, scratch_dir):
    """The reader really HOLDS between presses: the next seek is another single frame, not a resumed read."""
    path = _written(scratch_dir, 10)
    thread = _spawn(path, paused=True)
    frames = []
    thread.frame_ready.connect(frames.append)
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 4)
        _settle(qapp)

        thread.seek(0.5)
        assert _wait_for(qapp, lambda: len(frames) > 4)
        _settle(qapp)
        first = _pts(frames[4:])
        assert first == [0.5], f"first held seek: got {first}"

        thread.seek(0.2)
        assert _wait_for(qapp, lambda: len(frames) > 5)
        _settle(qapp)
        second = _pts(frames[5:])
        assert second == [0.2], f"second held seek: got {second}"

        # A press past everything the file holds is not answered by the frame the pane already stands on: the reader
        # attempts it, finds nothing at the target, and publishes NOTHING (the picture and the marker stay put). It used
        # to fall back to the published edge, i.e. re-publish 0.5 - a frame the press did not ask for, and the path that
        # moved the picture backwards live.
        refusals: list[str] = []
        thread.seek_unavailable.connect(refusals.append)
        thread.seek(9.0)
        _settle(qapp)
        assert _pts(frames[6:]) == [], (
            f"a press the file cannot satisfy publishes nothing, got {_pts(frames[6:])}")
        assert round(thread.produced_seconds, 3) == 0.5, (
            "the reader's own edge stays the highest frame it published - a refusal never lowers it")
        assert refusals and "9.000" in refusals[-1] and "0.200" in refusals[-1], (
            f"the refusal names the target and the frame that stays on screen, got {refusals}")
    finally:
        thread.stop_and_wait()


def test_a_held_seek_past_the_flushed_file_holds_the_picture(qapp, scratch_dir):
    """A press with no frame behind it publishes nothing, and the NEXT reachable press still lands exactly one frame.

    The live shape (2026-11-24, 12.5 % produced): the band says a place exists, the muxer has flushed less, and the
    target is past it. The answer must be "the file has no frame there" - not the stale published edge, which is where
    the picture already stands (or, worse, behind it).
    """
    path = _written(scratch_dir, 10)                 # the whole file is 0.0 .. 0.9 s
    thread = _spawn(path, paused=True)
    frames: list = []
    refusals: list[str] = []
    thread.frame_ready.connect(frames.append)
    thread.seek_unavailable.connect(refusals.append)
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 4)
        _settle(qapp)
        assert _pts(frames) == [0.0, 0.1, 0.2, 0.3], f"the held band (got {_pts(frames)})"
        standing = len(frames)
        before = round(thread.produced_seconds, 3)
        assert before == 0.3

        thread.seek(2.0)                             # past the file's own end
        _settle(qapp)
        assert _pts(frames[standing:]) == [], (
            f"nothing exists at 2.0 s, so nothing is published (got {_pts(frames[standing:])})")
        assert round(thread.produced_seconds, 3) == before, (
            "the reported playhead is the frame on screen - it does not move to a stale edge")
        assert refusals, "the attempt is legible in the log"
        assert "2.000" in refusals[-1], f"the log names the target, got {refusals[-1]!r}"

        thread.seek(0.5)                             # a target the file does have
        assert _wait_for(qapp, lambda: len(frames) > standing), "the next reachable press still acts"
        _settle(qapp)
        assert _pts(frames[standing:]) == [0.5], (
            f"one frame, at the target, after an unreachable press (got {_pts(frames[standing:])})")
    finally:
        thread.stop_and_wait()


def test_a_held_press_polls_the_file_the_muxer_grew_since_the_pause(qapp, scratch_dir):
    """A held pane judges its press on FRESH information: the file may have grown between the pause and the press.

    That is why the press is not clamped to the published edge before the read: the clamp was decided on a snapshot
    from the pause, while the muxer kept flushing. Here the frame written after the pause must be found.
    """
    path = _written(scratch_dir, 3)                  # 0.0 .. 0.2 s: fewer than the per-poll cap, so nothing is buffered
    thread = _spawn(path, paused=True)
    frames: list = []
    thread.frame_ready.connect(frames.append)
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 3)
        _settle(qapp)
        assert _pts(frames) == [0.0, 0.1, 0.2], f"the held band (got {_pts(frames)})"

        _append(path, 6, start=3)                    # the renderer keeps writing while the pane is held: 0.3 .. 0.8
        _settle(qapp)
        assert _pts(frames) == [0.0, 0.1, 0.2], "a held pane does not read on its own"

        thread.seek(0.7)
        assert _wait_for(qapp, lambda: len(frames) > 3), "the press polls"
        _settle(qapp)
        assert _pts(frames[3:]) == [0.7], (
            f"the press must find the frame flushed after the pause, got {_pts(frames[3:])}")
    finally:
        thread.stop_and_wait()


def test_the_playhead_never_moves_backwards_across_held_presses(qapp, scratch_dir):
    """The live sequence: a reachable press, then presses past the flush, then one at the deepest flushed frame.

    The picture must only ever move forward through such a run, must not be re-published onto a stale edge, and must
    still reach the deepest frame the file holds (the band edge) at the end.
    """
    path = _written(scratch_dir, 14)                 # 0.0 .. 1.3 s
    thread = _spawn(path, paused=True)
    frames: list = []
    refusals: list[str] = []
    thread.frame_ready.connect(frames.append)
    thread.seek_unavailable.connect(refusals.append)
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 4)
        _settle(qapp)
        seen = _pts(frames)
        held = len(frames)

        for target, reachable in ((0.9, True), (1.1, True), (2.0, False), (3.0, False), (1.3, True)):
            before = len(frames)
            edge_before = round(thread.produced_seconds, 3)
            thread.seek(target)
            if reachable:
                assert _wait_for(qapp, lambda: len(frames) > before), f"the press at {target} s must act"
            _settle(qapp)
            emitted = _pts(frames[before:])
            if reachable:
                assert emitted == [round(target, 3)], (
                    f"a reachable held press shows exactly its target, got {emitted} for {target} s")
            else:
                assert emitted == [], (
                    f"a press at {target} s has no frame behind it, so it publishes nothing (got {emitted}); "
                    f"the frame on screen is {edge_before} s")
                assert round(thread.produced_seconds, 3) == edge_before, (
                    f"an unreachable press must not move the reported playhead off {edge_before} s")
            seen = _pts(frames)
            assert seen == sorted(seen), f"the playhead never decreases over these presses, got {seen}"

        assert refusals, "and every unreachable press says so in the log"
        assert any("2.000" in message for message in refusals)
        assert any("3.000" in message for message in refusals)
        assert _pts(frames)[-1] == 1.3, "the deepest flushed frame is still reachable"
        assert len(frames) == held + 3, (
            f"three reachable presses publish three frames, nothing more (got {len(frames) - held})")
    finally:
        thread.stop_and_wait()


# -- (c): playing is unchanged ------------------------------------------------------------------------


def test_a_seek_while_playing_still_runs_without_duplicates(qapp, scratch_dir):
    """The 2026-11-23 behaviour: playing, a seek replays from the target and then follows new frames."""
    path = _written(scratch_dir, 10)
    thread = _spawn(path, paused=False)
    frames = []
    thread.frame_ready.connect(frames.append)
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 10), "a playing tail reads the whole written part"
        _settle(qapp)
        assert _pts(frames) == [round(i / FPS, 3) for i in range(10)]

        before = len(frames)
        thread.seek(0.2)
        assert _wait_for(qapp, lambda: len(frames) >= before + 8), "the replay runs from the target to the edge"
        _settle(qapp)
        run = _pts(frames[before:])
        assert run == [round(i / FPS, 3) for i in range(2, 10)], (
            f"a playing seek emits the run from the target, each frame once: got {run}")

        _append(path, 2, start=10)
        assert _wait_for(qapp, lambda: len(frames) >= before + 10), "following continues after the replayed run"
        _settle(qapp)
        assert _pts(frames[before + 8:]) == [1.0, 1.1], (
            f"new frames arrive once, in order: got {_pts(frames[before + 8:])}")
    finally:
        thread.stop_and_wait()


# -- (d): the LMB compare asks at the time of the picture on screen -----------------------------------


def _pane() -> VideoPreviewSurface:
    """The real surface, fed exactly the way `ProcessTab._on_preview_frame` feeds it."""
    widget = VideoPreviewSurface()
    widget.resize(*PANE_SIZE)
    return widget


def _live_tab(pane: VideoPreviewSurface, thread: LiveTailThread) -> ProcessTab:
    """A `ProcessTab` whose only job is the preview frame slot (the `object.__new__` house style)."""
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=pane))
    tab._display = None
    tab.isPreview = True
    tab._live_tail = thread
    tab._preview_decoder = None
    tab._preview_playhead = 0.0
    tab._preview_duration = 0.0
    tab.currentRenderOptions = SimpleNamespace(videoFrameCount=10, videoFps=int(FPS))   # a 1 s clip
    tab.renderProcess = None
    tab.currentFrame = 9
    tab.max_value = 10
    return tab


def _press(widget, point: QPointF) -> None:
    event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(point), QPointF(point),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def _release(widget, point: QPointF) -> None:
    event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(point), QPointF(point),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def test_the_lmb_compare_asks_at_the_pts_of_the_frame_on_screen(qapp, scratch_dir):
    """The press must decode the source at the picture's own time, never a marker ahead of it.

    The pane publishes its playhead from the frame it was just given (`set_frame` then `set_progress` in the same
    slot), so "the pane carries the painted frame's pts" is a property of ONE publisher, not a second piece of state.
    After a held seek that frame is the target, so press and picture agree at the target.
    """
    path = _written(scratch_dir, 10)
    thread = _spawn(path, paused=True)
    frames = []
    thread.frame_ready.connect(frames.append)
    pane = _pane()
    tab = _live_tab(pane, thread)
    thread.start()
    try:
        assert _wait_for(qapp, lambda: len(frames) >= 4)
        _settle(qapp)

        thread.seek(0.4)
        assert _wait_for(qapp, lambda: len(frames) > 4)
        _settle(qapp)
        assert _pts(frames[4:]) == [0.4], "the held seek published its target only"

        # Paint every published frame through the REAL slot, in order, on this (the GUI) thread.
        for frame in frames:
            tab._on_preview_frame(frame)
        on_screen = round(frames[-1].pts, 3)
        assert on_screen == 0.4, f"the pane was last given the frame at 0.4, not {on_screen}"
        assert pane.has_frame() is True

        asked: list[float] = []
        pane.originalRequested.connect(asked.append)
        middle = QPointF(pane.width() / 2.0, pane.height() / 2.0)

        _press(pane, middle)
        _release(pane, middle)

        assert asked == [pytest.approx(on_screen)], (
            f"the LMB compare asked for {asked}, but the picture on screen is the frame at {on_screen} s")
        assert round(pane._progress[2], 3) == on_screen, (
            "the pane's playhead IS the painted frame's pts - the press cannot be ahead of the picture")
    finally:
        thread.stop_and_wait()
