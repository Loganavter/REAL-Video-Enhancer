"""A preview's strip may address the WHOLE timeline: the renderer can render a frame the tail has not reached.

Measured live (2026-11-24, a slow preview held at 12.5 % produced): a press at 30 % of the strip published nothing and
the only trace was `seek: 30.0% of the strip is past the produced band; nothing to show there`. The press never
reached `ProcessTab._on_preview_seek`, because the pane's `_seekable` compares against the produced share - and even if
it had, `_on_preview_seek` returned early for `isPreview`. Yet the preview's render command carries the same
`--display_state_id`/`--seek_state_id` block an export uses, `RenderVideo._seek_preview` renders any requested
timestamp, and the GUI already writes a newest-wins request for an export.

What is pinned here, offscreen and without a GPU:

* **the reach**: with the block usable, a preview's ceiling is 1.0 (the press is accepted), and a press past the
  produced band writes exactly one `(seconds, generation + 1)` request and sets `wanted`;
* **one painter**: a tail frame arriving while that request is pending does not reach `set_frame`;
* **the picture**: the frame the block publishes is painted (even though `isPreview`), carries the requested playhead,
  and the one-frame loan ends (`wanted` back to 0, pending cleared);
* **hands back on play**: pressing play clears the hold and the next tail frame paints again;
* **an export is unchanged**: a press still takes the export route (and a playing export is not paused by it);
* **an unusable block**: a preview keeps the produced-edge refusal and the log says so.

The tab-level tests build `ProcessTab` with `object.__new__` (the house style of `test_display_wiring.py`), with a fake
pane that records what reached the screen and a fake block reader for the request slot. Two tests drive the real
`VideoPreviewSurface` (the refusal and the ceiling live there) against a real `DisplayReader` block, so the chain
"press -> ceiling -> request" is pinned without fakes.
"""

from __future__ import annotations

import os
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# BOTH the app and the backend live in a `src` directory (namespace packages). The suite's conftest puts the backend
# first, which makes `src.constants` the backend's module - `ProcessTab` needs the app's. The house pattern is
# `test_seek_ceiling.py`/`test_display_wiring.py`; keep it identical so module order cannot change the answer.
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them.
try:  # pragma: no cover - the branch taken depends on the interpreter
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

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.ProcessTab import ProcessTab  # noqa: E402
from src.ui.video.channel import DisplayReader  # noqa: E402
from src.ui.video.frames import FrameRef  # noqa: E402
from src.ui.widgets.video_preview import VideoPreviewSurface  # noqa: E402

# Leave the suite's path order as it was (see `test_seek_ceiling.py`): the app modules are imported and bound now.
sys.path[:] = _ORIGINAL_SYS_PATH

#: 120 frames at 24 fps: a 5 s clip, so 30 % of the strip is 1.5 s and 12 % produced is 0.6 s.
CLIP = SimpleNamespace(videoFrameCount=120, videoFps=24)
PRODUCED_FRAMES = 12.0
TOTAL_FRAMES = 100.0
PANE_SIZE = (800, 600)
FRAME_SIZE = (1600, 900)

_SEGMENT_COUNTER = 0


def _segment_name(slug: str) -> str:
    """A fresh block name; the pid and a counter keep parallel runs out of each other's segments."""
    global _SEGMENT_COUNTER
    _SEGMENT_COUNTER += 1
    return f"/rve-preview-ondemand-{slug}-{os.getpid()}-{_SEGMENT_COUNTER}"


def _picture(width: int, height: int, colour: str = "#204060") -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(colour))
    return image


def _strip_point(surface, fraction: float) -> QPointF:
    strip = surface._strip_rect()
    assert strip.width() > 0, "the strip must be laid out for this test"
    return QPointF(strip.left() + strip.width() * fraction, strip.center().y())


def _press(widget, point: QPointF) -> None:
    event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(point), QPointF(point),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


# --- fakes: the pane, the block reader, the tail ------------------------------------------------------


class _Signal:
    """A Qt signal's `connect`/`emit` as the wiring and the fakes use them."""

    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self.slots):
            slot(*args)


class _FakePane:
    """The pane API the page touches, recording what reached the screen and the ceiling it was given."""

    def __init__(self, visible: bool = True) -> None:
        self.seekRequested = _Signal()
        self.originalRequested = _Signal()
        self.seekBlocked = _Signal()
        self.ceilings: list[float | None] = []
        self.frames: list = []
        self.progress: list[dict] = []
        self._progress = None
        self.visible = bool(visible)

    def set_seek_ceiling(self, fraction) -> None:
        self.ceilings.append(fraction)

    def isVisible(self) -> bool:  # noqa: N802 - Qt spelling
        return self.visible

    def set_frame(self, image, *, gui_ms: float | None = None, index: int | None = None,
                  seq: int | None = None, media_time: float | None = None) -> None:
        self.frames.append(image)

    def set_progress(self, **values) -> None:
        self.progress.append(dict(values))
        self._progress = (values.get("rendered", 0.0), values.get("total", 0.0),
                          values.get("playhead", 0.0), values.get("duration", 0.0))

    def has_frame(self) -> bool:
        return bool(self.frames)

    def view_mode(self) -> str:
        return "single"


class _FakeBlockReader:
    """`DisplayReader` as the page uses it: the newest-wins slot and the one wanted byte, without shared memory.

    `poll` publishes the image it was given (`display_frame`), which is the shape `_tick_display` drives, and
    `request_seek` answers with `generation + 1` exactly as the real block does.
    """

    attached = True

    def __init__(self, generation: int = 0) -> None:
        self.generation = int(generation)
        self.seeks: list[float] = []
        self.seek_past_edge: list[bool] = []
        self.wants: list[bool] = []
        self.display_frame = _Signal()
        self.released = False
        self._next_image = None

    def set_wanted(self, wanted: bool) -> None:
        self.wants.append(bool(wanted))

    def request_seek(self, seconds: float) -> int:
        self.seeks.append(float(seconds))
        self.generation += 1
        return self.generation

    def publish(self, image) -> None:
        """What the renderer does: one frame is waiting for the next poll."""
        self._next_image = image

    def poll(self):
        image, self._next_image = self._next_image, None
        if image is not None:
            self.display_frame.emit(image)
        return image

    def release(self) -> None:
        self.released = True


class _FakeTail:
    """The live tail as the page reads it: a produced edge, a seek, and the pause the transport sets."""

    def __init__(self, produced_seconds: float = 0.0) -> None:
        self.produced_seconds = float(produced_seconds)
        self.seeks: list[float] = []
        self.paused = True
        self.pause_calls: list[bool] = []

    def seek(self, seconds: float, *, attempt_past_edge: bool = False) -> None:
        # The real thread takes the flag: a resume asks for the RENDERER's head, which this tail has not read, so the ask
        # must not be clamped to its own published edge (that clamp is what the live run showed: 2.544 s -> 0.334 s).
        self.seeks.append(float(seconds))
        self.seek_past_edge.append(bool(attempt_past_edge))

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        self.pause_calls.append(bool(paused))


class _Process:
    """`subprocess.Popen` as the wiring reads it: `poll()` is None while the render is alive."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = bool(alive)

    def poll(self):
        return None if self.alive else 0


def _tab(*, pane=None, preview: bool = True, paused: bool = True, alive: bool = True,
         reader=None) -> ProcessTab:
    tab = object.__new__(ProcessTab)
    pane = pane if pane is not None else _FakePane()
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=pane))
    tab.isPreview = bool(preview)
    tab._preview_paused = bool(paused)
    tab.renderProcess = _Process(alive)
    tab._display = reader
    tab._display_seek_pending = False
    tab._preview_strip_wired_for = None
    tab._live_tail = None
    tab._preview_decoder = None
    tab._preview_playhead = 0.0
    tab._preview_duration = 0.0
    tab._seek_watch = None
    tab._second_surface_pts = None
    tab.currentFrame = PRODUCED_FRAMES
    tab.max_value = TOTAL_FRAMES
    tab.currentRenderOptions = CLIP
    if reader is not None and hasattr(reader, "display_frame"):
        reader.display_frame.connect(tab._on_display_frame)
    return tab


def _frame(pts: float) -> FrameRef:
    return FrameRef(buffer=bytes(2 * 2 * 3), width=2, height=2, format="rgb24", stride=6, pts=float(pts),
                    index=0)


# --- (a) the press reaches the renderer ---------------------------------------------------------------


def test_a_preview_press_past_the_produced_band_asks_for_exactly_one_frame():
    """Produced 12 %, press at 30 %: one newest-wins request and one `wanted`, never a tail seek."""
    pane = _FakePane()
    reader = _FakeBlockReader(generation=4)
    tail = _FakeTail(produced_seconds=0.6)
    tab = _tab(pane=pane, preview=True, paused=True, reader=reader)
    tab._live_tail = tail
    tab._wire_preview_strip()

    tab._apply_seek_ceiling()
    assert pane.ceilings[-1] == pytest.approx(1.0), (
        "a preview whose block is usable may seek the whole timeline: the renderer can render that frame")

    pane.seekRequested.emit(0.30)

    assert reader.seeks == [pytest.approx(1.5)], "30 % of the 5 s clip, written once"
    assert reader.generation == 5, "the slot carries generation + 1, never the generation it replaced"
    assert reader.wants[-1] is True, "the renderer must publish the frame the pane asked for"
    assert tab._display_seek_pending is True
    assert tail.seeks == [], "the tail only holds the produced band, so it was not asked for a frame it cannot show"


def test_a_real_press_on_a_preview_strip_writes_one_request_into_the_real_block(qapp):
    """The whole chain, no fakes: ceiling 1.0 -> the press is accepted -> one `(seconds, generation + 1)` write."""
    surface = VideoPreviewSurface()
    surface.resize(*PANE_SIZE)
    surface.set_frame(_picture(*FRAME_SIZE))
    surface.set_progress(rendered=PRODUCED_FRAMES, total=TOTAL_FRAMES)
    tail = _FakeTail(produced_seconds=0.6)
    reader = DisplayReader(_segment_name("block"), box=(2, 2))
    try:
        struct.pack_into("<dI", reader._seek_shm.buf, 0, 0.0, 5)
        tab = _tab(pane=surface, preview=True, paused=True, reader=reader)
        tab._live_tail = tail
        tab._wire_preview_strip()
        tab._apply_seek_ceiling()
        assert surface._seek_ceiling == pytest.approx(1.0)

        _press(surface, _strip_point(surface, 0.30))

        seconds, generation = reader.seek_request()
        assert seconds == pytest.approx(1.5, abs=0.02)
        assert generation == 6, "exactly one write, bumping the generation it read (5)"
        assert reader.block_wanted() == 1
        assert tab._display_seek_pending is True
        assert tail.seeks == []
    finally:
        reader.release()


# --- (b) one painter ---------------------------------------------------------------------------------


def test_a_tail_frame_does_not_paint_over_a_pending_seek_frame():
    pane = _FakePane()
    reader = _FakeBlockReader()
    tab = _tab(pane=pane, preview=True, paused=True, reader=reader)
    tab._live_tail = _FakeTail(produced_seconds=0.6)
    tab._wire_preview_strip()
    pane.seekRequested.emit(0.30)
    assert tab._display_seek_pending is True

    tab._on_preview_frame(_frame(1.4))

    assert pane.frames == [], "the feed owns the pane until the requested frame is up"


# --- (c) the requested frame is the picture ----------------------------------------------------------


def test_the_published_frame_clears_the_pending_hold_and_is_painted():
    pane = _FakePane()
    reader = _FakeBlockReader()
    tab = _tab(pane=pane, preview=True, paused=True, reader=reader)
    tab._live_tail = _FakeTail(produced_seconds=0.6)
    tab._wire_preview_strip()
    pane.seekRequested.emit(0.30)

    image = _picture(4, 2, "#a04020")
    reader.publish(image)
    tab._tick_display()

    assert pane.frames == [image], "the seek frame the block published is the one on the pane"
    assert tab._display_seek_pending is False
    assert reader.wants[-1] is False, "one frame was the whole loan: the held pane costs the renderer nothing again"
    assert pane.progress[-1]["playhead"] == pytest.approx(1.5), (
        "the marker stands on the requested frame, not on the tail's stale edge")
    assert pane.progress[-1]["total"] == pytest.approx(TOTAL_FRAMES)


# --- (d) play hands the pane back ---------------------------------------------------------------------


def test_pressing_play_hands_the_pane_back_to_the_tail():
    pane = _FakePane()
    reader = _FakeBlockReader()
    tail = _FakeTail(produced_seconds=0.6)
    tab = _tab(pane=pane, preview=True, paused=True, reader=reader)
    tab._live_tail = tail
    tab._wire_preview_strip()
    pane.seekRequested.emit(0.30)
    assert tab._display_seek_pending is True

    tab._on_preview_pause_toggled(False)

    assert tab._preview_paused is False
    assert tab._display_seek_pending is False, "the one-frame loan belongs to the hold it was asked in"
    assert tail.paused is False
    assert reader.wants[-1] is False, "a playing preview is painted by its tail, so its feed stays quiet"

    tab._on_preview_frame(_frame(0.8))
    assert len(pane.frames) == 1, "the next tail frame paints again"


def test_a_press_on_a_playing_preview_holds_the_frame_it_shows():
    """A playing tail cannot pass a frame it has not produced, so the press parks the transport on the requested one."""
    pane = _FakePane()
    reader = _FakeBlockReader()
    tail = _FakeTail(produced_seconds=0.6)
    tab = _tab(pane=pane, preview=True, paused=False, reader=reader)
    tab._live_tail = tail
    tab._wire_preview_strip()

    pane.seekRequested.emit(0.30)

    assert tab._preview_paused is True
    assert tail.paused is True
    assert reader.seeks == [pytest.approx(1.5)]
    assert tab._display_seek_pending is True


# --- (e) an export is unchanged -----------------------------------------------------------------------


def test_an_export_press_still_goes_through_the_export_route():
    pane = _FakePane()
    reader = _FakeBlockReader(generation=2)
    tab = _tab(pane=pane, preview=False, paused=True, reader=reader)
    tab._wire_preview_strip()

    pane.seekRequested.emit(0.50)

    assert reader.seeks == [pytest.approx(2.5)]
    assert reader.generation == 3
    assert tab._display_seek_pending is True
    assert reader.wants[-1] is True


def test_a_playing_export_is_not_paused_by_a_press():
    """The export's feed keeps painting the moving band; only the requested frame is injected."""
    pane = _FakePane()
    reader = _FakeBlockReader(generation=1)
    tab = _tab(pane=pane, preview=False, paused=False, reader=reader)
    tab._wire_preview_strip()

    pane.seekRequested.emit(0.50)

    assert reader.seeks == [pytest.approx(2.5)]
    assert tab._preview_paused is False, "an export's transport is not a preview hold"
    assert tab._display_seek_pending is False, "a playing export gets the seek frame through the wanted byte it already has"

    tab._on_display_frame(_picture(2, 2, "#00ff00"))
    assert len(pane.frames) == 1, "the feed keeps painting the export's pane"


# --- (f) an unusable block keeps today's refusal -------------------------------------------------------


def test_a_preview_without_a_usable_block_still_refuses_and_logs(qapp, monkeypatch):
    module = sys.modules[ProcessTab.__module__]
    lines: list[str] = []
    monkeypatch.setattr(module, "log", lines.append)

    surface = VideoPreviewSurface()
    surface.resize(*PANE_SIZE)
    surface.set_frame(_picture(*FRAME_SIZE))
    surface.set_progress(rendered=PRODUCED_FRAMES, total=TOTAL_FRAMES)
    tail = _FakeTail(produced_seconds=0.6)
    tab = _tab(pane=surface, preview=True, paused=True, reader=None)
    tab._live_tail = tail
    tab._wire_preview_strip()
    tab._apply_seek_ceiling()

    assert surface._seek_ceiling is None, "no block means the produced edge is still the honest reach"

    _press(surface, _strip_point(surface, 0.30))

    assert surface._scrubbing is False
    assert tail.seeks == [], "nothing was asked, because nothing could answer"
    assert any("past the produced band" in line for line in lines), (
        "the refusal is legible in log.txt: a press that does nothing and one that cannot are not the same event")
