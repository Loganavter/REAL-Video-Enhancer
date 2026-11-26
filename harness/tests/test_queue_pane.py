"""The process page's preview pane follows the INNER TAB, and the Render Queue tab gets its own video.

Owner, 2026-11-24: "Add the same video pane to the Render Queue tab. There a click on a queue row shows the video on
the right ... switching to the inference tab shows the current video with the right seeker position; switching to the
queue tab shows that field EMPTY - unless a queue video was selected before, then it shows that video at its
remembered seeker position. On ANY switch between tabs, playback pauses automatically."

What is pinned here, offscreen and with fakes (no GPU, no `av`, no window):

(a) a switch to the queue tab pauses the transport and, with no remembered selection, CLEARS the pane;
(b) a remembered selection is shown at its remembered playhead, held;
(c) a row click loads that row's file and the playhead is remembered per PATH across a leave/return;
(d) leaving the queue tab pauses and hands the pane back to the render owner at ITS position (the produced edge);
(e) a switch between two NON-queue tabs pauses too (General <-> Encoder Options);
(f) one painter: a render frame arriving while the queue owns the pane never reaches `set_frame`, and a queue frame
    arriving after the queue tab was left does not either;
(g) the queue decoder opens at the PANE's box, not the file's full resolution (the playback-cost cap).

The tab-level tests build `ProcessTab` with `object.__new__` (the house style of `test_display_wiring.py`), so the
methods under test are the shipped ones and nothing else is constructed.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# The app and the backend both live in a `src` namespace package; the suite's conftest puts the backend first. The
# app's `ProcessTab` needs the app's `src.constants`, so the app root goes first for this module (same pattern as
# `test_display_wiring.py` / `test_preview_frame_on_demand.py`).
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

from PySide6.QtGui import QImage  # noqa: E402

from src.ui import video as video_package  # noqa: E402
from src.ui.ProcessTab import ProcessTab  # noqa: E402
from src.ui.video.decode import VideoDecoder  # noqa: E402
from src.ui.video.frames import FrameRef  # noqa: E402

# Leave the suite's path order as it was (the app modules are imported and bound now).
sys.path[:] = _ORIGINAL_SYS_PATH

_QUEUE_TAB = 2  # General, Encoder Options, Render Queue, Advanced, Logs


# --- fakes ------------------------------------------------------------------------------------------


class _Signal:
    """A Qt signal's `connect`/`emit` as the wiring and the fakes use them."""

    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self.slots):
            slot(*args)


class _Pane:
    """The pane API the page touches, recording what reached the screen.

    No strip, no seek signal, no ceiling: the shipped `VideoPreviewSurface` answers none of those any more (audit
    2026-11-27 §6.6), so a double that offers them would model a pane that does not exist. The seek in these tests is
    the page's own route (`_on_preview_seek`), which is what the transport bar's backend calls.
    """

    def __init__(self) -> None:
        self.originalRequested = _Signal()
        self.frames: list = []
        self.progress: list[dict] = []

    def set_frame(self, image, *, gui_ms: float | None = None, index: int | None = None,
                  seq: int | None = None, media_time: float | None = None,
                  generation: int | None = None) -> None:
        self.frames.append(image)

    def set_progress(self, **values) -> None:
        self.progress.append(dict(values))

    def has_frame(self) -> bool:
        return bool(self.frames) and self.frames[-1] is not None

    def view_mode(self) -> str:
        return "single"

    def isVisible(self) -> bool:  # noqa: N802 - Qt spelling
        return True


class _Reader:
    """`DisplayReader` as the page reads it: one wanted byte and the newest-wins seek slot."""

    attached = True

    def __init__(self) -> None:
        self.wants: list[bool] = []
        self.seeks: list[float] = []
        self.generation = 0

    def set_wanted(self, wanted: bool) -> None:
        self.wants.append(bool(wanted))

    def request_seek(self, seconds: float) -> int:
        self.seeks.append(float(seconds))
        self.generation += 1
        return self.generation


class _Process:
    """`subprocess.Popen` as the wiring reads it: `poll()` is None while the render is alive."""

    def __init__(self, alive: bool = False) -> None:
        self.alive = bool(alive)

    def poll(self):
        return None if self.alive else 0


class _Tail:
    """The live tail as the page reads it: a produced edge, a seek, and the pause the transport sets."""

    def __init__(self, produced_seconds: float = 0.0) -> None:
        self.produced_seconds = float(produced_seconds)
        self.seeks: list[float] = []
        self.sought_past_edge: list[bool] = []
        self.paused = True
        self.pause_calls: list[bool] = []

    def seek(self, seconds: float, *, attempt_past_edge: bool = False) -> None:
        self.seeks.append(float(seconds))
        self.sought_past_edge.append(bool(attempt_past_edge))

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        self.pause_calls.append(bool(paused))


class _FakeDecoder:
    """`VideoDecoder` as the queue pane uses it: open/close, one held frame per seek, play/pause.

    The class records every instance it built (the tests read the LAST one), and `seek` emits a frame at the target -
    which is what the real decoder does while paused: one published frame at the requested PTS.
    """

    instances: list["_FakeDecoder"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.frame_ready = _Signal()
        self.opened = _Signal()
        self.seeked = _Signal()
        self.error = _Signal()
        self.path: str | None = None
        self.playing = False
        self.closed = False
        self.seeks: list[float] = []
        self.pause_calls: list[bool] = []
        self._duration = 0.0
        self._pts = 0.0
        _FakeDecoder.instances.append(self)

    def open(self, path: str) -> bool:
        self.path = str(path)
        self._duration = 10.0
        self.opened.emit(self._duration, 24.0, 240.0)
        return True

    def close(self) -> None:
        self.closed = True
        self.playing = False

    def seek(self, seconds: float) -> None:
        self.seeks.append(float(seconds))
        self._pts = float(seconds)
        self.frame_ready.emit(_frame(float(seconds)))

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False
        self.pause_calls.append(True)

    def is_playing(self) -> bool:
        return self.playing

    @property
    def duration(self) -> float:
        return self._duration


def _frame(pts: float) -> FrameRef:
    return FrameRef(buffer=bytes(2 * 2 * 3), width=2, height=2, format="rgb24", stride=6, pts=float(pts),
                    index=0)


def _job(path: str):
    return SimpleNamespace(inputFile=path)


def _tab(*, pane=None, queue=None, paused=False, alive=False, reader=None, tail=None,
         decoder=None) -> ProcessTab:
    tab = object.__new__(ProcessTab)
    pane = pane if pane is not None else _Pane()
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=pane))
    tab.parent = SimpleNamespace(renderQueue=queue)
    tab.isPreview = False
    tab._preview_paused = bool(paused)
    tab.renderProcess = _Process(alive)
    tab._display = reader
    tab._display_box = (960, 540)
    tab.currentRenderOptions = None
    tab.currentFrame = 0
    tab.max_value = 0
    tab._preview_playhead = 0.0
    tab._preview_duration = 10.0
    tab._preview_decoder = decoder
    tab._live_tail = tail
    tab._display_seek_pending = False
    tab._second_surface_pts = None
    tab._preview_strip_wired_for = None
    tab._queue_tab_index = _QUEUE_TAB
    tab._queue_pane_path = None
    tab._queue_pane_positions = {}
    tab._queue_pane_decoder = None
    tab._queue_pane_decoder_path = None
    tab._queue_pane_owns = False
    tab._queue_pane_position = 0.0
    tab._queue_pane_duration = 0.0
    return tab


@pytest.fixture()
def fake_decoder(monkeypatch):
    """Patch the decoder the pane builds: the module object is what the lazy `from .video.decode import` resolves."""
    _FakeDecoder.instances = []
    monkeypatch.setattr(video_package.decode, "VideoDecoder", _FakeDecoder)
    return _FakeDecoder


# --- (a) entering the queue tab: pause, and empty without a remembered selection ----------------------


def test_switching_to_the_queue_tab_pauses_and_clears_the_pane():
    pane = _Pane()
    pane.set_frame(QImage(2, 2, QImage.Format.Format_RGB888))       # the render's standing picture
    decoder = _FakeDecoder()
    decoder.playing = True
    tab = _tab(pane=pane, paused=False, decoder=decoder)

    tab._on_inner_tab_changed(_QUEUE_TAB)

    assert tab._preview_paused is True, "every tab switch holds the picture"
    assert decoder.playing is False
    assert tab._queue_owns_pane() is True
    assert pane.frames[-1] is None, "no remembered selection: the queue tab's field is EMPTY"
    assert pane.progress[-1] == {"rendered": 0, "total": 0, "playhead": 0.0, "duration": 0.0}


# --- (b) a remembered selection comes back at its own playhead ---------------------------------------


def test_a_remembered_queue_selection_is_shown_at_its_position_paused(fake_decoder):
    pane = _Pane()
    tab = _tab(pane=pane, paused=False)
    tab._queue_pane_path = "/in/a.mkv"
    tab._queue_pane_positions["/in/a.mkv"] = 3.5

    tab._on_inner_tab_changed(_QUEUE_TAB)

    decoder = fake_decoder.instances[-1]
    assert decoder.path == "/in/a.mkv"
    assert decoder.seeks == [pytest.approx(3.5)], "the remembered playhead is the frame shown"
    assert decoder.playing is False, "a queue item is shown held, never auto-playing"
    assert len(pane.frames) == 1 and pane.frames[-1] is not None
    assert pane.progress[-1]["playhead"] == pytest.approx(3.5)
    assert pane.progress[-1]["duration"] == pytest.approx(10.0)


# --- (c) a click loads a row; the position is remembered per path ------------------------------------


def test_clicking_a_row_loads_it_and_remembers_the_position_across_a_leave_and_return(fake_decoder):
    pane = _Pane()
    queue = SimpleNamespace(getQueue=lambda: [_job("/in/a.mkv"), _job("/in/b.mkv")])
    tab = _tab(pane=pane, queue=queue, paused=True)
    tab._queue_pane_owns = True

    tab._on_queue_row_changed(1)
    first = fake_decoder.instances[-1]
    assert first.path == "/in/b.mkv"
    assert tab._queue_pane_path == "/in/b.mkv"

    # playback reaches 4.2 s and the pane is held/left there
    tab._on_queue_pane_frame(_frame(4.2))
    assert tab._queue_pane_positions["/in/b.mkv"] == pytest.approx(4.2)

    tab._on_inner_tab_changed(0)                                    # leave: the decoder is released
    assert first.closed is True, "nothing decodes while the queue tab does not own the pane"
    assert tab._queue_pane_positions["/in/b.mkv"] == pytest.approx(4.2)

    tab._on_inner_tab_changed(_QUEUE_TAB)                           # return: same file, same playhead
    second = fake_decoder.instances[-1]
    assert second is not first
    assert second.path == "/in/b.mkv"
    assert second.seeks == [pytest.approx(4.2)]
    assert second.playing is False


# --- (d) leaving: pause, and the render owner repaints at ITS position -------------------------------


def test_leaving_the_queue_tab_pauses_and_the_tail_repaints_at_the_render_head(fake_decoder):
    pane = _Pane()
    tail = _Tail(produced_seconds=1.0)
    tab = _tab(pane=pane, paused=False, tail=tail)
    tab.max_value = 100
    tab.currentFrame = 60                                           # 60 % of a 10 s clip = 6.0 s
    tab._queue_pane_path = "/in/a.mkv"                              # a queue item owned the pane before the leave
    tab._on_inner_tab_changed(_QUEUE_TAB)
    queue_decoder = fake_decoder.instances[-1]
    assert tab._queue_owns_pane() is True

    tab._on_inner_tab_changed(0)

    assert tab._queue_owns_pane() is False
    assert queue_decoder.closed is True
    assert tab._preview_paused is True, "leaving the tab holds the render's picture"
    assert tail.paused is True
    assert tail.seeks == [pytest.approx(6.0)], "the render owner stands at ITS produced edge, not at the queue item's"
    assert tail.sought_past_edge == [True], "the head is the renderer's, not the tail's own stale edge"


def test_leaving_the_queue_tab_asks_the_display_feed_for_the_render_edge(fake_decoder):
    pane = _Pane()
    reader = _Reader()
    tab = _tab(pane=pane, paused=False, reader=reader, alive=True)
    tab.max_value = 100
    tab.currentFrame = 60
    tab._on_inner_tab_changed(_QUEUE_TAB)                           # no remembered selection: empty

    tab._on_inner_tab_changed(0)

    assert reader.seeks == [pytest.approx(6.0)], "the feed is asked for the frame at the render's produced edge"
    assert tab._display_seek_pending is True, "a held pane buys exactly that one frame"


# --- (e) every switch pauses, queue or not -----------------------------------------------------------


def test_a_switch_between_two_non_queue_tabs_pauses(fake_decoder):
    pane = _Pane()
    tail = _Tail(produced_seconds=2.0)
    decoder = _FakeDecoder()
    decoder.path = "/out/preview.mkv"
    decoder.playing = True
    tab = _tab(pane=pane, paused=False, tail=tail, decoder=decoder)
    fake_decoder.instances.clear()                                   # the preview decoder is not a queue item

    tab._on_inner_tab_changed(0)                                    # General
    assert tab._preview_paused is True
    assert decoder.playing is False

    decoder.playing = True
    tab._preview_paused = False
    tab._on_inner_tab_changed(1)                                    # Encoder Options

    assert tab._preview_paused is True, "a switch between two non-queue tabs holds the picture too"
    assert decoder.playing is False
    assert tail.paused is True
    assert fake_decoder.instances == [], "no queue item was ever opened"
    assert pane.frames == [], "and the pane was not cleared: only the queue tab owns its content"


# --- (f) one painter ---------------------------------------------------------------------------------


def test_a_render_frame_does_not_reach_the_pane_while_the_queue_owns_it(fake_decoder):
    pane = _Pane()
    reader = _Reader()
    tail = _Tail(produced_seconds=2.0)
    queue = SimpleNamespace(getQueue=lambda: [_job("/in/a.mkv")])
    tab = _tab(pane=pane, queue=queue, paused=False, reader=reader, tail=tail, alive=True)
    tab.currentRenderOptions = SimpleNamespace(videoFrameCount=240, videoFps=24)
    image = QImage(2, 2, QImage.Format.Format_RGB888)

    tab._on_inner_tab_changed(_QUEUE_TAB)
    tab._on_queue_row_changed(0)
    painted = len(pane.frames)

    tab._on_display_frame(image)                                    # the export feed's frame
    tab._on_preview_frame(_frame(1.0))                              # the tail's frame

    assert len(pane.frames) == painted, "the queue item is the pane's one owner while its tab is current"
    assert reader.wants[-1] is False, "and the renderer is asked to stage nothing"


def test_a_queue_frame_does_not_reach_the_pane_once_the_queue_tab_was_left(fake_decoder):
    pane = _Pane()
    queue = SimpleNamespace(getQueue=lambda: [_job("/in/a.mkv")])
    tab = _tab(pane=pane, queue=queue, paused=True)
    tab._on_inner_tab_changed(_QUEUE_TAB)
    tab._on_queue_row_changed(0)
    decoder = fake_decoder.instances[-1]

    tab._on_inner_tab_changed(0)
    painted = len(pane.frames)

    decoder.frame_ready.emit(_frame(5.0))                           # a queued delivery after the hand-back

    assert len(pane.frames) == painted, "a retired queue decoder is not a painter"


def test_clearing_the_queue_empties_its_tab_but_keeps_the_remembered_path(fake_decoder):
    pane = _Pane()
    items = [_job("/in/a.mkv")]
    queue = SimpleNamespace(getQueue=lambda: list(items))
    tab = _tab(pane=pane, queue=queue, paused=True)
    tab._on_inner_tab_changed(_QUEUE_TAB)
    tab._on_queue_row_changed(0)
    assert pane.frames[-1] is not None

    items.clear()
    tab._on_queue_row_changed(-1)                                   # `setItems`/`clear` emit "nothing current"

    assert pane.frames[-1] is None, "a cleared queue must not leave a picture of an item that is gone"
    assert tab._queue_pane_path == "/in/a.mkv", "the selection memory is the path, not the row"


# --- (g) the decode cap: the pane's box, not the file's full resolution -------------------------------


class _StubFrame:
    """A decoded frame only through the duck-typed surface `_fit` reads (no PyAV needed)."""

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.reformatted: list[tuple[int, int]] = []

    def reformat(self, *, width: int, height: int):
        self.reformatted.append((int(width), int(height)))
        return SimpleNamespace(width=int(width), height=int(height), reformatted=self.reformatted)


def test_the_queue_decoder_caps_the_decode_size_to_the_pane():
    decoder = VideoDecoder(max_size=(960, 540))
    frame = _StubFrame(3840, 2160)

    fitted = decoder._fit(frame)

    assert frame.reformatted == [(960, 540)], "a 4K file decodes at the pane's box, aspect kept"
    assert (fitted.width, fitted.height) == (960, 540)


def test_the_queue_decoder_leaves_a_frame_that_already_fits_alone():
    decoder = VideoDecoder(max_size=(960, 540))
    frame = _StubFrame(640, 360)

    assert decoder._fit(frame) is frame
    assert frame.reformatted == []


def test_without_a_cap_the_decoder_is_unchanged():
    decoder = VideoDecoder()
    frame = _StubFrame(3840, 2160)

    assert decoder._fit(frame) is frame
    assert frame.reformatted == []


def test_the_queue_pane_opens_its_decoder_at_the_pane_box(fake_decoder):
    pane = _Pane()
    queue = SimpleNamespace(getQueue=lambda: [_job("/in/a.mkv")])
    tab = _tab(pane=pane, queue=queue, paused=True)
    tab._queue_pane_owns = True

    tab._on_queue_row_changed(0)

    decoder = fake_decoder.instances[-1]
    assert decoder.kwargs.get("max_size") == (960, 540), (
        "the pane's box is the decode cap: playback must not cost a full-resolution decode")


def test_a_press_on_the_queue_item_seeks_it(app=None):
    """The strip belongs to the pane's owner: with the queue item up, a press seeks THAT decoder.

    Measured live (2026-11-24): the queue item showed its first frame and the strip appeared, but a press at 40 % left
    the playhead at 0.0 - the host's seek route had no queue branch at the top and fell through to the render's owners,
    none of which was alive. The press is one decoder seek at `fraction * duration`, and the position it lands on is what
    the remembered map keeps.
    """
    _FakeDecoder.instances.clear()
    pane = _Pane()
    tab = _tab(pane=pane)
    tab._queue_pane_owns = True
    tab._queue_pane_path = "/films/a.mkv"
    tab._queue_pane_duration = 20.0
    decoder = _FakeDecoder()
    decoder._duration = 20.0
    tab._queue_pane_decoder = decoder

    tab._on_preview_seek(0.40)

    assert decoder.seeks == [8.0], f"the press must reach the queue decoder at 40 % of 20 s, got {decoder.seeks}"
    # The remembered position is written by the frame the decoder publishes (`_on_queue_pane_frame`), not by the press -
    # a press that lands nowhere must not move the map either.


def test_the_finished_preview_decoder_is_capped_to_the_pane_box(fake_decoder):
    """The preview PLAYBACK reads the render's 4K output; it must decode at the box the pane can show.

    Measured live 2026-11-25: the finished playback delivered 24 frames/s but the pane painted only ~9/s, and the
    picture jumped two to three frames per paint. A 2880x2160 frame costs 24.9 MB to pack and 33 MB to convert to
    RGBA and upload; at the pane's box the same frame costs 1.8 MB. The queue decoder has carried this cap since it
    existed - the playback path was the one that did not.
    """
    tab = _tab(paused=True)
    tab._stop_preview_playback = lambda: None
    tab._wire_preview_strip = lambda: None
    tab._sync_preview_transport = lambda: None

    tab._start_preview_playback("/out/preview.mkv")

    decoder = fake_decoder.instances[-1]
    assert decoder.kwargs.get("max_size") == (960, 540), (
        "the playback decoder must be capped to the pane's box, not the file's 4K resolution")
    assert decoder.kwargs.get("max_frames_per_tick") == 1, (
        "one frame per tick: on the GUI thread a burst is painted once and the rest are lost")
    assert decoder.kwargs.get("content_time") is True, "a rendered output is read in content time"
    assert decoder.path == "/out/preview.mkv"
