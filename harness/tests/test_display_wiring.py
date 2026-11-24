"""The two merged lanes wired into the process page: the export feed and the standing original.

Offscreen and without a GPU. What is pinned here is the seam the deleted 640x360 channel left behind:

* **the command** carries the pane's three flags (`--display_state_id`, `--display_size WxH`, `--seek_state_id`) for
  BOTH job kinds, and leaves them out when a caller models a job with no pane (the pre-move golden stays byte-equal);
* **the reader** (`QTcustom.DisplayReader`) turns a hand-written block - 15-byte header plus a payload slot - into a
  `QImage` of the announced size, and copies the slot the writer's `seq` says is complete;
* **`wanted`** follows "pane visible AND transport playing AND a render is publishing", and a pause clears it;
* **a strip press during an export** writes `(seconds, generation + 1)` into the seek block - never `(seconds,
  generation)` - and writes nothing when no render is alive;
* **one painter**: a display frame is dropped while a preview's tail owns the pane, and a tail frame is dropped while
  the export's display feed owns it.

The tests that touch `ProcessTab` build the object with `object.__new__`: the wiring they exercise is a handful of
plain methods, and constructing the whole window would buy nothing but a slower, flakier test. `cpuinfo` and
`pypresence` are probed, not required: the app imports them at module scope, the harness interpreter may not have them.
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
# BOTH the app and the backend live in a `src` directory (they are namespace packages). The suite's conftest puts the
# backend first, which makes `src.constants` the backend's module - the app's `QTcustom`/`ProcessTab` need THEIR
# `src.constants` (it owns `has_network` and the pid-scoped names), so the app root goes first for this module. Backend
# modules (`src.utils`, `src.pytorch`, ...) do not exist under the app root, so they still resolve.
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
# The backend owns a `src.constants` too (backend detection), and whichever module was imported first is cached under
# that name for the rest of the session. Drop that cache entry so the app's `constants` (has_network, the pid-scoped
# shared-memory names) is the one `QTcustom` and `ProcessTab` resolve. Nothing imports the backend's by that name.
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them, so this module never
# shadows a real install for the rest of the suite.
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

from PySide6.QtGui import QColor, QImage  # noqa: E402

from src.ui import shared_flag  # noqa: E402
from src.ui.ProcessTab import ProcessTab  # noqa: E402
from src.ui.video.channel import (  # noqa: E402
    DEFAULT_DISPLAY_BOX,
    DISPLAY_FMT_RGB888,
    DISPLAY_TIME_UNSET,
    DISPLAY_HEADER,
    DISPLAY_HEADER_SIZE,
    DISPLAY_OFFSET_SEQ,
    DISPLAY_OFFSET_WANTED,
    DISPLAY_STATE_SHARED_MEMORY_ID,
    SEEK_STATE_SHARED_MEMORY_ID,
    DisplayReader,
)
from src.ui.render_command import RenderPaths, build_command  # noqa: E402
from src.ui.video.frames import FrameRef  # noqa: E402
from src.ui.RenderQueue import RenderOptions  # noqa: E402

# Leave the suite's path order as it was: these modules are imported (and bound) now; the app's `src.constants` stays
# cached for their lazy imports, while backend `src.utils`/`src.pytorch` resolve from the backend root as before.
sys.path[:] = _ORIGINAL_SYS_PATH

_SEGMENT_COUNTER = 0


def _segment_name(slug: str) -> str:
    """A fresh segment name; the pid and a counter keep parallel runs out of each other's blocks."""
    global _SEGMENT_COUNTER
    _SEGMENT_COUNTER += 1
    return f"/rve-display-wiring-{slug}-{os.getpid()}-{_SEGMENT_COUNTER}"


def _drop_block(block) -> None:
    """Close a test block the reader may already have unlinked, without upsetting the resource tracker."""
    if block is None:
        return
    try:
        block.close()
    except Exception:
        pass
    try:
        block.unlink()
    except FileNotFoundError:
        pass


# --- the command -----------------------------------------------------------------------------------

SETTINGS = {
    "precision": "float16",
    "tensorrt_optimization_level": "3",
    "ncnn_gpu_id": "0",
    "pytorch_gpu_id": "1",
    "auto_border_cropping": "False",
    "dynamic_tensorrt_engine": "False",
    "quality_preset": "best",
    "temporal_reuse": "True",
    "temporal_reuse_statistic": "mean_abs",
    "preview_enabled": "True",
    "scene_change_detection_enabled": "True",
    "scene_change_detection_method": "sudo_scene_detect",
    "scene_change_detection_threshold": "3.5",
    "uhd_mode": "False",
}


class _Settings:
    def __init__(self):
        self.settings = dict(SETTINGS)
        self.allowedSettings = {"quality_preset": ("best", "fast")}


def _options(*, is_preview: bool) -> RenderOptions:
    return RenderOptions(
        inputFile="/in/clip.mkv", outputPath="/out/clip.mkv", videoWidth=1920, videoHeight=1080,
        videoFps=24, tilingEnabled=False, tilesize="256", videoFrameCount=120, backend="ncnn",
        modelScale=2, interpolateTimes=2, benchmarkMode=False, sloMoMode=False,
        dyanmicScaleOpticalFlow=False, ensemble=False, isPreview=is_preview,
    )


def _paths(**overrides) -> RenderPaths:
    values = dict(
        python="/opt/rve/python/bin/python",
        backend_path="/opt/rve/backend",
        cwd="/opt/rve",
        ffmpeg_path="/opt/rve/bin/ffmpeg",
        models_path="/opt/rve/models",
        custom_models_path="/opt/rve/models/custom",
        paused_shared_memory_id="/paused_state1",
    )
    values.update(overrides)
    return RenderPaths(**values)


def _command(*, is_preview: bool, paths: RenderPaths) -> list[str]:
    return build_command(_options(is_preview=is_preview), _Settings(), paths=paths, notice=lambda _message: None)


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command, f"{flag} is missing from the command"
    index = command.index(flag)
    assert index + 1 < len(command), f"{flag} has no value"
    return command[index + 1]


def test_the_command_carries_the_pane_flags_and_size_for_both_job_kinds():
    paths = _paths(display_state_id="/image_display77", display_size="640x360",
                   seek_state_id="/seek_state77")
    for is_preview in (False, True):
        command = _command(is_preview=is_preview, paths=paths)
        assert _flag_value(command, "--display_state_id") == "/image_display77"
        assert _flag_value(command, "--display_size") == "640x360"
        assert _flag_value(command, "--seek_state_id") == "/seek_state77"
        # the preview/export difference is still only the preview flag
        assert ("--preview_render" in command) is is_preview


def test_the_pane_flags_are_absent_when_the_paths_model_no_pane():
    """The pre-move golden in `test_render_command.py` builds paths without a pane; it must stay byte-equal."""
    command = _command(is_preview=False, paths=_paths())
    assert "--display_state_id" not in command
    assert "--display_size" not in command
    assert "--seek_state_id" not in command


def test_the_process_tab_points_the_command_at_its_own_block():
    tab = object.__new__(ProcessTab)
    tab._display_box = (1280, 720)
    paths = tab._render_paths()
    assert paths.display_state_id == DISPLAY_STATE_SHARED_MEMORY_ID
    assert paths.seek_state_id == SEEK_STATE_SHARED_MEMORY_ID
    assert paths.display_size == "1280x720"


# --- the reader ------------------------------------------------------------------------------------

def _write_frame(block, *, seq: int, width: int, height: int, pixels: bytes) -> None:
    """Write a header and one payload the way the renderer does: payload into `seq & 1`... then seq + 1.

    The helper's callers pass the POST-publish `seq`, so the slot is `(seq - 1) & 1` - the writer fills the old
    counter's slot and only then bumps it (`DisplayChannel.publish`).
    """
    slot = (seq - 1) & 1
    slot_size = width * height * 3
    start = DISPLAY_HEADER_SIZE + slot * slot_size
    assert len(pixels) == slot_size
    block.buf[start:start + slot_size] = pixels
    # the 8th field is the picture's media time; the stand-ins here publish without one, so the sentinel
    DISPLAY_HEADER.pack_into(block.buf, 0, seq, width, height, DISPLAY_FMT_RGB888, 0, 0, 0, DISPLAY_TIME_UNSET)


def test_the_reader_turns_a_hand_written_block_into_a_qimage_of_the_announced_size():
    width, height = 4, 2
    name = _segment_name("read")
    size = DISPLAY_HEADER_SIZE + 2 * width * height * 3
    block = shared_flag.create(name, size)
    reader = None
    try:
        pixels = bytes(((index * 17) & 0xFF) for index in range(width * height * 3))
        _write_frame(block, seq=1, width=width, height=height, pixels=pixels)
        reader = DisplayReader(name, box=(width, height))
        assert reader.attached
        seen = []
        reader.display_frame.connect(seen.append)

        # Nobody asked: the renderer is not to be read, and the pane draws nothing.
        assert reader.poll() is None

        reader.set_wanted(True)
        assert block.buf[DISPLAY_OFFSET_WANTED] == 1
        image = reader.poll()
        assert isinstance(image, QImage)
        assert seen == [image]
        assert (image.width(), image.height()) == (width, height)
        assert image.pixelColor(0, 0) == QColor(pixels[0], pixels[1], pixels[2])
        assert image.pixelColor(width - 1, height - 1) == QColor(*pixels[-3:])
        # the reader re-asserts the request byte after a frame
        assert block.buf[DISPLAY_OFFSET_WANTED] == 1

        # An unchanged sequence is one read and no picture: one `seq` per tick at most.
        assert reader.poll() is None

        # The next publish lands in the OTHER slot; the reader must follow the counter, not slot 0 forever.
        second = bytes(((index * 31 + 5) & 0xFF) for index in range(width * height * 3))
        _write_frame(block, seq=2, width=width, height=height, pixels=second)
        image = reader.poll()
        assert image is not None
        assert image.pixelColor(0, 0) == QColor(second[0], second[1], second[2])
    finally:
        if reader is not None:
            reader.release()
        _drop_block(block)


def test_the_reader_and_the_real_writer_and_seek_reader_agree_byte_for_byte():
    """The cross-lane pin: the reader's slots and the seek request are read by the SHIPPED backend classes."""
    from src.utils.DisplayChannel import DisplayChannel, SeekState

    name = _segment_name("writer")
    reader = DisplayReader(name, box=(3, 2), seek_name=name + "_seek")
    channel = None
    seek_state = None
    try:
        channel = DisplayChannel(name, box=(3, 2))
        assert channel.attached
        reader.set_wanted(True)
        first = bytes(range(18))
        assert channel.publish(first, 3, 2) is True
        image = reader.poll()
        assert image is not None
        assert image.pixelColor(0, 0) == QColor(first[0], first[1], first[2])
        second = bytes(((value + 90) & 0xFF) for value in range(18))
        assert channel.publish(second, 3, 2) is True
        image = reader.poll()
        assert image is not None
        assert image.pixelColor(0, 0) == QColor(second[0], second[1], second[2])

        # The seek answer: the real writer names the generation it answers, the real reader reports it beside the
        # picture. Without it the pane's only rule was "the first picture with a media time", and the streaming path
        # names every frame it draws (measured 2026-11-24: a press at 1.888 s was closed by the frame at 4.046 s).
        third = bytes(((value + 180) & 0xFF) for value in range(18))
        assert channel.publish(third, 3, 2, time_s=1.888, seek_generation=2) is True
        assert channel.published_seek_generation() == 2
        assert reader.poll() is not None
        assert reader.seek_answer == 2, "the answered generation rides in the same header as the picture"
        fourth = bytes(((value + 45) & 0xFF) for value in range(18))
        assert channel.publish(fourth, 3, 2, time_s=4.046) is True
        assert reader.poll() is not None
        assert reader.seek_answer is None, "a streaming frame answers no request, even when it names its time"

        seek_state = SeekState(name + "_seek")
        assert seek_state.attached
        assert seek_state.pending() is None
        assert reader.request_seek(3.5) == 1
        assert seek_state.pending() == pytest.approx(3.5)
        assert reader.request_seek(7.25) == 2
        assert seek_state.pending() == pytest.approx(7.25)
        assert seek_state.pending() is None, "one slot, newest wins: an acted-on generation is not repeated"
    finally:
        if seek_state is not None:
            seek_state.close()
        if channel is not None:
            channel.close()
        reader.release()


def test_the_reader_follows_the_wanted_byte_it_is_given():
    name = _segment_name("wanted")
    block = shared_flag.create(name, DISPLAY_HEADER_SIZE + 2 * 8 * 8 * 3)
    reader = None
    try:
        _write_frame(block, seq=1, width=8, height=8, pixels=bytes(8 * 8 * 3))
        reader = DisplayReader(name, box=(8, 8))
        reader.set_wanted(True)
        assert block.buf[DISPLAY_OFFSET_WANTED] == 1
        reader.set_wanted(False)
        assert block.buf[DISPLAY_OFFSET_WANTED] == 0
        assert reader.poll() is None
    finally:
        if reader is not None:
            reader.release()
        _drop_block(block)


# --- the page's side -------------------------------------------------------------------------------

class _Pane:
    """The pane's two-surface API, without a window."""

    def __init__(self, visible: bool = True) -> None:
        self.visible = visible
        self.frames = []
        self.progress = []
        self.mode = "single"

    def isVisible(self) -> bool:  # noqa: N802 - Qt spelling
        return bool(self.visible)

    def set_frame(self, image, *, gui_ms: float | None = None, index: int | None = None,
                  seq: int | None = None, media_time: float | None = None) -> None:
        self.frames.append(image)

    def set_progress(self, **values) -> None:
        self.progress.append(values)

    def view_mode(self) -> str:
        return self.mode

    def has_frame(self) -> bool:
        return bool(self.frames)


class _Process:
    """`subprocess.Popen` as the wiring reads it: `poll()` is None while the render is alive."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive

    def poll(self):
        return None if self.alive else 0


class _Reader:
    """The parts of `DisplayReader` the page uses, with the byte writes recorded."""

    attached = True

    def __init__(self) -> None:
        self.wants = []
        self.seeks = []
        self.generation = 0

    def set_wanted(self, wanted: bool) -> None:
        self.wants.append(bool(wanted))

    def request_seek(self, seconds: float) -> int:
        self.seeks.append(float(seconds))
        self.generation += 1
        return self.generation


def _tab(*, pane=None, preview=False, paused=False, alive=True, reader=None) -> ProcessTab:
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=pane or _Pane()))
    tab.isPreview = bool(preview)
    tab._preview_paused = bool(paused)
    tab.renderProcess = _Process(alive)
    tab._display = reader
    tab.currentFrame = 0
    tab.max_value = 0
    tab._preview_playhead = 0.0
    tab._preview_duration = 0.0
    tab.currentRenderOptions = None
    tab._live_tail = None
    tab._preview_decoder = None
    tab._second_surface_pts = None
    return tab


def test_wanted_follows_a_visible_playing_pane_and_stops_when_paused():
    reader = _Reader()
    pane = _Pane(visible=True)
    tab = _tab(pane=pane, preview=False, paused=False, alive=True, reader=reader)

    tab._update_display_wanted()
    assert reader.wants[-1] is True

    tab._on_preview_pause_toggled(True)
    assert reader.wants[-1] is False, "a held picture must clear the request byte"
    tab._on_preview_pause_toggled(False)
    assert reader.wants[-1] is True

    pane.visible = False
    tab._update_display_wanted()
    assert reader.wants[-1] is False, "a pane that is not on screen costs the renderer nothing"

    pane.visible = True
    tab.renderProcess = _Process(alive=False)
    tab._update_display_wanted()
    assert reader.wants[-1] is False, "no live render means no frames to ask for"


def test_a_preview_never_asks_the_renderer_for_display_frames():
    reader = _Reader()
    tab = _tab(preview=True, paused=False, reader=reader)
    tab._update_display_wanted()
    assert reader.wants[-1] is False, "a preview's tail owns the pane; the display feed must not compete"


def test_a_strip_press_during_an_export_asks_for_the_next_generation():
    reader = DisplayReader(_segment_name("seek"), box=(2, 2))
    try:
        struct.pack_into("<dI", reader._seek_shm.buf, 0, 0.0, 5)
        tab = _tab(preview=False, paused=False, alive=True)
        tab._display = reader
        tab.currentRenderOptions = SimpleNamespace(videoFrameCount=100, videoFps=10)  # 10 s clip

        tab._on_preview_seek(0.5)

        seconds, generation = reader.seek_request()
        assert seconds == pytest.approx(5.0)
        assert generation == 6, "the writer must bump the generation it read (5), never repeat it"
        assert generation != 5

        # Nothing renders: the block is left alone (no process would answer the request).
        tab.renderProcess = _Process(alive=False)
        tab._on_preview_seek(0.9)
        assert reader.seek_request() == pytest.approx((5.0, 6))
    finally:
        reader.release()


def test_a_seek_while_paused_buys_exactly_that_one_frame():
    """A held pane has `wanted` 0; the requested frame is the one exception, then the hold resumes."""
    pane = _Pane()
    reader = _Reader()
    tab = _tab(pane=pane, paused=True, reader=reader)
    tab.currentRenderOptions = SimpleNamespace(videoFrameCount=100, videoFps=10)
    tab.max_value = 100
    tab.currentFrame = 50

    tab._on_preview_seek(0.5)
    assert reader.seeks == [pytest.approx(5.0)]
    assert reader.wants[-1] is True, "the renderer must publish the frame the pane asked for"
    assert tab._display_seek_pending is True

    tab._on_display_frame(QImage(2, 2, QImage.Format.Format_RGB888))
    assert pane.frames, "a press while held still shows its target"
    assert tab._display_seek_pending is False
    assert reader.wants[-1] is False, "after that one frame the hold costs the renderer nothing again"


def test_a_preview_without_a_usable_block_never_writes_the_renderer_seek_block():
    """No block, no answer: the tail is all there is, and a press it cannot serve writes nothing.

    A preview WITH a usable block is the frame-on-demand case (`test_preview_frame_on_demand.py`): the block renders
    the requested frame, so that press does write a request.
    """
    reader = _Reader()
    tab = _tab(preview=True, paused=False, alive=True, reader=reader)
    reader.attached = False
    tab.currentRenderOptions = SimpleNamespace(videoFrameCount=100, videoFps=10)
    tab._on_preview_seek(0.5)
    assert reader.seeks == []


def test_a_display_frame_is_dropped_while_a_previews_tail_owns_the_pane():
    pane = _Pane()
    reader = _Reader()
    image = QImage(2, 2, QImage.Format.Format_RGB888)

    preview = _tab(pane=pane, preview=True, paused=False, reader=reader)
    preview._on_display_frame(image)
    assert pane.frames == [], "the tail is the preview's only painter"

    export = _tab(pane=pane, preview=False, paused=False, reader=reader)
    export._on_display_frame(image)
    assert pane.frames == [image]
    assert pane.progress and pane.progress[-1]["total"] == export.max_value
    assert pane.progress[-1]["rendered"] == export.currentFrame


def test_a_tail_frame_is_dropped_while_the_exports_display_feed_owns_the_pane():
    pane = _Pane()
    reader = _Reader()
    tab = _tab(pane=pane, preview=False, paused=False, reader=reader)
    frame = FrameRef(buffer=bytes(2 * 2 * 3), width=2, height=2, format="rgb24", stride=6, pts=0.0,
                     index=0)
    tab._on_preview_frame(frame)
    assert pane.frames == [], "one picture, one painter: the display feed wins an export's pane"


# --- one painter, decided before the job starts ----------------------------------------------------


def _job(*, preview: bool = False, path: str = "/tmp/out.mkv"):
    """The shape `_tail_should_run` reads from a queued job (the same fields `_followable` uses)."""
    return SimpleNamespace(isPreview=bool(preview), outputPath=path)


def _painters(*, preview: bool, reader) -> ProcessTab:
    tab = _tab(preview=preview, paused=False, reader=reader)
    tab.settings = SimpleNamespace(settings={"preview_enabled": "True"})
    return tab


def test_an_export_whose_pane_is_fed_by_the_display_starts_no_tail():
    """The invariant is not "the display wins by arriving later": this job's tail is never built."""
    tab = _painters(preview=False, reader=_Reader())
    assert tab._tail_should_run(_job()) is False, (
        "an export whose pane the display block feeds must not also decode its own growing file")


def test_a_preview_still_follows_its_own_file():
    """The preview's tail IS the picture (and the strip's seek lives on it), so it always starts."""
    tab = _painters(preview=True, reader=_Reader())
    assert tab._tail_should_run(_job(preview=True, path="/tmp/preview.mp4")) is True


def test_an_export_without_a_usable_display_block_keeps_the_tail_as_its_fallback():
    """A pane with no block would otherwise stay black on a plain render (the reason the tail was added)."""
    tab = _painters(preview=False, reader=None)
    assert tab._tail_should_run(_job()) is True


def test_an_export_file_the_tail_cannot_follow_starts_no_tail():
    tab = _painters(preview=False, reader=None)
    assert tab._tail_should_run(_job(path="/tmp/out.mp4")) is False, "a growing mp4 has its index at the end"


def test_the_preview_switch_off_starts_no_tail():
    tab = _painters(preview=False, reader=None)
    tab.settings = SimpleNamespace(settings={"preview_enabled": "False"})
    assert tab._tail_should_run(_job()) is False


def test_a_tail_frame_from_a_tail_that_is_gone_does_not_reach_the_surface():
    """`frame_ready` is queued: a stopped tail can still deliver a frame, and it must not paint over the picture."""
    pane = _Pane()
    tab = _tab(pane=pane, preview=False, paused=False, alive=False, reader=None)
    frame = FrameRef(buffer=bytes(2 * 2 * 3), width=2, height=2, format="rgb24", stride=6, pts=8.97,
                     index=0)
    tab._on_preview_frame(frame)
    assert pane.frames == [], "a retired tail is not a painter"


def test_a_preview_paints_from_its_tail():
    """The other side of the same gate: a live preview's tail is the picture."""
    pane = _Pane()
    tab = _tab(pane=pane, preview=True, paused=False, alive=True, reader=_Reader())
    tab._live_tail = SimpleNamespace(produced_seconds=0.0)
    frame = FrameRef(buffer=bytes(2 * 2 * 3), width=2, height=2, format="rgb24", stride=6, pts=1.0,
                     index=0)
    tab._on_preview_frame(frame)
    assert len(pane.frames) == 1
    assert pane.progress and pane.progress[-1]["playhead"] == pytest.approx(1.0)


def test_a_paused_export_keeps_its_produced_band_honest_without_a_tail():
    """Removing the export's tail must not freeze the strip while the held pane waits: the feed paints, so the band
    still follows the renderer's counters (the picture's playhead stays where the hold left it)."""
    pane = _Pane()
    pane._progress = (0, 100, 5.0, 10.0)        # what the last display frame painted
    tab = _tab(pane=pane, preview=False, paused=True, reader=_Reader())
    tab.max_value = 100
    tab.currentFrame = 60

    tab._refresh_held_progress()

    assert pane.progress[-1] == {"rendered": 60, "total": 100, "playhead": 5.0, "duration": 10.0}


# --- the job's end: no painter outlives it ---------------------------------------------------------


class _Signal:
    def connect(self, slot) -> None:  # noqa: D401 - Qt spelling
        pass

    def disconnect(self) -> None:
        pass


class _Button:
    def __init__(self) -> None:
        self.clicked = _Signal()
        self.enabled = None
        self.visible = None

    def setEnabled(self, value) -> None:  # noqa: N802 - Qt spelling
        self.enabled = bool(value)

    def setVisible(self, value) -> None:  # noqa: N802 - Qt spelling
        self.visible = bool(value)


class _Text:
    def setText(self, text) -> None:  # noqa: N802 - Qt spelling
        self.text = str(text)

    def setEnabled(self, value) -> None:  # noqa: N802 - Qt spelling
        self.enabled = bool(value)


class _Tail:
    """A live tail as the completion path sees it: `stop_and_wait` is the whole contract there."""

    def __init__(self, events) -> None:
        self.events = events
        self.stops = 0
        self.produced_seconds = 0.0

    def stop_and_wait(self) -> None:
        self.stops += 1
        self.events.append("tail")


class _ReleasingReader(_Reader):
    """The pane's block; `release()` is recorded so the ordering against the tail is visible."""

    def __init__(self, events) -> None:
        super().__init__()
        self.events = events

    def release(self) -> None:
        self.events.append("block")


def _completion_tab(*, preview: bool, reader, tail) -> ProcessTab:
    tab = _tab(pane=_Pane(), preview=preview, paused=False, alive=False, reader=reader)
    tab._live_tail = tail
    tab.return_codes = [0]
    tab.userKilled = False
    tab._render_started_mono = 1.0
    tab.currentRenderOptions = SimpleNamespace(isPreview=preview, outputPath="/tmp/out.mkv")
    tab.parent = SimpleNamespace(enableProcessPage=lambda: None, startRender=lambda: None)
    tab.setProgressDetail = lambda text: None
    tab.setStatus = lambda status: None
    tab.syncRenderConsole = lambda: None
    tab._restoreQueueAfterPreview = lambda: None
    tab._start_preview_playback = lambda path: None
    proc = tab.page_handles.proc
    proc.onRenderButtonsContiainer = object()
    proc.startRenderButton = _Button()
    proc.FPS = _Text()
    proc.ETA = _Text()
    tab.page_handles.general = SimpleNamespace(outputFileText=_Text(), inputFileText=_Text())
    return tab


def test_a_plain_exports_completion_stops_its_tail_before_releasing_the_block(monkeypatch):
    """Measured live (2026-11-24): the tail outlived its export and painted a lagging cluster over the last frame."""
    events = []
    reader = _ReleasingReader(events)
    tail = _Tail(events)
    tab = _completion_tab(preview=False, reader=reader, tail=tail)
    # The completion path re-states the row through the glue's single writer; the stub tab has no widgets to
    # enable, so the writer is what gets neutralised (it used to be the `hide_layout_widgets` call it replaced).
    monkeypatch.setattr(ProcessTab, "_set_render_controls", lambda self, state: None)

    tab.guiChangesOnRenderCompletion()

    assert tail.stops == 1, "a tail left running keeps decoding a file nobody writes"
    assert tab._live_tail is None
    assert events == ["tail", "block"], "the tail goes first; the block it would paint over goes second"
    assert tab._display is None


def test_a_previews_completion_stops_the_tail_before_playback_starts(monkeypatch):
    events = []
    reader = _ReleasingReader(events)
    tail = _Tail(events)
    tab = _completion_tab(preview=True, reader=reader, tail=tail)
    tab._start_preview_playback = lambda path: events.append("playback")
    # The completion path re-states the row through the glue's single writer; the stub tab has no widgets to
    # enable, so the writer is what gets neutralised (it used to be the `hide_layout_widgets` call it replaced).
    monkeypatch.setattr(ProcessTab, "_set_render_controls", lambda self, state: None)

    tab.guiChangesOnRenderCompletion()

    assert tail.stops == 1
    assert events == ["tail", "block", "playback"], "the finished file plays after the tail that followed it stops"


def test_the_second_surface_asks_the_source_decoder_for_the_result_playhead():
    """In `sbs`/`split` the standing original follows the result on the pane's clock, one decode per moved playhead."""
    pane = _Pane()
    pane.mode = "sbs"
    pane.frames.append(QImage(2, 2, QImage.Format.Format_RGB888))       # a result exists
    reader = _Reader()
    tab = _tab(pane=pane, preview=False, paused=False, alive=True, reader=reader)
    tab.currentRenderOptions = SimpleNamespace(videoFrameCount=100, videoFps=10)  # 10 s
    tab.max_value = 100
    tab.currentFrame = 50
    asked = []
    tab._on_original_requested = lambda seconds: asked.append(float(seconds))

    tab._tick_second_surface()
    tab._tick_second_surface()
    assert asked == [pytest.approx(5.0)], "the same playhead is one decode, not one per tick"

    tab.currentFrame = 75
    tab._tick_second_surface()
    assert asked[-1] == pytest.approx(7.5)

    # `single` keeps today's behaviour: the source appears only while LMB is held, so the clock does not ask.
    pane.mode = "single"
    tab.currentFrame = 90
    tab._tick_second_surface()
    assert len(asked) == 2


def test_the_default_box_is_the_probe_constant():
    assert DEFAULT_DISPLAY_BOX == (960, 540)
    assert DISPLAY_HEADER_SIZE == 23      # + the 8-byte media time
    assert DISPLAY_OFFSET_SEQ == 0
    assert DISPLAY_OFFSET_WANTED == 9


# --- the merged transport: the glyph must never outlive the state ------------------------------------------------

class _WidgetStub:
    """The three calls `_set_render_controls` makes on a button; nothing else is needed to test the state machine."""

    def __init__(self) -> None:
        self.visible = False
        self.enabled = False
        self.checked = False

    def setVisible(self, value) -> None:  # noqa: N802 - Qt spelling
        self.visible = bool(value)

    def setEnabled(self, value) -> None:  # noqa: N802 - Qt spelling
        self.enabled = bool(value)

    def setChecked(self, value) -> None:  # noqa: N802 - Qt spelling
        self.checked = bool(value)


def test_a_failed_start_puts_the_transport_glyph_back():
    """The transport is a toggle, so a press flips it before the handler; a start that does nothing must un-flip it.

    Measured live 2026-11-24 on a fresh instance with no clip picked: pressing the transport left the pause glyph on a
    row where nothing ran, and the cancel button stayed disabled - the toggle lied about the state.
    """
    class _Proc:
        def __init__(self) -> None:
            self.startRenderButton = _WidgetStub()
            self.killRenderButton = _WidgetStub()
            self.onRenderButtonsContiainer = types.SimpleNamespace(setEnabled=lambda value: None)

    class _Parent:
        def startRender(self) -> None:      # a start that never reaches the running state (no file chosen)
            return

    tab = types.SimpleNamespace(
        page_handles=types.SimpleNamespace(proc=_Proc()), parent=_Parent(),
        _render_state="idle", _applying_controls=False,
        RENDER_CONTROLS=ProcessTab.RENDER_CONTROLS,     # a class attribute the stub does not inherit
    )
    # the `finally` writes the state back through the same method; bind it to the stub
    tab._set_render_controls = lambda state, _tab=tab: ProcessTab._set_render_controls(_tab, state)
    ProcessTab._set_render_controls(tab, "idle")
    assert tab.page_handles.proc.startRenderButton.checked is False
    assert tab.page_handles.proc.killRenderButton.enabled is False

    tab.page_handles.proc.startRenderButton.checked = True      # what the toggle does before the handler runs
    ProcessTab._on_transport_pressed(tab)

    assert tab.page_handles.proc.startRenderButton.checked is False, (
        "the glyph claims a job that never started")
    assert tab.page_handles.proc.killRenderButton.enabled is False


def test_the_transport_glyph_follows_the_running_state():
    """Checked == a job is running (the pause glyph); the same writer every transition uses."""
    class _Proc:
        def __init__(self) -> None:
            self.startRenderButton = _WidgetStub()
            self.killRenderButton = _WidgetStub()
            self.onRenderButtonsContiainer = types.SimpleNamespace(setEnabled=lambda value: None)

    tab = types.SimpleNamespace(page_handles=types.SimpleNamespace(proc=_Proc()), parent=None,
                                _render_state="idle", _applying_controls=False,
                                RENDER_CONTROLS=ProcessTab.RENDER_CONTROLS)
    for state, checked, cancel in (("idle", False, False), ("running", True, True), ("paused", False, True)):
        ProcessTab._set_render_controls(tab, state)
        assert tab.page_handles.proc.startRenderButton.checked is checked, state
        assert tab.page_handles.proc.killRenderButton.enabled is cancel, state
        assert tab.page_handles.proc.startRenderButton.enabled is True, state


# --- a seek closes on the picture that names itself, not on the next stream frame ------------------------------

#: The reader never grew the field: a stub, or a backend older than the header change.
_NO_FIELD = object()


def _seek_tab(media_time, seek_answer=_NO_FIELD, seek_generation=None, target_s=1.139):
    """The smallest tab `_on_display_frame` needs, with a display reader that has (or has not) a media time.

    `seek_answer` has three cases and they are NOT the same: an int is the generation the picture answers, `None` is a
    streaming frame from a producer that names generations, and `_NO_FIELD` is a producer that cannot say - where the
    pane has nothing but the media time to go on.
    """
    calls = []

    class _Pane:
        def set_frame(self, image, *, gui_ms=None, index=None, seq=None, media_time=None):
            calls.append(("frame", seq, media_time))
            self._shown = ("seq", seq) if seq is not None else None
            self._shown_time = media_time

        def shown(self):
            return getattr(self, "_shown", None)

        def shown_time(self):
            return getattr(self, "_shown_time", None)

    class _Reader:
        def __init__(self, stamp, answer=_NO_FIELD):
            self._stamp = stamp
            self.seq = 196
            if answer is not _NO_FIELD:
                self.seek_answer = answer

        @property
        def media_time(self):
            return self._stamp

    pane = _Pane()
    tab = types.SimpleNamespace(
        page_handles=types.SimpleNamespace(proc=types.SimpleNamespace(previewLabel=pane)),
        _display=_Reader(media_time, seek_answer), _seek_generation=seek_generation,
        _display_seek_pending=True, _seek_pressed_at=0.0,
        _seek_target_frame=27, _seek_target_s=target_s,
        _preview_paused=False, _display_paints_pane=lambda: True,
        _result_playhead=lambda: 0.0, _clip_duration=lambda: 20.646,
    )
    # The seek rules themselves, bound to the fake tab: stubbing them would pin the call, not the rule. `getattr`
    # because an older `ProcessTab` has no such rule - then the red-before run of this module fails on the ASSERTION
    # (the wrong picture closing the seek), which is the defect, instead of on a missing attribute.
    for _name in ("_picture_answers_seek", "_picture_seek_answer", "_picture_names_its_seek"):
        _rule = getattr(ProcessTab, _name, None)
        if _rule is not None:
            setattr(tab, _name, _rule.__get__(tab))
    return tab, calls


def test_a_named_streaming_frame_no_longer_closes_the_seek(monkeypatch):
    """The measured case: a press at 1.888 s was answered by the streaming frame at 4.046 s, +2.158 s off.

    `preview-log.txt`, 2026-11-24 23:11, lines 2281-2289: the press asked for 1.888 s (frame 45), the pane showed
    4.046 s and closed the request, and the answer (seq 99, 1.888 s) arrived one publication later with nobody waiting
    for it. The picture WAS named, which is all the old rule asked for - the streaming path now names every frame it
    draws, so "named" stopped meaning "asked for". Only the generation the writer puts in the header can tell them
    apart, and a picture that is not the answer must not be painted either: that flash is half the report.
    """
    notes = []
    monkeypatch.setattr("src.ui.video.telemetry.note_seek",
                        lambda ms, **kw: notes.append((ms, kw)))
    tab, calls = _seek_tab(4.046, seek_answer=None, seek_generation=7, target_s=1.888)

    ProcessTab._on_display_frame(tab, QImage(2, 2, QImage.Format.Format_RGB888))
    assert notes == [], "a streaming frame closed a seek it does not answer"
    assert calls == [], "and it was painted over the held picture"
    assert tab._seek_pressed_at == 0.0, "the request must stay open for its own answer"

    tab._display._stamp = 1.888          # the answer, one publication later
    tab._display.seek_answer = 7
    tab._display.seq = 99
    ProcessTab._on_display_frame(tab, QImage(2, 2, QImage.Format.Format_RGB888))
    assert len(notes) == 1, notes
    _ms, fields = notes[0]
    # `delta_s` is the formatter's job (`note_seek`, pinned in test_preview_telemetry.py); here the two times are the
    # evidence that the closed publication really is the requested picture.
    assert (fields["landed_time"], fields["target_s"]) == (1.888, 1.888), "the answer, not the playhead"
    assert (fields["target_gen"], fields["landed_gen"]) == (7, 7), "the log names both halves of the identity"
    assert len(calls) == 1 and calls[0][1] == 99
    assert tab._seek_pressed_at is None


def test_a_streaming_frame_does_not_close_a_seek(monkeypatch):
    """The renderer keeps publishing after the seek frame; an UNNAMED picture must not claim the landing.

    The producer here cannot say which request a picture answers (`_NO_FIELD`), so the picture has to name its own time
    to be evidence at all - and an unnamed one never does.
    """
    notes = []
    monkeypatch.setattr("src.ui.video.telemetry.note_seek",
                        lambda ms, **kw: notes.append((ms, kw)))
    tab, calls = _seek_tab(None)
    ProcessTab._on_display_frame(tab, QImage(2, 2, QImage.Format.Format_RGB888))
    assert notes == [], "a picture with no media time closed the seek"
    assert tab._seek_pressed_at == 0.0, "the seek must stay open for the stamped picture"
    assert calls and calls[0][2] is None


def test_the_stamped_picture_closes_the_seek_with_its_own_time(monkeypatch):
    """The renderer's stamped frame is the evidence; the pane must report ITS time, not the target's.

    The fallback path, kept honest: a producer that names no generation (`_NO_FIELD`) still closes on a named picture,
    because there is nothing better to go on. The real reader and feed always name one, so this is not the shipped
    rule - `test_a_named_streaming_frame_no_longer_closes_the_seek` is.
    """
    notes = []
    monkeypatch.setattr("src.ui.video.telemetry.note_seek",
                        lambda ms, **kw: notes.append((ms, kw)))
    tab, _calls = _seek_tab(1.139)
    ProcessTab._on_display_frame(tab, QImage(2, 2, QImage.Format.Format_RGB888))
    assert len(notes) == 1, notes
    _ms, fields = notes[0]
    assert fields["landed_time"] == 1.139
    assert fields["target_s"] == 1.139
    assert fields["landed"] == ("seq", 196)
    assert tab._seek_pressed_at is None
