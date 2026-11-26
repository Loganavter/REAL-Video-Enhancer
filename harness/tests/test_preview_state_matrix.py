"""The preview state matrix: one test per pane state, each written to fail when the behaviour it names is removed.

The previewer's 462 tests were green while the worst defect (a compare mode painting half a picture) was live: they pin
arithmetic and wiring, never STATES (`harness/docs/CODE_PATTERNS.md` §8; plan
`plan-rve-preview-use-cases-2026-11-27` §P0). This module is the yardstick for the refactor that follows: one function
per row of the matrix, named after its state, with the audit section or rule it pins in the docstring and the defect
that would make it red.

| row | test |
|---|---|
| paused, sbs, pair | `test_a_paused_sbs_pair_is_one_moment_and_asks_for_nothing` |
| paused, sbs, no source | `test_a_paused_sbs_pane_with_no_source_stays_empty_and_paints_no_lone_half` |
| playing, sbs, steady | `test_steady_sbs_playback_paints_every_pair_with_its_source_and_no_frame_delta` |
| playing, sbs, strip drag | `test_a_strip_drag_holds_the_last_matched_pair_and_bounds_the_gap` (xfail strict, RED today) |
| seek on a paused pane | `test_a_seek_on_a_paused_pane_gaps_then_pairs_the_new_moment` |
| render end -> finished file | `test_render_end_opens_a_new_generation_whose_first_pair_is_whole` |
| cleared pane | `test_a_cleared_pane_is_still_empty_at_two_seconds_and_at_ten` |
| refused press | `test_a_refused_press_leaves_the_generation_and_the_register_untouched` |
| export, sbs | `test_an_export_in_sbs_takes_both_halves_from_the_block_while_the_render_is_alive` |

How the rows are built. The pane rows drive the REAL `VideoPreviewSurface` through its two APIs (`set_result`,
`set_source`); the tab rows build the REAL `ProcessTab` with `object.__new__` (the house style of
`test_display_wiring.py`) and put a real pane on it. The renderer is wrapped, not replaced: its own `set_frame` /
`set_original` still run, so the recorded order is the order the GPU would receive while the pane's state stays honest.
There are no decoders and no `time.sleep`: where a row is about a hold, the clock is a number the test advances.

Everything the rows assert is a MOMENT or a PAYLOAD, never a bare count: "the pair is one picture" is two equal media
times on two halves that both exist, and "the register was untouched" is the same moments under the same keys carrying
the same pictures (`_colour_of`, since `set_source` re-creates the QImage it converts).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# Both the app and the backend live in a `src` namespace package; the suite's conftest puts the backend first, which
# would make `src.constants` the backend's module. The app root goes first for this module (the same house pattern as
# `test_display_wiring.py` / `test_preview_frame_on_demand.py`), and the cached entry is dropped so the app's own
# `constants` is what `ProcessTab` resolves.
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

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.ProcessTab import COMPARE_POOL_FRAMES, ProcessTab, _pane_result  # noqa: E402
from src.ui.preview.source_register import ResultPicture, SourcePicture, SourceRegister  # noqa: E402
from src.ui.video.frames import FrameRef  # noqa: E402
from src.ui.widgets import video_preview as pane_module  # noqa: E402
from src.ui.widgets.video_preview import (  # noqa: E402
    PAIR_QUEUE_MAX,
    PAIR_SYNC_HOLD_S,
    VideoPreviewSurface,
)

# Leave the suite's path order as it was (the house pattern): the app modules are imported and bound now.
sys.path[:] = _ORIGINAL_SYS_PATH

#: One frame at the scene's 24 fps - the unit every row below moves in.
FRAME = 1.0 / 24.0
PANE_SIZE = (800, 600)
FRAME_SIZE = (320, 180)
CLIP = SimpleNamespace(videoFrameCount=2400, videoFps=24, interpolateTimes=1, inputFile="/tmp/in.mkv",
                       outputPath="/tmp/out.mkv")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _picture(width: int = FRAME_SIZE[0], height: int = FRAME_SIZE[1], colour: str = "#204060") -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(colour))
    return image


def _colour_of(image: QImage) -> str:
    """The pane's held picture as a colour name: `set_source` converts and scales, so identity is the payload.

    A QImage is re-created by `convertToFormat`/`scaled`, so `is` would compare two different objects that carry the
    same picture. Every row paints each moment in its OWN colour, which makes "the picture for this moment" a
    comparison of what the pane holds against what was offered.
    """
    return image.pixelColor(1, 1).name()


def _surface(mode: str = "sbs", *, playhead: float = 0.0) -> VideoPreviewSurface:
    """A real pane in `sbs`, sized, with the strip's own progress published (the fixture shape of the sibling file)."""
    _app()
    widget = VideoPreviewSurface()
    widget.resize(*PANE_SIZE)
    widget.set_view_mode(mode)
    widget.set_progress(0.0, 0.0, playhead=playhead, duration=100.0)
    return widget


class _Clock:
    """A monotonic clock the test owns: the rows about a hold advance it instead of sleeping."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += float(seconds)
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    """Install a fake `time.monotonic` for the pane module's holds; restored by monkeypatch."""
    fake = _Clock()
    monkeypatch.setattr(pane_module.time, "monotonic", fake.monotonic)
    return fake


class _Paints:
    """Record what reached the renderer, in order, while the renderer's own state is still written.

    The rows are about identity, so the record keeps the picture's own media time (`pts`) and the fake clock reading at
    the call. Wrapping (rather than replacing) the two writers is what lets a row assert both "what the GPU was handed"
    and "what the pane now holds".
    """

    def __init__(self, surface: VideoPreviewSurface, clock: _Clock | None = None) -> None:
        self.calls: list[tuple[str, float | None, float | None]] = []
        self._clock = clock
        self._set_frame = surface._renderer.set_frame
        self._set_original = surface._renderer.set_original
        surface._renderer.set_frame = self._frame
        surface._renderer.set_original = self._original

    def _at(self) -> float | None:
        return None if self._clock is None else float(self._clock.t)

    def _frame(self, frame) -> None:
        self.calls.append(("frame", None if frame is None else float(frame.pts), self._at()))
        self._set_frame(frame)

    def _original(self, frame) -> None:
        self.calls.append(("orig", None if frame is None else float(frame.pts), self._at()))
        self._set_original(frame)

    @property
    def kinds(self) -> list[str]:
        return [kind for kind, _pts, _at in self.calls]

    @property
    def shape(self) -> list[tuple[str, float | None]]:
        """The (kind, pts) order without the clock reading - the form to compare when a fake clock is installed."""
        return [(kind, pts) for kind, pts, _at in self.calls]

    @property
    def frames(self) -> list[tuple[float | None, float | None]]:
        return [(pts, at) for kind, pts, at in self.calls if kind == "frame"]

    @property
    def originals(self) -> list[tuple[float | None, float | None]]:
        return [(pts, at) for kind, pts, at in self.calls if kind == "orig"]

    def clear(self) -> None:
        self.calls.clear()


class _Process:
    """`subprocess.Popen` as the wiring reads it: `poll()` is None while the render is alive."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = bool(alive)

    def poll(self):
        return None if self.alive else 0


class _Reader:
    """The parts of `DisplayReader` the page uses, with the seek slot recorded (no shared memory)."""

    attached = True

    def __init__(self, *, attached: bool = True) -> None:
        self.attached = bool(attached)
        self.wants: list[bool] = []
        self.seeks: list[float] = []
        self.generation = 0

    def set_wanted(self, wanted: bool) -> None:
        self.wants.append(bool(wanted))

    def request_seek(self, seconds: float) -> int:
        self.seeks.append(float(seconds))
        self.generation += 1
        return self.generation

    def seek_generation(self) -> int:
        return self.generation

    def release(self) -> None:
        self.attached = False


class _Tail:
    """A live tail as the press route sees it: its published edge and its seek."""

    media_origin = 0.0

    def __init__(self, produced_seconds: float = 100.0) -> None:
        self.produced_seconds = float(produced_seconds)
        self.seeks: list[float] = []
        self.stops = 0
        self.paused: list[bool] = []

    def seek(self, target, **_kwargs) -> None:
        self.seeks.append(float(target))

    def set_paused(self, value) -> None:
        self.paused.append(bool(value))

    def stop_and_wait(self) -> None:
        self.stops += 1


class _Feed:
    """The compare's block: one picture for one media time, offered by `_on_source_feed_frame`."""

    def __init__(self, media_time: float) -> None:
        self.media_time = float(media_time)
        self.attached = True
        self.released = 0
        self.wants: list[bool] = []

    def set_wanted(self, wanted) -> None:
        self.wants.append(bool(wanted))

    def poll(self):
        return None

    def release(self) -> None:
        self.released += 1


class _Tab:
    """The real `ProcessTab`, built without a window, with a real pane on it (the house `object.__new__` style)."""

    @staticmethod
    def build(*, pane=None, preview: bool = True, paused: bool = True, alive: bool = True,
              reader=None, tail=None) -> ProcessTab:
        pane = _surface(playhead=0.0) if pane is None else pane
        tab = object.__new__(ProcessTab)
        tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=pane))
        tab.isPreview = bool(preview)
        tab._preview_paused = bool(paused)
        tab.renderProcess = _Process(alive)
        tab._display = reader
        tab._display_seek_pending = False
        tab._preview_strip_wired_for = None
        tab._live_tail = tail
        tab._preview_decoder = None
        tab._preview_playhead = 0.0
        tab._preview_duration = 100.0
        tab._seek_watch = None
        tab.currentFrame = 50
        tab.max_value = 100
        tab.videoFrameCount = 2400
        tab.fps = 24
        tab.currentRenderOptions = SimpleNamespace(**vars(CLIP))
        tab._source_register = SourceRegister(COMPARE_POOL_FRAMES)
        tab._source_fill_pending = None
        tab._source_warm_tried = True          # never open a decoder in a unit test
        tab._source_decoder = None
        tab._result_generation = 0
        tab._compare_from_tick = False
        tab._compare_fill_stats = {"asks": 0, "arrivals": 0, "at": 0.0}
        # The decoder is the only stub a tab row needs: its ask is a tuple, not a picture.
        tab._asked: list[tuple[float, bool]] = []
        tab._on_original_requested = lambda seconds, *, span=False: tab._asked.append((float(seconds), bool(span)))
        return tab


def _frame_ref(pts: float, index: int = 0) -> FrameRef:
    return FrameRef(buffer=bytes(2 * 2 * 3), width=2, height=2, format="rgb24", stride=6, pts=float(pts),
                    index=int(index))


# --- row 1: paused, sbs, pair ---------------------------------------------------------------------------------------


def test_a_paused_sbs_pair_is_one_moment_and_asks_for_nothing():
    """`paused, sbs, pair`: both halves of ONE moment, `compare fill asks=0` over a quiet window, no gap.

    Pins `PREVIEWER_RULES.md` §5 and `code-patterns.md` §6: a cache may not evict the answer it is being asked for, and a
    paused pane is asked for one picture at its own moment, never for a decode lead. The defect it pins is the owner's
    "the original blinks a couple of times with playback never started" (`asks` climbing while the pane stands still, the
    lead walk evicting the picture on screen).

    Breaks it: removing the `paused` early return in `ProcessTab._source_fill` (the fill then asks for a lead on every
    tick), or eviction measured from "newest" instead of the asked moment (`SourceRegister._prune`).
    """
    pane = _surface("sbs", playhead=5.0)
    paints = _Paints(pane)
    tab = _Tab.build(pane=pane, preview=True, paused=True, reader=_Reader())
    tab._result_generation = 3
    tab._preview_playhead = 5.0

    result_picture = _picture(colour="#3060a0")
    source_picture = _picture(colour="#402060")
    _pane_result(tab, pane, 5.0, result_picture)
    tab._source_register.offer(5.0, source_picture, "decoder", generation=3)
    tab._tick_second_surface()

    assert paints.calls == [("frame", 5.0, None), ("orig", 5.0, None)], (
        "the quiet pane paints both halves of one moment, together")
    assert _colour_of(pane._original) == QColor("#402060").name(), (
        "the source half is the picture the register holds for THIS moment")
    held = pane._original
    assert pane.shown_moment() == pytest.approx(5.0) == pane.original_shown_time()
    assert pane.committed_time() == pytest.approx(5.0) == pane._committed_source_time
    assert pane.pair_in_sync() is True

    # The quiet window: the pane stands still, so nothing may be decoded for it.
    for _ in range(10):
        tab._tick_second_surface()
    assert tab._compare_fill_stats["asks"] == 0, "a paused pane asked for a decode (the lead is a playback cost)"
    assert tab._asked == [], f"the fill asked the decoder for {tab._asked} while the pane stood still"
    assert tab._source_fill_pending is None, "an ask was left in flight on a pane nobody is moving"
    assert pane._original is held, "the same picture is still on screen: no gap, no neighbour"
    assert pane.original_shown_time() == pytest.approx(5.0)
    assert paints.calls == [("frame", 5.0, None), ("orig", 5.0, None)], "ten idle ticks repainted the pair"


# --- row 2: paused, sbs, no source -----------------------------------------------------------------------------------


def test_a_paused_sbs_pane_with_no_source_stays_empty_and_paints_no_lone_half(clock):
    """`paused, sbs, no source`: the left half stays empty and the result is never handed over alone.

    Pins `PREVIEWER_RULES.md` §2 / audit §6.1: `_request_frame_paint` used to call `_commit_newest()` whenever
    `_original is None` - the normal state before the first source frame and after every refused one - so the upscale
    was painted alone and the original followed a frame later (owner: "оригинал не отрисовывается одновременно с
    апскейлом в режиме side by side"). The compare mode must paint a PAIR or nothing; the only permitted escape is the
    hold window, after which a still picture is better than a blank pane.

    Breaks it: the pre-fix branch at `video_preview.py:997` (`if self._view_mode == "single" or self._original is None:
    self._commit_newest()`), which paints the first result immediately and every later one too.
    """
    pane = _surface("sbs", playhead=2.0)
    paints = _Paints(pane, clock)
    tab = _Tab.build(pane=pane, preview=True, paused=True, reader=_Reader())
    tab._result_generation = 1
    tab._preview_playhead = 2.0

    # The result half exists from the first frame; the source never comes.
    _pane_result(tab, pane, 2.0, _picture(colour="#3060a0"))
    assert paints.calls == [], "a compare mode handed the renderer a result with no source"
    assert pane._original_rect().isEmpty(), "the left half is empty: there is nothing to draw there"
    assert pane.pair_in_sync() is False, "one half is not a synced pair"
    held_at = pane._pair_blocked_at
    assert held_at is not None, "the gate must be HOLDING the picture, not silently ignoring it"

    clock.advance(0.1)
    _pane_result(tab, pane, 2.0 + FRAME, _picture(colour="#3060a0"))
    assert paints.calls == [], "a second result inside the hold opened the gate"

    # Past the hold the hard limit is the only escape - and it may never paint a lone half EARLIER than that.
    clock.advance(PAIR_SYNC_HOLD_S)
    _pane_result(tab, pane, 2.0 + 2 * FRAME, _picture(colour="#3060a0"))
    for kind, pts, at in paints.calls:
        if kind == "frame" and pts is not None:
            assert at - held_at >= PAIR_SYNC_HOLD_S, (
                f"a lone result was painted {at - held_at:.3f}s into the hold (limit {PAIR_SYNC_HOLD_S}s)")

    # The gap is a gap, never a stale neighbour: with no source accepted, the renderer holds no source picture.
    if paints.calls:
        assert pane._original is None, "the escape painted a source half that never arrived"


# --- row 3: playing, sbs, steady -------------------------------------------------------------------------------------


def test_steady_sbs_playback_paints_every_pair_with_its_source_and_no_frame_delta(clock):
    """`playing, sbs, steady`: every painted pair carries the source, `delta_frames == 0`.

    Pins `PREVIEWER_RULES.md` §2 and the pair queue's contract (`_pick_pair` / `_commit_pair`): the renderer gets a
    result and a source of the SAME moment, whichever half arrived first, and never a half-updated pair. The defect is
    the 40 ms window of mismatched halves the owner read as flicker: one half committed on its own arrival, the other
    on the next.

    Breaks it: `_request_frame_paint` committing `_commit_newest()` on every arrival instead of `_commit_pair()`.
    """
    pane = _surface("sbs", playhead=1.0)
    paints = _Paints(pane, clock)
    tab = _Tab.build(pane=pane, preview=False, paused=False, reader=_Reader())
    tab._result_generation = 4

    moments = [1.0 + step * FRAME for step in range(8)]
    for moment in moments:
        _pane_result(tab, pane, moment, _picture(colour="#3060a0"))     # the result leads
        clock.advance(FRAME / 2.0)                                      # ...its source answers a decode later
        pane.set_source(SourcePicture(moment=moment, picture=_picture(colour="#402060"),
                                      origin="decoder", generation=4))
        clock.advance(FRAME / 2.0)

    kinds = paints.kinds
    assert kinds == ["frame", "orig"] * len(moments), f"every paint is a framed pair, got {kinds}"
    pairs = list(zip(paints.calls[0::2], paints.calls[1::2]))
    for (kind_a, result_pts, _), (kind_b, source_pts, _) in pairs:
        assert (kind_a, kind_b) == ("frame", "orig")
        assert result_pts == pytest.approx(source_pts), (
            f"delta_frames != 0: the result at {result_pts} was painted beside the source at {source_pts}")
    assert pane.committed_time() == pytest.approx(moments[-1])
    assert pane._committed_source_time == pytest.approx(moments[-1])
    assert pane.pair_in_sync() is True
    assert pane._pair_blocked_at is None, "a pair never needed the hold in steady playback"


# --- row 4: playing, sbs, strip drag (RED today) ----------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=(
    "RED on the 2026-11-27 tree, deliberately: a strip drag during playback bumps the result generation once per "
    "slider step and the source decoder answers one ask late, so every answer is a NEIGHBOUR of the moment the "
    "pane now shows and is refused (audit §6.3: the bump invalidates the pictures by design; the answers arrive "
    "stamped one generation or one frame behind). The pane's widget-side source half therefore becomes a gap and "
    "the ~1.3 s of drag repaints without the original (owner's log: 31 draws). The assertion below is the intended "
    "behaviour after plan P1's pair writer: the last matched pair HOLDS (source half included) for the whole drag, "
    "no paint hands the renderer a lone half, and the number of results held without a fresh source is bounded by "
    "MAX_STALE_FRAMES. Flip: remove this xfail when P1 lands."))
def test_a_strip_drag_holds_the_last_matched_pair_and_bounds_the_gap(clock):
    """`playing, sbs, strip drag`: the last matched pair holds; the gap is bounded; no lone-half flashes.

    Pins audit §6.3 (a drag clears the register 10-60x/second and every source offer is then refused as the wrong
    generation) plus the plan's P0 row. Written as the INTENDED behaviour: after the pair writer exists
    (`src/ui/preview/pair.py`, one `CommittedPair`), the committed pair is what the pane draws, so a refused neighbour
    cannot take the source half away from a picture that is still on screen.

    Breaks it (today): the refusal path in `_apply_source_picture(None)` clears `_original`, so the widget stops drawing
    the left half even though the renderer still holds a matched pair. When P1 lands, this row asserts: `_original` stays
    the committed source for every step of the drag, `set_frame` is never called without its `set_original`, and at most
    `MAX_STALE_FRAMES` results are held unpaired before the pair is painted again.
    """
    pane = _surface("sbs", playhead=3.0)
    paints = _Paints(pane, clock)
    tab = _Tab.build(pane=pane, preview=True, paused=False, reader=_Reader(), tail=_Tail(produced_seconds=100.0))
    tab._preview_playhead = 3.0
    tab._result_generation = 1

    # The last matched pair: 3.0, both halves.
    _pane_result(tab, pane, 3.0, _picture(colour="#3060a0"))
    tab._source_register.offer(3.0, _picture(colour="#402060"), "decoder", generation=1)
    tab._tick_second_surface()
    assert paints.shape == [("frame", 3.0), ("orig", 3.0)]
    paints.clear()

    # A drag: one press per slider step, each bumping the generation and clearing the register.
    newest = 8.0
    for step in range(6):
        tab._on_preview_seek(0.5)
        newest = 8.0 + step * FRAME
        _pane_result(tab, pane, newest, _picture(colour="#3060a0"))
        tab._tick_second_surface()                       # the gap: no picture for this generation yet
        # The decoder answers ONE ask late: a neighbour of the moment the pane shows.
        tab._source_register.offer(newest - FRAME, _picture(colour="#402060"), "decoder",
                                   generation=tab._result_generation)
        tab._tick_second_surface()                       # refused as a neighbour
        clock.advance(0.05)

        assert pane._original is not None, (
            f"step {step}: the source half went blank while the last matched pair was still the picture on screen")
        assert pane.original_shown_time() == pytest.approx(pane.committed_time()), (
            f"step {step}: the drawn source ({pane.original_shown_time()}) is not the committed pair's "
            f"({pane.committed_time()})")

    for (kind, pts, _at) in paints.calls:
        if kind == "frame" and pts is not None:
            raise AssertionError(f"a lone result half ({pts}) was handed to the renderer during the drag")

    limit = getattr(pane_module, "MAX_STALE_FRAMES", PAIR_QUEUE_MAX)
    assert len(pane._result_queue) <= limit, (
        f"{len(pane._result_queue)} results are held without a source; the documented limit is {limit}")

    # The true answer finally lands: one pair, both halves at the new moment.
    tab._source_register.offer(newest, _picture(colour="#402060"), "decoder", generation=tab._result_generation)
    tab._tick_second_surface()
    assert pane.committed_time() == pytest.approx(newest) == pane._committed_source_time


# --- row 5: seek on a paused pane -------------------------------------------------------------------------------------


def test_a_seek_on_a_paused_pane_gaps_then_pairs_the_new_moment():
    """`seek on a paused pane`: a gap, then the new moment; left and right are the same moment.

    Pins `PREVIEWER_RULES.md` §1/§2 and `_flush_on_jump`: a press is a new moment, so the pair the pane holds belongs to
    the moment it is leaving. The result of the seek arrives before its source (the tail re-decodes), and the pane must
    show a GAP for that first frame rather than pair the new result with the picture of the old moment - the
    `delta_frames=-80` transition the flush exists for.

    Breaks it: `_request_frame_paint` committing the newest of each half (`_commit_newest`) - the new result paints
    beside the source of the moment the press left, i.e. left and right are two moments.
    """
    pane = _surface("sbs", playhead=4.0)
    paints = _Paints(pane)
    tail = _Tail(produced_seconds=100.0)
    tab = _Tab.build(pane=pane, preview=True, paused=True, reader=_Reader(), tail=tail)
    tab._preview_playhead = 4.0
    tab._result_generation = 2

    # The moment the pane is standing on, as a matched pair.
    _pane_result(tab, pane, 4.0, _picture(colour="#3060a0"))
    tab._source_register.offer(4.0, _picture(colour="#402060"), "decoder", generation=2)
    tab._tick_second_surface()
    assert paints.shape[-2:] == [("frame", 4.0), ("orig", 4.0)], "a matched pair is on screen before the press"
    paints.clear()

    # A press inside the produced band: the tail is asked, the generation is bumped, the register emptied.
    tab._on_preview_seek(0.4)
    assert tail.seeks == [pytest.approx(40.0)], "the press asked the tail for its own timeline"
    assert tab._result_generation == 3, "a seek that was actually asked for mints the next generation"
    assert len(tab._source_register) == 0, "the previous moment's pictures are not answers after a seek"

    # The tail's answer for the new moment: the result half alone, one tick before its source.
    tab._on_preview_frame(_frame_ref(40.0, index=960))

    assert pane.shown_moment() == pytest.approx(40.0), "the pane is at the moment it was asked for"
    assert pane._original is None, "the new result was paired with the picture of the moment the press left"
    assert paints.calls == [], "nothing was painted: the seek's pair is not whole yet"
    assert pane.committed_time() == pytest.approx(4.0), "the renderer keeps the last consistent pair meanwhile"
    assert pane._renderer._original is not None and pane._renderer._frame is not None, (
        "the last matched pair is still what the renderer holds")

    # The source for the new moment arrives: one pair, both halves at 40.0 - and no paint mixed the two moments.
    tab._source_register.offer(40.0, _picture(colour="#402060"), "decoder", generation=3)
    tab._tick_second_surface()

    assert paints.calls == [("frame", 40.0, None), ("orig", 40.0, None)], (
        f"the new moment's pair, together; got {paints.calls}")
    assert pane.committed_time() == pytest.approx(40.0) == pane._committed_source_time
    assert pane.original_shown_time() == pytest.approx(40.0), "left and right are the same moment"
    for kind, pts, _at in paints.calls:
        if kind == "frame":
            assert pts == pytest.approx(40.0), "a frame of the moment the pane left was painted"


# --- row 6: render end -> finished file -------------------------------------------------------------------------------


class _Button:
    def __init__(self) -> None:
        self.clicked = SimpleNamespace(connect=lambda *_a, **_k: None)
        self.enabled = None
        self.visible = None
        self.checked = None

    def setEnabled(self, value) -> None:  # noqa: N802 - Qt spelling
        self.enabled = bool(value)

    def setVisible(self, value) -> None:  # noqa: N802 - Qt spelling
        self.visible = bool(value)

    def setChecked(self, value) -> None:  # noqa: N802 - Qt spelling
        self.checked = bool(value)


class _Text:
    def setText(self, text) -> None:
        self.text = str(text)

    def setEnabled(self, value) -> None:
        self.enabled = bool(value)

    def text(self) -> str:
        return getattr(self, "text", "")


class _FakeDecoder(QObject):
    """The finished file's own player: an open/seek/play that records, no media."""

    frame_ready = Signal(object)
    opened = Signal(object)
    error = Signal(str)
    fps = 24.0

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.seeks: list[float] = []
        self.plays = 0
        self.pauses = 0
        self.closed = 0
        self.path = None

    def open(self, path) -> bool:
        self.path = str(path)
        return True

    def seek(self, seconds) -> None:
        self.seeks.append(float(seconds))

    def play(self) -> None:
        self.plays += 1

    def pause(self) -> None:
        self.pauses += 1

    def close(self) -> None:
        self.closed += 1


def _completion_tab(pane, monkeypatch):
    """The tab `guiChangesOnRenderCompletion` needs, with the real pane and a fake finished-file decoder."""
    tab = _Tab.build(pane=pane, preview=True, paused=True, alive=False, reader=_Reader(), tail=_Tail())
    tab._preview_playhead = 12.0
    tab._result_generation = 5
    tab.return_codes = [0]
    tab.userKilled = False
    tab._render_started_mono = 1.0
    tab.currentRenderOptions = SimpleNamespace(**vars(CLIP))
    tab.currentRenderOptions.inputFile = "/tmp/in.mkv"
    tab.currentRenderOptions.outputPath = "/tmp/out.mkv"
    tab.currentRenderOptions.isPreview = True
    tab.page_handles.proc.onRenderButtonsContiainer = object()
    tab.page_handles.proc.startRenderButton = _Button()
    tab.page_handles.proc.killRenderButton = _Button()
    tab.page_handles.proc.previewPauseButton = None
    tab.page_handles.general = SimpleNamespace(outputFileText=_Text(), inputFileText=_Text())
    tab.parent = SimpleNamespace(enableProcessPage=lambda: None, startRender=lambda: None)
    tab.setProgressDetail = lambda text: None
    tab.setStatus = lambda status: None
    tab.syncRenderConsole = lambda: None
    tab._restoreQueueAfterPreview = lambda: None
    monkeypatch.setattr("src.ui.video.decode.VideoDecoder", _FakeDecoder)
    # The render row's own state machine is chrome, not the seam: `test_display_wiring` neutralises the same writer.
    monkeypatch.setattr(ProcessTab, "_set_render_controls", lambda self, state: None)
    return tab


def test_render_end_opens_a_new_generation_whose_first_pair_is_whole(monkeypatch):
    """`render end -> finished file`: no frame of the previous file; the first pair of the new generation is whole.

    Pins `PREVIEWER_RULES.md` §3 and `ProcessTab._start_preview_playback`: the finished file is a NEW SOURCE, so its
    frames are a new generation, and the pane keeps the last painted frame until that generation has a whole pair. The
    two defects it pins are the "previous file's frame after the render ended" (a picture promoted across the handover)
    and the lone half at the seam (a new result painted beside the old file's source).

    Breaks it: `_next_result_generation()` at the handover, or the pane's generation/moment guards (`_result_refusal`,
    `set_source`'s `same_generation` check) - without them the old file's pictures are accepted as the new file's.
    """
    pane = _surface("sbs", playhead=9.0)
    paints = _Paints(pane)
    tab = _completion_tab(pane, monkeypatch)
    old_generation = tab._result_generation

    # The live job's last pair: the growing file's picture at 9.0.
    _pane_result(tab, pane, 9.0, _picture(colour="#3060a0"))
    tab._source_register.offer(9.0, _picture(colour="#402060"), "decoder", generation=old_generation)
    tab._sync_compare_source()
    assert paints.calls == [("frame", 9.0, None), ("orig", 9.0, None)]
    paints.clear()

    tab.guiChangesOnRenderCompletion()

    assert tab._result_generation == old_generation + 1, "the finished file is a new generation, minted at the handover"
    decoder = tab._preview_decoder
    assert isinstance(decoder, _FakeDecoder) and decoder.path == "/tmp/out.mkv", (
        "the finished file became the pane's own player")
    assert decoder.seeks == [pytest.approx(12.0)], "a paused handover stands where the render stood"
    assert pane.shown_moment() == pytest.approx(9.0), "the last painted frame is kept through the job's end"
    assert pane._original is not None, "and so is its source half: the pane does not blank at the seam"

    # A source from the PREVIOUS file cannot join the new generation's first result.
    pane.set_result(ResultPicture(moment=12.0, picture=_picture(colour="#3060a0"),
                                  generation=tab._result_generation))
    pane.set_source(SourcePicture(moment=9.0, picture=_picture(colour="#402060"), origin="decoder",
                                  generation=old_generation))
    assert pane._original is None, "the previous file's source was accepted for the new generation"
    assert paints.calls == [], "a lone half of the previous file was painted at the seam"

    # The new file's own source for the same moment: the first pair of the new generation, whole.
    pane.set_source(SourcePicture(moment=12.0, picture=_picture(colour="#402060"), origin="decoder",
                                  generation=tab._result_generation))
    assert paints.calls == [("frame", 12.0, None), ("orig", 12.0, None)], (
        f"the first pair of the new generation is whole, got {paints.calls}")
    assert pane.committed_time() == pytest.approx(12.0) == pane._committed_source_time
    assert pane.original_shown_time() == pytest.approx(12.0)


# --- row 7: cleared pane ----------------------------------------------------------------------------------------------


def test_a_cleared_pane_is_still_empty_at_two_seconds_and_at_ten(clock):
    """`cleared pane`: empty at 2 s and at 10 s.

    Pins `PREVIEWER_RULES.md` §3 / audit §6.1b, probe G2: `set_result(None)` used to reset the picture but keep the pair
    queues, and with `shown_moment()` None the source guard was skipped by construction, so a late source re-keyed
    itself by its own moment and painted half a picture onto the pane that was told to be empty. The state clock is the
    pane's: the row advances a fake monotonic, never a wall clock.

    Breaks it: the clear branch of `_accept_result` (the queue/source flush and `_cleared_empty`), or `set_source`'s
    `_cleared_empty` refusal.
    """
    pane = _surface("sbs", playhead=20.0)
    paints = _Paints(pane, clock)
    tab = _Tab.build(pane=pane, preview=True, paused=True, reader=_Reader())
    tab._result_generation = 6
    tab._preview_playhead = 20.0

    # A pair is on the pane, and two more results are QUEUED behind it without a source (a stalled compare).
    _pane_result(tab, pane, 20.0, _picture(colour="#3060a0"))
    tab._source_register.offer(20.0, _picture(colour="#402060"), "decoder", generation=6)
    tab._sync_compare_source()
    _pane_result(tab, pane, 20.0 + FRAME, _picture(colour="#3060a0"))
    _pane_result(tab, pane, 20.0 + 2 * FRAME, _picture(colour="#3060a0"))
    assert paints.calls, "a pair was painted before the clear"
    paints.clear()

    pane.set_result(None)                                       # the pane is told to be EMPTY

    assert pane._frame is None and pane._planes_size is None, "the result half is empty"
    assert pane._original is None, "the source half is empty"
    assert pane._renderer._frame is None and pane._renderer._original is None, "the renderer holds nothing"
    assert pane._result_queue == {}, "the cleared pane kept the queued results of the picture it no longer shows"
    assert pane._source_queue == {}, "and the source queue with them"

    for seconds in (2.0, 10.0):
        clock.t = 1000.0 + float(seconds)
        # Everything a late producer can still do: a source for a PRE-clear moment, a result from the cleared queue,
        # a pair release whose timer was armed before the clear.
        pane.set_source(SourcePicture(moment=20.0, picture=_picture(colour="#402060"), origin="decoder",
                                     generation=6))
        pane.set_result(ResultPicture(moment=None, picture=None, generation=6))
        pane._request_frame_paint()
        pane._release_pair_paint()

        assert pane._frame is None, f"the cleared pane got a result back at {seconds:.0f}s"
        assert pane._original is None, f"the cleared pane got a source half back at {seconds:.0f}s"
        assert pane._renderer._frame is None, f"the renderer was handed a picture onto an empty pane at {seconds:.0f}s"
        assert pane._renderer._original is None, f"the renderer was handed a source onto an empty pane at {seconds:.0f}s"


# --- row 8: refused press ---------------------------------------------------------------------------------------------


def test_a_refused_press_leaves_the_generation_and_the_register_untouched(monkeypatch):
    """`refused press`: the result generation and the compare register are unchanged.

    Pins `PREVIEWER_RULES.md` §4 / audit §6.3, probe B: `_on_preview_seek` bumped the result generation before choosing
    a route, and every refusal returned after the bump; `_next_result_generation` also clears the register. A press the
    pane could not serve therefore took generation 7 -> 8 and the register 3 -> 0, and every source offer was refused as
    the wrong generation until the next result - the blank half after a click.

    Breaks it: moving `self._next_result_generation()` back to the top of `_on_preview_seek` (the pre-fix shape); the
    probe below is that exact shape.
    """
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("src.ui.video.telemetry.event", lambda kind, **fields: events.append((kind, fields)))
    pane = _surface("sbs", playhead=2.0)
    paints = _Paints(pane)
    # A preview with NO usable block: the documented refusal, "preview-without-a-block".
    tab = _Tab.build(pane=pane, preview=True, paused=True, reader=_Reader(attached=False))
    tab._result_generation = 7
    tab._preview_playhead = 2.0

    _pane_result(tab, pane, 2.0, _picture(colour="#3060a0"))
    colours = {1.0: "#402060", 2.0: "#403060", 3.0: "#404060"}
    for moment, colour in colours.items():
        tab._source_register.offer(moment, _picture(colour=colour), "decoder", generation=7)
    tab._sync_compare_source()
    assert len(tab._source_register) == 3
    assert _colour_of(pane._original) == QColor(colours[2.0]).name(), (
        "the compare's half is on screen before the press")
    pane_state_before = (pane.shown_moment(), int(pane._result_generation), _colour_of(pane._original))
    paints.clear()

    tab._on_preview_seek(0.5)

    assert tab._press is not None and not tab._press.open, "the press was recorded and refused"
    refusals = [fields for kind, fields in events if kind == "seek refused"]
    assert refusals and refusals[-1]["reason"] == "preview-without-a-block", (
        f"the refusal did not name its route: {refusals}")
    assert tab._result_generation == 7, "a refused press minted a new result generation"
    assert len(tab._source_register) == 3, "a refused press emptied the compare register"
    assert tab._source_register.owner_of(2.0, generation=7) == "decoder"
    assert _colour_of(tab._source_register.picture_for(2.0, 0.5 / 24.0, generation=7).picture) == QColor(
        colours[2.0]).name(), "the register still holds the same picture for the same moment"
    assert (pane.shown_moment(), int(pane._result_generation), _colour_of(pane._original)) == pane_state_before, (
        "a refused press changed the picture identity on the pane")
    assert paints.calls == [], "a refused press repainted the pane"


# --- row 9: export, sbs -----------------------------------------------------------------------------------------------


def test_an_export_in_sbs_takes_both_halves_from_the_block_while_the_render_is_alive(monkeypatch):
    """`export, sbs`: both halves from one owner; `held_block` only while the render is alive.

    Pins `PREVIEWER_RULES.md` §6 and `code-patterns.md` §2: during an export the renderer publishes the SOURCE of each
    composited frame into its own block, so both halves come from one production at one media time - the decoder must
    not spend a second decode on the moment the block already answers. The register's pool count is split by producer
    (`held_block`/`held_decoder`) so "24 held" cannot be read as "24 source frames", and the block's pictures belong to
    the render pass: `_close_display_state` drops them when the job ends.

    Breaks it: the `owner_of(playhead) == "block"` early return in `_source_fill` (the compare re-decodes the moment the
    block owns), or `register.drop_origin("block")` in `_close_display_state` (`held_block` outlives the render).
    """
    pane = _surface("sbs", playhead=5.0)
    paints = _Paints(pane)
    feed = _Feed(media_time=5.0)
    tab = _Tab.build(pane=pane, preview=False, paused=False, alive=True, reader=_Reader())
    tab._source_feed = feed
    tab._result_generation = 0

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("src.ui.video.telemetry.event", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr("src.ui.ProcessTab.time.monotonic", lambda: 5000.0)

    # The export's own picture: the feed publishes the result, the block publishes the source of the same moment.
    _pane_result(tab, pane, 5.0, _picture(colour="#3060a0"))
    block_picture = _picture(colour="#402060")
    tab._on_source_feed_frame(block_picture)

    assert paints.calls == [("frame", 5.0, None), ("orig", 5.0, None)], (
        f"the block's own picture is paired with the feed's frame, got {paints.calls}")
    assert _colour_of(pane._original) == QColor("#402060").name(), (
        "both halves come from one moment, one producer")
    assert tab._source_register.counts() == {"block": 1}

    # The fill must not pay for a decode of the moment the block already owns.
    tab._source_fill(5.0)
    assert tab._asked == [], f"the compare decoded the moment the block owns: {tab._asked}"

    # The fill line reports the pool by producer while the render is alive.
    tab._compare_fill_stats["at"] = 5000.0 - 2.0
    tab._compare_fill_report()
    fills = [fields for kind, fields in events if kind == "compare fill"]
    assert fills and fills[-1]["held_block"] == 1 and fills[-1]["held_decoder"] == 0, (
        f"the pool count is not split by producer: {fills[-1] if fills else None}")

    # The render ends: the block's pictures are not answers any more.
    tab._close_display_state()
    assert tab._source_feed is None and feed.released == 1
    assert tab._source_register.counts() == {}, "the block's pictures outlived the render that produced them"

    tab._compare_fill_stats["at"] = 5000.0 - 2.0
    tab._compare_fill_report()
    fills = [fields for kind, fields in events if kind == "compare fill"]
    assert fills[-1]["held_block"] == 0 and fills[-1]["pool"] == 0, (
        f"a finished render still reports source pictures: {fills[-1]}")

    # And the pane keeps the last pair it was given (a job's end does not blank the picture).
    assert pane.committed_time() == pytest.approx(5.0)
    assert _colour_of(pane._original) == QColor("#402060").name()
