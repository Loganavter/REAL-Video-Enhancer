"""The scrubber's drag: one authority for the thumb at a time, and the ask survives the release.

The live state matrix found this on a leased display (2026-11-27, row `rve-timeline-drag-glue-2026-11-27`):
`drag:preview_timeline@40,10:120,10` pressed, four moves ran, and the picture went on to the end while the thumb
stood where it had been pressed. The apply lines of that run:

    preview: scrub apply value=160 thumb=0.16 fill=0.4000 produced=1.0 scrubbing=1 finished=1 glued=0
    ...                                     fill 0.4167 ... 0.98                glued flips to 1 at 0.8667

Two facts in one line: the state's playhead stayed on the press (`thumb=0.16`) while the picture's own moment
(`fill`) marched, and `value` never moved off the press. The widget had no answer to "who owns the thumb", so the
answer was implicit: `sliderMoved` moved it and *any* state pass rewrote it. Three consequences, all of them here:

* a state pass that arrives after the release puts the thumb back on whatever the page's playhead says - and while a
  press is unanswered that playhead is the REQUESTED time, so the thumb lands on a place the finger has already left;
* a state pass that arrives while a move is outstanding is free to pull the thumb off the drag;
* nothing bounds a drag whose release signal never arrived: `_scrubbing` stays true and no state can ever move the
  thumb again (the "pinned" the owner saw).

What this module pins, in the order the task names them (each row is red on the pre-fix tree, proven by reverting the
guard - see `DONE.md`):

| row | test |
|---|---|
| press, no move | `test_a_press_with_no_move_asks_once_at_the_pressed_fraction` |
| four moves | `test_four_moves_ask_four_times_and_the_thumb_follows_the_last` |
| release | `test_a_release_re_asks_nothing_and_holds_the_last_target_against_a_stale_state` |
| drag in progress | `test_a_value_that_moves_without_the_move_signal_still_asks`, `test_a_state_pass_during_a_drag_does_not_move_the_thumb` |
| a forward seek it passed | `test_a_forward_seek_that_lands_a_tick_past_the_ask_does_not_freeze_the_thumb` |
| lost release | `test_a_drag_whose_release_never_arrives_ends_and_the_ask_still_owns_the_thumb` |
| the overlay's shape | `test_a_state_pass_does_not_reshape_the_overlay_under_the_pointer` |
| the glue | `test_the_bar_asks_the_controller_for_every_move_and_the_thumb_holds_the_last` |

The invariant, in one line: **the thumb == the last seek this control emitted == the moment the pane was asked for**;
the state may move it only when it agrees with that ask (or when the ask is past what the pane can show, or has stood
unanswered past `ASK_HOLD_WINDOW_S`). `glued` - the report's "the fill and the thumb are one number" - is false while
the user is dragging, by construction.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

# `ProcessTab` is not imported here, but the toolkit's theme and `src.ui.preview` are; stub the app's optional runtime
# imports exactly as the sibling control-surface tests do, so this module never depends on the interpreter's extras.
for _name, _attrs in (("cpuinfo", {"get_cpu_info": lambda *a, **k: {}}),
                      ("pypresence", {"Presence": object})):
    try:  # pragma: no cover - the branch depends on the interpreter
        __import__(_name)
    except ModuleNotFoundError:  # pragma: no cover
        _module = types.ModuleType(_name)
        for _attr, _value in _attrs.items():
            setattr(_module, _attr, _value)
        sys.modules[_name] = _module

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager  # noqa: E402

from src.ui.preview import timeline as timeline_module  # noqa: E402
from src.ui.preview.backends import NoBackend  # noqa: E402
from src.ui.preview.controller import PreviewController  # noqa: E402
from src.ui.preview.controls import PreviewControls  # noqa: E402
from src.ui.preview.state import PreviewBackendKind, PreviewCommand, PreviewPhase, PreviewState  # noqa: E402
from src.ui.preview.timeline import PreviewTimeline, TRACK_STEPS  # noqa: E402

#: The clip of the live run that found this (`scratch/seek-clip.mp4`): ~8.75 s, and the press landed at 16 % of it.
CLIP = 8.75
FPS = 24.0
#: The picture's moment at the press in that run: 0.4 of the clip, i.e. the press asked BACKWARD, which is the case
#: where "the state's playhead" and "the asked moment" are guaranteed to differ for as long as the seek is unanswered.
PICTURE_AT_PRESS = 0.4
#: One step of the 0..1000 track, and the tolerance a rounded value is compared with (two steps: the pointer position
#: and the value round independently).
TRACK = 1.0 / TRACK_STEPS
THUMB_TOLERANCE = 2.0 / TRACK_STEPS


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _toolkit_palette():
    """The app registers the palettes at start-up; a bare ThemeManager has none (the app is not booted here)."""
    from sli_ui_toolkit.palettes import FLUENT_DARK, FLUENT_LIGHT

    theme = ThemeManager.get_instance()
    previous = (dict(theme._light_palette), dict(theme._dark_palette))
    theme.register_palettes(FLUENT_LIGHT, FLUENT_DARK)
    yield theme
    theme._light_palette, theme._dark_palette = previous


def _state(playhead: float | None = None, *, played: float | None = None, phase=PreviewPhase.PLAYING,
           produced: float = CLIP) -> PreviewState:
    """A state of the live run's clip: fully produced (a finished file), 24 fps, the picture where `played` says."""
    moment = CLIP * PICTURE_AT_PRESS if playhead is None else float(playhead)
    return PreviewState(phase=phase, duration=CLIP, produced=produced, playhead=moment,
                        played=moment if played is None else float(played), fps=FPS)


@pytest.fixture()
def timeline():
    """The real scrubber in a real (offscreen) window: the geometry and the event route the live bar has."""
    _app()
    host = QWidget()
    host.resize(420, 64)
    widget = PreviewTimeline(host)
    widget.setGeometry(0, 20, 400, 24)
    host.show()
    QTest.qWaitForWindowExposed(host)
    widget.set_state(_state(playhead=CLIP * 0.4))
    # The host is the widget's parent in C++; the Python reference is what keeps it (and the widget) alive for the
    # whole test, since a fixture's locals are collected when it returns.
    widget._host_window = host                     # noqa: SLF001 - a lifetime reference, not a protocol
    return widget


def _point(widget: PreviewTimeline, fraction: float) -> QPoint:
    groove = widget._groove_rect()
    return QPoint(int(round(groove.left() + groove.width() * fraction)), int(groove.center().y()))


def _thumb(widget: PreviewTimeline) -> float:
    """Where the user SEES the thumb: the widget's own value, never the state's playhead."""
    return float(widget.value()) / float(TRACK_STEPS)


def _press(widget: PreviewTimeline, fraction: float) -> None:
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _point(widget, fraction))


def _move(widget: PreviewTimeline, fraction: float) -> None:
    QTest.mouseMove(widget, _point(widget, fraction), 5)


def _release(widget: PreviewTimeline, fraction: float) -> None:
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _point(widget, fraction))


def _asks(widget: PreviewTimeline) -> list[float]:
    seen: list[float] = []
    widget.seekRequested.connect(seen.append)
    return seen


# --- 1. press, no move ----------------------------------------------------------------------------------------------


def test_a_press_with_no_move_asks_once_at_the_pressed_fraction(timeline):
    """A click is ONE seek at the point pressed, and the thumb stands on it - not one frame of the old picture.

    Red before the fix for the *second* half only: the press was emitted correctly, but the next state pass moved the
    thumb back to the page's playhead, so "one seek at the pressed fraction" and "the thumb equals it" disagreed for a
    click as well as for a drag.
    """
    seen = _asks(timeline)
    _press(timeline, 0.25)
    _release(timeline, 0.25)

    assert len(seen) == 1, f"a click is one seek (got {seen})"
    assert seen[0] == pytest.approx(0.25 * CLIP, abs=0.05)
    assert _thumb(timeline) == pytest.approx(0.25, abs=THUMB_TOLERANCE)
    assert timeline.is_scrubbing() is False, "the release must clear the drag state"


# --- 2. four moves --------------------------------------------------------------------------------------------------


def test_four_moves_ask_four_times_and_the_thumb_follows_the_last(timeline):
    """Every move is its own ask at its own fraction, and the release leaves the thumb on the last one.

    A drag that asked once (at the press) and then let the state own the thumb is the owner's "the seeker is
    disconnected from the gesture": the host is never asked for the moment the finger is on.
    """
    seen = _asks(timeline)
    _press(timeline, 0.16)
    moves = (0.30, 0.45, 0.60, 0.75)
    for fraction in moves:
        _move(timeline, fraction)
    _release(timeline, moves[-1])

    assert len(seen) == 1 + len(moves), f"a press and four moves are five seeks (got {seen})"
    for asked, fraction in zip(seen[1:], moves, strict=True):
        assert asked == pytest.approx(fraction * CLIP, abs=0.06), f"the ask for {fraction} was {asked}"
    assert _thumb(timeline) == pytest.approx(moves[-1], abs=THUMB_TOLERANCE)
    assert timeline.is_scrubbing() is False


# --- 3. release -----------------------------------------------------------------------------------------------------


def test_a_release_re_asks_nothing_and_holds_the_last_target_against_a_stale_state(timeline):
    """THE defect: after the release the thumb stands on the last ask, not on the page's (stale) playhead.

    Reproduces the live shape without a host: the press asks BACKWARD (the picture is at 40 %, the finger at 16 %),
    the drag ends at 55 %, and the page's tick still reports the picture at 40 % - because the seek it forwarded has
    not been answered yet. Before the fix the very next state pass wrote 40 % into the widget: the thumb jumped back
    to the press, and the picture's own moment (the fill) then marched past it - the reported "pinned while the
    picture runs on".

    Then the ask lands (the tick reports the asked moment) and the state takes the thumb over again, so playback
    following is unchanged.
    """
    seen = _asks(timeline)
    _press(timeline, 0.16)
    _move(timeline, 0.55)
    _release(timeline, 0.55)
    assert _thumb(timeline) == pytest.approx(0.55, abs=THUMB_TOLERANCE)
    count_after_release = len(seen)

    # The page's tick: the picture is still at the press moment (the seek is in flight), and the fill with it.
    timeline.set_state(_state(playhead=CLIP * PICTURE_AT_PRESS))
    assert _thumb(timeline) == pytest.approx(0.55, abs=THUMB_TOLERANCE), (
        "a stale state pulled the thumb off the released target")
    assert len(seen) == count_after_release, "a release does not re-emit"
    assert timeline.scrub_report()["asked"]["holding"] is True, (
        "the reported authority is the ask, not the picture's last position")

    # The ask lands: the state and the ask are one number again, and the thumb follows the state from here on.
    timeline.set_state(_state(playhead=0.55 * CLIP))
    assert _thumb(timeline) == pytest.approx(0.55, abs=THUMB_TOLERANCE)
    assert timeline.scrub_report()["asked"]["holding"] is False, "a landed ask is no longer the authority"
    timeline.set_state(_state(playhead=0.62 * CLIP))
    assert _thumb(timeline) == pytest.approx(0.62, abs=THUMB_TOLERANCE), (
        "once the ask landed, playback must drive the thumb again")


# --- 4. drag in progress --------------------------------------------------------------------------------------------


def test_a_value_that_moves_without_the_move_signal_still_asks(timeline):
    """The value is the fact; `sliderMoved` is only one of its two producers.

    The toolkit owns the pointer geometry and calls `setValue` on every move, and the base class announces a drag
    through `sliderMoved` only while the toolkit's own pressed flag is up. A drag whose value moved without that
    signal moved the thumb and asked the host for nothing - the live run's `value` never left the press. Both signals
    are wired to one ask, deduped by `_last_emitted`, so this row fails if the second producer is deleted.
    """
    seen = _asks(timeline)
    timeline.sliderPressed.emit()                 # a drag in flight (no real pointer needed for this row)
    timeline.setValue(700)                        # what the toolkit's `_set_value_from_pos` does, signal-less
    assert len(seen) == 2, f"the press and the moved value are two asks (got {seen})"
    assert seen[-1] == pytest.approx(0.7 * CLIP, abs=0.06)
    assert _thumb(timeline) == pytest.approx(0.7, abs=THUMB_TOLERANCE)
    timeline.sliderReleased.emit()
    assert len(seen) == 2, f"the release must not re-ask (got {seen})"


def test_a_state_pass_during_a_drag_does_not_move_the_thumb(timeline):
    """While the button is down the pointer owns the thumb: the picture's own progress may not move it.

    The live lines show exactly this fight (`scrubbing=1` and a marching `fill`): the fill is the picture and it must
    be allowed to move, but the thumb under the finger may not.
    """
    _press(timeline, 0.30)
    for fraction in (0.45, 0.62, 0.80, 0.95):
        timeline.set_state(_state(playhead=CLIP * fraction, played=CLIP * fraction))
        assert _thumb(timeline) == pytest.approx(0.30, abs=THUMB_TOLERANCE), (
            f"a state at {fraction} moved the thumb during a drag")
        assert timeline.is_scrubbing() is True
    report = timeline.scrub_report()["paint"]
    assert report["glued"] is False, "the fill and the thumb may not be reported as one number during a drag"
    assert report["why_not_glued"] == "scrubbing (the mouse owns the thumb)"
    assert timeline.scrub_report()["asked"]["holding"] is True, (
        "the report must say the thumb is standing on an unanswered ask, not leave it to a screenshot")
    _release(timeline, 0.30)


def test_a_forward_seek_that_lands_a_tick_past_the_ask_does_not_freeze_the_thumb(timeline):
    """A forward ask is answered by the picture passing it: playback must not be mistaken for a refusal.

    The state arrives at 10 Hz, so a seek that landed perfectly can be three frames past the ask by the time the bar
    sees it. Holding the thumb until "the state equals the ask" would then freeze it mid-playback for the whole hold
    window - a new instance of the same bug. The direction recorded with the ask is what tells the two apart: an ask
    ahead of the picture is closed by the picture reaching OR passing it; an ask behind it only by the picture coming
    back (that is `test_a_release_re_asks_nothing...`, where the picture is ahead and must seek back).
    """
    ask = 0.62
    timeline.set_state(_state(playhead=CLIP * 0.40))          # the picture, before the press
    _press(timeline, ask)
    _release(timeline, ask)
    # The picture lands and plays on: 3.5 frames past the ask, then further.
    for fraction in (ask + 3.5 / FPS / CLIP, ask + 10.0 / FPS / CLIP):
        timeline.set_state(_state(playhead=CLIP * fraction))
        assert _thumb(timeline) == pytest.approx(fraction, abs=THUMB_TOLERANCE), (
            "a forward seek the picture passed was treated as unanswered")


# --- 5. a lost release ----------------------------------------------------------------------------------------------


def test_a_drag_whose_release_never_arrives_ends_and_the_ask_still_owns_the_thumb(timeline, monkeypatch):
    """A drag that never sees `sliderReleased` may not own the thumb for ever.

    The flag `_scrubbing` is set by one signal and cleared by another; a release lost to a window-system hiccup (the
    pointer leaving the window, a grab cancelled by a mask change, a lost event) left it true, and from then on no
    state could move the thumb: the permanent pin. The widget now asks the pointer (`QApplication.mouseButtons`) on
    every state pass, so a drag with no button down ends on its own - and the last ask still owns the thumb, so the
    thumb does not jump either.
    """
    _press(timeline, 0.30)
    assert timeline.is_scrubbing() is True
    # The release signal is lost: only state passes happen, with no mouse button down (`sliderPressed` was emitted by
    # hand here, exactly as a lost release leaves the widget).
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _point(timeline, 0.30))
    timeline.sliderPressed.emit()                       # claim a drag nobody is holding any more
    assert QGuiApplication.mouseButtons() == Qt.MouseButton.NoButton
    for _ in range(timeline_module.DRAG_LOST_TICKS + 1):
        timeline.set_state(_state(playhead=CLIP * PICTURE_AT_PRESS))
    assert timeline.is_scrubbing() is False, "a drag with no button down must end, not own the thumb for ever"
    assert _thumb(timeline) == pytest.approx(0.30, abs=THUMB_TOLERANCE), (
        "the end of a lost drag still leaves the thumb on what it asked for")

    # ... and an ask nobody answers cannot stand on the thumb for ever either: past the window the state wins, and
    # the snap-back is what records the refusal.
    monkeypatch.setattr(timeline_module, "ASK_HOLD_WINDOW_S", 0.0)
    timeline.set_state(_state(playhead=CLIP * 0.80))
    assert _thumb(timeline) == pytest.approx(0.80, abs=THUMB_TOLERANCE)
    assert any(entry["kind"] == "snapback" for entry in timeline.scrub_report()["recent"]), (
        "a thumb that leaves the ask without an answer must be recorded as a snap-back")


# --- the overlay's shape is not paint state --------------------------------------------------------------------------


def test_a_state_pass_does_not_reshape_the_overlay_under_the_pointer():
    """A 10 Hz state pass may not re-write the bars' geometry or the overlay's mask when nothing moved.

    The mask is what makes the bars hittable and the picture's middle the pane's; it is also the window's input shape.
    `set_state` runs ten times a second and once per seek, so on the pre-fix tree a drag reshaped the window under the
    pointer between every two moves - and a window-system shape change is exactly what can end an implicit grab (the
    gesture then continues with no press: the press landed, the moves did not). The layout is written when it CHANGES.
    """
    _app()
    pane = QWidget()
    pane.resize(621, 721)
    pane.show()
    QTest.qWaitForWindowExposed(pane)
    controls = PreviewControls(pane, pane=pane)
    control = _state(playhead=CLIP * PICTURE_AT_PRESS)

    shaped: list[object] = []
    original = controls.setMask
    controls.setMask = lambda region: (shaped.append(region), original(region))[1]   # noqa: SLF001 - a write counter
    moved: list[object] = []
    for bar in (controls.top_bar, controls.bottom_bar):
        bar_set = bar.setGeometry
        bar.setGeometry = lambda rect, _set=bar_set, _bar=bar: (moved.append((_bar, rect)), _set(rect))[1]

    for _ in range(5):
        controls.set_state(control)                  # the tick, unchanged layout
    assert shaped == [], f"a state pass re-shaped the overlay {len(shaped)} times"
    assert moved == [], f"a state pass re-laid the bars {len(moved)} times"

    # ... and a real change is still applied: the overlay is not frozen.
    pane.resize(500, 521)
    _app().processEvents()
    assert shaped, "a resize must still re-shape the overlay"
    assert controls.mask().contains(controls.bottom_bar.geometry().center())


# --- 6. the glue: the bar's seek path -------------------------------------------------------------------------------


class _Surface:
    def set_view_mode(self, mode: str) -> None: ...
    def set_zoom(self, zoom: float, *, fit: bool) -> None: ...
    def set_compare(self, held: bool) -> None: ...
    def set_fullscreen(self, fullscreen: bool) -> None: ...


class _Backend:
    """A finished file whose seeks are accepted and recorded; the PICTURE does not move (`position` stays)."""

    kind = PreviewBackendKind.FINISHED_FILE

    def __init__(self) -> None:
        self.seeks: list[float] = []
        self.position = CLIP * PICTURE_AT_PRESS

    def play(self) -> None: ...
    def pause(self) -> None: ...
    def step(self, frames: int) -> bool: return True
    def seek(self, seconds: float, *, exact: bool) -> bool:
        self.seeks.append(float(seconds))
        return True

    @property
    def produced(self) -> float: return CLIP
    @property
    def is_playing(self) -> bool: return True


def test_the_bar_asks_the_controller_for_every_move_and_the_thumb_holds_the_last(timeline):
    """The whole shipped glue: the bar's `seekRequested` -> `SEEK_TO` seconds -> the session's ticket, one per move.

    Runs `PreviewControls` over a real controller, with the page's tick pushed between the moves (`observe`), which is
    what a live drag has: the page reports the picture it is painting, not the request. The bar must ask for each
    move and, after the release, the thumb must stand on the last ask - the state pushed by the tick says otherwise.
    """
    pane = QWidget()
    pane.resize(621, 400)
    pane.show()
    QTest.qWaitForWindowExposed(pane)
    controls = PreviewControls(pane, pane=pane)
    backend = _Backend()
    controller = PreviewController(_Surface())
    controller.register_backend(_Backend.kind, backend)
    controller.set_backend(_Backend.kind)
    controls.commandRequested.connect(controller.dispatch)
    controller.stateChanged.connect(controls.set_state)
    controller.observe(is_playing=True, has_picture=True, picture=True, playhead=CLIP * PICTURE_AT_PRESS,
                       played=CLIP * PICTURE_AT_PRESS, duration=CLIP, produced=CLIP, fps=FPS,
                       seek_ceiling=CLIP)

    widget = controls.timeline
    moves = (0.30, 0.45, 0.60, 0.75)
    _press(widget, 0.16)
    seen: list[float] = []
    controller.status.connect(seen.append)
    for fraction in moves:
        _move(widget, fraction)
        # The page's 100 ms tick: the picture is still where it was when the drag started.
        controller.observe(playhead=CLIP * PICTURE_AT_PRESS, played=CLIP * PICTURE_AT_PRESS)
    _release(widget, moves[-1])

    assert len(backend.seeks) == 1 + len(moves), f"the drag asked the host {backend.seeks}"
    for asked, fraction in zip(backend.seeks[1:], moves, strict=True):
        assert asked == pytest.approx(fraction * CLIP, abs=0.06), f"the ask for {fraction} was {asked}"
    assert _thumb(widget) == pytest.approx(moves[-1], abs=THUMB_TOLERANCE), (
        "after the release the thumb must stand where the pane was asked to go")
    assert [line for line in seen if "refused" in line] == []
    assert controller.state().playhead == pytest.approx(CLIP * PICTURE_AT_PRESS), (
        "the controller's state is the picture (the tick says so); only the thumb holds the ask")
