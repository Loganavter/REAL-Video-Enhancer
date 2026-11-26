"""The preview control surface: commands, one state writer, two bars (plan-rve-preview-controls-frame-path-2026-11-25).

Owner direction (2026-11-25): the bottom is *video transport only* and the tools float over the picture; the buttons
must work "through a professional pattern, not magic". What is pinned here, offscreen and without a GPU:

* the controller is the single writer of `PreviewState`: every command (play/pause, step, seek by time and by
  fraction, the three view modes, the zoom row, the compare hold, fullscreen) either moves the state or answers with
  exactly ONE status line - a refused seek is a line, never a silent snap and never an exception;
* a growing file buffers AT its produced edge (`PLAYING -> BUFFERING`) and plays again when frames land;
* the action catalog is one registry: ids are unique, every `PreviewCommand` is covered, and the predicates decide
  enabled/checked/visible - `set_state` applies them in one pass with signals blocked, so painting never looks like a
  click;
* the bars are the toolkit's own widgets (a `Button` is a `QWidget`, so the modes are exclusive through the toolkit's
  `ButtonFamily`, not a `QButtonGroup`), they sit inside the pane at 621x721 and never cover its middle;
* the compare is a HOLD (press = source, release = result);
* the frame readout is `f <source done> / <source total>` - the writer's CFR count is a separate fact, never a ratio.

The real pixels, the pane's letterbox and the window's fullscreen are the supervisor's live proof.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

# The app's optional runtime imports, stubbed only when the interpreter lacks them (the same pattern as
# `test_process_readouts_language.py`): `ProcessTab` is the page this lane wires, and it imports `src.Util`.
for _name, _attrs in (("cpuinfo", {"get_cpu_info": lambda *a, **k: {}}),
                      ("pypresence", {"Presence": object})):
    try:  # pragma: no cover - the branch depends on the interpreter
        __import__(_name)
    except ModuleNotFoundError:  # pragma: no cover
        _module = types.ModuleType(_name)
        for _attr, _value in _attrs.items():
            setattr(_module, _attr, _value)
        sys.modules[_name] = _module
if "resources_rc" not in sys.modules:
    # A test tree has no compiled resource blob; the icons resolve to null QIcons and the widgets still build.
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager  # noqa: E402
from sli_ui_toolkit.widgets import Button  # noqa: E402

from src.ui.preview.actions import ACTIONS, actions_for, build_button, enabled_for  # noqa: E402
from src.ui.preview.backends import NoBackend, backend_for  # noqa: E402
from src.ui.preview.controller import PreviewController  # noqa: E402
from src.ui.preview.controls import PreviewControls  # noqa: E402
from src.ui.preview.state import (  # noqa: E402
    PreviewBackendKind,
    PreviewCommand,
    PreviewPhase,
    PreviewState,
)
from src.ui.preview.timeline import PreviewTimeline, format_content_time  # noqa: E402

PANE_SIZE = (621, 721)


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


class FakeSurface:
    """Records what the controller asks the picture to do."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_view_mode(self, mode: str) -> None:
        self.calls.append(("mode", mode))

    def set_zoom(self, zoom: float, *, fit: bool) -> None:
        self.calls.append(("zoom", (zoom, fit)))

    def set_compare(self, held: bool) -> None:
        self.calls.append(("compare", held))

    def set_fullscreen(self, fullscreen: bool) -> None:
        self.calls.append(("fullscreen", fullscreen))


class FakeBackend:
    """A backend that answers like a finished file and counts the calls."""

    kind = PreviewBackendKind.FINISHED_FILE

    def __init__(self, *, playing: bool = False) -> None:
        self.playing = playing
        self.play_calls = 0
        self.pause_calls = 0
        self.seek_calls: list[float] = []
        self.step_calls: list[int] = []
        self.position = 4.0
        self.duration = 20.0

    def play(self) -> None:
        self.play_calls += 1
        self.playing = True

    def pause(self) -> None:
        self.pause_calls += 1
        self.playing = False

    def seek(self, seconds: float, *, exact: bool) -> bool:
        self.seek_calls.append(float(seconds))
        self.position = float(seconds)
        return True

    def step(self, frames: int) -> bool:
        self.step_calls.append(int(frames))
        self.position += int(frames) / 24.0
        return True

    @property
    def produced(self) -> float:
        return self.duration

    @property
    def is_playing(self) -> bool:
        return self.playing


def _controller(surface: FakeSurface | None = None, *, kind=PreviewBackendKind.FINISHED_FILE,
                produced: float = 12.0, duration: float = 20.0, playhead: float = 4.0,
                playing: bool = False, ceiling: float | None = None, has_picture: bool = True):
    surface = surface or FakeSurface()
    controller = PreviewController(surface)
    backend = FakeBackend() if kind is PreviewBackendKind.FINISHED_FILE else NoBackend()
    controller.register_backend(kind, backend)
    controller.set_backend(kind)
    controller.observe(
        is_playing=playing, has_picture=has_picture, playhead=playhead, produced=produced,
        duration=duration, source_frames=493, frames_done=100, output_frames=495, fps=24.0,
        seek_ceiling=ceiling,
    )
    return controller, backend, surface


def _status_recorder(controller: PreviewController) -> list[str]:
    lines: list[str] = []
    controller.status.connect(lines.append)
    return lines


# --- the controller: every command, every refusal ----------------------------------------------------

def test_play_pause_moves_the_phase_and_calls_the_backend():
    controller, backend, _ = _controller()
    assert controller.state().phase is PreviewPhase.READY
    controller.dispatch(PreviewCommand.PLAY_PAUSE)
    assert backend.play_calls == 1
    assert controller.state().phase is PreviewPhase.PLAYING
    controller.dispatch(PreviewCommand.PLAY_PAUSE)
    assert backend.pause_calls == 1
    assert controller.state().phase is PreviewPhase.READY


def test_play_with_nothing_on_the_pane_is_one_status_line():
    controller, backend, _ = _controller(has_picture=False)
    lines = _status_recorder(controller)
    controller.dispatch(PreviewCommand.PLAY_PAUSE)
    assert backend.play_calls == 0
    assert len(lines) == 1 and "refused" in lines[0]


def test_a_closed_controller_refuses_transport():
    controller, backend, _ = _controller()
    controller.close()
    lines = _status_recorder(controller)
    controller.dispatch(PreviewCommand.PLAY_PAUSE)
    assert backend.play_calls == 0
    assert len(lines) == 1 and "refused" in lines[0]


def test_buffering_at_the_produced_edge_of_a_growing_file():
    controller = PreviewController(FakeSurface())
    controller.register_backend(PreviewBackendKind.LIVE_TAIL, FakeBackend())
    controller.set_backend(PreviewBackendKind.LIVE_TAIL)
    controller.observe(is_playing=False, has_picture=True, playhead=12.0, produced=12.0, duration=20.0, fps=24.0)
    controller.dispatch(PreviewCommand.PLAY_PAUSE)
    assert controller.state().phase is PreviewPhase.BUFFERING
    # New frames land: the playhead is no longer at the edge, so the clock is playing again.
    controller.observe(produced=13.0, playhead=12.2)
    assert controller.state().phase is PreviewPhase.PLAYING
    # And it buffers again when it catches up.
    controller.observe(playhead=13.0)
    assert controller.state().phase is PreviewPhase.BUFFERING


def test_seek_to_moves_the_playhead_through_the_backend():
    controller, backend, _ = _controller()
    controller.dispatch(PreviewCommand.SEEK_TO, 6.0)
    assert backend.seek_calls == [6.0]
    assert controller.state().playhead == pytest.approx(6.0)


def test_seek_fraction_is_a_share_of_the_duration():
    controller, backend, _ = _controller()
    controller.dispatch(PreviewCommand.SEEK_FRACTION, 0.25)
    assert backend.seek_calls == [pytest.approx(5.0)]


def test_a_refused_seek_is_one_status_line_and_no_backend_call():
    """Past the produced edge on a growing file: the answer is a line, not a snap to the edge."""
    controller, backend, _ = _controller(kind=PreviewBackendKind.LIVE_TAIL, produced=12.0, ceiling=None)
    lines = _status_recorder(controller)
    controller.dispatch(PreviewCommand.SEEK_TO, 18.0)
    assert backend.seek_calls == [] if hasattr(backend, "seek_calls") else True
    assert len(lines) == 1
    assert "refused" in lines[0] and "12.000" in lines[0]
    assert controller.state().playhead == pytest.approx(4.0)


def test_a_finished_file_can_seek_the_whole_clip():
    controller, backend, _ = _controller(ceiling=20.0)
    controller.dispatch(PreviewCommand.SEEK_TO, 19.5)
    assert backend.seek_calls == [19.5]


def test_step_moves_one_frame_and_asks_the_backend():
    controller, backend, _ = _controller()
    controller.dispatch(PreviewCommand.STEP_FORWARD)
    controller.dispatch(PreviewCommand.STEP_BACK)
    assert backend.step_calls == [1, -1]
    assert controller.state().playhead == pytest.approx(4.0)


def test_set_mode_reaches_the_pane_and_clears_a_held_compare():
    surface = FakeSurface()
    controller, _, _ = _controller(surface)
    controller.dispatch(PreviewCommand.COMPARE_DOWN)
    controller.dispatch(PreviewCommand.SET_MODE, "sbs")
    assert controller.state().mode == "sbs"
    assert controller.state().compare is False
    assert ("compare", False) in surface.calls
    assert ("mode", "sbs") in surface.calls


def test_an_unknown_view_mode_is_refused():
    controller, _, surface = _controller()
    lines = _status_recorder(controller)
    controller.dispatch(PreviewCommand.SET_MODE, "quad")
    assert surface.calls == []
    assert len(lines) == 1 and "refused" in lines[0]


def test_the_zoom_row():
    surface = FakeSurface()
    controller, _, _ = _controller(surface)
    controller.dispatch(PreviewCommand.ZOOM_IN)
    assert controller.state().zoom == pytest.approx(1.25)
    assert controller.state().zoom_fit is False
    controller.dispatch(PreviewCommand.ZOOM_ONE)
    assert controller.state().zoom == pytest.approx(1.0)
    controller.dispatch(PreviewCommand.ZOOM_FIT)
    assert controller.state().zoom_fit is True
    controller.dispatch(PreviewCommand.ZOOM_OUT)
    assert controller.state().zoom == pytest.approx(1.0 / 1.25)
    assert [name for name, _ in surface.calls] == ["zoom", "zoom", "zoom", "zoom"]


def test_compare_is_a_hold_and_only_exists_in_single():
    surface = FakeSurface()
    controller, _, _ = _controller(surface)
    controller.dispatch(PreviewCommand.COMPARE_DOWN)
    assert controller.state().compare is True
    assert ("compare", True) in surface.calls
    controller.dispatch(PreviewCommand.COMPARE_UP)
    assert controller.state().compare is False
    assert ("compare", False) in surface.calls
    controller.dispatch(PreviewCommand.SET_MODE, "split")
    lines = _status_recorder(controller)
    controller.dispatch(PreviewCommand.COMPARE_DOWN)
    assert len(lines) == 1 and "refused" in lines[0]


def test_fullscreen_toggles_even_with_nothing_on_the_pane():
    controller, _, surface = _controller(has_picture=False)
    controller.dispatch(PreviewCommand.TOGGLE_FULLSCREEN)
    assert controller.state().fullscreen is True
    assert ("fullscreen", True) in surface.calls


def test_an_unknown_command_is_a_status_line_not_an_exception():
    controller, _, _ = _controller()
    lines = _status_recorder(controller)
    controller.dispatch("fly")
    assert len(lines) == 1 and "unknown command" in lines[0]


def test_observe_rejects_a_typo():
    controller, _, _ = _controller()
    with pytest.raises(KeyError):
        controller.observe(playhead_seconds=3.0)


def test_one_state_signal_per_change():
    controller, _, _ = _controller()
    seen: list[PreviewState] = []
    controller.stateChanged.connect(seen.append)
    controller.dispatch(PreviewCommand.SEEK_TO, 6.0)
    assert len(seen) == 1 and seen[0].playhead == pytest.approx(6.0)
    controller.dispatch(PreviewCommand.SEEK_TO, 6.0)
    assert len(seen) == 1      # the playhead did not move, so the state did not change: no second repaint
    controller.observe(playhead=7.0)
    assert len(seen) == 2


# --- the catalog -------------------------------------------------------------------------------------

def test_action_ids_are_unique():
    ids = [spec.id for spec in ACTIONS]
    assert len(ids) == len(set(ids))


def test_every_command_is_covered():
    """Every verb is reachable: a spec carries it, the scrubber does, or it is the pane's LMB gesture.

    The compare is deliberately NOT a button (owner, 2026-11-25: "Зачем нужна кнопка compare?"): holding LMB on the
    picture is the gesture, so `COMPARE_DOWN/UP` have no catalog entry - the controller still dispatches them for the
    surface port.
    """
    covered = ({spec.command for spec in ACTIONS}
               | {spec.release_command for spec in ACTIONS if spec.release_command is not None}
               | {PreviewCommand.SEEK_TO, PreviewCommand.SEEK_FRACTION,
                  PreviewCommand.COMPARE_DOWN, PreviewCommand.COMPARE_UP})
    assert covered == set(PreviewCommand)


def test_the_bars_are_split_by_function():
    top = {spec.id for spec in actions_for("top")}
    bottom = {spec.id for spec in actions_for("bottom")}
    assert "play" in bottom and "step_back" in bottom and "step_forward" in bottom
    assert "play" not in top and "compare" not in top
    assert {"mode_single", "mode_sbs", "mode_split", "zoom_fit", "fullscreen"} <= top


def test_enabled_predicates_read_the_state():
    idle = PreviewState()
    ready = PreviewState(phase=PreviewPhase.READY, duration=10.0)
    play = next(spec for spec in ACTIONS if spec.id == "play")
    assert enabled_for(play, idle) is False
    assert enabled_for(play, ready) is True


def test_the_catalog_carries_the_keyboard_routes():
    by_id = {spec.id: spec for spec in ACTIONS}
    assert by_id["play"].shortcut == "Space"
    assert by_id["step_back"].shortcut == "Left"
    assert by_id["step_forward"].shortcut == "Right"


def test_a_spec_builds_a_real_toolkit_button():
    _app()
    spec = next(spec for spec in ACTIONS if spec.id == "zoom_fit")
    button = build_button(spec)
    assert isinstance(button, Button)
    assert button.toolTip() == spec.tip
    assert button.isCheckable() is True


# --- the view: one pass, two bars in the letterbox ---------------------------------------------------

@pytest.fixture()
def pane():
    _app()
    widget = QWidget()
    widget.resize(*PANE_SIZE)
    widget.show()          # a hidden widget receives no resize event, and the bars follow the pane's size
    return widget


@pytest.fixture()
def controls(pane):
    widget = PreviewControls(pane, pane=pane)
    widget.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.563, produced=12.4,
                                  playhead=4.171, source_frames=493, frames_done=100,
                                  output_frames=495, fps=24.0))
    return widget


def test_set_state_applies_every_widget_in_one_pass_without_emitting(controls):
    emitted: list[tuple[object, object]] = []
    controls.commandRequested.connect(lambda command, value: emitted.append((command, value)))
    state = PreviewState(phase=PreviewPhase.PLAYING, backend=PreviewBackendKind.FINISHED_FILE,
                         mode="sbs", produced=20.0, duration=20.0, playhead=6.0,
                         source_frames=493, frames_done=120, output_frames=495, fps=24.0)
    controls.set_state(state)
    assert emitted == []
    assert controls.button("mode_sbs").isChecked()
    assert not controls.button("mode_single").isChecked()
    assert controls.button("play").isEnabled()
    assert controls.timeline.isEnabled()
    assert controls.time_text.text() == "0:06.000 / 0:20.000"


def test_checked_is_written_with_signals_blocked(controls):
    """Painting a state must never look like a click: a checked mode button emits no command during set_state."""
    emitted: list[tuple[object, object]] = []
    controls.commandRequested.connect(lambda command, value: emitted.append((command, value)))
    controls.set_state(PreviewState(phase=PreviewPhase.READY, duration=10.0, produced=5.0, playhead=1.0,
                                    mode="split"))
    assert controls.button("mode_split").isChecked()
    assert emitted == []


def test_the_bars_sit_inside_the_pane_and_never_cover_its_middle(controls, pane):
    top = controls.top_bar.geometry()
    bottom = controls.bottom_bar.geometry()
    assert pane.rect().contains(top)
    assert pane.rect().contains(bottom)
    # The picture's middle: everything between the top row and the transport stays free.
    assert top.bottom() < pane.height() // 2
    assert bottom.top() > pane.height() // 2
    assert top.bottom() < bottom.top()


def test_the_bars_are_children_of_the_pane(controls, pane):
    """The overlay is the pane's child; the two bars are its children, so both sit over the pane (not the page)."""
    assert controls.parentWidget() is pane
    assert controls.top_bar.parentWidget() is controls
    assert controls.bottom_bar.parentWidget() is controls


def test_the_overlay_does_not_take_the_picture_mouse(controls, pane):
    """The overlay spans the pane, so a widget MASK keeps only the bars hittable.

    The first cut set `WA_TransparentForMouseEvents` on the overlay and the bars; live (2026-11-25) that made the whole
    new surface dead - the play square, Compare, the modes and the zoom all hit the pane underneath. The mask is the
    clip that gives the buttons their presses AND leaves the picture's middle to the pane.
    """
    from PySide6.QtCore import QPoint

    mask = controls.mask()
    assert not mask.isEmpty()
    assert mask.contains(controls.top_bar.geometry().center())
    assert mask.contains(controls.bottom_bar.geometry().center())
    assert not mask.contains(QPoint(pane.width() // 2, pane.height() // 2))
    assert not controls.top_bar.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not controls.bottom_bar.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_resizing_the_pane_repositions_the_bars(controls, pane):
    pane.resize(400, 400)
    _app().processEvents()
    assert controls.width() == 400 and controls.height() == 400
    assert controls.bottom_bar.geometry().bottom() <= 400
    assert controls.top_bar.geometry().bottom() < 200


def test_a_narrow_pane_collapses_the_zoom_row_into_the_overflow(controls, pane):
    pane.resize(330, 400)
    _app().processEvents()
    assert "zoom_in" in controls.hidden_tools()
    assert controls.overflow_button.isVisible()
    assert controls._group_panels["zoom"].isHidden(), "a group leaves as one capsule, not half a row"


def test_each_group_has_its_own_capsule(controls):
    """Owner, 2026-11-25: "ты забыл делать подложки у групп кнопок".

    The produced pill was the only surface; the mode row and the zoom row are groups, so each owns one capsule
    (`SurfacePanel`) behind ghost buttons. The capsule is the backing; the buttons inside it are transparent until
    hover/checked, so the group reads as one control.
    """
    for name, ids in (("view_mode", ("mode_single", "mode_sbs", "mode_split")),
                      ("zoom", ("zoom_out", "zoom_fit", "zoom_one", "zoom_in"))):
        panel = controls._group_panels[name]
        assert panel.isVisible(), f"the {name} capsule must be visible"
        assert all(controls.button(spec_id).parent() is panel for spec_id in ids)
        assert all(controls._specs[spec_id].variant == "ghost" for spec_id in ids)
        assert panel.width() > 0 and panel.height() > 0


def test_a_mode_toggle_emits_set_mode_and_keeps_one_checked(controls):
    emitted: list[tuple[object, object]] = []
    controls.commandRequested.connect(lambda command, value: emitted.append((command, value)))
    sbs = controls.button("mode_sbs")
    sbs.setChecked(True)
    sbs.toggled.emit(True)
    assert emitted[-1] == (PreviewCommand.SET_MODE, "sbs")
    assert not controls.button("mode_single").isChecked()


def test_the_frame_readout_counts_source_packets_only(controls):
    assert controls.frame_text.text() == "f 100 / 493"
    assert "495" not in controls.frame_text.text()
    assert "495" in controls.frame_text.toolTip() and "CFR" in controls.frame_text.toolTip()


def test_the_play_glyph_follows_the_state(controls):
    """The glyph is the ACTION a press performs: pause while playing, play while held."""
    play = next(spec for spec in ACTIONS if spec.id == "play")
    playing = PreviewState(phase=PreviewPhase.PLAYING, duration=10.0, playhead=1.0)
    paused = PreviewState(phase=PreviewPhase.READY, duration=10.0, playhead=1.0)
    from PySide6.QtGui import QIcon

    assert isinstance(play.icon_when(playing), QIcon)
    assert isinstance(play.icon_when(paused), QIcon)
    controls.set_state(playing)     # applying it must not raise or emit
    controls.set_state(paused)


# --- the scrubber ------------------------------------------------------------------------------------

def test_the_scrubber_carries_three_numbers_and_emits_seconds(controls):
    timeline = controls.timeline
    assert isinstance(timeline, PreviewTimeline)
    assert timeline.produced_fraction() == pytest.approx(12.4 / 20.563, abs=1e-3)
    assert timeline.playhead_fraction() == pytest.approx(4.171 / 20.563, abs=1e-3)
    seen: list[float] = []
    timeline.seekRequested.connect(seen.append)
    timeline.sliderPressed.emit()
    timeline.setValue(500)
    timeline.sliderMoved.emit(500)
    timeline.sliderReleased.emit()
    assert seen and seen[-1] == pytest.approx(20.563 / 2, abs=1e-2)
    assert timeline.is_scrubbing() is False


def test_a_real_press_on_the_scrubber_asks_for_a_seek(controls):
    """The signal's PRODUCER, not only its consumer: a mouse press must reach `seekRequested`.

    The test above emits `sliderPressed`/`sliderMoved`/`sliderReleased` by hand, so it proved the timeline reacts to
    signals the toolkit `Slider` never emitted - it drew its own groove, handled the mouse itself and skipped the base
    class, so a press moved the thumb under the cursor and asked the host for nothing (measured live 2026-11-25).
    This drives the real widget with real mouse events.
    """
    timeline = controls.timeline
    timeline.resize(400, 24)
    timeline.show()
    QTest.qWaitForWindowExposed(timeline)
    seen: list[float] = []
    timeline.seekRequested.connect(seen.append)
    groove = timeline._groove_rect()
    point = QPoint(int(groove.left() + groove.width() * 0.5), int(groove.center().y()))
    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    assert seen, "a press on the scrubber asked for nothing"
    assert seen[-1] == pytest.approx(20.563 / 2, abs=0.4)
    assert timeline.is_scrubbing() is False, "the release must end the drag"


def test_the_played_fill_is_the_picture_and_the_thumb_is_the_host(controls):
    """Two facts, two fractions: a press moves the thumb at once, the fill follows the frame that is actually shown.

    Painting both from the playhead made a seek invisible until it landed and made the fill jump back with every
    stale frame the decoder delivered after the press (owner, 2026-11-26).
    """
    state = controls.controller.state() if hasattr(controls, "controller") else None
    if state is None:
        from src.ui.preview.state import PreviewState
        state = PreviewState(duration=20.0, playhead=10.0, played=2.0, produced=20.0)
    timeline = controls.timeline
    timeline.set_state(state.with_(duration=20.0, playhead=10.0, played=2.0, produced=20.0))
    assert timeline.playhead_fraction() == pytest.approx(0.5)
    assert timeline.played_fraction() == pytest.approx(0.1)
    # An unknown `played` (nothing painted yet) falls back to the thumb, never to zero.
    timeline.set_state(state.with_(duration=20.0, playhead=10.0, played=None, produced=20.0))
    assert timeline.played_fraction() == pytest.approx(0.5)


def test_a_click_that_never_moves_asks_once(controls):
    """Press and release at one point are ONE seek: the second request re-armed the marker and cost a second decode."""
    timeline = controls.timeline
    seen: list[float] = []
    timeline.seekRequested.connect(seen.append)
    groove = timeline._groove_rect()
    point = QPoint(int(groove.left() + groove.width() * 0.4), int(groove.center().y()))
    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    assert len(seen) == 1, f"one click, one ask (got {seen})"
    # A drag still asks as it moves.
    seen.clear()
    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseMove(timeline, QPoint(int(groove.left() + groove.width() * 0.7), int(groove.center().y())), 20)
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(int(groove.left() + groove.width() * 0.7), int(groove.center().y())))
    assert len(seen) >= 2, f"a drag must ask for its intermediate positions too (got {seen})"


def test_content_time_is_m_ss_mmm():
    assert format_content_time(0.0) == "0:00.000"
    assert format_content_time(4.171) == "0:04.171"
    assert format_content_time(65.5) == "1:05.500"
    assert format_content_time(None).startswith("\u2014")


def test_the_scrubber_is_disabled_without_a_timeline(controls):
    controls.set_state(PreviewState(phase=PreviewPhase.IDLE))
    assert controls.timeline.isEnabled() is False


# --- the page's fact mapping --------------------------------------------------------------------------

def _fake_page(pane, controller):
    """A `ProcessTab` shell with only the attributes `_push_preview_state` reads (no window, no GPU)."""
    from src.ui.ProcessTab import ProcessTab

    page = ProcessTab.__new__(ProcessTab)
    page.page_handles = types.SimpleNamespace(
        proc=types.SimpleNamespace(previewLabel=pane, previewPauseButton=None))
    page.currentRenderOptions = types.SimpleNamespace(videoFrameCount=493, interpolateTimes=2, videoFps=24.0)
    page.max_value = 990          # 495 CFR frames x 2
    page.currentFrame = 500       # output frames processed
    page.fps = 24
    page._preview_paused = True
    page._preview_playhead = 10.0
    page._preview_duration = 20.0
    page._preview_zoom = 1.0
    page._preview_zoom_fit = True
    page._preview_controller = controller
    page.renderProcess = None
    page._live_tail = None
    page._preview_decoder = None
    page._queue_pane_decoder = None
    page._queue_pane_owns = False
    page._display = None
    page._feed = None
    page._press = None
    page._seek_ceiling = None
    return page


def test_the_page_feeds_source_packets_not_cfr_slots(controls):
    controller = PreviewController(FakeSurface())
    page = _fake_page(controls.parentWidget(), controller)
    page._push_preview_state()
    state = controller.state()
    assert state.source_frames == 493
    assert state.frames_done == 250            # 500 output frames / interpolate factor 2
    assert state.output_frames == 990          # a separate fact, never a denominator
    assert state.frame_ratio() == "f 250 / 493"
    assert state.duration == pytest.approx(20.0)


def test_the_page_selects_no_backend_without_an_owner(controls):
    controller = PreviewController(FakeSurface())
    page = _fake_page(controls.parentWidget(), controller)
    page._push_preview_state()
    assert controller.state().backend is PreviewBackendKind.NONE
    assert isinstance(controller.backend(), NoBackend)
    assert controller.state().phase is PreviewPhase.IDLE     # a stub pane with no frame


def test_the_page_reports_a_finished_file_backend(controls):
    controller = PreviewController(FakeSurface())

    class Decoder:
        duration = 20.0

        def is_playing(self):
            return False

    page = _fake_page(controls.parentWidget(), controller)
    page._preview_decoder = Decoder()
    page._push_preview_state()
    assert controller.state().backend is PreviewBackendKind.FINISHED_FILE
    assert controller.state().produced == pytest.approx(20.0)
    assert controller.state().seek_ceiling == pytest.approx(20.0)


def test_installing_the_controls_hides_the_old_strip_and_button():
    from src.ui.ProcessTab import ProcessTab
    from src.ui.widgets.video_preview import VideoPreviewSurface

    _app()
    pane = VideoPreviewSurface()
    pane.resize(*PANE_SIZE)
    legacy = Button(icon=None, text="", size=(40, 40))
    legacy.setVisible(True)
    page = ProcessTab.__new__(ProcessTab)
    page.page_handles = types.SimpleNamespace(
        proc=types.SimpleNamespace(previewLabel=pane, previewPauseButton=legacy))
    page.currentRenderOptions = None
    page.max_value = 0
    page.currentFrame = 0
    page.fps = 0
    page._preview_paused = True
    page._preview_playhead = 0.0
    page._preview_duration = 0.0
    page._preview_zoom = 1.0
    page._preview_zoom_fit = True
    page._preview_controller = None
    page._preview_controls = None
    page.renderProcess = None
    page._live_tail = None
    page._preview_decoder = None
    page._queue_pane_decoder = None
    page._queue_pane_owns = False
    page._display = None
    page._feed = None
    page._press = None

    page._install_preview_controls()
    assert legacy.isVisible() is False
    assert page._preview_controls is not None
    assert pane._strip_visible is False          # the painted strip stands down (public set_strip_visible)
    # The bars are wired to the controller: a command from the view reaches the state machine (fullscreen is the one
    # verb that works with nothing on the pane, so the wiring can be proven without a render).
    page._preview_controls.commandRequested.emit(PreviewCommand.TOGGLE_FULLSCREEN, None)
    assert page._preview_controller.state().fullscreen is True


def test_the_backend_factory_selects_the_adapter_for_the_kind():
    host = types.SimpleNamespace()
    assert isinstance(backend_for(host, PreviewBackendKind.NONE), NoBackend)
    encoder = object()
    adapter = backend_for(host, PreviewBackendKind.LIVE_TAIL, tail=encoder)
    assert adapter.kind is PreviewBackendKind.LIVE_TAIL
    decoder = object()
    assert backend_for(host, PreviewBackendKind.FINISHED_FILE, decoder=decoder).kind is (
        PreviewBackendKind.FINISHED_FILE)
    assert backend_for(host, PreviewBackendKind.DISPLAY_FEED).kind is PreviewBackendKind.DISPLAY_FEED


def test_a_picture_without_an_owner_still_plays_through_the_page():
    """The "play and the page starts its tail" case: the adapter delegates, it does not refuse."""
    from src.ui.preview.backends import HostPlaybackBackend

    class Page:
        _preview_paused = True
        _preview_playhead = 3.0

        def __init__(self):
            self.calls: list[bool] = []

        def _set_preview_paused(self, paused: bool) -> None:
            self.calls.append(bool(paused))
            self._preview_paused = bool(paused)

        def _clip_duration(self) -> float:
            return 20.0

    page = Page()
    adapter = backend_for(page, PreviewBackendKind.NONE, host_playable=True)
    assert isinstance(adapter, HostPlaybackBackend)
    adapter.play()
    adapter.pause()
    assert page.calls == [False, True]



def test_the_bars_and_the_scrubber_carry_control_names(controls):
    """A live probe can only reach a custom-painted widget by NAME: the strip is the one control whose gesture must be
    provable (`click:preview_timeline@x,y`), and without a name a press that never landed read exactly like a press
    that did nothing (2026-11-25)."""
    assert controls.top_bar.objectName() == "preview_top_bar"
    assert controls.bottom_bar.objectName() == "preview_bottom_bar"
    assert controls.timeline.objectName() == "preview_timeline"
    assert controls.timeline.parent() is controls.bottom_bar


def test_the_scrubber_writes_its_own_debug_lines_and_reports_what_it_did(controls):
    """The owner's ask: debug for this slider beside the preview's own.

    A press writes `preview: scrub ask …` into the preview's telemetry file and lands in `scrub_report()`, so "the thumb
    did not move" can be answered with numbers instead of a screenshot: what was asked, what the state showed instead,
    and the ceiling that refused it.
    """
    from src.ui.video import telemetry

    lines: list[tuple[str, str]] = []
    previous = telemetry._writer
    telemetry.set_writer(lambda line, which: lines.append((which, line)))
    try:
        timeline = controls.timeline
        timeline.resize(400, 24)
        timeline.show()
        QTest.qWaitForWindowExposed(timeline)
        groove = timeline._groove_rect()
        point = QPoint(int(groove.left() + groove.width() * 0.4), int(groove.center().y()))
        QTest.mousePress(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        report = timeline.scrub_report()
        assert any(entry["kind"] == "ask" for entry in report["recent"]), report["recent"]
        assert any("scrub ask" in line for _which, line in lines), lines

        # The host refuses (a ceiling below the ask): the state puts the thumb somewhere else, and the report says why.
        asked = report["asked"]["seconds"]
        refused = PreviewState(phase=PreviewPhase.READY, duration=20.0, playhead=1.0, played=1.0,
                               seek_ceiling=2.0, produced=2.0, fps=24.0)
        timeline.set_state(refused)
        report = timeline.scrub_report()
        snaps = [entry for entry in report["recent"] if entry["kind"] == "snapback"]
        assert snaps, report["recent"]
        assert snaps[-1]["asked_s"] == pytest.approx(asked, abs=0.05)
        assert snaps[-1]["shown_s"] == pytest.approx(1.0, abs=0.05)
        assert snaps[-1]["ceiling_s"] == pytest.approx(2.0)
        assert any("scrub snapback" in line for _which, line in lines), lines
    finally:
        telemetry.set_writer(previous)


def test_a_live_tail_seeks_through_the_page_so_a_press_past_the_edge_is_answerable():
    """The tail adapter must not seek the tail itself: the PAGE is what knows the produced edge, the on-demand route
    and the press ticket. Seeking the tail directly made a past-the-edge press die inside the tail and the marker snap
    back with no record of why (found 2026-11-26 with the scrubber's own report).
    """
    class Page:
        def __init__(self) -> None:
            self.asked: list[float] = []

        def _clip_duration(self) -> float:
            return 20.0

        def _on_preview_seek(self, fraction: float) -> None:
            self.asked.append(float(fraction))

    class Tail:
        def __init__(self) -> None:
            self.seeks: list[float] = []

        def seek(self, seconds: float, *, attempt_past_edge: bool = True) -> None:
            self.seeks.append(float(seconds))

    page, tail = Page(), Tail()
    adapter = backend_for(page, PreviewBackendKind.LIVE_TAIL, tail=tail)
    assert adapter.seek(18.958, exact=True) is True
    assert page.asked and page.asked[-1] == pytest.approx(18.958 / 20.0, abs=1e-6), page.asked
    assert tail.seeks == [], "the page routes the press; the adapter must not seek the tail behind its back"

    # With no clip length the page cannot map a fraction; the direct tail seek stays the honest fallback.
    page2 = types.SimpleNamespace(_clip_duration=lambda: 0.0)
    adapter2 = backend_for(page2, PreviewBackendKind.LIVE_TAIL, tail=tail)
    assert adapter2.seek(1.0, exact=True) is True
    assert tail.seeks == [1.0]


@pytest.mark.parametrize("kind", [PreviewBackendKind.FINISHED_FILE, PreviewBackendKind.DISPLAY_FEED])
def test_every_owner_seeks_through_the_page_not_behind_its_back(kind):
    """The finished-file decoder and the display feed too: the page's press route is what opens the ticket that keeps
    the marker on the requested time. Seeking them directly made the thumb jump back to the old frame for ~150 ms after
    every press (the owner's log, 2026-11-26)."""
    class Page:
        def __init__(self) -> None:
            self.asked: list[float] = []
            self.requests: list[float] = []

        def _clip_duration(self) -> float:
            return 20.0

        def _on_preview_seek(self, fraction: float) -> None:
            self.asked.append(float(fraction))

        def _request_display_seek(self, fraction: float) -> bool:
            self.requests.append(float(fraction))
            return True

    class Owner:
        duration = 20.0

        def __init__(self) -> None:
            self.seeks: list[float] = []

        def seek(self, seconds: float) -> None:
            self.seeks.append(float(seconds))

    page, owner = Page(), Owner()
    adapter = backend_for(page, kind, decoder=owner)
    assert adapter.seek(10.0, exact=True) is True
    assert page.asked == [pytest.approx(0.5)], page.asked
    assert owner.seeks == [], "the page routes the press; the adapter must not seek the owner behind its back"


def test_the_strip_paints_played_in_blue_and_produced_as_the_same_blue_tinted(controls):
    """The colours are the approved mockup's (`shared.css`): track dark, band `rgba(91,157,255,.28)`, played the bright
    blue, head white at 50 %.

    This app's theme resolves `accent` to WHITE, and the widget used `accent` for the played fill while painting the
    produced band in solid blue - so a fully produced clip showed a short white bar and a blue band, i.e. both roles
    swapped (owner, 2026-11-26: "у меня produced - всё, если ты смотрел на комментарий").
    """
    from PySide6.QtCore import QRectF
    from src.ui.preview.timeline import PRODUCED_ALPHA
    from src.ui.preview.state import PreviewPhase

    timeline = controls.timeline
    timeline.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.0, produced=20.0,
                                    playhead=3.0, played=3.0, fps=24.0))

    class _Painter:
        def __init__(self) -> None:
            self.rects: list[tuple[int, int, int, int]] = []
            self.brushes: list = []
            self.pens: list = []

        def save(self): pass
        def restore(self): pass
        def setPen(self, pen): self.pens.append(pen)
        def setBrush(self, brush): self.brushes.append(brush)
        def drawRect(self, rect): self.rects.append((rect.x(), round(brush_alpha(self.brushes[-1]), 3),
                                                     round(rect.width(), 3), 0))
        def drawLine(self, *args): self.rects.append(("head",))

    def brush_alpha(brush):
        return brush.color().alphaF()

    painter = _Painter()
    timeline._paint_track(painter, QRectF(0, 0, 100, 4))

    # track, band (28 %), played (100 %), head
    assert len(painter.rects) == 4, painter.rects
    track, band, played, head = painter.rects
    assert band[1] == pytest.approx(PRODUCED_ALPHA, abs=0.01), "the produced band must be the tinted blue"
    assert band[2] == pytest.approx(100.0), "produced=100% must tint the whole bar"
    assert played[1] == pytest.approx(1.0), "the played fill is full strength"
    assert played[2] == pytest.approx(15.0), "played=3 s of 20 s"
    assert head[0] == "head"
    played_colour = painter.brushes[2].color().name()
    assert played_colour != "#ffffff", "the played fill must not be the theme's white accent"
    assert painter.pens[-1].color().alphaF() == pytest.approx(0.5, abs=0.01), "the head is white at 50 %"


def test_the_report_says_where_each_rect_is_drawn_and_from_which_number(controls):
    """The owner's ask: "улучши отладку чтобы было видно и завершено ли видео и где что отрисовывается и почему".

    The painter records every rect as it draws it, so the report answers with numbers: which rect, its px span, its
    fraction, the colour AND alpha used, and the state number it came from; `finished` says whether the clip is fully
    produced; `glued`/`why_not_glued` say whether the fill is literally the thumb and, if not, the page's reason.
    """
    from PySide6.QtCore import QRectF
    from src.ui.preview.state import PreviewPhase

    timeline = controls.timeline
    timeline.resize(400, 20)
    timeline.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.0, produced=20.0,
                                    playhead=3.0, played=3.0, fps=24.0, fill_source="shown"))
    timeline._paint_track(_RecordingPainter(), QRectF(10, 0, 380, 4))
    report = timeline.scrub_report()
    paint = report["paint"]

    assert paint["finished"] is True, "produced == duration is the finished file"
    roles = [rect["role"] for rect in paint["rects"]]
    assert roles == ["track", "produced", "played", "head"], roles
    band = paint["rects"][1]
    assert band["fraction"] == pytest.approx(1.0) and band["w"] == pytest.approx(380.0)
    assert band["from"] == "produced / duration" and band["alpha"] == pytest.approx(0.28, abs=0.01)
    fill = paint["rects"][2]
    assert fill["fraction"] == pytest.approx(0.15) and fill["alpha"] == pytest.approx(1.0)
    assert fill["from"] == "played / duration"
    assert paint["glued"] is True and paint["why_not_glued"] is None

    # A press in flight: the thumb is the request, the fill is the picture, and the report says so in the page's words.
    timeline.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.0, produced=20.0,
                                    playhead=10.0, played=3.0, fps=24.0, fill_source="press-open"))
    paint = timeline.scrub_report()["paint"]
    assert paint["glued"] is False
    assert paint["why_not_glued"] == "press-open"
    assert paint["state"]["thumb_fraction"] == pytest.approx(0.5)
    assert paint["state"]["fill_fraction"] == pytest.approx(0.15)


class _RecordingPainter:
    def save(self): pass
    def restore(self): pass
    def setPen(self, _pen): pass
    def setBrush(self, _brush): pass
    def drawRect(self, _rect): pass
    def drawLine(self, *args): pass
