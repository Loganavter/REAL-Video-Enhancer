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
* the transport carries no run readout of its own: no frame counter, no ETA/FPS/status line - the video-info pane's
  run card is the one surface for those numbers (`f <source done> / <source total>` is `PreviewState.frame_ratio`, and
  the card prints the same source-packet pair).

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
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px  # noqa: E402
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
    surface port. A spec emits ONE verb (`checkable` -> on check, else on release); the old hold/release pair was the
    same button implementation of that gesture and is deleted (audit 2026-11-27 §6.6).
    """
    covered = ({spec.command for spec in ACTIONS}
               | {PreviewCommand.SEEK_TO, PreviewCommand.SEEK_FRACTION,
                  PreviewCommand.COMPARE_DOWN, PreviewCommand.COMPARE_UP})
    # `TOGGLE_FULLSCREEN` is WINDOW-owned now: the preview bar carries no control for it (owner, 2026-11-26: "саму кнопку
    # фуллскрина лучше потом убрать ибо она делает глупую работу по управлению окном"), and the window's own entry point
    # (`--fullscreen`, the WM) is outside this catalog. Named here so the gap stays a decision, not an oversight.
    window_owned = {PreviewCommand.TOGGLE_FULLSCREEN}
    assert covered | window_owned == set(PreviewCommand)


def test_the_bars_are_split_by_function():
    top = {spec.id for spec in actions_for("top")}
    bottom = {spec.id for spec in actions_for("bottom")}
    assert "play" in bottom and "step_back" in bottom and "step_forward" in bottom
    assert "play" not in top and "compare" not in top
    assert {"mode_single", "mode_sbs", "mode_split", "zoom_fit"} <= top


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
    """The zoom row leaves as ONE capsule when the width cannot hold it; the modes leave last.

    The width is MEASURED, not pinned: this threshold has moved twice (the produced pill removed ~150 design px, then the
    fullscreen button left the bar), so a magic number in the test would be a third place to update. What must hold is the
    ORDER and the unit: zoom collapses before the modes, and a group goes as a whole capsule, never half a row.
    """
    collapsed_at = None
    for width in range(400, 120, -10):
        pane.resize(width, 400)
        _app().processEvents()
        if controls._group_panels["zoom"].isHidden():
            collapsed_at = width
            break
        assert controls._group_panels["view_mode"].isVisible(), "the modes must survive every width the zoom survives"
    assert collapsed_at is not None, "the zoom row never collapses: the overflow is dead"
    assert controls.overflow_button.isVisible(), "a collapsed control needs its affordance"
    assert "zoom_in" in controls.hidden_tools() and "zoom_fit" in controls.hidden_tools(), (
        "the whole capsule moves, not a button at a time")
    assert controls._group_panels["view_mode"].isVisible(), "the modes leave LAST"


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


# --- the owner's 2026-11-26 pass: pill removed, checked ring, nested corners -------------------------

def test_the_produced_pill_is_gone_from_the_overlay(controls):
    """The produced number stays in the transport/strip; the overlay keeps no readout and no spare surface.

    A hidden widget still owns its stretch in the row, so the check is structural: the only SurfacePanels under the top
    bar are the two group capsules, and nothing in it paints the word `produced`.
    """
    from sli_ui_toolkit.ui.widgets.atomic.surface_panel import SurfacePanel
    from sli_ui_toolkit.widgets import Text as ToolkitText

    top = controls.top_bar
    assert not hasattr(controls, "produced_text")
    assert not hasattr(controls, "pill")
    assert not hasattr(controls, "pill_dot")
    panels = [child for child in top.children() if isinstance(child, SurfacePanel)]
    assert set(panels) == set(controls._group_panels.values()), "a pill panel survived in the row"
    texts = [widget.text() for widget in top.findChildren(ToolkitText)]
    assert all("produced" not in text for text in texts), texts
    # `set_state` has no produced write left: the same screen with a produced number adds no text either.
    controls.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.0, produced=12.4, playhead=4.0))
    assert all("produced" not in widget.text() for widget in top.findChildren(ToolkitText))


def test_the_ring_is_suppressed_while_the_pane_has_nothing_to_show(controls):
    """Owner, 2026-11-26: "убирать обводку на выбранной кнопки когда и показывать в превью ещё нечего".

    A checked ring on a mode that cannot act (no picture, so no mode) is a promise. The gate is ONE property on the
    button, written by `set_state` from `state.has_picture()` and read by the layer at paint time.
    """
    from src.ui.preview.controls import CheckedOutlineLayer
    from types import SimpleNamespace

    mode = controls.button("mode_single")
    ring = next(layer for layer in mode._painter.layers if isinstance(layer, CheckedOutlineLayer))

    assert controls.set_state(PreviewState()) is None
    assert mode.property("outlineAllowed") is False
    assert ring.applies(SimpleNamespace(states=frozenset(mode.region_states("_main")), widget=mode)) is False, (
        "an empty pane must not claim a mode")

    controls.set_state(PreviewState(phase=PreviewPhase.READY, mode="single", duration=10.0, produced=5.0, playhead=1.0,
                                    source_frames=100, output_frames=100, picture=True))
    assert mode.property("outlineAllowed") is True
    assert ring.applies(SimpleNamespace(states=frozenset(mode.region_states("_main")), widget=mode)) is True


def test_the_checked_mode_wears_the_accent_ring_and_the_ring_follows_the_check(controls):
    """The ring is the toolkit's own layer, gated on the toolkit's `CHECKED` state - not a page paint.

    `Button.setChecked` already writes `ButtonState.CHECKED`; the layer reads the same state set at paint time, so the
    ring appears on exactly the current mode and leaves the previous one when `set_state` moves the mode.
    """
    from src.ui.preview.controls import CheckedOutlineLayer
    from sli_ui_toolkit.ui.widgets.buttons.state import ButtonState

    mode_ids = ("mode_single", "mode_sbs", "mode_split")
    rings = {}
    for spec_id in mode_ids:
        button = controls.button(spec_id)
        found = [layer for layer in button._painter.layers if isinstance(layer, CheckedOutlineLayer)]
        assert len(found) == 1, f"{spec_id} has no checked ring layer"
        assert found[0].scope == "widget", "the ring must paint over the region fills"
        rings[spec_id] = found[0]

    def ring_applies(spec_id: str) -> bool:
        button = controls.button(spec_id)
        return rings[spec_id].applies(types.SimpleNamespace(states=frozenset(button.region_states("_main"))))

    assert controls.button("mode_single").isChecked()
    assert ring_applies("mode_single") is True
    assert ring_applies("mode_sbs") is False and ring_applies("mode_split") is False

    controls.set_state(PreviewState(phase=PreviewPhase.READY, mode="sbs", duration=10.0, produced=5.0, playhead=1.0))
    assert controls.button("mode_sbs").isChecked()
    assert ring_applies("mode_sbs") is True
    assert ring_applies("mode_single") is False, "the previous mode must lose the ring"

    controls.set_state(PreviewState(phase=PreviewPhase.READY, mode="split", duration=10.0, produced=5.0, playhead=1.0))
    assert ring_applies("mode_split") is True
    assert ring_applies("mode_sbs") is False


def test_the_group_buttons_nest_their_corners_into_the_plate(controls):
    """Owner, 2026-11-28: one identical outline for the three modes, and a taller/wider plate.

    The corner recipe belongs to the CAPSULE, so all three mode buttons carry the same four radii and no index is
    special-cased; the zoom row is the other recipe (outer corners only). The plate's pad is `CAPSULE_PAD` - one design
    point more than the gutter - while the buttons keep their distance and their size.
    """
    from src.ui.preview.controls import CAPSULE_PAD, GROUP_INSET, GROUP_RADIUS, NESTED_RADIUS

    plate = controls._group_panels["view_mode"]
    pad = scaled_px(CAPSULE_PAD)
    gutter = scaled_px(GROUP_INSET)
    assert plate.radius() == scaled_px(GROUP_RADIUS)
    row = plate.layout()
    margins = row.contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (pad, pad, pad, pad)
    assert row.spacing() == gutter, "the pad grew, the gutter between the buttons did not"

    # One recipe, four equal radii per button, the same value on every mode.
    assert CAPSULE_PAD == GROUP_INSET + 1
    mode_ids = ("mode_single", "mode_sbs", "mode_split")
    radii = [controls.button(spec_id)._design_corner_radii for spec_id in mode_ids]
    assert radii == [(NESTED_RADIUS,) * 4] * 3
    outer = scaled_px(GROUP_RADIUS) - scaled_px(GROUP_INSET)
    assert controls.button("mode_single")._corner_radii_px == (outer,) * 4

    # The nested radius is design px, so the toolkit rescales it with the interface: the design value is the
    # difference of the two plate numbers, never a preset that only fits at 1.0.
    assert NESTED_RADIUS + GROUP_INSET == GROUP_RADIUS


def test_the_zoom_group_keeps_its_controls_order_and_geometry(controls):
    assert controls._group_specs["view_mode"] == ("mode_single", "mode_sbs", "mode_split")
    assert controls._group_specs["zoom"] == ("zoom_out", "zoom_fit", "zoom_one", "zoom_in")
    plate = controls._group_panels["zoom"]
    buttons = [controls.button(spec_id) for spec_id in controls._group_specs["zoom"]]
    assert all(button.parent() is plate for button in buttons)
    assert [button.height() for button in buttons] == [scaled_px(30)] * 4
    assert buttons[0].width() == scaled_px(28) and buttons[-1].width() == scaled_px(28)
    # The fullscreen BUTTON is gone (owner, 2026-11-26: "саму кнопку фуллскрина лучше потом убрать ибо она делает
    # глупую работу по управлению окном"): the window owns that, and the bar keeps only pane controls.
    assert "fullscreen" not in controls._buttons

    row = controls.top_bar._row
    in_row = [row.itemAt(index).widget() for index in range(row.count()) if row.itemAt(index).widget() is not None]
    assert in_row.index(controls._group_panels["view_mode"]) < in_row.index(plate)
    assert in_row.index(plate) < in_row.index(controls.overflow_button)
    bottom = [controls.bottom_bar._row.itemAt(index).widget() for index in range(controls.bottom_bar._row.count())]
    assert bottom[:3] == [controls.button("step_back"), controls.button("play"), controls.button("step_forward")]


def test_a_mode_toggle_emits_set_mode_and_keeps_one_checked(controls):
    emitted: list[tuple[object, object]] = []
    controls.commandRequested.connect(lambda command, value: emitted.append((command, value)))
    sbs = controls.button("mode_sbs")
    sbs.setChecked(True)
    sbs.toggled.emit(True)
    assert emitted[-1] == (PreviewCommand.SET_MODE, "sbs")
    assert not controls.button("mode_single").isChecked()


def test_the_transport_carries_no_frame_counter(controls):
    """The run card owns the frame number; the transport row must not repeat it (owner, 2026-11-26).

    A hidden widget still owns its stretch in the row, so the check is structural: no `frame_text` attribute, no
    toolkit `Text` in the transport other than the content time, and the row is still exactly step-back / play /
    step-forward / time / scrubber.
    """
    from sli_ui_toolkit.widgets import Text as ToolkitText

    assert not hasattr(controls, "frame_text")
    texts = [widget for widget in controls.bottom_bar.findChildren(ToolkitText)]
    assert texts == [controls.time_text], f"a second readout survived in the transport: {texts}"
    row = [controls.bottom_bar._row.itemAt(index).widget()
           for index in range(controls.bottom_bar._row.count())]
    assert row == [controls.button("step_back"), controls.button("play"), controls.button("step_forward"),
                   controls.time_text, controls.timeline]
    # `set_state` has nothing left to write there: the same state adds no text to the transport either.
    controls.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.0, source_frames=493,
                                    frames_done=250, output_frames=495, playhead=4.171))
    assert [widget for widget in controls.bottom_bar.findChildren(ToolkitText)] == [controls.time_text]


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


def test_installing_the_controls_retires_the_old_strip_and_hides_the_button():
    """The install makes the bar the only transport: the page's pause button stands down, and the pane's own painted
    strip does not exist to hide (it is deleted, audit 2026-11-27 §6.6) - only its named, inert entry point remains.
    """
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
    # The pane's own painted strip is DELETED (audit 2026-11-27 §6.6), so there is nothing to hide: the page's call
    # is the named, inert survivor, and the bar is the only strip. What must never come back is the second organ.
    assert not hasattr(pane, "_strip_visible") and not hasattr(pane, "seekRequested"), (
        "the pane paints no strip and offers no seek: the transport bar is the window's only one")
    pane.set_strip_visible(False)     # the page's call still answers, and changes nothing
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
    assert adapter.seek_to(18.958) is True
    assert page.asked and page.asked[-1] == pytest.approx(18.958 / 20.0, abs=1e-6), page.asked
    assert tail.seeks == [], "the page routes the press; the adapter must not seek the tail behind its back"

    # With no clip length the page cannot map a fraction; the direct tail seek stays the honest fallback.
    page2 = types.SimpleNamespace(_clip_duration=lambda: 0.0)
    adapter2 = backend_for(page2, PreviewBackendKind.LIVE_TAIL, tail=tail)
    assert adapter2.seek_to(1.0) is True
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
    assert adapter.seek_to(10.0) is True
    assert page.asked == [pytest.approx(0.5)], page.asked
    assert owner.seeks == [], "the page routes the press; the adapter must not seek the owner behind its back"


def test_the_strip_is_one_bar_blue_for_produced_grey_for_not(controls):
    """Blue = upscaled. Grey = not upscaled yet. The pointer does not decide either (owner, 2026-11-26).

    Two annotated screenshots settled it: with produced=100 % the bar must be one blue line to the end, and while the
    clip is still being upscaled the rest is grey - not a tint, not a second "played" colour whose boundary sits under
    the thumb.
    """
    from PySide6.QtCore import QRectF
    from src.ui.preview.state import PreviewPhase

    timeline = controls.timeline
    timeline.resize(400, 20)

    class _Painter:
        def __init__(self) -> None:
            self.brushes = []
            self.rects = []

        def save(self): pass
        def restore(self): pass
        def setPen(self, _pen): pass
        def setBrush(self, brush): self.brushes.append(brush)
        def drawRect(self, rect): self.rects.append((round(rect.x(), 1), round(rect.width(), 1)))
        def drawLine(self, *args): self.rects.append(("head",))

    # Half produced, playback in the middle: the band must stop at the produced edge regardless of the thumb.
    timeline.set_state(PreviewState(phase=PreviewPhase.PLAYING, duration=20.0, produced=10.0,
                                    playhead=5.0, played=5.0, fps=24.0))
    painter = _Painter()
    timeline._paint_track(painter, QRectF(0, 0, 100, 4))
    assert painter.rects[0] == (0.0, 100.0), "the whole groove is painted first (grey)"
    assert painter.rects[1] == (0.0, 50.0), "blue ends at 50 % produced, not at the thumb"
    assert painter.rects[2] == ("head",)
    colours = [brush.color() for brush in painter.brushes]
    r, g, b = colours[0].red(), colours[0].green(), colours[0].blue()
    assert max(r, g, b) - min(r, g, b) <= 8, f"not-yet-upscaled must be a grey, got {colours[0].name()}"
    r, g, b = colours[1].red(), colours[1].green(), colours[1].blue()
    assert b > r + 60 and b > g + 30, f"produced must be a blue, got {colours[1].name()}"
    assert painter.brushes[1].color().alphaF() == pytest.approx(1.0), "the bar is opaque, not a tint"

    # Finished: the blue covers the whole groove, and the pointer still changes nothing about it.
    timeline.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.0, produced=20.0,
                                    playhead=3.0, played=3.0, fps=24.0))
    painter = _Painter()
    timeline._paint_track(painter, QRectF(0, 0, 100, 4))
    assert painter.rects[1] == (0.0, 100.0), "produced=100 % paints the whole bar blue"
    report = timeline.scrub_report()["paint"]
    assert [rect["role"] for rect in report["rects"]] == ["track", "produced", "head"]
    assert report["state"]["fill_fraction"] == pytest.approx(0.15), "the played number survives in the report"


# --- P5: the queue item is its own clock, the dump feeds on the state, the line carries it -----------------

class _QueueDecoder:
    """The queue pane's decoder as the adapter reads it: its own length, its own seek, its own play state."""

    def __init__(self, duration: float = 12.0) -> None:
        self.duration = float(duration)
        self.seeks: list[float] = []
        self._playing = False

    def seek(self, seconds: float) -> None:
        self.seeks.append(float(seconds))

    def is_playing(self) -> bool:
        return self._playing


class _AliveProcess:
    """A render child that is still running (`poll()` is None until it exits)."""

    def poll(self):
        return None


def controls_parent():
    """A bare QWidget to stand in for the pane (the file's other page tests build one the same way)."""
    _app()
    return QWidget()


def _timeline_widget():
    _app()
    timeline = PreviewTimeline()
    timeline.resize(400, 24)
    return timeline


def test_a_queue_item_is_its_own_clock_not_the_host():
    """P3's note, closed in P5: a press on the queue tab binds the QUEUE's adapter, not the host one.

    The host adapter reads `_preview_playhead` - the RENDER's position behind the tab - so a press there was judged
    against a different file's timeline. The queue adapter reads the item's own position and its own length, and still
    routes the press through the page (which is what asks the queue decoder and records the route).
    """
    controller = PreviewController(FakeSurface())
    page = _fake_page(controls_parent(), controller)
    decoder = _QueueDecoder()
    page._queue_pane_owns = True
    page._queue_pane_decoder = decoder
    page._queue_pane_duration = 12.0
    page._queue_pane_position = 4.0
    asked: list[float] = []
    page._on_preview_seek = lambda fraction: asked.append(float(fraction))

    page._push_preview_state()

    assert controller.state().backend is PreviewBackendKind.QUEUE_FILE
    assert controller.backend().kind is PreviewBackendKind.QUEUE_FILE
    assert controller.state().duration == pytest.approx(12.0)
    assert controller.state().produced == pytest.approx(12.0), "a queue item's whole length exists"
    assert controller.backend().position == pytest.approx(4.0), "the item's playhead, not the render's 10.0 s"

    controller.dispatch(PreviewCommand.SEEK_TO, 3.0)

    assert asked and asked[-1] == pytest.approx(0.25), "the press reaches the page's queue route, in queue seconds"
    assert decoder.seeks == [], "the adapter must not seek the decoder behind the page's back"


@pytest.fixture()
def preview_lines(monkeypatch):
    """Capture `preview-log.txt` lines with a controllable clock (the aggregate is a per-second instrument)."""
    from src.ui.video import telemetry

    lines: list[tuple[str, str]] = []
    now = {"t": 1000.0}
    monkeypatch.setenv(telemetry.ENV_NAME, "1")
    previous_writer, previous_clock = telemetry._writer, telemetry._clock
    telemetry.set_writer(lambda line, which: lines.append((which, line)))
    telemetry.set_clock(lambda: now["t"])
    telemetry._reset(session=True)
    try:
        yield lines, now
    finally:
        telemetry.set_writer(previous_writer)
        telemetry.set_clock(previous_clock)
        telemetry._reset(session=True)


def _snapshot_lines(lines) -> list[str]:
    return [line for _which, line in lines if line.startswith("preview: snapshot ")]


def _page_for(controller):
    return _fake_page(controls_parent(), controller)


def test_the_aggregate_line_says_finished_for_a_queue_item(preview_lines):
    """One line a second, and a queue item's band IS its whole file: `finished=1`."""
    lines, now = preview_lines
    controller = PreviewController(FakeSurface())
    page = _page_for(controller)
    page._queue_pane_owns = True
    page._queue_pane_decoder = _QueueDecoder()
    page._queue_pane_duration = 12.0

    page._push_preview_state()          # starts the line's window
    now["t"] += 1.2
    page._push_preview_state()

    snapshot = _snapshot_lines(lines)
    assert len(snapshot) == 1, snapshot
    assert "kind=queue_file" in snapshot[0], snapshot[0]
    assert "duration=12.000" in snapshot[0] and "finished=1" in snapshot[0], snapshot[0]
    now["t"] += 0.4
    page._push_preview_state()
    assert len(_snapshot_lines(lines)) == 1, "one line per second, not one per tick"


def test_the_aggregate_line_says_not_finished_while_a_render_runs(preview_lines):
    """An export at 14 % is not a finished file: the line must carry the band's own `finished`, not a second rule."""
    lines, now = preview_lines
    controller = PreviewController(FakeSurface())
    page = _page_for(controller)
    page.currentFrame, page.max_value = 67, 493
    page._display = types.SimpleNamespace(attached=True)
    page.renderProcess = _AliveProcess()
    page.isPreview = False

    page._push_preview_state()
    now["t"] += 1.2
    page._push_preview_state()

    snapshot = _snapshot_lines(lines)
    assert len(snapshot) == 1, snapshot
    assert "kind=display_feed" in snapshot[0], snapshot[0]
    assert "finished=0" in snapshot[0], snapshot[0]
    assert "produced=" in snapshot[0] and "played=" in snapshot[0] and "playhead=" in snapshot[0], snapshot[0]


def test_the_page_feeds_the_dump_the_state_it_paints():
    """`state:preview` must answer the numbers the BARS paint: the dump reads the controller's state, not a copy."""
    controller = PreviewController(FakeSurface())
    page = _page_for(controller)
    page._preview_controls = types.SimpleNamespace(timeline=None)
    page._push_preview_state()

    dump = page._preview_session().dump()

    state = controller.state()
    assert dump["snapshot"]["duration"] == pytest.approx(state.duration)
    assert dump["snapshot"]["produced"] == pytest.approx(state.produced)
    assert dump["snapshot"]["played"] == pytest.approx(state.played)
    assert dump["snapshot"]["playhead"] == pytest.approx(state.playhead)
    assert dump["snapshot"]["finished"] is state.finished()
    assert dump["snapshot"]["kind"] == state.backend.value
    assert dump["owner"]["kind"] == state.backend.value
    assert set(dump["owner"]) == {"kind", "may_paint", "may_want", "may_seek_any", "has_clock", "has_surface"}


def test_the_dump_reads_the_scrubbers_paint_without_touching_anything():
    """A `state:preview` read must not move the session, the state or the control: it is a diagnosis, not an action."""
    controller = PreviewController(FakeSurface())
    page = _page_for(controller)
    page._preview_controls = types.SimpleNamespace(timeline=_timeline_widget())
    page._push_preview_state()
    session = page._preview_session()

    before_state = controller.state()
    before_recent = list(page._preview_controls.timeline._recent)
    budget = (controller, session.ticket)

    dump = session.dump()

    assert dump["paint"] is not None and "rects" in dump["paint"], dump["paint"]
    assert controller.state() == before_state, "the dump moved the state"
    assert page._preview_controls.timeline._recent == before_recent, "the dump touched the scrubber"
    assert (controller, session.ticket) == budget and session.ticket is None, "the dump opened or moved a ticket"


def test_the_widget_state_scrub_block_keeps_its_keys_and_types():
    """The per-widget answer is byte-compatible: the pane-wide dump must not replace or reshape its `scrub` block."""
    timeline = _timeline_widget()
    timeline.set_state(PreviewState(phase=PreviewPhase.READY, duration=20.0, playhead=3.0, played=2.0,
                                    produced=8.0, fps=24.0))
    report = timeline.scrub_report()

    assert set(report) == {"duration", "value", "thumb", "fill", "produced", "scrubbing", "asked",
                           "finished", "paint", "recent"}
    assert isinstance(report["duration"], float) and isinstance(report["value"], int)
    assert isinstance(report["scrubbing"], bool) and isinstance(report["finished"], bool)
    for key in ("thumb", "fill", "produced"):
        assert set(report[key]) == {"fraction", "seconds"}, report[key]
        assert isinstance(report[key]["fraction"], float) and isinstance(report[key]["seconds"], float)
    assert isinstance(report["recent"], list) and isinstance(report["paint"], dict)


