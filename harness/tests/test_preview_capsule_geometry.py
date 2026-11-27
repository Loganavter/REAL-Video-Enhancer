"""The preview's group capsules: one identical outline for the three modes, and both plates one design point bigger.

Owner, 2026-11-28: "нужно в превью у выбора сравнения кнопок 3х сделать идентичные скругленную обводку, а не каждую
по отдельности настраивать + сделать обе подложки (справа тоже группа контролов) на 1 пункт больше по размеру".

What is pinned here, offscreen and on the real `PreviewControls` (no window, no GPU):

* the corner recipe belongs to the CAPSULE, not to a button's index: every mode button carries the same four equal
  radii, and the three siblings carry the same tuple as each other. The zoom row keeps the nested recipe - written as
  an explicit recipe, so the difference is a decision and not a leftover loop branch;
* only the checked mode wears the accent ring, and an empty pane claims no mode at all;
* both plates are exactly `2 * scaled_px(1)` wider and taller than with the old pad, while the buttons keep the size
  their specs ask for. The expectation is COMPOSED from the plate's own buttons and pads, so growing a button cannot
  silently satisfy it;
* the growth is resolved at call time: under a scale factor of 2.0 the plates grow by `2 * scaled_px(1)` = 4 device px
  and the radii double with the interface.

Red-before (2026-11-28): the old `_add_group` round the first/last buttons only, i.e. the mode tuples came out
`(8, 0, 0, 8)` / `(0, 0, 0, 0)` / `(0, 8, 8, 0)` - the first test fails on the very first button - and the plate used
`GROUP_INSET` as its pad, so its width and the bar height came out `2 * scaled_px(1)` short in the third and fourth
tests. Both were put back into `controls.py`, watched red, then restored.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

# The app's optional runtime imports, stubbed only when the interpreter lacks them (the pattern of
# `test_preview_controls.py`): the controls build toolkit buttons, which touch neither, but the module graph does.
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

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager  # noqa: E402
from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px  # noqa: E402

from src.ui.preview.controls import (  # noqa: E402
    CAPSULE_PAD,
    GROUP_INSET,
    GROUP_RADIUS,
    NESTED_RADIUS,
    TOP_HEIGHT,
    CheckedOutlineLayer,
    PreviewControls,
)
from src.ui.preview.state import PreviewPhase, PreviewState  # noqa: E402

MODE_IDS = ("mode_single", "mode_sbs", "mode_split")
ZOOM_IDS = ("zoom_out", "zoom_fit", "zoom_one", "zoom_in")
PANE_SIZE = (900, 721)


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


def _build(state: PreviewState | None = None, *, size: tuple[int, int] = PANE_SIZE) -> PreviewControls:
    """A pane with real controls on it, at the CURRENT ui scale (so a scale test builds after setting the factor)."""
    _app()
    pane = QWidget()
    pane.resize(*size)
    pane.show()          # a hidden widget receives no resize event, and the bars follow the pane's size
    controls = PreviewControls(pane, pane=pane)
    controls.set_state(state or PreviewState(phase=PreviewPhase.READY, duration=20.0, produced=10.0,
                                             playhead=4.0))
    return controls


@pytest.fixture()
def controls() -> PreviewControls:
    return _build()


@pytest.fixture()
def double_scale():
    """Run under a 2.0 interface factor, as the settings page may; the process-wide singleton is restored after."""
    scale = UiScale.get_instance()
    previous = scale.factor()
    scale.set_factor(2.0)
    try:
        yield scale
    finally:
        scale.set_factor(previous)


# --- one identical outlined pill per mode ------------------------------------------------------------

def test_every_mode_button_wears_the_same_four_corner_radii(controls):
    """One recipe for the group: no button is the "first" or the "last" any more.

    Read from what the toolkit actually built: `_design_corner_radii` is the tuple the group handed to `Button`, and
    `_corner_radii_px` is the one the painter uses. All four corners of all three buttons must be the same value.
    """
    assert len({tuple(controls.button(spec_id)._design_corner_radii) for spec_id in MODE_IDS}) == 1, (
        "the three modes did not get one recipe")
    for spec_id in MODE_IDS:
        button = controls.button(spec_id)
        design = button._design_corner_radii
        assert design is not None, f"{spec_id} kept the toolkit's own radius instead of the group's"
        assert len(set(design)) == 1, f"{spec_id} is not one shape: {design}"
        assert design == (NESTED_RADIUS,) * 4
        # Design px, so the toolkit resolves it at build time; the painter sees the scaled tuple.
        assert button._corner_radii_px == (scaled_px(NESTED_RADIUS),) * 4, spec_id
        assert len(set(button._corner_radii_px)) == 1, spec_id


def test_the_zoom_row_keeps_the_nested_recipe_as_an_explicit_parameter(controls):
    """The zoom row is deliberately NOT uniform: its outer corners make it one capsule. Explicit, not a loop branch."""
    outer = NESTED_RADIUS
    first = controls.button(ZOOM_IDS[0])
    last = controls.button(ZOOM_IDS[-1])
    assert first._design_corner_radii == (outer, 0, 0, outer)
    assert last._design_corner_radii == (0, outer, outer, 0)
    for spec_id in ZOOM_IDS[1:-1]:
        assert controls.button(spec_id)._design_corner_radii == (0, 0, 0, 0), spec_id


def test_only_the_checked_mode_wears_the_accent_ring(controls):
    """The ring is gated on the toolkit's CHECKED state through the painter's own layer, as before this pass."""
    rings = {}
    for spec_id in MODE_IDS:
        found = [layer for layer in controls.button(spec_id)._painter.layers
                 if isinstance(layer, CheckedOutlineLayer)]
        assert len(found) == 1, f"{spec_id} has no checked ring layer"
        rings[spec_id] = found[0]

    def wearing() -> list[str]:
        worn = []
        for spec_id in MODE_IDS:
            button = controls.button(spec_id)
            ctx = types.SimpleNamespace(states=frozenset(button.region_states("_main")), widget=button)
            if rings[spec_id].applies(ctx):
                worn.append(spec_id)
        return worn

    controls.set_state(PreviewState(phase=PreviewPhase.READY, mode="split", duration=10.0, produced=5.0,
                                    playhead=1.0, picture=True))
    assert controls.button("mode_split").isChecked()
    assert wearing() == ["mode_split"]

    controls.set_state(PreviewState(phase=PreviewPhase.READY, mode="sbs", duration=10.0, produced=5.0,
                                    playhead=1.0, picture=True))
    assert wearing() == ["mode_sbs"], "the previous mode must lose the ring"

    # An empty pane claims no mode: the ring disappears even though a mode button is still checked.
    controls.set_state(PreviewState())
    assert wearing() == [], "an empty pane must not claim a mode"


# --- both plates, one design point bigger -------------------------------------------------------------

def _composed_width(controls: PreviewControls, name: str, pad: int) -> int:
    """The plate's width composed from its own buttons: widths + the gutters between them + the pad on both sides."""
    panel = controls._group_panels[name]
    buttons = [controls.button(spec_id) for spec_id in controls._group_specs[name]]
    gutter = panel.layout().spacing()
    return sum(button.width() for button in buttons) + gutter * (len(buttons) - 1) + 2 * pad


def _assert_buttons_keep_their_size(controls: PreviewControls, name: str) -> None:
    for spec_id in controls._group_specs[name]:
        spec = controls._specs[spec_id]
        button = controls.button(spec_id)
        assert button.height() == scaled_px(spec.size[1]), f"{spec_id} changed height"
        if spec.size[0]:
            assert button.width() == scaled_px(spec.size[0]), f"{spec_id} changed width"


def test_both_plates_grew_by_exactly_one_design_point_per_side(controls):
    """`CAPSULE_PAD` is one design point more than the gutter, so every plate is `2 * scaled_px(1)` bigger."""
    old_pad = scaled_px(GROUP_INSET)
    new_pad = scaled_px(CAPSULE_PAD)
    assert new_pad - old_pad == scaled_px(1), "the pad must grow by one design point, resolved through scaled_px()"
    grown = 2 * scaled_px(1)

    # The pad is the plate's own padding; the gutter between the buttons did not move.
    for name in ("view_mode", "zoom"):
        panel = controls._group_panels[name]
        margins = panel.layout().contentsMargins()
        assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == ((new_pad,) * 4), name
        assert panel.layout().spacing() == old_pad, name
        assert panel.radius() == scaled_px(GROUP_RADIUS), name
        assert panel.width() == _composed_width(controls, name, new_pad), name
        assert _composed_width(controls, name, new_pad) - _composed_width(controls, name, old_pad) == grown, name
        _assert_buttons_keep_their_size(controls, name)

    # The bar grew by the same point on each side so the plate is actually VISIBLE two points taller: the plate is
    # stretched to the bar's height, and the bar's height is the one that holds it.
    old_top = scaled_px(TOP_HEIGHT - 2 * (CAPSULE_PAD - GROUP_INSET))
    assert scaled_px(TOP_HEIGHT) - old_top == grown
    for name in ("view_mode", "zoom"):
        panel = controls._group_panels[name]
        assert panel.height() == scaled_px(TOP_HEIGHT), name
        assert panel.height() - old_top == grown, name
        assert panel.height() > max(controls.button(spec_id).height()
                                    for spec_id in controls._group_specs[name]), name


def test_the_growth_is_resolved_at_call_time(double_scale):
    """At a 2.0 interface factor the same one design point is 4 device px: the pads and radii are read, never baked."""
    controls = _build(size=(1400, 1000))     # the row doubles too, so give the bar room to hold both capsules
    old_pad = scaled_px(GROUP_INSET)
    new_pad = scaled_px(CAPSULE_PAD)
    assert scaled_px(1) == 2, "a 2.0 factor makes one design point two device px"
    grown = 2 * scaled_px(1)

    for name in ("view_mode", "zoom"):
        panel = controls._group_panels[name]
        assert panel.isVisible(), f"the {name} capsule collapsed (the pane is too narrow for this test)"
        assert panel.width() == _composed_width(controls, name, new_pad), name
        assert _composed_width(controls, name, new_pad) - _composed_width(controls, name, old_pad) == grown, name
        assert panel.layout().contentsMargins().left() == new_pad, name
        # Every member of a capsule keeps the SAME size at this scale too: the growth above is the pad's, never a
        # button's (a button's own size is the catalog's, and the toolkit resolves it at build time).
        heights = {controls.button(spec_id).height() for spec_id in controls._group_specs[name]}
        assert len(heights) == 1, f"{name} stretched one member and not its siblings: {heights}"

    old_top = scaled_px(TOP_HEIGHT - 2 * (CAPSULE_PAD - GROUP_INSET))
    for name in ("view_mode", "zoom"):
        assert controls._group_panels[name].height() - old_top == grown, name

    # The outline scales with the interface: the design tuple is unchanged, the painted one doubles.
    for spec_id in MODE_IDS:
        button = controls.button(spec_id)
        assert button._design_corner_radii == (NESTED_RADIUS,) * 4
        assert button._corner_radii_px == (scaled_px(NESTED_RADIUS),) * 4
        assert scaled_px(NESTED_RADIUS) == 2 * NESTED_RADIUS, "the radius did not scale with the factor"
