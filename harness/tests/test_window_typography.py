"""The built shell's type follows the toolkit's font path - labels AND controls, live.

`harness/tests/test_typography_scale.py` pins `src/ui/typography.py` on bare widgets. The user-visible claim is one
layer up: the fonts a real `MainWindowShell` builds its pages with are the toolkit-resolved ones, so a `UiScale` or
family change made while the window is open reaches the widgets the user reads, with no rebuild. This file pins that
end-to-end path on the real shell (offscreen, resource blob stubbed - the same shape `test_window_scale.py` uses).

The red state it describes is the pre-`af163a50` tree: `typography.apply` produced a plain `QFont`, a toolkit `Text`
then overrode it from its own variant, and a built page's labels stayed at one size at every factor while
`UiScale.set_factor(2.0)` reached the toolkit's own widgets and nothing in RVE. The identity half is just as important:
at 1.0 the resolved sizes must stay the Designer's - a 16 px body label and a 12 px control, not a uniform 12 px - so
routing type through the toolkit does not silently redraw the app.

Everything here is offscreen. Pixels are not asserted; a leased display is where the look is proven.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtGui import QFontDatabase, QFontInfo  # noqa: E402
from PySide6.QtWidgets import QMainWindow  # noqa: E402

from sli_ui_toolkit.ui.managers import UiFont, UiScale  # noqa: E402

from src.ui import typography  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

#: The DESIGN pixel size each scale group resolves to at 1.0. Read from `typography`, never restated as a number.
DISPLAY_PX = typography.label_px(typography.DISPLAY_PT)   # 25 pt -> 33 px
BODY_PX = typography.label_px(typography.BODY_PT)         # 12 pt -> 16 px
GROUP_TITLE_PX = typography.label_px(typography.GROUP_TITLE_PT)  # 15 pt -> 20 px
CAPTION_PX = typography.label_px(typography.CAPTION_PT)   # 10 pt -> 13 px
CONTROL_PX = typography.CONTROL_PX                        # 12 px, its own scale

#: One named widget per scale group, from the pages the shell builds. Object names are the Designer's and are what the
#: app and the live control channel address, so they are stable.
TEXT_GROUPS = {
    "label": DISPLAY_PX,             # home page display heading (25 pt)
    "label_4": DISPLAY_PX,           # download page heading (25 pt, underlined)
    "encoderLabel_4": BODY_PX,       # a body label (12 pt)
    "label_41": GROUP_TITLE_PX,      # a section heading (15 pt)
    "videoInfoRemaining": CAPTION_PX,  # the run card's caption line (10 pt)
}
CONTROL_NAME = "uninstallAppBtn"     # a toolkit Button: RVE's control scale (12 design px)


@pytest.fixture()
def factor_state():
    """Leave the process-wide factor where it was: the singleton outlives the test and so does its window."""
    from src.ui import i18n

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _shell(qapp, factor: float):
    """The real window at `factor`, built, shown and settled."""
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    i18n.configure_i18n("en")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    UiScale.get_instance().set_factor(factor)
    window = QMainWindow()
    builder = MainWindowShell()
    builder.build(window)
    window.show()
    _settle(qapp)
    return builder, window


def _settle(qapp, turns: int = 8) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _text(window, name: str):
    from sli_ui_toolkit.widgets import Text

    widget = window.findChild(Text, name)
    assert widget is not None, f"the shell did not build the {name!r} label"
    return widget


def _control(window, name: str):
    from sli_ui_toolkit.widgets import Button

    widget = window.findChild(Button, name)
    assert widget is not None, f"the shell did not build the {name!r} button"
    return widget


def _px(widget) -> int:
    """The pixel size Qt actually resolves for the widget's font - what the user sees."""
    return QFontInfo(widget.font()).pixelSize()


# --- the 1.0 identity: the two scales keep their designed gap ------------------------------------------------


def test_one_point_zero_keeps_every_scale_group_at_its_designed_size(qapp, factor_state):
    """Routing type through the toolkit must not redraw the 1.0 app: labels in pt, controls in px."""
    _, window = _shell(qapp, 1.0)
    try:
        for name, expected in TEXT_GROUPS.items():
            assert _px(_text(window, name)) == expected, (
                f"{name!r} resolves to {_px(_text(window, name))} px at 1.0, not the designed {expected} px")
        assert _px(_control(window, CONTROL_NAME)) == CONTROL_PX, (
            "the control group left its own 12 px scale")
        assert typography.CONTROL_RATIO == pytest.approx(CONTROL_PX / BODY_PX), (
            "the label-vs-control gap is no longer 16/12")
    finally:
        window.close()


# --- the live change: the user's scale reaches what is already built -----------------------------------------


def test_a_live_scale_change_reaches_the_built_window_and_comes_back(qapp, factor_state):
    """The acceptance: with the window open, factor 2.0 doubles each group's resolved pixels; 1.0 restores them.

    Red before the change: the page labels were plain `QFont`s the toolkit's `Text` overrode, so every value here
    stayed put at 2.0.
    """
    _, window = _shell(qapp, 1.0)
    try:
        before = {name: _px(_text(window, name)) for name in TEXT_GROUPS}
        control_before = _px(_control(window, CONTROL_NAME))
        assert before == TEXT_GROUPS and control_before == CONTROL_PX

        factor_state.set_factor(2.0)
        _settle(qapp)
        scaled = UiScale.get_instance()
        for name, design_px in TEXT_GROUPS.items():
            assert _px(_text(window, name)) == scaled.scaled_px(design_px), (
                f"{name!r} did not follow the live factor to 2.0")
        assert _px(_control(window, CONTROL_NAME)) == scaled.scaled_px(CONTROL_PX), (
            "the control stayed at 12 px when the window scaled to 2.0")

        factor_state.set_factor(1.0)
        _settle(qapp)
        assert {name: _px(_text(window, name)) for name in TEXT_GROUPS} == before, (
            "returning to 1.0 did not restore the designed sizes")
        assert _px(_control(window, CONTROL_NAME)) == CONTROL_PX
    finally:
        window.close()


# --- the family axis of the same path ------------------------------------------------------------------------


def test_a_family_change_reaches_the_built_window(qapp, factor_state):
    """A face change is the other half of `UiFont`: an already-built label and control must repaint in the new face."""
    ui_font = UiFont.get_instance()
    previous = ui_font.family()
    candidates = [name for name in QFontDatabase.families() if name and name != previous]
    if not candidates:
        pytest.skip("this interpreter registers only one font family")

    _, window = _shell(qapp, 1.0)
    try:
        new_family = candidates[0]
        ui_font.set_family(new_family)
        _settle(qapp)

        assert _text(window, "encoderLabel_4").font().family() == new_family, (
            "a built body label kept the old face after UiFont.set_family")
        assert _control(window, CONTROL_NAME).font().family() == new_family, (
            "a built control kept the old face after UiFont.set_family")
    finally:
        ui_font.set_family(previous)
        _settle(qapp)
        window.close()
