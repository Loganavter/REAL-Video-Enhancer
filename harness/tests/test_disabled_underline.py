"""A disabled line edit draws a GREY underline, not the white one an editable field carries.

Owner, 2026-11-24: the disabled output-path field kept a bright white bottom edge, so "this is not editable" was
invisible - it looked like any other input. The underline came from one shared token with one alpha
(`button.default.bottom.edge`, white, `alpha=60`) for every state, and the theme has no mid-grey key to reach for:
`input.text.disabled` / `dialog.text.disabled` are the plate's own colour (`#2c313c`), which draws nothing.

So the control derives the colour: the plate token (`surface.background`) nudged toward the text token
(`dialog.text`) by `DISABLED_UNDERLINE_MIX` - a neutral grey that still reads as the field's edge - and a host can
override it outright with `setDisabledUnderlineColor`.

Measured on the real plate (disabled output field, factor 1.0): the drawn line is `#6b6f76` where the unfocused enabled
field draws `#8f9298` from the shared token. The pin below states the contract, not the pixels: an alpha drop alone was
tried first and vanished into the plate (24/255 of white), which is why the colour is asserted here.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")


@pytest.fixture()
def captured(qapp, monkeypatch):
    """The app's palette installed, and the underline config each paint asks for captured without painting."""
    from src.ui import toolkit_boot, toolkit_theme
    from sli_ui_toolkit.ui.widgets.atomic import custom_line_edit as module

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)     # the derived grey is read OUT of the palette, so the palette has to be the app's

    seen = []
    original = module.draw_bottom_underline

    def spy(painter, rect, theme_manager, config=None):
        seen.append(config)
        return None  # the paint itself is not what this gate is about

    monkeypatch.setattr(module, "draw_bottom_underline", spy)
    monkeypatch.setattr(module.draw_bottom_underline, "__wrapped__", original, raising=False)
    return module, seen


def _field(module, qapp, *, enabled: bool):
    widget = module.CustomLineEdit()
    widget.resize(320, 30)
    widget.show()
    for _ in range(6):
        qapp.processEvents()
    widget.clearFocus()          # the unfocused path is the one a disabled field takes
    widget.setEnabled(enabled)
    for _ in range(6):
        qapp.processEvents()
    return widget


def test_a_disabled_field_asks_for_a_grey_line_not_the_shared_white(captured, qapp):
    module, seen = captured
    field = _field(module, qapp, enabled=False)
    try:
        config = seen[-1]
        assert config.color is not None, (
            "the disabled field fell back to the shared white token - the defect this gate was written for")
        assert config.color.name() != "#ffffff", "the disabled underline is still the white one"
        expected = field._disabled_underline_colour()
        assert config.color == expected, f"the disabled underline is {config.color.name()}, not the derived grey"
        assert config.alpha is None, "the derived grey is passed as a solid colour; an alpha would dim it to nothing"
    finally:
        field.close()


def test_an_enabled_field_keeps_the_shared_token(captured, qapp):
    """The cure must not change the editable look: that one still comes from the theme, at its own alpha."""
    module, seen = captured
    field = _field(module, qapp, enabled=True)
    try:
        config = seen[-1]
        assert config.color is None, "an enabled field now paints a hard-coded colour instead of the theme's"
        assert config.alpha == 60
    finally:
        field.close()


def test_a_host_can_state_the_disabled_colour(captured, qapp):
    """`setDisabledUnderlineColor` wins over the derived grey - the API the theme cannot cover."""
    from PySide6.QtGui import QColor

    module, seen = captured
    field = _field(module, qapp, enabled=False)
    try:
        field.setDisabledUnderlineColor(QColor("#ff6b6b"))
        field.update()
        for _ in range(6):
            qapp.processEvents()
        assert seen[-1].color == QColor("#ff6b6b")
        assert field.disabledUnderlineColor() == QColor("#ff6b6b")
        field.setDisabledUnderlineColor(None)
        field.update()
        for _ in range(6):
            qapp.processEvents()
        assert seen[-1].color == field._disabled_underline_colour()
    finally:
        field.close()


def test_the_derived_grey_sits_between_the_plate_and_the_text(captured, qapp):
    """The colour is derived from two tokens, so every channel lands strictly between them."""
    module, seen = captured
    field = _field(module, qapp, enabled=False)
    try:
        grey = field._disabled_underline_colour()
        plate = module.QColor(field.theme_manager.get_color("surface.background"))
        text = module.QColor(field.theme_manager.get_color("dialog.text"))
        for channel in ("red", "green", "blue"):
            low, high = sorted((getattr(plate, channel)(), getattr(text, channel)()))
            value = getattr(grey, channel)()
            assert low <= value <= high, f"{channel}: {value} is outside [{low}, {high}]"
        assert grey.name() != text.name(), "the mix collapsed onto the text colour - that is the white line again"
    finally:
        field.close()
