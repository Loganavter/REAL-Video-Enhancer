"""A disabled line edit draws its OWN grey line, never the button family's white one.

History, because the file has changed sides once already. Owner, 2026-11-24: the disabled output-path field kept a
bright white bottom edge, so "not editable" was invisible. Then the underline had one shared token at one alpha
(`button.default.bottom.edge`, white, 60) for every state, and no mid-grey key existed to reach for.

Since then the toolkit lane gave the field its own token family (`input.underline` / `.focused` / `.disabled`, commit
`d9c154b`) and the RVE bridge stayed as the no-token fallback. So the contract this file pins is the PRECEDENCE, not
one implementation:

1. an explicit `setDisabledUnderlineColor` wins;
2. else the family's `input.underline.disabled` token;
3. else the derived grey (`_disabled_underline_colour`, the bridge).

Whatever the source, a disabled field may never paint the shared white line, and an enabled field must keep the theme's
own answer (the token, or - on a palette that has none - the legacy `None` + alpha 60).

The spy takes `**kwargs` on purpose: the toolkit passes `token_prefix=` to the painter now, and a spy narrower than the
real signature raises *inside* `paintEvent`, which pytest-qt then reports as a crash rather than a failure.
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
    toolkit_theme.install(qapp)     # the line's colours are read OUT of the palette, so it has to be the app's

    seen = []

    def spy(painter, rect, theme_manager, config=None, **kwargs):
        seen.append(config)
        return None  # the paint itself is not what this gate is about

    monkeypatch.setattr(module, "draw_bottom_underline", spy)
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


def _family_token(module, field):
    """The field's own disabled token, when the toolkit version has the family (it is the preferred source)."""
    from PySide6.QtGui import QColor

    name = getattr(module.CustomLineEdit, "UNDERLINE_DISABLED_TOKEN", None)
    if name is None:
        return None
    colour = field.theme_manager.try_get_color(name)
    return None if colour is None else QColor(colour)


def _expected_disabled(field):
    """The precedence the control promises: override, then the family token, then the derived grey."""
    from PySide6.QtGui import QColor

    override = field.disabledUnderlineColor()
    if override is not None:
        return QColor(override)
    token = _family_token(sys.modules[field.__class__.__module__], field)
    if token is not None:
        return token
    return field._disabled_underline_colour()


def test_a_disabled_field_never_paints_the_shared_white_line(captured, qapp):
    module, seen = captured
    field = _field(module, qapp, enabled=False)
    try:
        config = seen[-1]
        assert config.color is not None, (
            "the disabled field fell back to the shared white token - the defect this gate was written for")
        assert config.color.name() != "#ffffff", "the disabled underline is still the white one"
        assert config.color == _expected_disabled(field), (
            f"the disabled underline is {config.color.name()}, not the family token / derived grey")
        assert config.alpha is None, "the disabled colour is a solid one; an alpha would dim it to nothing"
    finally:
        field.close()


def test_an_enabled_field_keeps_the_theme_answer(captured, qapp):
    """No hard-coded colour for an editable field: the family token when the palette has one, else the legacy pair."""
    from PySide6.QtGui import QColor

    module, seen = captured
    field = _field(module, qapp, enabled=True)
    try:
        config = seen[-1]
        token_name = getattr(module.CustomLineEdit, "UNDERLINE_TOKEN", None)
        token = field.theme_manager.try_get_color(token_name) if token_name else None
        if token is None:
            assert config.color is None, "an enabled field invented a colour where the palette has no token"
            assert config.alpha == 60
        else:
            assert config.color == QColor(token)
            assert config.alpha is None
    finally:
        field.close()


def test_a_host_can_state_the_disabled_colour(captured, qapp):
    """`setDisabledUnderlineColor` outranks both the family token and the derived grey."""
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
        assert seen[-1].color == _expected_disabled(field)
    finally:
        field.close()


def test_the_derived_grey_sits_between_the_plate_and_the_text(captured, qapp):
    """The fallback (for a palette that predates `input.underline.disabled`) mixes the INPUT family's own two ends.

    Since the toolkit's input-family migration the mix is `input.text.disabled` into `input.background`, not
    `dialog.text` over `surface.background` (toolkit CHANGELOG, "The input family paints its own state tokens"), so the
    bounds this test names are the family's - a test that still named the dialog sources would pass by coincidence and
    stop protecting the contract.
    """
    module, seen = captured
    field = _field(module, qapp, enabled=False)
    try:
        grey = field._disabled_underline_colour()
        from PySide6.QtGui import QColor

        plate = QColor(field.theme_manager.get_color("input.background"))
        text = QColor(field.theme_manager.get_color("input.text.disabled"))
        for channel in ("red", "green", "blue"):
            low, high = sorted((getattr(plate, channel)(), getattr(text, channel)()))
            value = getattr(grey, channel)()
            assert low <= value <= high, f"{channel}: {value} is outside [{low}, {high}]"
        assert grey.name() != text.name(), "the mix collapsed onto the text colour - that is the white line again"
    finally:
        field.close()
