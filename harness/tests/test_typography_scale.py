"""RVE's type scale must follow the toolkit's ``UiScale`` - one design value, multiplied once.

Why this file exists (2026-11-24): ``src/ui/typography.py`` was the single owner of RVE's font sizes, but it resolved
them into a plain ``QFont`` once, at build time. ``UiScale.set_factor(2.0)`` therefore reached the toolkit's own widgets
(a ``Button`` sizeHint went 53x33 -> 80x48 -> 106x65) and nothing in RVE: the body text stayed 12 pt at every factor.
The red state this file was written against is exactly that - ``point_size_of(widget)`` did not move when the factor
did.

What the pins say:

* the body point size doubles at factor 2.0 and returns at 1.0 - the whole feature in one assertion;
* a control's pixel size does the same, through the same module, so the pt-vs-px split is scaled, not replaced;
* the deliberate ratio between the two scales (a 12 pt label is 16 px, controls are 12 px) is stated in code and may
  not drift;
* ``UiFont.font_changed`` re-applies the resolved font to a widget that was already styled - the "stays in sync" half
  of ``UiFont.apply``;
* ``src/ui/toolkit_theme.py`` does not carry a second 12 px literal.

All of it is offscreen; nothing here needs a window.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from sli_ui_toolkit.ui.managers import UiFont, UiScale  # noqa: E402

from src.ui import typography  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: The design point size the body group is pinned to; the test reads it rather than restating it.
BODY_PT = typography.BODY_PT


@pytest.fixture(autouse=True)
def _restore_ui_scale():
    """Leave the process-wide factor where it was: the singleton outlives a test."""
    scale = UiScale.get_instance()
    previous = scale.factor()
    yield
    scale.set_factor(previous)


def test_body_point_size_doubles_with_the_ui_scale(qapp):
    """Red before the change: the body point size ignored ``UiScale`` and stayed at 12 pt."""
    label = QLabel("body")
    typography.apply(label)

    assert typography.point_size_of(label) == pytest.approx(BODY_PT, abs=0.01)

    UiScale.get_instance().set_factor(2.0)
    assert typography.point_size_of(label) == pytest.approx(BODY_PT * 2, abs=0.01)

    UiScale.get_instance().set_factor(1.0)
    assert typography.point_size_of(label) == pytest.approx(BODY_PT, abs=0.01)


def test_control_pixel_size_doubles_with_the_ui_scale(qapp):
    """The control scale is px, the label scale is pt; both are design values and both follow one factor."""
    control = QLabel("control")
    typography.apply_control_font(control)

    assert control.font().pixelSize() == typography.CONTROL_PX

    UiScale.get_instance().set_factor(2.0)
    assert control.font().pixelSize() == typography.CONTROL_PX * 2

    UiScale.get_instance().set_factor(1.0)
    assert control.font().pixelSize() == typography.CONTROL_PX


def test_a_toolkit_text_keeps_its_design_size_and_follows_the_scale(qapp):
    """The painted labels are toolkit ``Text``, whose own style pass re-resolves on every scale change.

    A widget font would be overwritten by that pass, so ``apply`` pins the DESIGN pixel size on the widget instead; at
    factor 1.0 the resolved size is the one Qt already gives the point size, so the look does not move.
    """
    from sli_ui_toolkit.widgets import Text

    body = Text("body")
    typography.apply(body)
    assert body.font().pixelSize() == typography.label_px(BODY_PT) == 16

    display = Text("display")
    typography.apply(display, typography.DISPLAY_PT)
    assert display.font().pixelSize() == typography.label_px(typography.DISPLAY_PT) == 33

    UiScale.get_instance().set_factor(2.0)
    qapp.processEvents()
    assert body.font().pixelSize() == 32
    assert display.font().pixelSize() == 66

    UiScale.get_instance().set_factor(1.0)
    qapp.processEvents()
    assert body.font().pixelSize() == 16
    assert display.font().pixelSize() == 33


def test_the_two_scales_keep_their_declared_ratio():
    """12 pt is 16 px at Qt's 96 dpi, and a control is 12 px: the 0.75 ratio is a design decision, not a typo."""
    assert typography.label_px(BODY_PT) == 16
    assert typography.CONTROL_RATIO == pytest.approx(typography.CONTROL_PX / 16)
    assert typography.CONTROL_RATIO == pytest.approx(0.75)


def test_font_changed_reapplies_the_resolved_font(qapp):
    """``UiFont.apply`` re-applies on a real font change - a widget cannot keep a stale size that way."""
    label = QLabel("retouched")
    typography.apply(label)

    stale = QFont(label.font())
    stale.setPointSize(30)
    label.setFont(stale)
    assert typography.point_size_of(label) == pytest.approx(30, abs=0.01)

    UiFont.get_instance().font_changed.emit()
    assert typography.point_size_of(label) == pytest.approx(BODY_PT, abs=0.01)


def test_a_family_override_survives_a_scale_change(qapp):
    """The family is an override on the same resolved font, so scaling does not drop it."""
    label = QLabel("family")
    family = QFont().defaultFamily()
    typography.apply(label, family=family)

    UiScale.get_instance().set_factor(1.5)
    assert label.font().family() == family
    assert typography.point_size_of(label) == pytest.approx(BODY_PT * 1.5, abs=0.01)


def test_toolkit_theme_has_no_second_pixel_literal():
    """The application font's size comes from ``typography.CONTROL_PX``; a literal here would be a second source."""
    source = (ROOT / "src" / "ui" / "toolkit_theme.py").read_text(encoding="utf-8")
    assert not re.search(r"setPixelSize\s*\(\s*\d", source), (
        "src/ui/toolkit_theme.py must take its pixel size from src/ui/typography.py"
    )
