"""The video-info pane shows all nine fields at every UI factor - no hidden line under the edge fade.

Owner, 2026-11-24: "the fade is aggressive? the text fits, but it fades the bottom." Two numbers were behind it:

* the pane (`RichTextView`) always asked for the same height, so its own `sizeHint` could never say "I need 228 px";
* its container (`widget_26`) carried the Designer's `setMaximumSize(..., 200)`, a cap ABOVE the panel's 190 px
  floor - so the panel could not grow into its content even if it had asked.

At 2.0 the nine-field body measures 208 px in a 170 px viewport: the last line sat under the scroll area's edge fade
(a 38 px scroll range, `_EdgeFade` requires 23). The pane now reports its content height (`fit_content=True`) and the
container has no vertical ceiling, so the fade disappears because there is nothing left to fade.

The pin is the invariant: **the pane's scroll range is zero and its body is fully inside its viewport** at 1.0 (the
design factor) and at 2.0 (the raised one the report came from).
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

#: The nine rows a loaded clip puts in the pane, in the app's own shape (`REAL-Video-Enhancer.py` writes `setText`).
LOADED_VALUES = (
    "К/с: 24.0 -> 24.0\n"
    "Разрешение: 1440x1080 -> 2880x2160\n"
    "Количество кадров: 495 -> 495\n"
    "Кодировщик: hevc -> libx264\n"
    "Контейнер: mkv -> mkv\n"
    "Цветовое пространство: bt709 -> bt709\n"
    "Формат пикселей: yuv420p10le -> yuv420p10le\n"
    "Поддержка HDR: False -> False\n"
    "Глубина цвета: 10 bit -> 10 bit\n"
)

#: The pane's own floor (the Designer's 190 px): an EMPTY panel keeps it, a filled one may grow past it.
PANEL_FLOOR_PX = 190


@pytest.fixture()
def page_at(qapp):
    """Build a real General Settings page at a factor and hand back its video-info pane, laid out and settled."""
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.pages.general_settings import GeneralSettingsPage
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    factor_before, language_before = scale.factor(), i18n.current_language()
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    i18n.configure_i18n("ru")

    def build(factor: float):
        scale.set_factor(factor)
        page = GeneralSettingsPage()
        root = page.build()
        i18n.install(root)
        root.resize(900, 1200)
        root.show()
        for _ in range(8):
            qapp.processEvents()
        pane = page.inputVideoInfoTextEdit
        pane.setText(LOADED_VALUES)
        # The body is measured by the engine on a deferred pass after a resize/text change.
        for _ in range(10):
            qapp.processEvents()
        root.resize(900, 1200)
        for _ in range(6):
            qapp.processEvents()
        return root, page, pane

    roots = []
    try:
        yield build, roots
    finally:
        for root in roots:
            root.close()
        scale.set_factor(factor_before)
        i18n.configure_i18n(language_before)


@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_the_info_pane_shows_every_field_it_holds(page_at, factor):
    build, roots = page_at
    root, _page, pane = build(factor)
    roots.append(root)
    scroll = pane._scroll
    body = pane.label.height()
    viewport = scroll.viewport().height()
    assert pane.height() >= PANEL_FLOOR_PX, "the panel's floor keeps an empty pane readable"
    assert body <= viewport, (
        f"at factor {factor} the body is {body} px in a {viewport} px viewport - the last field line sits under the "
        "edge fade")
    assert scroll.verticalScrollBar().maximum() == 0, (
        f"at factor {factor} the pane scrolls by {scroll.verticalScrollBar().maximum()} px although its floor is "
        f"{PANEL_FLOOR_PX} and its body needs {body} - the container's ceiling blocked the growth")


def test_the_pane_reports_the_height_of_its_content(page_at):
    """The hint is what lets the layout give the pane its body's height instead of a hard-coded box."""
    build, roots = page_at
    root, _page, pane = build(2.0)
    roots.append(root)
    assert pane.sizeHint().height() >= pane._content_height(), "the hint must include the whole body and the inset"
    assert pane._content_height() > PANEL_FLOOR_PX, "the stand only proves something if the body is past the floor"
