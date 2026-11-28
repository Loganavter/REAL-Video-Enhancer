"""The General tab fits its column: no field is clipped at the column's edge, and a page return does not widen it.

Owner report, 2026-11-24: with a clip loaded and upscaling enabled, going to Settings and back to Inference broke the
middle of the layout - the path field, the model combos and the queue button were cut at the column's edge.

Two independent causes, both measured on this tree:

* **the row cap fed on its own output** (`src/ui/widgets/combo_width.py`). A `Maximum` combo is clamped into its row's
  minimum hint, and the cap was computed from that row's width: enabling upscale makes the vertical scrollbar appear,
  the viewport drops 478 -> 470, the row keeps its 438 px from the pre-scrollbar pass (content 478 against a 470
  viewport), and each return from Settings re-laid the row at a wider transient that the cap took as a new floor
  (438 -> 446, content 478 -> 486). Measured with the real window: `contentMin` 478 -> 486 after one return;
* **`outputFileText` carried the Designer's `MinimumExpanding`** - a policy without `ShrinkFlag`, so `qSmartMinSize`
  takes `sizeHint` (the field's TEXT width) as the row's MINIMUM. A normal output path (464 px of text) made the
  content 486 px wide. The input field next door is `Expanding` and never did this.

The pin is the invariant rather than a number per widget: **the scroll content's minimum width never exceeds the
viewport it is shown in**, before and after a page return, with the upscale section open and a long path in the field.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtWidgets import QMainWindow  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

#: The page the owner ships at 1.0: a 500 px column in the design window.
WINDOW = (1163, 900)

#: A path as long as the app's own auto-generated one - the string that used to set the pane's minimum width.
LONG_OUTPUT_PATH = "/home/jorj/native1440_20s_2x-futsuu-anime(1).pth_24.0fps_2880x2160_(6).mkv"

#: The stack the nav pages live in, in `PAGE_NAMES` order (`src/ui/pages/handles.py`).
PAGES = ("home", "more", "proc", "settings", "download")


@pytest.fixture()
def shell(qapp):
    """The real shell at 1.0 in the design window, on the process page, with the General tab's upscale section OPEN.

    The section is opened directly: in the app it is `updateVideoGUIGetails` that shows it once a clip has an upscale
    model, and that path needs a probed video (ffmpeg/torch) that this suite deliberately does not load. What the
    geometry sees is the same widget state.
    """
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    factor_before, language_before = scale.factor(), i18n.current_language()
    scale.set_factor(1.0)
    i18n.configure_i18n("ru")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)

    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    window.resize(*WINDOW)
    window.show()
    _settle(qapp)
    window.resize(*WINDOW)
    _settle(qapp)

    page = shell.procPage_page
    _show_page(window, "proc")
    page.generalSettings_page.generalUpscaleContainer.setVisible(True)
    _settle(qapp)
    try:
        yield shell, window, page
    finally:
        window.close()
        scale.set_factor(factor_before)
        i18n.configure_i18n(language_before)


def _settle(qapp, turns: int = 12) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _show_page(window, name: str) -> None:
    from PySide6.QtWidgets import QStackedWidget

    stack = window.findChild(QStackedWidget, "stackedWidget")
    assert stack is not None, "the nav page stack is missing"
    stack.setCurrentIndex(PAGES.index(name))


def _content(page, window):
    """`(viewport width, content)` of the General tab's own scroll surface."""
    from PySide6.QtWidgets import QWidget

    viewport = page.generalSettings_page.scrollArea_4.viewport()
    content = window.findChild(QWidget, "scrollAreaWidgetContents_4")
    assert content is not None, "the General tab's scroll content is missing"
    return viewport.width(), content


def test_the_general_tab_never_asks_for_more_width_than_its_pane(shell, qapp):
    """Upscale open + a long output path: the content must fit the viewport, and a page return must not widen it."""
    window, page = shell[1], shell[2]
    page.generalSettings_page.outputFileText.setText(LONG_OUTPUT_PATH)
    _settle(qapp)

    viewport, content = _content(page, window)
    assert content.minimumSizeHint().width() <= viewport, (
        f"the General tab asks for {content.minimumSizeHint().width()} px inside a {viewport} px viewport - every "
        "field is clipped at the column's edge")
    assert content.width() <= viewport

    first = content.minimumSizeHint().width()
    for visit in (1, 2):
        _show_page(window, "settings")
        _settle(qapp)
        _show_page(window, "proc")
        _settle(qapp)
        viewport, content = _content(page, window)
        assert content.minimumSizeHint().width() == first, (
            f"return #{visit} grew the General tab's demand {first} -> {content.minimumSizeHint().width()} px "
            "(the row cap fed on its own output)")
        assert content.width() <= viewport


def test_the_cap_follows_the_pane_not_the_row_it_sizes(shell, qapp):
    """A row laid out wider than its pane must not hand the combo that width: the row is the cap's own output.

    The page-level test above needs the app's own video path to reproduce the wider transient; this stand states the
    mechanism directly. A row stretched 60 px past its viewport (what the pre-scrollbar pass did in the live window)
    must still cap the combo at what the PANE can give it.
    """
    from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

    from sli_ui_toolkit.widgets import ComboBox, Text
    from src.ui.widgets.combo_width import cap_combo_to_row

    pane = QScrollArea()
    pane.setWidgetResizable(True)
    content = QWidget()
    column = QVBoxLayout(content)
    column.setContentsMargins(11, 11, 11, 11)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)
    label = Text(parent=row)
    label.setText("Upscale model")
    combo = ComboBox(row)
    combo.addItem("2x-FutsuuAnime_Compressed_Lite_v1.0_2024-01-31.pth")
    row_layout.addWidget(label)
    row_layout.addWidget(combo)
    column.addWidget(row)
    pane.setWidget(content)
    pane.resize(300, 300)
    pane.show()
    cap_combo_to_row(combo)
    _settle(qapp)

    viewport = pane.viewport().width()
    content.setMinimumWidth(viewport + 60)
    _settle(qapp)

    pane_room = viewport - 22 - label.sizeHint().width() - row_layout.spacing() * 1
    assert combo.minimumWidth() <= pane_room
    assert combo.maximumWidth() <= pane_room, (
        f"the cap is {combo.maximumWidth()} px for a {viewport} px pane: the row's width fed the cap its own output")
    pane.close()


def test_the_upscale_combo_cap_does_not_grow_across_page_returns(shell, qapp):
    """The cap is a function of the PANE, not of the row's width, so a return cannot ratchet it upward."""
    window, page = shell[1], shell[2]
    combo = page.generalSettings_page.upscaleModelComboBox
    _settle(qapp)
    first = combo.maximumWidth()
    for _ in range(2):
        _show_page(window, "settings")
        _settle(qapp)
        _show_page(window, "proc")
        _settle(qapp)
    assert combo.maximumWidth() == first, (
        f"the upscale combo's cap moved {first} -> {combo.maximumWidth()} px across a page return")
    viewport, _ = _content(page, window)
    label = page.generalSettings_page.label_7
    assert first <= viewport, "a cap wider than the pane it lives in is not a cap"
    assert label.width() + combo.maximumWidth() <= viewport, (
        "the capped row still does not fit the pane's viewport")


def test_the_queue_button_is_as_wide_as_its_own_text(shell, qapp):
    """Owner, 2026-11-28: the label rendered "Add to Render Que…".

    The button was built with a literal `size=(200, 35)` while its content (icon + text + paddings) is what decides how
    wide it must be; at a raised factor the English label no longer fit and elided. `size=(0, 35)` is the toolkit's
    "width from content, height fixed". The property pinned here is the one that matters: at 1.0 and at 1.25 the button
    is at least as wide as its own content hint, in the longer (English) label.
    """
    from PySide6.QtGui import QFontMetrics
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    _shell, window, page = shell
    button = page.generalSettings_page.addToRenderQueueButton
    scale = UiScale.get_instance()
    factor_before = scale.factor()
    try:
        for factor in (1.0, 1.25):
            scale.set_factor(factor)
            window.resize(window.width(), window.height())
            _settle(qapp)
            needed = button.minimumSizeHint().width()
            assert button.width() >= needed, (
                f"at {factor} the queue button is {button.width()} px for a {needed} px content - the label would elide")
            assert needed > 0
    finally:
        scale.set_factor(factor_before)
        _settle(qapp)
