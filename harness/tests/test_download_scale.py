"""The download/home/log pages' boxes follow the UI factor, and their text is not pinned to a point size.

Owner decision (2026-11-24, wave 4 of the UI-scale migration): *the boxes grow with the text they hold.* Wave 2 did
this for the process column; the defect this file pins is the same shape on three other pages - a raw Qt pixel ceiling
around a control whose painted content already follows ``UiScale``. Measured at the tree before this change:

* the download page's ten icon buttons are capped at ``QSize(50, 45)`` at every factor, while the toolkit paints
  their 30 design-px icon at 45 px at 1.5 and 60 px at 2.0 - the icon is clipped by its own button;
* the home page's two icon buttons are capped at 55 px wide while their 35 design-px icon paints 52/70 px;
* the log row's 100 px header and 50 px command row stay fixed while the 12 pt/13 pt text inside doubles, and the
  encoder command's 200 px floor does not grow with the row it guards.

Why the icon SIZE itself is not multiplied here: ``Button.setIconSize`` stores a **design** value and the toolkit's
style bridge scales it at paint time (``read_widget_style`` -> ``UiScale.scaled_px``), as do ``content_padding`` and
the corner radius. Multiplying it in the page would apply the factor twice, which is why the test pins that the stored
icon size stays at the design number while the box around it grows.

The inline `font-size` decision (home ``changeLogText``, log ``renderOutput``): removed, because the document font is
the pane's own face and already follows the factor. Qt's rich-text engine applies an inline `font-size` as an
ABSOLUTE value - measured here, the body block stayed at 11 pt at 1.0, 1.5 and 2.0 while the document default font
went 12 -> 18 -> 24 px. The home pane's visible changelog is fed through ``setMarkdown`` (``HomeTab.QConnect``), which
already used that document default, so the absolute 11 pt only ever sized the empty placeholder it replaced. The log
pane is a painted console whose ``setHtml`` keeps the plain text only, so its 13 pt never reached a glyph.

1.0 is the gate: every expected value below is today's number exactly, and the raised factors are the same design
values through ``scaled_px``.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtGui import QTextCursor  # noqa: E402

from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.pages.download_page import (  # noqa: E402
    ICON_BUTTON_CONTENT_PADDING, ICON_BUTTON_ICON_PX, ICON_BUTTON_MAX_HEIGHT_PX, ICON_BUTTON_MAX_WIDTH_PX,
    DownloadPagePage,
)
from src.ui.pages.home_page import (  # noqa: E402
    HOME_ICON_CONTENT_PADDING, HOME_ICON_PX, HOME_ICON_MAX_WIDTH_PX, HomePagePage,
)
from src.ui.pages.tab_6 import (  # noqa: E402
    ENCODER_COMMAND_MIN_WIDTH, ENCODER_HEADER_MAX_HEIGHT_PX, ENCODER_ROW_HEIGHT_PX, Tab6Page,
)

#: Today's geometry at 1.0 and the same design values through the factor - the pins the owner asked for.
DOWNLOAD_BOX = {1.0: (50, 45), 1.5: (75, 68), 2.0: (100, 90)}
HOME_MAX_WIDTH = {1.0: 55, 1.5: 82, 2.0: 110}
ENCODER_HEADER = {1.0: 100, 1.5: 150, 2.0: 200}
ENCODER_ROW = {1.0: 50, 1.5: 75, 2.0: 100}
ENCODER_FLOOR = {1.0: 200, 1.5: 300, 2.0: 400}

#: The document default font of the changelog pane, in px (design 12 px, multiplied once by the factor).
#: The changelog body's document font per factor. It was 12/18/24 - the toolkit's UNPINNED default, i.e. smaller than
#: the app's body (12 pt = 16 px at 96 DPI), which is what the owner saw as "the left pane does not follow the app's
#: typography" (2026-11-28). The pane now states the app's body size through `RichTextView.setTextPixelSize`
#: (`label_px(BODY_PT)`), which the toolkit multiplies by the factor exactly once.
CHANGELOG_DEFAULT_PX = {1.0: 16, 1.5: 24, 2.0: 32}

#: `QWIDGETSIZE_MAX`: the height side of a max-width-only box stays unbounded.
UNBOUNDED = 16777215


@pytest.fixture()
def scale_state():
    """Put the process-wide factor back: the singleton outlives a test and every later page inherits it."""
    from src.ui import i18n

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _download(scale: UiScale, factor: float) -> DownloadPagePage:
    scale.set_factor(factor)
    page = DownloadPagePage()
    page.build()
    return page


def _home(scale: UiScale, factor: float) -> HomePagePage:
    scale.set_factor(factor)
    page = HomePagePage()
    page.build()
    return page


def _encoder(scale: UiScale, factor: float) -> Tab6Page:
    scale.set_factor(factor)
    page = Tab6Page()
    page.build()
    return page


def _icon_buttons(page: DownloadPagePage):
    """Every download-page icon button, stated once so a new one cannot silently miss the follower."""
    return [
        page.uninstallAppBtn,
        page.downloadRecommendedBtn,
        page.downloadNCNNBtn,
        page.uninstallNCNNBtn,
        page.downloadTorchBtn,
        page.uninstallTorchBtn,
        page.downloadTensorRTBtn,
        page.uninstallTensorRTBtn,
        page.downloadDirectMLBtn,
        page.uninstallDirectMLBtn,
    ]


def _assert_box(button, width: int, height: int, what: str) -> None:
    assert (button.maximumWidth(), button.maximumHeight()) == (width, height), (
        f"{what}: {button.maximumWidth()}x{button.maximumHeight()} px, expected {width}x{height}")


# --- the design values are today's, named -------------------------------------------------------


def test_the_design_constants_are_the_geometry_the_pages_had(qapp, scale_state):
    """A constant is only an improvement if it still says today's number; 1.0 is that number, unscaled."""
    assert (ICON_BUTTON_MAX_WIDTH_PX, ICON_BUTTON_MAX_HEIGHT_PX) == DOWNLOAD_BOX[1.0]
    assert ICON_BUTTON_ICON_PX == 30
    assert ICON_BUTTON_CONTENT_PADDING == (10, 5, 10, 5)
    assert (HOME_ICON_MAX_WIDTH_PX, HOME_ICON_PX) == (HOME_MAX_WIDTH[1.0], 35)
    assert HOME_ICON_CONTENT_PADDING == (10, 5, 10, 5)
    assert ENCODER_HEADER_MAX_HEIGHT_PX == ENCODER_HEADER[1.0]
    assert ENCODER_ROW_HEIGHT_PX == ENCODER_ROW[1.0]
    assert ENCODER_COMMAND_MIN_WIDTH == ENCODER_FLOOR[1.0]
    for expected, design in ((DOWNLOAD_BOX, (50, 45)), (HOME_MAX_WIDTH, 55), (ENCODER_HEADER, 100),
                             (ENCODER_ROW, 50), (ENCODER_FLOOR, 200)):
        assert expected[1.0] == design


# --- the download page's icon buttons -------------------------------------------------------------


@pytest.mark.parametrize("factor", sorted(DOWNLOAD_BOX))
def test_the_download_icon_boxes_follow_the_factor(qapp, scale_state, factor):
    """10 buttons, one design box: 50x45 at 1.0, 75x68 at 1.5, 100x90 at 2.0."""
    page = _download(scale_state, factor)
    width, height = DOWNLOAD_BOX[factor]
    for button in _icon_buttons(page):
        _assert_box(button, width, height, f"{button.objectName()} at factor {factor}")
        # The design icon size is NOT multiplied here - the toolkit scales it at paint time, once.
        assert button.getIconSizePx() == ICON_BUTTON_ICON_PX, (
            f"{button.objectName()}: the stored icon size must stay the design value, or the factor is applied twice")


@pytest.mark.parametrize("factor", (1.5, 2.0))
def test_the_download_icon_box_holds_the_icon_the_toolkit_paints(qapp, scale_state, factor):
    """The box has to contain the painted icon plus its padding, or the control clips its own glyph."""
    page = _download(scale_state, factor)
    icon = scaled_px(ICON_BUTTON_ICON_PX)
    pad_x = scaled_px(ICON_BUTTON_CONTENT_PADDING[0]) + scaled_px(ICON_BUTTON_CONTENT_PADDING[2])
    pad_y = scaled_px(ICON_BUTTON_CONTENT_PADDING[1]) + scaled_px(ICON_BUTTON_CONTENT_PADDING[3])
    button = page.uninstallAppBtn
    assert button.maximumWidth() >= icon + pad_x, f"icon {icon} + padding {pad_x} in {button.maximumWidth()} px"
    assert button.maximumHeight() >= icon + pad_y, f"icon {icon} + padding {pad_y} in {button.maximumHeight()} px"


def test_a_live_factor_change_moves_the_download_boxes(qapp, scale_state):
    """The capture channel's `scale:<factor>` moves a built page - no restart, no rebuild."""
    page = _download(scale_state, 1.0)
    buttons = _icon_buttons(page)
    assert buttons[0].maximumWidth() == 50 and buttons[0].maximumHeight() == 45

    scale_state.set_factor(1.5)
    for _ in range(6):
        qapp.processEvents()
    for button in buttons:
        _assert_box(button, 75, 68, f"{button.objectName()} after a live 1.5")

    scale_state.set_factor(2.0)
    for _ in range(6):
        qapp.processEvents()
    for button in buttons:
        _assert_box(button, 100, 90, f"{button.objectName()} after a live 2.0")


# --- the home page's icon buttons -----------------------------------------------------------------


@pytest.mark.parametrize("factor", sorted(HOME_MAX_WIDTH))
def test_the_home_icon_boxes_follow_the_factor(qapp, scale_state, factor):
    """55 px wide at 1.0, 82 at 1.5, 110 at 2.0; the height side stays unbounded."""
    page = _home(scale_state, factor)
    for button in (page.kofiBtn, page.githubBtn):
        _assert_box(button, HOME_MAX_WIDTH[factor], UNBOUNDED, f"{button.objectName()} at factor {factor}")
        assert button.getIconSizePx() == HOME_ICON_PX


def test_a_live_factor_change_moves_the_home_boxes(qapp, scale_state):
    page = _home(scale_state, 1.0)
    assert page.githubBtn.maximumWidth() == 55
    scale_state.set_factor(2.0)
    for _ in range(6):
        qapp.processEvents()
    _assert_box(page.githubBtn, 110, UNBOUNDED, "githubBtn after a live 2.0")
    _assert_box(page.kofiBtn, 110, UNBOUNDED, "kofiBtn after a live 2.0")


# --- the log row ----------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", sorted(ENCODER_HEADER))
def test_the_encoder_log_row_follows_the_factor(qapp, scale_state, factor):
    """The 100 px header band, the 50 px command row and the 200 px field floor all take the factor."""
    page = _encoder(scale_state, factor)
    assert page.widget_33.maximumHeight() == ENCODER_HEADER[factor]
    row = ENCODER_ROW[factor]
    assert page.scrollArea_6.minimumHeight() == row
    assert page.scrollArea_6.maximumHeight() == row
    assert page.EncoderCommand.minimumWidth() == ENCODER_FLOOR[factor]


def test_a_live_factor_change_moves_the_log_row(qapp, scale_state):
    page = _encoder(scale_state, 1.0)
    assert page.widget_33.maximumHeight() == 100
    scale_state.set_factor(2.0)
    for _ in range(6):
        qapp.processEvents()
    assert page.widget_33.maximumHeight() == 200
    assert page.scrollArea_6.minimumHeight() == 100
    assert page.scrollArea_6.maximumHeight() == 100
    assert page.EncoderCommand.minimumWidth() == 400


# --- the inline HTML sizes ------------------------------------------------------------------------


@pytest.mark.parametrize("factor", sorted(CHANGELOG_DEFAULT_PX))
def test_the_changelog_body_uses_the_scaled_document_font(qapp, scale_state, factor):
    """No inline point size: the block takes the document font, which follows the factor once."""
    page = _home(scale_state, factor)
    engine = page.changeLogText.label.canvas()
    document = engine._rich.document()

    assert document.defaultFont().pixelSize() == CHANGELOG_DEFAULT_PX[factor]

    cursor = QTextCursor(document)
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    block_font = cursor.charFormat().font()
    assert block_font.pointSizeF() == pytest.approx(-1.0), (
        "an inline font-size is absolute in Qt's rich-text engine; this block must inherit the document font")
    assert block_font.pixelSize() in (-1, CHANGELOG_DEFAULT_PX[factor])


def test_the_log_pane_html_carries_no_absolute_font_size(qapp, scale_state, monkeypatch):
    """`ConsoleView.setHtml` keeps the plain text only, but the markup it is handed must not state a size either.

    The console paints from its own font (``apply(SMALL_PT)``), so an inline 13 pt in the Designer's empty paragraph
    was unreachable text - and would be an absolute block size the day the pane grew a rich engine.
    """
    captured: list[str] = []
    from sli_ui_toolkit.widgets import ConsoleView

    original = ConsoleView.setHtml

    def capture(self, html):
        captured.append(str(html))
        return original(self, html)

    monkeypatch.setattr(ConsoleView, "setHtml", capture)

    from src.ui.pages.tab_5 import Tab5Page

    scale_state.set_factor(1.0)
    Tab5Page().build()
    assert captured, "the log page never called setHtml"
    assert not any("font-size" in html for html in captured), (
        "the log pane's markup states an absolute font size; the console's own font is the live size")


def test_the_home_markup_carries_no_absolute_font_size(qapp, scale_state):
    """The source of the changelog body states no size either: the rendered block above is the behaviour, this is
    the string that produces it."""
    page = _home(scale_state, 1.0)
    assert "font-size" not in page.changeLogText._markup, page.changeLogText._markup[:200]
