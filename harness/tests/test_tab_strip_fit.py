"""A tab strip either fits its labels or exposes the overflow - never clips them.

Owner, 2026-11-21: "the tab labels do not fit: the Log label is clipped and scroller arrows appear". The toolkit's
``TopTabBar`` fixes each tab to its own ``sizeHint`` and the strip to their sum, so a bar narrower than that sum draws
its last tabs past its own edge; the overflow bar it grows is a hover-only affordance, so a user who does not already
know it is there cannot reach the tabs. The host's half of the answer lives in ``src/ui/pages/proc_page.py``
(``fit_tab_strip`` + ``TabStripHost``) because both pages' strips share it and neither page may say "it does not fit"
without saying by how much and where the user can reach it.

Two pins per strip: the arithmetic (a pure function, no display, no Qt widget) and the real behaviour (a narrow strip
scrolls its SELECTED tab whole into view and shows the overflow row). The label widths and the bar's own gap are the
numbers the two real strips report today - the process strip in its 500 px column, the settings strip in its wide one.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

# The pages import the compiled Qt resource blob by name; a test tree has no build step for it, so the module is
# stubbed before the page module (which is where the strip's host half lives) can be imported.
if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.pages.proc_page import TabButtonFactory, TabStripHost, fit_tab_strip, overflow_style  # noqa: E402

#: The process page's five labels at the app's font, as their buttons measure them (label + the shared style's
#: padding), the bar's own chrome and spacings around them, and the column the page caps itself at.
PROC_LABELS = ("General", "Encoder Options", "Render Queue", "Advanced", "Logs")
PROC_WIDTHS = (68, 118, 106, 79, 51)
PROC_GAP = 28
PROC_COLUMN = 500

#: The settings page's three labels and the wide column they sit in.
SETTINGS_LABELS = ("Output Settings", "Render Settings", "RVE Settings")
SETTINGS_WIDTHS = (113, 113, 94)
SETTINGS_GAP = 20
SETTINGS_COLUMN = 1027


# --- the arithmetic (pure: no display, no widget) ---------------------------------------------------------------


def test_the_process_strip_fits_its_real_column():
    fit = fit_tab_strip(PROC_WIDTHS, PROC_COLUMN, gap=PROC_GAP)

    assert fit.fits, f"the five process labels need {fit.needed} px and the column has {PROC_COLUMN}"
    assert fit.overflow == 0
    assert fit.needed == 450, "the pinned need changed - re-measure the labels before editing this number"


def test_the_process_strip_reports_its_overflow_when_the_column_narrows():
    """The defect: 382 px of bar, 450 px of tabs - the last two tabs sit outside it."""
    fit = fit_tab_strip(PROC_WIDTHS, 382, gap=PROC_GAP)

    assert not fit.fits
    assert fit.overflow == 68, "a strip that does not fit must say exactly how much does not"


def test_the_settings_strip_fits_its_real_column():
    fit = fit_tab_strip(SETTINGS_WIDTHS, SETTINGS_COLUMN, gap=SETTINGS_GAP)

    assert fit.fits
    assert fit.needed == 340


def test_a_single_huge_label_reports_the_whole_strip_as_overflow():
    fit = fit_tab_strip([900], 200, gap=12)

    assert fit.overflow == 712


def test_an_empty_strip_has_nothing_to_overflow():
    fit = fit_tab_strip([], 0, gap=0)

    assert fit.fits and fit.needed == 0


def test_the_answer_is_measured_in_whole_pixels():
    """Widths and the gap come from widgets, so they arrive as ints - and a fractional one must not skew the answer."""
    fit = fit_tab_strip([10.9, 20.2], 100.4, gap=6.6)

    assert fit.widths == (10, 20)
    assert (fit.available, fit.gap) == (100, 6)
    assert fit.needed == 36 and fit.fits


# --- the behaviour (offscreen: no display, the real widgets) -----------------------------------------------------


@pytest.fixture()
def narrow_strip(qapp):
    """The process page's strip in a bar too narrow for it: 450 px of tabs, ~382 px of room."""
    built = _build_strip(qapp, 400)
    yield built
    built[3].close()


def _build_strip(qapp, width, style=None):
    """A real `TopTabHost` with the process page's five labels, its style, factory and host half."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    from sli_ui_toolkit.widgets import TopTabHost

    from src.ui.widgets.tab_strip import TAB_HEIGHT, tab_style

    window = QtWidgets.QMainWindow()
    window.resize(width, 300)
    column = QtWidgets.QWidget(window)
    layout = QtWidgets.QVBoxLayout(column)
    window.setCentralWidget(column)
    style = style if style is not None else overflow_style(tab_style())
    factory = TabButtonFactory()
    host = TopTabHost(column, tab_height=TAB_HEIGHT, pane_radius=0, pane_inset=(10, 5, 10, 5), chrome_inset=0,
                      tab_radius=4, pane_fill_token=None, tab_factory=factory, tab_style=style)
    for label in PROC_LABELS:
        host.addTab(QtWidgets.QWidget(), label)
    layout.addWidget(host)
    strip = TabStripHost(host, style, factory)
    window.show()
    for _ in range(6):
        qapp.processEvents()
    return host, strip, factory, window


def _whole_in_view(host, factory, index) -> bool:
    bar = host.tabBar()
    button = factory.buttons[index]
    offset = bar.scroll_offset()
    return button.x() >= offset and button.x() + button.width() <= offset + bar.width()


def test_the_process_labels_fit_whole_in_the_real_column(qapp):
    """The owner's report, pinned: "Logs" is not clipped when the five labels fit the column."""
    host, strip, factory, window = _build_strip(qapp, 502)
    try:
        assert strip.strip_fit().fits, "the five labels must fit the process page's 500 px column"
        for index in range(host.count()):
            assert _whole_in_view(host, factory, index), (
                f"tab {host.tabText(index)!r} is cut at the strip's edge in a column that has room for it"
            )
    finally:
        window.close()


def test_the_overflow_is_visible_and_the_selected_tab_stays_whole(narrow_strip, qapp):
    host, strip, factory, _window = narrow_strip
    bar = host.tabBar()

    fit = strip.strip_fit()
    assert not fit.fits and fit.overflow > 0, "this pin needs a strip that does not fit"
    assert bar.height() > bar.sizeHint().height(), (
        "a strip that overflows must grow its overflow row (a visible affordance), not wait for the pointer"
    )

    for index in range(host.count()):
        host.setCurrentIndex(index)
        qapp.processEvents()
        assert _whole_in_view(host, factory, index), (
            f"tab {index} ({host.tabText(index)!r}) is selected but not whole inside the bar: "
            f"offset={bar.scroll_offset()} width={bar.width()}"
        )


def test_the_overflow_row_is_reachable_by_wheel(narrow_strip, qapp):
    """The row is the affordance; this proves the strip behind it actually moves."""
    host, strip, factory, _ = narrow_strip
    bar = host.tabBar()
    maximum = bar.required_width() - bar.width()
    assert maximum > 0

    bar.set_scroll_offset(maximum)
    qapp.processEvents()

    assert bar.scroll_offset() == maximum
    assert _whole_in_view(host, factory, host.count() - 1), "the last tab must be readable at the far end of the strip"


def test_a_widened_strip_drops_its_offset(narrow_strip, qapp):
    host, strip, _factory, _ = narrow_strip
    bar = host.tabBar()
    host.setCurrentIndex(host.count() - 1)
    qapp.processEvents()
    assert bar.scroll_offset() > 0

    host.window().resize(900, 300)
    for _ in range(6):
        qapp.processEvents()

    assert strip.strip_fit().fits
    assert bar.scroll_offset() == 0, "a strip that fits again must not leave its first tabs pushed out of the bar"


def _plate(button):
    """The exact fill the toolkit's painter uses for this button (its override channel), or None."""
    colour = button._override_bg_color
    return colour.name() if colour is not None else None


def test_the_selected_plate_is_the_strips_own_token(qapp):
    """A theme that restyles `TabStyle.selected_token` restyles the selected tab - the host's token, not a fixed one."""
    from dataclasses import replace

    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.widgets.tab_strip import tab_style

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    from sli_ui_toolkit.theme import ThemeManager

    theme = ThemeManager.get_instance()

    default_style = overflow_style(tab_style())
    accent_style = overflow_style(replace(tab_style(), selected_token="accent"))
    first = _build_strip(qapp, 502, default_style)
    second = _build_strip(qapp, 502, accent_style)
    try:
        default_plate = _plate(first[2].buttons[0])
        accent_plate = _plate(second[2].buttons[0])

        assert default_plate == theme.try_get_color(default_style.selected_token).name()
        assert accent_plate == theme.try_get_color("accent").name()
        assert default_plate != accent_plate, "two different tokens must paint two different plates"
        assert _plate(first[2].buttons[1]) is None, "only the selected tab carries the plate"
    finally:
        first[3].close()
        second[3].close()
