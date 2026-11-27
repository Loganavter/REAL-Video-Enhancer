"""The video-info pane's compact Render Queue card: the tab's rows, four of them, a pin, and a live mirror.

The owner (2026-11-28): *"добавить тот же самый виджет отображения очереди, но под карточками информации о прогоне в
виде виджета куда 4 строки влазеет + пролистывание, а также сверху справа пин который при нажатии открывает таб с
полным списком"*.

What is pinned, and why no assertion is a snapshot:

* **one list, not two renderers.** The tab's list and the card's list are asserted to share ONE painter object and
  ONE tooltip closure, built by `src/ui/queue_rows.py` - a second renderer cannot appear without failing here;
* **exactly four rows.** The height is re-derived from `TextListWidget.row_height()` and the list's own padding
  (`content_height() - count * row_height`), so the pin is `4 * row + padding` at any UI factor;
* **more items scroll, they do not grow the card.** Six items give a positive vertical range while the card's
  height is identical to the four-item state;
* **the pin selects by identity.** The test host places the Render Queue page at index 1, so an implementation with
  a hardcoded `2` cannot pass;
* **the card is live.** A row added to the model AFTER the build reaches the card through the glue's one refresh call
  (`ProcessTab.refreshQueueRows`), not through a build-time copy;
* **the catalogue follows a language switch without rebuilding the list.** The list object is asserted to be the
  same object before and after, and its rows byte-identical.

Red-before: at the parent commit the card does not exist at all - nor do the two catalogue keys it reads - so
collection fails on `from src.ui.pages.general_settings import INFO_QUEUE_EMPTY, ...` (`ImportError`) before a single
test runs. There is no older behaviour to revert to; the feature is absent. Verified by restoring the parent
revision of the three touched files and running this module: collection error, zero tests executed; then restored
and 5 passed.

Runs offscreen only: `QT_QPA_PLATFORM=offscreen` has no QRhi, so the pin glyph and the four rows at 1.5 are
`LIVE-PROOF WANTED` in `DONE.md`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

# The app's optional runtime imports are stubbed only when this interpreter lacks them (the same seam
# `test_general_settings_info.py` resolves before importing `ProcessTab`).
for _name, _attrs in (("cpuinfo", {"get_cpu_info": lambda *a, **k: {}}),
                      ("pypresence", {"Presence": object})):
    try:  # pragma: no cover - the branch depends on the interpreter
        __import__(_name)
    except ModuleNotFoundError:  # pragma: no cover
        _module = types.ModuleType(_name)
        for _attr, _value in _attrs.items():
            setattr(_module, _attr, _value)
        sys.modules[_name] = _module

#: The app checkout root. `ProcessTab` pulls the APP's `src.constants` while the suite's conftest puts the backend's
#: `src` first (the same seam `test_general_settings_info.py` resolves).
_ROOT = str(Path(__file__).resolve().parents[2])

from PySide6.QtWidgets import QWidget  # noqa: E402

from src.ui import i18n  # noqa: E402
from src.ui.RenderQueue import RenderQueue  # noqa: E402
from src.ui.pages.general_settings import (  # noqa: E402
    INFO_QUEUE_EMPTY,
    INFO_QUEUE_PIN_TOOLTIP,
    INFO_QUEUE_ROWS,
    GeneralSettingsPage,
)

#: The tab's own row height (`render_queue_tab.py` builds its list with 20): the card must state the SAME number.
TAB_ROW_HEIGHT = 20


@pytest.fixture()
def language(qapp):
    """The real i18n module, configured in English and restored afterwards."""
    before = i18n.current_language()
    i18n.configure_i18n("en")
    yield i18n
    i18n.emit_language_changed(before)
    i18n.configure_i18n(before)


def _settle(qapp, rounds: int = 8) -> None:
    for _ in range(rounds):
        qapp.processEvents()


def _page(qapp) -> GeneralSettingsPage:
    """A real General Settings page at factor 1.0, shown, with the translator installed on its root."""
    from src.ui import toolkit_boot, toolkit_theme
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    UiScale.get_instance().set_factor(1.0)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    page = GeneralSettingsPage()
    root = page.build()
    i18n.install(root)
    root.resize(900, 1200)
    root.show()
    _settle(qapp)
    page._test_root = root  # keep the root alive for the duration of the test
    return page


def _option(name: str):
    """One queue row's model shape: the painter reads `inputFile` and the job state's progress."""
    return SimpleNamespace(inputFile=name, videoFrameCount=120, videoFps=24)


def _host(qapp):
    """The real inner tab host with the Render Queue page at index 1 - NOT index 2, so a literal cannot pass."""
    from sli_ui_toolkit.widgets import TopTabHost

    host = TopTabHost()
    first, last = QWidget(), QWidget()
    queue_page = QWidget()
    host.addTab(first, "General")
    host.addTab(last, "Encoder Options")
    host.insertTab(1, queue_page, "Render Queue")
    host.resize(420, 240)
    host.show()
    _settle(qapp)
    return host, queue_page


def _process_tab():
    """`src.ui.ProcessTab`, imported with the APP's `src` ahead of the backend's (the repo's own dance)."""
    path_before = list(sys.path)
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)
    sys.modules.pop("src.constants", None)
    try:
        from src.ui.ProcessTab import ProcessTab
    finally:
        sys.path[:] = path_before
    return ProcessTab


def _wired(qapp):
    """The page, a real `RenderQueue`, and the glue's queue methods over them - the card's production path."""
    from sli_ui_toolkit.widgets import TextListWidget

    page = _page(qapp)
    tab_list = TextListWidget(row_height=TAB_ROW_HEIGHT, radius=10)
    tab_list.resize(320, 240)
    tab_list.show()
    queue = RenderQueue(tab_list)
    host, queue_page = _host(qapp)
    tab = object.__new__(_process_tab())
    tab.parent = SimpleNamespace(
        renderQueue=queue,
        shell=SimpleNamespace(procPage_page=SimpleNamespace(generalSettings_page=page)),
    )
    tab.page_handles = SimpleNamespace(
        queue=SimpleNamespace(renderQueueListWidget=tab_list),
        proc=SimpleNamespace(processSettingsContainer=host, renderQueueTab=queue_page),
    )
    tab._queue_tab_index = None
    tab.renderTextOutputList = None
    tab.installQueueRows()
    tab.refreshQueueRows()
    _settle(qapp)
    return SimpleNamespace(page=page, queue=queue, tab=tab, host=host, queue_page=queue_page, tab_list=tab_list)


# --- 1. the empty state ---------------------------------------------------------------------------


def test_an_empty_queue_shows_the_one_line_and_no_list(qapp, language):
    """No items -> one muted catalogue line, and the list (the four-row hole) is hidden.

    Red before: the card does not exist, so the module does not collect (see the module docstring).
    """
    wired = _wired(qapp)
    page = wired.page

    assert page.infoQueueCard.isVisible(), "the card is part of the pane even with nothing queued"
    assert page.infoQueueEmpty.isVisible()
    assert page.infoQueueEmpty.text() == i18n.translate(INFO_QUEUE_EMPTY, i18n.current_language())
    assert not page.infoQueueList.isVisible(), "an empty four-row hole is what the empty line replaces"
    assert page.infoQueueList.count() == 0
    # one line, not a four-row card: the empty card is shorter than the row budget it would need
    assert page.infoQueueCard.height() < INFO_QUEUE_ROWS * page.infoQueueList.row_height()


# --- 2. four rows, and the rest scrolls -----------------------------------------------------------


def test_six_items_scroll_inside_a_four_row_card(qapp, language):
    """Six items: four rows fit exactly, the list scrolls, and the card does not grow.

    Red before: the card does not exist, so the module does not collect (see the module docstring).
    """
    wired = _wired(qapp)
    page, queue, tab = wired.page, wired.queue, wired.tab
    panel = page.infoQueueList

    for index in range(6):
        queue.add(_option(f"/videos/clip_{index}.mkv"))
    tab.refreshQueueRows()
    _settle(qapp)

    row_height = panel.row_height()
    padding = panel.content_height() - panel.count() * row_height
    assert row_height == TAB_ROW_HEIGHT, "the card states a different row height from the tab's list"
    assert panel.count() == 6
    assert panel.isVisible()
    # the row arithmetic, not a pixel: FOUR rows plus the list's own padding is the whole visible height
    assert panel.height() == INFO_QUEUE_ROWS * row_height + padding, (
        f"the list is {panel.height()} px for {INFO_QUEUE_ROWS} x {row_height} px rows + {padding} px padding")
    assert page.queue_preview_scroll_maximum() > 0, "six rows in a four-row list must scroll, not clip"

    card_height_with_six = page.infoQueueCard.height()
    # the card's whole height is that four-row body plus the pin's band and the card's insets - derived, not stored
    margins = page.infoQueueLayout.contentsMargins()
    assert card_height_with_six == (
        panel.height() + page.infoQueuePin.height() + margins.top() + margins.bottom()
        + page.infoQueueLayout.spacing()
    ), "the card's height is not the row arithmetic plus its pin band and insets"

    # four items fit in the whole budget: no range, and the card is exactly as tall as it was with six
    queue.clear()
    for index in range(4):
        queue.add(_option(f"/videos/fits_{index}.mkv"))
    tab.refreshQueueRows()
    _settle(qapp)
    assert page.queue_preview_scroll_maximum() == 0
    assert page.infoQueueCard.height() == card_height_with_six, "more items must not grow the card"


# --- 3. the pin opens the full queue tab ----------------------------------------------------------


def test_the_pin_selects_the_render_queue_tab_by_identity(qapp, language):
    """The pin selects the host's Render Queue page - the test host puts it at index 1, not 2.

    Red before: neither the card nor its pin exists, so the module does not collect.
    """
    wired = _wired(qapp)
    host, queue_page, pin = wired.host, wired.queue_page, wired.page.infoQueuePin

    index = host.indexOf(queue_page)
    assert index == 1, "the test host's own placement: the queue page must not be at 2"
    host.setCurrentIndex(0)
    _settle(qapp)

    # the shipped path: `QConnect` resolved the index once by identity
    wired.tab._queue_tab_index = index
    pin.click()
    _settle(qapp)
    assert host.currentIndex() == index

    # the index may not be cached yet (a host built after the wiring): resolving it by identity must still work
    wired.tab._queue_tab_index = None
    host.setCurrentIndex(0)
    _settle(qapp)
    pin.click()
    _settle(qapp)
    assert host.currentIndex() == host.indexOf(queue_page)


# --- 4. the card is live, not a build-time copy ---------------------------------------------------


def test_an_item_added_after_the_build_reaches_the_card(qapp, language):
    """Add a row AFTER the build: the glue's one refresh call mirrors the model into the card.

    Red before: the card does not exist, so the module does not collect. The assertion that matters is that a
    *post-build* mutation arrives, which a build-time copy of the names could never satisfy.
    """
    wired = _wired(qapp)
    page, queue, tab = wired.page, wired.queue, wired.tab
    assert page.infoQueueList.count() == 0

    queue.add(_option("/videos/late.mkv"))
    tab.refreshQueueRows()  # the call a state change already makes - nothing card-specific added by the caller
    _settle(qapp)

    assert page.infoQueueList.count() == 1
    assert page.infoQueueList.itemText(0) == "/videos/late.mkv"
    assert page.infoQueueList.itemText(0) == wired.tab_list.itemText(0), "the two views disagree about the rows"
    assert not page.infoQueueEmpty.isVisible()
    # the SAME row renderer and tooltip, one factory call for both views
    assert page.infoQueueList.row_painter() is wired.tab_list.row_painter(), (
        "the card has its own row painter - that is a second renderer")
    assert page.infoQueueList.row_tooltip(0) == wired.tab_list.row_tooltip(0)

    queue.clear()
    tab.refreshQueueRows()
    _settle(qapp)
    assert page.infoQueueList.count() == 0
    assert page.infoQueueEmpty.isVisible(), "an emptied queue returns to the one line"


# --- 5. the catalogue, without rebuilding the list ------------------------------------------------


def test_the_card_text_follows_the_language_and_keeps_its_list(qapp, language):
    """A language switch re-states BOTH catalogue strings and never replaces the list.

    Red before: the card and the pin (with the catalogue key its tooltip reads) do not exist - no collection.
    """
    wired = _wired(qapp)
    page, queue, tab = wired.page, wired.queue, wired.tab

    # the catalogue really carries two different texts (a missing key would fall back to English)
    assert i18n.translate(INFO_QUEUE_EMPTY, "ru") != i18n.translate(INFO_QUEUE_EMPTY, "en")
    assert i18n.translate(INFO_QUEUE_PIN_TOOLTIP, "ru") != i18n.translate(INFO_QUEUE_PIN_TOOLTIP, "en")

    queue.add(_option("/videos/one.mkv"))
    queue.add(_option("/videos/two.mkv"))
    tab.refreshQueueRows()
    _settle(qapp)
    panel_before = page.infoQueueList
    rows_before = [panel_before.itemText(row) for row in range(panel_before.count())]
    assert "Open the Render Queue tab" in page.infoQueuePin.toolTip()

    language.emit_language_changed("ru")
    _settle(qapp)

    assert i18n.translate(INFO_QUEUE_PIN_TOOLTIP, "ru") in page.infoQueuePin.toolTip()
    assert page.infoQueueList is panel_before, "a language switch rebuilt the list instead of re-stating its text"
    assert [page.infoQueueList.itemText(row) for row in range(page.infoQueueList.count())] == rows_before

    # the empty line follows too, and the same list object survives that switch as well
    queue.clear()
    tab.refreshQueueRows()
    _settle(qapp)
    assert page.infoQueueEmpty.isVisible()
    assert page.infoQueueEmpty.text() == i18n.translate(INFO_QUEUE_EMPTY, "ru")
    language.emit_language_changed("en")
    _settle(qapp)
    assert page.infoQueueEmpty.text() == i18n.translate(INFO_QUEUE_EMPTY, "en")
    assert page.infoQueueList is panel_before
