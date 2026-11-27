"""The process page's run readouts live in ONE surface: the video-info pane's run card.

History: the page's chrome used to repeat the run as three `Text` labels under the preview (`ETA: …`, `FPS: …`,
`STATUS: …`), rebuilt by the 100 ms tick in `src/ui/ProcessTab.py::updateProcessTab` and by `setStatus` outside it.
That made the labels a language hazard (the tick's literals rewrote the chrome's Russian keys within one tick,
measured live 2026-11-24) - and then a duplicate, once the pane grew a run card that shows frame / percent / ETA / fps /
status / elapsed from the same state. The labels were removed on 2026-11-26 (`d87bbe98` added the card, this row
deletes the older readouts); this module now pins the removal rather than the strings:

1. **The duplicated chrome is gone.** The page carries no `ETA`/`FPS`/`STATUS` widget - no such attribute, no child
   with such an object name, and no text under the info pane that reads like one of the old lines.
2. **A language switch has nothing left to repaint there** and does not raise. The catalogue's dead prefix keys went
   with the labels; `FPS:` stays, because the info pane's own first row name is that key.
3. **The tick keeps the state and leaves no label.** `updateProcessTab` still writes `currentFrame`/`fps`/`eta`/
   `status`, and `_run_card_values` - the card's feed - returns the same tuple shape and numbers it returned before
   the removal.
4. **`setStatus` still records the phase the card reads**, and writes no widget.

The page is the real one, built and shown offscreen; the tab is the shipped `ProcessTab` with the tick's other pane
work stubbed (the house style of `test_queue_pane.py`), so a failure here is about the readouts, not about a window.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# The app and the backend both own a `src` namespace package; the suite's conftest puts the backend first. `ProcessTab`
# needs the APP's `src.constants`, so the app root goes first for this module (same pattern as `test_queue_pane.py` /
# `test_display_wiring.py`).
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them.
try:  # pragma: no cover - the branch taken depends on the interpreter
    import cpuinfo  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda *args, **kwargs: {}
    sys.modules["cpuinfo"] = _cpuinfo
try:  # pragma: no cover
    import pypresence  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _presence = types.ModuleType("pypresence")
    _presence.Presence = object
    sys.modules["pypresence"] = _presence

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui import i18n  # noqa: E402
from src.ui.ProcessTab import ProcessTab  # noqa: E402

# Leave the suite's path order as it was (the app modules are imported and bound now).
sys.path[:] = _ORIGINAL_SYS_PATH

#: The three labels the page used to carry, by the object names `proc_page.py` gave them.
REMOVED = ("ETA", "FPS", "STATUS")

#: The prefixes those labels printed. None may be the start of any text under the info pane.
REMOVED_LINES = ("ETA:", "ETA: ", "FPS:", "FPS: ", "Status: ", "STATUS:")

#: The catalogue keys that existed ONLY for the removed labels (the chrome's `ETA: `/`FPS:`/`STATUS:` spellings and
#: the tick's own `FPS: `/`Status: ` twins). `FPS:` is NOT one of them - the info pane's first row name is that key.
REMOVED_KEYS = ("ETA: ", "STATUS: ", "FPS: ", "Status: ")


@pytest.fixture()
def language(qapp):
    """The real i18n module at 1.0, English to start, everything process-wide restored afterwards."""
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    before = i18n.current_language()
    scale.set_factor(1.0)
    i18n.configure_i18n("en")
    yield i18n
    i18n.emit_language_changed(before)
    i18n.configure_i18n(before)
    scale.set_factor(factor_before)


def _page(qapp):
    """The real process page, shown, with the translator installed on its own root."""
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.pages.proc_page import ProcPagePage

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    page = ProcPagePage()
    root = page.build()
    i18n.install(root)
    root.resize(900, 900)
    root.show()
    for _ in range(6):
        qapp.processEvents()
    return page


def _tab(page) -> ProcessTab:
    """The shipped `ProcessTab` writers over the real page, with the tick's pane work stubbed out.

    `object.__new__` is the house style for the tab-level tests (`test_queue_pane.py`): the methods under test are the
    shipped ones, and nothing else is constructed (no render process, no window). The stubs are the tick's other
    owners - paint, tail, console, queue rows - so a failure here is a readout failure.
    """
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=page)
    # The card's owner is reached through the shell (`_general_settings_page`), exactly as the capture probes do.
    tab.parent = SimpleNamespace(shell=SimpleNamespace(procPage_page=page))
    tab.renderTextOutputList = []
    tab.max_value = 493
    tab.currentFrame = 44
    tab.fps = 42
    tab.eta = "00:01:23"
    tab.status = "Rendering"
    tab._render_started_mono = None
    tab.currentRenderOptions = None
    for name in ("_ensure_live_tail", "_sync_preview_transport",
                 "_refresh_held_progress", "_tick_display", "_tick_second_surface",
                 "_push_preview_state", "_ensure_picture_timer",
                 "syncRenderConsole", "refreshQueueRows"):
        setattr(tab, name, lambda *args, **kwargs: None)
    return tab


# --- 1. the removed chrome is gone, measured on the object, not in its source -------------------------


def test_the_page_carries_no_eta_fps_status_labels(qapp):
    """No handle (typed or built), no child by object name, and no old-prefix text under the info pane."""
    import dataclasses

    from PySide6.QtWidgets import QWidget

    from sli_ui_toolkit.widgets import Text

    from src.ui.pages.handles import ProcPageHandles

    page = _page(qapp)
    typed = {field.name for field in dataclasses.fields(ProcPageHandles)}
    assert typed.isdisjoint(REMOVED), f"the typed surface still names {typed & set(REMOVED)}"
    for name in REMOVED:
        assert not hasattr(page, name), f"the removed readout {name} is back on the page"
        assert page.procPage.findChild(QWidget, name) is None, f"a widget named {name} survived on the page"
    texts = [widget.text() for widget in page.processInfoContainer.findChildren(Text)]
    offenders = [text for text in texts if text.startswith(REMOVED_LINES)]
    assert offenders == [], f"the info pane still prints a removed readout: {offenders}"


# --- 2. nothing left for a language switch to repaint -------------------------------------------------


def test_the_catalogue_no_longer_carries_the_removed_prefix_keys(language):
    """The dead keys went with the labels; the info pane's own `FPS:` row name stays."""
    table = language.TRANSLATIONS["ru"]
    still_there = [key for key in REMOVED_KEYS if key in table]
    assert still_there == [], f"a catalogue key survived its only writer: {still_there}"
    assert language.translate("FPS:", "ru") == "К/с:", "the info pane's row name lost its translation"


def test_a_switch_repaints_no_readout_and_does_not_raise(language, qapp):
    """`retranslate` used to write the three labels; with them gone the switch only touches what remains."""
    page = _page(qapp)
    language.emit_language_changed("ru")
    for _ in range(6):
        qapp.processEvents()
    for name in REMOVED:
        assert not hasattr(page, name), f"a switch rebuilt the removed readout {name}"
    assert page.label_9.text() == "Начало (секунды)", "the chrome's real labels must still follow the switch"


# --- 3. the tick keeps the state and feeds the card --------------------------------------------------


def test_the_tick_leaves_no_label_and_still_feeds_the_card(qapp):
    """The numbers the card shows are the tab's state, produced by the same tick that used to paint the labels."""
    page = _page(qapp)
    tab = _tab(page)
    tab._render_started_mono = time.monotonic() - 8.2

    tab.updateProcessTab()

    for name in REMOVED:
        assert not hasattr(page, name), f"the tick rebuilt the removed readout {name}"
    values = tab._run_card_values()
    assert values[:6] == (44, 493, 9, "00:01:23", "42", "Rendering"), values
    # The tick handed that tuple to the card's owner (reached through the shell), unchanged by the removal.
    assert tab._run_card_state == values


# --- 4. the status written outside the tick ----------------------------------------------------------


def test_set_status_records_the_phase_the_card_reads(qapp):
    """`setStatus` runs on pause/resume/completion, not on the 100 ms clock - it is the card's state, not a label."""
    page = _page(qapp)
    tab = _tab(page)
    tab._render_started_mono = time.monotonic() - 1.0

    tab.setStatus("Paused")
    assert tab.status == "Paused"
    assert tab._run_card_values()[5] == "Paused"
    for name in REMOVED:
        assert not hasattr(page, name), f"`setStatus` wrote the removed readout {name}"
