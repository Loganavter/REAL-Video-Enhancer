"""The process page's per-tick readouts in Russian: the labels a switch has to survive the NEXT tick.

The page's chrome (`src/ui/pages/proc_page.py::retranslate`) restates `FPS:`, `ETA: ` and `STATUS:` on a language
switch, so a switch looked done - for up to 100 ms. The tick in `src/ui/ProcessTab.py::updateProcessTab` rebuilds the
same three labels from literals (`f"FPS: …"`, `f"ETA: …"`, `f"Status: …"`) and `setStatus` does it again outside the
tick, so the Russian labels flipped back to English within one tick (measured live in `ru`, 2026-11-24):

    after switch (chrome): 'К/с:' 'Осталось: ' 'СТАТУС:'
    ru tick:               'FPS: 42'   'ETA: 00:01:23' 'Status: Rendering'

What is pinned here, against the real page and the real `i18n` table (no translator mock):

1. **The keys are the tick's own prefixes, byte for byte.** `FPS: ` and `Status: ` are DIFFERENT keys from the
   chrome's `FPS:`/`STATUS:` (the trailing space is part of the key), and both sides must keep working: the fix adds
   keys, it does not rename the ones the chrome ships.
2. **English is unchanged, byte for byte.** The numbers, their formatting and the labels are exactly what 1.0 wrote
   before this row (`FPS: 42`, `ETA: 00:01:23`, `Status: Rendering`).
3. **A switch repaints on the next tick, without a restart.** Same page, same labels; one `updateProcessTab()` after
   `emit_language_changed("ru")` is enough, and switching back reads English again with the same values.
4. **`setStatus` outside the tick follows the switch too** - the idle phase is written when the user pauses, not on
   the render clock, so it cannot be left English by a tick that never comes.

The tick's other work (live tail, display feed, console) is not under test here: the six pane helpers are stubbed so
this file measures the strings and nothing else. The page itself is real, built and shown offscreen.
"""

from __future__ import annotations

import re
import sys
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

#: The tick's three readout prefixes, as the source writes them. The trailing spaces are the keys.
TICK_KEYS = {
    "FPS: ": "К/с: ",
    "ETA: ": "Осталось: ",
    "Status: ": "Статус: ",
}

#: The chrome's own keys (`proc_page.retranslate`) - they must keep their old spellings and translations.
CHROME_KEYS = {
    "FPS:": "К/с:",
    "STATUS:": "СТАТУС:",
}

#: What the English tick writes for this state. The pre-fix body produced these exact strings.
EN_READOUTS = ("FPS: 42", "ETA: 00:01:23", "Status: Rendering")

#: The same three after one tick in `ru` - labels turn, values do not.
RU_READOUTS = ("К/с: 42", "Осталось: 00:01:23", "Статус: Rendering")


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
    owners - paint, console, queue rows - so a failure here is a string failure.
    """
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=page)
    tab.renderTextOutputList = []
    tab.max_value = 100
    tab.currentFrame = 50
    tab.fps = "42"
    tab.eta = "00:01:23"
    tab.status = "Rendering"
    tab._render_started_mono = None
    tab.currentRenderOptions = None
    for name in ("_ensure_live_tail", "_apply_seek_ceiling", "_sync_preview_transport",
                 "_refresh_held_progress", "_tick_display", "_tick_second_surface",
                 "syncRenderConsole", "refreshQueueRows"):
        setattr(tab, name, lambda *args, **kwargs: None)
    return tab


def _readouts(page) -> tuple[str, str, str]:
    return (page.FPS.text(), page.ETA.text(), page.STATUS.text())


# --- the source gate: no readout prefix is rebuilt from a literal again ---------------------------


def test_no_readout_prefix_is_a_literal_in_the_tab_source():
    """The hazard is a literal at the WRITE site, not a key in the table: the tick rebuilt `f"FPS: …"` each 100 ms.

    A key that exists does not help if the writer never asks for it, so this gate reads the shipped module: no
    f-string may open with one of the three prefixes (the catalogue keys themselves are plain arguments to `tr`, not
    f-strings, so they are untouched by this check).
    """
    source = (Path(_ROOT) / "src" / "ui" / "ProcessTab.py").read_text(encoding="utf-8")
    offenders = [match.group(0) for match in re.finditer(r'f"(?:FPS|ETA|Status): ', source)]
    assert not offenders, f"readout literals rebuilt at a write site: {offenders}"
    for key in TICK_KEYS:
        assert f'i18n.tr("{key}")' in source, f"the writer does not ask the catalogue for {key!r}"


# --- the table -----------------------------------------------------------------------------------


def test_the_table_carries_the_tick_prefixes_and_keeps_the_chrome_keys(language):
    """The tick's keys are added beside the chrome's, not instead of them: both labels share these three widgets."""
    table = language.TRANSLATIONS["ru"]
    missing = [key for key in TICK_KEYS if key not in table]
    assert not missing, f"missing from the ru table: {missing}"
    for key, russian in TICK_KEYS.items():
        assert language.translate(key, "ru") == russian
        # The keys are the English source strings, byte for byte - `FPS:` and `FPS: ` are two keys.
        assert language.translate(key, "en") == key
    for key, russian in CHROME_KEYS.items():
        assert language.translate(key, "ru") == russian, f"the chrome key {key!r} was renamed or lost"
    # The two spellings must stay distinguishable, or one widget's key would answer for the other's.
    assert language.translate("FPS: ", "ru") != language.translate("FPS:", "ru")
    assert language.translate("Status: ", "ru") != language.translate("STATUS:", "ru")
    # `ETA: ` is the one prefix the chrome already carried; the tick reuses it rather than adding a twin.
    assert language.translate("ETA: ", "ru") == "Осталось: "


# --- English first: byte-identical to the pre-fix tick -------------------------------------------


def test_the_english_tick_writes_the_strings_1_0_wrote(language, qapp):
    page = _page(qapp)
    tab = _tab(page)
    tab.updateProcessTab()
    assert _readouts(page) == EN_READOUTS


# --- a switch, then one tick ---------------------------------------------------------------------


def test_a_switch_repaints_the_readouts_on_the_next_tick(language, qapp):
    """No restart and no rebuild: the same page's labels read Russian after ONE tick, values intact."""
    page = _page(qapp)
    tab = _tab(page)
    tab.updateProcessTab()
    assert _readouts(page) == EN_READOUTS

    language.emit_language_changed("ru")
    # The switch alone reaches the chrome (the page's own `retranslate`), which writes the value-less labels - the
    # state the old code left on screen for one tick before the literals won.
    assert page.FPS.text() == "К/с:", page.FPS.text()

    tab.updateProcessTab()
    assert _readouts(page) == RU_READOUTS, f"one tick after the switch the page reads {_readouts(page)!r}"

    # A second tick stays Russian (the rewrite is idempotent, not a one-shot) ...
    tab.updateProcessTab()
    assert _readouts(page) == RU_READOUTS

    # ... and switching back is equally one tick, with the same numbers.
    language.emit_language_changed("en")
    tab.updateProcessTab()
    assert _readouts(page) == EN_READOUTS


def test_the_values_are_the_only_thing_the_two_language_ticks_share(language, qapp):
    """Strict form: strip the label from each readout and compare the value halves across the switch."""
    page = _page(qapp)
    tab = _tab(page)
    tab.updateProcessTab()

    # Every spelling a label can carry on screen: the English key plus each language's translation.
    keys = list(TICK_KEYS) + list(CHROME_KEYS)
    spellings = [spelling for key in keys for spelling in (key, language.translate(key, "ru"))]

    def values(readouts):
        halves = []
        for text in readouts:
            label = next((s for s in spellings if text.startswith(s)), None)
            assert label is not None, f"no known label prefixes {text!r}"
            halves.append(text[len(label):])
        return halves

    before = values(_readouts(page))
    language.emit_language_changed("ru")
    tab.updateProcessTab()
    after = values(_readouts(page))
    assert after == before == ["42", "00:01:23", "Rendering"]


# --- the status written outside the tick ---------------------------------------------------------


def test_set_status_follows_the_switch_without_a_tick(language, qapp):
    """`setStatus` runs on pause/resume/construction, not on the 100 ms clock - it must read the catalogue itself."""
    page = _page(qapp)
    tab = _tab(page)
    tab.setStatus("Paused")
    assert page.STATUS.text() == "Status: Paused"

    language.emit_language_changed("ru")
    tab.setStatus("Paused")
    assert page.STATUS.text() == "Статус: Paused"

    language.emit_language_changed("en")
    tab.setStatus("Idle")
    assert page.STATUS.text() == "Status: Idle"
