"""The progress-detail line in Russian: `elapsed`, `left` and `frame` are catalogue keys, not tick literals.

`src/ui/ProcessTab.py::progressDetailText` builds

    22% · 12m3s elapsed · 16m40s left · frame 1234/5678

and the 100 ms tick (`updateProcessTab` -> `setProgressDetail`) writes it into `progressDetail` on every frame. The
previous lane translated `FPS: `, `ETA: ` and `Status: ` at the same write site and left this line out, so in `ru` the
readouts above it read Russian and this line stayed English for the whole render - it is rebuilt faster than any page
`retranslate` pass could reach it.

What is pinned here, against the real page and the real `i18n` table (no translator mock):

1. **The three words are keys, at the write site.** `" elapsed"`, `" left"` and `"frame "` are passed to `i18n.tr` with
   the spaces the literals carried (a leading/trailing space is part of the key), so a switch reaches a line that is
   rebuilt from code rather than from a `retranslate` pass.
2. **English is unchanged, byte for byte.** Same numbers, same order, same ` · ` separator: the string below is what
   the pre-fix body produced for this state.
3. **A switch repaints this line on the NEXT tick** - that is where it is built - with the numbers untouched, and
   switching back is equally one tick.
4. **Idle stays empty.** `progressDetailText` returns `""` before a render; a translated line must not invent content.

The tick's other work (live tail, display feed, console, queue rows) is stubbed exactly as in
`test_process_readouts_language.py`, so a failure here is a string failure.
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
# needs the APP's `src.constants`, so the app root goes first for this module (same pattern as
# `test_process_readouts_language.py`).
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

#: The three fragments, as the source writes them. The spaces are the keys: `" elapsed"`, `" left"`, `"frame "`.
DETAIL_KEYS = {
    " elapsed": " прошло",
    " left": " осталось",
    "frame ": "кадр ",
}

#: The state these tests drive: 1234 of 5678 frames (22%), 12m3s in, 16m40s to go.
MAX_FRAMES = 5678
CURRENT_FRAME = 1234
ELAPSED_S = 723.0          # short_duration -> "12m3s"
START_MONO = 1000.0
NOW_MONO = START_MONO + ELAPSED_S

#: What the English line is for that state - the exact string the pre-fix body wrote.
EN_LINE = "22% · 12m3s elapsed · 16m40s left · frame 1234/5678"

#: The same state after one tick in `ru`: the three words turn, every digit and separator stays.
RU_LINE = "22% · 12m3s прошло · 16m40s осталось · кадр 1234/5678"

#: The fragments as they can appear on screen, per language (for the value-only comparison).
_SPELLINGS = [s for key, ru in DETAIL_KEYS.items() for s in (key, ru)]


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


@pytest.fixture()
def clock(monkeypatch):
    """Pin the render clock: `progressDetailText` imports `monotonic` per call, so the patch is the number."""
    import time

    monkeypatch.setattr(time, "monotonic", lambda: NOW_MONO)


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


def _bare_tab() -> ProcessTab:
    """The shipped `progressDetailText` with no window at all: it reads attributes and the catalogue, nothing else."""
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace()
    return tab


def _tab(page) -> ProcessTab:
    """The shipped `ProcessTab` writers over the real page, with the tick's pane work stubbed out."""
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=page)
    tab.renderTextOutputList = []
    tab.max_value = MAX_FRAMES
    tab.currentFrame = CURRENT_FRAME
    tab.fps = "42"
    tab.eta = "16m40s"
    tab.status = "Rendering"
    tab._render_started_mono = START_MONO
    tab.currentRenderOptions = None
    for name in ("_ensure_live_tail", "_sync_preview_transport",
                 "_refresh_held_progress", "_tick_display", "_tick_second_surface",
                 "syncRenderConsole", "refreshQueueRows"):
        setattr(tab, name, lambda *args, **kwargs: None)
    return tab


# --- the source gate: no fragment is rebuilt from a literal again --------------------------------


def test_no_detail_fragment_is_a_literal_in_the_tab_source():
    """The hazard is the literal at the WRITE site: the tick rebuilt `f"{x} elapsed"` every 100 ms."""
    source = (Path(_ROOT) / "src" / "ui" / "ProcessTab.py").read_text(encoding="utf-8")
    offenders = [match.group(0) for match in re.finditer(r'f"(?:\{[^}]*\} (?:elapsed|left)|frame )', source)]
    assert not offenders, f"detail literals rebuilt at a write site: {offenders}"
    for key in DETAIL_KEYS:
        assert f'i18n.tr("{key}")' in source, f"the writer does not ask the catalogue for {key!r}"


# --- the table -----------------------------------------------------------------------------------


def test_the_table_carries_the_detail_fragments(language):
    """Keys are the English source strings byte for byte, and the spaces are part of them."""
    table = language.TRANSLATIONS["ru"]
    missing = [key for key in DETAIL_KEYS if key not in table]
    assert not missing, f"missing from the ru table: {missing}"
    for key, russian in DETAIL_KEYS.items():
        assert language.translate(key, "ru") == russian
        assert language.translate(key, "en") == key
    # A bare word is NOT the key: `" left"` and `"left"` are different strings, and the write site passes the spaced one.
    for bare in ("elapsed", "left", "frame"):
        assert language.translate(bare, "ru") == bare, f"{bare!r} became a key; the writer passes the spaced fragment"


# --- English first: byte-identical to the pre-fix line -------------------------------------------


def test_the_english_line_is_byte_for_byte_what_1_0_wrote(language, clock, qapp):
    page = _page(qapp)
    tab = _tab(page)
    assert tab.progressDetailText() == EN_LINE
    tab.updateProcessTab()
    assert page.progressDetail.text() == EN_LINE


@pytest.mark.parametrize(
    "max_value, current_frame, eta, elapsed, expected",
    [
        # the shipped example state
        (5678, 1234, "16m40s", 723.0, "22% · 12m3s elapsed · 16m40s left · frame 1234/5678"),
        # no start yet: the percent and the frame counter, and nothing else
        (100, 0, 0, None, "0% · frame 0/100"),
        # a frame count past the range is clamped, not printed
        (100, 250, "2s", 45.0, "100% · 45s elapsed · 2s left · frame 100/100"),
        # a negative counter is clamped to zero
        (100, -5, "2s", 45.0, "0% · 45s elapsed · 2s left · frame 0/100"),
        # an empty range is one frame, never a division by zero
        (0, 0, 0, 5.0, "0% · 5s elapsed · frame 0/1"),
        # 99.9 rounds to 100, and an hours-long render keeps its short form
        (1000, 999, "3s", 7261.0, "100% · 2hr1m elapsed · 3s left · frame 999/1000"),
    ],
)
def test_every_shape_of_the_line_keeps_the_1_0_formatting(language, clock, qapp, max_value, current_frame, eta, elapsed,
                                                           expected):
    """The swap is a string lookup: percent rounding, clamping, `short_duration` and the ` · ` joins do not move."""
    tab = _bare_tab()
    tab.max_value = max_value
    tab.currentFrame = current_frame
    tab.eta = eta
    tab._render_started_mono = None if elapsed is None else NOW_MONO - elapsed
    assert tab.progressDetailText() == expected


# --- a switch, then one tick ---------------------------------------------------------------------


def test_a_switch_repaints_the_detail_line_on_the_next_tick(language, clock, qapp):
    """No restart: the line is BUILT in the tick, so the tick is what makes it Russian."""
    page = _page(qapp)
    tab = _tab(page)
    tab.updateProcessTab()
    assert page.progressDetail.text() == EN_LINE

    language.emit_language_changed("ru")
    # The page's own `retranslate` does not own this line - it is written by the tick, so the switch alone changes
    # nothing here. That is the reason the next tick is the assertion, not a page pass.
    assert page.progressDetail.text() == EN_LINE

    tab.updateProcessTab()
    assert page.progressDetail.text() == RU_LINE, f"one tick after the switch the line reads {page.progressDetail.text()!r}"

    # A second tick stays Russian (the rewrite is idempotent, not a one-shot) ...
    tab.updateProcessTab()
    assert page.progressDetail.text() == RU_LINE

    # ... and switching back is equally one tick, with the same numbers.
    language.emit_language_changed("en")
    tab.updateProcessTab()
    assert page.progressDetail.text() == EN_LINE


def test_only_the_three_words_turn_across_the_switch(language, clock, qapp):
    """Strict form: strip the fragments and the value halves are identical in both languages."""
    page = _page(qapp)
    tab = _tab(page)

    def values(line):
        parts = line.split(" · ")
        assert len(parts) == 4, parts
        halves = [parts[0]]  # the percent carries no word at all
        for part in parts[1:]:
            spelling = next((s for s in _SPELLINGS if s in part), None)
            assert spelling is not None, f"no known fragment in {part!r}"
            halves.append(part.replace(spelling, "", 1))
        return halves

    english = values(tab.progressDetailText())
    assert english == ["22%", "12m3s", "16m40s", "1234/5678"]

    language.emit_language_changed("ru")
    tab.updateProcessTab()
    assert values(page.progressDetail.text()) == english
    # The digits themselves are shared substrings: nothing re-formatted, nothing re-ordered.
    for digit in ("22%", "12m3s", "16m40s", "1234/5678"):
        assert digit in page.progressDetail.text()


# --- idle ----------------------------------------------------------------------------------------


def test_the_idle_line_stays_empty_in_both_languages(language, clock, qapp):
    """Before a render there is nothing to say; the catalogue must not give the line a body."""
    page = _page(qapp)
    tab = _tab(page)
    tab.max_value = 0
    tab.currentFrame = 0
    tab.eta = 0
    tab._render_started_mono = None
    assert tab.progressDetailText() == ""
    language.emit_language_changed("ru")
    tab.updateProcessTab()
    assert page.progressDetail.text() == ""
