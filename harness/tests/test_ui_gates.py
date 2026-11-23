"""Pins for `harness/scripts/ui_gates.py`: each rule catches its violation shape, ignores its exemptions, and the
ratchet fails only in the direction that matters (a new finding, never a migrated one).

The checks are AST walks over source text, so every case here is a small synthetic module written to a temp directory
and handed to the check directly - no Qt, no page build, no display. That is also what makes the gate usable on the
portable interpreter, which is where this suite runs.

The last group pins the ratchet against THIS tree: `KNOWN` must equal what the checks report today, key by key and
count by count, in both directions. If a page migrates and loses a pinned pixel that test fails and names the stale
entries - the ratchet is meant to shrink. A line moving is not a change the ratchet can see: the key is
`(path, site)` where `site` is the rule's own text, so only a literal appearing or going away touches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.scripts import ui_gates


def _write(tmp_path: Path, source: str, name: str = "case.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _qss(tmp_path: Path, source: str) -> list[ui_gates.Finding]:
    return ui_gates.check_qss([_write(tmp_path, source)])


def _raw_px(tmp_path: Path, source: str) -> list[ui_gates.Finding]:
    return ui_gates.check_raw_px([_write(tmp_path, source)])


def _theme(tmp_path: Path, source: str) -> list[ui_gates.Finding]:
    return ui_gates.check_theme_reads([_write(tmp_path, source)])


def _empty_stylesheet(tmp_path: Path, source: str) -> list[ui_gates.Finding]:
    return ui_gates.check_empty_stylesheets([_write(tmp_path, source)])


def _swallow(tmp_path: Path, source: str) -> list[ui_gates.Finding]:
    return ui_gates.check_silent_swallows([_write(tmp_path, source)])


def _backend_ui(tmp_path: Path, source: str) -> list[ui_gates.Finding]:
    return ui_gates.check_backend_imports_ui([_write(tmp_path, source)])


def _page_io(tmp_path: Path, source: str) -> list[ui_gates.Finding]:
    return ui_gates.check_page_process_http([_write(tmp_path, source)])


# --- rule 1: no manual theming on a migrated widget -------------------------------------------------------

def test_qss_on_an_adapter_is_a_finding(tmp_path: Path) -> None:
    findings = _qss(tmp_path, """
class Page:
    def build(self):
        self.addButton = QtButton(self.container)
        self.addButton.setStyleSheet(u"QPushButton{\\ntext-align: center;\\n}")
""")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule == "qss-on-adapter"
    assert (finding.line, finding.where.endswith(":5")) == (5, True)
    # the ratchet key is the file and the text, never the line
    assert finding.key() == (finding.path, finding.detail)
    # the report has to name the widget, or the migrator cannot find what to delete
    assert "QtButton" in finding.detail and "addButton" in finding.detail
    assert "\n" not in str(finding)


def test_qss_on_a_stock_container_is_not_a_finding(tmp_path: Path) -> None:
    """The widgets that are still stock paint through the style, so their rules are doing real work."""
    findings = _qss(tmp_path, """
class Page:
    def build(self):
        self.page = QWidget()
        self.page.setStyleSheet(u"QWidget{background-color:#1f232a}")
        self.frame = QFrame(self.page)
        self.frame.setStyleSheet(u"QFrame{border:0}")
        self.scroll = QScrollArea(self.page)
        self.scroll.setStyleSheet(u"QScrollArea{background:transparent}")
        self.tabs = QTabWidget(self.page)
        self.tabs.setStyleSheet(u"QTabWidget::pane{border:0}")
        self.edit = QTextEdit(self.page)
        self.edit.setStyleSheet(u"QTextEdit:disabled{color:white;}")
        self.items = QListWidget(self.page)
        self.items.setStyleSheet(u"QListWidget{border:0}")
        self.spin = QSpinBox(self.page)
        self.spin.setStyleSheet(u"QSpinBox{border:1px}")
        self.dspin = QDoubleSpinBox(self.page)
        self.dspin.setStyleSheet(u"QDoubleSpinBox{border:1px}")
""")
    assert findings == []


def test_an_empty_stylesheet_reset_is_not_a_finding(tmp_path: Path) -> None:
    """`setStyleSheet("")` states no rule; the Designer writes it on every widget it wants unstyled."""
    findings = _qss(tmp_path, """
class Page:
    def build(self):
        self.button = QtButton(self.container)
        self.button.setStyleSheet(u"")
        self.label = QtLabel(self.container)
        self.label.setStyleSheet(u"   \\n")
""")
    assert findings == []


def test_qss_target_unknown_to_the_module_is_not_guessed(tmp_path: Path) -> None:
    """Only a widget the module is seen building with an adapter counts - a name that arrives some other way is left
    alone, which is what keeps the rule from accusing a stock widget somebody else constructed."""
    findings = _qss(tmp_path, """
class Page:
    def build(self, injected):
        injected.setStyleSheet(u"QPushButton{color:red}")
        self.container = QWidget()
        self.container.setStyleSheet(u"QWidget{color:red}")
""")
    assert findings == []


def test_a_name_is_resolved_inside_its_own_class(tmp_path: Path) -> None:
    """Two pages in one module, both with a `title`: only the one built by an adapter has a dead rule."""
    findings = _qss(tmp_path, """
class Migrated:
    def build(self):
        self.title = QtLabel(self.root)
        self.title.setStyleSheet(u"QLabel{color:red}")

    def retranslate(self):
        self.title.setStyleSheet(u"QLabel{color:blue}")


class Stock:
    def build(self):
        self.title = QLabel(self.root)
        self.title.setStyleSheet(u"QLabel{color:red}")
""")
    assert [finding.line for finding in findings] == [5, 8]
    assert {finding.detail.split(":")[0] for finding in findings} == {"QtLabel title"}


# --- rule 2: no raw chrome px ----------------------------------------------------------------------------

def test_pinned_geometry_is_a_finding(tmp_path: Path) -> None:
    findings = _raw_px(tmp_path, """
class Page:
    def build(self):
        self.button.setFixedWidth(222)
        self.button.setMinimumSize(QSize(351, 128))
        self.scroll.setGeometry(QRect(0, 0, 968, 528))
        icon = QSize(32, 32)
        self.label.setFixedHeight(self.fontMetrics().height())
""")
    details = {finding.line: finding.detail for finding in findings}
    assert details[4] == "setFixedWidth(222)"
    # the QSize inside the size call is reported once, by the call that pins it
    assert details[5] == "setMinimumSize(351, 128)"
    assert details[6] == "setGeometry(968, 528)"      # 0 is exempt, 968x528 are not
    assert details[7] == "QSize(32, 32)"              # a bare QSize pins two numbers
    assert 8 not in details                            # computed arguments are the direction we want


@pytest.mark.parametrize("source", [
    "self.w.setFixedWidth(0)",                                  # "off", not a pin
    "self.w.setFixedHeight(1)",                                 # one step, not a pin
    "self.w.setMaximumSize(QSize(16777215, 16777215))",         # QWIDGETSIZE_MAX = unbounded
    "self.w.setMinimumSize(QSize(width, height))",              # computed
    "self.w.setFixedSize(self.sizeHint())",                     # computed
    "self.w.resize(event.size())",                              # computed
    "self.w.setGeometry(QRect(0, 0, 0, 0))",                    # every argument exempt
])
def test_exempt_and_computed_geometry_is_not_a_finding(tmp_path: Path, source: str) -> None:
    findings = _raw_px(tmp_path, f"def build(self):\n    {source}\n")
    assert findings == [], f"{source} -> {findings}"


def test_every_size_call_in_the_contract_is_scanned(tmp_path: Path) -> None:
    """The list is the contract with the reference host: a call added there must be scanned here."""
    assert {"setMinimumSize", "setMaximumSize", "setFixedSize", "setFixedWidth", "setFixedHeight",
            "setGeometry", "resize"} <= set(ui_gates.SIZE_CALLS)
    for name in sorted(ui_gates.SIZE_CALLS - {"setGeometry", "resize"}):
        findings = _raw_px(tmp_path, f"def build(self):\n    self.w.{name}(40)\n")
        assert [f.line for f in findings] == [2], f"{name} not scanned"


# --- rule 3: no theme read at build time -----------------------------------------------------------------

def test_theme_reads_at_import_are_findings(tmp_path: Path) -> None:
    findings = _theme(tmp_path, '''
from PySide6.QtGui import QColor

PAGE_BACKGROUND = QColor("#1f232a")
ACCENT = get_color("accent")


class Palette:
    EDGE = QColor("#2c313c")


def paint_me(self):
    return QColor("#ffffff"), try_get_color("dialog.text")
''')
    lines = sorted(f.line for f in findings)
    assert lines == [4, 5, 9], [str(f) for f in findings]
    assert "QColor('#1f232a')" in findings[0].detail
    assert "get_color(" in findings[1].detail


def test_theme_reads_inside_a_function_are_the_whole_point(tmp_path: Path) -> None:
    """A colour resolved while painting follows a theme flip; one resolved at import cannot."""
    findings = _theme(tmp_path, '''
from PySide6.QtGui import QColor

TITLE_PX = 16          # a design px constant is not a colour and is fine at module level


def build(self):
    background = QColor("#1f232a")
    edge = try_get_color("dialog.border")


def connect(self):
    self.w.clicked.connect(lambda: QColor("#ffffff"))
''')
    assert findings == []


def test_a_non_hex_qcolor_at_import_is_left_to_the_literal_colour_gate(tmp_path: Path) -> None:
    """`QColor(255, 255, 255)` is a literal colour question (the reference host's allow-list gate), not a build-time
    theme read: this rule names the `"#..."` form so the two gates do not report the same line twice."""
    findings = _theme(tmp_path, 'WHITE = QColor(255, 255, 255)\n')
    assert findings == []


# --- rule 4: an empty stylesheet is dead text -------------------------------------------------------------

def test_an_empty_stylesheet_is_a_finding(tmp_path: Path) -> None:
    """`setStyleSheet("")` states no rule, so the call can never change a pixel - and it reads like theming.

    Both quote styles and a whitespace-only literal are the same reset. The report names the widget, because the fix
    is to delete one line at one widget, and the key stays `(file, that text)` like every other rule's.
    """
    findings = _empty_stylesheet(tmp_path, """
class Page:
    def build(self):
        self.panel = QWidget(self.root)
        self.panel.setStyleSheet(u"")
        self.label = QLabel(self.root)
        self.label.setStyleSheet('')
        self.hint = QLabel(self.root)
        self.hint.setStyleSheet(u"   \\n")
""")
    assert [finding.line for finding in findings] == [5, 7, 9]
    assert {finding.rule for finding in findings} == {"empty-stylesheet"}
    details = {finding.line: finding.detail for finding in findings}
    assert details[5] == 'panel.setStyleSheet("")'
    assert details[7] == 'label.setStyleSheet("")'


@pytest.mark.parametrize("source", [
    'self.w.setStyleSheet(u"QWidget{background-color:#1f232a}")',   # a rule: the QSS/adapter question
    "self.w.setStyleSheet(self._rule)",                              # computed: may well be a rule at build time
    "self.w.setStyleSheet(self._rule or '')",                        # computed, empty at runtime only
])
def test_a_stylesheet_that_may_state_a_rule_is_not_dead(tmp_path: Path, source: str) -> None:
    findings = _empty_stylesheet(tmp_path, f"def build(self):\n    {source}\n")
    assert findings == [], f"{source} -> {findings}"


def test_the_two_stylesheet_rules_ask_different_questions(tmp_path: Path) -> None:
    """One call, two rules: the adapter rule has no rule to retire, the dead-text rule has nothing else to say.

    Pinning both halves keeps a later edit from "simplifying" the empty reset into `qss-on-adapter` (where it would
    accuse every stock widget the Designer reset) or out of the gate entirely (where it stays unread dead text).
    """
    path = _write(tmp_path, """
class Page:
    def build(self):
        self.addButton = QtButton(self.root)
        self.addButton.setStyleSheet(u"")
""")
    assert ui_gates.check_qss([path]) == []
    assert [finding.rule for finding in ui_gates.check_empty_stylesheets([path])] == ["empty-stylesheet"]


# --- rule 5: a swallow that leaves no trace ---------------------------------------------------------------

def test_a_silent_swallow_is_a_finding(tmp_path: Path) -> None:
    """Bare, named and tuple handlers are the same shape; the finding sits on the `except`, where a reader looks."""
    findings = _swallow(tmp_path, """
def teardown(self):
    try:
        self.item.destroy()
    except Exception:
        pass
    try:
        self.thread.wait()
    except (RuntimeError, TypeError):
        pass
    try:
        self.window.close()
    except:
        pass
""")
    assert [finding.line for finding in findings] == [5, 9, 13]
    assert [finding.detail for finding in findings] == [
        "except Exception: pass", "except (RuntimeError, TypeError): pass", "except: pass"]


def test_a_swallow_with_a_reason_or_a_trace_is_not_silent(tmp_path: Path) -> None:
    """The escape hatch is a comment (this swallow is deliberate) or a body that does something (this one is not a
    swallow at all). Both are how the next reader learns why the failure is acceptable."""
    findings = _swallow(tmp_path, '''
def teardown(self):
    try:
        self.item.destroy()
    except RuntimeError:
        pass  # already destroyed when the page tears down twice
    try:
        self.thread.wait()
    except Exception:
        log.warning("thread did not stop")
    try:
        self.thread.quit()
    except Exception:
        raise
''')
    assert findings == []


def test_a_comment_on_the_except_line_is_a_note_too(tmp_path: Path) -> None:
    """Teardown swallows are often one line; refusing a comment that does not sit on the `pass` would flag them."""
    findings = _swallow(tmp_path, """
try:
    widget.close()
except RuntimeError:  # the widget is gone on the second teardown
    pass
""")
    assert findings == []


# --- rule 6: the backend never imports the UI --------------------------------------------------------------

def test_a_backend_module_importing_the_ui_is_a_finding(tmp_path: Path) -> None:
    """Three spellings of the same edge: the package, a submodule, and `from src import ui`."""
    findings = _backend_ui(tmp_path, """
from src.ui.pages.download_page import DownloadPage
import src.ui.widgets.qt_shaped as qt_shaped
from src import ui
""")
    assert [finding.line for finding in findings] == [2, 3, 4]
    assert [finding.detail for finding in findings] == [
        "imports src.ui.pages.download_page", "imports src.ui.widgets.qt_shaped", "imports src.ui"]


def test_the_backend_importing_its_own_tree_is_left_alone(tmp_path: Path) -> None:
    """`backend/src` is the `src` package on the backend's path: its own imports and a relative one are not the UI."""
    findings = _backend_ui(tmp_path, """
from src.utils.Util import log
import src.RenderVideo
from src.utils import GetFFMpeg
from .ui import helpers
""")
    assert findings == []


def test_a_module_whose_name_merely_starts_with_src_ui_is_not_the_ui(tmp_path: Path) -> None:
    """`src.ui_helpers` and `src.uix` are different packages; a prefix test without a dot boundary would flag them."""
    findings = _backend_ui(tmp_path, """
from src.ui_kit import thing
import src.uix
""")
    assert findings == []


# --- rule 7: a page or shell module does not spawn processes or do HTTP -----------------------------------

def test_a_page_that_spawns_or_fetches_is_a_finding(tmp_path: Path) -> None:
    """The import half catches the four sanctioned spellings plus `from os import system`; the call half catches
    `os.system`, which imports only `os` and would otherwise be invisible."""
    findings = _page_io(tmp_path, """
import subprocess
from os import system
from requests import Session
import urllib.request
import httpx


def build(self):
    os.system("ffmpeg -i in.mp4 out.mkv")
    os.popen("ffmpeg -version")
""")
    assert [finding.line for finding in findings] == [2, 3, 4, 5, 6, 10, 11]
    assert [finding.detail for finding in findings] == [
        "imports subprocess (process)", "imports os.system (process)", "imports requests (http)",
        "imports urllib.request (http)", "imports httpx (http)", "os.system(...)", "os.popen(...)"]


@pytest.mark.parametrize("source", [
    "import os",
    "import urllib.parse",                     # string handling, not a request
    "from pathlib import Path",
    "import subprocess_helpers",               # a different module
    "self.os.system('not the os module')",
    "os.path.join('a', 'b')",
])
def test_the_page_io_rule_leaves_the_neighbours_alone(tmp_path: Path, source: str) -> None:
    findings = _page_io(tmp_path, f"def build(self):\n    {source}\n")
    assert findings == [], f"{source} -> {findings}"


def test_the_new_rules_scan_the_layer_they_protect() -> None:
    """The boundaries are part of the contract, not an accident of `rglob`.

    The legacy top-level tab modules stay out (the migration owns their catches and resets wholesale - a ratchet over
    them would chase a rewrite), and `page-process-http` stops at the page/shell layer because the download dialog is
    the one UI module whose job is a request and a child process.
    """
    migrated = {"src/ui/pages", "src/ui/shell", "src/ui/widgets", "src/ui/dialogs"}
    assert set(ui_gates.SCOPES["empty-stylesheet"]) == migrated
    assert set(ui_gates.SCOPES["silent-swallow"]) == migrated
    assert ui_gates.SCOPES["backend-imports-ui"] == ("backend",)
    assert set(ui_gates.SCOPES["page-process-http"]) == {"src/ui/pages", "src/ui/shell"}


# --- the ratchet -----------------------------------------------------------------------------------------

def test_the_module_does_not_import_qt() -> None:
    """The suite runs on the portable interpreter (no PySide6), so the gates are source text only."""
    source = Path(ui_gates.__file__).read_text(encoding="utf-8")
    imports = [line.strip() for line in source.splitlines()
               if line.startswith(("import ", "from ")) and "PySide6" in line]
    assert imports == [], imports


@pytest.mark.parametrize("rule", sorted(ui_gates.SCOPES))
def test_ratchet_matches_the_tree(rule: str) -> None:
    """`KNOWN` is today's count per key, in both directions.

    Over a recorded count (or a key the list never had) means an unrecorded violation - the gate would have stayed
    silent. Under it means a page migrated or a literal went away. Either way the fix is the same edit, and the
    message says which. A line inserted above a literal appears in neither list: the key carries no line number.
    """
    current = ui_gates.counts(ui_gates.CHECKS[rule](ui_gates.SCOPES[rule]))
    known = ui_gates.KNOWN[rule]
    grown = {key: (known.get(key, 0), found) for key, found in current.items() if found > known.get(key, 0)}
    assert grown == {}, (
        f"{rule}: {len(grown)} key(s) over their recorded count - retire the rule or the literal, do not add them "
        f"to KNOWN: {grown}")
    shrunk = {key: (recorded, current.get(key, 0)) for key, recorded in known.items()
              if current.get(key, 0) < recorded}
    assert shrunk == {}, (
        f"{rule}: {len(shrunk)} KNOWN key(s) no longer reproduced - a page migrated or a literal went away; "
        f"delete them so the ratchet shrinks: {shrunk}")


def test_every_known_rule_has_a_check() -> None:
    assert set(ui_gates.KNOWN) == set(ui_gates.SCOPES) == set(ui_gates.CHECKS)


def test_ratchet_lets_a_known_finding_through_and_fails_a_new_one(tmp_path: Path) -> None:
    """The ratchet's one job: today's debt does not fail the build, tomorrow's violation does."""
    known_rule = next(rule for rule, entries in ui_gates.KNOWN.items() if entries)
    path, site = sorted(ui_gates.KNOWN[known_rule])[0]
    recorded = ui_gates.Finding(path, 1234, known_rule, site)     # any line: the key does not carry one
    assert ui_gates.new_findings([recorded]) == []

    fresh = _raw_px(tmp_path, "def build(self):\n    self.w.setFixedWidth(40)\n")
    assert fresh, "the synthetic violation was not reported at all"
    assert ui_gates.new_findings(fresh) == fresh


def test_one_occurrence_over_the_recorded_count_is_new(monkeypatch: pytest.MonkeyPatch) -> None:
    """A count is the record: the same literal one more time in one file is growth, not the same debt again.

    This is what the key's count is for - a plain set keyed by `(path, site)` would call the second one a duplicate.
    """
    monkeypatch.setitem(ui_gates.KNOWN, "fake", {("a.py", "setFixedWidth(40)"): 1})
    findings = [ui_gates.Finding("a.py", 1, "fake", "setFixedWidth(40)"),
                ui_gates.Finding("a.py", 9, "fake", "setFixedWidth(40)")]
    assert [finding.line for finding in ui_gates.new_findings(findings)] == [9]


def test_a_line_shift_does_not_change_the_key(tmp_path: Path) -> None:
    """The tax this ratchet replaces: +1 line at the top of `general_settings.py` re-keyed 29 entries, red two tests,
    and three commits in one day exist only to move fingerprints."""
    body = "class Page:\n    def build(self):\n        self.w.setFixedWidth(222)\n        icon = QSize(32, 32)\n"
    first = _raw_px(tmp_path, body)
    _write(tmp_path, "\n\n\n" + body)          # same file, three blank lines above the literals
    second = ui_gates.check_raw_px([tmp_path / "case.py"])
    assert [finding.line for finding in first] != [finding.line for finding in second]      # the report moved
    assert [finding.key() for finding in first] == [finding.key() for finding in second]    # the ratchet did not


@pytest.mark.parametrize("rule", ["empty-stylesheet", "silent-swallow", "backend-imports-ui", "page-process-http"])
def test_every_new_rule_ratchets_in_both_directions(rule: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per rule: a recorded finding passes, one more of the same fails, and an empty KNOWN covers nothing.

    The last step is the one the two green rules (backend, page IO) depend on: their `KNOWN` is `{}` today, and the
    test proves that `{}` is a refusal, not "this rule is decorative". Run for a synthetic key so the verdict does not
    move when the tree's real counts move.
    """
    recorded = ui_gates.Finding("a.py", 10, rule, "site")
    monkeypatch.setitem(ui_gates.KNOWN, rule, {("a.py", "site"): 1})
    assert ui_gates.new_findings([recorded]) == []
    assert [finding.line for finding in ui_gates.new_findings([recorded, ui_gates.Finding("a.py", 20, rule, "site")])
            ] == [20]
    monkeypatch.setitem(ui_gates.KNOWN, rule, {})
    assert ui_gates.new_findings([recorded]) == [recorded]


def test_stale_entries_are_named() -> None:
    stale = ui_gates.stale_known([])
    assert len(stale) == sum(len(entries) for entries in ui_gates.KNOWN.values())
    assert all(rule in ui_gates.SCOPES for rule, *_ in stale)


def test_a_short_count_is_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ten `QSize(30, 30)` becoming nine is the list owing an entry, even though the key still exists."""
    monkeypatch.setitem(ui_gates.KNOWN, "fake", {("a.py", "QSize(30, 30)"): 10})
    findings = [ui_gates.Finding("a.py", 5, "fake", "QSize(30, 30)")]
    assert [entry for entry in ui_gates.stale_known(findings) if entry[0] == "fake"] == [
        ("fake", ("a.py", "QSize(30, 30)"), 10, 1)]


def test_cli_is_clean_on_this_tree(capsys: pytest.CaptureFixture[str]) -> None:
    assert ui_gates.main() == 0
    out = capsys.readouterr().out
    assert "0 new" in out
    assert "no new violations" in out
    # the four rules added on 2026-11-23 are part of the table, not a second mechanism beside it
    for rule in ("empty-stylesheet", "silent-swallow", "backend-imports-ui", "page-process-http"):
        assert rule in out, rule


def test_cli_table_marks_a_new_finding(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path) -> None:
    """Same tree, one extra file in scope: the CLI prints it as NEW and exits 1 - the failure a migration would see."""
    scratch = tmp_path / "pages"
    scratch.mkdir()
    _write(scratch, "def build(self):\n    self.w.setFixedWidth(40)\n")
    monkeypatch.setitem(ui_gates.SCOPES, "raw-chrome-px", (scratch.as_posix(),))
    assert ui_gates.main() == 1
    out = capsys.readouterr().out
    assert "NEW" in out and "setFixedWidth(40)" in out


# --- the ratchet's regeneration is guarded ----------------------------------------------------------------

def _fake_rekey_env(monkeypatch, tmp_path: Path, findings, known):
    module_copy = tmp_path / "ui_gates_copy.py"
    module_copy.write_text(Path(ui_gates.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(ui_gates, "__file__", str(module_copy))
    monkeypatch.setattr(ui_gates, "CHECKS", {"fake": lambda paths: findings})
    monkeypatch.setattr(ui_gates, "SCOPES", {"fake": ("whatever.py",)})
    monkeypatch.setattr(ui_gates, "KNOWN", known)
    return module_copy


def test_rekey_refuses_a_rule_that_would_grow(monkeypatch, tmp_path):
    findings = [ui_gates.Finding("a.py", 1, "fake", "d")]
    copy = _fake_rekey_env(monkeypatch, tmp_path, findings, {"fake": {}})
    assert ui_gates._rekey() == 2
    # Nothing was written: the baseline still has no entry for the rule.
    assert "'a.py'" not in copy.read_text(encoding="utf-8")


def test_rekey_refuses_a_substitution(monkeypatch, tmp_path):
    """One key retired, another introduced: the total is unchanged, so the old total-only guard passed it silently."""
    findings = [ui_gates.Finding("a.py", 1, "fake", "setFixedWidth(40)"),
                ui_gates.Finding("a.py", 5, "fake", "setFixedWidth(222)")]
    known = {"fake": {("a.py", "setFixedWidth(40)"): 1, ("a.py", "setFixedWidth(999)"): 1}}
    copy = _fake_rekey_env(monkeypatch, tmp_path, findings, known)
    assert len(findings) == sum(known["fake"].values())          # the totals match: this is the hole
    assert ui_gates._rekey() == 2
    assert "'setFixedWidth(222)'" not in copy.read_text(encoding="utf-8")


def test_rekey_rewrites_a_baseline_whose_lines_moved(monkeypatch, tmp_path):
    """The key no longer carries a line, so a shifted literal is not growth - and no line number is persisted."""
    findings = [ui_gates.Finding("a.py", 77, "fake", "d")]
    copy = _fake_rekey_env(monkeypatch, tmp_path, findings, {"fake": {("a.py", "d"): 1}})
    assert ui_gates._rekey() == 0
    text = copy.read_text(encoding="utf-8")
    assert "('a.py', 'd'): 1" in text
    assert "a.py:77" not in text


def test_rekey_drops_entries_that_are_gone(monkeypatch, tmp_path):
    """Shrinking is the one direction regeneration is for: a retired literal leaves the baseline."""
    findings = [ui_gates.Finding("a.py", 2, "fake", "d")]
    known = {"fake": {("a.py", "d"): 1, ("a.py", "retired"): 1}}
    copy = _fake_rekey_env(monkeypatch, tmp_path, findings, known)
    assert ui_gates._rekey() == 0
    text = copy.read_text(encoding="utf-8")
    assert "('a.py', 'd'): 1" in text
    assert "'retired'" not in text
