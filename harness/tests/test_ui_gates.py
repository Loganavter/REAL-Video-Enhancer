"""Pins for `harness/scripts/ui_gates.py`: each rule catches its violation shape, ignores its exemptions, and the
ratchet fails only in the direction that matters (a new finding, never a migrated one).

The checks are AST walks over source text, so every case here is a small synthetic module written to a temp directory
and handed to the check directly - no Qt, no page build, no display. That is also what makes the gate usable on the
portable interpreter, which is where this suite runs.

The last group pins the ratchet against THIS tree: `KNOWN` must equal what the checks report today, in both
directions. If a page migrates and loses a pinned pixel, or its lines move, that test fails and names the stale
entries - the ratchet is meant to shrink, and only a human knows whether the literal went away or the line moved.
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
    assert (finding.line, finding.fingerprint().endswith(":5")) == (5, True)
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


# --- the ratchet -----------------------------------------------------------------------------------------

def test_the_module_does_not_import_qt() -> None:
    """The suite runs on the portable interpreter (no PySide6), so the gates are source text only."""
    source = Path(ui_gates.__file__).read_text(encoding="utf-8")
    imports = [line.strip() for line in source.splitlines()
               if line.startswith(("import ", "from ")) and "PySide6" in line]
    assert imports == [], imports


@pytest.mark.parametrize("rule", sorted(ui_gates.SCOPES))
def test_ratchet_matches_the_tree(rule: str) -> None:
    """`KNOWN` is today's count, in both directions.

    Extra here means an unrecorded violation (the gate would have stayed silent); missing means a page migrated or a
    line moved. Either way the fix is the same edit, and the message says which.
    """
    current = {f.fingerprint() for f in ui_gates.CHECKS[rule](ui_gates.SCOPES[rule])}
    known = set(ui_gates.KNOWN[rule])
    assert current - known == set(), (
        f"{rule}: {len(current - known)} unrecorded violation(s) - retire the rule or the literal, do not add them "
        f"to KNOWN: {sorted(current - known)}")
    assert known - current == set(), (
        f"{rule}: {len(known - current)} KNOWN entr(y/ies) no longer reproduced - a page migrated or a line moved; "
        f"delete them so the ratchet shrinks: {sorted(known - current)}")


def test_every_known_rule_has_a_check() -> None:
    assert set(ui_gates.KNOWN) == set(ui_gates.SCOPES) == set(ui_gates.CHECKS)


def test_ratchet_lets_a_known_finding_through_and_fails_a_new_one(tmp_path: Path) -> None:
    """The ratchet's one job: today's debt does not fail the build, tomorrow's violation does."""
    known_rule = next(rule for rule, fingerprints in ui_gates.KNOWN.items() if fingerprints)
    fingerprint = sorted(ui_gates.KNOWN[known_rule])[0]
    path, line = fingerprint.rsplit(":", 1)
    recorded = ui_gates.Finding(path, int(line), known_rule, "recorded")
    assert ui_gates.new_findings([recorded]) == []

    fresh = _raw_px(tmp_path, "def build(self):\n    self.w.setFixedWidth(40)\n")
    assert fresh, "the synthetic violation was not reported at all"
    assert ui_gates.new_findings(fresh) == fresh


def test_stale_entries_are_named(tmp_path: Path) -> None:
    stale = ui_gates.stale_known([])
    assert len(stale) == sum(len(v) for v in ui_gates.KNOWN.values())
    assert all(rule in ui_gates.SCOPES for rule, _ in stale)


def test_cli_is_clean_on_this_tree(capsys: pytest.CaptureFixture[str]) -> None:
    assert ui_gates.main() == 0
    out = capsys.readouterr().out
    assert "0 new" in out
    assert "no new violations" in out


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
