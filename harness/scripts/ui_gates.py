"""Three static gates over the UI tree, ported from the reference host (Improve-ImgSLI / BlinkTable).

Why this file exists: the migration from the Designer's generated class to code-assembled pages changes *which class*
answers to a widget name (a stock `QLabel` becomes `QtLabel`, a `QPushButton` becomes `QtButton`), and two of RVE's
habits silently stop working when that happens:

* an inline `setStyleSheet` rule that styled the stock widget is a **no-op on a toolkit widget** - a toolkit control is
  a custom `QWidget`, never styled by the widget style, and its painter owns the pixels. Keeping the rule means the
  page looks "slightly off" for a reason nothing in the log names. That is `no manual theming` in
  `tests/contracts/test_no_manual_theming.py` on the reference host, and rule 1 here;
* a pinned pixel (`setMinimumSize(..., 30)`, `QSize(32, 32)`) is the Designer's geometry, not the interface's: the
  toolkit's control metrics are its own and every fight against them cost a measurement cycle. That is
  `test_no_raw_chrome_px.py` there, and rule 2 here - chrome geometry belongs in *design px*, resolved through
  `UiScale` at call time, never at import;
* a theme colour read while the module is being imported is a colour that can never follow a theme change. That is the
  QColor half of `test_no_manual_theming.py`, and rule 3 here.

The three checks are plain AST functions returning `Finding`s ("rule, where, what"), so they run on the portable
interpreter without PySide6, and each has its own source scope (declared in `SCOPES`).

**The ratchet.** Now is the worst moment for all three gates: the pages are the Designer's own statements, so they
carry 88 pinned pixels and two QSS rules on widgets that are already toolkit controls. A gate that is red on day one
is a gate nobody reads, so `KNOWN` records today's findings per rule and the gate fails only when a **new** one
appears - and equally when a recorded one is gone, so the list cannot rot. The docstring of the ratchet is the rule:
*`KNOWN` only shrinks.* A page that migrates and loses its pinning must drop its entries in the same change; a page
that keeps them keeps its place in the list. Delete an entry without deleting the literal and the gate names it as
new, which is the only direction that can grow a migration.

The key is `(path, site)` with a count, not `file:line`: `site` is the rule's own stable text (`setFixedWidth(222)`,
`QtButton addButton: QPushButton{...}`) and the count is how many times one file repeats it. Line numbers are
reported to humans and never used for identity, so an edit that inserts a line above a recorded literal leaves the
ratchet untouched - with the old `file:line` key one blank line at the top of `general_settings.py` re-keyed 29
entries and red two tests.

Usage:

    python3 -m harness.scripts.ui_gates          # table of findings + ratchet status; exit 1 on a new finding
    python3 harness/scripts/ui_gates.py          # same, from a checkout without the package on sys.path
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:          # `python3 harness/scripts/ui_gates.py` puts only scripts/ on the path
    sys.path.insert(0, str(_ROOT))

#: rule name -> the source roots it scans, relative to the repo root. Rule 1 covers the shell too: its nav buttons are
#: toolkit controls now, so a QSS rule there would be dead text exactly like a page's.
SCOPES: dict[str, tuple[str, ...]] = {
    "qss-on-adapter": ("src/ui/pages", "src/ui/shell", "src/ui/widgets"),
    "raw-chrome-px": ("src/ui/pages", "src/ui/shell", "src/ui/widgets"),
    "theme-read-at-build": ("src/ui/pages", "src/ui/shell", "src/ui/widgets"),
    "stock-widget": ("src/ui/pages",),
}

#: The adapters in `src/ui/widgets/qt_shaped.py`. A widget built by one of these paints itself, so a QSS rule aimed at
#: it is dead text; anything else a page builds is still a stock Qt widget and may legitimately be styled.
ADAPTERS = frozenset({"QtLabel", "QtButton", "QtCheckBox", "QtComboBox", "QtLineEdit"})

#: Calls whose arguments are chrome geometry (the reference host's list minus the margin/spacing ones: RVE's pages
#: have not been rebuilt around layout margins yet, so flagging those today would drown the real findings).
SIZE_CALLS = frozenset({
    "setMinimumSize", "setMaximumSize", "setFixedSize",
    "setFixedWidth", "setFixedHeight",
    "setGeometry", "resize",
})

#: Values that are not a pin at all: 0 and 1 are "off"/"one step", 16777215 is `QWIDGETSIZE_MAX` (Qt's "unbounded",
#: which is what a maximum size is set to when a widget should be free to grow).
EXEMPT_PX = frozenset({0, 1, 16777215})

#: Calls that read the live theme. Reading one at import freezes the colour of the theme that happened to be installed
#: (or of no theme at all) into a module-level constant.
THEME_CALLS = frozenset({"get_color", "try_get_color"})

#: rule -> `(path, site)` -> how many times the file repeats it. Generated from the tree at this commit, not invented:
#: `python3 -m harness.scripts.ui_gates` prints the current list. **It only shrinks** - a page that migrates removes
#: its entries; nothing is ever added, because a new entry would not be debt but a new violation.
#:
#: `site` is the rule's own stable text, and the count is what keeps two identical literals apart: a plain set keyed
#: by `(path, site)` would stop noticing a third `QSize(30, 30)` in `download_page.py`, while a count going 10 -> 11
#: is growth. No key carries a line number, so inserting a line above a recorded literal re-keys nothing.
KNOWN: dict[str, dict[tuple[str, str], int]] = {
    "qss-on-adapter": {},
    "raw-chrome-px": {
        ('src/ui/pages/advanced_settings.py', 'setGeometry(460, 516)'): 1,
        ('src/ui/pages/advanced_settings.py', 'setMinimumSize(351, 128)'): 1,
        ('src/ui/pages/download_page.py', 'QSize(30, 30)'): 10,
        ('src/ui/pages/download_page.py', 'setGeometry(928, 302)'): 1,
        ('src/ui/pages/download_page.py', 'setMaximumSize(50, 45)'): 10,
        ('src/ui/pages/general_settings.py', 'QSize(230)'): 1,
        ('src/ui/pages/general_settings.py', 'QSize(32, 32)'): 1,
        ('src/ui/pages/general_settings.py', 'setGeometry(470, 1040)'): 1,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(1000, 30)'): 1,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(10000, 30)'): 1,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(100000, 180)'): 1,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(200, 35)'): 1,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(230, 30)'): 2,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(250, 30)'): 4,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(30)'): 4,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(42344, 200)'): 1,
        ('src/ui/pages/general_settings.py', 'setMaximumSize(75, 30)'): 2,
        ('src/ui/pages/general_settings.py', 'setMinimumSize(190)'): 1,
        ('src/ui/pages/general_settings.py', 'setMinimumSize(30)'): 6,
        ('src/ui/pages/general_settings.py', 'setMinimumSize(75, 30)'): 2,
        ('src/ui/pages/home_page.py', 'QSize(35, 35)'): 2,
        ('src/ui/pages/home_page.py', 'setMaximumSize(55)'): 2,
        ('src/ui/pages/proc_page.py', 'QSize(40, 45)'): 3,
        ('src/ui/pages/proc_page.py', 'setMaximumSize(500)'): 3,
        ('src/ui/pages/proc_page.py', 'setMaximumSize(59, 55)'): 3,
        ('src/ui/pages/render_queue_tab.py', 'setMaximumSize(335)'): 1,
        ('src/ui/pages/settings_page.py', 'setGeometry(1026, 652)'): 1,
        ('src/ui/pages/settings_page.py', 'setMaximumSize(112312)'): 1,
        ('src/ui/pages/tab_2.py', 'setGeometry(968, 528)'): 1,
        ('src/ui/pages/tab_2.py', 'setMinimumSize(25)'): 1,
        ('src/ui/pages/tab_2.py', 'setMinimumSize(75, 30)'): 1,
        ('src/ui/pages/tab_6.py', 'setGeometry(453, 581)'): 1,
        ('src/ui/pages/tab_6.py', 'setGeometry(918, 48)'): 1,
        ('src/ui/pages/tab_6.py', 'setMaximumSize(100)'): 1,
        ('src/ui/pages/tab_6.py', 'setMaximumSize(50)'): 1,
        ('src/ui/pages/tab_6.py', 'setMinimumSize(50)'): 1,
        ('src/ui/pages/tab_6.py', 'setMinimumSize(900, 30)'): 1,
        ('src/ui/shell/window.py', 'QSize(35, 35)'): 4,
        ('src/ui/shell/window.py', 'resize(1163, 700)'): 1,
        ('src/ui/shell/window.py', 'setMaximumSize(55)'): 4,
        ('src/ui/shell/window.py', 'setMinimumSize(1050, 700)'): 1,
        ('src/ui/shell/window.py', 'setMinimumSize(31)'): 1,
    },
    "stock-widget": {},
    "theme-read-at-build": {},
}


@dataclass(frozen=True, order=True)
class Finding:
    """One violation: which rule, where, and what the line does. Sorted by (path, line, rule)."""

    path: str
    line: int
    rule: str
    detail: str

    def key(self) -> tuple[str, str]:
        """The ratchet's key: `(path, site)`, where `site` is the rule's own text for what the line does.

        Line numbers are deliberately absent. Inserting a line above a recorded literal moves nothing that matters -
        the same file still pins the same geometry the same number of times - so the ratchet stays put instead of
        demanding a regeneration. The count `KNOWN` stores alongside the key is what keeps two identical literals
        apart: a plain set keyed by `(path, site)` would stop noticing a third one.
        """
        return (self.path, self.detail)

    @property
    def where(self) -> str:
        """`file:line` - for the table and for a failure message a human has to act on. Never part of the key."""
        return f"{self.path}:{self.line}"

    def __str__(self) -> str:
        return f"{self.where}: {self.rule}: {self.detail}"


# --- plumbing ---------------------------------------------------------------------------------------------

def _display(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (a synthetic file the tests write outside the tree)."""
    try:
        return path.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _python_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    """Every `.py` under each path, in a stable order - a directory is expanded, a file is taken as it comes."""
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = _ROOT / path
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)
        elif path.suffix == ".py":
            yield path


def scope_paths(rule: str) -> list[Path]:
    """The files a rule scans today, from `SCOPES`."""
    return list(_python_files(SCOPES[rule]))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(node: ast.Call) -> str:
    """`f(...)`, `mod.f(...)` and `self.f(...)` all answer with the bare name."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _int_constants(node: ast.AST) -> list[ast.Constant]:
    """Integer literals anywhere inside `node`. `True`/`False` are ints to Python and not geometry, so they are out."""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Constant) and type(n.value) is int]


def _literal_text(node: ast.AST) -> str | None:
    """The concatenated text of a literal string (implicit `"a" "b"` concatenation is one Constant already)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# --- rule 1: no manual theming on a migrated widget --------------------------------------------------------

class _Scope:
    """The constructors one body declares, and the body that encloses it.

    A page class is the unit: `build()` declares `self.button = QtButton(...)` and a later `retranslate()` styles it, so
    methods share their class's names. The module body is the root, which is where the shell and a module-level
    function land. Two pages in one module therefore cannot confuse each other's widget of the same name - which is
    exactly what happens while W4 migrates page by page inside a module that still holds stock siblings.
    """

    def __init__(self, parent: _Scope | None = None) -> None:
        self.parent = parent
        self.names: dict[str, str] = {}

    def record(self, targets: Iterable[ast.expr], value: ast.AST) -> None:
        """Remember `name = Class(...)` for every plain name and `self.<name>` target; later assignments win."""
        if not isinstance(value, ast.Call):
            return
        built = _call_name(value)
        if not built:
            return
        for target in targets:
            if isinstance(target, ast.Attribute):
                self.names[target.attr] = built
            elif isinstance(target, ast.Name):
                self.names[target.id] = built

    def resolve(self, name: str) -> str | None:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None


class _QssScan(ast.NodeVisitor):
    """Every `setStyleSheet` call paired with the scope it sits in, so the target can be resolved after the walk.

    Functions do not open a scope of their own on purpose: the assignment and the styling usually live in different
    methods of the same page (`build()` builds, `retranslate()` restates).
    """

    def __init__(self) -> None:
        self.scope = _Scope()
        self.calls: list[tuple[ast.Call, _Scope]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:            # noqa: N802 - ast spelling
        outer = self.scope
        self.scope = _Scope(outer)
        for statement in node.body:
            self.visit(statement)
        self.scope = outer

    def visit_Assign(self, node: ast.Assign) -> None:               # noqa: N802
        self.scope.record(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:          # noqa: N802
        self.scope.record([node.target], node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:                    # noqa: N802
        self.calls.append((node, self.scope))
        self.generic_visit(node)


def _styled_name(node: ast.Call) -> str:
    """The widget a `x.setStyleSheet(...)` call addresses: `button` for `self.button`, `field` for `field`."""
    target = node.func.value if isinstance(node.func, ast.Attribute) else None
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def check_qss(paths: Iterable[str | Path]) -> list[Finding]:
    """`setStyleSheet` aimed at a widget the toolkit paints: a rule that cannot take effect.

    The target is resolved through the constructor calls in its own class body (or module body), so the check reports
    only a widget it has seen built by an adapter - a rule on a stock container (`QWidget`, `QFrame`, `QScrollArea`,
    `QTabWidget`, `QTextEdit`, `QListWidget`, the spin boxes) is not a finding, because those widgets still paint
    through the style, and a name built somewhere this module cannot see is left alone rather than guessed at.

    An EMPTY stylesheet is not a finding: `setStyleSheet("")` states no rule, so there is nothing to retire, and
    reporting the Designer's resets would bury the rules that actually have to move.
    """
    findings: list[Finding] = []
    for path in _python_files(paths):
        scan = _QssScan()
        scan.visit(_parse(path))
        for node, scope in scan.calls:
            if _call_name(node) != "setStyleSheet" or not node.args:
                continue
            name = _styled_name(node)
            adapter = scope.resolve(name) if name else None
            if adapter not in ADAPTERS:
                continue
            rule = _literal_text(node.args[0])
            if rule is not None and not rule.strip():
                continue
            preview = " ".join((rule or "<computed>").split())
            if len(preview) > 48:
                preview = preview[:45] + "..."
            findings.append(Finding(_display(path), node.lineno, "qss-on-adapter", f"{adapter} {name}: {preview}"))
    return sorted(findings)


# --- rule 2: no raw chrome px -----------------------------------------------------------------------------

def check_raw_px(paths: Iterable[str | Path]) -> list[Finding]:
    """Integer literals pinned into a widget's geometry, where the interface should let content decide.

    Two shapes are scanned: any integer literal inside the arguments of `SIZE_CALLS` (including a `QSize`/`QRect`
    built inline), and a `QSize(<int>, <int>)` anywhere. `EXEMPT_PX` are not pins. A call whose arguments are computed
    - a name, a layout geometry, `fontMetrics().height()` - is not a finding: that is the direction the migration is
    moving in, and the literal is what this gate is for.

    The gate is a ratchet: `KNOWN` holds today's literals (88 of them at this commit, in 10 modules, 42 `(path, site)`
    keys) and that list only shrinks - each page that migrates deletes its entries, and nothing is ever added to it.
    """
    findings: list[Finding] = []
    for path in _python_files(paths):
        tree = _parse(path)
        per_line: dict[int, list[str]] = {}
        wrapped: set[int] = set()          # QSize calls already reported through the size call that wraps them
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in SIZE_CALLS:
                continue
            arguments = list(node.args) + [keyword.value for keyword in node.keywords]
            for argument in node.args:
                if isinstance(argument, ast.Call) and _call_name(argument) == "QSize":
                    wrapped.add(id(argument))
            values = [v for argument in arguments for v in _int_constants(argument) if v.value not in EXEMPT_PX]
            if values:
                line = per_line.setdefault(node.lineno, [])
                line.append(f"{name}({', '.join(str(v.value) for v in values)})")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "QSize" or id(node) in wrapped:
                continue
            if len(node.args) < 2:
                continue
            if not all(isinstance(a, ast.Constant) and type(a.value) is int for a in node.args[:2]):
                continue                    # `QSize(hint)`, `QSize(a.width(), h)` - computed, not pinned
            values = [a for a in node.args[:2] if a.value not in EXEMPT_PX]
            if values:
                per_line.setdefault(node.lineno, []).append(
                    f"QSize({', '.join(str(v.value) for v in values)})")
        for line in sorted(per_line):
            findings.append(Finding(_display(path), line, "raw-chrome-px", "; ".join(per_line[line])))
    return sorted(findings)


# --- rule 3: no theme read at build time ------------------------------------------------------------------

class _ModuleLevelCalls(ast.NodeVisitor):
    """Calls in module body and class bodies - code that runs at import - and never inside a function.

    A class body counts because it runs with the module; a method does not, because it runs when the page is built or
    painted, which is exactly when a theme colour may be read. A nested class keeps counting.
    """

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:        # noqa: N802 - ast spelling
        """A def body runs later; its decorators/defaults are skipped with it, on purpose."""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:   # noqa: N802
        """Same as `visit_FunctionDef`."""

    def visit_Lambda(self, node: ast.Lambda) -> None:                  # noqa: N802
        """A lambda body runs later."""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:              # noqa: N802
        for statement in node.body:
            self.visit(statement)

    def visit_Call(self, node: ast.Call) -> None:                      # noqa: N802
        self.calls.append(node)
        self.generic_visit(node)


def check_theme_reads(paths: Iterable[str | Path]) -> list[Finding]:
    """A theme colour read while the module is imported: `QColor("#...")` or `get_color(...)` outside a function.

    Both freeze a value into the module: a literal hex bypasses the token table entirely (and a theme flip can never
    reach it), and a `get_color` call at import runs before or after the theme is installed depending on import order -
    the same order-dependence the toolkit's scrollbar sweep has. Design px constants are ints and are not affected.
    """
    findings: list[Finding] = []
    for path in _python_files(paths):
        tree = _parse(path)
        visitor = _ModuleLevelCalls()
        for statement in tree.body:
            visitor.visit(statement)
        for call in visitor.calls:
            name = _call_name(call)
            if name in THEME_CALLS:
                argument = _literal_text(call.args[0]) if call.args else None
                findings.append(Finding(_display(path), call.lineno, "theme-read-at-build",
                                         f"{name}({(argument or '...')!r}) at module level"))
                continue
            if name != "QColor":
                continue
            literal = next((t for t in (_literal_text(a) for a in call.args) if t and t.startswith("#")), None)
            if literal is not None:
                findings.append(Finding(_display(path), call.lineno, "theme-read-at-build",
                                         f"QColor({literal!r}) at module level"))
    return sorted(findings)


# --- the ratchet ------------------------------------------------------------------------------------------

#: Widget classes the toolkit provides a counterpart for. A page that constructs one of these directly is a page the
#: migration has not reached yet - and the host's law is "no stock Q* widget except what the toolkit gives".
STOCK_WIDGETS = {
    "QLabel": "Text", "QPushButton": "Button", "QCheckBox": "CheckBox", "QComboBox": "ComboBox",
    "QLineEdit": "CustomLineEdit", "QSpinBox": "SpinBox", "QDoubleSpinBox": "DoubleSpinBox",
    "QProgressBar": "ProgressBar/ToastProgressBar", "QListWidget": "IconListWidget",
    "QTextEdit": "Text(mode='document')/ConsoleView", "QScrollArea": "SurfaceScrollArea/OverlayScrollArea",
    "QTabWidget": "TopTabHost", "QSlider": "Slider", "QRadioButton": "RadioButton",
}


def check_stock_widgets(paths: Iterable[str | Path]) -> list[Finding]:
    """A page module that builds a STOCK widget class instead of its toolkit counterpart.

    The counters in `harness/tests/test_page_primitives.py` pin what a page builds today; this check is the ratchet
    above them: the count of stock constructions per class may only go down. It is the enforcement of the rule "no QSS
    and no Q* widget except what the toolkit provides" - without it, a regeneration quietly reintroduces `QLabel`.
    """
    findings: list[Finding] = []
    for path in _python_files(paths):
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name in STOCK_WIDGETS:
                findings.append(Finding(_display(path), getattr(node, "lineno", 0), "stock-widget",
                                        f"{name} (toolkit: {STOCK_WIDGETS[name]})"))
    return findings


CHECKS = {
    "qss-on-adapter": check_qss,
    "raw-chrome-px": check_raw_px,
    "theme-read-at-build": check_theme_reads,
    "stock-widget": check_stock_widgets,
}


def all_findings() -> list[Finding]:
    """Every finding from every rule over its declared scope."""
    findings: list[Finding] = []
    for rule, check in CHECKS.items():
        findings.extend(check(SCOPES[rule]))
    return sorted(findings)


def counts(findings: Iterable[Finding]) -> dict[tuple[str, str], int]:
    """How many times each `(path, site)` occurs - the value side of the ratchet."""
    tally: dict[tuple[str, str], int] = {}
    for finding in findings:
        tally[finding.key()] = tally.get(finding.key(), 0) + 1
    return tally


def _allocation(findings: Iterable[Finding]) -> list[tuple[Finding, bool]]:
    """Every finding paired with whether the ratchet's recorded count already covers it.

    The recorded count is spent in sorted order (lowest line first), so the verdict is deterministic and the
    `file:line` a failure names is a real one in today's tree, not a reconstruction from the key.
    """
    remaining = {rule: dict(recorded) for rule, recorded in KNOWN.items()}
    rows: list[tuple[Finding, bool]] = []
    for finding in sorted(findings):
        allowance = remaining.get(finding.rule, {})
        left = allowance.get(finding.key(), 0)
        if left:
            allowance[finding.key()] = left - 1
            rows.append((finding, True))
        else:
            rows.append((finding, False))
    return rows


def new_findings(findings: Iterable[Finding]) -> list[Finding]:
    """The findings the ratchet does not cover: a key it never recorded, or one occurrence over its count."""
    return [finding for finding, known in _allocation(findings) if not known]


def stale_known(findings: Iterable[Finding]) -> list[tuple[str, tuple[str, str], int, int]]:
    """`(rule, key, recorded, found)` entries in `KNOWN` the tree no longer fills: a page migrated, or a literal went.

    Short counts are stale exactly like missing keys - the count is the record, so ten `QSize(30, 30)` becoming nine
    is the list owing an entry. Stale is not a new violation, so it does not fail the CLI - but it is the signal the
    list must shrink, and the suite pins it so the list cannot quietly keep dead entries.
    """
    present: dict[str, dict[tuple[str, str], int]] = {}
    for finding in findings:
        per_rule = present.setdefault(finding.rule, {})
        per_rule[finding.key()] = per_rule.get(finding.key(), 0) + 1
    return sorted((rule, key, recorded, present.get(rule, {}).get(key, 0))
                  for rule, entries in KNOWN.items()
                  for key, recorded in entries.items()
                  if present.get(rule, {}).get(key, 0) < recorded)


def _rekey() -> int:
    """Rewrite `KNOWN` from the tree, refusing if ANY key would GROW (a ratchet is a shrink-only record).

    The guard is per key, not per total. Comparing totals let a substitution through: retire one pin, introduce
    another in the same commit, and the count is unchanged while the baseline silently absorbs a new violation. A key
    the baseline never recorded counts as 0 recorded, so its first occurrence is growth like any other.
    """
    import re as _re

    current = {rule: counts(check(SCOPES[rule])) for rule, check in CHECKS.items()}
    grown = [(rule, key, KNOWN.get(rule, {}).get(key, 0), found)
             for rule, tally in current.items()
             for key, found in tally.items()
             if found > KNOWN.get(rule, {}).get(key, 0)]
    if grown:
        for rule, (path, site), recorded, found in sorted(grown):
            print(f"REFUSED: {rule}: {path}: {site!r} would grow {recorded} -> {found}: "
                  "retire the violation, do not re-baseline it")
        return 2
    lines = ["KNOWN: dict[str, dict[tuple[str, str], int]] = {"]
    for rule in sorted(current):
        tally = current[rule]
        if not tally:
            lines.append(f'    "{rule}": {{}},')
            continue
        lines.append(f'    "{rule}": {{')
        lines.extend(f"        ({path!r}, {site!r}): {found},"
                     for (path, site), found in sorted(tally.items()))
        lines.append("    },")
    lines.append("}")
    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    pattern = _re.compile(r"KNOWN: dict\[str, dict\[tuple\[str, str\], int\]\] = \{.*?\n\}", _re.S)
    if not pattern.search(text):
        print("REFUSED: the KNOWN block is not in the expected shape")
        return 2
    path.write_text(pattern.sub("\n".join(lines), text, count=1), encoding="utf-8")
    print("re-keyed: " + ", ".join(f"{rule} {sum(tally.values())} findings / {len(tally)} keys"
                                   for rule, tally in sorted(current.items())))
    return 0


def main() -> int:
    """Print the table of findings and the ratchet's verdict. 0 = nothing new, 1 = a new violation."""
    if "--rekey" in sys.argv[1:]:
        return _rekey()
    findings = all_findings()
    rows = _allocation(findings)
    fresh = [finding for finding, known in rows if not known]
    stale = stale_known(findings)

    width = max((len(f.where) for f in findings), default=0)
    print(f"{'where'.ljust(width)}  rule                 detail")
    for finding, known in rows:
        mark = "known  " if known else "NEW    "
        print(f"{finding.where.ljust(width)}  {mark}{finding.rule:<20} {finding.detail}")

    print()
    print(f"{len(findings)} finding(s): {len(fresh)} new, "
          f"{sum(1 for _, known in rows if known)} recorded in KNOWN (the ratchet)")
    for rule in SCOPES:
        count = sum(1 for f in findings if f.rule == rule)
        recorded = sum(KNOWN.get(rule, {}).values())
        print(f"  {rule:<20} {count:>4}  (KNOWN {recorded})")
    if stale:
        print()
        print(f"{len(stale)} stale KNOWN entr{'y' if len(stale) == 1 else 'ies'} - a page migrated or a literal went "
              "away; delete them so the ratchet keeps shrinking:")
        for rule, (path, site), recorded, found in stale:
            print(f"  {rule}: {path}: {site!r} (recorded {recorded}, found {found})")
    if fresh:
        print()
        print("NEW findings are the failure: retire the rule or the literal, do not add it to KNOWN.")
        return 1
    print()
    print("no new violations; the recorded ones are the pages this migration still has to reach")
    return 0


if __name__ == "__main__":
    sys.exit(main())
