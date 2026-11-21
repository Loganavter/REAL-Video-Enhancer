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
carry 118 pinned pixels and two QSS rules on widgets that are already toolkit controls. A gate that is red on day one
is a gate nobody reads, so `KNOWN` records today's findings by `file:line` fingerprint per rule and the gate fails
only when a **new** one appears - and equally when a recorded one is gone, so the list cannot rot. The docstring of
the ratchet is the rule: *`KNOWN` only shrinks.* A page that migrates and loses its pinning must drop its entries in
the same change; a page that keeps them keeps its place in the list. Delete an entry without deleting the literal and
the gate names it as new, which is the only direction that can grow a migration.

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

#: rule name -> the source roots it scans, relative to the repo root. Rule 1 skips `shell/`: the shell is not migrated
#: yet, so a QSS rule there is still doing its job on a stock widget.
SCOPES: dict[str, tuple[str, ...]] = {
    "qss-on-adapter": ("src/ui/pages", "src/ui/widgets"),
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

#: rule -> `file:line` fingerprints recorded as today's debt. Generated from the tree at this commit, not invented:
#: `python3 -m harness.scripts.ui_gates` prints the current list. **It only shrinks** - a page that migrates removes its
#: entries; nothing is ever added, because they would not be debt but a new violation.
KNOWN: dict[str, frozenset[str]] = {
    "qss-on-adapter": frozenset(),
    "raw-chrome-px": frozenset({
        "src/ui/pages/advanced_settings.py:109",
        "src/ui/pages/advanced_settings.py:123",
        "src/ui/pages/advanced_settings.py:147",
        "src/ui/pages/advanced_settings.py:202",
        "src/ui/pages/advanced_settings.py:223",
        "src/ui/pages/advanced_settings.py:74",
        "src/ui/pages/download_page.py:107",
        "src/ui/pages/download_page.py:111",
        "src/ui/pages/download_page.py:142",
        "src/ui/pages/download_page.py:153",
        "src/ui/pages/download_page.py:173",
        "src/ui/pages/download_page.py:190",
        "src/ui/pages/download_page.py:194",
        "src/ui/pages/download_page.py:211",
        "src/ui/pages/download_page.py:213",
        "src/ui/pages/download_page.py:219",
        "src/ui/pages/download_page.py:221",
        "src/ui/pages/download_page.py:236",
        "src/ui/pages/download_page.py:238",
        "src/ui/pages/download_page.py:244",
        "src/ui/pages/download_page.py:246",
        "src/ui/pages/download_page.py:263",
        "src/ui/pages/download_page.py:265",
        "src/ui/pages/download_page.py:271",
        "src/ui/pages/download_page.py:273",
        "src/ui/pages/download_page.py:288",
        "src/ui/pages/download_page.py:290",
        "src/ui/pages/download_page.py:296",
        "src/ui/pages/download_page.py:298",
        "src/ui/pages/general_settings.py:104",
        "src/ui/pages/general_settings.py:117",
        "src/ui/pages/general_settings.py:118",
        "src/ui/pages/general_settings.py:124",
        "src/ui/pages/general_settings.py:125",
        "src/ui/pages/general_settings.py:137",
        "src/ui/pages/general_settings.py:138",
        "src/ui/pages/general_settings.py:149",
        "src/ui/pages/general_settings.py:150",
        "src/ui/pages/general_settings.py:157",
        "src/ui/pages/general_settings.py:158",
        "src/ui/pages/general_settings.py:171",
        "src/ui/pages/general_settings.py:172",
        "src/ui/pages/general_settings.py:195",
        "src/ui/pages/general_settings.py:241",
        "src/ui/pages/general_settings.py:242",
        "src/ui/pages/general_settings.py:258",
        "src/ui/pages/general_settings.py:259",
        "src/ui/pages/general_settings.py:280",
        "src/ui/pages/general_settings.py:317",
        "src/ui/pages/general_settings.py:353",
        "src/ui/pages/general_settings.py:392",
        "src/ui/pages/general_settings.py:437",
        "src/ui/pages/general_settings.py:453",
        "src/ui/pages/general_settings.py:461",
        "src/ui/pages/general_settings.py:462",
        "src/ui/pages/general_settings.py:490",
        "src/ui/pages/general_settings.py:520",
        "src/ui/pages/general_settings.py:525",
        "src/ui/pages/general_settings.py:532",
        "src/ui/pages/general_settings.py:542",
        "src/ui/pages/general_settings.py:543",
        "src/ui/pages/home_page.py:63",
        "src/ui/pages/home_page.py:67",
        "src/ui/pages/home_page.py:75",
        "src/ui/pages/home_page.py:79",
        "src/ui/pages/proc_page.py:145",
        "src/ui/pages/proc_page.py:182",
        "src/ui/pages/proc_page.py:197",
        "src/ui/pages/proc_page.py:235",
        "src/ui/pages/proc_page.py:250",
        "src/ui/pages/proc_page.py:255",
        "src/ui/pages/proc_page.py:267",
        "src/ui/pages/proc_page.py:272",
        "src/ui/pages/proc_page.py:283",
        "src/ui/pages/proc_page.py:287",
        "src/ui/pages/render_queue_tab.py:83",
        "src/ui/pages/settings_page.py:52",
        "src/ui/pages/settings_page.py:56",
        "src/ui/pages/tab_2.py:117",
        "src/ui/pages/tab_2.py:143",
        "src/ui/pages/tab_2.py:149",
        "src/ui/pages/tab_2.py:169",
        "src/ui/pages/tab_2.py:58",
        "src/ui/pages/tab_2.py:75",
        "src/ui/pages/tab_2.py:95",
        "src/ui/pages/tab_3.py:101",
        "src/ui/pages/tab_3.py:125",
        "src/ui/pages/tab_3.py:146",
        "src/ui/pages/tab_3.py:163",
        "src/ui/pages/tab_3.py:181",
        "src/ui/pages/tab_3.py:57",
        "src/ui/pages/tab_3.py:77",
        "src/ui/pages/tab_4.py:79",
        "src/ui/pages/tab_6.py:107",
        "src/ui/pages/tab_6.py:108",
        "src/ui/pages/tab_6.py:113",
        "src/ui/pages/tab_6.py:118",
        "src/ui/pages/tab_6.py:125",
        "src/ui/pages/tab_6.py:149",
        "src/ui/pages/tab_6.py:181",
        "src/ui/pages/tab_6.py:229",
        "src/ui/pages/tab_6.py:255",
        "src/ui/pages/tab_6.py:283",
        "src/ui/pages/tab_6.py:309",
        "src/ui/pages/tab_6.py:335",
        "src/ui/pages/tab_6.py:72",
        "src/ui/pages/tab_6.py:90",
        "src/ui/shell/window.py:326",
        "src/ui/shell/window.py:327",
        "src/ui/shell/window.py:331",
        "src/ui/shell/window.py:337",
        "src/ui/shell/window.py:341",
        "src/ui/shell/window.py:357",
        "src/ui/shell/window.py:365",
        "src/ui/shell/window.py:370",
        "src/ui/shell/window.py:372",
        "src/ui/shell/window.py:61",
        "src/ui/shell/window.py:67",
    }),
    "stock-widget": frozenset({
        "src/ui/pages/render_queue_tab.py:108",
        "src/ui/pages/render_queue_tab.py:78",
    }),
    "theme-read-at-build": frozenset(),
}


@dataclass(frozen=True, order=True)
class Finding:
    """One violation: which rule, where, and what the line does. Sorted by (path, line, rule)."""

    path: str
    line: int
    rule: str
    detail: str

    def fingerprint(self) -> str:
        """The ratchet's key: `file:line`.

        Deliberately brittle: an edit above a recorded literal re-keys it, so a merge that only moves lines needs a
        regeneration (`python3 - <<'PY'` recipe in the plan). A stable key would have to count occurrences per
        `(file, rule, detail)` - two identical literals in one file currently share a fingerprint, and a plain set
        would then stop noticing a third one (measured: 118 findings collapsed to 50 keys).
        """
        return f"{self.path}:{self.line}"

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


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

    The gate is a ratchet: `KNOWN` holds today's literals (118 of them at this commit, in 12 modules) and that list
    only shrinks - each page that migrates deletes its entries, and nothing is ever added to it.
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


def new_findings(findings: Iterable[Finding]) -> list[Finding]:
    """The findings the ratchet does not know about - the reason to fail a build."""
    return [f for f in findings if f.fingerprint() not in KNOWN.get(f.rule, frozenset())]


def stale_known(findings: Iterable[Finding]) -> list[tuple[str, str]]:
    """`(rule, fingerprint)` entries in `KNOWN` that nothing reproduces any more: a page migrated, or a line moved.

    Stale is not a new violation, so it does not fail the CLI - but it is the signal the list must shrink, and the
    suite pins it so the list cannot quietly keep dead entries.
    """
    present = {(f.rule, f.fingerprint()) for f in findings}
    return sorted((rule, fingerprint) for rule, fingerprints in KNOWN.items()
                  for fingerprint in fingerprints if (rule, fingerprint) not in present)


def _rekey() -> int:
    """Rewrite `KNOWN` from the tree, refusing if any rule would GROW (a ratchet is a shrink-only record).

    Line-number fingerprints go stale on every edit above a recorded literal, which in an active migration is most
    commits. Regenerating them by hand was a three-step dance with no guard against the one thing that must never
    happen - a new violation entering the baseline. This makes the regeneration explicit and the grow-check automatic.
    """
    import re as _re

    current = {rule: sorted(f.fingerprint() for f in check(SCOPES[rule])) for rule, check in CHECKS.items()}
    grown = {rule: (len(got), len(KNOWN.get(rule, ()))) for rule, got in current.items()
             if len(got) > len(KNOWN.get(rule, ()))}
    if grown:
        for rule, (got, known) in sorted(grown.items()):
            print(f"REFUSED: {rule} would grow {known} -> {got}: retire the violation, do not re-baseline it")
        return 2
    lines = ["KNOWN: dict[str, frozenset[str]] = {"]
    for rule in sorted(current):
        fingerprints = current[rule]
        if not fingerprints:
            lines.append(f'    "{rule}": frozenset(),')
            continue
        lines.append(f'    "{rule}": frozenset({{')
        lines.extend(f'        "{fp}",' for fp in fingerprints)
        lines.append("    }),")
    lines.append("}")
    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    pattern = _re.compile(r"KNOWN: dict\[str, frozenset\[str\]\] = \{.*?\n\}", _re.S)
    if not pattern.search(text):
        print("REFUSED: the KNOWN block is not in the expected shape")
        return 2
    path.write_text(pattern.sub("\n".join(lines), text, count=1), encoding="utf-8")
    print("re-keyed: " + ", ".join(f"{rule} {len(got)}" for rule, got in sorted(current.items())))
    return 0


def main() -> int:
    """Print the table of findings and the ratchet's verdict. 0 = nothing new, 1 = a new violation."""
    if "--rekey" in sys.argv[1:]:
        return _rekey()
    findings = all_findings()
    known = {f for f in findings if f.fingerprint() in KNOWN.get(f.rule, frozenset())}
    fresh = new_findings(findings)
    stale = stale_known(findings)

    width = max((len(f"{f.path}:{f.line}") for f in findings), default=0)
    print(f"{'where'.ljust(width)}  rule                 detail")
    for finding in findings:
        mark = "known  " if finding in known else "NEW    "
        print(f"{f'{finding.path}:{finding.line}'.ljust(width)}  {mark}{finding.rule:<20} {finding.detail}")

    print()
    print(f"{len(findings)} finding(s): {len(fresh)} new, {len(known)} recorded in KNOWN (the ratchet)")
    for rule in SCOPES:
        count = sum(1 for f in findings if f.rule == rule)
        print(f"  {rule:<20} {count:>4}  (KNOWN {len(KNOWN.get(rule, frozenset()))})")
    if stale:
        print()
        print(f"{len(stale)} stale KNOWN entr{'y' if len(stale) == 1 else 'ies'} - a page migrated or a line moved; "
              "delete them so the ratchet keeps shrinking:")
        for rule, fingerprint in stale:
            print(f"  {rule}: {fingerprint}")
    if fresh:
        print()
        print("NEW findings are the failure: retire the rule or the literal, do not add it to KNOWN.")
        return 1
    print()
    print("no new violations; the recorded ones are the pages this migration still has to reach")
    return 0


if __name__ == "__main__":
    sys.exit(main())
