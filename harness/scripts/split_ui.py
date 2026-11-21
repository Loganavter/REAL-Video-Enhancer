"""Split the Designer-generated `mainwindow.py` into page modules - the mechanical half of W3.

Why split rather than regenerate from XML: `mainwindow.py` already *is* the XML compiled into Python, by Qt's own
tool. Re-implementing that compiler (1 095 properties, 12 property types, 17 widget classes) buys nothing; moving the
statements does. The emitted code is the same statements in the same order, only relocated - so the gate can be
brute-force honest: the built page must be the same widget tree as the Designer's own.

How the split works: `setupUi` is a flat sequence (measured: 2 067 statements, one `if` at the top, no loops), so a
page is a contiguous range - from `self.<page> = ...` to the `addWidget`/`addTab` that places it. Pages nest
(`settingsPage` holds `tabWidget`, which holds `tab_2/3/4`), so the ranges are applied recursively; a nested page is
built where it was placed, and its statements are not repeated in the parent.

The one thing that cannot be moved in isolation is a temporary (`font4`, `sizePolicy`) defined before a page's range
and used inside it, so the splitter replays the definitions a page consumes, in order, at the top of its builder.

Usage:
  python3 harness/scripts/split_ui.py --report          # partition + external references, writes nothing
  python3 harness/scripts/split_ui.py --write DIR       # emit one module per page
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys

PAGE_CONTAINERS = {"stackedWidget"}
"""Widget names whose `addWidget` marks a top-level page (the nav stack)."""

#: Qt class -> (adapter class in src/ui/widgets/qt_shaped.py, calls that make the toolkit widget unfit).
#: A widget is migrated only when it uses none of the blocking calls - the toolkit has no tristate box, no checkable
#: push button with RVE's semantics, and `Text` paints no pixmaps. Those stay stock and become toolkit rows.
#:
#: The value controls (W4d): the blocking sets are the Qt API the toolkit has no counterpart for, so a later
#: regeneration keeps such a widget stock instead of building an adapter that would fail at runtime.
TOOLKIT = {
    "QLabel": ("QtLabel", {"setPixmap", "setScaledContents"}),
    # `QtButton` takes `toggle=True` at construction; `setCheckable(True)` is emitted as that keyword (see
    # `_toggle_call`). `setAutoExclusive`/`setAutoRepeat` have no toolkit counterpart, so such buttons stay stock.
    "QPushButton": ("QtButton", {"setAutoExclusive", "setAutoRepeat"}),
    "QCheckBox": ("QtCheckBox", {"setTristate"}),
    "QComboBox": ("QtComboBox", set()),
    "QLineEdit": ("QtLineEdit", set()),
    # `QtSpinBox`/`QtDoubleSpinBox` carry setRange/setMinimum/setMaximum/setValue/value/text/textChanged and map
    # ButtonSymbols onto the stepper's visibility; prefix/suffix/special-value text/wrapping have no counterpart.
    "QSpinBox": ("QtSpinBox", {
        "setPrefix", "setSuffix", "setSpecialValueText", "setWrapping", "setAccelerated",
        "setGroupSeparatorShown", "setCorrectionMode", "setInvertedControls", "setKeyboardTracking",
    }),
    "QDoubleSpinBox": ("QtDoubleSpinBox", {
        "setPrefix", "setSuffix", "setSpecialValueText", "setWrapping", "setAccelerated",
        "setGroupSeparatorShown", "setCorrectionMode", "setInvertedControls", "setKeyboardTracking",
    }),
    # `QtProgressBar` paints a token track/chunk and has setRange/setMinimum/setMaximum/setValue/value; Qt's
    # label/format/orientation switches have no counterpart (the toolkit bar paints no text).
    # A log/read-only pane is the toolkit's painted console; it carries setPlainText/setHtml/verticalScrollBar. Qt's
    # editor-only calls have no counterpart, so a pane that uses them stays stock.
    # A queue of names is the compact flat list, not the navigation list (`IconListWidget` cost a migrated page 3.2 %).
    # Qt's list API that the flat list does not carry keeps such a list stock.
    "QListWidget": ("TextListWidget", {
        "setSelectionMode", "setDragDropMode", "setDragEnabled", "setAcceptDrops", "setDefaultDropAction",
        "setEditTriggers", "setViewMode", "setIconSize", "setUniformItemSizes", "setAlternatingRowColors",
        "setWordWrap", "setSpacing", "setVerticalScrollBarPolicy", "setHorizontalScrollBarPolicy", "setItemWidget",
    }),
    "QTextEdit": ("ConsoleView", {
        "setLineWrapMode",
        "setAcceptRichText", "setUndoRedoEnabled", "setTextInteractionFlags", "setPlaceholderText",
        "setTabStopDistance", "setWordWrapMode", "setCurrentCharFormat", "mergeCurrentCharFormat",
    }),
    "QProgressBar": ("QtProgressBar", {
        "setFormat", "setInvertedAppearance", "setTextDirection", "setOrientation", "setAlignment", "reset",
    }),
}



class Stmt:
    __slots__ = ("index", "node", "text", "lineno")

    def __init__(self, index: int, node: ast.stmt, text: str) -> None:
        self.index = index
        self.node = node
        self.text = text
        self.lineno = node.lineno

    def __repr__(self) -> str:
        return f"Stmt({self.index}, {self.lineno}, {self.text.splitlines()[0][:60]!r})"


class Page:
    def __init__(self, name: str, cls: str, create: int, place: int, parent: str, kind: str) -> None:
        self.name = name
        self.cls = cls
        self.create = create
        self.place = place
        self.parent = parent
        self.kind = kind
        self.children: list[Page] = []
        self.own: list[int] = []
        self.retranslate: list[int] = []

    @property
    def title(self) -> str:
        if "_" in self.name:
            return "".join(part.capitalize() for part in re.split(r"[_\W]+", self.name) if part)
        return self.name[:1].upper() + self.name[1:]


def _segment(lines: list[str], node: ast.stmt) -> str:
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def _class_and_name(text: str) -> tuple[str, str] | None:
    match = re.match(r"\s*self\.(\w+) = (\w+)\(", text)
    return (match.group(2), match.group(1)) if match else None


def _placement(text: str) -> tuple[str, str, str, str] | None:
    """`self.<parent>.addWidget(self.<name>)` / `addTab(self.<name>, u"…")` -> (kind, parent, name, tail)."""
    match = re.match(r"\s*self\.(\w+)\.addWidget\(self\.(\w+)\)", text)
    if match:
        return ("addWidget", match.group(1), match.group(2), "")
    match = re.match(r"\s*self\.(\w+)\.addTab\(self\.(\w+),(.*)\)\s*$", text)
    if match:
        return ("addTab", match.group(1), match.group(2), match.group(3).strip())
    return None


def _names_used(text: str) -> set[str]:
    return set(re.findall(r"\bself\.(\w+)", text))


def _names_assigned(text: str) -> set[str]:
    assigned = set(re.findall(r"self\.(\w+)\s*=", text))
    for call in re.finditer(r"self\.(\w+)\.add(?:Widget|Item|Layout|Tab)\(\s*(?:self\.(\w+)|(\w+))", text):
        if call.group(2):
            assigned.add(call.group(2))
    return assigned


def load(path: str) -> tuple[list[str], list[Stmt], list[Stmt], dict[str, int]]:
    source = open(path, encoding="utf-8").read()
    lines = source.splitlines()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    functions = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    if "setupUi" not in functions:
        raise SystemExit(f"{path}: no setupUi in {cls.name}")

    def wrap(body: list[ast.stmt]) -> list[Stmt]:
        return [Stmt(index, node, _segment(lines, node)) for index, node in enumerate(body)]

    locals_defined: dict[str, int] = {}
    for index, stmt in enumerate(functions["setupUi"].body):
        match = re.match(r"\s*(\w+) = ", stmt_text(lines, stmt))
        if match and not stmt_text(lines, stmt).lstrip().startswith("self."):
            locals_defined.setdefault(match.group(1), index)
    return lines, wrap(functions["setupUi"].body), wrap(functions["retranslateUi"].body), locals_defined


def stmt_text(lines: list[str], node: ast.stmt) -> str:
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def build_partition(setup: list[Stmt]) -> tuple[list[Page], list[int]]:
    """All pages with their own statement ranges, plus the indices that belong to the shell."""
    created: dict[str, tuple[int, str]] = {}
    placed: dict[str, tuple[int, str, str]] = {}
    for stmt in setup:
        shape = _class_and_name(stmt.text)
        if shape:
            created.setdefault(shape[1], (stmt.index, shape[0]))
        place = _placement(stmt.text)
        if place:
            placed.setdefault(place[2], (stmt.index, place[1], place[0]))

    pages: list[Page] = []
    for name, (place_index, parent, kind) in placed.items():
        if name not in created:
            continue
        is_page = kind == "addTab" or parent in PAGE_CONTAINERS
        if not is_page:
            continue
        pages.append(Page(name, created[name][1], created[name][0], place_index, parent, kind))
    pages.sort(key=lambda page: page.create)

    # nesting: a page's own range is (create, place), minus any page created inside that range
    by_range = {page.name: page for page in pages}
    for page in pages:
        inner = [other for other in pages if page.create < other.create < page.place]
        direct = [other for other in inner if not any(
            candidate is not other and candidate.create < other.create and other.place <= candidate.place
            for candidate in inner)]
        page.children = sorted(direct, key=lambda other: other.create)

    covered: set[int] = set()

    def claim(page: Page) -> None:
        # A child's range is [create, place): the placement statement itself belongs to the PARENT, which emits the
        # child's build call there. Excluding `place` as well was the bug that left nested pages unbuilt.
        child_ranges = [(child.create, child.place) for child in page.children]
        for index in range(page.create, page.place):
            if any(low <= index < high for low, high in child_ranges):
                continue
            page.own.append(index)
            covered.add(index)

    roots = [page for page in pages if page.parent in PAGE_CONTAINERS]
    stack = list(roots)
    while stack:
        page = stack.pop()
        claim(page)
        stack.extend(page.children)

    shell: list[int] = []
    for stmt in setup:
        if stmt.index in covered:
            continue
        place = _placement(stmt.text)
        if place and place[2] in by_range:
            continue                 # emitted by the owning parent, not by the shell
        shell.append(stmt.index)
    return pages, shell


def external_references(page: Page, setup: list[Stmt]) -> list[str]:
    """`self.X` used by a page but neither created in it nor in its descendants - a broken cut."""
    local: set[str] = set()
    for index in page.own:
        local |= _names_assigned(setup[index].text)
    external = set()
    for index in page.own:
        for name in _names_used(setup[index].text):
            if name not in local:
                external.add(name)
    return sorted(external)


def _bare_names(text: str) -> set[str]:
    """Identifiers used as values: `font4`, `sizePolicy7`, `QWidget`. Method names (after a dot) are excluded."""
    return set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)", text))


SKIPPED: dict[str, list[tuple[int, list[str]]]] = {}


def _hoisted_locals(page: Page, setup: list[Stmt], locals_defined: dict[str, int]) -> list[int]:
    """Statements outside the page that define or mutate a temporary the page consumes, in original order.

    `uic` shares temporaries across the whole file (`font4` is defined on the Process page and used on a Settings
    tab), and a temporary usually takes more than one statement (`icon = QIcon()`, then `icon.addFile(...)`). So the
    replay is statement-based, not assignment-based: every line whose subject is a needed temporary and that sits
    outside the page's own range is re-emitted, in file order, before the page's statements.
    """
    skipped = SKIPPED.setdefault(page.name, [])
    own = set(page.own)
    used: set[str] = set()
    owned_widgets: set[str] = set()
    for index in page.own:
        used |= _bare_names(setup[index].text)
        owned_widgets |= _names_assigned(setup[index].text)
    assigned_inside: set[str] = set()
    for index in page.own:
        match = re.match(r"\s*([A-Za-z_]\w*)\s*=", setup[index].text)
        if match:
            assigned_inside.add(match.group(1))
    needed = {name for name in used - assigned_inside if name in locals_defined}
    replay: list[int] = []
    for stmt in setup:
        if stmt.index in own:
            continue
        subject = re.match(r"\s*([A-Za-z_]\w*)\s*(?:=|\.)", stmt.text)
        if not (subject and subject.group(1) in needed):
            continue
        # A hoisted statement must be about the temporary and nothing else. `setHeightForWidth(self.<widget>...)`
        # copies a flag from a widget that may belong to the shell or to a child page that is built later, and
        # replaying it would either move a shell reference into the page or touch an attribute that does not exist yet
        # (both measured). Such lines are skipped and named in the generated module.
        external = set(_names_used(stmt.text))
        if re.search(r"\bMainWindow\b", stmt.text):
            external.add("MainWindow")
        if external:
            skipped.append((stmt.index, sorted(external)))
            continue
        replay.append(stmt.index)
    return replay


def _widget_calls(page: Page, setup: list[Stmt]) -> dict[str, set[str]]:
    """name -> methods called on it inside this page, used to decide which widgets the toolkit can take."""
    calls: dict[str, set[str]] = {}
    for index in page.own:
        for match in re.finditer(r"self\.(\w+)\.(\w+)\s*\(", setup[index].text):
            calls.setdefault(match.group(1), set()).add(match.group(2))
        for match in re.finditer(r"self\.(\w+)\.setCheckable\(\s*True", setup[index].text):
            calls.setdefault(match.group(1), set()).add("setCheckable(True)")
    return calls


#: Widgets that stay stock even though their call set looks migratable - the HOST feeds them an API the replacement
#: lacks, which a page-only scan cannot see. Regenerating without this would silently swap them for the wrong class.
STOCK_WIDGETS: dict[str, str] = {}

#: A `QTextEdit` is one of two widgets and the page cannot say which: a log (console, line-oriented) or a rich-text pane
#: (a plate, wrapping, markdown). The splitter takes the answer from the host's own usage, stated here per widget.
WIDGET_ADAPTERS: dict[str, str] = {
    "renderOutput": "ConsoleView",         # src/ui/ProcessTab.py feeds it setPlainText + a bottom pin
    "changeLogText": "RichTextView",       # HomeTab renders markdown; the pane is taller than its panel
    "systemInfoText": "RichTextView",      # a read-only information plate
    "inputVideoInfoTextEdit": "RichTextView",  # a read-only HTML plate
}


def _toolkit_class(qt_class: str, widget: str, calls: dict[str, set[str]]) -> str | None:
    if widget in STOCK_WIDGETS:
        return None
    if widget in WIDGET_ADAPTERS:
        return WIDGET_ADAPTERS[widget]
    entry = TOOLKIT.get(qt_class)
    if entry is None:
        return None
    adapter, blocking = entry
    used = calls.get(widget, set())
    return None if (used & blocking) else adapter


def _toggle_call(text: str, widget: str, adapter: str, calls: dict[str, set[str]]) -> str:
    """`QtButton(self.parent)` -> `QtButton(self.parent, toggle=True)` for a checkable page button.

    `Button`'s toggle mode is a constructor flag (`Button(toggle=True)`), and `QtButton` cannot turn it on after the
    fact. The splitter sees the Designer's `setCheckable(True)` in the same page, so the flag is stated where the
    widget is built - the only place the toolkit accepts it. A button with auto-exclusive semantics is not here: the
    splitter keeps those stock (`setAutoExclusive` is in the blocking set), because `Button` has no button group.
    """
    if adapter != "QtButton" or "setCheckable(True)" not in calls.get(widget, set()):
        return text
    return re.sub(rf"(self\.{re.escape(widget)} = {adapter}\([^()]*)\)", r"\1, toggle=True)", text, count=1)


def module_for(page: Page, setup: list[Stmt], retranslate: list[Stmt], locals_defined: dict[str, int],
               header: str, toolkit: bool = False) -> str:
    child_at = {child.place: child for child in page.children}
    # The Designer emits its statements at eight spaces - exactly the depth of a method body - and statements that
    # carry multi-line string literals cannot be re-indented without corrupting the literal. So they are emitted
    # verbatim, and only the splitter's own lines are indented to match.
    calls = _widget_calls(page, setup) if toolkit else {}
    substituted: dict[str, str] = {}      # Qt class -> adapter, for the import block
    adapted: dict[str, str] = {}          # widget name -> adapter, so its stylesheet can be dropped
    body: list[str] = []
    for index in _hoisted_locals(page, setup, locals_defined):
        body.append(setup[index].text)
    for index, names in SKIPPED.get(page.name, []):
        body.append(f"        # splitter: skipped {setup[index].text.strip()!r} ({names} live outside this page)")
    for index in page.own:
        if index in child_at:
            child = child_at[index]
            body.append(f"        self.{child.name}_page = {child.title}Page()")
            body.append(f"        self.{child.name} = self.{child.name}_page.build()")
            place = _placement(setup[index].text)
            tail = f", {place[3]}" if place[3] else ""
            body.append(f"        self.{place[1]}.{place[0]}(self.{place[2]}{tail})")
            continue
        text = setup[index].text
        if toolkit:
            # A stylesheet on a widget that became a toolkit control is dead code: the painter ignores QSS. Dropping the
            # statement here is what keeps a regenerated page free of QSS the gate would flag (the hand-migrated pages
            # delete these lines; a regeneration must not put them back).
            sheet = re.match(r"\s*self\.(\w+)\.setStyleSheet\(", text)
            if sheet and sheet.group(1) in adapted:
                body.append(f"        # splitter: dropped the stylesheet of {sheet.group(1)} (a toolkit widget paints "
                            f"itself, QSS would be ignored)")
                continue
            shape = _class_and_name(text)
            if shape:
                qt_class, widget = shape
                adapter = _toolkit_class(qt_class, widget, calls)
                if adapter:
                    # Substitution is per WIDGET, not per module: a page can hold both a `QLabel` that paints a pixmap
                    # (stays stock) and a `QLabel` that paints text (migrated). The adapter is named at the call site
                    # so the module says which widgets are migrated.
                    substituted[adapter] = adapter
                    adapted[widget] = adapter
                    text = re.sub(rf"(self\.{re.escape(widget)} = ){re.escape(qt_class)}\(",
                                  rf"\g<1>{adapter}(", text, count=1)
                    text = _toggle_call(text, widget, adapter, calls)
        body.append(text)
    if page.retranslate:
        body.append("        self.retranslate()")
    body.append(f"        return self.{page.name}")
    retranslate_body = "\n".join(retranslate[index].text for index in page.retranslate)
    retr = (f"\n    def retranslate(self) -> None:\n{retranslate_body}\n" if retranslate_body else "")
    child_imports = "\n".join(
        f"from src.ui.pages.{_snake(child.name)} import {child.title}Page" for child in page.children)
    if child_imports:
        child_imports = "\n" + child_imports
    if substituted:
        child_imports += ("\nfrom src.ui.widgets.qt_shaped import "
                          + ", ".join(sorted(substituted)))
    return f'''"""Page `{page.name}` ({page.cls}), split out of the Designer-generated mainwindow.py.

Generated by harness/scripts/split_ui.py from testRVEInterface.ui's compiled form: the statements are the Designer's
own, in the Designer's order, with `self` now meaning this page object. Do not hand-edit the layout numbers unless the
look is intentionally changing - the structural harness compares this page against the Designer build.
"""

from __future__ import annotations

{header}{child_imports}


class {page.title}Page:
    """`{page.name}` - placed by its parent with `{page.kind}`."""

    page_id = "{page.name}"

    def build(self):
        """Build the widget tree and return the page root."""
{chr(10).join(body)}
{retr}'''


def _rename_window(text: str) -> str:
    """`MainWindow` -> `window`, outside string literals (the window title and translate context are strings)."""
    return re.sub(r'("(?:[^"\\]|\\.)*")|\bMainWindow\b', lambda m: m.group(1) or "window", text)


def shell_module(pages: list[Page], setup: list[Stmt], retranslate: list[Stmt], locals_defined: dict[str, int],
                 header: str) -> str:
    """The shell: the Designer's own non-page statements, the five root pages built where it placed them."""
    roots = sorted((page for page in pages if page.parent in PAGE_CONTAINERS), key=lambda page: page.place)
    shell_indices = [stmt.index for stmt in setup if stmt.index not in
                     {index for page in pages for index in page.own}
                     and stmt.index not in {page.place for page in pages}]
    # the shell's own statements are exactly the ones no page claimed; rebuild the list from the partition instead
    _pages2, shell_indices = build_partition(setup)
    _migrate_page_defaults(_pages2, shell_indices, setup)
    # the shell owns the five nav placements: `build_partition` leaves a page's placement out of every list because
    # the parent emits it, and for a root page the parent is this shell.
    shell_indices = sorted(set(shell_indices) | {page.place for page in roots})
    child_at = {page.place: page for page in roots}
    pseudo = Page("MainWindow", "QMainWindow", -1, len(setup), "", "")
    pseudo.own = list(shell_indices)
    pseudo.children = roots

    body: list[str] = []
    for index in _hoisted_locals(pseudo, setup, locals_defined):
        body.append(_rename_window(setup[index].text))
    for index, names in SKIPPED.get(pseudo.name, []):
        body.append(f"        # splitter: skipped {setup[index].text.strip()!r} ({names} live outside this page)")
    for index in shell_indices:
        if index in child_at:
            child = child_at[index]
            body.append(f"        self.{child.name}_page = {child.title}Page()")
            body.append(f"        self.{child.name} = self.{child.name}_page.build()")
            place = _placement(setup[index].text)
            tail = f", {place[3]}" if place[3] else ""
            body.append(f"        self.{place[1]}.{place[0]}(self.{place[2]}{tail})")
            continue
        text = setup[index].text
        if re.match(r"\s*self\.retranslateUi\(", text):
            continue                      # pages retranslate themselves; the Designer's dispatcher only did this
        body.append(_rename_window(text))
    body.append("        expose(window, self)")

    window_retr = [stmt.text for stmt in retranslate if re.match(r"\s*MainWindow\.", stmt.text)]
    retr = ""
    if window_retr:
        retr = "\n    def retranslate(self, window) -> None:\n" + "\n".join(
            _rename_window(text) for text in window_retr) + "\n"

    imports = "\n".join(f"from src.ui.pages.{_snake(page.name)} import {page.title}Page" for page in roots)
    return f'''"""The window shell: pages built by their own modules, chrome built here.

Generated by harness/scripts/split_ui.py from the Designer's compiled `mainwindow.py`. It carries the Designer's own
non-page statements (central widget, nav rail, window metrics, inline QSS) so the window a page lands in is the window
it was designed for; the five nav pages come from `src/ui/pages/`.

The one deliberate difference: RVE's glue binds widgets as attributes of the window (`self.output_folder_location`,
`self.label_20`, ...), and pages keep theirs as attributes of the page object. `expose()` merges the two worlds once,
recursively, so glue written against `Ui_MainWindow` keeps working unchanged - which is what makes the shell removable
from the glue's point of view.
"""

from __future__ import annotations

{header}

{imports}


def expose(window, page) -> None:
    """Put a page's (and its descendants') widgets on the window - the surface RVE's glue already binds."""
    for name, value in vars(page).items():
        if name.endswith("_page"):
            expose(window, value)
        elif not name.startswith("_"):
            setattr(window, name, value)


class MainWindowShell:
    """Builds the main window's chrome and the five nav pages into a `QMainWindow`."""

    def build(self, window):
{chr(10).join(body)}
{retr}'''


#: The pages are hand-maintained since 2026-11-22: every one of them now builds toolkit widgets directly, with
#: constructor arguments this generator does not emit, and the compatibility layer it imported is deleted. A
#: regeneration would overwrite the migrated pages and re-create the layer, so `--write` refuses unless the caller
#: states that they know (the generator stays for `--report` and as the record of how the pages were produced).
RETIRED = True


def main() -> int:
    ap = argparse.ArgumentParser()
    default_source = "mainwindow.py"
    if not os.path.exists(default_source):
        default_source = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "harness", "fixtures", "designer_mainwindow.py")
    ap.add_argument("--source", default=default_source)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--allow-retired", action="store_true",
                    help="acknowledge that --write overwrites the hand-maintained pages")
    ap.add_argument("--write", default="")
    ap.add_argument("--toolkit", action="store_true",
                    help="migrate the classes whose toolkit counterpart has the Qt API (W4); default is stock Qt")
    args = ap.parse_args()
    if args.write and not args.allow_retired:
        ap.error("--write is retired: since 2026-11-22 the pages build toolkit widgets directly and a regeneration "
                 "would restore the deleted compatibility layer (see RETIRED in this module)")

    lines, setup, retranslate, locals_defined = load(args.source)
    pages, shell = build_partition(setup)

    # retranslate statements belong to the page that owns the widget they touch
    for stmt in retranslate:
        if re.match(r"\s*MainWindow\.", stmt.text):
            continue
        name = re.match(r"\s*self\.(\w+)", stmt.text)
        if not name:
            continue
        owner = _owner_of(name.group(1), pages, setup)
        if owner is not None:
            owner.retranslate.append(stmt.index)

    _migrate_page_defaults(pages, shell, setup)
    external: dict[str, list[str]] = {page.name: external_references(page, setup) for page in pages}

    if args.report or not args.write:
        print(f"{len(pages)} pages, {len(shell)} shell statements, {len(setup)} total")
        for page in pages:
            nested = ", ".join(child.name for child in page.children) or "-"
            print(f"  {page.name:28s} {page.cls:14s} own={len(page.own):4d} retr={len(page.retranslate):3d} "
                  f"children={nested}")
            if external[page.name]:
                print(f"      EXTERNAL: {external[page.name]}")

    if args.write:
        os.makedirs(args.write, exist_ok=True)
        with open(args.source, encoding="utf-8") as handle:
            source = handle.read()
        header = _header(source)
        shell_path = os.path.join(os.path.dirname(args.write.rstrip(os.sep)) or ".", "shell", "window.py")
        os.makedirs(os.path.dirname(shell_path), exist_ok=True)
        with open(shell_path, "w", encoding="utf-8") as handle:
            handle.write(shell_module(pages, setup, retranslate, locals_defined, header))
        print(f"wrote shell -> {shell_path}")
        written = []
        for page in pages:
            path = os.path.join(args.write, f"{_snake(page.name)}.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(module_for(page, setup, retranslate, locals_defined, header, args.toolkit))
            written.append(path)
        print(f"wrote {len(written)} modules into {args.write}")
        for path in written:
            print("  ", path)
    return 0


def _migrate_page_defaults(pages: list[Page], shell: list[int], setup: list[Stmt]) -> None:
    """Give a page the `setCurrentIndex`-style defaults that name one of its own widgets.

    `setupUi` ends with lines like `self.tabWidget.setCurrentIndex(0)` - they live in the shell's range but they belong
    to the page that owns the widget, otherwise a standalone page build starts on the wrong tab and the identity gate
    blames the split for the shell's line.
    """
    owner = {}
    for page in pages:
        for index in page.own:
            for name in _names_assigned(setup[index].text):
                owner.setdefault(name, page)
    keep = []
    for index in shell:
        match = re.match(r"\s*self\.(\w+)\.(\w+)\(", setup[index].text)
        page = owner.get(match.group(1)) if match else None
        if page is None:
            keep.append(index)
            continue
        page.own.append(index)
        page.own.sort()
    shell[:] = keep


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _owner_of(widget: str, pages: list[Page], setup: list[Stmt]) -> Page | None:
    for page in pages:
        for index in page.own:
            if re.search(rf"self\.{re.escape(widget)}\s*=", setup[index].text):
                return page
    return None


def _header(source: str) -> str:
    imports: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segment = ast.get_source_segment(source, node)
            if segment:
                imports.append(segment)
    imports.append("import resources_rc")
    return "\n".join(imports)


if __name__ == "__main__":
    sys.exit(main())
