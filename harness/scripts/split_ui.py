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


def module_for(page: Page, setup: list[Stmt], retranslate: list[Stmt], locals_defined: dict[str, int],
               header: str) -> str:
    child_at = {child.place: child for child in page.children}
    # The Designer emits its statements at eight spaces - exactly the depth of a method body - and statements that
    # carry multi-line string literals cannot be re-indented without corrupting the literal. So they are emitted
    # verbatim, and only the splitter's own lines are indented to match.
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
        body.append(setup[index].text)
    if page.retranslate:
        body.append("        self.retranslate()")
    body.append(f"        return self.{page.name}")
    retranslate_body = "\n".join(retranslate[index].text for index in page.retranslate)
    retr = (f"\n    def retranslate(self) -> None:\n{retranslate_body}\n" if retranslate_body else "")
    child_imports = "\n".join(
        f"from src.ui.pages.{_snake(child.name)} import {child.title}Page" for child in page.children)
    if child_imports:
        child_imports = "\n" + child_imports
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mainwindow.py")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--write", default="")
    args = ap.parse_args()

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
        written = []
        for page in pages:
            path = os.path.join(args.write, f"{_snake(page.name)}.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(module_for(page, setup, retranslate, locals_defined, header))
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
