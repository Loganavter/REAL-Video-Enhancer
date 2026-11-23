"""The typed handle edge between the pages and their glue.

`window.page_handles` (built in `src/ui/pages/handles.py`) is the contract this file pins. The expected handle set
is not copied here: it is read out of the glue's own source (`ProcessTab.py`, `SettingsTab.py`), so a read that
survives a page refactor without a handle fails here, and a handle whose field the page no longer builds fails here
too - the two failure modes a `getattr(self.parent, ..., None)` used to hide.

No Qt is imported: the dataclass module is annotations-only at runtime, and the page modules are parsed as AST.
That keeps this pin runnable on the interpreter that has no PySide6, where the page build tests skip.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from src.ui.pages import handles

ROOT = Path(__file__).resolve().parents[2]
GLUE = (ROOT / "src" / "ui" / "ProcessTab.py", ROOT / "src" / "ui" / "SettingsTab.py")
PAGES = ROOT / "src" / "ui" / "pages"
SHELL = ROOT / "src" / "ui" / "shell" / "window.py"

#: The names the task cut first (reads x modules). A migration that quietly stops reading them, or renames a
#: handle, fails here before the rest of the surface is checked.
PRIORITY_NAMES = {
    "inputFileText",
    "startRenderButton",
    "upscaleModelComboBox",
    "interpolateModelComboBox",
    "backendComboBox",
    "EncoderCommand",
    "pytorch_backend",
    "pytorch_version",
    "outputFileText",
    "frameReuse",
    "celReuse",
}


def _chain(node: ast.AST) -> list[str]:
    """`self.page_handles.general.inputFileText` -> ['self', 'page_handles', 'general', 'inputFileText']."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _glue_reads() -> set[tuple[str, str]]:
    """Every `(group, name)` read through `page_handles`, wherever the receiver object came from."""
    reads: set[tuple[str, str]] = set()
    for path in GLUE:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            chain = _chain(node)
            if "page_handles" not in chain:
                continue
            tail = chain[chain.index("page_handles") + 1:]
            if len(tail) >= 2:
                reads.add((tail[0], tail[1]))
    return reads


def _group_class(group: str):
    """The dataclass behind a `WindowHandles` field, resolved without importing Qt."""
    for field in dataclasses.fields(handles.WindowHandles):
        if field.name != group:
            continue
        annotation = field.type
        name = annotation if isinstance(annotation, str) else annotation.__name__
        return getattr(handles, name)
    raise AssertionError(f"WindowHandles has no group {group!r}; add it in src/ui/pages/handles.py")


def _field_names(cls) -> set[str]:
    return {field.name for field in dataclasses.fields(cls)}


def _constructions() -> dict[str, tuple[Path, ast.Call]]:
    """`{dataclass name: (module, call)}` for every handle dataclass built in pages or the shell.

    Nested constructions count: `MenuHandles(...)` lives inside `WindowHandles(...)`, exactly as the shell
    composes it.
    """
    class_names = {
        name for name in dir(handles)
        if dataclasses.is_dataclass(getattr(handles, name)) and isinstance(getattr(handles, name), type)
    }
    found: dict[str, tuple[Path, ast.Call]] = {}
    modules = list(PAGES.glob("*.py")) + [SHELL]
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in class_names:
                found[node.func.id] = (path, node)
    return found


def _self_assignments(path: Path) -> set[str]:
    """`self.<name> = ...` names in a module, AS A WHOLE (the page class is the only owner)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id == "self":
                    names.add(target.attr)
    return names


# -- the expected set is the glue's reads ------------------------------------------------------------


def test_every_handle_the_glue_reads_resolves() -> None:
    reads = _glue_reads()
    groups = {group for group, _name in reads}
    # The coherence check: this migration covers the two glue modules' entire page surface, not one name in
    # one method. A new group here is fine; an empty set means the glue was reverted to `self.parent.<name>`.
    assert groups == {
        "general", "proc", "queue", "advanced", "encoder", "logs",
        "settings", "output", "render_settings", "app_settings", "download",
    }, f"unexpected handle groups: {sorted(groups)}"

    names = {name for _group, name in reads}
    assert len(names) >= 60, f"only {len(names)} handle names are read - the migration was reverted?"
    assert PRIORITY_NAMES <= names, f"the glue stopped reading: {sorted(PRIORITY_NAMES - names)}"

    missing = [
        f"{group}.{name}"
        for group, name in sorted(reads)
        if name not in _field_names(_group_class(group))
    ]
    assert not missing, (
        "the glue reads handles the dataclass does not declare: " + ", ".join(missing)
        + " - add the field in src/ui/pages/handles.py and the required argument in the page's build()")


def test_every_handle_field_is_required() -> None:
    """No defaults: a page that stops building a widget must fail the handle construction, never hand out None."""
    for field in dataclasses.fields(handles.WindowHandles):
        cls = _group_class(field.name)
        for member in dataclasses.fields(cls):
            assert member.default is dataclasses.MISSING
            assert getattr(member, "default_factory", dataclasses.MISSING) is dataclasses.MISSING


def test_every_handle_group_is_constructed_with_exactly_its_fields() -> None:
    constructions = _constructions()
    for field in dataclasses.fields(handles.WindowHandles):
        cls = _group_class(field.name)
        assert cls.__name__ in constructions, (
            f"{cls.__name__} is declared but never built; construct it where its widgets live")
        path, call = constructions[cls.__name__]
        kwargs = {keyword.arg for keyword in call.keywords}
        declared = _field_names(cls)
        assert kwargs == declared, (
            f"{path.relative_to(ROOT)}: {cls.__name__} is built with {sorted(kwargs)} but declares "
            f"{sorted(declared)} - a missing required field is allowed to fail, a made-up one is not"
        )


def test_page_handle_fields_are_attributes_of_the_page_that_builds_them() -> None:
    """Each handle must point at a `self.<name>` the page actually assigns - not at a window attribute."""
    constructions = _constructions()
    for field in dataclasses.fields(handles.WindowHandles):
        cls = _group_class(field.name)
        path, _call = constructions[cls.__name__]
        if path.parent.name != "pages":
            continue  # MenuHandles lives in the shell, where the nav buttons are the shell's own
        assigned = _self_assignments(path)
        missing = sorted(_field_names(cls) - assigned)
        assert not missing, (
            f"{path.relative_to(ROOT)}: {cls.__name__} handles {missing} but the page never assigns "
            f"`self.<name>` for them")


def test_the_glue_has_no_direct_read_or_getattr_fallback_left() -> None:
    """The defect this task removes: a direct `self.parent.<name>` or a silent `getattr(self.parent, ...)`."""
    names: set[str] = set()
    for field in dataclasses.fields(handles.WindowHandles):
        names |= _field_names(_group_class(field.name))
    offenders: list[str] = []
    for path in GLUE:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in names:
                value = node.value
                if isinstance(value, ast.Attribute) and value.attr == "parent" \
                        and isinstance(value.value, ast.Name) and value.value.id == "self":
                    offenders.append(f"{path.name}:{node.lineno} self.parent.{node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                    and len(node.args) >= 2:
                owner, asked = node.args[0], node.args[1]
                if isinstance(owner, ast.Attribute) and owner.attr == "parent" \
                        and isinstance(owner.value, ast.Name) and owner.value.id == "self" \
                        and isinstance(asked, ast.Constant) and asked.value in names:
                    offenders.append(f"{path.name}:{node.lineno} getattr(self.parent, {asked.value!r}, ...)")
    assert not offenders, "page widgets are still read around the typed edge: " + "; ".join(offenders)


# -- the named page selector -------------------------------------------------------------------------


class _FakeStack:
    """The two calls `PageSelector` makes, without Qt."""

    def __init__(self, count: int = len(handles.PAGE_NAMES)) -> None:
        self._count = count
        self._current = 0

    def count(self) -> int:
        return self._count

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt's spelling
        self._current = index

    def currentIndex(self) -> int:  # noqa: N802 - Qt's spelling
        return self._current


def test_page_selector_maps_a_name_not_an_index() -> None:
    stack = _FakeStack()
    selector = handles.PageSelector(stack=stack)
    assert handles.PAGE_NAMES == ("home", "more", "proc", "settings", "download")
    assert selector.index("proc") == 2
    assert selector.show("settings") == 3
    assert stack.currentIndex() == 3
    assert selector.name_of() == "settings"
    assert selector.name_of(0) == "home"
    with pytest.raises(KeyError, match="unknown page 'nope'"):
        selector.index("nope")


def test_page_selector_refuses_a_stack_it_does_not_name() -> None:
    with pytest.raises(ValueError, match="5 names"):
        handles.PageSelector(stack=_FakeStack(count=4))


def test_page_names_match_the_shell_stack_order() -> None:
    """The names are only useful if they follow the order the shell adds the pages in."""
    tree = ast.parse(SHELL.read_text(encoding="utf-8"))
    order: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "addWidget":
            continue
        owner = node.func.value
        if isinstance(owner, ast.Attribute) and owner.attr == "stackedWidget" and node.args:
            argument = node.args[0]
            order.append(argument.attr if isinstance(argument, ast.Attribute) else ast.unparse(argument))
    assert order == ["homePage", "morePage", "procPage", "settingsPage", "downloadPage"]
    assert dict(zip(handles.PAGE_NAMES, order, strict=True)) == {
        "home": "homePage",
        "more": "morePage",
        "proc": "procPage",
        "settings": "settingsPage",
        "download": "downloadPage",
    }
