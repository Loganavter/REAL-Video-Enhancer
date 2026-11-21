"""The parity pin for W4c: a page may only name a size, never invent one.

`src/ui/typography.py` owns every font size in `src/ui/pages/`. This test reads both sides as
source - the Designer's compiled output (`harness/fixtures/designer_mainwindow.py`, with its nine
`QFont()` temporaries) and every page module - and compares, per widget, the *resolved* spec:
point size, bold, italic, underline and family. A drifted size (or a lost underline, or a
dropped ``Fira Mono``) fails here, which is the gate: changing one page font without changing the
fixture is a regression, not a cleanup.

Everything is AST, so the test runs on the portable interpreter that has no PySide6. The one
runtime check is skipped when Qt is absent.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "src" / "ui" / "pages"
FIXTURE = ROOT / "harness" / "fixtures" / "designer_mainwindow.py"
TYPOGRAPHY = ROOT / "src" / "ui" / "typography.py"

#: The spec of a ``QFont()`` before any setter runs. ``pt=None`` means "no explicit size", which is
#: not the same as 0: the widget then resolves the application default, and two Designer widgets
#: (`renderPreviewBtn`, `low_storage_label`) depend on exactly that.
_DEFAULTS = {"pt": None, "bold": False, "italic": False, "underline": False, "family": None}
_MUTATORS = {
    "setPointSize": "pt",
    "setPointSizeF": "pt",
    "setBold": "bold",
    "setItalic": "italic",
    "setUnderline": "underline",
    "setFamilies": "family",
    "setFamily": "family",
}
_REQUIRED_GROUPS = ("BODY_PT", "GROUP_TITLE_PT", "CAPTION_PT", "SMALL_PT")


def _value(node: ast.AST, constants: dict[str, object]):
    """Resolve a literal, a ``None``, or a ``*_PT`` name from typography.py."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.List):
        return [_value(element, constants) for element in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_value(node.operand, constants)
    raise AssertionError(f"cannot resolve {ast.unparse(node)!r} - typography values must be literal or *_PT")


def _widget_name(node: ast.AST) -> str | None:
    """``self.label_12`` -> ``"label_12"``; anything else is not a widget reference."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _family(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _spec(call: ast.Call, constants: dict[str, object]):
    """A font spec from an ``apply(widget, pt, ...)`` call."""
    spec = dict(_DEFAULTS)
    if len(call.args) > 1:
        spec["pt"] = _value(call.args[1], constants)
    for keyword in call.keywords:
        if keyword.arg not in _DEFAULTS:
            raise AssertionError(f"unknown font option {keyword.arg!r} in {ast.unparse(call)}")
        spec[keyword.arg] = _value(keyword.value, constants)
    spec["family"] = _family(spec["family"])
    return spec


def _ordered(tree: ast.AST):
    """Nodes in source order, so a temp's setter is seen after its assignment."""
    return sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))


def _collect(path: Path, constants: dict[str, object]) -> dict[str, dict]:
    """widget name -> font spec, from ``apply(...)`` calls and (for the fixture) QFont temporaries."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    temps: dict[str, dict] = {}
    found: dict[str, dict] = {}

    for node in _ordered(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if (isinstance(target, ast.Name) and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name) and value.func.id == "QFont"):
                temps[target.id] = dict(_DEFAULTS)
                continue

        if not isinstance(node, ast.Call):
            continue

        function = node.func

        # `apply(self.<widget>, <pt>, ...)` - the page shape (`from src.ui.typography import apply`).
        if isinstance(function, ast.Name) and function.id == "apply" or (
                isinstance(function, ast.Attribute) and function.attr == "apply"):
            if not node.args:
                raise AssertionError(f"{path.name}:{node.lineno}: apply() needs a widget")
            widget = _widget_name(node.args[0])
            if widget is None:
                raise AssertionError(f"{path.name}:{node.lineno}: apply() target is not self.<widget>")
            if widget in found:
                raise AssertionError(f"{path.name}:{node.lineno}: {widget} gets two fonts")
            found[widget] = _spec(node, constants)
            continue

        if not isinstance(function, ast.Attribute):
            continue

        # `self.<widget>.setFont(<temporary>)` - the Designer's shape, kept working so the fixture parses.
        if function.attr == "setFont" and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            widget = _widget_name(function.value)
            temporary = node.args[0].id
            if widget is None or temporary not in temps:
                raise AssertionError(f"{path.name}:{node.lineno}: setFont({temporary}) has no QFont temporary")
            found[widget] = dict(temps[temporary])
            continue

        # A temporary being shaped: `font4.setPointSize(12)`, `font5.setUnderline(True)`, ...
        if function.attr in _MUTATORS and isinstance(function.value, ast.Name):
            temporary = function.value.id
            if temporary in temps and node.args:
                key = _MUTATORS[function.attr]
                value = _value(node.args[0], constants)
                temps[temporary][key] = _family(value) if key == "family" else value

    return found


def _constants() -> dict[str, object]:
    """Module-level ``NAME = <literal>`` values from typography.py (the *_PT groups)."""
    tree = ast.parse(TYPOGRAPHY.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant):
                out[node.targets[0].id] = node.value.value
    return out


def _normalised(spec: dict) -> tuple:
    return (spec["pt"], spec["bold"], spec["italic"], spec["underline"], spec["family"])


def test_pages_match_the_designer_fixture_widget_by_widget() -> None:
    constants = _constants()
    designer = _collect(FIXTURE, constants)

    pages: dict[str, tuple[str, dict]] = {}
    for path in sorted(PAGES.glob("*.py")):
        for widget, spec in _collect(path, constants).items():
            if widget in pages:
                raise AssertionError(f"{widget} gets a font twice: {pages[widget][0]} and {path.name}")
            pages[widget] = (path.name, spec)

    missing = sorted(set(designer) - set(pages))
    extra = sorted(set(pages) - set(designer))
    assert not missing, f"pages lost the Designer's font on: {missing}"
    assert not extra, f"pages set fonts the Designer never did: {extra}"

    drifted = {
        widget: (_normalised(designer[widget]), _normalised(pages[widget][1]), pages[widget][0])
        for widget in designer
        if _normalised(designer[widget]) != _normalised(pages[widget][1])
    }
    assert not drifted, "resolved font drifted from the Designer:\n" + "\n".join(
        f"  {widget} ({where}): fixture {want} != page {got}" for widget, (want, got, where) in drifted.items())


def test_groups_cover_every_explicit_designer_size() -> None:
    constants = _constants()
    designer_sizes = {
        spec["pt"] for spec in _collect(FIXTURE, constants).values() if spec["pt"] is not None
    }
    group_names = [name for name in constants if name.endswith("_PT")]
    for required in _REQUIRED_GROUPS:
        assert required in constants, f"typography.py lost {required}"
    defined = {constants[name] for name in group_names}
    assert defined == designer_sizes, (
        f"typography groups {defined} must equal the fixture's explicit sizes {designer_sizes}")


def test_pages_do_not_construct_fonts() -> None:
    offenders: list[str] = []
    for path in sorted(PAGES.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name in {"QFont", "setPointSize", "setPointSizeF"}:
                    offenders.append(f"{path.name}:{node.lineno}: {ast.unparse(node)}")
    assert not offenders, "page code must go through src/ui/typography.py:\n" + "\n".join(offenders)


def _load_typography():
    spec = importlib.util.spec_from_file_location("rve_typography", TYPOGRAPHY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_font_matches_the_designer_temporaries() -> None:
    pytest.importorskip("PySide6.QtGui")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QFont, QFontInfo
    from PySide6.QtWidgets import QApplication, QLabel

    app = QApplication.instance() or QApplication([])  # noqa: F841 - kept alive for the widgets
    typography = _load_typography()

    def resolved(font: QFont) -> tuple[float, int]:
        """What the widget actually renders with - QFontInfo, the same resolver toolkit ``Text`` needs (E5)."""
        widget = QLabel()
        widget.setFont(font)
        info = QFontInfo(widget.font())
        return (info.pointSizeF(), info.pixelSize())

    # Body: the Designer's font4 is `QFont(); setPointSize(12)`.
    body_designer = QFont()
    body_designer.setPointSize(12)
    assert resolved(typography.font(typography.BODY_PT)) == resolved(body_designer)

    # Group title: font5 is 15 pt + underline; component order must not matter.
    title_designer = QFont()
    title_designer.setPointSize(15)
    title_designer.setUnderline(True)
    title = QLabel()
    typography.apply(title, typography.GROUP_TITLE_PT, underline=True)
    assert typography.point_size_of(title) == resolved(title_designer)[0]
    assert title.font().underline() is True and QFontInfo(title.font()).underline() is True

    # Display: font is 25 pt in "Fira Mono" (the Designer's font.setFamilies call).
    display_designer = QFont()
    display_designer.setFamilies(["Fira Mono"])
    display_designer.setPointSize(25)
    display = QLabel()
    typography.apply(display, typography.DISPLAY_PT, family="Fira Mono")
    assert typography.point_size_of(display) == resolved(display_designer)[0]
    assert QFontInfo(display.font()).pixelSize() == resolved(display_designer)[1]
    assert display.font().families()[0] == "Fira Mono"

    # A bare QFont(): no point size is set on the font itself, so the widget resolves the application
    # default (with a QApplication alive, QFont() already carries it) - it must not become 12 pt.
    unsized_designer = QFont()
    unsized_designer.setBold(True)
    unsized_font = typography.font(None, bold=True)
    assert (unsized_font.pointSize(), unsized_font.pointSizeF()) == (
        unsized_designer.pointSize(), unsized_designer.pointSizeF())
    assert unsized_font.bold() is True
    unsized = QLabel()
    typography.apply(unsized, None, bold=True)
    assert typography.point_size_of(unsized) == resolved(unsized_designer)[0]

