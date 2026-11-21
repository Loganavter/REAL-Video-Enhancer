"""The primitive classes each page builds, pinned widget by widget.

Why this file exists: the pages are emitted by `harness/scripts/split_ui.py`, and its class substitution is one regex
away from silently reverting a migration - a regeneration that emits `QDoubleSpinBox` again would compile, build and
look plausible, while the toolkit adapter (and the token it paints from) quietly left the tree. Two layers catch that:

* **the pin** (`PAGES`, `ADAPTERS`) states, per page module and per widget, the class the page builds and, for an
  adapter, the geometry policy it declares (height constant, width source, size policy, bases). A change without an
  updated pin fails here;
* **the cross-check** re-runs the splitter's own decision over the Designer fixture (`harness/fixtures/
  designer_mainwindow.py`) and compares it with the class the page module actually builds, so the splitter and the
  pages cannot drift apart either.

Both layers are pure AST: no PySide6, so they run on the portable harness interpreter as well as in the app venv.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "src" / "ui" / "pages"
SPLIT = ROOT / "harness" / "scripts" / "split_ui.py"
QT_SHAPED = ROOT / "src" / "ui" / "widgets" / "qt_shaped.py"
DESIGNER = ROOT / "harness" / "fixtures" / "designer_mainwindow.py"

#: The Qt classes this pin owns (W4d), and the adapter that replaces them where one exists.
WATCHED = ("QSpinBox", "QDoubleSpinBox", "QProgressBar", "QListWidget", "QTextEdit")
ADAPTER_NAMES = ("QtSpinBox", "QtDoubleSpinBox", "QtProgressBar", "QtListWidget", "QtTextEdit")

#: page module -> {designer widget name: the class the page builds}.
#: Stock entries carry the reason in `GAPS` - they are not accidental leftovers.
PAGES_PIN: dict[str, dict[str, str]] = {
    "tab_2": {"scene_change_detection_threshold": "QtDoubleSpinBox"},
    "tab_3": {"ncnn_gpu_id": "QtSpinBox", "pytorch_gpu_id": "QtSpinBox"},
    "general_settings": {
        "interpolationMultiplierSpinBox": "QtDoubleSpinBox",
        "upscaleScaleSpinBox": "QtDoubleSpinBox",
        "inputVideoInfoTextEdit": "QTextEdit",
    },
    "proc_page": {
        "startTimeSpinBox": "QtDoubleSpinBox",
        "endTimeSpinBox": "QtDoubleSpinBox",
        "progressBar": "QtProgressBar",
    },
    "render_queue_tab": {
        "renderQueueListWidget": "QListWidget",
        "OutputFilesListWidget": "QListWidget",
    },
    "home_page": {"changeLogText": "QTextEdit", "systemInfoText": "QTextEdit"},
    "tab_5": {"renderOutput": "QTextEdit"},
}

#: Widgets that stay stock, and why. These are toolkit rows, not local workarounds: the pin keeps the reason next to
#: the class so a later reader does not have to re-derive it from the toolkit's source.
GAPS: dict[str, str] = {
    "renderQueueListWidget": "IconListWidget is a nav list (per-row Button, 44 px rows, 8 px host spacing) and has no "
                             "insertItem/takeItem, which RenderQueue reorders the queue with",
    "OutputFilesListWidget": "same nav list - no compact text row, no addItem/insertItem",
    "inputVideoInfoTextEdit": "Text(mode='document') has no HTML setter; this panel's content is HTML",
    "changeLogText": "Text(mode='document') never scrolls and the changelog is longer than its panel; ConsoleView has "
                     "no markdown",
    "systemInfoText": "a plain read-only panel with the same QTextEdit QSS as the changelog it sits beside",
    "renderOutput": "ConsoleView is a console (radius 6, padding 6, floating bar) and Text never scrolls: migrating "
                    "the log would move a page that is currently pixel-identical and retire no QSS",
}

#: adapter -> (height policy, width policy, size policy, bases) exactly as the adapter declares it.
ADAPTERS_PIN: dict[str, tuple[str, str, str, str]] = {
    "QtSpinBox": ("HEIGHT=32", "stock-oracle:QSpinBox", "Minimum/Fixed", "_QtSpinShape, Stepper"),
    "QtDoubleSpinBox": ("HEIGHT=32", "stock-oracle:QDoubleSpinBox", "Minimum/Fixed",
                        "_QtSpinShape, StepButtonsMixin, DoubleSpinBox"),
    "QtProgressBar": ("HEIGHT=25", "WIDTH=91", "Expanding/Fixed", "ToastProgressBar"),
}

#: the shared mixin's stated geometry (RVE's QSS: `padding:5px 10px`, `border-radius:10px`, a 20 px arrow column).
SHAPE_CONSTANTS = {
    "HEIGHT": 32,
    "RADIUS": 10,
    "H_PADDING": 10,
    "STEP_BUTTON_WIDTH": 20,
    "STEP_BUTTON_GAP": 0,
    "STEP_ARROW_SIZE": 10,
}

#: The splitter's map after W4d: the three value controls are enabled; the two classes with no counterpart are not.
TOOLKIT_PIN: dict[str, frozenset[str]] = {
    "QLabel": frozenset({"setPixmap", "setScaledContents"}),
    "QPushButton": frozenset({"setAutoExclusive", "setAutoRepeat"}),
    "QCheckBox": frozenset({"setTristate"}),
    "QComboBox": frozenset(),
    "QLineEdit": frozenset(),
    "QSpinBox": frozenset({"setPrefix", "setSuffix", "setSpecialValueText", "setWrapping", "setAccelerated",
                           "setGroupSeparatorShown", "setCorrectionMode", "setInvertedControls",
                           "setKeyboardTracking"}),
    "QDoubleSpinBox": frozenset({"setPrefix", "setSuffix", "setSpecialValueText", "setWrapping", "setAccelerated",
                                 "setGroupSeparatorShown", "setCorrectionMode", "setInvertedControls",
                                 "setKeyboardTracking"}),
    "QProgressBar": frozenset({"setFormat", "setInvertedAppearance", "setTextDirection", "setOrientation",
                               "setAlignment", "reset"}),
}

HINT = "update the pin in harness/tests/test_page_primitives.py if the change is intended"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _page_widgets(module: str) -> dict[str, str]:
    """`self.<name> = <Class>(...)` for the watched primitives, in source order."""
    found: dict[str, str] = {}
    for node in ast.walk(_parse(PAGES / f"{module}.py")):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
            continue
        if value.func.id in WATCHED or value.func.id in ADAPTER_NAMES:
            found[target.attr] = value.func.id
    return found


def _chain(name: str, classes: dict[str, ast.ClassDef]) -> list[ast.ClassDef]:
    """The class plus its in-module bases (the adapters share `_QtSpinShape`)."""
    out: list[ast.ClassDef] = []
    stack = [name]
    while stack:
        current = stack.pop(0)
        node = classes.get(current)
        if node is None or node in out:
            continue
        out.append(node)
        stack.extend(base.id for base in node.bases if isinstance(base, ast.Name))
    return out


def _constant(nodes: list[ast.ClassDef], name: str) -> int | None:
    for node in nodes:
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
            if name in targets and isinstance(statement.value, ast.Constant) \
                    and isinstance(statement.value.value, int):
                return statement.value.value
    return None


def _oracle_class(nodes: list[ast.ClassDef]) -> str | None:
    for node in nodes:
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == "_stock_oracle":
                for inner in ast.walk(statement):
                    if isinstance(inner, ast.ImportFrom) and inner.module == "PySide6.QtWidgets":
                        return inner.names[0].name
    return None


def _size_policy(nodes: list[ast.ClassDef]) -> str | None:
    for node in nodes:
        source = ast.unparse(node)
        match = re.search(
            r"setSizePolicy\(QSizePolicy\.Policy\.(\w+), QSizePolicy\.Policy\.(\w+)\)", source)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    return None


def _adapter_facts(name: str, classes: dict[str, ast.ClassDef]) -> tuple[str, str, str, str]:
    nodes = _chain(name, classes)
    height = _constant(nodes, "HEIGHT")
    width = _constant(nodes, "WIDTH")
    oracle = _oracle_class(nodes)
    width_policy = f"WIDTH={width}" if width is not None else (
        f"stock-oracle:{oracle}" if oracle else "toolkit")
    height_policy = f"HEIGHT={height}" if height is not None else "toolkit"
    bases = ", ".join(ast.unparse(base) for base in classes[name].bases)
    policy = _size_policy(nodes) or "toolkit"
    return height_policy, width_policy, policy, bases


def _split_ui():
    spec = importlib.util.spec_from_file_location("split_ui_pin", SPLIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _splitter_decision() -> dict[str, dict[str, str]]:
    """What the splitter would emit for the Designer fixture, page module -> widget -> class."""
    tool = _split_ui()
    _lines, setup, _retranslate, _locals = tool.load(str(DESIGNER))
    pages, shell = tool.build_partition(setup)
    tool._migrate_page_defaults(pages, shell, setup)
    decision: dict[str, dict[str, str]] = {}
    for page in pages:
        calls = tool._widget_calls(page, setup)
        for index in page.own:
            shape = tool._class_and_name(setup[index].text)
            if not shape:
                continue
            qt_class, widget = shape
            adapter = tool._toolkit_class(qt_class, widget, calls)
            if qt_class in WATCHED or adapter in ADAPTER_NAMES:
                decision.setdefault(tool._snake(page.name), {})[widget] = adapter or qt_class
    return decision


# ----------------------------------------------------------------------------------------------------------------
# the pins
# ----------------------------------------------------------------------------------------------------------------

def test_every_primitive_widget_is_pinned_with_the_class_the_page_builds() -> None:
    for module, expected in PAGES_PIN.items():
        actual = _page_widgets(module)
        assert actual == expected, (
            f"{module}.py builds {actual}, the pin says {expected} - {HINT}")


def test_no_page_builds_an_unpinned_primitive() -> None:
    for path in sorted(PAGES.glob("*.py")):
        module = path.stem
        actual = _page_widgets(module)
        if not actual:
            continue
        assert module in PAGES_PIN, f"{module}.py builds {sorted(actual)} but has no pin - {HINT}"
        assert actual.keys() == PAGES_PIN[module].keys(), (
            f"{module}.py widgets {sorted(actual)} vs pin {sorted(PAGES_PIN[module])} - {HINT}")


def test_stock_widgets_carry_their_reason() -> None:
    stock = {widget for widgets in PAGES_PIN.values() for widget, cls in widgets.items() if cls in WATCHED}
    assert stock == set(GAPS), f"stock primitives {sorted(stock)} vs recorded gaps {sorted(GAPS)} - {HINT}"


def test_adapters_declare_the_pinned_geometry() -> None:
    classes = _classes(_parse(QT_SHAPED))
    actual = {name: _adapter_facts(name, classes) for name in ADAPTERS_PIN if name in classes}
    assert actual == ADAPTERS_PIN, f"adapter geometry changed: {actual} - {HINT}"


def test_every_adapter_named_by_a_page_exists_and_is_pinned() -> None:
    classes = _classes(_parse(QT_SHAPED))
    used = {cls for widgets in PAGES_PIN.values() for cls in widgets.values() if cls.startswith("Qt")}
    assert used == set(ADAPTERS_PIN), f"pages use {sorted(used)}, adapters pinned {sorted(ADAPTERS_PIN)} - {HINT}"
    assert used <= set(classes), f"missing adapter classes in qt_shaped.py: {sorted(used - set(classes))}"


def test_spin_box_geometry_constants_are_rve_numbers() -> None:
    classes = _classes(_parse(QT_SHAPED))
    shape = classes.get("_QtSpinShape")
    assert shape is not None, "the shared spin-box shape mixin is gone - " + HINT
    actual = {name: _constant([shape], name) for name in SHAPE_CONSTANTS}
    assert actual == SHAPE_CONSTANTS, (
        f"spin-box constants {actual} vs {SHAPE_CONSTANTS} (RVE's QSS: 32 px row, radius 10, padding 10, a 20 px "
        f"arrow column) - {HINT}")


def test_spin_boxes_keep_qt_text_alignment() -> None:
    """`QAbstractSpinBox` aligns left (`QSpinBox().alignment()` measured); the toolkit field defaults to centre."""
    source = QT_SHAPED.read_text(encoding="utf-8")
    assert "AlignLeft | Qt.AlignmentFlag.AlignVCenter" in source, (
        "the Qt spin-box alignment constant is gone - the toolkit would centre every value - " + HINT)
    classes = _classes(_parse(QT_SHAPED))
    for name in ("QtSpinBox", "QtDoubleSpinBox"):
        constructors = [node for node in classes[name].body
                        if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
        assert constructors, f"{name} has no __init__ left - {HINT}"
        assert "alignment=_qt_alignment()" in ast.unparse(constructors[0]), (
            f"{name} no longer passes Qt's alignment - {HINT}")


# ----------------------------------------------------------------------------------------------------------------
# the cross-checks: splitter decision vs the pages, and the classes the splitter enables
# ----------------------------------------------------------------------------------------------------------------

def test_splitter_enables_exactly_the_pinned_classes() -> None:
    tool = _split_ui()
    actual = {qt_class: frozenset(blocking) for qt_class, (_adapter, blocking) in tool.TOOLKIT.items()}
    assert actual == TOOLKIT_PIN, f"the splitter's TOOLKIT map changed: {actual} - {HINT}"


def test_splitter_would_emit_the_class_each_page_builds() -> None:
    """Regeneration gate: the splitter's own decision equals the page module's constructor."""
    decision = _splitter_decision()
    for module, widgets in PAGES_PIN.items():
        assert module in decision, f"the splitter would not emit {module}.py at all - {HINT}"
        for widget, cls in widgets.items():
            assert widget in decision[module], (
                f"splitter: {module}.py has no {widget} - the fixture or the split lost it - {HINT}")
            assert decision[module][widget] == cls, (
                f"splitter would emit {decision[module][widget]} for {module}/{widget}, the page builds {cls} - "
                f"{HINT}")
        extra = set(decision[module]) - set(widgets)
        assert not extra, f"splitter would emit unpinned primitives in {module}: {sorted(extra)} - {HINT}"


@pytest.mark.parametrize("qt_class", ["QListWidget", "QTextEdit"])
def test_unmigrated_classes_are_absent_from_the_map(qt_class: str) -> None:
    """A gap is a decision: enabling these without an adapter would build a class that cannot carry the Qt API."""
    tool = _split_ui()
    assert qt_class not in tool.TOOLKIT, f"{qt_class} was enabled in the splitter without an adapter - {HINT}"
