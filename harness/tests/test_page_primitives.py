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

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "src" / "ui" / "pages"
SPLIT = ROOT / "harness" / "scripts" / "split_ui.py"
DESIGNER = ROOT / "harness" / "fixtures" / "designer_mainwindow.py"

#: The Qt classes this pin owns (W4d), and the adapter that replaces them where one exists.
WATCHED = ("QSpinBox", "QDoubleSpinBox", "QProgressBar", "QListWidget", "QTextEdit")
#: Replacement classes: the `Qt*` adapters from `qt_shaped.py`, plus toolkit classes a page constructs directly (a page
#: that migrates off the compatibility layer names the toolkit class itself, e.g. `DoubleSpinBox`).
ADAPTER_NAMES = ("QtSpinBox", "QtDoubleSpinBox", "QtProgressBar", "QtListWidget", "QtTextEdit", "ConsoleView",
                 "DoubleSpinBox", "Stepper", "ToastProgressBar", "Text", "Button", "CheckBox", "ComboBox",
                 "RichTextView", "TextListWidget", "DoubleSpinBox")

#: page module -> {designer widget name: the class the page builds}. Every watched widget is migrated to an adapter;
#: a stock one coming back would have to be pinned here and argued for in review.
#: Pages migrated off the compatibility layer are DERIVED, not listed: a page is hand-maintained the moment it no
#: longer imports `qt_shaped`. The splitter's cross-check therefore keeps its meaning for the pages still on the layer
#: without every migration wave editing this file (three parallel cells would fight over one literal). The generator
#: and this cross-check retire together with `qt_shaped.py`.
def _hand_maintained() -> frozenset[str]:
    root = Path(__file__).resolve().parents[2] / "src" / "ui" / "pages"
    return frozenset(page.stem for page in root.glob("*.py") if "qt_shaped" not in page.read_text(encoding="utf-8"))


HAND_MAINTAINED = _hand_maintained()

#: Adapter -> the toolkit class that replaces it. A pinned widget is accepted as either until the layer is deleted, so a
#: migration wave never has to edit this file; the pin still fails on any OTHER class (a stock control, a wrong widget).
TOOLKIT_SUCCESSORS = {
    "QtSpinBox": "Stepper",
    "QtDoubleSpinBox": "DoubleSpinBox",
    "QtProgressBar": "ToastProgressBar",
    "QtComboBox": "ComboBox",
    "QtCheckBox": "CheckBox",
    "QtButton": "Button",
    "QtLabel": "Text",
    "QtLineEdit": "CustomLineEdit",
}

PAGES_PIN: dict[str, dict[str, str]] = {
    "tab_2": {"scene_change_detection_threshold": "DoubleSpinBox"},
    "tab_3": {"ncnn_gpu_id": "QtSpinBox", "pytorch_gpu_id": "QtSpinBox"},
    "general_settings": {
        "interpolationMultiplierSpinBox": "QtDoubleSpinBox",
        "upscaleScaleSpinBox": "QtDoubleSpinBox",
        "inputVideoInfoTextEdit": "RichTextView",
    },
    "proc_page": {
        "startTimeSpinBox": "QtDoubleSpinBox",
        "endTimeSpinBox": "QtDoubleSpinBox",
        "progressBar": "QtProgressBar",
    },
    "render_queue_tab": {
        "renderQueueListWidget": "TextListWidget",
        "OutputFilesListWidget": "TextListWidget",
    },
    "home_page": {"changeLogText": "RichTextView", "systemInfoText": "RichTextView"},
    "tab_5": {"renderOutput": "ConsoleView"},
}




#: The splitter's map: the value controls and the log pane are enabled; `QListWidget` is not (the toolkit has no
#: compact text list - `IconListWidget` is a navigation list, measured at 3.2 % on the queue page).
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
    "QTextEdit": frozenset({"setLineWrapMode", "setAcceptRichText", "setUndoRedoEnabled", "setTextInteractionFlags",
                            "setPlaceholderText", "setTabStopDistance", "setWordWrapMode", "setCurrentCharFormat",
                            "mergeCurrentCharFormat"}),
    "QListWidget": frozenset({"setSelectionMode", "setDragDropMode", "setDragEnabled", "setAcceptDrops",
                              "setDefaultDropAction", "setEditTriggers", "setViewMode", "setIconSize",
                              "setUniformItemSizes", "setAlternatingRowColors", "setWordWrap", "setSpacing",
                              "setVerticalScrollBarPolicy", "setHorizontalScrollBarPolicy", "setItemWidget"}),
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

def _pinned_only(module: str, actual: dict[str, str]) -> dict[str, str]:
    """A migrated page builds `Text`/`Button` on every row; the pin speaks about named widgets only."""
    pinned = PAGES_PIN.get(module, {})
    return {name: cls for name, cls in actual.items() if name in pinned} if pinned else actual


def _mismatches(actual: dict[str, str], expected: dict[str, str]) -> list[str]:
    """Per-widget comparison that accepts the toolkit successor of a pinned adapter, never another class."""
    if actual.keys() != expected.keys():
        return [f"widgets {sorted(actual)} vs pin {sorted(expected)}"]
    return [
        f"{name}: builds {actual[name]}, pin {want} (or {TOOLKIT_SUCCESSORS.get(want)})"
        for name, want in expected.items()
        if actual[name] != want and TOOLKIT_SUCCESSORS.get(want) != actual[name]
    ]


def test_every_primitive_widget_is_pinned_with_the_class_the_page_builds() -> None:
    for module, expected in PAGES_PIN.items():
        problems = _mismatches(_pinned_only(module, _page_widgets(module)), expected)
        assert not problems, f"{module}.py: {'; '.join(problems)} - {HINT}"


def test_no_page_builds_an_unpinned_primitive() -> None:
    for path in sorted(PAGES.glob("*.py")):
        module = path.stem
        actual = _page_widgets(module)
        if not actual:
            continue
        if module in HAND_MAINTAINED:
            continue
        assert module in PAGES_PIN, f"{module}.py builds {sorted(actual)} but has no pin - {HINT}"
        assert actual.keys() == PAGES_PIN[module].keys(), (
            f"{module}.py widgets {sorted(actual)} vs pin {sorted(PAGES_PIN[module])} - {HINT}")


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
        if module in HAND_MAINTAINED:
            continue
        assert module in decision, f"the splitter would not emit {module}.py at all - {HINT}"
        for widget, cls in widgets.items():
            assert widget in decision[module], (
                f"splitter: {module}.py has no {widget} - the fixture or the split lost it - {HINT}")
            assert decision[module][widget] == cls, (
                f"splitter would emit {decision[module][widget]} for {module}/{widget}, the page builds {cls} - "
                f"{HINT}")
        extra = set(decision[module]) - set(widgets)
        assert not extra, f"splitter would emit unpinned primitives in {module}: {sorted(extra)} - {HINT}"
