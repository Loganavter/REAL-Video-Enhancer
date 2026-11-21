"""Dump every page's widget tree as JSON, so a rebuilt interface can be compared structurally, not only by pixels.

Why this exists on top of `page_capture.py`: a picture says "3.1% of pixels differ"; it cannot say *which* of 300
widgets moved, and it cannot say "this button lost its text while painting the same shape". Rebuilding pages in code
(the BlinkTable-level goal) needs a stronger invariant than "the screenshot looks close": the tree must be the same -
same classes where nothing was migrated, same object names, same parent-child order, same geometry, same text, same
visibility, same current index. That is what this dump records, and `harness/scripts/ui_layout_diff.py` compares two of
them.

Shape: one file per page, `<page>.json`, with a recursive `widgets` tree. Walks the same pages in the same order as the
capture walker (`page_capture.steps`), because two instruments over two different page sets would produce a table that
describes nothing.

Enabled by ``RVE_UI_LAYOUT_DUMP=<dir>``; same 40 s settle as the capture (see `page_capture` for why).
"""

from __future__ import annotations

import json
import os

from PySide6.QtWidgets import QWidget


def enabled() -> bool:
    return bool(os.environ.get("RVE_UI_LAYOUT_DUMP"))


def _text_of(widget: QWidget) -> str:
    """Best-effort text/value of a widget, with the getter that Qt actually exposes for its class."""
    for getter in ("text", "currentText", "title", "placeholderText"):
        value = getattr(widget, getter, None)
        if callable(value):
            try:
                text = value()
            except Exception:
                continue
            if isinstance(text, str) and text:
                return text
    # Toolkit widgets keep their label in a field rather than a getter (`Button.setText` writes `_text`, and there is
    # no `text()` to read it back). The dump has to see what the widget displays, so that field is the documented
    # fallback - reading nothing there would have reported every migrated button as empty (measured).
    private = getattr(widget, "_text", None)
    if isinstance(private, str) and private:
        return private
    to_plain = getattr(widget, "toPlainText", None)
    if callable(to_plain):
        try:
            text = to_plain()
        except Exception:
            text = ""
        if text:
            return text if len(text) <= 200 else text[:200] + "..."
    return ""


def _number_of(widget: QWidget) -> dict:
    """State that changes what a page shows without changing its tree: index, value, check state."""
    state: dict = {}
    for getter in ("currentIndex", "value"):
        fn = getattr(widget, getter, None)
        if callable(fn):
            try:
                value = fn()
            except Exception:
                continue
            # Some widgets expose currentIndex() as a QModelIndex; only a scalar describes visible state here, and
            # anything else is not JSON and not comparable across builds.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                state[getter] = value
    # Only a CHECKED widget records its check state. A `False` default is not interface information, and toolkit
    # widgets expose `isChecked()` on classes that are not check boxes at all - recording it turned a class
    # substitution into dozens of fake state differences (measured on the first W4 run).
    checked = getattr(widget, "isChecked", None)
    if callable(checked):
        try:
            if checked():
                state["checked"] = True
        except Exception:
            pass
    return state


def _icon_of(widget: QWidget) -> tuple[bool, list[int]]:
    """Whether a control actually has a pixmap icon, and how big it asks to draw it.

    The dump recorded text and geometry only, so a button that lost its icon (a migration that rebuilt the content)
    looked identical to the stock one in every table the harness had. That is exactly the regression the owner
    reported, and it was invisible by construction.
    """
    icon = None
    getter = getattr(widget, "icon", None)
    if callable(getter):
        try:
            icon = getter()
        except Exception:
            icon = None
    has = bool(icon is not None and not icon.isNull())
    if not has:
        has = bool(getattr(widget, "_has_icon", False))
    if not has:
        return False, [0, 0]
    size = getattr(widget, "iconSize", None)
    icon_size = size() if callable(size) else getattr(widget, "_icon_box", None)
    if icon_size is None:
        return True, [0, 0]
    return True, [int(icon_size.width()), int(icon_size.height())]


def _node(widget: QWidget) -> dict:
    node: dict = {
        "class": type(widget).__name__,
        "name": widget.objectName() or "",
        # QWidget.geometry() is parent-local; only a top-level window is screen-relative (and this dump has one root).
        "geometry": list(widget.geometry().getRect()),
        "visible": bool(widget.isVisible()),
        "enabled": bool(widget.isEnabled()),
    }
    layout = widget.layout()
    if layout is not None:
        node["layout"] = type(layout).__name__
    text = _text_of(widget)
    if text:
        node["text"] = text
    node.update(_number_of(widget))
    has_icon, icon_size = _icon_of(widget)
    if has_icon:
        node["icon"] = True
        node["icon_size"] = icon_size
    try:
        from PySide6.QtGui import QFontInfo

        node["font_px"] = int(QFontInfo(widget.font()).pixelSize())
    except Exception:
        pass
    try:
        palette = widget.palette()
        node["colors"] = {
            role: palette.color(getattr(palette.ColorRole, role)).name()
            for role in ("Button", "Base", "Window", "Text")
        }
    except Exception:
        pass
    children = [child for child in widget.children() if isinstance(child, QWidget)]
    if children:
        node["children"] = [_node(child) for child in children]
    return node


def dump(window: QWidget) -> dict:
    """The whole window as one JSON-ready dict (root node plus the geometry convention stated in the file)."""
    root = _node(window)
    return {"geometry_space": "parent-local", "root": root}


def install(window) -> None:
    """Arm the dumper: after the settle delay, walk every page, write one file per page, quit."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from src.ui import page_capture

    out_dir = os.environ["RVE_UI_LAYOUT_DUMP"]
    os.makedirs(out_dir, exist_ok=True)
    delay = int(os.environ.get("RVE_UI_LAYOUT_DELAY_MS", os.environ.get("RVE_UI_CAPTURE_DELAY_MS", "40000")))
    per_page = int(os.environ.get("RVE_UI_LAYOUT_STEP_MS", "260"))

    steps = page_capture.steps(window)
    if not steps:
        return

    def warm_up(index: int) -> None:
        """First pass: make every page current once, so its layout has run before anything is written.

        A page's children keep their construction-time geometry until the page has been current at least once; a
        single-pass walk wrote dumps where the *next* pages were 100x30 boxes and 640x480 stubs, which reads as a
        layout difference that is not there (measured twice, on both sides of a comparison).
        """
        if index >= len(steps):
            QTimer.singleShot(0, lambda: walk(0))
            return
        _name, action = steps[index]
        action()
        QApplication.processEvents()
        QTimer.singleShot(per_page, lambda: warm_up(index + 1))

    def walk(index: int) -> None:
        if index >= len(steps):
            QApplication.quit()
            return
        name, action = steps[index]
        action()
        QApplication.processEvents()
        page_capture.stabilise(window)
        QTimer.singleShot(per_page, lambda: write(name, index))

    def write(name: str, index: int) -> None:
        write_page(window, name)
        QTimer.singleShot(0, lambda: walk(index + 1))

    QTimer.singleShot(delay, lambda: warm_up(0))


def write_page(window, name: str) -> None:
    """Write one page's dump into `RVE_UI_LAYOUT_DUMP`. Shared with the capture walker (one walk, both artefacts)."""
    out_dir = os.environ.get("RVE_UI_LAYOUT_DUMP", "")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    try:
        tree = dump(window)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(tree, handle, indent=1, sort_keys=True)
        print(f"layout: {path} {_count(tree)} widgets", flush=True)
    except Exception as exc:      # an instrument that dies silently hangs the run until the timeout
        import traceback
        print(f"layout: FAILED on {name}: {exc}\n{traceback.format_exc()}", flush=True)


def _count(tree: dict) -> int:
    node = tree.get("root", {})
    total = 1
    for child in node.get("children", []):
        total += _count({"root": child})
    return total
