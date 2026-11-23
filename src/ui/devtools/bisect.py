"""Radical bisection of the window's widget tree, so a visual artifact can be localised in a few runs.

The owner's rule: stop probing with look-alike windows and cut the real one down - all at once, then by halves. A
look-alike window can only say "some other window is clean"; the same window with nothing inside says it about *this*
window, this surface and this process, which is the fact that ends the guessing.

`RVE_UI_BISECT=<mode>` (boot-time, because a repaint may wipe the artifact before it can be observed):

    empty            hide the central widget: the window paints only its own backdrop
    only:<name>      of the central widget's direct children, keep the one containing objectName=NAME
    hide:<a,b,...>   hide exactly these objectNames

Every mode logs what it did and how many widgets are still visible, so a run is self-describing.

Protocol: `empty` first. Artifact gone -> it lives inside the central widget, so bisect down (`only:stackedWidget`,
`only:procPage`, ...). Artifact still there over a uniform fill -> the widget tree is irrelevant; the window, the surface
or the stack produces it, and the next bisect is at that level.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import QWidget

ENV_FLAG = "RVE_UI_BISECT"


def enabled() -> bool:
    return bool(os.environ.get(ENV_FLAG))


def _visible_root(window) -> QWidget | None:
    # The shell's own container, by the object name it sets (`centralWidget`, inside `centralwidget`). The window used
    # to carry that QWidget as an attribute of the same name; it is looked up now that nothing is copied onto it.
    return window.findChild(QWidget, "centralWidget")


def _log(window, message: str) -> None:
    logger = getattr(window, "log", None) or getattr(window, "_log", None)
    if callable(logger):
        logger(message)
        return
    print(f"[bisect] {message}", flush=True)


def _hide_subtree(widget: QWidget) -> None:
    widget.hide()


def apply_mode(window) -> bool:
    """Apply `RVE_UI_BISECT`; returns True when a mode was applied. Idempotent enough to call once at startup."""
    mode = os.environ.get(ENV_FLAG, "").strip()
    if not mode or getattr(window, "_ui_bisect_applied", False):
        return False
    root = _visible_root(window)
    if root is None:
        _log(window, f"{mode}: no central widget, nothing to cut")
        return False

    if mode == "empty":
        _hide_subtree(root)
        _log(window, "empty: central widget hidden - the window paints only its backdrop")
    elif mode.startswith("only:"):
        name = mode.split(":", 1)[1].strip()
        target = window.findChild(QWidget, name)
        if target is None:
            _log(window, f"only:{name}: no widget with that objectName")
            return False
        # Keep the target's branch: at every level hide its siblings and show the node, up to the window. Works for any
        # depth, unlike hiding the central widget's direct children (most pages are several containers deep).
        node = target
        hidden = 0
        while node is not None and node is not window:
            parent = node.parentWidget()
            if parent is None:
                break
            for sibling in parent.findChildren(QWidget, options=_direct_children()):
                if sibling is not node:
                    sibling.hide()
                    hidden += 1
            node.show()
            node = parent
        _log(window, f"only:{name}: kept its branch, hid {hidden} sibling(s)")
    elif mode.startswith("hide:"):
        names = [part.strip() for part in mode.split(":", 1)[1].split(",") if part.strip()]
        done = []
        for name in names:
            for widget in window.findChildren(QWidget, name):
                _hide_subtree(widget)
                done.append(name)
        _log(window, f"hide:{','.join(names)}: hid {len(done)} widget(s) {sorted(set(done))}")
    else:
        _log(window, f"{mode}: unknown mode (empty | only:<name> | hide:<a,b>)")
        return False

    visible = sum(1 for widget in window.findChildren(QWidget) if not widget.isHidden())
    _log(window, f"{mode}: {visible} widgets still visible")
    window._ui_bisect_applied = True
    return True


def _direct_children():
    from PySide6.QtCore import Qt

    return Qt.FindChildOption.FindDirectChildrenOnly
