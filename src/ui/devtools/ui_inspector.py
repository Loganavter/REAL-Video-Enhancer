"""The toolkit's UI inspector, wired into RVE behind an opt-in flag.

The toolkit ships a full inspector (`InspectorWindow` + `InspectorController`: widget tree, family, config, state, colours,
painted regions, layers, theme tokens, live layout and the constructor/code site), but RVE never installed it - so the
question "what is that rectangle on my screen" had no in-app answer, only screenshots and guesses.

`RVE_UI_INSPECTOR=1` installs it: the tool window opens next to the app, and a **Shift+click** on any widget in the app
commits it to the inspector. Ctrl+Shift+I brings the tool window back after the app takes focus.

Why a flag and not always-on: the controller installs an application-wide event filter (hover and Shift+click handling)
and the tool window is a second top-level dialog. That is right for development and wrong for a render session.
"""

from __future__ import annotations

import os

#: Environment variable that arms the inspector.
ENV_FLAG = "RVE_UI_INSPECTOR"

#: Tool-window shortcut (the controller itself only handles Shift+click).
SHORTCUT = "Ctrl+Shift+I"


def enabled() -> bool:
    """True when the host asked for the inspector (`RVE_UI_INSPECTOR=1`)."""
    return bool(os.environ.get(ENV_FLAG))


def install(window) -> object | None:
    """Wire the toolkit inspector onto `window`; returns the controller, or None if already installed.

    The controller is application-wide (it filters the app's events, not just this window), so it is stored on the
    window only to keep it alive and to make a second call a no-op.
    """
    if getattr(window, "_ui_inspector_controller", None) is not None:
        return None

    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import QApplication

    from sli_ui_toolkit.theme import ThemeManager
    from sli_ui_toolkit.ui.inspector.controller import InspectorController
    from sli_ui_toolkit.ui.inspector.view import InspectorWindow

    app = QApplication.instance()
    if app is None:
        return None

    inspector = InspectorWindow()
    inspector.setWindowTitle("UI Inspector - REAL Video Enhancer")
    controller = InspectorController(app, inspector, ThemeManager.get_instance())
    window._ui_inspector_window = inspector
    window._ui_inspector_controller = controller

    def _raise_inspector() -> None:
        inspector.show()
        inspector.raise_()
        inspector.activateWindow()

    window._ui_inspector_shortcut = QShortcut(QKeySequence(SHORTCUT), window)
    window._ui_inspector_shortcut.activated.connect(_raise_inspector)
    inspector.show()
    return controller
