"""P0 spike: put toolkit widgets into the live RVE window and compare against the stock ones.

Not a migration - a measurement. The question the plan has to answer with evidence is whether `sli-ui-toolkit`
widgets can sit inside RVE's Designer-built window at all, and what that costs:

* a stock `QPushButton` is configured by the `.ui` (`setText`, `setIcon`, `setMaximumSize`, `setToolTip`) because it
  subclasses `QPushButton`;
* the toolkit's `Button` is a `QWidget` that paints itself and takes `text=`/`icon=` in the constructor, with no
  `setText`. Designer can only promote it as a `QWidget` placeholder, and everything visual has to be applied in code.

So this module builds the toolkit widgets for two visible places of the Home tab, drops them into the same layouts,
hides the originals and re-wires the same slots - which is exactly what a migration would do, one widget at a time.

Enabled only with ``RVE_TOOLKIT_SPIKE=1`` so the default app is untouched.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon


def enabled() -> bool:
    return os.environ.get("RVE_TOOLKIT_SPIKE", "0") == "1"


def install(window) -> list[str]:
    """Replace the Home tab's GitHub button and its caption with toolkit widgets. Returns what was replaced."""
    from PySide6.QtWidgets import QApplication
    from sli_ui_toolkit.palettes import FLUENT_DARK, FLUENT_LIGHT
    from sli_ui_toolkit.theme import ThemeManager
    from sli_ui_toolkit.widgets import Button as ToolkitButton
    from sli_ui_toolkit.widgets import Text as ToolkitText

    # A toolkit widget paints itself FROM THE THEME: without palettes registered and a theme set, a Button draws its
    # icon on nothing (measured in the first spike run: a black rectangle where the stock plate was). Every RVE
    # migration step depends on this line, so the spike owns it rather than each widget doing it.
    theme = ThemeManager.get_instance()
    theme.register_palettes(FLUENT_LIGHT, FLUENT_DARK)
    theme.set_theme("dark", QApplication.instance(), await_ripples=False)
    # RVE's own QSS (src/ui/QTstyle.py:56-72) for a QPushButton: #2c313c, hover #343b47, checked #676e7b. The toolkit
    # paints its Button from theme tokens, so matching the look is a token map, not a widget rewrite - this is the
    # first two entries of that map.
    from PySide6.QtGui import QColor
    for key in ("dialog.button.background", "surface.list"):
        theme.set_color(key, QColor("#2c313c"))

    replaced: list[str] = []

    stock_button = getattr(window, "githubBtn", None)
    if stock_button is not None:
        layout = stock_button.parentWidget().layout()
        index = layout.indexOf(stock_button) if layout is not None else -1
        toolkit_button = ToolkitButton(
            QIcon(":/icons/icons/github.svg"),
            text="",
            icon_size=35,          # the stock QPushButton sets iconSize 35x35 in the .ui
            parent=stock_button.parentWidget(),
        )
        toolkit_button.setObjectName("githubBtnToolkit")
        toolkit_button.setFixedSize(QSize(55, 34))
        toolkit_button.setToolTip("GitHub Page")
        # the same slot the stock button is wired to in src/ui/HomeTab.py
        toolkit_button.clicked.connect(stock_button.click)
        if layout is not None and index >= 0:
            layout.insertWidget(index, toolkit_button)
        stock_button.hide()
        replaced.append("githubBtn -> sli_ui_toolkit.Button")

    caption = getattr(window, "label_16", None)
    if caption is not None:
        layout = caption.parentWidget().layout()
        index = layout.indexOf(caption) if layout is not None else -1
        toolkit_text = ToolkitText(caption.text(), parent=caption.parentWidget())
        toolkit_text.setObjectName("label_16_toolkit")
        if layout is not None and index >= 0:
            layout.insertWidget(index, toolkit_text)
        caption.hide()
        replaced.append("label_16 -> sli_ui_toolkit.Text")

    return replaced
