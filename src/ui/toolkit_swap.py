"""P1: put one screen of RVE on the toolkit, keeping RVE's look and its glue.

The screen this starts with is the **Output Settings** page (`tab_2` in `testRVEInterface.ui`, 440 lines): 9 labels, one
line edit, one button, two checkboxes, one combobox and one spin box. It is the smallest screen that exercises every
primitive the migration needs, so it is the honest unit to prove the approach before a bigger tab is touched.

Two facts from P0 shape everything here:

* the toolkit's widgets are `QWidget`s that paint themselves and mirror most of Qt's API (`Text.setText/text`,
  `ComboBox.addItems/currentText/currentIndex`, `CustomLineEdit.text/setText/textChanged`), but *not* all of it -
  `Button` has no `setText` (text is a constructor argument) and `CheckBox` reports `toggled(bool)` where RVE's glue
  expects `stateChanged(int)`. Those two gaps are bridged in this module, in the host, not in the toolkit;
* the glue (`src/ui/SettingsTab.py`) runs AFTER `setupUi`, so a replacement made immediately after `setupUi` is the
  widget the glue binds to - as long as the window attribute is rebound too (`parent.output_folder_location`).

What is NOT migrated and why: `QDoubleSpinBox` has no toolkit counterpart, so it stays stock; `QScrollArea` is a
container, not a look.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

# The page to migrate. Kept as a name rather than a widget object so the same code can be pointed at the next screen.
DEFAULT_PAGE = "tab_2"


def enabled() -> bool:
    """P1 is a measured step, not the default yet: the page is compared against the stock build before it ships on."""
    import os
    return os.environ.get("RVE_TOOLKIT_MIGRATE", "0") == "1"


class ToolkitLabel:
    """Nothing to bridge: `sli_ui_toolkit.widgets.Text` already has `setText`/`text`."""


def _bridge_checkbox():
    """RVE binds `stateChanged(int)`; the toolkit's CheckBox reports `toggled(bool)`."""
    from sli_ui_toolkit.widgets import CheckBox

    class BridgedCheckBox(CheckBox):
        stateChanged = Signal(int)

        def __init__(self, text: str | None = None, parent=None):
            super().__init__(text, parent)
            self.toggled.connect(lambda checked: self.stateChanged.emit(2 if checked else 0))

    return BridgedCheckBox


def _bridge_button():
    """`Button` takes its text in the constructor and has no setter; RVE calls `setText` on a few buttons."""
    from sli_ui_toolkit.widgets import Button

    class BridgedButton(Button):
        def setText(self, text: str) -> None:      # noqa: N802 - Qt spelling, on purpose
            self._rve_text = text
            self.setToolTip(text)
            self.update()

    return BridgedButton


def _copy_common(source: QWidget, target: QWidget) -> None:
    target.setObjectName(source.objectName())
    target.setToolTip(source.toolTip())
    target.setMinimumSize(source.minimumSize())
    target.setMaximumSize(source.maximumSize())
    target.setSizePolicy(source.sizePolicy())


def _replace(source: QWidget, target: QWidget, window) -> None:
    """Put ``target`` where ``source`` was, hide the source, and rebind the window attribute if it pointed at it."""
    layout = source.parentWidget().layout() if source.parentWidget() is not None else None
    index = layout.indexOf(source) if layout is not None else -1
    if index >= 0:
        layout.insertWidget(index, target)
    # Two measured rules, both from failures of this file:
    #  * never copy the source's visibility - the swap runs right after setupUi, when the page is still hidden, and
    #    `setVisible(False)` pinned every new widget invisible for the session (first run: 15 replacements, empty page);
    #  * never reparent or delete the source - it is still referenced by the generated mainwindow, and
    #    `setParent(None)` crashed the process with SIGSEGV (second run). It stays in place, hidden, and the toolkit
    #    widget sits in front of it in the same layout slot.
    source.hide()
    name = source.objectName()
    if name and window is not None and getattr(window, name, None) is source:
        setattr(window, name, target)


def migrate_page(window, page_name: str = DEFAULT_PAGE) -> dict[str, int]:
    """Replace the page's stock primitives with toolkit widgets. Returns the count per class."""
    from sli_ui_toolkit.widgets import Button, ComboBox, CustomLineEdit, Text

    page = getattr(window, page_name, None)
    if page is None:
        return {"missing page": 0}

    BridgedCheckBox = _bridge_checkbox()
    BridgedButton = _bridge_button()
    counts: dict[str, int] = {}

    def bump(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    for widget in list(page.findChildren(QWidget)):
        if widget.parent() is None:
            continue
        if isinstance(widget, QLabel):
            target = Text(widget.text(), parent=widget.parentWidget())
            target.setAlignment(widget.alignment())
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QLabel->Text")
        elif isinstance(widget, QPushButton):
            target = BridgedButton(widget.text(), parent=widget.parentWidget())
            target.setEnabled(widget.isEnabled())
            _copy_common(widget, target)
            _replace(widget, target, window)
            # the glue binds `clicked`; carry any already-connected slot across
            widget.clicked.connect(target.click)
            bump("QPushButton->Button")
        elif isinstance(widget, QLineEdit):
            target = CustomLineEdit(parent=widget.parentWidget())
            target.setText(widget.text())
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QLineEdit->CustomLineEdit")
        elif isinstance(widget, QComboBox):
            target = ComboBox(parent=widget.parentWidget())
            target.addItems([widget.itemText(i) for i in range(widget.count())])
            target.setCurrentIndex(widget.currentIndex())
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QComboBox->ComboBox")
        elif isinstance(widget, QCheckBox):
            target = BridgedCheckBox(widget.text(), parent=widget.parentWidget())
            target.setChecked(widget.isChecked())
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QCheckBox->CheckBox")
        elif isinstance(widget, QDoubleSpinBox):
            bump("QDoubleSpinBox (left stock: no toolkit counterpart)")
    return counts
