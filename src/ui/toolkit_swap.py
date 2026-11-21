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
    # A toolkit widget paints itself and does not derive a usable size hint from its content while its page is
    # hidden, so the stock widget's hint is the geometry source: without this the migrated Button collapsed to an
    # empty square and the ComboBox truncated to 'sudo_scene_dete...' (measured, P1 run 3).
    hint = source.sizeHint()
    if hint.width() > 0 and hint.height() > 0:
        target.setMinimumSize(hint)
    else:
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


def _wanted() -> set[str] | None:
    """Optional class filter, for bisecting which replacement changes behaviour (`RVE_TOOLKIT_SWAP=qcombobox,…`)."""
    import os
    raw = os.environ.get("RVE_TOOLKIT_SWAP", "").strip()
    return {part.strip().lower() for part in raw.split(",")} if raw else None


def _trace_threshold(window) -> None:
    """Trace mode: who changes the scene-detection threshold, with the stack that did it."""
    import os
    if os.environ.get("RVE_TOOLKIT_TRACE") != "1":
        return
    import traceback
    from backend.src.utils.Util import log

    spin = getattr(window, "scene_change_detection_threshold", None)
    if spin is None:
        log("trace: no scene_change_detection_threshold on the window")
        return
    log(f"trace: threshold at swap time = {spin.value()}")

    def on_change(value):
        stack = "".join(traceback.format_stack(limit=6)[:-1])
        log(f"trace: threshold -> {value}\n{stack}")

    spin.valueChanged.connect(on_change)


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

    _trace_threshold(window)
    wanted = _wanted()
    # EXCLUDED, with what the trace actually measured (RVE_TOOLKIT_TRACE=1):
    #  * the threshold widget's own history is clean: 1.0 from the .ui, then 3.5 at SettingsTab.connectSettingText:437
    #    and never changed again - the swap does not touch its value;
    #  * `output_folder_location.textChanged` fires once with the right path, and writeOutputFolder does not run in
    #    the default flow;
    #  * BUT with the line edit swapped the page INTERMITTENTLY renders the threshold as 1.0: stock build 3 runs ->
    #    identical to the baseline (0.00); migrated build -> 1.0 in some runs, 3.5 in another with the same class set.
    # So the failing runs are showing a DIFFERENT spin box than the one the glue writes to (the trace proves the
    # captured one receives 3.5), which is a swap/geometry problem, not the value. Until that is pinned the field
    # stays stock: the verified configuration is 4 classes, mean 3.28 / 3.51%, threshold correct.
    EXCLUDED: set[type] = {QLineEdit}

    def allowed(cls: type) -> bool:
        import os as _os
        if cls in EXCLUDED and _os.environ.get("RVE_TOOLKIT_TRACE") != "1":
            return False
        return wanted is None or cls.__name__.lower() in wanted

    for widget in list(page.findChildren(QWidget)):
        if widget.parent() is None:
            continue
        if isinstance(widget, QLabel) and allowed(QLabel):
            # The toolkit paints its own font: without the source's, a label renders at the variant's size and the
            # row shifts by a few pixels against the Designer layout (measured: labels-only run, 3.06% of the card,
            # and every fixed-coordinate comparison after it failed by an offset rather than by colour).
            font = widget.font()
            size = font.pixelSize() if font.pixelSize() > 0 else None
            target = Text(
                widget.text(),
                parent=widget.parentWidget(),
                variant="body",
                family=font.family() or None,
                pixel_size=size,
                bold=font.bold() or None,
                italic=font.italic() or None,
            )
            target.setAlignment(widget.alignment())
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QLabel->Text")
        elif isinstance(widget, QPushButton) and allowed(QPushButton):
            # Button's first positional is the ICON (text is keyword-only): passing the label there made QIcon
            # interpret it as a path, and the migrated button painted nothing (measured, P1 run 4).
            target = BridgedButton(text=widget.text(), parent=widget.parentWidget())
            target.setEnabled(widget.isEnabled())
            _copy_common(widget, target)
            _replace(widget, target, window)
            # the glue binds `clicked`; carry any already-connected slot across
            widget.clicked.connect(target.click)
            bump("QPushButton->Button")
        elif isinstance(widget, QLineEdit) and allowed(QLineEdit):
            import os as _os
            if _os.environ.get("RVE_TOOLKIT_TRACE") == "1":
                from backend.src.utils.Util import log as _log
                widget.textChanged.connect(lambda text, n=widget.objectName(): _log(f"trace: stock {n} textChanged -> {text!r}"))
            target = CustomLineEdit(parent=widget.parentWidget())
            target.setText(widget.text())
            import os as _os
            if _os.environ.get("RVE_TOOLKIT_TRACE") == "1":
                from backend.src.utils.Util import log as _log
                from sli_ui_toolkit.widgets import CustomLineEdit as _CLE
                target.textChanged.connect(lambda text, n=widget.objectName(): _log(f"trace: toolkit {n} textChanged -> {text!r}"))
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QLineEdit->CustomLineEdit")
        elif isinstance(widget, QComboBox) and allowed(QComboBox):
            target = ComboBox(parent=widget.parentWidget())
            target.addItems([widget.itemText(i) for i in range(widget.count())])
            target.setCurrentIndex(widget.currentIndex())
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QComboBox->ComboBox")
        elif isinstance(widget, QCheckBox) and allowed(QCheckBox):
            target = BridgedCheckBox(widget.text(), parent=widget.parentWidget())
            target.setChecked(widget.isChecked())
            _copy_common(widget, target)
            _replace(widget, target, window)
            bump("QCheckBox->CheckBox")
        elif isinstance(widget, QDoubleSpinBox):
            bump("QDoubleSpinBox (left stock: no toolkit counterpart)")
    return counts
