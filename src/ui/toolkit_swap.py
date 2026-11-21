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

    # Which spin box is actually on screen? The intermittency said "a different widget than the one written to", so
    # dump every QDoubleSpinBox the window owns, with its value, visibility and parent chain, three seconds in.
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDoubleSpinBox

    def dump() -> None:
        log(f"trace: window.scene_change_detection_threshold = {getattr(window, 'scene_change_detection_threshold', None)}"
            f" value={spin.value()}")
        for box in window.findChildren(QDoubleSpinBox):
            chain, parent = [], box.parentWidget()
            while parent is not None and len(chain) < 4:
                chain.append(parent.objectName() or type(parent).__name__)
                parent = parent.parentWidget()
            log(f"trace: spin {box.objectName() or type(box).__name__} value={box.value()} visible={box.isVisible()}"
                f" geom={box.geometry().getRect()} parents={'<'.join(chain)}")

    QTimer.singleShot(3000, dump)


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
    # EXCLUDED: including the line edit makes the scene-detection threshold row render 1.0 instead of 3.5, and the
    # evidence is contradictory in a way that is written down rather than papered over:
    #   * excluded  -> 3.5 on screen, page deviation mean 3.28 / 3.51% (verified twice, settled captures);
    #   * included  -> 1.0 on screen, deviation 5.69% (settled capture at 42 s as well - so NOT a startup transient,
    #     which was the first wrong guess);
    #   * with RVE_TOOLKIT_TRACE=1 the widget the window attribute points at reads value=3.5, visible=True,
    #     geometry laid out - while the same frame's screenshot shows 1.0 at that row. So two objects are involved
    #     and the next step is a dump scoped to that row (siblings by geometry), not a whole-window one.
    # The exclusion keeps the verified configuration until that is answered.
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
