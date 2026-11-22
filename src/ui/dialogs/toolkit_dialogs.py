"""Toolkit-backed bodies for the seven bespoke popups in :mod:`src.ui.QTcustom`.

Why this module exists
----------------------
W5 of ``harness/docs/wsp/plan-rve-ui-on-toolkit-2026-11-21.md``: a toolkit widget is a custom painter and never
receives ``WA_StyledBackground``, so the ``styleSheet()`` the old popups shared was a silent no-op once the widgets
moved onto the toolkit. The reference host (Improve-ImgSLI) uses ``ThemedDialog``/``BaseDialog`` for windows and
``ToastManager``/``ToastAction`` for transient messages, and never styles either with QSS. These classes follow that
shape:

* every window is a ``sli_ui_toolkit.ui.dialogs.BaseDialog`` subclass (the toolkit owns the surface paint, the icon and
  the theme repaint), so modality still means the same thing: a popup that used to call ``exec()`` still calls it;
* labels are ``Text``, buttons are ``Button``, output is the painted ``ConsoleView`` (the toolkit has no rich-text
  widget, and ``Text`` deliberately renders text, not HTML), the progress track is ``ToastProgressBar``;
* the transient message is a ``ToastManager`` toast, which is the reference host's non-modal notification surface;
* no colour is written here - the toolkit's painter reads its tokens, and the ANSI -> console conversion in
  ``TextOutputPopup`` only names foreground colours from the log content itself (content, not a surface).

Import policy
-------------
``src.ui.QTcustom`` is imported near the top of ``REAL-Video-Enhancer.py``, *before* ``toolkit_boot.ensure()`` runs.
This module therefore boots the toolkit path itself (the same resolver the app uses) and then imports the toolkit.
If neither is available the module still imports - ``harness/tests`` imports ``UpdateGUIThread`` from ``QTcustom`` on
interpreters without the toolkit - but constructing a dialog raises a clear ``RuntimeError`` instead of an
``ImportError`` three imports deep.

What the app's glue relies on
-----------------------------
``RegularQTPopup(message)``, ``SettingUpBackendPopup(command, ...)``, ``DownloadProgressPopup(link, location, ...)``
and friends keep their names, positional arguments and blocking behaviour. The only added argument is the
keyword-only ``blocking``: it defaults to ``True`` (the app's path) and exists so ``harness/scripts/ui_dialogs_smoke.py``
can build each window without entering an event loop or starting a real download/subprocess. It changes nothing the
app passes today.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys

import requests

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...Util import log, subprocess_popen_without_terminal

# --------------------------------------------------------------------------------------------------------------------
# Toolkit bootstrap. `ensure()` is the app's own resolver (`src/ui/toolkit_boot.py`): already importable -> override ->
# sibling checkout. Calling it here keeps `import src.ui.QTcustom` working before `main()` gets to run it.
# --------------------------------------------------------------------------------------------------------------------
_TOOLKIT_ERROR: BaseException | None = None
try:
    from ..toolkit_boot import ensure as _ensure_toolkit_importable

    _ensure_toolkit_importable()
except Exception as _boot_error:  # pragma: no cover - exercised only on a machine without the toolkit
    _TOOLKIT_ERROR = _boot_error

try:
    from sli_ui_toolkit.ui.dialogs import BaseDialog
    from sli_ui_toolkit.ui.widgets.composite.console_view import ConsoleLine, ConsoleView
    from sli_ui_toolkit.widgets import Button, Text, ToastManager, ToastProgressBar

    TOOLKIT_AVAILABLE = True
except Exception as _import_error:  # pragma: no cover - the app cannot run without the toolkit
    TOOLKIT_AVAILABLE = False
    _TOOLKIT_ERROR = _TOOLKIT_ERROR or _import_error
    BaseDialog = QDialog
    Button = Text = ToastManager = ToastProgressBar = None  # type: ignore[assignment]
    ConsoleLine = None  # type: ignore[assignment]
    ConsoleView = QWidget  # type: ignore[assignment,misc]


def _require_toolkit() -> None:
    """Fail at construction, not at import (the harness imports this module on interpreters without Qt/toolkit)."""
    if not TOOLKIT_AVAILABLE:
        raise RuntimeError(
            "the migrated dialogs need sli-ui-toolkit, which could not be imported: "
            f"{_TOOLKIT_ERROR}"
        )


# --------------------------------------------------------------------------------------------------------------------
# ANSI -> painted console. The old popup converted SGR to HTML for a stock QTextEdit; a toolkit surface paints lines,
# so the same codes become a per-line colour/weight. `ansi_to_html` is kept (re-exported by QTcustom) for callers that
# still want the string form, and is unchanged from the original implementation.
# --------------------------------------------------------------------------------------------------------------------
_ANSI_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")

_ANSI_COLORS = {
    30: "black",
    31: "red",
    32: "green",
    33: "yellow",
    34: "blue",
    35: "magenta",
    36: "cyan",
    37: "white",
    90: "gray",
}


def ansi_to_html(text: str) -> str:
    """Convert text with ANSI SGR sequences to a safe HTML string (kept for API compatibility)."""
    if not text:
        return ""

    escaped = html.escape(text)

    parts = _ANSI_SGR_RE.split(escaped)
    out = []
    open_spans = 0
    i = 0
    while i < len(parts):
        chunk = parts[i]
        out.append(chunk)
        i += 1
        if i < len(parts):
            code_chunk = parts[i]
            i += 1
            codes = [int(c) for c in code_chunk.split(";") if c != ""] if code_chunk else [0]
            if 0 in codes:
                if open_spans:
                    out.append("</span>" * open_spans)
                    open_spans = 0
                continue

            styles = []
            if 1 in codes:
                styles.append("font-weight:bold")
            for c in codes:
                if c in _ANSI_COLORS:
                    styles.append(f"color: {_ANSI_COLORS[c]}")
                elif 90 <= c <= 97:
                    color = _ANSI_COLORS.get(c - 60, None)
                    if color:
                        styles.append(f"color: {color}")

            if styles:
                out.append(f"<span style=\"{';'.join(styles)}\">")
                open_spans += 1

    if open_spans:
        out.append("</span>" * open_spans)

    return "".join(out).replace("\n", "<br>")


def ansi_to_console_lines(text: str):
    """One :class:`ConsoleLine` per source line, carrying the line's last SGR colour and bold state."""
    lines = []
    for raw in str(text).splitlines():
        color = None
        bold = False
        for match in _ANSI_SGR_RE.finditer(raw):
            codes = [int(c) for c in match.group(1).split(";") if c != ""] if match.group(1) else [0]
            for code in codes:
                if code == 0:
                    color = None
                    bold = False
                elif code == 1:
                    bold = True
                elif code in _ANSI_COLORS:
                    color = _ANSI_COLORS[code]
                elif 90 <= code <= 97:
                    color = _ANSI_COLORS.get(code - 60, color)
        plain = _ANSI_SGR_RE.sub("", raw)
        lines.append(
            ConsoleLine(plain, QColor(color) if color is not None else None, bold)
        )
    return lines


# --------------------------------------------------------------------------------------------------------------------
# Threads the dialogs own. They used to live in QTcustom; they are dialog plumbing and moving them here keeps the
# import graph one-way (`QTcustom` -> `toolkit_dialogs`). QTcustom re-exports both names.
# --------------------------------------------------------------------------------------------------------------------
class DownloadAndReportToQTThread(QThread):
    """Downloads a file while reporting the actual bytes downloaded."""

    finished = Signal()
    progress = Signal(int)

    def __init__(self, link, downloadLocation, parent=None):
        super().__init__(parent)
        self.link = link
        self.downloadLocation = downloadLocation

    def run(self):
        try:
            response = requests.get(self.link, stream=True)
            log("Downloading: " + self.link)
            if "Content-Length" in response.headers:
                totalByteSize = int(response.headers["Content-Length"])
            else:
                print(
                    "Warning: missing key 'Content-Length' in request headers; "
                    "taking default length of 100 for progress bar."
                )
                totalByteSize = 10000000

            totalSize = 0
            with open(self.downloadLocation, "wb") as f:
                chunk_size = 1024
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    totalSize += chunk_size
                    size = totalSize / totalByteSize * 100
                    self.progress.emit(size)
        except Exception as exc:  # a dead link must close the popup, not leave it on screen forever
            log(f"Download failed: {exc}")
        finally:
            self.finished.emit()


class SubprocessThread(QThread):
    output = Signal(str)
    return_code = Signal(int)
    fullOutput = Signal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "universal_newlines": True,
            "bufsize": 1,
        }

        self.process = subprocess_popen_without_terminal(self.command, **kwargs)
        totalOutput = ""
        for line in iter(self.process.stdout.readline, ""):
            totalOutput += line
            self.output.emit(line.strip())
            log(line.strip())
        self.fullOutput.emit(totalOutput)
        self.process.stdout.close()
        return_code = self.process.wait()
        self.output.emit(f"Process finished with return code {return_code}")
        self.return_code.emit(return_code)


# --------------------------------------------------------------------------------------------------------------------
# Shared toolkit window + small widget helpers.
# --------------------------------------------------------------------------------------------------------------------
def _tk_button(text: str, parent: QWidget) -> "Button":
    """A toolkit dialog button with RVE's padding/radius and no fixed 36 px control box.

    ``Button`` gives every instance a 36x36 minimum *and* maximum; clearing both lets the dialog layout size it, which
    is the same adjustment the toolkit's ``Button`` makes on the pages (``content_padding`` + corner radius).
    """
    button = Button(
        icon=None,
        text=text,
        variant="surface",
        parent=parent,
        content_padding=(10, 5, 10, 5),
        corner_radius=10,
        content_align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
    )
    button.setMinimumSize(0, 0)
    button.setMaximumSize(16777215, 16777215)
    button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    return button


def _action_row() -> QHBoxLayout:
    """A right-aligned row for the dialog's buttons."""
    row = QHBoxLayout()
    row.addStretch(1)
    return row


class ToolkitDialog(BaseDialog):
    """A toolkit ``BaseDialog`` with the shared content layout and an explicit modality flag."""

    def __init__(
        self,
        parent: QWidget | None = None,
        title: str = "",
        min_width: int = 0,
        min_height: int = 0,
        *,
        modal: bool = True,
    ) -> None:
        _require_toolkit()
        super().__init__(parent=parent, title=title, min_width=min_width, min_height=min_height)
        # `exec()` makes a dialog application-modal while it runs and restores the old modality afterwards. The old
        # popups were all `exec()`-driven, so the flag is stated once here and survives a non-blocking construction
        # (the smoke script reads it back with `isModal()`).
        self.setModal(bool(modal))
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(16, 16, 16, 16)
        self.root_layout.setSpacing(10)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Give the dialog RVE's type sizes on its first show.

        A dialog is its own top level, so `typography.apply_control_fonts(window)` (the one place the app states "labels
        12 pt, controls 12 px") never reached it: its text ran at the toolkit's design size while everything else in the
        app is larger (owner, 2026-11-22: "the text is somehow small"). Applied once, on the first show, so every dialog -
        including the ones written later - inherits the app's typography."""
        super().showEvent(event)
        if getattr(self, "_typography_applied", False):
            return
        self._typography_applied = True
        from ..typography import BODY_PT, apply as apply_font, apply_control_fonts

        for label in self.findChildren(Text):
            apply_font(label, BODY_PT)
        apply_control_fonts(self)


class ConsoleSurface(ConsoleView):  # type: ignore[misc,valid-type]
    """``ConsoleView`` with the stock text-edit names the old popups exposed.

    The painted console is the toolkit's read-only multi-line surface (it owns selection, copy, wrapping and the
    scrollbar). ``setPlainText`` is kept because the old classes filled a ``QPlainTextEdit``/``QTextEdit`` and callers
    that reach for those names should keep working; it rebuilds the console model instead of storing one string.
    """

    def setPlainText(self, text: str) -> None:
        self.set_lines(
            [ConsoleLine(line) for line in str(text).splitlines()]
        )

    def toPlainText(self) -> str:
        return self.text()

    def setHtml(self, markup: str) -> None:
        # The toolkit has no rich-text surface; HTML is only ever produced by `ansi_to_html` here, so a tag strip is
        # the honest fallback rather than pretending a stock rich-text widget came back.
        self.setPlainText(html.unescape(re.sub(r"<[^>]+>", "", str(markup))))

    def setAcceptRichText(self, accept: bool) -> None:  # noqa: N802 - Qt spelling kept on purpose
        del accept

    def setReadOnly(self, read_only: bool) -> None:  # noqa: N802
        del read_only                       # ConsoleView is read-only by construction


# --------------------------------------------------------------------------------------------------------------------
# 1. NotificationOverlay - non-modal transient message (toolkit toast)
# --------------------------------------------------------------------------------------------------------------------
class NotificationOverlay(QWidget):
    """A transient message surface over the host window, now a toolkit ``ToastManager`` toast.

    The old class was a transparent child widget carrying a QSS label. The toolkit toast is the same non-modal
    behaviour without QSS (stacking, duration and anchor are the manager's job). ``label`` still resolves to the
    painted message label for callers that reach for it.
    """

    def __init__(self, message, parent=None, timeout=3000):
        _require_toolkit()
        host = parent if parent is not None else QApplication.activeWindow()
        if host is None:
            raise ValueError(
                "NotificationOverlay needs a parent window: a toolkit toast is anchored inside one"
            )
        super().__init__(host)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._message = message
        self._manager = _toast_manager_for(host)
        self._toast_id = self._manager.show_toast(str(message), duration=int(timeout))
        self.hide()

    def _toast(self):
        toasts = getattr(self._manager, "_toasts", {})
        return toasts.get(self._toast_id)

    @property
    def label(self):
        """The toast's painted message label (``None`` once the toast has closed)."""
        try:
            toast = self._toast()
            return getattr(toast, "message_label", None) if toast is not None else None
        except RuntimeError:
            # the toast's C++ object was deleted between the registry lookup and the attribute read
            return None

    def close(self) -> bool:
        toast_id = getattr(self, "_toast_id", None)
        if toast_id is not None:
            self._toast_id = None
            try:
                self._manager.close_toast(toast_id)
            except Exception:
                pass
        return super().close()


def _toast_manager_for(host: QWidget) -> "ToastManager":
    """One manager per host window, stored on the host so its Python overrides outlive a one-shot overlay."""
    manager = getattr(host, "_rve_toast_manager", None)
    if manager is None:
        manager = ToastManager(host)
        try:
            host._rve_toast_manager = manager
        except Exception:
            pass
    return manager


# --------------------------------------------------------------------------------------------------------------------
# 2. DownloadProgressPopup - modal download window
# --------------------------------------------------------------------------------------------------------------------
class DownloadProgressPopup(ToolkitDialog):
    """Downloads a file in a thread while a toolkit progress track reports it; blocks like the old `exec()` popup."""

    canceled = Signal()

    def __init__(self, link: str, downloadLocation: str, title: str = None, *, blocking: bool = True):
        super().__init__(title=title or "Downloading", min_width=300, min_height=200, modal=True)
        self.link = link
        self.title = title
        self.downloadLocation = downloadLocation
        self.workerThread = None
        self._canceled = False

        # The old QProgressDialog set this; without a visible main window it is what makes the app exit when the
        # download popup closes.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, True)
        self.setFixedSize(300, 200)
        self.label = Text(title or "", self, word_wrap=True)
        self.root_layout.addWidget(self.label, 1)

        self.progressBar = ToastProgressBar(self, height=8)
        self.progressBar.setRange(0, 100)
        self.root_layout.addWidget(self.progressBar)

        row = _action_row()
        self.cancel_button = _tk_button("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_process)
        row.addWidget(self.cancel_button)
        self.root_layout.addLayout(row)

        if blocking:
            self.startDownload()
            self.exec()
            if self.workerThread is not None:
                self.workerThread.wait()

    def setup_ui(self) -> None:
        """Kept as a public name; the window is built in ``__init__`` because the toolkit widgets are its content."""
        return None

    def startDownload(self) -> None:
        self.workerThread = DownloadAndReportToQTThread(
            link=self.link, downloadLocation=self.downloadLocation
        )
        self.workerThread.progress.connect(self.setProgress)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(self.workerThread.wait)
        self.workerThread.start()

    def cancel_process(self) -> None:
        self._canceled = True
        self.canceled.emit()
        QApplication.quit()
        sys.exit()

    def setProgress(self, value) -> None:  # noqa: N802 - public name the old class exposed
        if self.wasCanceled():
            sys.exit()
        self.setValue(value + 10)

    # -- QProgressDialog-shaped surface the old class inherited --------------------------------------------------
    def setValue(self, value) -> None:  # noqa: N802
        self.progressBar.setValue(int(value))

    def value(self) -> int:
        return self.progressBar.value()

    def setRange(self, minimum: int, maximum: int) -> None:  # noqa: N802
        self.progressBar.setRange(minimum, maximum)

    def setLabelText(self, text: str) -> None:  # noqa: N802
        self.label.setText(text)

    def wasCanceled(self) -> bool:  # noqa: N802
        return self._canceled


# --------------------------------------------------------------------------------------------------------------------
# 3. SettingUpBackendPopup - modal command runner that keeps the command's output
# --------------------------------------------------------------------------------------------------------------------
class SettingUpBackendPopup(ToolkitDialog):
    """Runs a command and blocks until it finishes, keeping the output/return code the caller reads afterwards."""

    def __init__(
        self,
        command: list,
        title: str = "Setting Up Backend",
        progressBarLength: int = None,
        *,
        blocking: bool = True,
    ):
        super().__init__(title=title, min_width=350, min_height=400, modal=True)
        self.command = command
        self.totalCommandOutput = ""
        self.title = title
        self.totalIters = 0
        self.progressBarLength = progressBarLength
        self.output = None
        self.return_code = None
        self.workerThread = None

        self.layout2 = self.root_layout      # the old class exposed its vertical layout under this name
        self.setFixedSize(350, 400)

        self.iconLabel = QLabel(self)
        self.iconLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QIcon()
        icon.addFile(":/icons/icons/logo-v2.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        pixmap = icon.pixmap(QSize(200, 200))
        if not pixmap.isNull():
            self.iconLabel.setPixmap(pixmap)
        self.root_layout.addWidget(self.iconLabel)

        self.label = Text(self.title, self, variant="group-title", word_wrap=True)
        self.root_layout.addWidget(self.label)
        self.root_layout.addStretch(1)

        if blocking:
            self.startThread()
            self.exec()
            if self.workerThread is not None:
                self.workerThread.wait()

    def closeEvent(self, event):  # noqa: N802 - Qt spelling
        thread = getattr(self, "workerThread", None)
        if thread is not None:
            try:
                thread.quit()
                thread.wait()
            except RuntimeError:
                pass
        super().closeEvent(event)

    def startThread(self) -> None:  # noqa: N802
        self.workerThread = SubprocessThread(command=self.command)
        self.workerThread.fullOutput.connect(self.setOutput)
        self.workerThread.return_code.connect(self.setReturnCode)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(self.workerThread.wait)
        self.workerThread.start()

    def setReturnCode(self, return_code) -> None:  # noqa: N802
        self.return_code = return_code

    def setOutput(self, output) -> None:  # noqa: N802
        self.output = output

    def getReturnCode(self):  # noqa: N802
        return self.return_code

    def getOutput(self):  # noqa: N802
        return self.output


# --------------------------------------------------------------------------------------------------------------------
# 4. DisplayCommandOutputPopup - modal command runner with a live console
# --------------------------------------------------------------------------------------------------------------------
class DisplayCommandOutputPopup(ToolkitDialog):
    """Runs a command and shows its output in a painted console, with a toolkit cancel button and progress track."""

    def __init__(
        self,
        command: str,
        title: str = "Running Command",
        progressBarLength: int = None,
        *,
        blocking: bool = True,
    ):
        super().__init__(title=title, min_width=700, min_height=100, modal=True)
        self.command = command
        self.totalCommandOutput = ""
        self.title = title
        self.totalIters = 0
        self.progressBarLength = progressBarLength
        self.return_code = None
        self.workerThread = None

        self.plainTextEdit = ConsoleSurface(self)
        self.root_layout.addWidget(self.plainTextEdit, 1)

        self.progressBar = ToastProgressBar(self, height=8)
        self.progressBar.setRange(0, 100)
        if self.progressBarLength:
            self.progressBar.setRange(0, self.progressBarLength)
        self.root_layout.addWidget(self.progressBar)

        row = _action_row()
        self.pushButton = _tk_button("Cancel", self)
        self.pushButton.clicked.connect(self.close)
        row.addWidget(self.pushButton)
        self.root_layout.addLayout(row)

        if blocking:
            self.startDownload()
            self.exec()
            if self.workerThread is not None:
                self.workerThread.wait()

    def closeEvent(self, event):  # noqa: N802 - Qt spelling
        thread = getattr(self, "workerThread", None)
        if thread is not None:
            process = getattr(thread, "process", None)
            if process is not None:
                try:
                    process.terminate()
                except Exception:
                    pass
            try:
                thread.quit()
                thread.wait(2000)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)

    def startDownload(self) -> None:  # noqa: N802 - public name the old class exposed
        self.workerThread = SubprocessThread(command=self.command)
        self.workerThread.output.connect(self.setProgress)
        self.workerThread.return_code.connect(self.set_return_code)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(self.workerThread.wait)
        self.workerThread.start()

    def set_return_code(self, return_code) -> None:
        self.return_code = return_code

    def get_return_code(self) -> int:
        try:
            return self.return_code
        except Exception:
            return 1

    def setProgress(self, value) -> None:  # noqa: N802
        self.totalCommandOutput += value + "\n"
        if self.progressBarLength is not None:
            if self.totalCommandOutput.count("satisfied") > self.totalIters:
                self.totalIters = self.totalCommandOutput.count("satisfied")
            if self.totalCommandOutput.count("Collecting") > self.totalIters:
                self.totalIters = self.totalCommandOutput.count("Collecting")
            self.progressBar.setValue(self.totalIters)
        self.plainTextEdit.setPlainText(self.totalCommandOutput)


# --------------------------------------------------------------------------------------------------------------------
# 5. TextOutputPopup - modal read-only text window
# --------------------------------------------------------------------------------------------------------------------
class TextOutputPopup(ToolkitDialog):
    """Shows text (ANSI colour preserved as line colour) on the toolkit's painted console."""

    def __init__(self, message: str, title: str = "Information", *, blocking: bool = True):
        super().__init__(title=title, min_width=700, min_height=400, modal=True)
        self.message = message
        self.title = title

        self.textEdit = ConsoleSurface(self)
        self.textEdit.set_lines(ansi_to_console_lines(message))
        self.root_layout.addWidget(self.textEdit, 1)

        if blocking:
            self.exec()


# --------------------------------------------------------------------------------------------------------------------
# 6. RegularQTPopup - modal one-message window
# --------------------------------------------------------------------------------------------------------------------
class RegularQTPopup(ToolkitDialog):
    """The app-wide message box: one label, one OK button, blocks like the old ``exec()`` dialog."""

    def __init__(self, message, parent: QWidget | None = None, *, title: str = "REAL Video Enhancer",
                 blocking: bool = True):
        super().__init__(parent=parent, title=title, min_width=400, min_height=100, modal=True)
        self.message = message
        self.setFixedSize(400, 100)

        self.label = Text(str(message), self, word_wrap=True)
        self.root_layout.addWidget(self.label, 1)

        row = _action_row()
        self.ok_button = _tk_button("OK", self)
        self.ok_button.clicked.connect(self.accept)
        row.addWidget(self.ok_button)
        self.root_layout.addLayout(row)

        if blocking:
            self.exec()


# --------------------------------------------------------------------------------------------------------------------
# 6b. ConfirmPopup - the toolkit counterpart of `QMessageBox.question`
# --------------------------------------------------------------------------------------------------------------------
class ConfirmPopup(ToolkitDialog):
    """A Yes/No question on the toolkit's dialog chrome - the exit and install prompts.

    `QMessageBox.question` was the last stock window in the app: its own frame, its own palette, a stock question-mark
    icon and stock buttons, none of which follow the theme (owner, 2026-11-22: "why are the non-main windows not on the
    toolkit"). This is the same dialog in the toolkit's clothes: painted surface, `Button`s with the app's padding and
    radius, and **No** pre-focused, which is the default the stock question had.
    """

    def __init__(
        self,
        message,
        parent: QWidget | None = None,
        *,
        title: str = "REAL Video Enhancer",
        yes_text: str = "Yes",
        no_text: str = "No",
        blocking: bool = True,
    ) -> None:
        super().__init__(parent=parent, title=title, min_width=420, min_height=110, modal=True)
        self.message = str(message)
        self.label = Text(self.message, self, word_wrap=True)
        self.root_layout.addWidget(self.label, 1)

        row = _action_row()
        self.no_button = _tk_button(no_text, self)
        self.no_button.clicked.connect(self.reject)
        self.yes_button = _tk_button(yes_text, self)
        self.yes_button.clicked.connect(self.accept)
        row.addWidget(self.no_button)
        row.addWidget(self.yes_button)
        self.root_layout.addLayout(row)
        # The stock question defaulted to No; keep the same safe default (focus, not Enter-on-the-wrong-button).
        self.no_button.setFocus(Qt.FocusReason.OtherFocusReason)

        if blocking:
            self.exec()

    def confirmed(self) -> bool:
        """True only when the user pressed Yes (a closed dialog is a No)."""
        return self.result() == QDialog.DialogCode.Accepted


def confirm(parent, message, *, title: str = "REAL Video Enhancer", yes_text: str = "Yes",
            no_text: str = "No") -> bool:
    """Ask a Yes/No question on the toolkit dialog; True for Yes. Replaces `QMessageBox.question`."""
    dialog = ConfirmPopup(message, parent, title=title, yes_text=yes_text, no_text=no_text, blocking=False)
    dialog.exec()
    return dialog.confirmed()


def warn(parent, message, *, title: str = "REAL Video Enhancer") -> None:
    """A one-button message on the toolkit dialog; replaces `QMessageBox.warning`."""
    dialog = RegularQTPopup(message, parent=parent, blocking=False, title=title)
    dialog.exec()


# --------------------------------------------------------------------------------------------------------------------
# 7. IndependentQTPopup - message window that can stand alone (used by the disabled process-based path)
# --------------------------------------------------------------------------------------------------------------------
class IndependentQTPopup(ToolkitDialog):
    """A message window that may create its own ``QApplication`` when none exists (the old class did the same)."""

    def __init__(self, message=None, *, blocking: bool = True):
        if message is None:
            super().__init__(title="REAL Video Enhancer", min_width=400, min_height=100, modal=True)
            self._built = False
            return

        app = QApplication.instance()
        created_app = False
        if app is None:
            app = QApplication(sys.argv)
            created_app = True
            try:
                from ..toolkit_theme import install as _install_theme

                _install_theme(app)
            except Exception as exc:
                log(f"IndependentQTPopup: toolkit theme failed: {exc}")

        super().__init__(title="REAL Video Enhancer", min_width=400, min_height=100, modal=True)
        self.setFixedSize(400, 100)
        self.label = Text(str(message), self, word_wrap=True)
        self.root_layout.addWidget(self.label, 1)

        row = _action_row()
        self.ok_button = _tk_button("OK", self)
        self.ok_button.clicked.connect(self.accept)
        row.addWidget(self.ok_button)
        self.root_layout.addLayout(row)

        if blocking:
            self.exec()
            if created_app:
                app.exec()
