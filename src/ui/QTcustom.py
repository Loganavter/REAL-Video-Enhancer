import sys
import os
import struct
import subprocess
import requests
import time
from multiprocessing import shared_memory

import re
import html

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QIcon
from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    QTime,
    QUrl,
    Qt,QTimer
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSpacerItem,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QMainWindow,
)
from multiprocessing import Process
from ..constants import PLATFORM, has_network
from ..Util import log, networkCheck, subprocess_popen_without_terminal

# W5: the seven popups are toolkit-backed. Their bodies live in `src/ui/dialogs/toolkit_dialogs.py`; this is the
# import surface the app already uses, plus the two thread classes the dialogs own and the ANSI helper that used to
# live here. Keeping the names here means no call site (and no harness test) changes.
from .dialogs.toolkit_dialogs import (
    DisplayCommandOutputPopup,
    DownloadAndReportToQTThread,
    DownloadProgressPopup,
    IndependentQTPopup,
    NotificationOverlay,
    RegularQTPopup,
    SettingUpBackendPopup,
    SubprocessThread,
    TextOutputPopup,
    ansi_to_html,
)

__all__ = [
    "DisplayCommandOutputPopup",
    "DownloadAndReportToQTThread",
    "DownloadProgressPopup",
    "IndependentQTPopup",
    "NetworkCheckPopup",
    "NotificationOverlay",
    "RegularQTPopup",
    "SettingUpBackendPopup",
    "SubprocessThread",
    "TextOutputPopup",
    "UpdateGUIThread",
    "addNotificationToButton",
    "ansi_to_html",
    "disable_combobox_item",
    "disable_combobox_item_by_text",
    "hide_layout_widgets",
    "needs_network_else_exit",
    "remove_combobox_item_by_text",
    "show_layout_widgets",
]

def disable_combobox_item(combobox: QComboBox, index):
    """
    Disable a specific item in a QComboBox (making it visible but unselectable)
    
    Parameters:
        combobox: QComboBox widget
        index: Index of the item to disable
    """
    if 0 <= index < combobox.count():
        model = combobox.model()
        item = model.item(index)
        if item:
            # Remove the enabled flag to disable the item
            flags = item.flags() & ~Qt.ItemIsEnabled
            item.setFlags(flags)
            item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray color

def disable_combobox_item_by_text(combobox: QComboBox, text):
    """
    Disable a specific item in a QComboBox by its text
    
    Parameters:
        combobox: QComboBox widget
        text: Text of the item to disable
    """
    index = combobox.findText(text)
    if index >= 0:
        disable_combobox_item(combobox, index)

def remove_combobox_item_by_text(combobox: QComboBox, text:str,):
    
    """
    removes a specific item in a QComboBox by its text (not case sensitive)
    """
    for idx in range(combobox.count()):
        if text.lower() in combobox.itemText(idx).lower():
            combobox.removeItem(idx)

def hide_layout_widgets(layout):
    # Iterate through all items in the layout and hide the widgets
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.setVisible(False)  # Hide the widget


def show_layout_widgets(layout):
    # Iterate through all items in the layout and show the widgets
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.setVisible(True)  # Show the widget


#: The pane's block, as the reader sees it. The writer's copy is `backend/src/utils/DisplayChannel.py`; both sides
#: derive every offset from the struct, never from a literal, because a field at the wrong offset does not raise -
#: it draws a torn or colour-shifted picture. 15 bytes: `seq`, `width`, `height`, `fmt`, `wanted`, `flags`, reserved.
DISPLAY_HEADER = struct.Struct("<IHHBBBI")
DISPLAY_HEADER_SIZE = DISPLAY_HEADER.size
DISPLAY_OFFSET_SEQ = 0
DISPLAY_OFFSET_WIDTH = 4
DISPLAY_OFFSET_HEIGHT = 6
DISPLAY_OFFSET_FMT = 8
DISPLAY_OFFSET_WANTED = 9
DISPLAY_FMT_RGB888 = 1
DISPLAY_RGB_BYTES_PER_PIXEL = 3

#: The newest-wins seek request: `requested_seconds` (double) + `generation` (uint32) in a 16-byte block, the same
#: layout `DisplayChannel.SEEK_STRUCT` writes and the renderer's `SeekState` reads.
SEEK_REQUEST = struct.Struct("<dI")
SEEK_BLOCK_SIZE = 16

#: The block's size when the pane has not been laid out yet (or reports no size). The probe measured its channel at
#: this box; the live pane sizes its own block from its device pixels where that is available.
DEFAULT_DISPLAY_BOX = (960, 540)


def display_state_id(pid: int | None = None) -> str:
    """The frame block's name: one per GUI process, exactly like `PAUSED_STATE_SHARED_MEMORY_ID`."""
    return "/image_display" + str(os.getpid() if pid is None else int(pid))


def seek_state_id(pid: int | None = None) -> str:
    """The seek request block's name: same process-scoped convention."""
    return "/seek_state" + str(os.getpid() if pid is None else int(pid))


DISPLAY_STATE_SHARED_MEMORY_ID = display_state_id()
SEEK_STATE_SHARED_MEMORY_ID = seek_state_id()


class DisplayReader(QObject):
    """The pane's end of the display channel (`backend/src/utils/DisplayChannel.py` is the writer's).

    The GUI allocates the block before the backend starts (`ProcessTab._prepare_display_state`), because the pane is
    what knows its own box and the renderer must only ever attach. On the page's 100 ms tick `poll` reads the header,
    skips an unchanged `seq`, copies the newest complete slot and hands the picture on as a `QImage`.

    Two invariants live here and nowhere else:

    * **The newest complete slot is `(seq - 1) & 1`.** The writer fills slot `seq & 1` and only *then* writes
      `seq + 1`, so after a publish the payload is one slot behind the header's counter. (The probe reads only the
      header; this offset is the writer's own contract, pinned by `test_display_channel.py`.)
    * **One `seq` per tick.** A frame that arrived while the previous one was being copied is not chased in a loop:
      the next tick shows the newest one, which is the point of the two slots.
    """

    #: One composited frame, already a Qt image owned by the GUI (the shared buffer is copied, not aliased).
    display_frame = Signal(QImage)

    def __init__(self, name: str, box=DEFAULT_DISPLAY_BOX, seek_name: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.name = str(name)
        self.seek_name = str(seek_name) if seek_name else f"{self.name}_seek"
        width, height = int(box[0]), int(box[1])
        self.box = (max(1, width), max(1, height))
        self._slot_size = 0
        self._last_seq = 0
        self._want = False
        self._shm = None
        self._seek_shm = None
        self._attached = False
        self._open()

    # -- attach / release ---------------------------------------------------------------------------

    def _open(self) -> None:
        from . import shared_flag

        wanted_size = DISPLAY_HEADER_SIZE + 2 * (self.box[0] * self.box[1] * DISPLAY_RGB_BYTES_PER_PIXEL)
        segment = None
        created = False
        try:
            try:
                segment = shared_flag.attach(self.name)
            except FileNotFoundError:
                segment = None
            if segment is None or len(segment.buf) < wanted_size:
                if segment is not None:
                    # A previous process with this pid left a smaller block: the name is ours (it carries the pid), so
                    # replacing it cannot take a live pane's block away.
                    shared_flag.release(segment)
                segment = shared_flag.create(self.name, wanted_size)
                created = True
        except Exception:
            segment = None
        self._shm = segment
        self._attached = segment is not None and len(segment.buf) >= DISPLAY_HEADER_SIZE + 3
        if self._attached:
            # The segment is the authority on the slot size: a larger stale block still fits this pane's frames,
            # and its own size is what the writer validated against.
            self._slot_size = (len(segment.buf) - DISPLAY_HEADER_SIZE) // 2
            if created:
                # A freshly owned block starts at `seq` 0: an attached one (the pane reopening, a hand-written test
                # block) keeps whatever its writer already put there.
                DISPLAY_HEADER.pack_into(segment.buf, 0, 0, 0, 0, 0, 0, 0, 0)
            self._write_wanted(0)
        try:
            self._seek_shm = shared_flag.create(self.seek_name, SEEK_BLOCK_SIZE)
        except Exception:
            self._seek_shm = None
        # `/dev/shm` must not keep these after a quit that skips every close handler (the names carry the pid, so
        # nothing else would ever find them) - the same lifetime rule as the paused flag.
        shared_flag.register(lambda: self._shm)
        shared_flag.register(lambda: self._seek_shm)

    @property
    def attached(self) -> bool:
        return bool(self._attached and self._shm is not None)

    @property
    def slot_size(self) -> int:
        return int(self._slot_size)

    @property
    def seq(self) -> int:
        if not self.attached:
            return 0
        return int(DISPLAY_HEADER.unpack_from(self._shm.buf, 0)[0])

    def release(self) -> None:
        """Close and unlink both blocks; safe twice and never raises."""
        from . import shared_flag

        self._want = False
        for attribute in ("_seek_shm", "_shm"):
            segment = getattr(self, attribute, None)
            setattr(self, attribute, None)
            if segment is not None:
                shared_flag.release(segment)
        self._attached = False
        self._slot_size = 0
        self._last_seq = 0

    # -- asking for frames --------------------------------------------------------------------------

    def _write_wanted(self, value: int) -> None:
        if self._shm is None:
            return
        try:
            self._shm.buf[DISPLAY_OFFSET_WANTED] = int(value) & 0xFF
        except Exception:
            pass

    def set_wanted(self, wanted: bool) -> None:
        """Tell the renderer whether a picture is wanted right now (one byte; the renderer reads it per frame)."""
        self._want = bool(wanted)
        self._write_wanted(1 if self._want else 0)

    def wanted(self) -> bool:
        return bool(self._want)

    def block_wanted(self) -> int:
        """The byte the renderer actually reads (for tests and for "is the backend asking?")."""
        if not self.attached:
            return 0
        return int(self._shm.buf[DISPLAY_OFFSET_WANTED])

    # -- the tick -----------------------------------------------------------------------------------

    def poll(self) -> QImage | None:
        """Read at most one new frame and emit it; `None` when nothing new (or nothing wanted) is there."""
        if not self.attached or not self._want:
            return None
        buf = self._shm.buf
        seq, width, height, fmt, _wanted, _flags, _reserved = DISPLAY_HEADER.unpack_from(buf, 0)
        seq = int(seq)
        if seq == self._last_seq:
            return None
        width, height = int(width), int(height)
        if fmt != DISPLAY_FMT_RGB888 or width <= 0 or height <= 0:
            return None
        needed = width * height * DISPLAY_RGB_BYTES_PER_PIXEL
        slot = (seq - 1) & 1
        start = DISPLAY_HEADER_SIZE + slot * self._slot_size
        if self._slot_size <= 0 or start + needed > len(buf):
            return None
        payload = bytes(buf[start:start + needed])
        # A frame published while the copy ran would make the picture a mix of two: drop this tick rather than draw it.
        if int(DISPLAY_HEADER.unpack_from(buf, 0)[0]) != seq:
            return None
        self._last_seq = seq
        # Re-assert the request: the writer preserves this byte, and this is where the pane proves it is still here.
        self._write_wanted(1)
        image = QImage(payload, width, height, width * DISPLAY_RGB_BYTES_PER_PIXEL,
                       QImage.Format.Format_RGB888).copy()
        self.display_frame.emit(image)
        return image

    # -- the newest-wins seek -----------------------------------------------------------------------

    def seek_generation(self) -> int:
        if self._seek_shm is None:
            return 0
        return int(SEEK_REQUEST.unpack_from(self._seek_shm.buf, 0)[1])

    def request_seek(self, seconds: float) -> int:
        """Write `(seconds, generation + 1)`; returns the generation written (0 when the block is missing)."""
        if self._seek_shm is None:
            return 0
        generation = (self.seek_generation() + 1) & 0xFFFFFFFF
        SEEK_REQUEST.pack_into(self._seek_shm.buf, 0, float(seconds), generation)
        return generation

    def seek_request(self) -> tuple[float, int]:
        """The bytes a renderer would read: `(seconds, generation)`."""
        if self._seek_shm is None:
            return 0.0, 0
        seconds, generation = SEEK_REQUEST.unpack_from(self._seek_shm.buf, 0)
        return float(seconds), int(generation)


class UpdateGUIThread(QThread):
    """The process page's 100 ms clock.

    It used to read a 640x360 thumbnail the renderer published through a shared-memory block. That channel is gone
    (owner, 2026-11-23: "какая нахуй миниатюра? Я думал этот код удален уже как день"), and the pane is fed at full
    resolution from the preview file the renderer writes anyway. It is also the tick of the export's display channel
    (`DisplayReader`, 2026-11-24): one `seq` per tick, only while the pane is visible and the transport is playing.
    What is left is the clock - the log box, FPS/ETA, the status label and the pane's two painters run on it, and a
    render with no preview target used to freeze them.
    """

    #: One panel refresh. No payload: the picture is read from the pane's own block on this tick.
    tick = Signal()
    TICK_SECONDS = 0.1

    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent
        self._stop_flag = False
        self._mutex = QMutex()

    def run(self):
        while True:
            with QMutexLocker(self._mutex):
                if self._stop_flag:
                    break
            self.tick.emit()
            time.sleep(self.TICK_SECONDS)

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_flag = True


# The threads the popups own and the seven popups themselves are toolkit-backed now. Their bodies live in
# `src/ui/dialogs/toolkit_dialogs.py` (W5 of plan-rve-ui-on-toolkit); this module keeps the import surface the app
# already uses (`from .ui.QTcustom import ...`) and the helpers below that never belonged to a popup.
def NetworkCheckPopup(hostname="https://raw.githubusercontent.com") -> bool:
    if not networkCheck(hostname=hostname):
        RegularQTPopup("No Network Connection")
        return False
    # return true if network connection
    return True

def needs_network_else_exit():
    if not has_network():
        RegularQTPopup("Network is required for this action!\nPlease connect to a network.")
        os._exit(1)


def addNotificationToButton(button: QPushButton):
    notification = QLabel(button)
    notification.setFixedSize(10, 10)
    notification.setStyleSheet("background-color: red; border-radius: 5px;")
    notification.move(button.width() - 15, 5)
    notification.show()


if __name__ == "__main__":
    DownloadProgressPopup(
        link="https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg",
        downloadLocation="ffmpeg",
        title="Downloading Python",
    )