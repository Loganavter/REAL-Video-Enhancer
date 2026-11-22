import sys
import os
import subprocess
import requests
import struct
import time
import numpy as np
from multiprocessing import shared_memory

from . import shared_flag
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


class UpdateGUIThread(QThread):
    """
    Reads the preview thumbnail the renderer publishes and hands it to the GUI as a QImage.

    The block is small (a 640x360 thumbnail is 0.69 MB, a 4K frame is 24.9 MB), carries a
    header that says what the pixels are, and has two slots so a reader never sees half of
    one frame and half of the next. The `wanted` byte is the other half of the bargain: the
    renderer builds a thumbnail only after this thread consumed the previous one, so a
    viewer at 5 Hz cannot make the render loop pay 30 times a second.
    """

    latestPreviewPixmap = Signal(QtGui.QImage)

    # Mirrors HEADER in backend/src/utils/PreviewChannel.py: two processes, two trees, one
    # layout. Kept as a literal here so the GUI never imports backend code.
    PREVIEW_HEADER = struct.Struct("<IHHBBBI")
    PREVIEW_FMT_RGB888 = 1
    PREVIEW_FMT_YUV420 = 2
    PREVIEW_OFF_WANTED = 9
    PREVIEW_POLL_SECONDS = 0.1

    def __init__(self, parent, imagePreviewSharedMemoryID):
        super().__init__()
        self._parent = parent
        self.shm = None
        self._stop_flag = False  # Boolean flag to control stopping
        self._mutex = QMutex()  # Atomic flag to control stopping
        self.imagePreviewSharedMemoryID = imagePreviewSharedMemoryID
        self.previewWidth = 0
        self.previewHeight = 0

    def createNewSharedMemory(self, width: int, height: int):
        """Allocate the thumbnail block: header + two slots of `width` x `height` RGB888."""
        if self.shm is not None:
            # a queued render re-creates the block; close the old mapping instead of leaking it
            try:
                self.shm.close()
            finally:
                self.shm = None
        self.previewWidth = int(width)
        self.previewHeight = int(height)
        self.previewBoxW = int(width)
        self.previewBoxH = int(height)
        size = (
            self.PREVIEW_HEADER.size
            + 2 * self.previewBoxW * self.previewBoxH * 3
        )
        try:
            self.shm = shared_flag.create_many(self.imagePreviewSharedMemoryID, size)
        except FileExistsError:
            # a crashed run leaves the block behind (POSIX keeps it until unlinked), and a
            # restarted session can be handed the same name: drop the stale block, then
            # allocate ours. Attaching to the old one would keep its size and header.
            stale = shared_flag.attach(self.imagePreviewSharedMemoryID)
            shared_flag.release(stale)
            self.shm = shared_flag.create_many(self.imagePreviewSharedMemoryID, size)
        self.shm.buf[self.PREVIEW_OFF_WANTED] = 1  # ask for the first frame
        shared_flag.register(lambda: getattr(self, "shm", None))

    def deleteSharedMemory(self):
        shm, self.shm = self.shm, None
        shared_flag.release(shm)

    def run(self):
        last_seq = 0
        while True:
            with QMutexLocker(self._mutex):
                if self._stop_flag:
                    self.deleteSharedMemory()
                    break
            try:
                if self.shm is not None:
                    seq, width, height, fmt, _wanted, _flags, _reserved = (
                        self.PREVIEW_HEADER.unpack_from(self.shm.buf, 0)
                    )
                    if seq != last_seq and width and height and fmt in (
                        self.PREVIEW_FMT_RGB888, self.PREVIEW_FMT_YUV420
                    ):
                        last_seq = seq
                        # the slots were laid out for the BOX: a 4:3 frame's 640x480
                        # thumbnail must not make the reader step into the other slot
                        offset = (
                            self.PREVIEW_HEADER.size
                            + (seq & 1) * self.previewBoxW * self.previewBoxH * 3
                        )
                        length = self._payload_length(fmt, width, height)
                        # bytes(): own the pixels, so the QImage cannot end up pointing at
                        # a slot the renderer is already overwriting (the old reader built
                        # a QImage over a numpy buffer that died on return)
                        pixels = bytes(self.shm.buf[offset:offset + length])
                        self.latestPreviewPixmap.emit(
                            self._to_image(pixels, width, height, fmt)
                        )
                        self.shm.buf[self.PREVIEW_OFF_WANTED] = 1
            except Exception as e:
                log(f"Error in UpdateGUIThread: {e}")

            # This loop is also the panel's refresh clock: the log box, FPS/ETA and the
            # status label are painted by the same slot that shows a preview frame, so
            # with no preview target (no shared block, or preview disabled) the whole
            # panel used to freeze while the render ran. A null image is a tick, not a
            # frame - the slot skips the pixmap branch and repaints the text.
            self.latestPreviewPixmap.emit(QtGui.QImage())

            time.sleep(self.PREVIEW_POLL_SECONDS)

    @staticmethod
    def _payload_length(fmt, width, height):
        if fmt == UpdateGUIThread.PREVIEW_FMT_RGB888:
            return width * height * 3
        return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)

    @staticmethod
    def _to_image(pixels, width, height, fmt):
        """QImage from the block's payload; the colour matrix lives here, not in the render loop."""
        if fmt == UpdateGUIThread.PREVIEW_FMT_RGB888:
            return QtGui.QImage(
                pixels, width, height, width * 3, QtGui.QImage.Format_RGB888
            ).copy()
        n = width * height
        ch, cw = (height + 1) // 2, (width + 1) // 2
        y = np.frombuffer(pixels[:n], np.uint8).reshape(height, width).astype(np.float32) - 16.0
        u = np.frombuffer(pixels[n:n + ch * cw], np.uint8).reshape(ch, cw).astype(np.float32) - 128.0
        v = np.frombuffer(pixels[n + ch * cw:n + 2 * ch * cw], np.uint8).reshape(ch, cw).astype(np.float32) - 128.0
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:height, :width]
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:height, :width]
        r = np.clip(1.164 * y + 1.793 * v, 0.0, 255.0)
        g = np.clip(1.164 * y - 0.213 * u - 0.533 * v, 0.0, 255.0)
        b = np.clip(1.164 * y + 2.112 * u, 0.0, 255.0)
        rgb = np.dstack((r, g, b)).astype(np.uint8)
        return QtGui.QImage(
            rgb.data, width, height, width * 3, QtGui.QImage.Format_RGB888
        ).copy()

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