import sys
import os
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


class UpdateGUIThread(QThread):
    """The process page's 100 ms clock.

    It used to read a 640x360 thumbnail the renderer published through a shared-memory block. That channel is gone
    (owner, 2026-11-23: "какая нахуй миниатюра? Я думал этот код удален уже как день"): the pane is fed at full
    resolution from the preview file the renderer writes anyway, so the renderer pays nothing for the picture and the
    pane has exactly one painter while a render runs. What is left is the tick - the log box, FPS/ETA and the status
    label are repainted on the same 100 ms clock, and a render with no preview target used to freeze them.
    """

    #: One panel refresh. No payload on purpose: a picture comes from the preview file, never from this thread.
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