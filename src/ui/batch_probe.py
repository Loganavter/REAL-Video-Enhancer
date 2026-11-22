"""The thread wrapper around `video_scan`: keeps a folder read off the GUI thread.

`video_scan` holds the logic (`video_files`, `probe_many`); this file only turns it into a `QThread` with per-record
progress, so the page can show "Reading videos: 30 / 200" while the window stays live.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..constants import BACKEND_PATH, FFMPEG_PATH, PYTHON_EXECUTABLE_PATH
from .video_scan import probe_many


class BatchProbeThread(QThread):
    """Probe a folder off the GUI thread; one signal per record so the page can show progress."""

    record = Signal(dict)
    completed = Signal(list)
    failed = Signal(str)

    def __init__(self, paths: list[str], parent=None) -> None:
        super().__init__(parent)
        self._paths = list(paths)

    def run(self) -> None:  # noqa: D102 - QThread
        try:
            records = probe_many(
                self._paths,
                python=PYTHON_EXECUTABLE_PATH,
                backend=BACKEND_PATH,
                ffmpeg=FFMPEG_PATH,
                on_record=self.record.emit,
            )
        except Exception as error:      # the page must hear about a backend that would not start at all
            self.failed.emit(str(error))
            return
        self.completed.emit(records)
