import subprocess
import os
from threading import Thread
import re
from time import sleep
from multiprocessing import shared_memory

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor, QFontMetrics
from PySide6.QtCore import Qt, QSize, QUrl

from .RenderQueue import RenderQueue
from ..render_jobs import short_duration
from .render_log import LogLine, RenderLogFilter
from . import shared_flag

from .AnimationHandler import AnimationHandler
from .QTcustom import (
    RegularQTPopup,
    UpdateGUIThread,
    show_layout_widgets,
    hide_layout_widgets,
)
from ..constants import (
    BACKEND_PATH,
    PYTHON_EXECUTABLE_PATH,
    MODELS_PATH,
    CUSTOM_MODELS_PATH,
    IMAGE_SHARED_MEMORY_ID,
    PAUSED_STATE_SHARED_MEMORY_ID,
    PREVIEW_WIDTH,
    PREVIEW_HEIGHT,
    INPUT_TEXT_FILE,
    FFMPEG_PATH,
    CWD,
    PLATFORM,
)
from ..Util import (
    log,
)
from ..DownloadModels import DownloadModel
from .SettingsTab import Settings
from ..DiscordRPC import DiscordRPC
from ..ModelHandler import getModels
from .RenderQueue import RenderOptions


class ProcessTab:
    def __init__(self, parent, settings: Settings):
        self.parent = parent
        self.renderTextOutputList = None
        #: The curated lines (`LogLine`: text + level) behind `renderTextOutputList`; the console paints these.
        self.renderLogLines: list[LogLine] | None = None
        self._log_filter = RenderLogFilter()
        self.isOverwrite = False
        self.outputVideoHeight = None
        self.outputVideoWidth = None
        self.currentFrame = 0
        self.fps = 0
        self.eta = 0
        self.max_value = 0
        self.isPreview = False
        self.userKilled = False
        self.currentRenderOptions = None
        self.status = "Idle"
        #: monotonic stamp of the current render, for the "elapsed" half of the progress line
        self._render_started_mono = None
        self.animationHandler = AnimationHandler()
        self.tileUpAnimationHandler = AnimationHandler()
        self.tileDownAnimationHandler = AnimationHandler()
        self.settings = settings
        self.return_codes = []

        self.qualityToCRF = {
            "Low": "28",
            "Medium": "23",
            "High": "18",
            "Very High": "15",
        }
        # encoder dict
        # key is the name in RVE gui
        # value is the encoder used

        # get default backend
        self.QConnect()
        self.populateModels(self.parent.backendComboBox.currentText())
        self.setStatus("Idle")

    def setStatus(self, text: str) -> None:
        """The one place that decides what `STATUS:` says - never leave a stale phase on screen."""
        self.status = str(text)
        try:
            self.parent.STATUS.setText(f"Status: {self.status}")
        except Exception:
            pass

    def setProgressDetail(self, text: str) -> None:
        try:
            self.parent.progressDetail.setText(str(text))
        except Exception:
            pass

    def progressDetailText(self) -> str:
        """`42% · 12m3s elapsed · 16m40s left · frame 1234/5678` - empty while idle."""
        if self._render_started_mono is None and not self.max_value:
            return ""
        total = max(1, int(self.max_value or 0))
        done = max(0, min(total, int(self.currentFrame or 0)))
        parts = [f"{int(round(100.0 * done / total))}%"]
        if self._render_started_mono is not None:
            from time import monotonic

            elapsed = max(0.0, monotonic() - self._render_started_mono)
            parts.append(f"{short_duration(elapsed)} elapsed")
        if self.eta:
            parts.append(f"{self.eta} left")
        parts.append(f"frame {done}/{total}")
        return " · ".join(parts)

    def populateModels(self, backend) -> dict:
        """
        returns
        the current models available given a method (interpolate, upscale) and a backend (ncnn, tensorrt, pytorch)
        """
        interpolateModels, upscaleModels, deblurModels, denoiseModels, decompressModels, _ = getModels(backend)
        self.parent.interpolateModelComboBox.clear()
        self.parent.upscaleModelComboBox.clear()
        self.parent.deblurModelComboBox.clear()
        self.parent.denoiseModelComboBox.clear()
        self.parent.decompressModelComboBox.clear()
        self.parent.interpolateModelComboBox.addItems(
            list(interpolateModels.keys())
        )
        self.parent.interpolateModelComboBox.setCurrentIndex(len(list(interpolateModels.keys()))-1)
        self.parent.upscaleModelComboBox.addItems(list(upscaleModels.keys()))
        self.parent.deblurModelComboBox.addItems(list(deblurModels.keys()))
        self.parent.denoiseModelComboBox.addItems(list(denoiseModels.keys()))
        self.parent.decompressModelComboBox.addItems(list(decompressModels.keys()))
    def onTilingSwitch(self):
        if self.parent.tilingCheckBox.isChecked():
            self.parent.tileSizeContainer.setVisible(True)
            self.tileDownAnimationHandler.dropDownAnimation(
                self.parent.tileSizeContainer
            )
        else:
            self.tileUpAnimationHandler.moveUpAnimation(self.parent.tileSizeContainer)
            self.parent.tileSizeContainer.setVisible(False)

    def openOutputFolderInExplorer(self):
        output_folder = os.path.dirname(self.parent.outputFileText.text())
        if os.path.isdir(output_folder):
            if PLATFORM == "win32":
                os.startfile(output_folder)
            elif PLATFORM == "darwin":
                subprocess.Popen(["open", output_folder])
            else:
                subprocess.Popen(["xdg-open", output_folder])

    def installQueueRows(self) -> None:
        """Queue rows carry a job's state: a thin progress bar, the percentage, and a hover line.

        The rows are painted by the host - `TextListWidget` stays a text list (see `src/ui/render_jobs.py`).
        """
        from .queue_rows import queue_row_painter, queue_tooltip

        widget = self.parent.renderQueueListWidget
        widget.set_row_painter(queue_row_painter(self.parent.renderQueue))
        widget.set_row_tooltip_provider(queue_tooltip(self.parent.renderQueue))

    def refreshQueueRows(self) -> None:
        """Repaint the queue after a state change (a tick, a finish): the painter reads the model, no row rebuild."""
        try:
            self.parent.renderQueueListWidget.update()
        except Exception:
            pass

    def installQueueMenu(self) -> None:
        """The queue's right-click menu - the replacement for the three buttons the page used to carry.

        Built from the toolkit's own `ContextMenu` (theme surface, keyboard focus, the danger colour on the destructive
        rows). The rows are selected by the list before `customContextMenuRequested` fires, so every action applies to
        the whole selection: remove the picked items, or slide the block up/down. The entries are rebuilt on every
        request because their enabled state depends on the selection and on the render lock.
        """
        from sli_ui_toolkit.widgets import ContextMenu, ContextMenuAction, ContextMenuSeparator

        widget = self.parent.renderQueueListWidget

        def entries_for(queue):
            locked = queue.is_locked()
            has_selection = bool(queue.selectedRows())
            # A locked queue is being consumed by the running render: the actions are shown DISABLED rather than the
            # whole list being disabled, so the user can still scroll it while the render runs.
            return (
                ContextMenuAction(
                    "queue.remove", "Remove from queue",
                    enabled=has_selection and not locked, danger=True,
                ),
                ContextMenuSeparator(),
                ContextMenuAction("queue.up", "Move Up", enabled=has_selection and not locked),
                ContextMenuAction("queue.down", "Move Down", enabled=has_selection and not locked),
                ContextMenuSeparator(),
                ContextMenuAction(
                    "queue.clear", "Clear queue", enabled=bool(queue.getQueue()) and not locked, danger=True,
                ),
            )

        def on_action(action_id: str, _data=None) -> None:
            queue = self.parent.renderQueue
            if action_id == "queue.remove":
                queue.removeSelected()
            elif action_id == "queue.up":
                queue.moveSelected("up")
            elif action_id == "queue.down":
                queue.moveSelected("down")
            elif action_id == "queue.clear":
                queue.clear()

        def on_context_menu(point) -> None:
            # A FRESH menu per request: the toolkit's popup surface destroys itself when it hides, so reusing one
            # instance raised "Internal C++ object (ContextMenu) already deleted" on the second right-click (measured).
            # NO `surface="popup"`: this was the app's only Qt.Popup, and the reference host's investigation of exactly
            # this symptom ends with one line - "a Qt.Popup whose QWidget parent is the top-level window breaks the
            # window's compositing on this Wayland/XWayland+NVIDIA stack: in-window translucent surfaces (toasts) render
            # opaque black, existing QRhiWidgets slow down" (blinktable docs/dev/investigations/
            # inv-2026-10-06-ql-rhi-menu-first.md, Root cause RESOLVED). The toolkit's default surface is the in-window
            # one, which is what every other menu in that app uses; the popup mode is opt-in and we opted in by accident.
            menu = ContextMenu(widget, entries=entries_for(self.parent.renderQueue),
                               on_triggered=on_action)
            # `on_triggered` is constructor state (the toolkit keeps it as `_on_triggered`), so it is passed here and
            # not assigned afterwards - an attribute set after construction would never fire.
            self._queue_menu = menu
            menu.popup_at(widget.mapToGlobal(point))

        widget.customContextMenuRequested.connect(on_context_menu)

    def QConnect(self):
        # connect file select buttons
        self.parent.addToRenderQueueButton.clicked.connect(self.parent.addToRenderQueue)
        self.installQueueMenu()
        self.installQueueRows()
        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.batchSelectButton.clicked.connect(self.parent.openBatchFiles)
        self.parent.inputFileText.textChanged.connect(self.parent.loadVideo)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        self.parent.openOutputFolderButton.clicked.connect(self.openOutputFolderInExplorer)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        # set tile size visible to false by default
        self.parent.tileSizeContainer.setVisible(False)
        # set slo mo container visable to false by default
        
        self.parent.interpolateContainer_2.setVisible(False)
        # connect up tilesize container visiable
        self.parent.tilingCheckBox.stateChanged.connect(self.onTilingSwitch)

        self.parent.interpolationMultiplierSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )

        self.parent.upscaleModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.upscaleScaleSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.interpolateModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.decompressModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.interpolateCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.upscaleCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.deblurCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.denoiseCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.decompressCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)   
        self.parent.sloMoModeCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)  
        self.parent.backendComboBox.currentIndexChanged.connect(
            lambda: self.populateModels(self.parent.backendComboBox.currentText())
        )
        self.parent.EncoderCommand.textChanged.connect(lambda: self.parent.EncoderCommand.setFixedWidth(max(50, QFontMetrics(self.parent.EncoderCommand.font()).horizontalAdvance(self.parent.EncoderCommand.text()) + 10)))
        # connect up pausing
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.pauseRenderButton.clicked.connect(self.pauseRender)
        self.parent.killRenderButton.clicked.connect(self.killRenderProcess)

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.userKilled = True
            self.renderProcess.terminate()
        except AttributeError:
            log("No render process!")

    def pauseRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 1  # 1 = True
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)
        self.setStatus("Paused")

    def resumeRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 0  # 0 = False
        from time import monotonic

        self._render_started_mono = monotonic()
        self.setStatus("Preparing")
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.onRenderButtonsContiainer.setEnabled(True)
        self.parent.startRenderButton.setVisible(False)

    def startGUIUpdate(self):
        
        self.workerThread = UpdateGUIThread(
            parent=self,
            imagePreviewSharedMemoryID=IMAGE_SHARED_MEMORY_ID,
        )
        self.workerThread.latestPreviewPixmap.connect(self.updateProcessTab)
        self.workerThread.finished.connect(self.guiChangesOnRenderCompletion)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def splitListIntoStringWithNewLines(self, string_list: list[str]):
        # Join the strings with newline characters
        return "\n".join(string_list)
        # Set the text to the QTextEdit

    def questionToOverride(self):
        from src.ui.dialogs.toolkit_dialogs import confirm

        return confirm(
            self.parent,
            "Output files in render queue already exist, do you want to overwrite?",
        )

    def askForOverwrite(self, outputPath):
        if os.path.isfile(outputPath):
            self.isOverwrite = self.questionToOverride()
            if not self.isOverwrite:
                self.onRenderCompletion()
                self.guiChangesOnRenderCompletion()
                return False
        return True

    def checkForOverwrite(self, renderQueue: RenderQueue):
        for renderOptions in renderQueue.getQueue():
            if not self.askForOverwrite(renderOptions.outputPath):
                return False
        return True

    def run(
        self,
        renderQueue: RenderQueue,
    ):
        self.return_codes = [] # reset return codes
        self.userKilled = False # reset userkilled
        # gui changes
        from time import monotonic

        self._render_started_mono = monotonic()
        self.setStatus("Preparing")
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(False)
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.resumeRender)

        if not self.checkForOverwrite(renderQueue):
            return

        self.startDiscordRPC()
        self.settings.readSettings()

        self.startGUIUpdate()
        writeThread = Thread(target=lambda: self.renderToPipeThread(renderQueue))
        writeThread.start()

    def startDiscordRPC(self):
        if self.settings.settings["discord_rich_presence"] == "True":
            try:
                self.discordRPC = DiscordRPC()
                self.discordRPC.start_discordRPC()
            except Exception:
                pass

    def createPausedSharedMemory(self):
        """The one-byte paused flag the backend polls - created here, removed by `closePausedSharedMemory`.

        `track=False` on purpose. With the default the segment is registered with `multiprocessing`'s resource tracker,
        whose at-exit sweep then prints two warnings the owner saw on quit (2026-11-22): "1 leaked shared_memory
        objects to clean up" and "No such file or directory: '//paused_state<pid>'" - CPython's `unlink` raises before
        it unregisters, so a segment removed from another path (or by a previous run's cleanup) stays registered and is
        reported as leaked. The name carries the pid, nothing else can own it, and this object releases it explicitly:
        at the end of a render and in the window's close handler.
        """
        self.pausedSharedMemory = shared_flag.create(PAUSED_STATE_SHARED_MEMORY_ID)
        shared_flag.register(lambda: getattr(self, "pausedSharedMemory", None))

    def closePausedSharedMemory(self):
        """Release the paused flag; safe to call twice, and safe when nothing was rendered."""
        shm = getattr(self, "pausedSharedMemory", None)
        self.pausedSharedMemory = None
        shared_flag.release(shm)

    def renderToPipeThread(
        self,
        renderQueue: RenderQueue,
    ):
        self.createPausedSharedMemory()

        for renderOptions in renderQueue.pending_jobs():
            
            self.isPreview = renderOptions.isPreview
            self.currentRenderOptions = renderOptions
            renderQueue.mark_rendering(renderOptions)
            self.refreshQueueRows()

            # The preview block is a fixed-size thumbnail now, not the output frame: its
            # geometry comes from one constant on both sides, so it is allocated before the
            # backend starts and never depends on the output resolution.
            self.workerThread.createNewSharedMemory(PREVIEW_WIDTH, PREVIEW_HEIGHT)
            self.max_value = renderOptions.videoFrameCount * renderOptions.interpolateTimes
            command = self.build_command(renderOptions)
            log(str(command))

            kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "universal_newlines": True,
                "text": True,                 # return str instead of bytes
                "encoding": "utf-8",          # decode using utf-8
                "errors": "replace",
            }

            if PLATFORM == "win32":
                kwargs["startupinfo"] = subprocess.STARTUPINFO()
                kwargs["startupinfo"].dwFlags |= subprocess.STARTF_USESHOWWINDOW


            self.renderProcess = subprocess.Popen(
                command,
                **kwargs,
            )
            textOutput = []
            self._log_filter = RenderLogFilter()
            self.renderLogLines = []
            for line in iter(self.renderProcess.stdout.readline, b""):
                if self.renderProcess.poll() is not None:
                    break  # Exit the loop if the process has terminated

                try:
                    line = str(line.strip())
                    if "it/s" in line:
                        textOutput = textOutput[:-1]
                    if "FPS" in line:
                        textOutput = textOutput[
                            :-1
                        ]  # slice the list to only get the last updated data
                        self.currentFrame = int(
                            re.search(r"Current Frame: (\d+)", line).group(1)
                        )
                        self.fps = re.search(r"FPS: (\d+)", line).group(1)
                        # bounded to h:mm:ss: the stream can carry a log record on the same line
                        self.eta = re.search(r"ETA: (\d+(?::\d+){2})", line).group(1)
                        self.setStatus("Rendering")

                    if "this may take a while" in line.lower():
                        self.setStatus("Building engine…")


                    # The pane gets the EVENTS, not the telemetry: per-frame timings, the live progress line it already
                    # shows as FPS/ETA, and the setup chatter repeats are dropped here (src/ui/render_log.py explains
                    # each class and the numbers behind them).
                    for entry in self._log_filter.feed(line):
                        textOutput.append(entry.text)
                        self.renderLogLines.append(entry)
                    self.renderTextOutputList = textOutput.copy()
                except Exception as e:
                    pass
                if "Time to complete render" in line:
                    break
            self.return_codes.append(self.renderProcess.wait())
            summary = self._log_filter.hidden_summary()
            if summary is not None:
                textOutput.append(summary.text)
                self.renderLogLines.append(summary)
                self.renderTextOutputList = textOutput.copy()
            for line in textOutput:
                if len(line) > 2:
                    log(line)

            renderQueue.mark_finished(renderOptions, ok=self.return_codes[-1] in (0, 3221225477))
            self.refreshQueueRows()
            self.workerThread.deleteSharedMemory()

        self.closePausedSharedMemory()

        # NO automatic `renderQueue.clear()`: a finished job stays in the list with its final state (owner, 2026-11-22),
        # and a job that never ran because the render stopped must stay too. Clearing is a USER action - the context
        # menu's "Clear queue" or removing the selection. Two automatic clears used to wipe the queue at the end of every
        # pass, which is exactly the "items disappear without me deleting them" report.
        self.onRenderCompletion()

    def _restoreQueueAfterPreview(self) -> None:
        """Put back what `renderPreview` set aside, and drop the preview job itself from the list."""
        backup = getattr(self.parent, "_queue_after_preview", None)
        if not backup:
            return
        self.parent._queue_after_preview = None
        queue = self.parent.renderQueue
        # the preview job is history, not work: it is not re-added
        queue.clear()
        for job in backup:
            queue.add(job)

    def guiChangesOnRenderCompletion(self):
        if all(return_code == 0 or return_code == 3221225477 for return_code in self.return_codes) or self.userKilled: # 3221225477 comes up when using ncnn on windows for some reason, but no error in output itself.
            log("All render processes completed successfully")
        else:
            log("Some render processes failed: Error code: " + str(self.return_codes))
            RegularQTPopup("Rendering failed! Please check the logs tab!")
        # Have to swap the visibility of these here otherwise crash for some reason
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setEnabled(True)
        self.parent.outputFileText.setEnabled(True)
        self.parent.inputFileText.setEnabled(True)
        if getattr(self.parent, "preview_label_is_surface", False):
            self.parent.previewLabel.set_frame(None)
        else:
            self.parent.previewLabel.clear()
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        self.parent.enableProcessPage()
        self.parent.startRenderButton.setVisible(True)
        self._render_started_mono = None
        self.setProgressDetail("")
        finished_ok = all(code == 0 or code == 3221225477 for code in self.return_codes) or self.userKilled
        self.setStatus("Idle" if finished_ok else "Failed")
        self.parent.FPS.setText("FPS: ")
        self.parent.ETA.setText("ETA: ")

        self._restoreQueueAfterPreview()
        if self.currentRenderOptions.isPreview:
            self._start_preview_playback(self.currentRenderOptions.outputPath)


    def _start_preview_playback(self, path: str) -> None:
        """Play the finished preview file through our own decoder into the preview pane.

        Replaces the old `QMediaPlayer` stub, which handed a plain QWidget to `setVideoOutput` (no surface at all) and
        wired the scrollbar with two different time units - it could not have shown a frame.
        """
        from .video.decode import VideoDecoder

        try:
            self._stop_preview_playback()
            decoder = VideoDecoder(pixel_format="rgb24")
            decoder.frame_ready.connect(self._on_preview_frame)
            decoder.opened.connect(self._on_preview_opened)
            decoder.error.connect(lambda message: log(f"preview decoder: {message}"))
            if not decoder.open(str(path)):
                log(f"preview decoder: cannot open {path}")
                return
            self._preview_decoder = decoder
            pane = self.parent.previewLabel
            if hasattr(pane, "seekRequested"):
                try:
                    pane.seekRequested.connect(self._on_preview_seek)
                except Exception:
                    pass
            decoder.play()
        except Exception as exc:  # a preview that cannot play must not take the render down
            log(f"preview playback failed: {exc}")

    def _stop_preview_playback(self) -> None:
        decoder = getattr(self, "_preview_decoder", None)
        if decoder is not None:
            try:
                decoder.close()
            except Exception:
                pass
            self._preview_decoder = None

    def _on_preview_seek(self, fraction: float) -> None:
        decoder = getattr(self, "_preview_decoder", None)
        duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
        if decoder is not None and duration > 0:
            decoder.seek(max(0.0, min(1.0, float(fraction))) * duration)

    def _on_preview_opened(self, duration: float, fps: float, frames_total: float) -> None:
        self._preview_duration = float(duration) or 0.0

    def _on_preview_frame(self, frame) -> None:
        """One decoded frame from a finished preview: RGB24 bytes -> QImage -> the pane."""
        from PySide6.QtGui import QImage

        image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB888).copy()
        pane = self.parent.previewLabel
        if getattr(self.parent, "preview_label_is_surface", False):
            pane.set_frame(image)
            duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
            pane.set_progress(rendered=frame.pts, total=duration, playhead=frame.pts, duration=duration)
        else:
            pixmap = QtGui.QPixmap.fromImage(image)
            pane.setPixmap(self.getRoundedPixmap(pixmap, corner_radius=10))

    def onRenderCompletion(self):
        self.eta = 0
        self.fps = 0
        try:
            self.renderProcess.wait()
        except Exception:
            pass
        # Have to swap the visibility of these here otherwise crash for some reason
        if (
            self.settings.settings["discord_rich_presence"] == "True"
        ):  # only close if it exists
            self.discordRPC.closeRPC()
        try:
            self.workerThread.stop()
            self.workerThread.quit()
            self.workerThread.wait()
        except Exception:
            pass  # pass just incase internet error caused a skip

    def getRoundedPixmap(self, pixmap, corner_radius):
        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)  # type: ignore

        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, size.width(), size.height(), corner_radius, corner_radius
        )

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        rounded_pixmap = QPixmap(size)
        rounded_pixmap.fill(Qt.transparent)  # type: ignore

        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore
        painter.drawPixmap(0, 0, mask)
        painter.end()

        return rounded_pixmap

    def modelNameToFile(self):
        pass

    def syncRenderConsole(self) -> None:
        """Push the render log into the console INCREMENTALLY.

        `updateProcessTab` runs on the update thread's 100 ms clock, and the old body re-set the whole accumulated log on
        every tick. The toolkit console re-wraps what it is given, so the cost grew with the log: after a couple of
        minutes a single tick took longer than the tick interval and the GUI thread never returned to the event loop -
        the window froze with the render still going and stayed pegged at ~80 % CPU afterwards (py-spy: setPlainText ->
        set_lines -> _wrap_line, owner 2026-11-22).

        Append what is new, and rebuild only when the log was replaced (a new render, a reset): the prefix check makes
        that decision O(1) on the common path.
        """
        console = self.parent.renderOutput
        entries = list(self.renderLogLines or [])
        sent = getattr(self, "_console_sent_entries", [])
        if len(entries) >= len(sent) and entries[:len(sent)] == sent:
            new = entries[len(sent):]
            if new:
                console.append_console_lines(self._console_lines(new))
                self._console_sent_entries = entries
            return
        console.set_lines(self._console_lines(entries))
        self._console_sent_entries = entries

    @staticmethod
    def _console_lines(entries) -> list["ConsoleLine"]:
        """Map the filter's levels onto the console's paint slots (colour + emphasis), from theme tokens only."""
        from PySide6.QtGui import QColor

        from sli_ui_toolkit.theme import ThemeManager

        manager = ThemeManager.get_instance()

        def token(name: str, fallback: str) -> QColor:
            try:
                colour = QColor(manager.get_color(name))
            except Exception:
                colour = QColor()
            return colour if colour.isValid() else QColor(fallback)

        red = token("BrightText", "#ff0000")
        accent = token("accent", "#0078d4")
        from sli_ui_toolkit.ui.widgets.composite.console_view import ConsoleLine

        lines: list[ConsoleLine] = []
        for entry in entries:
            level = getattr(entry, "level", "info")
            text = getattr(entry, "text", str(entry))
            if level == "error":
                lines.append(ConsoleLine(text, QColor(red), True))
            elif level == "warn":
                lines.append(ConsoleLine(text, None, True))
            elif level == "phase":
                lines.append(ConsoleLine(text, QColor(accent), True))
            else:
                lines.append(ConsoleLine(text))
        return lines

    def updateProcessTab(self, qimage: QtGui.QImage):
        """
        Called by the worker QThread, and updates the GUI elements: Progressbar, Preview, FPS
        """

        if self.renderTextOutputList is not None:
            self.parent.progressBar.setRange(
                0,
                # only set the range to multiply the frame count if the method is interpolate
                self.max_value
            )
            self.syncRenderConsole()
            try:
                options = getattr(self, "currentRenderOptions", None)
                if options is not None and self.max_value:
                    self.parent.renderQueue.update_progress(options, self.currentFrame / max(1, self.max_value))
                    self.refreshQueueRows()
            except Exception:
                pass
            self.parent.progressBar.setValue(self.currentFrame)
            self.setProgressDetail(self.progressDetailText())
            if self.fps != 0:
                self.parent.FPS.setVisible(True)
                self.parent.FPS.setText(f"FPS: {self.fps}")
            else:
                self.parent.FPS.setVisible(False)
            if self.eta != 0:
                self.parent.ETA.setVisible(True)
                self.parent.ETA.setText(f"ETA: {self.eta}")
            else:
                self.parent.ETA.setVisible(False)
            self.parent.STATUS.setText(f"Status: {self.status}")
        if not qimage.isNull():
            preview = self.parent.previewLabel
            if getattr(self.parent, "preview_label_is_surface", False):
                # The surface fits, zooms and letterboxes itself and always keeps the frame's own resolution; the
                # backend's thumbnail is what a live render can afford (the full-resolution path is the preview file).
                preview.set_frame(qimage)
                preview.set_progress(self.currentFrame, self.max_value)
            else:
                # The backend already shrank the frame to the preview box, so nothing 4K ever
                # reaches this thread. The thumbnail keeps the picture's aspect ratio (640x480
                # for a 4:3 source), so it can still be a little larger than the panel: fit it
                # down when it is, never scale it up.
                area = preview.size()
                if qimage.width() > area.width() or qimage.height() > area.height():
                    qimage = qimage.scaled(
                        area, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                pixmap = QtGui.QPixmap.fromImage(qimage)
                roundedPixmap = self.getRoundedPixmap(pixmap, corner_radius=10)
                preview.setPixmap(roundedPixmap)

    def build_command(self, renderOptions: RenderOptions):
        if (
            renderOptions.backend == "pytorch (cuda)"
            or renderOptions.backend == "pytorch (rocm)"
            or renderOptions.backend == "pytorch (xpu)"
            or renderOptions.backend == "pytorch (mps)"
        ):
            renderOptions.backend = (
                "pytorch"  # pytorch is the same for both cuda and rocm
            )

        command = [
            f"{PYTHON_EXECUTABLE_PATH}",
            "-W",
            "ignore",
            os.path.join(BACKEND_PATH, "rve-backend.py"),
            "-i",
            renderOptions.inputFile,
            "-o",
            renderOptions.outputPath,
            "-b",
            f"{renderOptions.backend}",
            "--precision",
            f"{self.settings.settings['precision']}",
            # the run report is written next to the output, always: it is what makes "it got slower" a
            # number with a commit and a card (python3 -m harness.ingest <report>)
            "--report",
            f"{os.path.splitext(renderOptions.outputPath)[0]}.report.json",
            "--custom_encoder",
            f" {renderOptions.encoderCommand} ",
            "--tensorrt_opt_profile",
            f"{self.settings.settings['tensorrt_optimization_level']}",
            "--pause_shared_memory_id",
            f"{PAUSED_STATE_SHARED_MEMORY_ID}",
            "--ncnn_gpu_id",
            f"{self.settings.settings['ncnn_gpu_id']}",
            "--pytorch_gpu_id",
            f"{self.settings.settings['pytorch_gpu_id']}",
            "--cwd",
            f"{CWD}",
            "--ffmpeg_path",
            f"{FFMPEG_PATH}",

        ]

        if renderOptions.upscaleModelFile:
            modelPath = os.path.join(MODELS_PATH, renderOptions.upscaleModelFile)
            if renderOptions.upscaleModelArch == "custom":
                modelPath = os.path.join(
                    CUSTOM_MODELS_PATH, renderOptions.upscaleModelFile
                )
            command += [
                "--upscale_model",
                modelPath,
            ]
            command += ["--override_upscale_scale", f"{renderOptions.overrideUpscaleScale}"]
            
        if renderOptions.tilingEnabled:
            command += [
                "--tilesize",
                f"{renderOptions.tilesize}",
            ]
            

        if renderOptions.interpolateModelFile:
            command += [
                "--interpolate_model",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.interpolateModelFile,
                ),
                "--interpolate_factor",
                f"{renderOptions.interpolateTimes}",
            ]
            if renderOptions.sloMoMode:
                command += [
                    "--slomo_mode",
                ]
            if renderOptions.dyanmicScaleOpticalFlow:
                command += [
                    "--dynamic_scaled_optical_flow",
                ]
            if renderOptions.ensemble:
                command += [
                    "--ensemble",
                ]
        
        if renderOptions.deblurModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.deblurModelFile,
                ),
            ]
        if renderOptions.denoiseModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.denoiseModelFile,
                ),
            ]
        if renderOptions.decompressModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.decompressModelFile,
                ),
            ]

        if self.settings.settings["auto_border_cropping"] == "True":
            command += [
                "--border_detect",
            ]

        if self.settings.settings["dynamic_tensorrt_engine"] == "True":
            command += [
                "--tensorrt_dynamic_shapes",
            ]

        # always explicit, even though "best" is the backend default, and tolerant of an
        # older settings.txt that predates the key.
        quality_preset = self.settings.settings.get("quality_preset", "best")
        if quality_preset not in self.settings.allowedSettings["quality_preset"]:
            quality_preset = "best"
        command += [
            "--quality_preset",
            f"{quality_preset}",
        ]

        # the backend defaults to reuse ON; the switch is therefore an opt-out, and it is passed explicitly so
        # an old settings.txt cannot silently mean something else than what the checkbox shows
        if self.settings.settings.get("temporal_reuse", "True") == "False":
            command += ["--no_temporal_reuse"]
        # "by cel": compare the drawing, not the film (3 px low-pass + the scan's translation compensated, and
        # the reused frame is moved by that estimate). Passed only when chosen - the backend's default is the
        # shipped statistic, so an old settings.txt that predates the key cannot mean something else.
        elif self.settings.settings.get("temporal_reuse_statistic", "mean_abs") == "coarse":
            command += ["--temporal_reuse_statistic", "coarse"]

        if self.settings.settings["preview_enabled"] == "True":
            command += [
                "--preview_shared_memory_id",
                f"{IMAGE_SHARED_MEMORY_ID}",
                "--preview_size",
                f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}",
            ]

        if self.settings.settings["scene_change_detection_enabled"] == "False":
            command += ["--scene_detect_method", "none"]
        else:
            command += [
                "--scene_detect_method",
                self.settings.settings["scene_change_detection_method"],
                "--scene_detect_threshold",
                self.settings.settings["scene_change_detection_threshold"],
            ]
            if renderOptions.sceneChangeModelFile:
                command += [
                    "--scene_detect_model",
                    os.path.join(
                        MODELS_PATH,
                        renderOptions.sceneChangeModelFile,
                    ),
                ]

        if renderOptions.benchmarkMode:
            command += ["--benchmark"]

        if self.settings.settings["uhd_mode"] == "True":
            if renderOptions.videoWidth > 1920 or renderOptions.videoHeight > 1080:
                command += ["--UHD_mode"]
                log("UHD mode enabled")

        if self.isOverwrite:
            command += ["--overwrite"]

        if renderOptions.hdrMode:
            command += ["--hdr_mode"]



        return command