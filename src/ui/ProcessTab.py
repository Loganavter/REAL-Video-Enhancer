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
from .render_command import build_command as build_render_command
from ..DiscordRPC import DiscordRPC
from ..ModelHandler import getModels
from .RenderQueue import RenderOptions
from .pages.handles import WindowHandles


#: The read side of the backend's one crop record (`backend/src/RenderVideo.py: format_borders_line`).
#: The pattern lives inside the parser so the shipped function is exactly what the harness executes: no
#: module-level regex to re-create in a test, and no second copy of the format to drift.
def parse_borders_line(line: object) -> tuple[int, int, int, int] | None:
    """``(w, h, x, y)`` from a render's ``BORDERS w=.. h=.. x=.. y=..`` line, or ``None``.

    ``None`` for anything that is not exactly that record (a truncated line, an extra field, a non-numeric
    value, an empty or missing line): the caller then keeps the preview's uncropped decode - today's
    behaviour - instead of applying half a box. ``BACKEND: `` is tolerated because the backend's ``log()``
    prefixes its records; the bare print this record uses does not. ``fullmatch`` is the point: a line that
    carries anything else is not this record, and guessing around noise crops the wrong rectangle.
    """
    match = re.fullmatch(r"(?:BACKEND:\s+)?BORDERS w=(\d+) h=(\d+) x=(\d+) y=(\d+)", str(line).strip())
    if match is None:
        return None
    return tuple(int(group) for group in match.groups())


def crop_region(box, source_width: int, source_height: int) -> tuple[int, int, int, int] | None:
    """``(x, y, w, h)`` for a QImage crop of a ``source_width``x``source_height`` frame, or ``None``.

    ``box`` is the backend's order (``w, h, x, y``); the answer is QImage's (``x, y, w, h``). A box whose
    rectangle runs off the source (a stale ``.crop.json`` from another resolution, a scaled preview) is
    refused as a whole: a partially sampled source would show a picture shifted against the render, which
    reads as a render bug rather than as a stale box.
    """
    if box is None or len(tuple(box)) != 4:
        return None
    width, height, x, y = (int(value) for value in box)
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        return None
    if x + width > int(source_width) or y + height > int(source_height):
        return None
    return x, y, width, height


class ProcessTab:
    def __init__(self, parent, settings: Settings):
        self.parent = parent
        #: The typed page surface (`window.page_handles`): every control this glue touches is a required field on
        #: one of its group dataclasses, so a page that stops building one fails at build time, not at first click.
        self.page_handles: WindowHandles = parent.page_handles
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
        #: The crop the CURRENT job's backend reported (`BORDERS w/h/x/y`), or ``None`` until a render says.
        #: The LMB compare decodes the source with this box, so it shows the same framing the renderer used.
        self._source_crop = None
        #: True once "the box is unknown" was logged for this job - one line per render, not per LMB press.
        self._source_crop_unknown_logged = False
        #: The video pane whose strip signals are already wired. Wired once per pane object, because the live
        #: tail and the finished-file playback both need them and two connections would double every seek.
        self._preview_strip_wired_for = None
        #: The preview file this job wants followed, and whether the GUI tick has built the reader for it yet.
        #: The request is made from the render's worker thread, which has no Qt event loop (see `_start_live_tail`).
        self._live_tail_path = None
        self._live_tail_wanted = False
        self._live_tail = None
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
        self.populateModels(self.page_handles.general.backendComboBox.currentText())
        self.setStatus("Idle")

    def setStatus(self, text: str) -> None:
        """The one place that decides what `STATUS:` says - never leave a stale phase on screen."""
        self.status = str(text)
        try:
            self.page_handles.proc.STATUS.setText(f"Status: {self.status}")
        except Exception:
            pass

    def setProgressDetail(self, text: str) -> None:
        try:
            self.page_handles.proc.progressDetail.setText(str(text))
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
        self.page_handles.general.interpolateModelComboBox.clear()
        self.page_handles.general.upscaleModelComboBox.clear()
        self.page_handles.general.deblurModelComboBox.clear()
        self.page_handles.general.denoiseModelComboBox.clear()
        self.page_handles.general.decompressModelComboBox.clear()
        self.page_handles.general.interpolateModelComboBox.addItems(
            list(interpolateModels.keys())
        )
        self.page_handles.general.interpolateModelComboBox.setCurrentIndex(len(list(interpolateModels.keys()))-1)
        self.page_handles.general.upscaleModelComboBox.addItems(list(upscaleModels.keys()))
        self.page_handles.general.deblurModelComboBox.addItems(list(deblurModels.keys()))
        self.page_handles.general.denoiseModelComboBox.addItems(list(denoiseModels.keys()))
        self.page_handles.general.decompressModelComboBox.addItems(list(decompressModels.keys()))
    def onTilingSwitch(self):
        if self.page_handles.advanced.tilingCheckBox.isChecked():
            self.page_handles.advanced.tileSizeContainer.setVisible(True)
            self.tileDownAnimationHandler.dropDownAnimation(
                self.page_handles.advanced.tileSizeContainer
            )
        else:
            self.tileUpAnimationHandler.moveUpAnimation(self.page_handles.advanced.tileSizeContainer)
            self.page_handles.advanced.tileSizeContainer.setVisible(False)

    def openOutputFolderInExplorer(self):
        output_folder = os.path.dirname(self.page_handles.general.outputFileText.text())
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

        widget = self.page_handles.queue.renderQueueListWidget
        widget.set_row_painter(queue_row_painter(self.parent.renderQueue))
        widget.set_row_tooltip_provider(queue_tooltip(self.parent.renderQueue))

    def refreshQueueRows(self) -> None:
        """Repaint the queue after a state change (a tick, a finish): the painter reads the model, no row rebuild."""
        try:
            self.page_handles.queue.renderQueueListWidget.update()
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

        widget = self.page_handles.queue.renderQueueListWidget

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
        self.page_handles.general.addToRenderQueueButton.clicked.connect(self.parent.addToRenderQueue)
        self.installQueueMenu()
        self.installQueueRows()
        self.page_handles.general.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.page_handles.general.batchSelectButton.clicked.connect(self.parent.openBatchFiles)
        self.page_handles.general.inputFileText.textChanged.connect(self.parent.loadVideo)
        self.page_handles.general.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        self.page_handles.general.openOutputFolderButton.clicked.connect(self.openOutputFolderInExplorer)
        # connect render button
        self.page_handles.proc.startRenderButton.clicked.connect(self.parent.startRender)
        # set tile size visible to false by default
        self.page_handles.advanced.tileSizeContainer.setVisible(False)
        # set slo mo container visable to false by default
        
        self.page_handles.advanced.interpolateContainer_2.setVisible(False)
        # connect up tilesize container visiable
        self.page_handles.advanced.tilingCheckBox.stateChanged.connect(self.onTilingSwitch)

        self.page_handles.general.interpolationMultiplierSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )

        self.page_handles.general.upscaleModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.page_handles.general.upscaleScaleSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.page_handles.general.interpolateModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.page_handles.general.decompressModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.page_handles.general.interpolateCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.page_handles.general.upscaleCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.page_handles.general.deblurCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.page_handles.general.denoiseCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.page_handles.general.decompressCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)   
        self.page_handles.general.sloMoModeCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)  
        self.page_handles.general.backendComboBox.currentIndexChanged.connect(
            lambda: self.populateModels(self.page_handles.general.backendComboBox.currentText())
        )
        self.page_handles.encoder.EncoderCommand.textChanged.connect(lambda: self.page_handles.encoder.EncoderCommand.setFixedWidth(max(50, QFontMetrics(self.page_handles.encoder.EncoderCommand.font()).horizontalAdvance(self.page_handles.encoder.EncoderCommand.text()) + 10)))
        # connect up pausing
        hide_layout_widgets(self.page_handles.proc.onRenderButtonsContiainer)
        self.page_handles.proc.pauseRenderButton.clicked.connect(self.pauseRender)
        self.page_handles.proc.killRenderButton.clicked.connect(self.killRenderProcess)

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.userKilled = True
            self.renderProcess.terminate()
        except AttributeError:
            log("No render process!")

    def pauseRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 1  # 1 = True
        hide_layout_widgets(self.page_handles.proc.onRenderButtonsContiainer)
        self.page_handles.proc.startRenderButton.setVisible(True)
        self.page_handles.proc.startRenderButton.setEnabled(True)
        self.setStatus("Paused")

    def resumeRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 0  # 0 = False
        from time import monotonic

        self._render_started_mono = monotonic()
        self.setStatus("Preparing")
        show_layout_widgets(self.page_handles.proc.onRenderButtonsContiainer)
        self.page_handles.proc.onRenderButtonsContiainer.setEnabled(True)
        self.page_handles.proc.startRenderButton.setVisible(False)

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
        show_layout_widgets(self.page_handles.proc.onRenderButtonsContiainer)
        self.page_handles.proc.startRenderButton.setVisible(False)
        self.page_handles.proc.startRenderButton.clicked.disconnect()
        self.page_handles.proc.startRenderButton.clicked.connect(self.resumeRender)

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
            # A job's crop belongs to THAT job: the next render's box arrives in its own log, and until it
            # does the compare must not reuse the previous job's rectangle.
            self._source_crop = None
            self._source_crop_unknown_logged = False
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
            if renderOptions.isPreview:
                self._start_live_tail(renderOptions.outputPath)
            textOutput = []
            self._log_filter = RenderLogFilter()
            self.renderLogLines = []
            render_failure = ""
            # Read to EOF, not "until poll() says the process is gone": the old loop checked poll()
            # BEFORE handling the line it had just read, so the last lines of a dying render - the
            # `RENDER FAILED: ...` summary included - were dropped on the floor and the job ended
            # with no reason anywhere (2026-11-22). `readline` returns "" at EOF in text mode, which
            # is the sentinel that actually terminates this iterator.
            for line in iter(self.renderProcess.stdout.readline, ""):
                try:
                    line = str(line.strip())
                    if "BORDERS" in line:
                        # The crop the backend actually handed its reader, kept with this job so the LMB
                        # compare decodes the same rectangle. Parsed here, before the log filter, because the
                        # filter's job is what a person reads - this record is the job's own state.
                        box = parse_borders_line(line)
                        if box is not None:
                            self._source_crop = box
                    if "RENDER FAILED" in line:
                        # The backend's thread wrapper wrote its one summary line: this job is over,
                        # whatever else arrives belongs to a traceback the pane should show.
                        render_failure = line[len("BACKEND: "):] if line.startswith("BACKEND: ") else line
                        self.setStatus("Failed")
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
            code = self.return_codes[-1]
            ok = code in (0, 3221225477) and not render_failure
            if not ok and not render_failure:
                render_failure = f"render process exited with code {code} before the render finished"
                self.setStatus("Failed")
            if not ok:
                # Whatever the exit code, the row's hover line and the popup both read this: the
                # mark_finished(ok=False) path IS the error surface (src/ui/render_jobs.py state_text
                # appends state.error to the row's line).
                log(f"render failed: {render_failure}")
            summary = self._log_filter.hidden_summary()
            if summary is not None:
                textOutput.append(summary.text)
                self.renderLogLines.append(summary)
                self.renderTextOutputList = textOutput.copy()
            for line in textOutput:
                if len(line) > 2:
                    log(line)

            renderQueue.mark_finished(renderOptions, ok=ok, error="" if ok else render_failure)
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
        hide_layout_widgets(self.page_handles.proc.onRenderButtonsContiainer)
        self.page_handles.proc.startRenderButton.setEnabled(True)
        self.page_handles.general.outputFileText.setEnabled(True)
        self.page_handles.general.inputFileText.setEnabled(True)
        if hasattr(self.page_handles.proc.previewLabel, "set_frame"):
            self.page_handles.proc.previewLabel.set_frame(None)
        else:
            self.page_handles.proc.previewLabel.clear()
        self.page_handles.proc.startRenderButton.clicked.disconnect()
        self.page_handles.proc.startRenderButton.clicked.connect(self.parent.startRender)
        self.parent.enableProcessPage()
        self.page_handles.proc.startRenderButton.setVisible(True)
        self._render_started_mono = None
        self.setProgressDetail("")
        finished_ok = all(code == 0 or code == 3221225477 for code in self.return_codes) or self.userKilled
        self.setStatus("Idle" if finished_ok else "Failed")
        # The failure line has to be ON SCREEN, not only in the list: the pane is fed by the update
        # thread's 100 ms tick and the last lines of a dying render can arrive after its last tick
        # (2026-11-22: the process page showed `Status: Rendering` with no error). This runs on the
        # GUI thread, so pushing the accumulated log here is legal and cannot be skipped.
        try:
            self.syncRenderConsole()
        except Exception:
            pass
        self.page_handles.proc.FPS.setText("FPS: ")
        self.page_handles.proc.ETA.setText("ETA: ")

        self._restoreQueueAfterPreview()
        if self.currentRenderOptions.isPreview:
            self._stop_live_tail()
            self._start_preview_playback(self.currentRenderOptions.outputPath)


    def _start_live_tail(self, path: str) -> None:
        """Ask for the live tail; the pane's own GUI tick creates it.

        This is the live preview: the renderer pays nothing for it (it writes the file anyway), the pane decodes what
        has been produced so far, and the strip's render head tracks the produced range.

        It is called from the render's worker (`threading.Thread`, ProcessTab.py:429) - a thread with no Qt event
        loop. A `QTimer` built here never fires, and a signal connected here is queued to a thread that never runs
        its queue: both mistakes were live at once (measured 2026-11-23: the pane showed the 640x360 thumbnail for a
        whole 12 s preview and a strip press during the render did nothing, while the seek wiring and the timer looked
        present in the code). So this only records the request; `_ensure_live_tail` runs on the GUI thread from
        `updateProcessTab`, which is the pane's 100 ms clock.
        """
        self._stop_live_tail()
        self._live_tail_path = str(path)
        self._live_tail_wanted = True

    def _ensure_live_tail(self) -> None:
        """Build the tail and wire the strip - on the GUI thread, once per job (see `_start_live_tail`)."""
        if not getattr(self, "_live_tail_wanted", False) or getattr(self, "_live_tail", None) is not None:
            return
        path = getattr(self, "_live_tail_path", None)
        if not path:
            return
        from .video.tail import LiveTailThread

        try:
            tail = LiveTailThread(path)
            tail.frame_ready.connect(self._on_preview_frame)
            tail.error.connect(lambda message: log(f"preview tail: {message}"))
            # The strip's signals too: a press must seek the tail, and the LMB compare must decode the source at the
            # current playhead. Connected here (GUI thread) or the seek lands in a queue nobody drains.
            self._wire_preview_strip()
            self._live_tail = tail
            self._live_tail_wanted = False
            tail.start()
        except Exception as exc:  # a preview that cannot follow must not take the render down
            self._live_tail_wanted = False
            log(f"preview tail failed: {exc}")

    def _stop_live_tail(self) -> None:
        self._live_tail_wanted = False
        tail = getattr(self, "_live_tail", None)
        if tail is not None:
            try:
                tail.stop_and_wait()
            except Exception:
                pass
            self._live_tail = None

    def _wire_preview_strip(self) -> None:
        """Connect the pane's strip signals once, whoever owns the picture (live tail or finished file).

        Both paths need them: live, `seekRequested` must move the playhead inside what the renderer has already
        written; finished, it seeks the file decoder. A second connection per pane would run every seek twice.
        """
        pane = self.page_handles.proc.previewLabel
        if getattr(self, "_preview_strip_wired_for", None) is pane:
            return
        if hasattr(pane, "seekRequested"):
            try:
                pane.seekRequested.connect(self._on_preview_seek)
            except Exception:
                pass
        if hasattr(pane, "originalRequested"):
            try:
                pane.originalRequested.connect(self._on_original_requested)
            except Exception:
                pass
        self._preview_strip_wired_for = pane

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
            self._wire_preview_strip()
            decoder.play()
        except Exception as exc:  # a preview that cannot play must not take the render down
            log(f"preview playback failed: {exc}")

    def _stop_preview_playback(self) -> None:
        self._stop_source_decoder()
        decoder = getattr(self, "_preview_decoder", None)
        if decoder is not None:
            try:
                decoder.close()
            except Exception:
                pass
            self._preview_decoder = None

    def _on_preview_seek(self, fraction: float) -> None:
        """A press or drag on the strip seeks whoever owns the picture.

        Live, that is the tail: it repositions inside what the renderer has already written, and the next poll
        (120 ms) shows the target. A finished file seeks its own decoder. The strip reports a share of the WHOLE
        timeline, so while the timeline length is unknown the produced share stands in for it - which is the same
        assumption the strip itself makes when it fills its white band.
        """
        fraction = max(0.0, min(1.0, float(fraction)))
        live = getattr(self, "_live_tail", None)
        if live is not None:
            edge = float(getattr(live, "produced_seconds", 0.0) or 0.0)
            if edge <= 0.0:
                return                      # nothing is written yet, so there is nothing to seek to
            produced = self._live_produced_fraction()
            timeline = edge / produced if produced > 0.0 else edge
            live.seek(fraction * timeline)
            return
        decoder = getattr(self, "_preview_decoder", None)
        duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
        if decoder is not None and duration > 0:
            decoder.seek(fraction * duration)

    def _live_produced_fraction(self) -> float:
        """The share of the timeline the renderer has produced, from the same counters the strip fills with."""
        rendered = float(self.currentFrame or 0)
        total = float(self.max_value or 0)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, rendered / total))

    def _live_duration_estimate(self) -> float:
        """The timeline length the produced share implies, so the strip can draw a playhead mid-render.

        `edge seconds / share produced` is exact while the renderer emits its frames uniformly, which is what the
        strip's own `rendered/total` fill already assumes. A finished file reports its real duration instead; a
        live render with no frame yet reports 0, and the strip hides a marker it cannot place honestly.
        """
        live = getattr(self, "_live_tail", None)
        if live is None:
            return 0.0
        edge = float(getattr(live, "produced_seconds", 0.0) or 0.0)
        produced = self._live_produced_fraction()
        if edge <= 0.0 or produced <= 0.0:
            return 0.0
        return edge / produced

    def _on_original_requested(self, seconds: float) -> None:
        """Decode ONE source frame at the playhead for the LMB compare (the input file, never the renderer).

        The image is what the platform's decoder produces from the input, which is the honest comparison: the renderer's
        own source buffer would need a second channel and would make the render pay for a preview-only action. It is
        then cropped to the box the render reported (`BORDERS`, parsed out of this job's output) - see
        `_on_original_frame`.
        """
        decoder = getattr(self, "_source_decoder", None)
        try:
            if decoder is None:
                from .video.decode import VideoDecoder

                decoder = VideoDecoder(pixel_format="rgb24")
                decoder.frame_ready.connect(self._on_original_frame)
                decoder.error.connect(lambda message: log(f"source decoder: {message}"))
                path = getattr(self.currentRenderOptions, "inputFile", "") or ""
                if not decoder.open(str(path)):
                    log(f"source decoder: cannot open {path}")
                    return
                self._source_decoder = decoder
            decoder.request_frame(max(0.0, float(seconds)))
        except Exception as exc:
            log(f"original frame failed: {exc}")

    def _on_original_frame(self, frame) -> None:
        """The source frame from the decoder, cropped to the box THE RENDER used, then handed to the pane.

        The crop is applied here rather than in the decoder because the decoder exposes no crop and the surface
        (`src/ui/widgets/video_preview.py`) is another lane's file: `QImage.copy` of the same rectangle is the same
        pixels, and the surface then scales it to the rendered frame's size - which is what makes the two pictures
        comparable. Without the crop the compare shows the whole source (black bars included) and the eye reads the
        letterbox as a scale difference.
        """
        from PySide6.QtGui import QImage

        image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB888).copy()
        box = getattr(self, "_source_crop", None)
        region = crop_region(box, image.width(), image.height())
        if region is None:
            # Today's picture (the whole source) is the fallback, and the log says once why: a silent
            # uncropped frame is indistinguishable from "crop detection found nothing".
            if box is None:
                self._log_source_crop_unknown("this render's output carried no readable BORDERS line")
            else:
                self._log_source_crop_unknown(
                    f"the render reported {box} but the source frame is {image.width()}x{image.height()}")
        else:
            image = image.copy(*region)
        pane = self.page_handles.proc.previewLabel
        if hasattr(pane, "set_original"):
            pane.set_original(image)

    def _log_source_crop_unknown(self, reason: str) -> None:
        """One line per job when the compare cannot crop: the preview then shows the uncropped source."""
        if getattr(self, "_source_crop_unknown_logged", False):
            return
        self._source_crop_unknown_logged = True
        log(f"source crop: unknown ({reason}); the LMB compare shows the uncropped source")

    def _stop_source_decoder(self) -> None:
        decoder = getattr(self, "_source_decoder", None)
        if decoder is not None:
            try:
                decoder.close()
            except Exception:
                pass
            self._source_decoder = None

    def _on_preview_opened(self, duration: float, fps: float, frames_total: float) -> None:
        self._preview_duration = float(duration) or 0.0

    def _on_preview_frame(self, frame) -> None:
        """One decoded frame from a finished preview: RGB24 bytes -> QImage -> the pane."""
        from PySide6.QtGui import QImage

        image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB888).copy()
        pane = self.page_handles.proc.previewLabel
        if hasattr(self.page_handles.proc.previewLabel, "set_frame"):
            pane.set_frame(image)
            duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
            live_tail = getattr(self, "_live_tail", None)
            running = self.renderProcess is not None and self.renderProcess.poll() is None
            if live_tail is not None or running:
                # Live: the white part is what the RENDERER has produced (`currentFrame/max_value`), the playhead is
                # where the tail stands - which lags the edge after a seek, so the strip must show it. The duration is
                # the estimate the produced share implies; without it the strip would draw no playhead at all and a
                # seek would be invisible until the renderer caught up with the target.
                pane.set_progress(rendered=self.currentFrame, total=self.max_value,
                                  playhead=frame.pts, duration=self._live_duration_estimate())
            else:
                # Finished file: everything exists, so the strip is fully produced; the playhead is the only marker.
                pane.set_progress(rendered=duration, total=duration,
                                  playhead=frame.pts, duration=duration)
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
        console = self.page_handles.logs.renderOutput
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

        # The pane's 100 ms clock is also the live tail's start point: it is the only GUI-thread tick during a
        # render, and the tail must be built and wired there (see `_start_live_tail`).
        self._ensure_live_tail()

        if self.renderTextOutputList is not None:
            self.page_handles.proc.progressBar.setRange(
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
            self.page_handles.proc.progressBar.setValue(self.currentFrame)
            self.setProgressDetail(self.progressDetailText())
            if self.fps != 0:
                self.page_handles.proc.FPS.setVisible(True)
                self.page_handles.proc.FPS.setText(f"FPS: {self.fps}")
            else:
                self.page_handles.proc.FPS.setVisible(False)
            if self.eta != 0:
                self.page_handles.proc.ETA.setVisible(True)
                self.page_handles.proc.ETA.setText(f"ETA: {self.eta}")
            else:
                self.page_handles.proc.ETA.setVisible(False)
            self.page_handles.proc.STATUS.setText(f"Status: {self.status}")
        if not qimage.isNull():
            live_tail = getattr(self, "_live_tail", None)
            if live_tail is not None and getattr(live_tail, "published_frames", 0) > 0:
                # The pane belongs to the 4K picture the tail reads from the preview file. A 640x360 thumbnail
                # landing between two tail frames would stutter the resolution AND drag the picture back to the
                # renderer's edge - the thumbnail IS the edge - undoing a seek 100 ms after it was made. The text
                # above keeps updating on this tick; only the picture is skipped. (Until the tail's first frame
                # the thumbnail still fills the pane, so it is never blank while the decoder warms up.)
                return
            preview = self.page_handles.proc.previewLabel
            if hasattr(self.page_handles.proc.previewLabel, "set_frame"):
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
        """The backend command for one job.

        The assembly lives in `src/ui/render_command.py` (P3 of the seam plan): it is a function of the options and the
        settings, so the command can be asked for without a window. This method only supplies this tab's two pieces of
        state - the settings store and the overwrite answer - and nothing about the string changed.
        """
        return build_render_command(
            renderOptions,
            self.settings,
            is_overwrite=self.isOverwrite,
        )
