import subprocess
import os
from threading import Thread
import re
from time import sleep
from multiprocessing import shared_memory

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor
from PySide6.QtCore import Qt, QSize, QUrl

from .RenderQueue import RenderQueue
from ..render_jobs import short_duration
from .render_log import LogLine, RenderLogFilter
from . import shared_flag

from .AnimationHandler import AnimationHandler
from .QTcustom import (
    DEFAULT_DISPLAY_BOX,
    DISPLAY_STATE_SHARED_MEMORY_ID,
    SEEK_STATE_SHARED_MEMORY_ID,
    DisplayReader,
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
    PAUSED_STATE_SHARED_MEMORY_ID,
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
        #: A preview does not run away by itself: it stands paused on a frame until the transport is pressed (owner,
        #: 2026-11-23: "добавь паузу и чтобы изначально на ней и стояло"). One flag for both owners of the pane.
        self._preview_paused = True
        #: The media time of the last frame the pane was given, so a paused hand-over to the finished file stands
        #: where the picture already is instead of jumping to 0.
        self._preview_playhead = 0.0
        #: The export pane's display channel (`QTcustom.DisplayReader`): the block the GUI allocates before the backend
        #: starts and the renderer publishes into while `wanted` is 1. `None` when no render has run.
        self._display = None
        #: The box the block was allocated at (the pane's device pixels, or `DEFAULT_DISPLAY_BOX`); the command's
        #: `--display_size` has to be the same box, so it is kept for the whole render pass.
        self._display_box = DEFAULT_DISPLAY_BOX
        #: The playhead the standing compare (`sbs`/`split`) was last asked for: the 100 ms clock must not re-seek the
        #: source decoder for a picture that is already on screen.
        self._second_surface_pts = None
        #: A paused pane may still be shown ONE frame - the one a strip press asked for. While this is set the request
        #: byte stays 1 and the next published frame paints even though the transport is holding.
        self._display_seek_pending = False
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
        # The first entry is not a default, it is an accident of the file listing: with a 4x model on top, a 2x request
        # ran the 4x forward pass and downscaled its 5760x4320 output - 668.7 ms/frame against 75.2 ms/frame for the
        # installed 2x model on the same clip (owner, 2026-11-23: "4x SPAN model использовал бы futsuu"). A model whose
        # own scale is the requested one costs a fraction and produces the same picture at that scale.
        self.prefer_scale_matched_model()
        self.page_handles.general.deblurModelComboBox.addItems(list(deblurModels.keys()))
        self.page_handles.general.denoiseModelComboBox.addItems(list(denoiseModels.keys()))
        self.page_handles.general.decompressModelComboBox.addItems(list(decompressModels.keys()))
    def prefer_scale_matched_model(self) -> None:
        """Select the first upscale model whose own scale IS the requested one (a default, not a policy).

        Only the default: once the user picks a model, nothing here runs again until the backend changes - switching
        backends rebuilds the list and the old choice is gone anyway.
        """
        import re

        combo = self.page_handles.general.upscaleModelComboBox
        if combo.count() == 0:
            return
        remembered = str(self.settings.settings.get("last_upscale_model", "") or "")
        if remembered:
            index = combo.findText(remembered)
            if index >= 0:
                combo.setCurrentIndex(index)
                return
        try:
            wanted = int(round(float(self.page_handles.general.upscaleScaleSpinBox.value())))
        except Exception:
            return
        if wanted < 2:
            return
            return
        for index in range(combo.count()):
            match = re.match(r"\s*(\d+)x", combo.itemText(index))
            if match and int(match.group(1)) == wanted:
                combo.setCurrentIndex(index)
                return

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
        # No width is driven from the encoder command's text: the field belongs to its row and scrolls its own text
        # (`src/ui/pages/tab_6.py`). Sizing it to `fontMetrics().horizontalAdvance(text)` gave it a ~900 px width in a
        # ~480 px column, so it was painted outside the page and over the preview pane (owner, 2026-11-24; the gate is
        # `harness/tests/test_page_overflow.py`).
        # connect up pausing
        hide_layout_widgets(self.page_handles.proc.onRenderButtonsContiainer)
        self.page_handles.proc.pauseRenderButton.clicked.connect(self.pauseRender)
        self.page_handles.proc.killRenderButton.clicked.connect(self.killRenderProcess)
        # The preview's own transport: it stands PAUSED until this is pressed (owner, 2026-11-23), and the toggle's
        # checked state is the truth for both owners of the pane (the live tail and the finished-file decoder).
        # `toggled(bool)` rather than `clicked()`: the handler then receives the state the user asked for instead of
        # reading it back out of the widget - a read-back depends on when the toolkit lands its own toggle, and one
        # early run showed a click that left the button checked with nothing moved (2026-11-23, not reproduced after
        # the switch, and the state can no longer disagree with the signal that carries it).
        self.page_handles.proc.previewPauseButton.toggled.connect(self._on_preview_pause_toggled)
        self._sync_preview_transport()

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
        
        self.workerThread = UpdateGUIThread(parent=self)
        self.workerThread.tick.connect(self.updateProcessTab)
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
        # The previous job's picture must not keep playing into this one's pane: a finished preview left a file
        # decoder running, and its frames would land on top of the live render (and onto the transport's state).
        self._stop_preview_playback()
        self._stop_live_tail()
        # A render is an owner of the pane from its first thumbnail, so the transport must say so now rather than
        # keep a state left over from the previous job (an enabled button that holds nothing is the trap this closes).
        self._sync_preview_transport()
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

        # The pane's block must exist before the first backend does: the renderer only attaches and publishes into
        # what the GUI sized, so allocating after `Popen` would lose the first frames (and `wanted` would be a byte in
        # a block nobody owns). This runs on the GUI thread, where the pane's size and the reader's signal belong.
        # A preview render leaves `wanted` at 0 - its tail owns the pane - but the flags go on every job.
        self._prepare_display_state()

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

    # -- the export pane's display channel ----------------------------------------------------------

    def _pane_display_box(self) -> tuple[int, int]:
        """The box the block is sized for: the pane's own device pixels, or the probe's constant.

        The writer fits every frame into this box with the aspect kept (`DisplayStager.fit_box`), so the picture is
        never stretched - a wrong box only letterboxes. Device pixels are cheap to read here (one `QWidget` call on the
        GUI thread) and they are what makes the pane's own picture sharp instead of a scaled constant.
        """
        pane = getattr(self.page_handles.proc, "previewLabel", None)
        try:
            width = int(pane.width())
            height = int(pane.height())
            ratio = float(pane.devicePixelRatioF()) if hasattr(pane, "devicePixelRatioF") else 1.0
            width, height = int(width * ratio), int(height * ratio)
        except Exception:
            return DEFAULT_DISPLAY_BOX
        if width < 32 or height < 18:
            # Not laid out yet (a page built but never shown): the probe's box is the honest stand-in.
            return DEFAULT_DISPLAY_BOX
        # RGB888 rows are happy at any width, but even dimensions keep the two slots and the strip geometry stable.
        return width - (width & 1), height - (height & 1)

    def _prepare_display_state(self) -> None:
        """Allocate the pane's frame block (and the seek block) before the backend starts - GUI thread only.

        The GUI is the owner: the renderer attaches to this name and never creates it, so a render started without a
        window (or after it closed) pays nothing. One block per render pass; `_close_display_state` releases it.
        """
        self._close_display_state()
        box = self._pane_display_box()
        self._display_box = box
        try:
            reader = DisplayReader(
                DISPLAY_STATE_SHARED_MEMORY_ID,
                box=box,
                seek_name=SEEK_STATE_SHARED_MEMORY_ID,
            )
        except Exception as exc:  # a pane that cannot allocate must not take the export down
            log(f"display: cannot allocate the pane's block ({exc}); the export runs without it")
            self._display = None
            return
        if not reader.attached:
            log("display: the pane's block is not usable; the export runs without it")
            reader.release()
            self._display = None
            return
        reader.display_frame.connect(self._on_display_frame)
        self._display = reader
        # The strip and the compare need their signals for an export too: without this only a preview ever wire them,
        # and a strip press during an export reached nothing (the seek then had two routes, neither connected).
        self._wire_preview_strip()
        log(f"display: block {DISPLAY_STATE_SHARED_MEMORY_ID} at {box[0]}x{box[1]}, "
            f"seek {SEEK_STATE_SHARED_MEMORY_ID}")

    def _close_display_state(self) -> None:
        """Release the frame and seek blocks; safe twice, and safe when nothing was prepared."""
        reader = getattr(self, "_display", None)
        self._display = None
        if reader is not None:
            try:
                reader.release()
            except Exception:
                pass

    def _display_owns_pane(self) -> bool:
        """Whether the display feed is the pane's painter right now (never during a preview: its tail owns it)."""
        reader = getattr(self, "_display", None)
        return reader is not None and reader.attached and not bool(getattr(self, "isPreview", False))

    def _display_should_want(self) -> bool:
        """`wanted` = the pane is visible AND the transport is playing AND a render is publishing into the block.

        A paused picture must cost the renderer nothing: that is the one byte this whole channel exists to save.
        """
        if not self._display_owns_pane():
            return False
        if bool(getattr(self, "_preview_paused", True)) and not getattr(self, "_display_seek_pending", False):
            # A held picture costs nothing - except the one frame a strip press asked for while it is held.
            return False
        process = getattr(self, "renderProcess", None)
        if process is None or process.poll() is not None:
            return False
        pane = getattr(self.page_handles.proc, "previewLabel", None)
        try:
            return bool(pane.isVisible())
        except Exception:
            return True

    def _update_display_wanted(self) -> None:
        """Re-assert the request byte after anything that changes the answer (pause, job start, page hide)."""
        reader = getattr(self, "_display", None)
        if reader is None or not reader.attached:
            return
        reader.set_wanted(self._display_should_want())

    def _tick_display(self) -> None:
        """One 100 ms tick of the export feed: ask, read at most one new `seq`, paint (GUI thread)."""
        reader = getattr(self, "_display", None)
        if reader is None or not reader.attached:
            return
        wanted = self._display_should_want()
        reader.set_wanted(wanted)
        if not wanted:
            return
        reader.poll()          # emits `display_frame` only when the sequence moved

    def _on_display_frame(self, image) -> None:
        """The renderer's composited frame for an EXPORT's pane (one picture, one painter).

        A preview's pane is painted by its live tail, so a display frame is dropped there; while the transport holds a
        frame, display frames are dropped too and the picture stays. The progress is republished with every fresh
        frame because the strip's produced band is the render's own counters, not the picture's timestamp.
        """
        if image is None:
            return
        if not self._display_owns_pane():
            return
        holding = bool(getattr(self, "_preview_paused", True))
        if holding and not getattr(self, "_display_seek_pending", False):
            return
        pane = self.page_handles.proc.previewLabel
        if hasattr(pane, "set_frame"):
            pane.set_frame(image)
            if hasattr(pane, "set_progress"):
                pane.set_progress(
                    rendered=self.currentFrame,
                    total=self.max_value,
                    playhead=self._result_playhead(),
                    duration=self._clip_duration(),
                )
        else:
            pixmap = QtGui.QPixmap.fromImage(image)
            pane.setPixmap(self.getRoundedPixmap(pixmap, corner_radius=10))
        if holding:
            # That was the requested frame (a held pane gets no other published ones): hold again from here, and let
            # the renderer stop staging for a picture nobody is following.
            self._display_seek_pending = False
            self._update_display_wanted()

    def _clip_duration(self) -> float:
        """The clip's length in seconds, as far as the pane knows it: the opened file's duration, else frames / fps."""
        duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
        if duration > 0.0:
            return duration
        options = getattr(self, "currentRenderOptions", None)
        frames = float(getattr(options, "videoFrameCount", 0) or 0)
        fps = float(getattr(options, "videoFps", 0) or 0)
        if frames > 0.0 and fps > 0.0:
            return frames / fps
        return 0.0

    def _result_playhead(self) -> float:
        """Where the RESULT's playhead is: the tail's or file decoder's PTS, else the render's produced share."""
        playhead = float(getattr(self, "_preview_playhead", 0.0) or 0.0)
        if (getattr(self, "_live_tail", None) is not None
                or getattr(self, "_preview_decoder", None) is not None) and playhead > 0.0:
            return playhead
        total = float(self.max_value or 0)
        if total > 0.0:
            produced = max(0.0, min(1.0, float(self.currentFrame or 0) / total))
            return produced * self._clip_duration()
        return playhead

    def _tick_second_surface(self) -> None:
        """Feed the standing original in `sbs`/`split` from the source decoder, on the pane's own 100 ms clock.

        The surface also asks by itself when its playhead moves (`set_progress` -> `originalRequested`); this tick is
        what keeps the left picture at the result's playhead when the result is a render (whose display frames carry no
        timestamp), and `_second_surface_pts` makes the two routes ask for one decode, not two. `single` is untouched:
        there the source appears only while LMB is held.
        """
        pane = getattr(self.page_handles.proc, "previewLabel", None)
        view_mode = getattr(pane, "view_mode", None)
        if view_mode is None:
            return
        try:
            mode = view_mode()
        except Exception:
            return
        if mode == "single":
            self._second_surface_pts = None
            return
        has_frame = getattr(pane, "has_frame", None)
        if callable(has_frame):
            try:
                if not has_frame():
                    return
            except Exception:
                return
        playhead = self._result_playhead()
        if playhead <= 0.0:
            return
        previous = getattr(self, "_second_surface_pts", None)
        if previous is not None and abs(playhead - previous) < 1e-3:
            return
        self._second_surface_pts = playhead
        self._on_original_requested(playhead)

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
            # The previous job's playhead and duration say nothing about this clip, and the compare's clock must
            # start from this job's first picture rather than from a stale second.
            self._preview_playhead = 0.0
            self._preview_duration = 0.0
            self._second_surface_pts = None
            self._display_seek_pending = False
            # An EXPORT's pane follows the feed from its first published frame: its transport starts playing. A
            # PREVIEW keeps the owner's default (stands paused until the button is pressed) - that is the preview's
            # whole transport. The flag is read by the GUI tick, so setting it here is enough; the button paints on
            # the next `_sync_preview_transport`.
            if not self.isPreview and self._preview_paused:
                self._preview_paused = False
                self._sync_preview_transport()
            renderQueue.mark_rendering(renderOptions)
            self.refreshQueueRows()

            # The display block is allocated once for the render pass at the pane's box (see `run`), and its size is
            # in every job's command; here only the progress denominator belongs to the job.
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
            if self._tail_should_run(renderOptions):
                # A PREVIEW follows its own file: the tail decodes what the renderer has written so far, which is the
                # pane's only painter there. An export with a usable display block does NOT start it - see
                # `_tail_should_run` (measured 2026-11-24: two painters, one surface).
                self._start_live_tail(renderOptions.outputPath)
            else:
                # One picture, one painter: this job's pane is fed by the display block (or by nothing), and a tail
                # left over from the previous job in the same pass would paint on top of it.
                self._stop_live_tail()
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
        # The job is over: no backend publishes any more, so the block and the seek request are released here (the
        # GUI thread) rather than left for a tick that may never come - and whoever followed the job's file stops
        # with it (see `_end_job_painters`). The pane's own clear below follows.
        self._second_surface_pts = None
        self._display_seek_pending = False
        self._end_job_painters()
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
            self._start_preview_playback(self.currentRenderOptions.outputPath)

    def _end_job_painters(self) -> None:
        """Stop whoever followed the finished job's file, then release the block the renderer published into.

        A tail outliving its job keeps decoding a file nobody writes any more, and its frames land on the pane after
        the picture was finished - measured live (2026-11-24): the export's last display frame stood at 18.9 s while
        the tail, still running, painted its own lagging cluster at 8.97 s over it. The tail goes FIRST: the block is
        what a still-live tail's frames would have been painted over, and `_on_preview_frame` refuses frames once no
        owner is current, so a frame already queued by the stopped tail is not painted either.
        """
        self._stop_live_tail()
        self._close_display_state()

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

    def _refresh_held_progress(self) -> None:
        """While the preview is paused, keep the strip's PRODUCED band honest without touching the playhead.

        The pane's `rendered` is a snapshot taken when a frame was painted, so a paused preview froze the band at
        whatever the renderer had produced at the moment of the pause - and a seek could not reach what had been
        produced since. The playhead must not move: only the two produced counters are refreshed.

        An EXPORT's pane has no tail (its painter is the display feed, see `_tail_should_run`), so "a tail is running"
        is not the condition: the condition is that a painter owns the pane and left a progress snapshot to refresh.
        """
        if not getattr(self, "_preview_paused", False):
            return
        if (getattr(self, "_live_tail", None) is None
                and not self._display_owns_pane()):
            return
        pane = self.page_handles.proc.previewLabel
        progress = getattr(pane, "_progress", None)
        if not progress or len(progress) < 4:
            return
        playhead, duration = float(progress[2]), float(progress[3])
        pane.set_progress(rendered=self.currentFrame, total=self.max_value,
                          playhead=playhead, duration=duration)

    @staticmethod
    def _followable(renderOptions) -> bool:
        """Whether the file this job writes can be read while it is still being written.

        Matroska/TS are streamable; the mp4 family is not, because `+faststart` relocates the index at close (a preview
        mp4 is written fragmented and IS readable - the preview job says so through `isPreview`).
        """
        import os

        if getattr(renderOptions, "isPreview", False):
            return True
        extension = os.path.splitext(str(getattr(renderOptions, "outputPath", "") or ""))[1].lower()
        return extension not in (".mp4", ".mov", ".m4v")

    def _tail_should_run(self, renderOptions) -> bool:
        """Whether THIS job's pane follows the job's own output file with a tail - decided before it starts.

        One picture, one painter, decided structurally rather than by who publishes last (measured 2026-11-24, a
        495-frame 1440x1080 mkv export: the display feed and a tail both painted the pane, so its playhead jumped
        between the composited frame at 18.9 s and the tail's cluster at 8.97 s):

        * a PREVIEW follows its own file - the tail is the picture there and the strip's seek lives on it;
        * an EXPORT whose pane the display block feeds does not start a tail at all - the feed is that pane's painter;
        * an EXPORT whose block could not be allocated keeps the tail as its fallback: that is what stopped the pane
          going black on a plain render when the thumbnail channel was deleted (owner, 2026-11-24: "видео куда-то
          пропало"), and it cannot race the feed because there is no feed.
        """
        if not self._preview_enabled() or not self._followable(renderOptions):
            return False
        if getattr(renderOptions, "isPreview", False):
            return True
        return not self._display_owns_pane()

    def _preview_enabled(self) -> bool:
        """The user's "Preview Enabled" switch (`Settings -> GUI Options`).

        It used to gate the shared-memory thumbnail; with that channel gone it gates the one thing it can still mean:
        whether a preview render follows its own output file at full resolution.
        """
        try:
            return str(self.settings.settings.get("preview_enabled", "True")) == "True"
        except Exception:
            return True

    def _ensure_live_tail(self) -> None:
        """Build the tail and wire the strip - on the GUI thread, once per job (see `_start_live_tail`)."""
        if not getattr(self, "_live_tail_wanted", False) or getattr(self, "_live_tail", None) is not None:
            return
        path = getattr(self, "_live_tail_path", None)
        if not path:
            return
        from .video.tail import LiveTailThread

        try:
            tail = LiveTailThread(path, paused=self._preview_paused)
            tail.frame_ready.connect(self._on_preview_frame)
            tail.error.connect(lambda message: log(f"preview tail: {message}"))
            # The strip's signals too: a press must seek the tail, and the LMB compare must decode the source at the
            # current playhead. Connected here (GUI thread) or the seek lands in a queue nobody drains.
            self._wire_preview_strip()
            self._live_tail = tail
            self._live_tail_wanted = False
            tail.start()
            self._sync_preview_transport()
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
        self._sync_preview_transport()

    def _on_preview_pause_toggled(self, paused: bool) -> None:
        """Hold the preview on this frame, or let it follow the render again (the button's `toggled(bool)`).

        One state for both owners of the pane: the live tail simply stops polling (no decode, no I/O, the picture
        stands, and a strip press still repositions it), the finished-file decoder pauses on its frame. The state
        comes IN with the signal, so the handler cannot act on a half-applied toggle.
        """
        self._preview_paused = bool(paused)
        # A one-shot seek frame belongs to the state it was asked in: leaving or re-entering the hold ends it.
        self._display_seek_pending = False
        # One line per change, with the owner: the pane can be painted by three different things, and "the button did
        # nothing" needs to be answerable from log.txt alone.
        owner = "tail" if getattr(self, "_live_tail", None) is not None else (
            "file" if getattr(self, "_preview_decoder", None) is not None else "thumbnail")
        log(f"preview transport: {'paused' if self._preview_paused else 'playing'} (owner={owner})")
        tail = getattr(self, "_live_tail", None)
        if tail is not None:
            try:
                tail.set_paused(self._preview_paused)
            except Exception as exc:
                log(f"preview transport: {exc}")
        decoder = getattr(self, "_preview_decoder", None)
        if decoder is not None:
            try:
                decoder.pause() if self._preview_paused else decoder.play()
            except Exception as exc:
                log(f"preview transport: {exc}")
        # A held picture clears `wanted`: the renderer must stop staging frames for a pane that is not following
        # (the whole cost of the channel when nobody watches is one byte read per frame).
        self._update_display_wanted()
        self._sync_preview_transport()

    def _sync_preview_transport(self) -> None:
        """Paint the toggle's state, its tooltip and whether it can do anything at all (GUI thread only).

        The glyphs are the ACTION a press performs, like a player's own button: playing shows a pause symbol, paused
        shows a play symbol. `enabled` is honest about the idle pane - there is nothing to hold when nothing owns it.
        """
        button = getattr(self.page_handles.proc, "previewPauseButton", None)
        if button is None:
            return
        from PySide6.QtCore import QThread, QTimer

        if QThread.currentThread() != button.thread():
            # `_stop_live_tail` runs in the render's worker thread, which has no event loop to paint from: hand the
            # widget work to the button's own thread instead of writing to a widget cross-thread.
            QTimer.singleShot(0, button, self._sync_preview_transport)
            return
        paused = bool(getattr(self, "_preview_paused", True))
        try:
            # `emit=False`: painting the state must not look like a user toggle (that would call this handler again).
            button.setChecked(paused, emit=False)
        except TypeError:      # a toolkit older than the keyword
            button.blockSignals(True)
            try:
                button.setChecked(paused)
            finally:
                button.blockSignals(False)
        button.setToolTip(
            "Paused: the preview holds this frame. Press to follow the render."
            if paused else "Following the render. Press to hold this frame."
        )
        # A render paints the pane through the thumbnail even before a tail exists, so a running render is an owner
        # too: the transport must be able to hold THAT picture, and must not look pressable when nothing paints.
        process = getattr(self, "renderProcess", None)
        rendering = process is not None and process.poll() is None
        owns_picture = (getattr(self, "_live_tail", None) is not None
                        or getattr(self, "_preview_decoder", None) is not None
                        or rendering)
        button.setEnabled(bool(owns_picture))

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
            if self._preview_paused:
                # A paused preview stands where the picture already was (the render's edge after a live tail), not at
                # 0: `seek` publishes exactly one frame and stays paused.
                decoder.seek(max(0.0, float(getattr(self, "_preview_playhead", 0.0))))
            else:
                decoder.play()
            self._sync_preview_transport()
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
            return
        # EXPORT, third route: no tail and no file decoder, but a renderer is alive and publishing into the pane's
        # block. The strip's share becomes a newest-wins request - `(seconds, generation + 1)` - and the renderer
        # decodes that one frame through the model path and publishes it without writing it to the export. A request
        # only ever goes out while something renders; with no process the block is left alone (nothing would answer).
        if bool(getattr(self, "isPreview", False)):
            return
        reader = getattr(self, "_display", None)
        if reader is None or not reader.attached:
            return
        process = getattr(self, "renderProcess", None)
        if process is None or process.poll() is not None:
            return
        clip = self._clip_duration()
        if clip <= 0.0:
            return                      # the timeline length is unknown, so a share cannot become seconds
        target = fraction * clip
        generation = reader.request_seek(target)
        if bool(getattr(self, "_preview_paused", True)):
            # A held pane has `wanted` 0, so the renderer would stage the requested frame and never publish it. Ask
            # for exactly that one frame: `_on_display_frame` paints it and clears this again.
            self._display_seek_pending = True
            reader.set_wanted(True)
        log(f"seek: requested {target:.3f}s (generation {generation}) from the running render")

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
        # The pane's own `originalRequested` (a compare mode's playhead moved) and the 100 ms tick both arrive here;
        # recording what was asked keeps one moved playhead to one source decode.
        self._second_surface_pts = float(seconds)
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
        """One decoded frame from a preview's live tail (or a finished file): RGB24 bytes -> QImage -> the pane.

        An EXPORT's pane has two possible painters if the output is followable: the display feed and the tail. One
        picture, one painter - the display feed wins and the tail's frame is dropped here. That is the rule the deleted
        640x360 channel broke: two writers raced for the same surface, so the pane flickered between a thumbnail and a
        decoded frame and neither could be trusted. A preview keeps its tail as the only painter (`wanted` stays 0).

        `frame_ready` is delivered queued, so a tail stopped at the job's end can still deliver a frame it emitted
        before the stop. Such a frame has no current owner (the tail is gone, and playback of a finished file installs
        its own decoder) and is dropped: a zombie painter is the same rule violation as an early one.
        """
        from PySide6.QtGui import QImage

        if self._display_owns_pane():
            return
        if (getattr(self, "_live_tail", None) is None
                and getattr(self, "_preview_decoder", None) is None):
            return
        image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB888).copy()
        pane = self.page_handles.proc.previewLabel
        if hasattr(self.page_handles.proc.previewLabel, "set_frame"):
            pane.set_frame(image)
            self._preview_playhead = float(getattr(frame, "pts", 0.0) or 0.0)
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

    def updateProcessTab(self):
        """The panel's 100 ms tick: progress, log, FPS/ETA/status - and the pane's owners.

        It used to be called with a 640x360 thumbnail from shared memory. The picture now has two routes, both starting
        here: a preview's live tail decodes the file it follows (`_ensure_live_tail`), an export's display channel reads
        one new `seq` from the block the GUI owns (`_tick_display`). Exactly one of them paints a given pane.
        """

        # The pane's 100 ms clock is also the live tail's start point: it is the only GUI-thread tick during a
        # render, and the tail must be built and wired there (see `_start_live_tail`).
        self._ensure_live_tail()
        # Keep the transport honest for whichever painter owns the pane this tick (the tail or the file decoder): an
        # enabled button that holds nothing, or a disabled one that holds the picture, is worse than no button.
        self._sync_preview_transport()
        self._refresh_held_progress()
        # The export pane's feed and the standing original both run on this same clock: the reader takes at most one
        # new `seq` per tick, and the compare's source decode is coalesced to the playhead the result just reached.
        self._tick_display()
        self._tick_second_surface()


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
            paths=self._render_paths(),
        )

    def _render_paths(self):
        """The install's paths plus THIS pane's display block, which no constant can know.

        `render_command.app_paths()` stays free of the pane (it is importable without a window); the block's name comes
        from the pid like the paused flag's, and its box from the block actually allocated, so the command's
        `--display_size` and the segment can never disagree.
        """
        from dataclasses import replace

        from .render_command import app_paths

        box = getattr(self, "_display_box", None) or DEFAULT_DISPLAY_BOX
        width, height = int(box[0]), int(box[1])
        return replace(
            app_paths(),
            display_state_id=DISPLAY_STATE_SHARED_MEMORY_ID,
            display_size=f"{width}x{height}",
            seek_state_id=SEEK_STATE_SHARED_MEMORY_ID,
        )
