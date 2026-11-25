import subprocess
import os
import time
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
from .video import telemetry
from .video.press import PressTicket
from . import shared_flag
from . import i18n

from .AnimationHandler import AnimationHandler
from .QTcustom import (
    RegularQTPopup,
    UpdateGUIThread,
)
from .video.channel import (
    DEFAULT_DISPLAY_BOX,
    DISPLAY_BOX_SHARED_MEMORY_ID,
    DISPLAY_STATE_SHARED_MEMORY_ID,
    PHASE_BUILDING_ENGINE,
    PHASE_FAILED,
    PHASE_FINALISING,
    PHASE_NAMES,
    PHASE_PAUSED,
    PHASE_PREPARING,
    PHASE_RENDERING,
    SEEK_STATE_SHARED_MEMORY_ID,
    STATUS_STATE_SHARED_MEMORY_ID,
    DisplayReader,
    format_eta,
)
from .video.feed import PreviewFeed
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
from .dialogs.queue_properties import QueuePropertiesDialog


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


def queue_menu_entries(queue):
    """The rows of the queue's right-click menu, built from the current queue state.

    Module level (not a closure inside `installQueueMenu`) so the menu the app shows is the menu the harness pins:
    the previous inline version could only be reached by building a whole page. The entries are rebuilt on every
    request because their enabled state depends on the selection and on the render lock.

    **Properties** is read-only and therefore does NOT follow the render lock: a locked queue is being consumed by the
    writer thread, but what a queued file IS does not change, and looking it up while the render runs is exactly when
    a user asks. The mutating rows keep following it.
    """
    from sli_ui_toolkit.widgets import ContextMenuAction, ContextMenuSeparator

    locked = queue.is_locked()
    has_selection = bool(queue.selectedRows())
    # A locked queue is being consumed by the running render: the actions are shown DISABLED rather than the
    # whole list being disabled, so the user can still scroll it while the render runs.
    return (
        ContextMenuAction("queue.properties", "Properties…", enabled=has_selection),
        ContextMenuSeparator(),
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


def queue_properties_target(queue):
    """``(job, row, row_count, selected_count)`` for the row Properties applies to, or ``None``.

    Properties describes ONE file, so it applies to the FIRST selected row - the lowest index, i.e. the topmost of a
    rubber-band selection - and the window says which row that is. ``selected_count`` lets the window add "3 rows
    selected — showing row 1" instead of silently dropping the other two.
    """
    rows = sorted({int(row) for row in queue.selectedRows() if int(row) >= 0})
    items = queue.getQueue()
    valid = [row for row in rows if row < len(items)]
    if not valid:
        return None
    row = valid[0]
    return items[row], row, len(items), len(rows)


class ProcessTab:
    def __init__(self, parent, settings: Settings):
        self.parent = parent
        #: The typed page surface (`window.page_handles`): every control this glue touches is a required field on
        #: one of its group dataclasses, so a page that stops building one fails at build time, not at first click.
        self.page_handles: WindowHandles = parent.page_handles
        #: The page layout mode last logged, so a resize that keeps the mode does not write a line per event.
        self._page_layout_mode: str | None = None
        self._install_page_layout_glue()
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
        #: The queue pane's owner (owner, 2026-11-24): the selected queue file's path, the playhead remembered per path,
        #: and the one decoder that serves the queue tab. The pane has exactly one owner at a time - `render` (the
        #: live feed/tail/finished playback) or `queue-item` - and every switch between inner tabs hands it over.
        self._queue_pane_path: str | None = None
        self._queue_pane_positions: dict[str, float] = {}
        self._queue_pane_position = 0.0
        self._queue_pane_duration = 0.0
        self._queue_pane_decoder = None
        self._queue_pane_decoder_path: str | None = None
        self._queue_pane_owns = False
        #: The index of the Render Queue page in `processSettingsContainer`, resolved by identity in `QConnect`.
        self._queue_tab_index: int | None = None
        #: The export pane's display channel (`video.channel.DisplayReader`): the block the GUI allocates before the backend
        #: starts and the renderer publishes into while `wanted` is 1. `None` when no render has run. The attachment
        #: itself stays the reader's fact: `_display_can_seek_any_frame`/`_display_owns_pane`/`_display_paints_pane`
        #: ask it about the block, and the seek/wanted writes reach the block through it.
        self._display = None
        #: The chain over that block (`video.feed.PreviewFeed`): the newest picture, its identity, its drop count and
        #: the seek request. Built with the reader in `_prepare_display_state`, so the two describe one block; `None`
        #: whenever `_display` is (see `_display_chain`).
        self._feed = None
        #: The box the block was allocated at (the pane's device pixels, or `DEFAULT_DISPLAY_BOX`); the command's
        #: `--display_size` has to be the same box, so it is kept for the whole render pass.
        self._display_box = DEFAULT_DISPLAY_BOX
        #: The last render-status generation applied to the readouts: the record is rewritten every frame, so
        #: an unchanged one must not repaint the pane - and must not overwrite a status the pane itself set
        #: after it (Paused, Failed) with a stale "Rendering".
        self._status_generation = None
        #: The playhead the standing compare (`sbs`/`split`) was last asked for: the 100 ms clock must not re-seek the
        #: source decoder for a picture that is already on screen.
        self._second_surface_pts = None
        #: A paused pane may still be shown ONE frame - the one a strip press asked for. While this is set the request
        #: byte stays 1 and the next published frame paints even though the transport is holding.
        self._display_seek_pending = False
        #: The seek generation the pane is waiting for (`None` before the first press): a picture closes the seek only
        #: when the writer says it answers THIS one - "a named picture" is not the same thing, because the streaming
        #: path names every frame it draws.
        self._seek_generation = None
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
        # The glue just hid the pause/kill pair and populated the model combos; the row's hint and the General tab's
        # content are final now, so the page takes one more policy pass (the footer height is read from that row).
        self.page_handles.proc.layoutState.reapplyRequested.emit()

    def _install_page_layout_glue(self) -> None:
        """The typed edge into the page's layout policy (owner, 2026-11-24).

        The page owns the geometry (mode, column, band height, sticky footer); this glue does the two things the page
        cannot see: it logs the mode when it changes (one line per change, not per resize) and asks for a final pass
        after the glue's own construction, because hiding the pause/kill pair is what settles the render row's hint.
        """
        proc = self.page_handles.proc
        proc.layoutState.changed.connect(self._on_page_layout_changed)

    def _on_page_layout_changed(self, mode: str, column: int, collapsed: bool) -> None:
        """One log line per mode change: the layout a bug report has to name, without a line per resize."""
        if mode == self._page_layout_mode:
            return
        self._page_layout_mode = mode
        band = "collapsed" if collapsed else "expanded"
        log(f"layout: process page {mode} at {column} px (preview band {band})")

    def setStatus(self, text: str) -> None:
        """The one place that decides what `STATUS:` says - never leave a stale phase on screen.

        The prefix is a catalogue key (`i18n.tr`, keyed by the English source string), not a literal: this label is
        written outside the 100 ms tick as well, so a switch while idle would otherwise wait for the next job to read
        Russian. The phase itself (`Idle`, `Rendering`, …) is state, not chrome - it is not translated.
        """
        self.status = str(text)
        try:
            self.page_handles.proc.STATUS.setText(i18n.tr("Status: ") + self.status)
        except Exception:
            pass

    def setProgressDetail(self, text: str) -> None:
        try:
            self.page_handles.proc.progressDetail.setText(str(text))
        except Exception:
            pass

    def progressDetailText(self) -> str:
        """`42% · 12m3s elapsed · 16m40s left · frame 1234/5678` - empty while idle.

        The three words are catalogue keys (`i18n.tr`, keyed by the English source string) because the tick rebuilds
        this line every 100 ms: a literal here put `elapsed`/`left`/`frame` back on screen one tick after a switch to
        `ru` (measured 2026-11-24). The fragments keep their own spaces exactly as the literals did (`" elapsed"`,
        `" left"`, `"frame "` - the space is part of the key), so the English line and every number in it stay byte for
        byte what 1.0 wrote.
        """
        if self._render_started_mono is None and not self.max_value:
            return ""
        total = max(1, int(self.max_value or 0))
        done = max(0, min(total, int(self.currentFrame or 0)))
        parts = [f"{int(round(100.0 * done / total))}%"]
        if self._render_started_mono is not None:
            from time import monotonic

            elapsed = max(0.0, monotonic() - self._render_started_mono)
            parts.append(short_duration(elapsed) + i18n.tr(" elapsed"))
        if self.eta:
            parts.append(str(self.eta) + i18n.tr(" left"))
        parts.append(i18n.tr("frame ") + f"{done}/{total}")
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
        the whole selection: remove the picked items, or slide the block up/down; **Properties…** describes the first
        selected row. The entries themselves live in `queue_menu_entries` (module level) because their enabled state
        depends on the selection and on the render lock, and the harness pins exactly the rows the app shows.
        """
        from sli_ui_toolkit.widgets import ContextMenu

        widget = self.page_handles.queue.renderQueueListWidget

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
            elif action_id == "queue.properties":
                self.openQueueProperties()

        def on_context_menu(point) -> None:
            # A FRESH menu per request: the toolkit's popup surface destroys itself when it hides, so reusing one
            # instance raised "Internal C++ object (ContextMenu) already deleted" on the second right-click (measured).
            # NO `surface="popup"`: this was the app's only Qt.Popup, and the reference host's investigation of exactly
            # this symptom ends with one line - "a Qt.Popup whose QWidget parent is the top-level window breaks the
            # window's compositing on this Wayland/XWayland+NVIDIA stack: in-window translucent surfaces (toasts) render
            # opaque black, existing QRhiWidgets slow down" (blinktable docs/dev/investigations/
            # inv-2026-10-06-ql-rhi-menu-first.md, Root cause RESOLVED). The toolkit's default surface is the in-window
            # one, which is what every other menu in that app uses; the popup mode is opt-in and we opted in by accident.
            menu = ContextMenu(widget, entries=queue_menu_entries(self.parent.renderQueue),
                               on_triggered=on_action)
            # `on_triggered` is constructor state (the toolkit keeps it as `_on_triggered`), so it is passed here and
            # not assigned afterwards - an attribute set after construction would never fire.
            self._queue_menu = menu
            menu.popup_at(widget.mapToGlobal(point))

        widget.customContextMenuRequested.connect(on_context_menu)

    def openQueueProperties(self) -> None:
        """Open the Properties window for the queue's first selected row (nothing selected -> nothing to show).

        The probe runs at open time, on this thread, reading the file (see `queue_properties`): it is two short
        subprocesses with hard timeouts, not a render, so the window can wait for it. A locked queue is fine - this
        path mutates nothing.
        """
        target = queue_properties_target(self.parent.renderQueue)
        if target is None:
            return
        job, row, row_count, selected_count = target
        dialog = QueuePropertiesDialog(
            job,
            row=row,
            row_count=row_count,
            selected_count=selected_count,
            parent=self.parent,
            blocking=False,
        )
        self._queue_properties_dialog = dialog
        dialog.exec()

    def QConnect(self):
        # connect file select buttons
        self.page_handles.general.addToRenderQueueButton.clicked.connect(self.parent.addToRenderQueue)
        self.installQueueMenu()
        self.installQueueRows()
        # The pane follows the INNER tab (owner, 2026-11-24): every switch holds the picture, entering the queue tab
        # shows the remembered queue item (or nothing), leaving it hands the pane back to the render. The queue tab's
        # index is resolved by identity, so inserting a tab cannot make this listen to the wrong page.
        host = self.page_handles.proc.processSettingsContainer
        self._queue_tab_index = host.indexOf(self.page_handles.proc.renderQueueTab)
        host.currentChanged.connect(self._on_inner_tab_changed)
        # A click on a queue row selects it (`TextListWidget`'s own single-click behaviour) and the pane follows.
        self.page_handles.queue.renderQueueListWidget.currentRowChanged.connect(self._on_queue_row_changed)
        self.page_handles.general.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.page_handles.general.batchSelectButton.clicked.connect(self.parent.openBatchFiles)
        self.page_handles.general.inputFileText.textChanged.connect(self.parent.loadVideo)
        self.page_handles.general.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        self.page_handles.general.openOutputFolderButton.clicked.connect(self.openOutputFolderInExplorer)
        # One press handler for the transport (play / pause / resume): it dispatches on `_render_state`, so the button
        # never has to be re-connected when the job changes state (owner, 2026-11-24: one button, the icon follows).
        self.page_handles.proc.startRenderButton.clicked.connect(self._on_transport_pressed)
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
        self._set_render_controls("idle")
        self.page_handles.proc.killRenderButton.clicked.connect(self.killRenderProcess)
        # The preview's own transport: it stands PAUSED until this is pressed (owner, 2026-11-23), and the toggle's
        # checked state is the truth for both owners of the pane (the live tail and the finished-file decoder).
        # `toggled(bool)` rather than `clicked()`: the handler then receives the state the user asked for instead of
        # reading it back out of the widget - a read-back depends on when the toolkit lands its own toggle, and one
        # early run showed a click that left the button checked with nothing moved (2026-11-23, not reproduced after
        # the switch, and the state can no longer disagree with the signal that carries it).
        self.page_handles.proc.previewPauseButton.toggled.connect(self._on_preview_pause_toggled)
        self._sync_preview_transport()

    #: Which render button each state leaves USABLE. Visibility never changes (owner, 2026-11-24: "show the cancel
    #: button always, just inactive while nothing runs") - hiding a button made the row either re-flow at the press or
    #: reserve its box as an empty gap, and a disabled button avoids both while still reading as unavailable.
    RENDER_CONTROLS = {
        "idle": {"transport": True, "kill": False},
        "running": {"transport": True, "kill": True},
        "paused": {"transport": True, "kill": True},
    }

    def _set_render_controls(self, state: str) -> None:
        """Apply `state`: the transport stays usable and its glyph follows, the cancel follows the job.

        One button carries play, pause and resume (owner, 2026-11-24: "join the pause and play buttons - just change the
        icon"). Checked means "a job is running", i.e. the pause glyph; the press is dispatched by `_render_state`, not
        by the toggle value, because a toggle flips before the handler runs.
        """
        proc = self.page_handles.proc
        wanted = self.RENDER_CONTROLS[state]
        proc.onRenderButtonsContiainer.setEnabled(True)
        for name, widget in (("transport", proc.startRenderButton), ("kill", proc.killRenderButton)):
            widget.setVisible(True)
            widget.setEnabled(bool(wanted[name]))
        self._render_state = state
        # `setChecked` emits `toggled`; the guard keeps our own state write out of the dispatcher.
        self._applying_controls = True
        try:
            proc.startRenderButton.setChecked(state == "running")
        finally:
            self._applying_controls = False

    def _on_transport_pressed(self) -> None:
        """The one press handler: dispatch on the state the glue is in, never on the glyph.

        The button is a toggle, so the press flips it BEFORE this runs. A start that does not happen (no file picked,
        the overwrite dialog cancelled, a backend missing) would then leave the glyph on "pause" with nothing running -
        measured live 2026-11-24 on a fresh instance. The state is re-asserted in `finally`, which is the same single
        writer every other transition uses.
        """
        if getattr(self, "_applying_controls", False):
            return
        state = getattr(self, "_render_state", "idle")
        try:
            if state == "running":
                self.pauseRender()
            elif state == "paused":
                self.resumeRender()
            else:
                self.parent.startRender()
        finally:
            self._set_render_controls(getattr(self, "_render_state", "idle"))

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.userKilled = True
            self.renderProcess.terminate()
        except AttributeError:
            log("No render process!")

    def pauseRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 1  # 1 = True
        self._set_render_controls("paused")
        self.setStatus("Paused")

    def resumeRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 0  # 0 = False
        from time import monotonic

        self._render_started_mono = monotonic()
        self.setStatus("Preparing")
        self._set_render_controls("running")

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
        self._set_render_controls("running")

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

    def _display_allocation_box(self) -> tuple[int, int]:
        """The box the block is ALLOCATED for: a maximum the pane may grow into without a reallocation.

        The pane announces its real size separately (`DisplayBoxWriter`); this is only how much memory is reserved. The
        screen's device pixels are the honest ceiling - the window can be maximised - capped so a preview cannot reserve
        a 4K pair of slots for a 400 px pane, and floored at the pane's own current box.
        """
        pane_box = self._pane_display_box()
        width, height = pane_box
        try:
            from PySide6.QtGui import QGuiApplication

            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                geometry = screen.availableGeometry()
                ratio = float(screen.devicePixelRatio() or 1.0)
                width = max(width, int(geometry.width() * ratio))
                height = max(height, int(geometry.height() * ratio))
        except Exception:
            pass
        width = min(width, 2560)
        height = min(height, 1440)
        return max(2, width) & ~1, max(2, height) & ~1

    def _prepare_display_state(self) -> None:
        """Allocate the pane's frame block (and the seek block) before the backend starts - GUI thread only.

        The GUI is the owner: the renderer attaches to this name and never creates it, so a render started without a
        window (or after it closed) pays nothing. One block per render pass; `_close_display_state` releases it.
        """
        self._close_display_state()
        # The block is allocated for a MAXIMUM and the pane announces its real box through the record; a resize inside
        # the maximum then needs no reallocation (see `_display_allocation_box`).
        box = self._display_allocation_box()
        self._display_box = box
        # A new pass starts at generation 0 of a fresh record, whatever the last render left in the readouts.
        self._status_generation = None
        try:
            reader = DisplayReader(
                DISPLAY_STATE_SHARED_MEMORY_ID,
                box=box,
                seek_name=SEEK_STATE_SHARED_MEMORY_ID,
            )
        except Exception as exc:  # a pane that cannot allocate must not take the export down
            log(f"display: cannot allocate the pane's block ({exc}); the export runs without it")
            self._display = None
            self._feed = None
            return
        # The pane's own box, announced through its own record: the writer fits into it and answers with what the
        # allocated block really holds. Created with the state so the FIRST frame is already the right size.
        try:
            from .video.channel import DisplayBoxWriter

            self._display_box_writer = DisplayBoxWriter(DISPLAY_BOX_SHARED_MEMORY_ID)
            if self._display_box_writer.attached:
                pane_box = self._pane_display_box()
                self._display_box_writer.announce(*pane_box)
                self._announced_box = pane_box
            else:
                self._announced_box = None
        except Exception as exc:
            log(f"display: the announced-box record is not usable ({exc}); the block keeps its allocation size")
            self._display_box_writer = None
            self._announced_box = None
        if not reader.attached:
            log("display: the pane's block is not usable; the export runs without it")
            reader.release()
            self._display = None
            self._feed = None
            return
        # The chain over that block: it decides what the newest picture is, reads its identity off the reader at the
        # moment it is accepted, counts the pictures it replaces, and writes the seek request. The pane paints from
        # `picture_changed`, so nothing has to re-derive "which frame is this" mid-paint any more.
        feed = PreviewFeed(
            DISPLAY_STATE_SHARED_MEMORY_ID,
            box=box,
            seek_name=SEEK_STATE_SHARED_MEMORY_ID,
            reader=reader,
        )
        if not feed.attach():  # the block is usable (checked above), so this only fails on a broken reader
            log("display: the pane's feed cannot take the block; the export runs without it")
            feed.detach()
            self._display = None
            self._feed = None
            return
        feed.picture_changed.connect(self._on_display_frame)
        self._display = reader
        self._feed = feed
        # The strip and the compare need their signals for an export too: without this only a preview ever wire them,
        # and a strip press during an export reached nothing (the seek then had two routes, neither connected).
        self._wire_preview_strip()
        # An export's feed renders any frame on demand, so the strip may seek the whole timeline instead of only the
        # produced band (measured live, 2026-11-24: a 30 % drag during a 7.5 %-produced export did nothing). This also
        # clears the ceiling when the allocation failed or the job is a preview (whose tail owns the edge).
        self._apply_seek_ceiling()
        log(f"display: block {DISPLAY_STATE_SHARED_MEMORY_ID} at {box[0]}x{box[1]}, "
            f"seek {SEEK_STATE_SHARED_MEMORY_ID}, status {STATUS_STATE_SHARED_MEMORY_ID}")

    def _close_display_state(self) -> None:
        """Release the frame, seek and announced-box blocks; safe twice, and safe when nothing was prepared."""
        announcer = getattr(self, "_display_box_writer", None)
        self._display_box_writer = None
        self._announced_box = None
        if announcer is not None:
            try:
                announcer.release()
            except Exception:
                pass
        feed = getattr(self, "_feed", None)
        reader = getattr(self, "_display", None)
        self._feed = None
        self._display = None
        if feed is not None:
            try:
                feed.detach()           # releases the reader's two blocks
            except Exception:
                pass
        elif reader is not None:
            # A tab a test wired by hand (or one whose feed never attached): the reader is still the block's owner.
            try:
                reader.release()
            except Exception:
                pass
        # Whoever owns the pane next (a tail, a finished file) decides its own reach: the feed's ceiling must not
        # outlive the feed, or a live tail would accept seeks into frames nobody has written.
        self._apply_seek_ceiling()

    def _apply_seek_ceiling(self) -> None:
        """Tell the pane how far its strip may seek, from what can actually serve a frame there.

        A usable block renders ANY timeline frame on demand (`RenderVideo._seek_preview`): an export's feed paints
        continuously from it, and a PREVIEW's same block answers a press past the produced band with that one frame
        (the tail only holds what the renderer has written). So the ceiling is 1.0 whenever the block is usable;
        without one the pane keeps its produced-edge rule, which is the honest limit for a tail.
        """
        handles = getattr(self, "page_handles", None)
        pane = getattr(getattr(handles, "proc", None), "previewLabel", None)
        setter = getattr(pane, "set_seek_ceiling", None)
        if setter is None:
            return
        if self._queue_owns_pane():
            # The queue item is a finished file: every frame of it can be decoded, so the strip may address all of it.
            reach = 1.0 if float(getattr(self, "_queue_pane_duration", 0.0) or 0.0) > 0.0 else None
            try:
                setter(reach)
            except Exception:
                pass
            return
        try:
            setter(1.0 if self._display_can_seek_any_frame() else None)
        except Exception:
            pass

    def _display_can_seek_any_frame(self) -> bool:
        """Whether the pane's block is attached, i.e. a press anywhere on the strip can be answered with one frame.

        The block is what makes a seek exact: the renderer decodes the requested timestamp through the model path and
        publishes it without writing it to the export. A preview's tail cannot do that - it only has what the file
        already holds - so this is the strip's REACH, not the pane's painter (see `_display_paints_pane`).
        """
        reader = getattr(self, "_display", None)
        return reader is not None and bool(getattr(reader, "attached", False))

    def _display_owns_pane(self) -> bool:
        """Whether the display feed is the pane's standing painter (never during a preview: its tail owns it).

        Use `_display_paints_pane` for "who paints the next frame": a preview lets the feed paint exactly the one
        frame a strip press asked for, which this predicate deliberately does not include.
        """
        reader = getattr(self, "_display", None)
        return reader is not None and reader.attached and not bool(getattr(self, "isPreview", False))

    def _display_paints_pane(self) -> bool:
        """Which painter owns the pane for the NEXT frame (one picture, one painter).

        An export's feed owns it whenever the block is usable. A preview's tail owns it - except while the pane waits
        for the one frame a strip press asked for (`_display_seek_pending`): the feed owns it for exactly that frame,
        and `_on_display_frame` hands it back. That is the same one-frame loan an export's held pane uses; there is no
        second mechanism.
        """
        reader = getattr(self, "_display", None)
        if reader is None or not bool(getattr(reader, "attached", False)):
            return False
        if self._queue_owns_pane():
            # The queue tab's item is the pane's one owner there: the feed stages nothing and its frames are dropped.
            return False
        if not bool(getattr(self, "isPreview", False)):
            # An export's feed owns the pane UNLESS a tail is painting it: during playback the tail is the timeline
            # (position, rate, seek), and the feed publishing at the same time would be the two-painter race the tail
            # rule was written to stop. A held export has no tail, so the feed's one-frame loan is untouched.
            if getattr(self, "_live_tail", None) is not None and not bool(getattr(self, "_preview_paused", True)):
                return False
            return True
        return bool(getattr(self, "_display_seek_pending", False))

    def _display_should_want(self) -> bool:
        """`wanted` = the pane is visible AND the transport is playing AND a render is publishing into the block.

        A paused picture must cost the renderer nothing: that is the one byte this whole channel exists to save. The
        exception is the one frame a strip press asked for - for an export's held pane and for a preview alike, the
        feed paints that frame and only that frame (`_display_paints_pane`).
        """
        if not self._display_paints_pane():
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

    def _display_chain(self):
        """The object that carries the picture and the seek request: the feed when there is one, else the reader.

        Both answer the same calls (`attached`, `set_wanted`, `poll`, `status`), which is why one expression is enough:
        the feed composes the reader, and a tab wired by hand (a test stub) has the reader alone.
        """
        return getattr(self, "_feed", None) or getattr(self, "_display", None)

    def _update_display_wanted(self) -> None:
        """Re-assert the request byte after anything that changes the answer (pause, job start, page hide)."""
        display = self._display_chain()
        if display is None or not display.attached:
            return
        display.set_wanted(self._display_should_want())

    #: The phases whose word the pane takes from the render-status block. Idle/Done are the pane's own
    #: transitions (the job's completion path sets them); taking them from the slot would announce "Idle"
    #: while the process is still flushing.
    _SLOT_STATUS_PHASES = (PHASE_PREPARING, PHASE_BUILDING_ENGINE, PHASE_RENDERING,
                           PHASE_FINALISING, PHASE_PAUSED, PHASE_FAILED)

    def _sync_render_status(self) -> None:
        """Take FPS/ETA/frame count/status from the render-status slot, not from the backend's prose.

        The block is written once per rendered frame and on every phase change, so it is fresher than the
        twice-a-second log line it replaces. An unchanged `generation` is a no-op - that is what keeps the
        pane's own Paused/Failed/Idle from being overwritten by a record that has not moved since. A torn
        record answers `None` and the previous readouts stay on screen.
        """
        display = self._display_chain()
        if display is None:
            return
        getter = getattr(display, "status", None)
        if not callable(getter):
            return
        try:
            status = getter()
        except Exception:
            return
        if not status:
            return
        generation = status.get("generation")
        if generation is None or generation == getattr(self, "_status_generation", None):
            return
        self._status_generation = generation
        frames = status.get("frames_rendered")
        if isinstance(frames, int) and frames >= 0:
            self.currentFrame = frames
        fps = status.get("fps")
        try:
            fps = float(fps or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        if fps > 0:
            self.fps = int(round(fps))
        eta = status.get("eta_seconds")
        if eta is not None and float(eta) >= 0:
            self.eta = format_eta(eta)
        phase = status.get("phase")
        if phase in self._SLOT_STATUS_PHASES:
            name = PHASE_NAMES.get(phase)
            if name is not None:
                self.setStatus(name)

    def _announce_pane_box(self) -> None:
        """Publish a new pane box when it changed; one call per tick, no signal plumbing.

        The pane is a widget and knows nothing about shared memory; the tab reads its box on the clock it already runs.
        A miss of one tick is 100 ms of the old size, which is the same latency the picture itself has.
        """
        writer = getattr(self, "_display_box_writer", None)
        if writer is None or not getattr(writer, "attached", False):
            return
        box = self._pane_display_box()
        if box == getattr(self, "_announced_box", None):
            return
        generation = writer.announce(*box)
        self._announced_box = box
        log(f"display: pane box {box[0]}x{box[1]} announced (generation {generation})")

    def _tick_display(self) -> None:
        """One 100 ms tick of the export feed: ask, read at most one new `seq`, paint (GUI thread)."""
        self._announce_pane_box()
        display = self._display_chain()
        if display is None or not display.attached:
            return
        wanted = self._display_should_want()
        display.set_wanted(wanted)
        if not wanted:
            return
        display.poll()          # the feed emits `picture_changed` only when the sequence moved

    def _on_display_frame(self, image=None) -> None:
        """The renderer's composited frame for the pane (one picture, one painter).

        An export's feed paints it whenever the block is usable. A preview's tail is that pane's painter, so a display
        frame is dropped there - except the ONE frame a strip press asked for, which `_display_paints_pane` admits
        while `_display_seek_pending` is set. While the transport holds a frame, display frames are dropped too and the
        picture stays. The progress is republished with every fresh frame because the strip's produced band is the
        render's own counters, not the picture's timestamp.

        `image` is passed only by a caller that already holds the picture (a test's stub reader emitting its own
        signal); the feed's `picture_changed` carries nothing, so the feed is where the picture and its identity are
        read from.
        """
        feed = getattr(self, "_feed", None)
        if image is None and feed is not None:
            image = feed.image()
        if image is None:
            return
        if not self._display_paints_pane():
            return
        holding = bool(getattr(self, "_preview_paused", True))
        pending = bool(getattr(self, "_display_seek_pending", False))
        if holding and not pending:
            return
        if pending and not self._picture_answers_seek():
            # A seek is pending and this is not its answer: paint NOTHING. A held pane waits for one picture, and
            # showing the in-flight streaming frame first is the flash this gate exists to stop - measured live
            # (2026-11-24): a press at 1.888 s painted the streaming frame at 4.046 s, then the answer at 1.888 s.
            # The request byte stays 1, so the answer still arrives - and the log names the picture it refused, once
            # per press, because "the pane painted something else than my press" has to be readable afterwards.
            ticket = getattr(self, "_press", None)
            if ticket is not None and ticket.open:
                feed_named = getattr(self, "_feed", None)
                if feed_named is not None and callable(getattr(feed_named, "shown", None)):
                    named = feed_named.shown()
                    ticket.note_stream(seq=named[1] if named else None,
                                       media_time=feed_named.shown_time(),
                                       landed_gen=self._picture_seek_answer())
                else:
                    reader_named = getattr(self, "_display", None)
                    ticket.note_stream(seq=getattr(reader_named, "seq", None),
                                       media_time=getattr(reader_named, "media_time", None),
                                       landed_gen=self._picture_seek_answer())
            return
        pane = self.page_handles.proc.previewLabel
        if hasattr(pane, "set_frame"):
            # Only the seek's clock is closed here: the FRAME itself is counted by the pane (`set_frame`), because this
            # function and the panel would otherwise count the same picture twice (measured 2026-11-24: `frames=2`
            # for one frame on screen).
            pressed_at = getattr(self, "_seek_pressed_at", None)
            # The display channel's own sequence is the identity of a composited picture: the renderer's frame counter
            # is a different fact (it is what the log used to print, and it named frames the pane was not showing).
            # The feed read that identity when it accepted this picture, so a repaint cannot pair a picture with the
            # counter of the one that replaced it.
            if feed is not None and callable(getattr(feed, "shown", None)):
                named = feed.shown()
                seq = named[1] if named else None
                media_time = feed.shown_time()
            else:
                # A tab wired by hand (a test's stub reader): the reader still carries the two attributes.
                reader = getattr(self, "_display", None)
                seq = getattr(reader, "seq", None)
                media_time = getattr(reader, "media_time", None)
            answer_of = getattr(self, "_picture_seek_answer", None)
            answer = answer_of() if callable(answer_of) else None
            # A seek answer is named by the reader's FITTED grid (a few milliseconds off the container); a streaming
            # picture is named by the frame's own PTS. The compare needs to know which, because a fitted name needs the
            # half-frame slack and an exact one must have none.
            self._shown_name_is_fitted = answer is not None
            pane.set_frame(image, seq=seq, media_time=media_time, generation=answer)
            if pressed_at is not None and pending and media_time is not None:
                # Close the seek here, and only here: the gate above already refused every picture that does not
                # answer the request, so this publication IS the answer. The log names both generations as well as the
                # two times, because a press -> answer turnaround is only a cost when the picture is the right one.
                shown = getattr(pane, "shown", None)
                shown_time = getattr(pane, "shown_time", None)
                fields = dict(target=getattr(self, "_seek_target_frame", None),
                              landed=shown() if callable(shown) else None,
                              target_s=getattr(self, "_seek_target_s", None),
                              landed_time=shown_time() if callable(shown_time) else None,
                              target_gen=getattr(self, "_seek_generation", None),
                              landed_gen=self._picture_seek_answer())
                turnaround_ms = (time.perf_counter() - pressed_at) * 1000.0
                ticket = getattr(self, "_press", None)
                if ticket is not None and ticket.open:
                    # The press record settles: the log line is a projection of it, not a sentence written here.
                    ticket.settled(landed=fields["landed"], landed_time=fields["landed_time"],
                                   turnaround_ms=turnaround_ms, landed_gen=fields["landed_gen"])
                else:
                    telemetry.note_seek(turnaround_ms, **fields)
                self._seek_pressed_at = None
                self._seek_generation = None
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

    def _picture_seek_answer(self) -> int | None:
        """The seek generation the pane's current picture answers, or None when it is a streaming frame."""
        feed = getattr(self, "_feed", None)
        getter = getattr(feed, "shown_seek", None)
        if callable(getter):
            return getter()
        return getattr(getattr(self, "_display", None), "seek_answer", None)

    def _picture_names_its_seek(self) -> bool:
        """Whether this producer can say WHICH request a picture answers at all.

        A stub reader and a backend older than the header field cannot, and there the media time is the only evidence
        there is. The real `DisplayReader` and `PreviewFeed` always can - and then `None` means "a streaming frame",
        which is the distinction that was missing.
        """
        feed = getattr(self, "_feed", None)
        if callable(getattr(feed, "shown_seek", None)):
            return True
        return hasattr(getattr(self, "_display", None), "seek_answer")

    def _picture_answers_seek(self) -> bool:
        """Whether the picture on hand is the answer to the seek the pane asked for.

        Identity, not proximity, and it fails CLOSED: a producer that names generations can only close a seek on the
        generation it asked for, so a named streaming frame is refused even though its media time is present. That is
        the defect this rule replaces - measured live (2026-11-24): a press at 1.888 s was closed by the streaming
        frame at 4.046 s (+2.158 s), because the streaming path names every frame it draws.
        """
        if not self._picture_names_its_seek():
            return True                     # an older producer: a named picture is all the evidence there is
        answer = self._picture_seek_answer()
        if answer is None:
            return False                    # a streaming frame answers no request, whatever it names
        wanted = getattr(self, "_seek_generation", None)
        if wanted is None:
            return True
        try:
            return int(answer) >= int(wanted)
        except (TypeError, ValueError):
            return True

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
        """Where the RESULT's playhead is: the tail's or file decoder's PTS, else the render's produced share.

        While a press has no answer yet the marker stands on the REQUESTED time. It used to be recomputed from
        `currentFrame / max_value` on every painted frame, so a press was followed by the marker jumping to the produced
        edge - "увела туда же головку произведения несмотря на паузу" (owner, 2026-11-24).
        """
        ticket = getattr(self, "_press", None)
        if ticket is not None and ticket.open and ticket.target_s is not None:
            return float(ticket.target_s)
        # The picture on screen names its own time: a display-channel frame carries the export's media time, so the
        # marker follows what is SHOWN instead of being recomputed from `currentFrame / max_value`. The old arithmetic
        # sent the marker to the produced edge right after a seek had landed (measured live 2026-11-24: press at
        # 12.370 s, marker at 7.88 s - the produced share at that instant).
        pane = getattr(getattr(self, "page_handles", None), "proc", None)
        shown_time = getattr(getattr(pane, "previewLabel", None), "shown_time", None)
        if callable(shown_time):
            named = shown_time()
            if named is not None:
                return float(named)
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
        if self._queue_owns_pane():
            # A queue item is the SOURCE, not a render: there is no second surface to compare it against.
            self._second_surface_pts = None
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
        # The tick asks every 100 ms while the playhead moves: its own requests are not presses and must not
        # fill log.txt, so the compare's one-line-per-press instrument is told which one this is.
        self._compare_from_tick = True
        try:
            self._on_original_requested(playhead)
        finally:
            self._compare_from_tick = False

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
            # The backend's raw stream, kept in its own file: the tab filter decides what a person reads, this keeps
            # what the backend actually printed (owner, 2026-11-25: "логично тогда иметь условно файл logs-terminal.txt").
            terminal_log = None
            try:
                from .terminal_log import TerminalLog

                terminal_log = TerminalLog()
            except Exception as exc:
                log(f"terminal log: not available ({exc}); the backend stream stays on its tab only")
            # Read to EOF, not "until poll() says the process is gone": the old loop checked poll()
            # BEFORE handling the line it had just read, so the last lines of a dying render - the
            # `RENDER FAILED: ...` summary included - were dropped on the floor and the job ended
            # with no reason anywhere (2026-11-22). `readline` returns "" at EOF in text mode, which
            # is the sentinel that actually terminates this iterator.
            for line in iter(self.renderProcess.stdout.readline, ""):
                try:
                    if terminal_log is not None:
                        terminal_log.write(line)
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
                    # FPS, ETA, the frame count and the status words are NOT parsed out of this stream any
                    # more: the backend writes them to its status block (`video/channel.py: STATUS`) and
                    # `_sync_render_status` reads them on the 100 ms tick. The render log below still gets
                    # every event, and the status lines are still printed for the log file a person reads.
                    #
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
            if terminal_log is not None:
                terminal_log.close()
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
        # The row goes back to its idle state: start usable, the cancel pair visible but disabled.
        self._set_render_controls("idle")
        self.page_handles.general.outputFileText.setEnabled(True)
        self.page_handles.general.inputFileText.setEnabled(True)
        # The job is over: no backend publishes any more, so the block and the seek request are released here (the
        # GUI thread) rather than left for a tick that may never come - and whoever followed the job's file stops
        # with it (see `_end_job_painters`). The pane's own clear below follows.
        self._second_surface_pts = None
        self._display_seek_pending = False
        self._end_job_painters()
        # The pane KEEPS the last picture at the end of a job. It used to be cleared here unconditionally, which left
        # an empty pane after every plain render - "the preview is cleared when processing ends, even when it is the
        # only video in the queue" (owner, 2026-11-24). A preview job is repainted a moment later by the finished-file
        # playback, and a plain render has no second painter, so the display feed's last frame IS the result the user
        # wants to look at. The frame is replaced by the next job's first frame; nothing stale is left behind.
        log("preview: keeping the last painted frame at the end of the job")
        self.parent.enableProcessPage()
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
        self.page_handles.proc.FPS.setText(i18n.tr("FPS: "))
        self.page_handles.proc.ETA.setText(i18n.tr("ETA: "))

        self._restoreQueueAfterPreview()
        if self.currentRenderOptions.isPreview:
            self._start_preview_playback(self.currentRenderOptions.outputPath)
        elif finished_ok:
            # The export is done, and its pane had NO source after this point: the display feed is released, the tail
            # was stopped, and `_request_display_seek` refuses a press because no process will answer it. That is why
            # "the last seeks do nothing on a fully rendered file" (owner, 2026-11-25). The finished file becomes the
            # pane's source, HELD at the position the render stood on: the strip scrubs it, Play runs it at the media
            # rate (the decoder paces by PTS), and the picture is the real output rather than the last live frame.
            output = str(getattr(self.currentRenderOptions, "outputPath", "") or "")
            try:
                import os as _os

                readable = bool(output) and _os.path.exists(output) and _os.path.getsize(output) > 0
            except Exception:
                readable = False
            if readable:
                self._preview_paused = True
                self._start_preview_playback(output)
                self._sync_preview_transport()
                log(f"preview: the finished file is the pane's source ({output}); the strip scrubs it")
            else:
                log(f"preview: no finished file to hand the pane ({output!r}); the last painted frame stays")

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
        if self._queue_owns_pane():
            # The queue item's progress is its own file's; the render's counters must not repaint over it.
            return
        if (getattr(self, "_live_tail", None) is None
                and not self._display_owns_pane()):
            return
        pane = self.page_handles.proc.previewLabel
        progress = getattr(pane, "_progress", None)
        if not progress or len(progress) < 4:
            return
        produced = (float(getattr(self, "currentFrame", 0) or 0), float(getattr(self, "max_value", 0) or 0))
        if (float(progress[0]), float(progress[1])) == produced:
            # Nothing has been produced since the last publish, and `set_progress` repaints unconditionally (it is the
            # strip's only notification). A held pane with nothing new therefore repainted ten times a second for
            # nothing - measured in the owner's own session (2026-11-24): `frames=0 paints=10 int_ms=100` while the
            # picture and both counters stood still.
            return
        playhead, duration = float(progress[2]), float(progress[3])
        pane.set_progress(rendered=produced[0], total=produced[1], playhead=playhead, duration=duration)

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
            box = self._pane_display_box()
            tail = LiveTailThread(path, paused=self._preview_paused, target_size=box)
            tail.frame_ready.connect(self._on_preview_frame)
            tail.error.connect(lambda message: log(f"preview tail: {message}"))
            # A press the file could not answer must be legible in log.txt: "it did nothing" and "it snapped to the
            # end of the upscale" are the same silence from the outside (owner, 2026-11-24).
            tail.seek_unavailable.connect(lambda message: log(f"preview tail: {message}"))
            # The strip's signals too: a press must seek the tail, and the LMB compare must decode the source at the
            # current playhead. Connected here (GUI thread) or the seek lands in a queue nobody drains.
            self._wire_preview_strip()
            self._live_tail = tail
            self._live_tail_wanted = False
            # Playback continues where the picture stands - but only where FRAMES EXIST. A seek past the produced edge is
            # answered on demand by the renderer's display channel (it decodes one frame), which does NOT write that
            # frame to the output; the file therefore holds nothing after the edge, and a tail asked to start there
            # cannot read it and would play from the file's beginning (measured live 2026-11-25: seek to 9.384 s on a
            # render that had written 4.9 s, playback started at 1.1 s). So the start is clamped to the produced edge and
            # the log says both numbers: an honest "from here, because that is all there is".
            start = float(getattr(self, "_preview_playhead", 0.0) or 0.0)
            clip = self._clip_duration()
            produced = self._live_produced_fraction()
            edge = produced * clip if clip > 0.0 and produced > 0.0 else 0.0
            if edge > 0.0 and start > edge:
                log(f"preview: playback starts at the render's edge {edge:.3f}s - {start:.3f}s is not written yet")
                start = edge
            if start > 0.0:
                try:
                    tail.seek(start, attempt_past_edge=True)
                except Exception:
                    pass
            tail.start()
            self._sync_preview_transport()
        except Exception as exc:  # a preview that cannot follow must not take the render down
            self._live_tail_wanted = False
            log(f"preview tail failed: {exc}")

    def _stop_live_tail(self) -> None:
        self._live_tail_wanted = False
        self._playback_tail = False
        self._tail_origin_logged = False
        tail = getattr(self, "_live_tail", None)
        if tail is not None:
            try:
                tail.stop_and_wait()
            except Exception:
                pass
            self._live_tail = None
        self._sync_preview_transport()

    def _resume_tail_at_the_pane(self, tail) -> None:
        """On play, continue from where the PANE stands - the seeker position - not from the renderer's head.

        Owner, 2026-11-25: "кнопка play начинает играть видео не с позиции сикера". The previous rule jumped to the
        produced head on purpose, to avoid replaying the backlog a held pane had accumulated; that cured a stall but
        made play ignore the seeker, which is the behaviour a viewer actually relies on. Both are satisfied here: the
        pane's position is the target, clamped into the produced band, so play continues where the picture stands and a
        backlog is still played at the media rate rather than skipped.

        The pane's clock is `_preview_playhead` (set by every tail frame and by a held seek). A pane that never painted
        one has no position and the head is the only honest answer.
        """
        clip = self._clip_duration()
        produced = self._live_produced_fraction()
        head = (produced * clip if clip > 0.0 and produced > 0.0
                else float(getattr(tail, "produced_seconds", 0.0) or 0.0))
        pane = float(getattr(self, "_preview_playhead", 0.0) or 0.0)
        if clip > 0.0:
            pane = min(max(0.0, pane), clip)
        target = pane if pane > 0.0 else head
        if target <= 0.0:
            return
        tail.seek(target, attempt_past_edge=pane > 0.0)
        log(f"preview transport: resuming at the pane's position {target:.3f}s "
            f"(pane {pane:.3f}s, render head {head:.3f}s = {produced:.1%} of {clip:.3f}s)")

    def _on_preview_pause_toggled(self, paused: bool) -> None:
        telemetry.event("transport", paused=int(bool(paused)))
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
            "queue" if getattr(self, "_queue_pane_decoder", None) is not None else (
                "file" if getattr(self, "_preview_decoder", None) is not None else "thumbnail"))
        log(f"preview transport: {'paused' if self._preview_paused else 'playing'} (owner={owner})")
        holding = bool(self._preview_paused)
        # ONE producer at a time: while the queue tab owns the pane, pressing play starts the QUEUE item, not the
        # render's tail/decoder behind it - a decoder playing under a picture nobody shows is pure cost.
        queue_owns = self._queue_owns_pane()
        if holding and getattr(self, "_playback_tail", False):
            # The tail this PLAY created exists only while the transport plays; pausing drops it and the feed's cheap
            # path (the renderer's own composited frame) takes the pane back. A PREVIEW's tail and a tail somebody else
            # owns (the queue tab's hand-back) are NOT ours to stop - they are the picture there.
            self._stop_live_tail()
        tail = getattr(self, "_live_tail", None)
        if not holding and tail is None and not queue_owns and not bool(getattr(self, "isPreview", False)):
            # PLAY, export, no tail: the pane was following the renderer's newest frames, which is not playback. The
            # output file is the timeline; start the tail here (the GUI tick builds it) at the pane's own position, and
            # the feed stands down while it paints. Pausing drops it again, so the feed's cheap path is what a held
            # pane uses (owner, 2026-11-25: "воспроизведение всё ещё просто переключает нас на слежение за последними
            # апкад кадрами").
            options = getattr(self, "currentRenderOptions", None)
            path = str(getattr(options, "outputPath", "") or "")
            if path:
                self._start_live_tail(path)
                self._ensure_live_tail()
                self._playback_tail = getattr(self, "_live_tail", None) is not None
                tail = getattr(self, "_live_tail", None)
        if tail is not None:
            if not holding and not queue_owns:
                # Its own try: a resume that cannot work out the head must still unpause the tail (the first version
                # put both in one block, and an AttributeError in the head arithmetic skipped `set_paused` entirely).
                try:
                    self._resume_tail_at_the_pane(tail)
                except Exception as exc:
                    log(f"preview transport: resume at the head failed ({exc})")
            try:
                tail.set_paused(holding or queue_owns)
            except Exception as exc:
                log(f"preview transport: {exc}")
        decoder = getattr(self, "_preview_decoder", None)
        if decoder is not None:
            try:
                decoder.pause() if (holding or queue_owns) else decoder.play()
            except Exception as exc:
                log(f"preview transport: {exc}")
        queue_decoder = getattr(self, "_queue_pane_decoder", None)
        if queue_decoder is not None:
            try:
                queue_decoder.pause() if (holding or not queue_owns) else queue_decoder.play()
            except Exception as exc:
                log(f"queue pane: {exc}")
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
                        or getattr(self, "_queue_pane_decoder", None) is not None
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
        if hasattr(pane, "seekBlocked"):
            # The pane refuses a press past the produced band before the host ever sees it. That is the right answer
            # while the render is still writing, but it is also the third way a press can "do nothing", so it says so.
            try:
                pane.seekBlocked.connect(
                    lambda fraction: log(
                        f"seek: {float(fraction):.1%} of the strip is past the produced band; nothing to show there"))
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
            # CONTENT time: this decoder reads the render's OWN output, whose container timeline starts at the
            # encoder's B-frame delay (0.083 s on the owner's clip) while its first frame is the source's first.
            # Without this the pane's times, its strip and the LMB compare are all two frames late.
            # Capped to the pane's box, like the queue decoder: the pane cannot show more than that, and a 2880x2160
            # frame costs 24.9 MB of pack + a 33 MB RGBA upload per picture - the cost that kept the paint at ~9 fps
            # while the decoder delivered 24 (measured 2026-11-25).
            # One frame per tick: this decoder runs on the GUI thread, so a burst of due frames lands as several
            # `set_frame` calls before Qt paints ONCE - the pane then shows only the last of the burst and skips the
            # rest (measured: 96 frames delivered, 66 painted, every second picture lost). One per tick leaves the
            # event loop a paint between frames.
            decoder = VideoDecoder(pixel_format="rgb24", content_time=True,
                                   max_size=self._pane_display_box(), max_frames_per_tick=1)
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

    # -- the queue pane: the Render Queue tab's own owner of the surface ---------------------------------

    def _queue_owns_pane(self) -> bool:
        """Whether the selected queue item, not the render, is the picture on the pane right now.

        The one predicate every painter asks. The pane has ONE owner: while its tab is current the queue item paints
        (the render's feed and tail drop their frames at the door), and after the hand-back the queue decoder's own
        queued frames are dropped the same way.
        """
        return bool(getattr(self, "_queue_pane_owns", False))

    def _on_inner_tab_changed(self, index: int) -> None:
        """One inner-tab switch: hold the picture, then hand the pane over if the queue tab is involved.

        The owner's rule is "on ANY switch between tabs, playback pauses automatically" (2026-11-24), so the pause is
        not conditional on which tabs are involved - General <-> Encoder Options included. Only the hand-over itself
        is about the queue tab.
        """
        queue_index = getattr(self, "_queue_tab_index", None)
        entering = queue_index is not None and int(index) == int(queue_index)
        leaving = (not entering) and self._queue_owns_pane()
        self._pause_pane_transport()
        if entering:
            self._enter_queue_pane()
        elif leaving:
            self._leave_queue_pane()

    def _pause_pane_transport(self) -> None:
        """Pause whoever paints: the tail, the finished-file decoder and the queue decoder, in one place."""
        if not bool(getattr(self, "_preview_paused", True)):
            # The shipped handler already pauses every owner and repaints the toggle; reusing it keeps one pause path.
            self._on_preview_pause_toggled(True)
            return
        decoder = getattr(self, "_queue_pane_decoder", None)
        if decoder is not None:
            try:
                decoder.pause()
            except Exception as exc:
                log(f"queue pane: pause failed ({exc})")
        self._update_display_wanted()

    def _enter_queue_pane(self) -> None:
        """The queue tab now owns the pane: the remembered item at its playhead, or an EMPTY field."""
        self._queue_pane_owns = True
        # The strip's signals are wired by the render owners (`_prepare_display_state`, the tail, the finished-file
        # playback), so on an instance that has not rendered yet the queue item's strip was deaf: a press was accepted by
        # the widget and emitted into nothing (measured live 2026-11-24: the queue item showed its first frame and a press
        # at 40 % left the playhead at 0.0 with no `seek:` line at all). Wiring here is idempotent.
        self._wire_preview_strip()
        path = getattr(self, "_queue_pane_path", None)
        if path:
            self._show_queue_item(path)
        else:
            self._clear_pane()
        self._apply_seek_ceiling()
        self._sync_preview_transport()

    def _leave_queue_pane(self) -> None:
        """Give the pane back to the render owner at ITS position, and stop the queue decoder entirely."""
        self._remember_queue_position()
        self._queue_pane_owns = False
        self._stop_queue_decoder()
        self._repaint_render_owner()
        self._apply_seek_ceiling()
        self._sync_preview_transport()

    def _on_queue_row_changed(self, row: int) -> None:
        """A queue row was selected (a click, or the list's own selection move): that file goes on the pane.

        `-1` is what the list emits when it is rebuilt or cleared, not a selection: a cleared queue empties the pane
        (nothing it shows is still queued) and any other rebuild leaves the picture alone. A row outside the queue (a
        stale signal after a removal) is dropped rather than guessed at. A selection must not paint while another tab
        is current - the render owns the pane there.
        """
        try:
            row = int(row)
        except (TypeError, ValueError):
            return
        queue = getattr(self.parent, "renderQueue", None)
        items = list(getattr(queue, "getQueue", lambda: [])() or [])
        if row < 0:
            # `-1` is the list saying "nothing is current" (a rebuild, or the queue emptied). A cleared queue must not
            # leave a picture of an item that is gone on the pane; the remembered PATH is kept - it is the selection
            # memory, not the row.
            if self._queue_owns_pane() and not items:
                self._clear_pane()
            return
        if not 0 <= row < len(items):
            return
        path = str(getattr(items[row], "inputFile", "") or "")
        if not path:
            return
        self._queue_pane_path = path
        if not self._queue_owns_pane():
            return
        self._show_queue_item(path)

    def _show_queue_item(self, path: str) -> None:
        """One decoder on the selected file, held at its remembered playhead (`self._queue_pane_positions`).

        Re-entering the same path reuses the open decoder; a different file closes the previous one first, so exactly
        one queue decoder exists. Decoding happens at the PANE's box (`_pane_display_box`), never at the file's own
        resolution - see `VideoDecoder(max_size=...)`.
        """
        path = str(path or "")
        if not path:
            self._queue_pane_path = None
            self._clear_pane()
            return
        self._queue_pane_path = path
        position = float(self._queue_pane_positions.get(path, 0.0) or 0.0)
        if self._queue_pane_decoder is None or self._queue_pane_decoder_path != path:
            self._stop_queue_decoder()
            decoder = self._new_queue_decoder()
            if decoder is None:
                self._clear_pane()
                return
            try:
                opened = decoder.open(path)
            except Exception as exc:
                opened = False
                log(f"queue pane: cannot open {path!r} ({exc})")
            if not opened:
                log(f"queue pane: cannot open {path!r}")
                try:
                    decoder.close()
                except Exception:
                    pass
                self._clear_pane()
                return
            self._queue_pane_decoder = decoder
            self._queue_pane_decoder_path = path
        decoder = self._queue_pane_decoder
        try:
            decoder.pause()
        except Exception as exc:
            log(f"queue pane: pause failed ({exc})")
        self._queue_pane_position = position
        try:
            # A paused seek publishes exactly one frame at the target and stays held: the remembered position is shown,
            # playback does not start by itself (the transport's own rule for this pane).
            decoder.seek(position)
        except Exception as exc:
            log(f"queue pane: seek to {position:.3f}s failed ({exc})")
        self._apply_seek_ceiling()

    def _new_queue_decoder(self):
        """Build the queue decoder at the pane's box; `None` when this build has no `max_size` (an older decoder)."""
        from .video.decode import VideoDecoder

        try:
            decoder = VideoDecoder(pixel_format="rgb24", max_size=self._pane_display_box())
        except TypeError:              # a decoder without the cap: full resolution, still one painter
            decoder = VideoDecoder(pixel_format="rgb24")
        decoder.frame_ready.connect(self._on_queue_pane_frame)
        decoder.opened.connect(self._on_queue_pane_opened)
        decoder.error.connect(lambda message: log(f"queue pane: {message}"))
        return decoder

    def _on_queue_pane_opened(self, duration: float, fps: float, frames_total: float) -> None:
        self._queue_pane_duration = float(duration) or 0.0

    def _on_queue_pane_frame(self, frame) -> None:
        """One frame from the queue decoder: painted only while the queue tab still owns the pane.

        `frame_ready` is delivered queued, so a decoder closed by a tab switch can deliver a frame it emitted before
        the close. That frame has no owner any more and is dropped: the hand-back is the rule, not the delivery order.
        """
        if not self._queue_owns_pane():
            return
        if frame is None:
            return
        from PySide6.QtGui import QImage

        path_started = time.perf_counter()
        image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB888).copy()
        pane = self.page_handles.proc.previewLabel
        if hasattr(pane, "set_frame"):
            pane.set_frame(image, gui_ms=(time.perf_counter() - path_started) * 1000.0,
                           index=getattr(frame, "index", None), media_time=getattr(frame, "pts", None))
        elif hasattr(pane, "setPixmap"):
            pane.setPixmap(self.getRoundedPixmap(QtGui.QPixmap.fromImage(image), corner_radius=10))
        pts = float(getattr(frame, "pts", 0.0) or 0.0)
        self._queue_pane_position = pts
        path = getattr(self, "_queue_pane_path", None)
        if path:
            self._queue_pane_positions[path] = pts
        duration = float(getattr(self, "_queue_pane_duration", 0.0) or 0.0)
        try:
            if hasattr(pane, "set_progress"):
                # A finished, fully decodable file: everything exists, so the whole strip is produced.
                pane.set_progress(rendered=duration, total=duration, playhead=pts, duration=duration)
        except Exception:
            pass

    def _remember_queue_position(self) -> None:
        path = getattr(self, "_queue_pane_path", None)
        if path:
            self._queue_pane_positions[path] = float(getattr(self, "_queue_pane_position", 0.0) or 0.0)

    def _stop_queue_decoder(self) -> None:
        """Release the queue decoder: nothing decodes while the queue tab does not own the pane."""
        decoder = getattr(self, "_queue_pane_decoder", None)
        self._queue_pane_decoder = None
        self._queue_pane_decoder_path = None
        if decoder is not None:
            try:
                decoder.close()
            except Exception:
                pass

    def _repaint_render_owner(self) -> None:
        """Hand the pane back to whoever followed the render, standing where IT stood.

        Three possible render owners, checked in the order the pane's painters are installed: the live tail (a
        preview), the finished-file decoder (after a preview job), the display feed (an export). The position is the
        render's own - the produced edge for a live feed/tail, the held playhead for a finished file - never the queue
        item's playhead the pane was just showing.
        """
        tail = getattr(self, "_live_tail", None)
        if tail is not None:
            target = self._render_head_seconds(tail)
            try:
                if target > 0.0:
                    tail.seek(target, attempt_past_edge=True)
                tail.set_paused(True)
            except Exception as exc:
                log(f"queue pane: handing back to the tail failed ({exc})")
            return
        decoder = getattr(self, "_preview_decoder", None)
        if decoder is not None:
            try:
                decoder.pause()
                decoder.seek(max(0.0, float(getattr(self, "_preview_playhead", 0.0) or 0.0)))
            except Exception as exc:
                log(f"queue pane: handing back to the finished file failed ({exc})")
            return
        reader = getattr(self, "_display", None)
        if reader is not None and bool(getattr(reader, "attached", False)):
            clip = self._clip_duration()
            fraction = self._live_produced_fraction()
            if fraction <= 0.0 and clip > 0.0:
                fraction = min(1.0, max(0.0, float(getattr(self, "_preview_playhead", 0.0) or 0.0) / clip))
            if self._request_display_seek(fraction, hold=True):
                return
        # No render owner at all: nobody may paint a queue item outside its tab.
        self._clear_pane()

    def _render_head_seconds(self, tail) -> float:
        """The head the renderer has reached, for a hand-back: produced share x clip length, else the tail's edge."""
        clip = self._clip_duration()
        produced = self._live_produced_fraction()
        if clip > 0.0 and produced > 0.0:
            return produced * clip
        return float(getattr(tail, "produced_seconds", 0.0) or 0.0)

    def _clear_pane(self) -> None:
        """The EMPTY field: no picture and no markers (the field a queue tab shows before anything was selected).

        The render's own playhead is deliberately NOT reset: it is where the render owner stands when the pane is
        handed back to it, and zeroing it here would lose the position the owner asked to keep.
        """
        pane = getattr(getattr(self.page_handles, "proc", None), "previewLabel", None)
        if pane is None:
            return
        try:
            setter = getattr(pane, "set_frame", None)
            if callable(setter):
                setter(None)
            elif hasattr(pane, "clear"):
                pane.clear()
            elif hasattr(pane, "setPixmap"):
                pane.setPixmap(QtGui.QPixmap())
        except Exception:
            pass
        self._queue_pane_position = 0.0
        try:
            if hasattr(pane, "set_progress"):
                pane.set_progress(rendered=0, total=0, playhead=0.0, duration=0.0)
        except Exception:
            pass

    def _on_preview_seek(self, fraction: float) -> None:
        """A press or drag on the strip seeks whoever owns the picture.

        Live, that is the tail: it repositions inside what the renderer has already written, and the next poll
        (120 ms) shows the target. A press PAST the produced band is different only while the pane's block is usable:
        the tail has nothing there, but the renderer can render that one frame on demand - so the press becomes a
        newest-wins request (`_request_display_seek`) and the pane holds it until play hands it back. A finished file
        seeks its own decoder. The strip reports a share of the WHOLE timeline, so while the timeline length is
        unknown the produced share stands in for it - which is the same assumption the strip itself makes when it
        fills its white band.
        """
        fraction = max(0.0, min(1.0, float(fraction)))
        self._seek_pressed_at = time.perf_counter()
        # The press names the FRAME it points at, not only the share: a turnaround in milliseconds says nothing about
        # whether the picture is the one asked for (owner, 2026-11-24: "write where the seek went and what is shown").
        total_frames = float(getattr(self, "max_value", 0) or 0)
        if total_frames <= 0:
            fps = float(getattr(getattr(self, "_preview_decoder", None), "fps", 0) or 0)
            duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
            total_frames = round(duration * fps)
        if total_frames <= 0:
            # The render publishes its total late; the CLIP's own frame count is known from the probe at load time and
            # is the right denominator for a share -> frame conversion in the meantime.
            total_frames = float(getattr(self, "videoFrameCount", 0) or 0)
        target_frame = int(round(fraction * total_frames)) if total_frames > 0 else None
        self._seek_target_frame = target_frame
        # The produced share goes with the press: a press past what the renderer has written is a different event from
        # a press inside it, and only the log can tell them apart afterwards.
        produced = 0.0
        total = float(getattr(self, "max_value", 0) or 0)
        if total > 0:
            produced = max(0.0, min(1.0, float(getattr(self, "currentFrame", 0) or 0) / total))
        target_s = None
        duration_s = float(self._clip_duration() or 0.0)
        if duration_s > 0.0:
            target_s = fraction * duration_s
        self._seek_target_s = target_s
        telemetry.event("seek press", fraction=f"{fraction:.4f}", produced=f"{produced:.4f}",
                        target_s=None if target_s is None else f"{target_s:.3f}",
                        target_frame=target_frame)
        # One record per press, created BEFORE a route is chosen: the intent, the route it took, whether that route
        # can answer it, and (from whichever painter answers) the picture. Without this a press could end in silence -
        # the owner's three sessions pressed four times each and left no line about the result at all (2026-11-24).
        edge_s = produced * duration_s if duration_s > 0.0 else None
        ticket = PressTicket(target_s=target_s, target_frame=target_frame, produced=produced, edge_s=edge_s)
        previous = getattr(self, "_press", None)
        if previous is not None and previous.open:
            previous.superseded(ticket.id)
        self._press = ticket
        if self._queue_owns_pane():
            decoder = getattr(self, "_queue_pane_decoder", None)
            duration = float(getattr(decoder, "duration", 0.0) or 0.0)
            if decoder is not None and duration > 0.0:
                ticket.decide(route="queue", answerable=True)
                try:
                    decoder.seek(fraction * duration)
                except Exception as exc:
                    ticket.refused(f"queue-decode-failed: {exc}")
                    log(f"queue pane: seek failed ({exc})")
            else:
                ticket.refused("queue-has-no-duration")
            return
        live = getattr(self, "_live_tail", None)
        produced = self._live_produced_fraction()
        clip = self._clip_duration()
        # The instrument: every press says what it meant, in the units the answer will be checked against. One line per
        # press (a drag sends many), which is why the target is what is logged and not the fraction alone.
        log(f"seek: press {fraction:.1%} of the strip (clip {clip:.3f}s, produced {produced:.1%}, "
            f"tail={'yes' if live is not None else 'no'}, preview={bool(getattr(self, 'isPreview', False))})")
        if (bool(getattr(self, "isPreview", False))
                and self._display_can_seek_any_frame()
                and (live is None or fraction > produced + 1e-6)):
            # The renderer can render this frame even though the tail cannot: the same block an export uses. Inside
            # the produced band the tail stays the answer (exact, and it costs the renderer nothing); past it the
            # press is one request and a hold on the picture it yields.
            if self._request_display_seek(fraction, hold=True):
                return
            ticket.refused("no-process-or-unknown-length")
            log("seek: refused - the running render cannot answer that press (no process, or the clip length is unknown)")
            return
        if live is not None:
            edge = float(getattr(live, "produced_seconds", 0.0) or 0.0)
            if edge <= 0.0:
                ticket.decide(route="tail", answerable=False, why="nothing-written-yet")
                ticket.refused("nothing-written-yet")
                log("seek: refused - nothing is written yet")
                return                      # nothing is written yet, so there is nothing to seek to
            target = fraction * self._live_timeline(edge)
            self._seek_watch = target
            live.seek(target)
            ticket.decide(route="tail", answerable=True)
            log(f"seek: tail asked for {target:.3f}s (published edge {edge:.3f}s)")
            return
        decoder = getattr(self, "_preview_decoder", None)
        duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
        if decoder is not None and duration > 0:
            ticket.decide(route="decoder", answerable=True)
            decoder.seek(fraction * duration)
            return
        # EXPORT, third route: no tail and no file decoder, but a renderer is alive and publishing into the pane's
        # block. The strip's share becomes a newest-wins request - `(seconds, generation + 1)` - and the renderer
        # decodes that one frame through the model path and publishes it without writing it to the export. A request
        # only ever goes out while something renders; with no process the block is left alone (nothing would answer).
        # A PREVIEW with no usable block keeps today's refusal: its tail is all there is.
        if bool(getattr(self, "isPreview", False)):
            ticket.refused("preview-without-a-block")
            return
        # An export press used to pass `hold=False` - "its feed keeps painting the moving band" - which is exactly what
        # the owner saw: the newest upscaled frame painted over his request and the marker dragged to the produced edge,
        # pause notwithstanding (2026-11-24). A press on a PAUSED pane parks on the requested picture; while playing it
        # stays a live follow, and the answer is one publication in the stream.
        if not self._request_display_seek(fraction, hold=bool(getattr(self, "_preview_paused", False))):
            # Nobody will answer: the render has finished (or never started), and an export has no file decoder behind
            # the pane. Refusing keeps the pane honest - it also clears the pending flag, because a parked pane that
            # waits forever for a picture is the "video became unavailable after the render" the owner reported
            # (2026-11-25). Reported instead of silent: the press was logged as decided and then vanished.
            ticket.refused("no-render-and-no-file-decoder")
            self._display_seek_pending = False
            self._update_display_wanted()

    def _request_display_seek(self, fraction: float, *, hold: bool) -> bool:
        """Ask the running render for ONE frame at `fraction` of the clip; True when the request went out.

        The block holds one `(seconds, generation + 1)` slot, so a drag replaces its predecessor (newest wins) and the
        renderer decodes the requested timestamp between frames (`RenderVideo._seek_preview`), publishing it without
        writing it to the export. `_on_display_frame` paints it.

        `hold` is a PREVIEW press: its tail cannot pass a frame it has not produced, so the transport parks on the
        requested picture (and play hands the pane back, following the produced edge as before). An export's press
        does not park anything - its feed keeps painting the moving band.
        """
        reader = getattr(self, "_display", None)
        if reader is None or not bool(getattr(reader, "attached", False)):
            return False
        process = getattr(self, "renderProcess", None)
        if process is None or process.poll() is not None:
            return False                # nothing would answer a request; the block is left alone
        clip = self._clip_duration()
        if clip <= 0.0:
            return False                # the timeline length is unknown, so a share cannot become seconds
        target = fraction * clip
        # The seek slot belongs to the reader's block; with a feed attached the write goes through the feed, which
        # is the object that owns "the pane asked for this generation" (newest wins, one slot).
        chain = self._display_chain()
        if hold and not bool(getattr(self, "_preview_paused", True)):
            # Park the transport on the frame it asked for. A playing tail would otherwise paint its own (produced)
            # frame over it a moment later; play resumes the follow from the produced edge, as before.
            self._on_preview_pause_toggled(True)
        unanswered = getattr(self, "_seek_pressed_at", None) is not None
        previous_generation = getattr(self, "_seek_generation", None)
        generation = chain.request_seek(target)
        self._seek_generation = generation or None
        spot = getattr(self, "_press", None)
        if spot is not None and spot.open:
            # "Can this be answered" is known now, in milliseconds: a produced frame exists or it does not. No stopwatch
            # - `answerable=0` with `why=target-past-produced-edge` is the explanation for a press that used to leave no
            # line at all (owner, 2026-11-24: four presses, produced=0.0000, nothing drawn, nothing logged).
            produced_s = float(self._live_produced_fraction() or 0.0) * clip
            ahead = target > produced_s + 1e-6
            spot.decide(route="display", answerable=True, ahead=ahead,
                        why="decoded-on-demand" if ahead else None, generation=generation)
        if hold or bool(getattr(self, "_preview_paused", True)):
            # A held pane has `wanted` 0, so the renderer would stage the requested frame and never publish it. Ask
            # for exactly that one frame: `_on_display_frame` paints it and clears this again.
            self._display_seek_pending = True
            chain.set_wanted(True)
        if hold:
            # The marker and the compare's source clock stand on the requested frame; the produced band keeps
            # advancing from the renderer's own counters (republished with the frame).
            self._preview_playhead = float(target)
        log(f"seek: requested {target:.3f}s (generation {generation}) from the running render")
        return True

    def _live_timeline(self, edge: float) -> float:
        """The whole timeline a strip fraction is a share of, for the live tail.

        The clip's own length when the job knows it (`videoFrameCount / videoFps`): the strip's white band is
        `currentFrame / max_value`, i.e. the produced share of THAT length, so a press at 30 % has to mean 30 %
        of it or the marker lands somewhere the finger did not. `edge / produced` is only the fallback: it is
        exact while the tail stands on the produced edge, and a HELD pane does not - it stops polling, so `edge`
        freezes at the last frame it published while the renderer (and the band) keeps going. Recomputing the
        timeline from that frozen edge made the same finger position mean a different second on every press.

        Returns 0.0 when neither is known: the caller then has nothing honest to seek to.
        """
        clip = self._clip_duration()
        if clip > 0.0:
            return float(clip)
        produced = self._live_produced_fraction()
        return float(edge) / produced if produced > 0.0 else float(edge)

    def _live_produced_fraction(self) -> float:
        """The share of the timeline the renderer has produced, from the same counters the strip fills with."""
        rendered = float(self.currentFrame or 0)
        total = float(self.max_value or 0)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, rendered / total))

    def _live_duration_estimate(self) -> float:
        """The timeline length the strip draws its playhead against, so the marker is a place on the clip.

        The clip's own length when the job knows it: the band is the produced share of it (`_live_produced_fraction`),
        and the seek maps a press with the same number (`_live_timeline`), so press, marker and band share one axis.
        Otherwise `edge seconds / share produced` is the estimate: exact while the renderer emits frames uniformly,
        which is what the strip's own `rendered/total` fill already assumes. A live render with no frame yet reports
        0, and the strip hides a marker it cannot place honestly.
        """
        live = getattr(self, "_live_tail", None)
        if live is None:
            return 0.0
        clip = self._clip_duration()
        if clip > 0.0:
            return float(clip)
        edge = float(getattr(live, "produced_seconds", 0.0) or 0.0)
        produced = self._live_produced_fraction()
        if edge <= 0.0 or produced <= 0.0:
            return 0.0
        return edge / produced

    def _on_original_requested(self, seconds: float) -> None:
        """Decode ONE source frame at the playhead for the LMB compare (the input file, never the renderer).

        The image is what the platform's decoder produces from the input, which is the honest comparison: the
        renderer's own source buffer would need a second channel and would make the render pay for a preview-only
        action. It is then cropped to the box the render reported (`BORDERS`, parsed out of this job's output) - see
        `_on_original_frame`.
        """
        if self._queue_owns_pane():
            # A queue item IS the source: there is no render to compare it against, and decoding the last render's
            # input file over it would show a picture of a different video.
            return
        # The pane's own `originalRequested` (a compare mode's playhead moved) and the 100 ms tick both arrive here;
        # recording what was asked keeps one moved playhead to one source decode.
        self._second_surface_pts = float(seconds)
        # The compare's own instrument: what was asked, and (in `_on_original_frame`) what came back. The owner's
        # report was "a couple of frames after the right one" and neither number existed anywhere before this line.
        self._compare_watch = float(seconds)
        if not getattr(self, "_compare_from_tick", False):
            pane = self.page_handles.proc.previewLabel
            showing = "-"
            try:
                shown = pane.shown_time() if hasattr(pane, "shown_time") else None
                showing = "-" if shown is None else f"{float(shown):.3f}"
            except Exception:
                pass
            log(f"compare: asked for the source at {seconds:.3f}s (pane playhead "
                f"{float(getattr(self, '_preview_playhead', 0.0)):.3f}s, showing {showing}s, "
                f"mode {getattr(pane, '_view_mode', '?')})")
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
            decoder.request_frame(max(0.0, float(seconds)),
                                  exact=not bool(getattr(self, "_shown_name_is_fitted", False)))
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
        watched = getattr(self, "_compare_watch", None)
        self._compare_watch = None
        pts = float(getattr(frame, "pts", 0.0) or 0.0)
        index = getattr(frame, "index", None)
        pane = self.page_handles.proc.previewLabel
        # The LAST step's other half: what the pane was showing when the press asked for this source frame. The labels
        # can agree and the pictures still differ (a shifted file timeline names every frame late), so the draw line in
        # the pane's own log carries a content fingerprint next to this one.
        result_time = None
        result_kind = None
        try:
            result_time = pane.shown_time() if hasattr(pane, "shown_time") else None
            result_kind = pane.shown() if hasattr(pane, "shown") else None
        except Exception:
            pass
        if watched is not None:
            fps = 0.0
            try:
                fps = float(self._source_decoder.fps()) if getattr(self, "_source_decoder", None) is not None else 0.0
            except Exception:
                fps = 0.0
            fps = fps if fps > 0.0 else 24.0
            gap = "" if result_time is None else (
                f", result {float(result_time):.3f}s {result_kind}"
                f" -> gap {pts - float(result_time):+.3f}s ({(pts - float(result_time)) * fps:+.1f} frames)")
            log(f"compare: the source frame is {pts:.3f}s idx={index} for an ask of {watched:.3f}s "
                f"(delta {pts - watched:+.3f}s, {abs(pts - watched) * fps:.1f} frames at {fps:.2f} fps{gap})")
        if hasattr(pane, "set_original"):
            pane.set_original(image, media_time=pts, index=index)

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
        decoded frame and neither could be trusted. A preview keeps its tail as the only painter (`wanted` stays 0) -
        except while the pane waits for the one frame a strip press asked the block for: `_display_paints_pane` admits
        the feed for exactly that frame, so a tail frame arriving in the meantime is dropped here, not painted over it.

        `frame_ready` is delivered queued, so a tail stopped at the job's end can still deliver a frame it emitted
        before the stop. Such a frame has no current owner (the tail is gone, and playback of a finished file installs
        its own decoder) and is dropped: a zombie painter is the same rule violation as an early one.
        """
        from PySide6.QtGui import QImage

        if self._display_paints_pane():
            return
        if self._queue_owns_pane():
            return
        if (getattr(self, "_live_tail", None) is None
                and getattr(self, "_preview_decoder", None) is None):
            return
        path_started = time.perf_counter()
        image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB888).copy()
        # The file's own clock, once: the writer's `make_zero` muxes a content-zero frame at the encoder's B-frame
        # delay, and the tail subtracts that origin from every name (measured 2026-11-25: 0.083 s, two frames).
        if getattr(self, "_live_tail", None) is not None and not getattr(self, "_tail_origin_logged", False):
            origin = getattr(self._live_tail, "media_origin", None)
            if origin is not None:
                self._tail_origin_logged = True
                named = float(getattr(frame, "pts", 0.0) or 0.0)
                log(f"preview: file timeline starts at {float(origin):.3f}s; picture names are CONTENT times "
                    f"(this one is {named:.3f}s = {named + float(origin):.3f}s in the file)")
        pane = self.page_handles.proc.previewLabel
        if hasattr(self.page_handles.proc.previewLabel, "set_frame"):
            pane.set_frame(image, gui_ms=(time.perf_counter() - path_started) * 1000.0,
                           index=getattr(frame, "index", None), media_time=getattr(frame, "pts", None))
            self._preview_playhead = float(getattr(frame, "pts", 0.0) or 0.0)
            watched = getattr(self, "_seek_watch", None)
            if watched is not None:
                # The seek instrument the owner asked for: one line per PRESS (not per frame), naming where the press
                # asked to go and where the picture actually landed. Without it a press and the frame it produced were
                # two separate facts and "it still does not work" could not be argued either way.
                self._seek_watch = None
                pressed_at = getattr(self, "_seek_pressed_at", None)
                if pressed_at is not None:
                    shown = getattr(pane, "shown", None)
                    turnaround_ms = (time.perf_counter() - pressed_at) * 1000.0
                    spot = getattr(self, "_press", None)
                    if spot is not None and spot.open:
                        spot.settled(landed=shown() if callable(shown) else None,
                                     landed_time=float(getattr(frame, "pts", 0.0) or 0.0),
                                     turnaround_ms=turnaround_ms, landed_gen=None)
                    else:
                        telemetry.note_seek(turnaround_ms, target=getattr(self, "_seek_target_frame", None),
                                            landed=shown() if callable(shown) else None)
                    self._seek_pressed_at = None
                    # The request is answered, by this path or not at all: a later press must not call it superseded.
                    self._seek_generation = None
                log(f"seek: press wanted {watched:.3f}s, the picture landed at {self._preview_playhead:.3f}s "
                    f"(delta {self._preview_playhead - watched:+.3f}s)")
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
        # The block is allocated once per pass, before the job's own `isPreview` is known (the worker sets it), so the
        # pane's seek reach is re-asserted here, on the GUI thread, once the owner is unambiguous.
        self._apply_seek_ceiling()
        # Keep the transport honest for whichever painter owns the pane this tick (the tail or the file decoder): an
        # enabled button that holds nothing, or a disabled one that holds the picture, is worse than no button.
        self._sync_preview_transport()
        self._refresh_held_progress()
        # The export pane's feed and the standing original both run on this same clock: the reader takes at most one
        # new `seq` per tick, and the compare's source decode is coalesced to the playhead the result just reached.
        self._tick_display()
        self._tick_second_surface()


        if self.renderTextOutputList is not None:
            # The numbers the labels below show come from the render-status slot (the backend writes them
            # beside its log lines); nothing here parses the backend's stdout for them any more.
            self._sync_render_status()
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
            # The three readouts are rebuilt here EVERY tick, so their labels go through the catalogue (`i18n.tr`,
            # keyed by the English source string) rather than being literals: a literal made a switch to `ru` survive
            # exactly until the next tick (measured 2026-11-24, this line rewrote the chrome's `К/с:` back to `FPS:`).
            # The keys are the prefixes as written - `FPS: ` and `Status: ` are NOT the chrome's `FPS:`/`STATUS:`.
            if self.fps != 0:
                self.page_handles.proc.FPS.setVisible(True)
                self.page_handles.proc.FPS.setText(i18n.tr("FPS: ") + f"{self.fps}")
            else:
                self.page_handles.proc.FPS.setVisible(False)
            if self.eta != 0:
                self.page_handles.proc.ETA.setVisible(True)
                self.page_handles.proc.ETA.setText(i18n.tr("ETA: ") + f"{self.eta}")
            else:
                self.page_handles.proc.ETA.setVisible(False)
            self.page_handles.proc.STATUS.setText(i18n.tr("Status: ") + self.status)

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
            display_box_id=DISPLAY_BOX_SHARED_MEMORY_ID,
            status_state_id=STATUS_STATE_SHARED_MEMORY_ID,
        )
