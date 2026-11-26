import subprocess
import os
import time
from threading import Thread
import re
from time import sleep
from multiprocessing import shared_memory

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor
from PySide6.QtCore import QObject, QSize, Qt, QUrl, Slot

from .RenderQueue import RenderQueue
from ..render_jobs import short_duration
from .render_log import LogLine, RenderLogFilter
from .video import telemetry
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
    SOURCE_STATE_SHARED_MEMORY_ID,
    STATUS_STATE_SHARED_MEMORY_ID,
    DisplayReader,
    format_eta,
)
from .video.feed import PreviewFeed
from .preview.owner import owner_of
from .preview.session import SEEK_WATCH_TIMEOUT_S as SESSION_SEEK_WATCH_TIMEOUT_S
from .preview.state import PreviewBackendKind
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


#: How far AHEAD of the result's playhead the compare's source half is decoded, in frames. The point is HEADROOM, not
#: throughput: a source frame costs 12.5 ms to decode (measured on the 1440p HEVC source: 48 frames in 0.60 s, an 80 fps
#: ceiling against a 24 fps stream), so three frames ahead cost 37 ms once and 12.5 ms per 41.7 ms after that - ~30 % of
#: one core, and in exchange the frame the pane is about to show is already decoded when the result arrives, so both
#: halves change in ONE repaint (owner, 2026-11-26: "нам в теории наоборот с запасом надо сделать сразу").
COMPARE_LEAD_FRAMES = 3

#: How long one source ask counts as "in flight". A span of `COMPARE_LEAD_FRAMES` costs ~40 ms of decode, so this is a
#: retry deadline, not a throttle: an ask nothing answers is re-sent on the next tick after it, which keeps a paused
#: pane's first frame arriving as quickly as the retry itself. It must stay BELOW the picture interval of a live tail
#: (~60 ms at 16 fps): at 250 ms a pending blocked two to three fills per frame, the pool fell 1-3 frames behind the
#: result and every pair timed out (measured 2026-11-26, split during inference).
COMPARE_FILL_WINDOW_S = 0.08

#: How many decoded source frames are kept. One second at 24 fps: a live tail's pictures are SPARSE (its lateness rule
#: skips frames, so consecutive ones can be six frames apart) and the pool is pruned to this window, so a six-frame jump
#: against a 0.25 s window emptied it on every skip and no pair ever closed (measured 2026-11-26). The frames are decoded
#: at the pane's box (1.5 MB), so a second of them is 36 MB - the cap is a memory budget, not a scheduling knob.
COMPARE_POOL_FRAMES = 24


class _FrameRelay(QObject):
    """Deliver one decoder frame on the GUI thread.

    `ProcessTab` is a plain object, not a `QObject`, so a signal connected straight to one of its methods runs IN THE
    EMITTING THREAD. With the compare's decoder on its own `QThread` that meant `set_original` -> `update()` from a
    worker thread: Qt's "Cannot make QOpenGLContext current in a different thread" and a dead window (measured
    2026-11-26, the first run after the decoder was moved off the GUI thread). The relay LIVES in the GUI thread, so
    the connection to it is queued and the callback runs where widgets may be touched.
    """

    def __init__(self, callback) -> None:
        super().__init__()
        self._callback = callback

    @Slot(object)
    def deliver(self, frame) -> None:
        self._callback(frame)


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


class _PreviewSurfacePort:
    """The page's picture verbs for `PreviewController` (view mode, zoom, compare, fullscreen).

    The controller owns the state and the commands; this object is the only place the state reaches the pane/widgets,
    and it uses the pane's PUBLIC API (`set_view_mode`, `set_zoom`, `reset_view`, `set_original`) - the internals of
    `video_preview.py` belong to another lane.
    """

    def __init__(self, page: "ProcessTab") -> None:
        self._page = page

    def _pane(self):
        return getattr(getattr(self._page.page_handles, "proc", None), "previewLabel", None)

    def set_view_mode(self, mode: str) -> None:
        pane = self._pane()
        setter = getattr(pane, "set_view_mode", None)
        if callable(setter):
            try:
                setter(str(mode))
            except Exception as exc:
                log(f"preview: view mode {mode} failed ({exc})")

    def set_zoom(self, zoom: float, *, fit: bool) -> None:
        pane = self._pane()
        if pane is None:
            return
        try:
            if fit:
                pane.reset_view()
            else:
                setter = getattr(pane, "set_zoom", None)
                if callable(setter):
                    setter(float(zoom))
        except Exception as exc:
            log(f"preview: zoom {zoom} (fit={fit}) failed ({exc})")

    def set_compare(self, held: bool) -> None:
        """Hold the SOURCE at the playhead, or give the pane back to the result.

        The decode is the page's existing compare route (`_on_original_requested` -> decoder -> `_on_original_frame` ->
        `pane.set_original`): one source decode, cropped to the render's box. Releasing clears the held picture.
        """
        pane = self._pane()
        if held:
            try:
                self._page._on_original_requested(float(getattr(self._page, "_preview_playhead", 0.0) or 0.0))
            except Exception as exc:
                log(f"preview: compare source decode failed ({exc})")
            return
        setter = getattr(pane, "set_original", None)
        if callable(setter):
            try:
                setter(None)
            except Exception:
                pass

    def set_fullscreen(self, fullscreen: bool) -> None:
        window = self._page.window() if callable(getattr(self._page, "window", None)) else None
        if window is None:
            window = getattr(self._page, "parent", None)
        if window is None:
            return
        try:
            if fullscreen:
                window.showFullScreen()
            else:
                window.showNormal()
        except Exception as exc:
            log(f"preview: fullscreen {fullscreen} failed ({exc})")


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
        #: The control surface's one source of truth (`src/ui/preview/`): the page feeds it facts, the bars render it.
        #: Built in `_install_preview_controls` (QConnect) once the page's widgets exist; `None` on a page without a
        #: preview surface (the raster fallback or a stub in a test).
        self._preview_controller = None
        self._preview_controls = None
        self._preview_surface = None
        self._preview_zoom = 1.0
        self._preview_zoom_fit = True
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
        #: itself stays the reader's fact, and it is one of the raw inputs `preview.owner.owner_of` reads: that function
        #: is the ONE answer to "who paints, who may ask for a frame, how far may a press reach" (plan P2), and the
        #: seek/wanted writes reach the block through this object.
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
        #: The compare's decoded source frames: `pts -> (index, QImage)`, kept AHEAD of the playhead (see
        #: `COMPARE_LEAD_FRAMES`) so the pane never waits for a decode to draw the other half.
        self._source_pool: dict[float, tuple[object, object]] = {}
        #: The pts currently handed to the pane, so a repaint happens only when the PICTURE changes.
        self._source_shown_pts: float | None = None
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
        # The new transport (plan-rve-preview-controls-frame-path-2026-11-25, layout E): the floating tools over the
        # picture and the transport in the letterbox replace the painted strip and the page's one pause button.
        self._install_preview_controls()
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
        """Start the page's 100 ms clock - a GUI-thread clock, which is not obvious from the connection (plan P6).

        `ProcessTab` is not a `QObject`, so `tick.connect(self.updateProcessTab)` binds a plain callable: there is no
        receiver object whose thread affinity Qt could read from the page, so the connection is resolved by the proxy
        PySide builds for that callable - and that proxy's delivery lands on the thread that owns the `QApplication`,
        never on the emitting thread and not on the connecting one (measured both ways). Measured on this host, PySide6
        6.12.0 / Qt 6.12.0, in this repo's own shape (`UpdateGUIThread` emitting from its `run()`, a thread with no event
        loop of its own): no call happens at emit time, and every call lands on the GUI thread at `TICK_SECONDS`. So the
        whole body (labels, console, the two painters, `_tick_press_deadline`) already runs on the GUI thread, and a slow
        tick starves the GUI event loop rather than the tick thread (that is the 2026-11-22 freeze `syncRenderConsole`
        was written for).

        The contract is pinned by `harness/tests/test_preview_state_tick.py::test_the_tick_body_runs_on_the_gui_thread`
        (real thread, real `updateProcessTab`, both hooks recording `threading.get_ident()`). It is why the page may touch
        widgets from `updateProcessTab` at all; re-measure it before moving the tick, since P6 lands last behind it.
        """
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
        #: The COMPARE's block: the renderer publishes the source picture of each composited frame into it, so `sbs` and
        #: `split` need no second decode of the input. Allocated with the display block (same box, same lifetime), and an
        #: unattached one is simply absent - every reader then falls back to its own decoder.
        self._source_feed = None
        self._source_feed_alive_at = None
        try:
            source_feed = DisplayReader(SOURCE_STATE_SHARED_MEMORY_ID, box=box)
            if source_feed.attached:
                source_feed.display_frame.connect(self._on_source_feed_frame)
                self._source_feed = source_feed
            else:
                source_feed.release()
        except Exception as exc:
            log(f"source: the compare's block is not usable ({exc}); compare modes decode their own half")
            self._source_feed = None
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
        """Release the frame, seek, announced-box and compare blocks; safe twice, and safe when nothing was prepared."""
        source_feed = getattr(self, "_source_feed", None)
        self._source_feed = None
        self._source_feed_alive_at = None
        if source_feed is not None:
            try:
                source_feed.release()
            except Exception:
                pass
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
        owner = owner_of(self)
        if owner.kind is PreviewBackendKind.QUEUE_FILE:
            # The queue item is a finished file: every frame of it can be decoded, so the strip may address all of it.
            reach = 1.0 if float(getattr(self, "_queue_pane_duration", 0.0) or 0.0) > 0.0 else None
        else:
            reach = 1.0 if owner.may_seek_any else None
        try:
            setter(reach)
        except Exception:
            pass

    def _display_pane_shown(self) -> bool:
        """Whether the pane is on screen - the one WIDGET read that is not the owner's.

        `owner.may_want` is "the transport wants a frame"; this is "somebody is there to see it". The two are ANDed
        where the request byte is written (`_update_display_wanted`, `_tick_display`): a hidden pane must cost the
        renderer nothing, and a widget call may not live in a Qt-free decision.
        """
        pane = getattr(getattr(getattr(self, "page_handles", None), "proc", None), "previewLabel", None)
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
        """Re-assert the request byte after anything that changes the answer (pause, job start, page hide).

        The answer is the owner's (`may_want`: paints, the transport is not holding, a render is alive) ANDed with the
        one widget fact it may not carry (the pane is on screen).
        """
        display = self._display_chain()
        if display is None or not display.attached:
            return
        display.set_wanted(owner_of(self).may_want and self._display_pane_shown())

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

    #: The picture clock, in ms. `updateProcessTab` is the page's 100 ms READOUT tick; the export pane's feed used to ride
    #: it, which capped the preview at 10 pictures a second during a render no matter how fast the model was (measured
    #: 2026-11-26: paints at exactly 100 ms while the render produced ~29 fps of content). The picture path is cheap - one
    #: `poll` that takes at most one new picture, plus a pool lookup - so it gets a clock of its own. 33 ms is ~30 Hz: the
    #: panel cannot show more, and the reader keeps only the newest picture anyway.
    PICTURE_TICK_MS = 33

    def _ensure_picture_timer(self) -> None:
        """Create the picture clock once, on the GUI thread (`updateProcessTab` runs there - see `startGUIUpdate`)."""
        if getattr(self, "_picture_timer", None) is not None:
            return
        from PySide6.QtCore import QTimer

        timer = QTimer()
        timer.setInterval(self.PICTURE_TICK_MS)
        timer.timeout.connect(self._picture_tick)
        timer.start()
        self._picture_timer = timer

    def _picture_tick(self) -> None:
        """The PICTURE path's own clock: the export's feed and the compare's standing original."""
        self._tick_display()
        self._tick_source_feed()
        self._tick_second_surface()
        self._compare_fill_report()

    def _compare_fill_report(self) -> None:
        """One line per second of the compare's supply: asks, frames that came back, and what the pool holds.

        The pair's own counters say the halves disagree; they cannot say whether the source was never ASKED or asked and
        never answered. This is that line, and it is how a frozen pool (asks climbing, arrivals flat) is told apart from a
        clock that picks the wrong frame.
        """
        stats = getattr(self, "_compare_fill_stats", None)
        if stats is None:
            stats = self._compare_fill_stats = {"asks": 0, "arrivals": 0, "at": time.monotonic()}
        now = time.monotonic()
        if now - float(stats["at"]) < 1.0:
            return
        stats["at"] = now
        pool = getattr(self, "_source_pool", None) or {}
        try:
            ph = round(float(self._compare_playhead()), 3)
        except Exception:
            ph = "-"
        newest = None
        if pool:
            newest = round(max(pool), 3)
        telemetry.event("compare fill", asks=stats["asks"], arrivals=stats["arrivals"], pool=len(pool),
                        playhead=ph, newest=newest, target=stats.get("target"), span=stats.get("span"),
                        pool_min=None if not pool else round(min(pool), 3))
        stats["asks"] = 0
        stats["arrivals"] = 0

    def _tick_source_feed(self) -> None:
        """Ask the renderer for the compare's picture, but only in a compare mode with the pane on screen.

        One byte of request and one `poll` per picture tick while `sbs`/`split` is up; nothing at all otherwise, which is
        the whole cost of the block when nobody compares. The picture arrives with the renderer's own media time, i.e.
        the SAME time as the composited frame it belongs to - the two halves are one moment by construction.
        """
        feed = getattr(self, "_source_feed", None)
        if feed is None or not getattr(feed, "attached", False):
            return
        pane = getattr(getattr(getattr(self, "page_handles", None), "proc", None), "previewLabel", None)
        mode = getattr(pane, "view_mode", None)
        comparing = False
        if callable(mode):
            try:
                comparing = mode() in ("sbs", "split")
            except Exception:
                comparing = False
        try:
            visible = bool(pane.isVisible()) if pane is not None else False
        except Exception:
            visible = True
        wanted = bool(comparing and visible)
        feed.set_wanted(wanted)
        if wanted:
            feed.poll()

    def _on_source_feed_frame(self, image) -> None:
        """The renderer's own source picture for the composited frame it just published."""
        feed = getattr(self, "_source_feed", None)
        if feed is None or image is None:
            return
        self._source_feed_alive_at = time.monotonic()
        pane = self.page_handles.proc.previewLabel
        named = getattr(feed, "media_time", None)
        if named is not None:
            # The block names the picture with the RENDERER's media time (the input frame's own PTS), while the pane's
            # clock is the TAIL's content time (the output file's start subtracted). The pair compares them, so the source
            # is moved onto the pane's clock here - the same origin the tail subtracts, taken from the tail itself. Without
            # this the halves sit the encoder's B-frame delay apart and every paint is a mismatch (measured 2026-11-26,
            # with the block already supplying the picture: `asks=0`, `pair_off` equal to the paint count).
            tail = getattr(self, "_live_tail", None)
            origin = getattr(tail, "media_origin", None) if tail is not None else None
            try:
                if origin is not None:
                    named = float(named) - float(origin)
            except Exception:
                pass
        if named is not None:
            # The block carries the source of the frame the RENDERER is compositing - the head. While the pane plays a
            # WRITTEN part of the file the head is seconds away, and taking this picture would compare two different
            # moments (measured 2026-11-26: `result=0.125 source=9.635`, 228 frames apart, because the pane was replaying
            # from 0.125 while the render stood at 9.6). Only a picture of the pane's OWN moment is taken; otherwise the
            # comparison decoder supplies the half, as it must for any rewind.
            try:
                here = float(self._compare_playhead())
            except Exception:
                here = None
            if here is not None and abs(float(named) - here) > 0.25:
                return
        setter = getattr(pane, "set_original", None)
        if callable(setter):
            setter(image, media_time=named)

    def _source_feed_is_live(self) -> bool:
        """True while the renderer has handed the compare a picture in the last second."""
        stamp = getattr(self, "_source_feed_alive_at", None)
        return stamp is not None and (time.monotonic() - float(stamp)) < 1.0

    def _tick_display(self) -> None:
        """One 100 ms tick of the export feed: ask, read at most one new `seq`, paint (GUI thread)."""
        self._announce_pane_box()
        display = self._display_chain()
        if display is None or not display.attached:
            return
        wanted = owner_of(self).may_want and self._display_pane_shown()
        display.set_wanted(wanted)
        if not wanted:
            return
        display.poll()          # the feed emits `picture_changed` only when the sequence moved

    def _on_display_frame(self, image=None) -> None:
        """The renderer's composited frame for the pane (one picture, one painter).

        The owner decides whether the feed may paint at all (`may_paint`): an export's feed does whenever the block is
        usable, a preview's tail is that pane's painter - except the ONE frame a strip press asked for, which the owner
        admits while `_display_seek_pending` is set. While the transport holds a frame, display frames are dropped too
        and the picture stays. The progress is republished with every fresh frame because the strip's produced band is
        the render's own counters, not the picture's timestamp.

        The two checks that follow are about THIS picture, not about the pane, so they stay here: a held pane paints
        only when it asked for one frame, and a pending seek paints only the picture that ANSWERS it (its generation,
        see `_picture_answers_seek`).

        `image` is passed only by a caller that already holds the picture (a test's stub reader emitting its own
        signal); the feed's `picture_changed` carries nothing, so the feed is where the picture and its identity are
        read from.
        """
        feed = getattr(self, "_feed", None)
        if image is None and feed is not None:
            image = feed.image()
        if image is None:
            return
        if not owner_of(self).may_paint:
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
            if media_time is not None:
                # The picture's own time IS the pane's position. Without this the feed's pane left the position at
                # 0.0 and Play then started the file at its beginning instead of where the render stood (measured
                # 2026-11-25: an export at 21 % and a Play that replayed from 0.000 s).
                self._preview_playhead = float(media_time)
            # The pair paints together: the source for this moment is already in the pool, so it is handed over in the
            # same repaint instead of letting the source's own arrival repaint the picture a moment later - which is the
            # "one drawn, then the other" the owner sees (2026-11-26).
            paint_pair = getattr(self, "_source_paint_pair", None)
            if callable(paint_pair):
                paint_pair()
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
            if media_time is not None:
                # The painter reports the picture to the pane's press session (plan P2), AFTER this slot's own settle
                # block has run: the press telemetry keeps its order and its one line per press, and the session then
                # records the picture that stands (or answers) as the pane's position. A display picture's identity is
                # the channel's `seq`, not a frame index, so no index is claimed here; the producers that number
                # frames report theirs in `_on_preview_frame`.
                from .preview.session import Painted

                session_of = getattr(self, "_press_session", None)
                if callable(session_of):
                    session_of().observe(Painted(media_time=media_time))
            if hasattr(pane, "set_progress"):
                pane.set_progress(
                    rendered=self.currentFrame,
                    total=self.max_value,
                    playhead=self._result_playhead(),
                    duration=self._clip_duration(),
                )
        else:
            image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                           QImage.Format.Format_RGB888).copy()
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

    def _clip_duration(self, owner=None) -> float:
        """The clip's length in seconds (`preview.facts.clip_duration` is the one derivation; the page names inputs).

        The decoder holds the opened file's length - it is what its `opened` signal copied into `_preview_duration` - so
        a decoder that can answer is passed as-is, and that same opened length stands in when there is none. While the
        pane's owner is `QUEUE_FILE` the queue item's own length is the ONE axis (its bar, band, reach and thumb are
        shares of it); every other owner passes 0.0 and keeps the preview/export numbers exactly.

        `owner` is the snapshot's one decision when the caller already has it (`_push_preview_state`); a caller without
        one gets a fresh read of the same pure function.
        """
        from types import SimpleNamespace

        from .preview.facts import clip_duration

        owner = owner_of(self) if owner is None else owner
        if owner.kind is PreviewBackendKind.QUEUE_FILE:
            # The queue item's own length IS the axis - including 0.0 while its field is empty. Falling back to the
            # render's clip here is the defect this kind removes: the bar drew the render's duration over the queue
            # item's picture.
            return clip_duration(None, None, queue_duration=float(getattr(self, "_queue_pane_duration", 0.0) or 0.0))
        decoder = getattr(self, "_preview_decoder", None)
        if decoder is None or float(getattr(decoder, "duration", 0.0) or 0.0) <= 0.0:
            decoder = SimpleNamespace(duration=float(getattr(self, "_preview_duration", 0.0) or 0.0))
        return clip_duration(getattr(self, "currentRenderOptions", None), decoder)

    #: A press whose frame never lands (a decode that died, a target the tail cannot reach) must still close, or the
    #: thumb stands on the requested time forever. The deadline itself lives in `preview.session` (ONE definition,
    #: plan P4); this name is the page's alias for the reader who looks for the deadline on the tab.
    SEEK_WATCH_TIMEOUT_S = SESSION_SEEK_WATCH_TIMEOUT_S

    def _result_playhead(self, owner=None) -> float:
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
            return produced * self._clip_duration(owner)
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
        if owner_of(self).kind is PreviewBackendKind.QUEUE_FILE:
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
        # Show what the pool already has for this playhead - a PAUSED pane has no other clock, and a press must land
        # while nothing else paints - then top the lead up. Both are cheap: the show is a lookup, the fill is at most
        # one decode, and an ask the decoder never answers is retried here (`_source_pool_newest` still lacks the frame).
        self._source_show(self._compare_playhead())
        self._source_fill(self._compare_supply_playhead())

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
            owner = owner_of(self)
            if (self._preview_enabled() and self._followable(renderOptions)
                    and (bool(getattr(renderOptions, "isPreview", False)) or not owner.may_seek_any)):
                # A PREVIEW follows its own file: the tail decodes what the renderer has written so far, which is the
                # pane's only painter there. An EXPORT whose pane the display block feeds does NOT start one (measured
                # 2026-11-24: two painters, one surface); one whose block could not be allocated keeps the tail as its
                # fallback, which is what stopped the pane going black on a plain render.
                self._start_live_tail(renderOptions.outputPath)
            else:
                # One picture, one painter: this job's pane is fed by the display block (or by nothing), and a tail
                # left over from the previous job in the same pass would paint on top of it. The previous job's press
                # cannot outlive the hand-over either (plan P4).
                self._reset_press_for_job()
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

        The job's press is closed here too (plan P4): nothing that has not answered by the end of its job can answer
        after it, and the marker must fall back to the picture rather than the request.
        """
        self._reset_press_for_job()
        self._stop_live_tail()
        self._close_display_state()

    def _start_live_tail(self, path: str) -> None:
        """Ask for the live tail; the pane's own GUI tick creates it.

        This is the live preview: the renderer pays nothing for it (it writes the file anyway), the pane decodes what
        has been produced so far, and the strip's render head tracks the produced range.

        The call is a JOB change (or the user pressing Play, which supersedes a press as well): an open ticket from the
        previous picture is closed here, so `_result_playhead` can never answer with the old job's `target_s`.

        It is called from the render's worker (`threading.Thread`, ProcessTab.py:429) - a thread with no Qt event
        loop. A `QTimer` built here never fires, and a signal connected here is queued to a thread that never runs
        its queue: both mistakes were live at once (measured 2026-11-23: the pane showed the 640x360 thumbnail for a
        whole 12 s preview and a strip press during the render did nothing, while the seek wiring and the timer looked
        present in the code). So this only records the request; `_ensure_live_tail` runs on the GUI thread from
        `updateProcessTab`, which is the pane's 100 ms clock.
        """
        self._reset_press_for_job()
        self._stop_live_tail()
        self._live_tail_path = str(path)
        self._live_tail_wanted = True

    def _refresh_held_progress(self) -> None:
        """While the preview is paused, keep the strip's PRODUCED band honest without touching the playhead.

        The pane's `rendered` is a snapshot taken when a frame was painted, so a paused preview froze the band at
        whatever the renderer had produced at the moment of the pause - and a seek could not reach what had been
        produced since. The playhead must not move: only the two produced counters are refreshed.

        An EXPORT's pane has no tail (its painter is the display feed, see the owner's `kind`), so "a tail is running"
        is not the condition: the condition is that the feed owns the pane and left a progress snapshot to refresh. A
        QUEUE item owns its own readout (`QUEUE_FILE`): the render's counters must not repaint over it.
        """
        if not getattr(self, "_preview_paused", False):
            return
        owner = owner_of(self)
        if owner.kind is PreviewBackendKind.QUEUE_FILE:
            return
        if getattr(self, "_live_tail", None) is None and owner.kind is not PreviewBackendKind.DISPLAY_FEED:
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

    def _publish_pane_progress(self, duration: float, produced: float, playhead: float) -> None:
        """Feed the pane's own (hidden) strip from the SNAPSHOT's numbers, never from a queue-specific derivation.

        The visible bar reads `PreviewState`; this only keeps the pane's painted strip - hidden while the control bars
        are installed, but still the widget a legacy A/B shows - on the same numbers, without a second call site that
        could publish a different duration. `set_progress` repaints unconditionally (it is the strip's only
        notification), so an unchanged quartet is not published at all: that guard is what keeps a held pane from
        repainting ten times a second.
        """
        pane = getattr(getattr(self, "page_handles", None), "proc", None)
        pane = getattr(pane, "previewLabel", None)
        setter = getattr(pane, "set_progress", None)
        if not callable(setter):
            return
        values = (float(produced), float(duration), float(playhead), float(duration))
        previous = getattr(pane, "_progress", None)
        try:
            if previous is not None and tuple(float(part) for part in previous) == values:
                return
        except (TypeError, ValueError):
            pass
        try:
            setter(rendered=values[0], total=values[1], playhead=values[2], duration=values[3])
        except Exception:
            pass

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
            # YUV420 planes, not RGB24: the pane uploads three plane textures and the shader converts. The 10-bit
            # arm is the preview file's own format; `pixel_format="yuv420p"` is the 8-bit fallback if R16 sampling
            # is unusable on the host's RHI backend.
            tail = LiveTailThread(path, paused=self._preview_paused, target_size=box,
                                  pixel_format="yuv420p10le")
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
        queue_owns = owner_of(self).kind is PreviewBackendKind.QUEUE_FILE
        if holding and getattr(self, "_playback_tail", False):
            # The tail this PLAY created exists only while the transport plays; pausing drops it and the feed's cheap
            # path (the renderer's own composited frame) takes the pane back. A PREVIEW's tail and a tail somebody else
            # owns (the queue tab's hand-back) are NOT ours to stop - they are the picture there.
            self._stop_live_tail()
        tail = getattr(self, "_live_tail", None)
        decoder = getattr(self, "_preview_decoder", None)
        if not holding and decoder is not None and getattr(self, "_playback_tail", False):
            # A finished file has a real player (media-rate pacing, one frame per tick, box-capped). The playback tail
            # polls the file every 120 ms and hands a backlog burst to the GUI, which paints once per burst - measured
            # 2026-11-25: 18 frames/s delivered, 9 painted, the drawn index stepping by 2. The tail is the player for a
            # file still being WRITTEN; once the job is done its decoder owns the timeline.
            self._stop_live_tail()
            tail = None
        if (not holding and tail is None and not queue_owns and not bool(getattr(self, "isPreview", False))
                and decoder is None):
            # PLAY, export, no tail and no finished-file decoder: the pane was following the renderer's newest frames,
            # which is not playback. The output file is the timeline; start the tail here (the GUI tick builds it) at
            # the pane's own position, and the feed stands down while it paints. Pausing drops it again, so the feed's
            # cheap path is what a held pane uses (owner, 2026-11-25: "воспроизведение всё ещё просто переключает нас
            # на слежение за последними апкад кадрами").
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
            self._push_preview_state()
            return
        from PySide6.QtCore import QThread, QTimer

        if QThread.currentThread() != button.thread():
            # `_stop_live_tail` runs in the render's worker thread, which has no event loop to paint from: hand the
            # widget work to the button's own thread instead of writing to a widget cross-thread.
            #
            # This branch is NOT about the tick: the 100 ms tick already arrives on the GUI thread (its plain-callable
            # connection is queued there - see `startGUIUpdate`, plan P6), so `updateProcessTab` never takes it. It is
            # for the other callers of this method: `renderToPipeThread` starts the job's tail from a plain `Thread`
            # (`_start_live_tail` -> `_stop_live_tail` -> here, ProcessTab.py:781/1432), and a widget write from there
            # is a cross-thread write whatever the tick does. The three-argument `singleShot` is the sanctioned form:
            # `button` is the CONTEXT object, so the retry lands on the button's thread and not on the caller's.
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
        # The bars read the same truth the legacy button does: one push per change, no second timer.
        self._push_preview_state()

    # -- the control surface (plan-rve-preview-controls-frame-path-2026-11-25) -------------------------

    def _install_preview_controls(self) -> None:
        """Build the controller, the backend adapters and the two bars over the pane.

        The old painted strip and the page's one pause button stand down here: the strip is hidden with the pane's
        PUBLIC `set_strip_visible(False)` (its signals stay wired, the transport is the new one), and the button is
        hidden rather than removed because the page owns it. The page keeps its own LMB compare/pan: the bars are
        children of the pane and only the controls take the mouse.
        """
        pane = getattr(getattr(self.page_handles, "proc", None), "previewLabel", None)
        if pane is None or not hasattr(pane, "set_view_mode"):
            return
        from .preview.controller import PreviewController

        try:
            pane.set_strip_visible(False)
        except Exception as exc:
            log(f"preview: the pane's painted strip could not be hidden ({exc})")
        legacy = getattr(self.page_handles.proc, "previewPauseButton", None)
        if legacy is not None:
            # The new transport replaces it. Its `toggled` connection stays (unused while hidden) so `_sync_preview_transport`
            # keeps painting a widget somebody may still show for an A/B.
            legacy.setVisible(False)
        controller = PreviewController(self)
        controller.set_logger(log)
        self._preview_surface = _PreviewSurfacePort(self)
        controller.set_surface(self._preview_surface)
        self._preview_controller = controller
        # The debug layer's dump reads through THIS page (plan P5): point it now, so a pane that has never pressed a
        # strip still answers `state:preview` with its state numbers, its paint and its owner.
        self._attach_dump_source(controller.session())
        # The pane's wheel moves the zoom without the controller: mirror it so the Fit / 1:1 buttons stay honest.
        zoom_changed = getattr(pane, "zoomChanged", None)
        if zoom_changed is not None and hasattr(zoom_changed, "connect"):
            zoom_changed.connect(self._on_preview_zoom_changed)
        try:
            from .preview.controls import PreviewControls

            controls = PreviewControls(pane, pane=pane)
        except Exception as exc:
            log(f"preview: the control bars could not be built ({exc})")
            self._preview_controller = None
            return
        controls.commandRequested.connect(controller.dispatch)
        controller.stateChanged.connect(controls.set_state)
        controller.stateChanged.connect(self._on_preview_state_changed)
        controller.status.connect(self._on_preview_status)
        self._preview_controls = controls
        self._push_preview_state()

    def _on_preview_state_changed(self, state) -> None:
        """Mirror the state's own view fields so the next fact push cannot overwrite them with stale page values."""
        self._preview_zoom = float(getattr(state, "zoom", 1.0) or 1.0)
        self._preview_zoom_fit = bool(getattr(state, "zoom_fit", True))

    def _on_preview_status(self, line: str) -> None:
        """One controller line (a refused command, a log record) - always visible in log.txt."""
        log(str(line))

    def _on_preview_zoom_changed(self, zoom: float) -> None:
        """The pane's own wheel zoom: the state follows the picture, the controller re-derives Fit vs 1:1."""
        try:
            self._preview_zoom = float(zoom)
        except (TypeError, ValueError):
            return
        self._preview_zoom_fit = False
        controller = getattr(self, "_preview_controller", None)
        if controller is not None:
            controller.observe(zoom=self._preview_zoom, zoom_fit=False)

    def _set_preview_paused(self, paused: bool) -> None:
        """The controller's play/pause entry: the shipped handler, which selects and moves the pane's owner.

        Kept as a method because it is the page's public edge for the backend adapters (they never re-implement the
        tail/decoder choice).
        """
        self._on_preview_pause_toggled(bool(paused))

    def _preview_seek_ceiling(self, owner, duration: float) -> float | None:
        """The furthest content time the transport may ask for, in seconds (`preview.facts.reach_seconds`).

        A finished file - the queue item and the preview's own file alike - and a feed that renders any frame on demand
        reach the whole clip; a growing file's tail only has what the renderer has written, so it reaches the produced
        edge. `owner.may_seek_any` already carries the queue's own rule (a finished item is fully reachable and the
        feed stages nothing there), so no second read of the ownership flag is needed here.
        """
        from .preview.facts import reach_seconds

        produced = self._produced_for_band(owner.kind, duration, fraction=self._live_produced_fraction(),
                                           rendering=True)
        return reach_seconds(owner.kind, duration=duration, produced=produced,
                             any_frame=owner.may_seek_any)

    def _produced_for_band(self, kind, duration: float, *, fraction: float, rendering: bool) -> float:
        """What the strip's BLUE means: seconds the upscaler has produced (`preview.facts` is the one derivation).

        A finished file is fully produced. Everything else is a live render and the band is its progress - including an
        EXPORT whose pane follows the display feed: that pane used to report the whole clip as produced, so a render at
        14 % showed the whole bar blue and the grey "not upscaled yet" half never appeared (owner, 2026-11-26,
        screenshot of an export at frame 67/493 with a fully blue strip). The feed can seek anywhere on demand, but that
        is the CEILING (`_preview_seek_ceiling`), not the produced share.
        """
        from .preview.facts import produced_seconds

        return produced_seconds(kind, fraction=fraction, duration=duration, rendering=rendering)

    def _push_preview_state(self) -> None:
        """Feed the controller the facts it cannot see: the pane's numbers, its owners, the probe's frame counts.

        ONE owner per snapshot: `preview.owner.owner_of` is read once here, at the top, and every decision this push
        makes (the backend kind, the produced band, the reach) reads that one object. The frame branches that run on
        their own signals build their own read, at their own frame - a cached owner would go stale exactly where a
        one-frame loan lives.
        """
        controller = getattr(self, "_preview_controller", None)
        pane = getattr(getattr(self, "page_handles", None), "proc", None)
        pane = getattr(pane, "previewLabel", None)
        if pane is None:
            return
        owner = owner_of(self)
        kind = owner.kind
        duration = float(self._clip_duration(owner) or 0.0)
        produced_fraction = self._live_produced_fraction()
        process = getattr(self, "renderProcess", None)
        rendering = process is not None and process.poll() is None
        produced = self._produced_for_band(kind, duration, fraction=produced_fraction, rendering=rendering)
        queue_owns = kind is PreviewBackendKind.QUEUE_FILE
        if queue_owns:
            # The queue item's own file is the axis: its playhead, its length - never the render's numbers behind it.
            playhead = self._queue_pane_playhead()
        else:
            try:
                playhead = float(self._result_playhead(owner))
            except Exception:
                playhead = float(getattr(self, "_preview_playhead", 0.0) or 0.0)
        try:
            shown = pane.shown_time() if callable(getattr(pane, "shown_time", None)) else None
        except Exception:
            shown = None
        if shown is not None:
            played = float(shown)
        else:
            played = playhead if queue_owns else float(getattr(self, "_preview_playhead", 0.0) or 0.0)
        if queue_owns:
            # The pane's own (hidden) strip is fed from THIS snapshot, not from a queue-specific `set_progress`: one
            # derivation per number, and the preview's `_progress` is never overwritten by the queue's numbers.
            self._publish_pane_progress(duration, produced, playhead)
        if controller is None:
            return
        from .preview.backends import backend_for

        options = getattr(self, "currentRenderOptions", None)
        source_frames = int(getattr(options, "videoFrameCount", 0) or 0)
        factor = max(1, int(getattr(options, "interpolateTimes", 1) or 1))
        output_frames = int(getattr(self, "max_value", 0) or 0) or source_frames * factor
        frames_done = int(round(float(getattr(self, "currentFrame", 0) or 0) / factor))
        if source_frames > 0:
            frames_done = min(frames_done, source_frames)
        fps = 0.0
        try:
            fps = float(getattr(options, "videoFps", 0.0) or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        if fps <= 0.0:
            try:
                fps = float(getattr(self, "fps", 0) or 0)
            except (TypeError, ValueError):
                fps = 0.0
        owners = (getattr(self, "_live_tail", None) is not None
                  or getattr(self, "_preview_decoder", None) is not None
                  or getattr(self, "_queue_pane_decoder", None) is not None)
        try:
            has_frame = bool(pane.has_frame()) if callable(getattr(pane, "has_frame", None)) else False
        except Exception:
            has_frame = False
        has_picture = bool(has_frame or owners or rendering)
        playing = bool(not self._preview_paused and has_picture)
        try:
            mode = str(pane.view_mode()) if callable(getattr(pane, "view_mode", None)) else "single"
        except Exception:
            mode = "single"
        zoom = float(getattr(self, "_preview_zoom", 1.0) or 1.0)
        pane_zoom = getattr(pane, "zoom", None)
        if callable(pane_zoom):
            try:
                # The pane's APPLIED factor (the fit scale at open) is what the zoom buttons step from: stepping from
                # a hard-coded 1.0 would jump a fitted 0.35 picture to 1.25 on the first press.
                zoom = float(pane_zoom())
                self._preview_zoom = zoom
            except Exception:
                pass
        tail = getattr(self, "_live_tail", None)
        decoder = getattr(self, "_preview_decoder", None)
        if kind is PreviewBackendKind.QUEUE_FILE:
            # A queue item is its OWN clock (P3's note, closed in P5): the adapter the session binds must be the
            # queue's, or a press there is judged against the render's position behind the tab.
            decoder = getattr(self, "_queue_pane_decoder", None)
        controller.register_backend(
            kind, backend_for(self, kind, tail=tail, decoder=decoder, host_playable=bool(has_picture)))
        controller.set_backend(kind)
        # Why the fill and the thumb can differ, in words the report can print: an open press is the only case where the
        # thumb is a REQUEST and the fill is the picture. Without this the report could only say "they differ", which is
        # what a screenshot already says.
        ticket = getattr(self, "_press", None)
        if ticket is not None and ticket.open:
            fill_source = "press-open"
        elif shown is not None:
            fill_source = "shown"
        else:
            fill_source = "playhead"
        controller.observe(
            is_playing=playing,
            has_picture=has_picture,
            playhead=max(0.0, playhead),
            played=max(0.0, played),
            fill_source=fill_source,
            produced=max(0.0, produced),
            duration=duration,
            source_frames=source_frames,
            frames_done=frames_done,
            output_frames=output_frames,
            fps=fps,
            seek_ceiling=self._preview_seek_ceiling(owner, duration),
            mode=mode,
            zoom=zoom,
            zoom_fit=bool(getattr(self, "_preview_zoom_fit", True)),
        )
        # The state is now this tick's truth, and it is the ONE reading the per-second line prints (plan P5): the same
        # numbers the bars paint, so the file cannot describe a pane the user is not looking at.
        state_now = getattr(controller, "state", None)
        state_now = state_now() if callable(state_now) else None
        if state_now is not None:
            self._note_state_snapshot(state_now)

    def _note_state_snapshot(self, state) -> None:
        """One aggregate `snapshot` line per second (plan P5): the dump's keys, read from the state the bars paint.

        The file used to have no per-second reading of the STATE at all - the scrubber's own lines are events (a press,
        a refusal) and every derived number lived only in a widget. This line names the owner, the clip's length, what
        is produced, where the picture is and whether a press ticket is open, so "the band says 20.6 s while the picture
        stands at 8.9 s" is a number in the log instead of an argument over a screenshot. Events are untouched: they are
        the owner's evidence, and telemetry writes them at level 0.
        """
        session = self._preview_session_existing()
        ticket = getattr(session, "ticket", None) if session is not None else None
        epoch = getattr(session, "epoch", None) if session is not None else None
        telemetry.note_snapshot(**{
            "kind": state.backend.value,
            "duration": float(state.duration),
            "produced": float(state.produced),
            "played": None if state.played is None else float(state.played),
            "playhead": float(state.playhead),
            "finished": bool(state.finished()),
            "ticket.open": bool(getattr(ticket, "open", False)) if ticket is not None else False,
            "epoch": None if epoch is None else int(epoch),
        })

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
            decoder = VideoDecoder(pixel_format="yuv420p10le", content_time=True,
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
        queued frames are dropped the same way. Painting decisions read the owner's `QUEUE_FILE` kind now; the raw
        flag stays for the queue's own bookkeeping (entering/leaving, row selection) and for the frame gate below.
        """
        return bool(getattr(self, "_queue_pane_owns", False))

    def _queue_pane_playhead(self) -> float:
        """Where the QUEUE ITEM's picture stands, in ITS file's time - the playhead its bar must show.

        The picture's own name when the pane kept it (`shown_time`: `_on_queue_pane_frame` hands the decoder's PTS),
        else the position the frame path remembered. This never reads `_preview_playhead`: that is the RENDER's
        position, and the hand-back must still find the render standing where it stood.
        """
        pane = getattr(getattr(self, "page_handles", None), "proc", None)
        shown_time = getattr(getattr(pane, "previewLabel", None), "shown_time", None)
        if callable(shown_time):
            try:
                named = shown_time()
            except Exception:
                named = None
            if named is not None:
                return max(0.0, float(named))
        return max(0.0, float(getattr(self, "_queue_pane_position", 0.0) or 0.0))

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
        self._reset_press_for_job()
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
        self._reset_press_for_job()
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
            self._reset_press_for_job()
            self._clear_pane()
            return
        if path != getattr(self, "_queue_pane_path", None):
            # Another item on the pane is another job: a press that was open on the previous file cannot answer here.
            self._reset_press_for_job()
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
        The raw ownership flag is the right read HERE and only here: this is a FRAME gate (does the picture that just
        arrived still have a painter), while every decision about the pane reads the owner's `QUEUE_FILE` kind.
        """
        if not self._queue_owns_pane():
            return
        if frame is None:
            return
        from PySide6.QtGui import QImage

        path_started = time.perf_counter()
        # ONE copy, not two: `bytes()` already owns a private buffer, so the QImage is built over it and only the CROP
        # (or the whole frame when there is no crop) is copied. The first cut copied the full 1440p frame and then
        # copied the crop out of it - 12 MB of memcpy per source frame at 24 fps for nothing.
        raw = bytes(frame.buffer)
        image = QImage(raw, frame.width, frame.height, frame.stride, QImage.Format.Format_RGB888)
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
        # A press on the queue item is answered by the frame its decoder publishes: the session closes the ticket on
        # the picture, the same closure the tail and the finished file use (plan P4).
        session_of = getattr(self, "_press_session", None)
        if callable(session_of):
            from .preview.session import Painted

            try:
                session_of().observe(Painted(media_time=pts, index=getattr(frame, "index", None)))
            except Exception:
                pass
        # The queue's numbers reach the bar through the SNAPSHOT (one derivation: its own length is the axis and a
        # finished item is fully produced). The pane's hidden strip is fed from that same snapshot inside
        # `_push_preview_state`, so there is no queue-specific `set_progress` left to overwrite the preview's
        # `_progress` on the pane the next owner paints.
        self._push_preview_state()

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

    def _preview_session(self):
        """The pane's press session: the controller's one, or a page-local one for a tab wired by hand.

        The page does not create press records itself any more (plan P1): the session owns the ticket and the epoch.
        A tab built without a controller (the harness's `object.__new__` style, the pane's own legacy strip signal)
        still gets a session, so the routing below has one code path. Every access re-points the session's `dump()`
        at THIS page's read (plan P5), so the debug channel never describes a pane with another page's facts.
        """
        session = self._preview_session_existing()
        if session is None:
            from .preview.session import PreviewSession

            session = PreviewSession()
            self._preview_session_local = session
        self._attach_dump_source(session)
        return session

    def _preview_session_existing(self):
        """The pane's session if one ALREADY exists, else None - a read that never creates one.

        The debug dump and the aggregate log line must be able to say "no session" instead of conjuring one: reading
        the pane is not the same act as pressing it.
        """
        controller = getattr(self, "_preview_controller", None)
        getter = getattr(controller, "session", None)
        session = getter() if callable(getter) else None
        if session is not None:
            return session
        return getattr(self, "_preview_session_local", None)

    def _attach_dump_source(self, session) -> None:
        """Point `session.dump()` at this page's one read (plan P5); idempotent, and the dump itself writes nothing."""
        setter = getattr(session, "set_dump_source", None)
        if callable(setter):
            setter(self._preview_dump_facts)

    def _preview_dump_facts(self) -> dict:
        """The page's half of `PreviewSession.dump()` (plan P5): ONE read of what the session cannot see.

        Reads the one owner decision, the controller's state and the scrubber's own last paint, and writes nothing: no
        widget is asked to paint, no ticket is moved, no state is pushed. The numbers are the state's - the same object
        the bars are painted from - so the dump cannot disagree with what the user sees.
        """
        from .preview.owner import owner_of

        owner = owner_of(self)
        controller = getattr(self, "_preview_controller", None)
        getter = getattr(controller, "state", None)
        state = getter() if callable(getter) else None
        return {
            "kind": owner.kind.value,
            "duration": float(getattr(state, "duration", 0.0) or 0.0),
            "produced": float(getattr(state, "produced", 0.0) or 0.0),
            "played": None if getattr(state, "played", None) is None else float(state.played),
            "playhead": float(getattr(state, "playhead", 0.0) or 0.0),
            "finished": bool(state.finished()) if state is not None else None,
            "paint": self._preview_paint_record(),
            "owner": {
                "kind": owner.kind.value,
                "may_paint": bool(owner.may_paint),
                "may_want": bool(owner.may_want),
                "may_seek_any": bool(owner.may_seek_any),
                "has_clock": owner.clock is not None,
                "has_surface": owner.surface is not None,
            },
        }

    def _preview_paint_record(self):
        """What the scrubber last painted (`scrub_report()["paint"]`), or None when this pane has no scrubber."""
        timeline = getattr(getattr(self, "_preview_controls", None), "timeline", None)
        report = getattr(timeline, "scrub_report", None)
        if not callable(report):
            return None
        try:
            return report().get("paint")
        except Exception:
            return None

    def _press_session(self):
        """The pane's session with THIS page's landing-line formatter attached (plan P4).

        The session owns the closure (a painted answer, the deadline, a job change); the page only formats the one line
        the owner reads. Idempotent: `set_settle` is a plain assignment, so every access re-points it at the same bound
        method of this page.
        """
        session = self._preview_session()
        setter = getattr(session, "set_settle", None)
        if callable(setter):
            setter(self._on_preview_settled)
        return session

    def _on_preview_settled(self, settle) -> None:
        """Format the owner's landing line for a ticket the SESSION closed - the page decides nothing here.

        `asked` / `landed` / the turnaround come from the session (the picture's own time and the moment the ticket
        ended); the text below is the same line, with the same numbers, the owner has read since 2026-11-24. A deadline
        that closed the press says so with the same `NOT the asked frame (gave up)` tail.
        """
        asked = getattr(settle, "asked", None)
        if asked is None:
            return
        landed = getattr(settle, "landed", None)
        if landed is None:
            landed = float(getattr(self, "_preview_playhead", 0.0) or 0.0)
        asked = float(asked)
        landed = float(landed)
        turnaround_ms = getattr(settle, "turnaround_ms", None)
        timing = "" if turnaround_ms is None else f", {float(turnaround_ms):.0f} ms"
        verdict = "" if bool(getattr(settle, "matched", False)) else ", NOT the asked frame (gave up)"
        log(f"seek: press wanted {asked:.3f}s, the picture landed at {landed:.3f}s "
            f"(delta {landed - asked:+.3f}s{timing}{verdict})")
        # The press is answered (or gave up): the next picture is an ordinary one, not the answer to this press.
        self._seek_watch = None
        self._seek_pressed_at = None
        self._seek_generation = None

    def _tick_press_deadline(self) -> None:
        """Report the tick's monotonically increasing time to the press session (the 100 ms clock, plan P4).

        The deadline lives in the session; this is the page's only job in it - tell it that time passed, so a press
        whose frame never lands cannot hold the marker on the requested time forever.

        It runs on the tick's clock, which is the GUI thread (`updateProcessTab` is queued there - measured, see
        `startGUIUpdate` and `test_preview_state_tick.py::test_the_tick_body_runs_on_the_gui_thread`), so the session's
        deadline and the playhead it releases are read and written on one thread.
        """
        session_of = getattr(self, "_press_session", None)
        if not callable(session_of):
            return
        try:
            session = session_of()
        except Exception:
            return
        check = getattr(session, "check_deadline", None)
        if callable(check):
            try:
                check()
            except Exception:
                pass

    def _reset_press_for_job(self) -> None:
        """Close the pane's open ticket because the JOB changed: a press cannot outlive it.

        A job change is a new tail (`_start_live_tail`), the end of the painters (`_end_job_painters`) or the queue
        hand-over (entering/leaving the queue tab, selecting another item). The session closes its own ticket on the
        `JobChanged` event; the page drops its reference so `_result_playhead` can never answer with the old job's
        `target_s` - the marker falls back to the picture (`shown_time`) immediately.
        """
        session_of = getattr(self, "_press_session", None)
        if callable(session_of):
            from .preview.session import JobChanged

            try:
                session_of().observe(JobChanged())
            except Exception:
                pass
        self._press = None
        self._seek_watch = None
        self._seek_pressed_at = None
        self._seek_generation = None

    def _on_preview_seek(self, fraction: float) -> None:
        """A press or drag on the strip seeks whoever owns the picture - the ROUTE is all that is decided here.

        Live, that is the tail: it repositions inside what the renderer has already written, and the next poll
        (120 ms) shows the target. A press PAST the produced band is different only while the pane's block is usable:
        the tail has nothing there, but the renderer can render that one frame on demand - so the press becomes a
        newest-wins request (`_request_display_seek`) and the pane holds it until play hands it back. A finished file
        seeks its own decoder. The strip reports a share of the WHOLE timeline, so while the timeline length is
        unknown the produced share stands in for it - which is the same assumption the strip itself makes when it
        fills its white band.

        The REACH is not checked here: `PreviewSession.seek_to` has already refused a target past it before this
        method is called (plan P1 - the clamp lives in one place). The press record is the session's too: this method
        asks for the ticket for the press it is routing and fills in the route facts only the page knows.
        """
        fraction = float(fraction)
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
        owner = owner_of(self)
        duration_s = float(self._clip_duration(owner) or 0.0)
        if duration_s > 0.0:
            target_s = fraction * duration_s
        self._seek_target_s = target_s
        telemetry.event("seek press", fraction=f"{fraction:.4f}", produced=f"{produced:.4f}",
                        target_s=None if target_s is None else f"{target_s:.3f}",
                        target_frame=target_frame)
        # One record per press, created BEFORE a route is chosen: the intent, the route it took, whether that route
        # can answer it, and (from whichever painter answers) the picture. Without this a press could end in silence -
        # the owner's three sessions pressed four times each and left no line about the result at all (2026-11-24).
        # The record lives in the pane's press session now (plan P1): a press that came through the controller already
        # owns one, and `press()` hands that back instead of opening a second; a direct press opens its own.
        edge_s = produced * duration_s if duration_s > 0.0 else None
        ticket = self._press_session().press(
            target_s=target_s, target_frame=target_frame, produced=produced, edge_s=edge_s)
        self._press = ticket
        if owner.kind is PreviewBackendKind.QUEUE_FILE:
            decoder = getattr(self, "_queue_pane_decoder", None)
            duration = float(getattr(decoder, "duration", 0.0) or 0.0)
            if decoder is not None and duration > 0.0:
                ticket.decide(route="queue", answerable=True)
                # The queue item is a finished file: the frame that lands answers the press, so the session watches
                # the same seconds the decoder is asked for (plan P4).
                self._press_session().watch(fraction * duration)
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
        # Two edges, not one: the band is the RENDERER's progress (`currentFrame / max_value`) and the tail can only
        # serve what the muxer has FLUSHED - a preview at 60% whose file holds 20% refused every press between them
        # inside the tail ("not in the flushed file"), with the strip advertising the whole 60% as reachable (owner's
        # log, 2026-11-26: press at 8.164 s with produced=0.603, no seek record, the thumb stayed at 4.17 s).
        tail_edge = float(getattr(live, "produced_seconds", 0.0) or 0.0) if live is not None else 0.0
        tail_tolerance = 0.5 / max(1.0, float(getattr(self, "fps", 0) or 0.0) or 24.0)
        past_tail = (live is not None and target_s is not None and tail_edge > 0.0
                     and float(target_s) > tail_edge + tail_tolerance)
        if (bool(getattr(self, "isPreview", False))
                and owner.may_seek_any
                and (live is None or fraction > produced + 1e-6 or past_tail)):
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
            # The tail maps a strip share onto ITS timeline (the clip length, or the flushed edge when it is unknown),
            # so the asked time it must be judged against is this one, not the ticket's whole-clip target.
            self._press_session().watch(target)
            live.seek(target)
            ticket.decide(route="tail", answerable=True)
            log(f"seek: tail asked for {target:.3f}s (published edge {edge:.3f}s)")
            return
        decoder = getattr(self, "_preview_decoder", None)
        duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
        if decoder is not None and duration > 0:
            ticket.decide(route="decoder", answerable=True)
            # The same landing instrument as the tail's: the session watches the time the decoder was asked for, and
            # the frame it publishes closes the press - not the page's timeout.
            self._press_session().watch(fraction * duration)
            decoder.seek(fraction * duration)
            log(f"seek: decoder asked for {fraction * duration:.3f}s")
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

    def _on_original_requested(self, seconds: float, *, span: bool = False) -> None:
        """Decode ONE source frame at the playhead for the LMB compare (the input file, never the renderer).

        The image is what the platform's decoder produces from the input, which is the honest comparison: the
        renderer's own source buffer would need a second channel and would make the render pay for a preview-only
        action. It is then cropped to the box the render reported (`BORDERS`, parsed out of this job's output) - see
        `_on_original_frame`.
        """
        if not getattr(self, "_compare_from_tick", False):
            pane_now = self.page_handles.proc.previewLabel
            mode_now = getattr(pane_now, "view_mode", None)
            comparing = False
            if callable(mode_now):
                try:
                    comparing = mode_now() != "single"
                except Exception:
                    comparing = False
            if comparing:
                # The pane's clock moved (a compare mode): the pool is the one that decides WHAT to decode, so this goes
                # through the fill instead of asking for this one timestamp. A direct ask overwrote the fill's span (the
                # decoder keeps only the last request), which left the pool permanently a frame behind - measured
                # 2026-11-26: the pane asked 14.139 / 14.181 / 14.223 while the pool's newest stayed at 0.292.
                playhead_now = self._compare_playhead()
                self._source_show(playhead_now)
                self._source_fill(playhead_now)
                return
        if self._queue_owns_pane():
            # A queue item IS the source: there is no render to compare it against, and decoding the last render's
            # input file over it would show a picture of a different video. A FRAME gate (which picture the compare
            # would decode), hence the raw flag; pane decisions read the owner's `QUEUE_FILE` kind.
            return
        # The pane's own `originalRequested` (a compare mode's playhead moved) and the 100 ms tick both arrive here.
        # The one-decode-per-playhead marker is recorded when a frame ARRIVES (`_on_original_frame`), not here: an ask
        # that never lands (the decoder still opening, a dropped request) used to leave the marker on the asked time, so
        # the tick saw "already asked for this playhead" and never retried - the half stayed empty for as long as the
        # pane was paused (measured 2026-11-26: `compare: asked for the source at 0.125s ... mode split` with no
        # "source frame" line after it, and no divider, because the divider is drawn only when the source exists).
        self._compare_watch = float(seconds)
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
                from PySide6.QtCore import QThread

                from .video.decode import VideoDecoder

                # ITS OWN THREAD, which `decode.py` is written for ("the caller moves it to a QThread") and nothing did:
                # on the GUI thread a 1440p source frame costs 12.5 ms of decode + a 6.2 MB pack, and the paint queue
                # waits behind it. Measured 2026-11-26: split playback ran at 0.20x wall with both decoders on the GUI
                # thread, while the same two decoders in their own event loops ran at 0.94x - the decode was never the
                # limit, the thread was. The lead below only has capacity once this decode is off the paint thread.
                # AT THE PANE'S BOX, like the result's own decoder (`_start_preview_playback`): a 1440p source handed
                # over whole costs a 6.2 MB copy, a GUI-thread scale down to the result's size, a 6.2 MB upload and a
                # 1440p mip chain per frame - to show 504x378. The fit happens in swscale inside the decode thread, and
                # the two halves then arrive at the SAME size, so the pane scales nothing.
                # CONTENT time, like both of the pane's other painters (the tail subtracts the output's start, the
                # finished-file decoder does the same): the compare names its halves by timestamp and, before this, the
                # source's own file times sat a fraction of a frame off the tail's derived ones, so the pair gate saw
                # two moments for stretches and painted on its timeout (measured 2026-11-26: `pair_off` 1-9 of 15-22
                # paints during inference, while the halves were the same picture). The decoder seeks by adding the
                # origin back (`_rewind`), so an ask in content time still lands on the right frame.
                decoder = VideoDecoder(pixel_format="rgb24", max_size=self._pane_display_box(), content_time=True)
                thread = QThread()
                decoder.moveToThread(thread)
                thread.start()
                self._source_thread = thread
                # Through the relay, not straight to `self`: see `_FrameRelay` (this class is not a QObject).
                relay = _FrameRelay(self._on_original_frame)
                self._compare_relay = relay
                decoder.frame_ready.connect(relay.deliver)
                decoder.error.connect(lambda message: log(f"source decoder: {message}"))
                path = getattr(self.currentRenderOptions, "inputFile", "") or ""
                if not decoder.open(str(path)):
                    log(f"source decoder: cannot open {path}")
                    return
                log(f"source decoder: opened {path}")
                self._source_decoder = decoder
            # A step FORWARD from the frame already on screen is one decode (12.5 ms measured on this 1440p source, an
            # 80 fps ceiling); a jump pays a keyframe rewind (median 176 ms, p95 321). The compare asks once per playhead
            # step, and at the rewind price the pair ran at a quarter of real time with the halves arriving apart
            # (owner, 2026-11-26: "прям видно как сначала один рисуется, а затем другой"). Forward motion is the one
            # case that can be cheap, so only it skips the seek; a scrub backwards still rewinds.
            landed = getattr(self, "_second_surface_pts", None)
            step_forward = (landed is not None
                            and 0.0 <= float(seconds) - float(landed) <= 0.6)
            # EXACT only for a press: its target is a frame's own name, and "the frame before it" is the desync the
            # exactness rule was written for. The POOL's asks walk a live grid (the tail's content times are derived, not
            # the source's stored PTS), and a target a fraction of a frame off the grid made the walk miss, run to the end
            # of the source and disable the decoder for the rest of the session - measured 2026-11-26: `arrivals=57` for
            # two seconds, then `arrivals=0 pool=0` while the asks kept coming. Half a frame of slack picks the same frame.
            exact = (not bool(getattr(self, "_compare_from_tick", False))
                     and not bool(getattr(self, "_shown_name_is_fitted", False)))
            decoder.request_frame(max(0.0, float(seconds)), exact=exact,
                                  seek=not step_forward,
                                  span=bool(span))
        except Exception as exc:
            log(f"original frame failed: {exc}")

    # -- the compare's source half: decoded AHEAD of the playhead, shown in step --------------------

    def _compare_fps(self) -> float:
        """The source's frame rate, with a usable fallback: the lead and the matching slack are measured in frames."""
        try:
            fps = float(self._source_decoder.fps) if getattr(self, "_source_decoder", None) is not None else 0.0
        except Exception:
            fps = 0.0
        return fps if fps > 0.0 else 24.0

    def _compare_lead(self) -> float:
        """How far ahead of the playhead the source half is kept, in seconds (`COMPARE_LEAD_FRAMES`)."""
        return COMPARE_LEAD_FRAMES / self._compare_fps()

    def _source_pool_put(self, pts: float, index, image) -> None:
        """Keep one decoded source frame, bounded: the oldest go first, so a long playback cannot grow the pane."""
        pool = getattr(self, "_source_pool", None)
        if pool is None:
            pool = self._source_pool = {}
        pool[float(pts)] = (index, image)
        while len(pool) > COMPARE_POOL_FRAMES:
            del pool[min(pool)]

    def _source_pool_newest(self) -> float | None:
        pool = getattr(self, "_source_pool", None) or {}
        return max(pool) if pool else None

    def _source_pool_for(self, playhead: float):
        """The decoded frame that BELONGS to `playhead`: the newest one at or before it (half a frame of slack).

        Not the newest decoded one: with a lead, the newest frame is ahead of the picture by construction, and showing
        it would put the source half in the future - the same mismatch this pool exists to remove, mirrored.
        """
        pool = getattr(self, "_source_pool", None) or {}
        if not pool:
            return None
        slack = 0.5 / self._compare_fps()
        best = None
        for pts in pool:
            if pts <= playhead + slack and (best is None or pts > best):
                best = pts
        if best is None:
            return None
        index, image = pool[best]
        return best, index, image

    def _source_prune(self, playhead: float) -> None:
        """Drop only what a long playback could never show again; the COUNT cap in `_source_pool_put` is the real bound.

        A playhead-based horizon looked right and emptied the pool: the decoder runs AHEAD, so a frame it has already
        passed is never emitted again, and dropping it (measured 2026-11-26: `pool=0.292` alone while the playhead was
        at 0.375, four paints in a row) leaves the pair one or two frames apart for ever. History is what a small
        backwards step needs, and it is still bounded.
        """
        pool = getattr(self, "_source_pool", None) or {}
        span = COMPARE_POOL_FRAMES / self._compare_fps()
        # BOTH directions: a seek dose not only leave OLD frames behind, it leaves the whole pool in the FUTURE of the
        # new playhead. Those frames made `_source_pool_newest` report a position eight seconds ahead, so the fill
        # decided the lead was already full and never asked again - the source half froze on the old picture
        # (measured 2026-11-26: `gap +8.049s (+193.2 frames)`, then `frames=0 up=0 mb=0.0` for as long as the pane ran).
        for pts in [value for value in pool if value < playhead - span or value > playhead + span]:
            del pool[pts]

    def _source_show(self, playhead: float) -> None:
        """Hand the pane the source frame that matches `playhead`, and only when it is a DIFFERENT frame.

        One `set_original` per new source frame means one repaint per change instead of one per arrival of either half:
        the two halves are then painted from frames that belong to the same moment.
        """
        pane = self.page_handles.proc.previewLabel
        mode = getattr(pane, "view_mode", None)
        if callable(mode):
            try:
                if mode() == "single":
                    return       # the LMB flash has its own path and its own timing
            except Exception:
                return
        self._source_prune(playhead)
        chosen = self._source_pool_for(playhead)
        if chosen is None:
            return
        pts, index, image = chosen
        stats = getattr(self, "_compare_fill_stats", None)
        if stats is not None:
            stats["arrivals"] += 1
        if telemetry.level() >= 2:
            # The pair's own number: how far the source half is from the result's playhead at the moment it is SHOWN.
            # One line per changed source frame, level 2 only - the compare's "the halves disagree" claim needs it.
            telemetry.event("pair", source=round(pts, 3), playhead=round(float(playhead), 3),
                            delta_frames=round((pts - float(playhead)) * self._compare_fps(), 2),
                            pool="/".join(f"{value:.3f}" for value in
                                          sorted(getattr(self, "_source_pool", None) or {})))
        shown_pts = getattr(self, "_source_shown_pts", None)
        if shown_pts is not None and abs(pts - shown_pts) < 1e-6:
            return
        self._source_shown_pts = pts
        setter = getattr(pane, "set_original", None)
        if callable(setter):
            setter(image, media_time=pts, index=index)

    def _compare_playhead(self) -> float:
        """The time of the RESULT picture on screen - ONE clock for the pair.

        `_result_playhead` is the transport's notion (it follows a press's target before the answer lands), while the
        pane names the picture it actually painted. Two clocks meant two different "now"s: the source was selected
        against one and the result painted at the other, so the pair flipped between a frame and its neighbour - measured
        2026-11-26, 135 source pictures in 4 s against ~96 result frames, a third of them one frame behind. The pane's own
        time wins when it has one; the transport's is the fallback for a pane that names nothing.
        """
        pane = self.page_handles.proc.previewLabel
        shown = getattr(pane, "shown_time", None)
        if callable(shown):
            try:
                value = shown()
            except Exception:
                value = None
            if value is not None:
                return float(value)
        return float(self._result_playhead())

    def _compare_supply_playhead(self) -> float:
        """Where the source pool should be AHEAD of: the picture on screen, not the newest arrival.

        A lag that equals the arrival rate never closes (each result frame moves the target by one frame and the source
        moves by one frame), so the pair gate saw a mismatch for ever and painted only on its timeout. The supply runs off
        the COMMITTED result - a stable clock - while `_compare_playhead` (presentation) stays on the newest one.
        """
        # The NEWEST result: the pool has to hold the frame that is about to be painted, and the only way to have it
        # there in time is to chase the newest one with a lead (see `COMPARE_LEAD_FRAMES`). Chasing the COMMITTED frame
        # instead kept the pool behind the newest result - every paint then showed T+1 against the source's T.
        return self._compare_playhead()

    def _source_paint_pair(self, playhead: float | None = None) -> None:
        """One entry point for "the result has just been painted; give it its other half in the same repaint".

        A minimal host (a harness that builds the tab without `__init__`) reaches the paint paths with nothing to pair,
        which is why the call sites look it up instead of calling it.
        """
        self._source_show(self._compare_playhead() if playhead is None else float(playhead))

    def _source_fill(self, playhead: float) -> None:
        """Top the pool up to `COMPARE_LEAD_FRAMES` ahead of the playhead, one frame per call.

        One frame per call, and the next call comes from the frame's ARRIVAL (`_on_original_frame`): the decoder then
        runs flat out until the lead is full and stops, instead of being polled at a timer's rate - a 10 Hz poll cannot
        feed a 24 fps stream, so the lead would drain exactly when it is needed.
        """
        if self._source_feed_is_live():
            # The renderer is publishing the source of every composited frame: decoding the same file a second time is
            # exactly the cost this block exists to remove.
            return
        playhead = float(playhead)
        interval = 1.0 / self._compare_fps()
        self._source_prune(playhead)
        newest = self._source_pool_newest()
        # ONE ask in flight at a time: the arrival of each frame calls this again, and without the guard a burst would
        # start a second span for the same moment (the decoder keeps only the last request, so the work is wasted).
        pending = getattr(self, "_source_fill_pending", None)
        if pending is not None:
            aged = time.monotonic() - float(pending[1])
            if newest is not None and newest >= float(pending[0]) - 0.5 * interval:
                self._source_fill_pending = None
            elif aged < COMPARE_FILL_WINDOW_S:
                return
            else:
                self._source_fill_pending = None      # a request nothing answered: ask again
        span = True
        if self._source_pool_for(playhead) is None:
            # Nothing in the pool answers for THIS playhead (an empty pool, a seek, a jump back): ask at it, and take the
            # single frame that answers. A seek's answer is one picture, not a burst from wherever the decoder stands.
            target = playhead
            span = False
        elif newest is None:
            target = playhead
            span = False
        elif playhead - newest > 2.0 * interval:
            # The pool is BEHIND the playhead - a scrub, or a SPARSE tail (its lateness rule skips pictures, so two
            # consecutive ones can be six frames apart). One frame per call cannot close a gap the playhead reopens on
            # every picture: measured 2026-11-26, `pool=1` for ever and every pair a timeout. Ask for the far edge of the
            # lead and take the whole walk - the burst IS the point.
            # The burst is decoded FORWARD, not rewound: measured 2026-11-26, forcing a rewind on a gap over 0.3 s made
            # every paint a timeout in ten consecutive seconds - the tail's sparse times are not on the source's frame
            # grid, so the rewind's ceil landed a frame off while the forward walk lands on the frame itself.
            target = playhead + self._compare_lead()
            span = True
        elif newest < playhead + self._compare_lead():
            # Ask for the WHOLE lead in one call and take every frame of the walk: one round trip per lead, not one per
            # frame (a frame-by-frame ask is latency-bound at ~8 frames/s; the decode itself runs at 80).
            target = playhead + self._compare_lead()
        else:
            return
        self._source_fill_pending = (float(target), time.monotonic())
        stats = getattr(self, "_compare_fill_stats", None)
        if stats is not None:
            stats["asks"] += 1
            stats["target"] = round(float(target), 3)
            stats["span"] = int(bool(span))
            stats["newest"] = None if newest is None else round(float(newest), 3)
        self._compare_from_tick = True
        try:
            self._on_original_requested(max(0.0, target), span=span)
        finally:
            self._compare_from_tick = False

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
        if box is not None:
            # The box is written in SOURCE pixels; this decoder hands over a fitted picture, so it is scaled by the same
            # ratio (`stream_size` is the coded size, the one the render's BORDERS line describes).
            source_size = None
            decoder = getattr(self, "_source_decoder", None)
            size_of = getattr(decoder, "stream_size", None)
            if callable(size_of):
                try:
                    source_size = size_of()
                except Exception:
                    source_size = None
            if source_size and source_size[0] > 0 and image.width() != source_size[0]:
                scale = image.width() / float(source_size[0])
                box = tuple(int(round(value * scale)) for value in box)
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
        if region is None:
            # No crop: the QImage still points into `raw`, which dies with this function - own the pixels.
            image = image.copy()
        # What the decoder actually produced, before the crop and the scale: a compare half that ends up black is
        # either an empty decode or a viewer-side problem, and only this line can tell them apart.
        if telemetry.level() >= 2:
            # Three `pixelColor` reads per source frame exist for one line of evidence; at 24 fps they are 72 a second
            # on the GUI thread for a log nobody is reading. Level 2 is the "one line per frame" mode, so ask there.
            _samples = []
            for _fx, _fy in ((1, 1), (image.width() // 2, image.height() // 2),
                             (max(0, image.width() - 2), max(0, image.height() - 2))):
                _c = image.pixelColor(int(_fx), int(_fy))
                _samples.append(f"{_c.red()},{_c.green()},{_c.blue()}")
            telemetry.event("source frame", w=image.width(), h=image.height(), fmt=image.format().name,
                            px="/".join(_samples))
        else:
            telemetry.event("source frame", w=image.width(), h=image.height(), fmt=image.format().name)
        watched = getattr(self, "_compare_watch", None)
        self._compare_watch = None
        pts = float(getattr(frame, "pts", 0.0) or 0.0)
        # The ONE-decode-per-playhead marker: a LANDED frame, so a request the decoder never answered does not silence
        # the retry (`_tick_second_surface` compares the playhead against this).
        self._second_surface_pts = pts
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
        # Into the pool, then: `single` shows it at once (the LMB flash is one picture, nothing to pair); a compare mode
        # shows the frame that belongs to the RESULT's playhead - usually this one, because the pool runs AHEAD of it -
        # and asks for the next frame so the lead stays full.
        self._source_pool_put(pts, index, image)
        if telemetry.level() >= 2:
            telemetry.event("compare put", pts=round(float(pts), 3), pool=len(getattr(self, "_source_pool", None) or {}))
        mode = getattr(pane, "view_mode", None)
        single = True
        if callable(mode):
            try:
                single = mode() == "single"
            except Exception:
                single = True
        if single:
            if hasattr(pane, "set_original"):
                self._source_shown_pts = pts
                pane.set_original(image, media_time=pts, index=index)
        else:
            playhead = self._compare_playhead()
            self._source_show(playhead)
            self._source_fill(self._compare_supply_playhead())

    def _log_source_crop_unknown(self, reason: str) -> None:
        """One line per job when the compare cannot crop: the preview then shows the uncropped source."""
        if getattr(self, "_source_crop_unknown_logged", False):
            return
        self._source_crop_unknown_logged = True
        log(f"source crop: unknown ({reason}); the LMB compare shows the uncropped source")

    def _stop_source_decoder(self) -> None:
        # The pool belongs to that decoder: keeping pictures from it after it is closed would show a frame of a file
        # that is no longer being compared.
        self._source_pool.clear()
        self._source_shown_pts = None
        self._source_fill_pending = None
        thread = getattr(self, "_source_thread", None)
        if thread is not None:
            # Stop the loop FIRST: `close()` under a running pump would pull the container out from under a decode.
            thread.quit()
            thread.wait(2000)
            self._source_thread = None
        decoder = getattr(self, "_source_decoder", None)
        if decoder is not None:
            try:
                decoder.frame_ready.disconnect()
            except Exception:
                pass
            self._compare_relay = None
            try:
                decoder.close()
            except Exception:
                pass
            self._source_decoder = None

    def _on_preview_opened(self, duration: float, fps: float, frames_total: float) -> None:
        self._preview_duration = float(duration) or 0.0

    def _pane_frame_report(self, stats) -> None:
        """One line per second: what the tail/decoder handed the pane and where it stopped.

        A frame that arrives and paints nothing has three possible reasons (the feed owns the pane, the owner kind
        refuses it, the pane is not there) and no number existed for any of them - "play does nothing during a render"
        could only be read as a stall.
        """
        now = time.monotonic()
        if now - float(stats.get("at") or 0.0) < 1.0:
            return
        stats["at"] = now
        telemetry.event("pane frames", got=stats["got"], dropped_feed=stats["may"],
                        dropped_kind=stats["kind"], painted=stats["painted"])
        stats.update(got=0, may=0, kind=0, painted=0)

    def _on_preview_frame(self, frame) -> None:
        """One decoded frame from a preview's live tail (or a finished file): RGB24 bytes -> QImage -> the pane.

        An EXPORT's pane has two possible painters if the output is followable: the display feed and the tail. One
        picture, one painter - the display feed wins and the tail's frame is dropped here. That is the rule the deleted
        640x360 channel broke: two writers raced for the same surface, so the pane flickered between a thumbnail and a
        decoded frame and neither could be trusted. A preview keeps its tail as the only painter (`wanted` stays 0) -
        except while the pane waits for the one frame a strip press asked the block for: the owner admits the feed for
        exactly that frame (`may_paint`), so a tail frame arriving in the meantime is dropped here, not painted over it.

        `frame_ready` is delivered queued, so a tail stopped at the job's end can still deliver a frame it emitted
        before the stop. Such a frame has no current owner (the owner's `kind` is `NONE`: the tail is gone, and
        playback of a finished file installs its own decoder) and is dropped: a zombie painter is the same rule
        violation as an early one.
        """
        from PySide6.QtGui import QImage

        owner = owner_of(self)
        stats = getattr(self, "_pane_frame_stats", None)
        if stats is None:
            stats = self._pane_frame_stats = {"got": 0, "may": 0, "kind": 0, "painted": 0, "at": time.monotonic()}
        stats["got"] += 1
        if owner.may_paint:
            stats["may"] += 1
            self._pane_frame_report(stats)
            return
        if owner.kind in (PreviewBackendKind.NONE, PreviewBackendKind.QUEUE_FILE):
            stats["kind"] += 1
            self._pane_frame_report(stats)
            # `QUEUE_FILE`: the queue item's own decoder is the pane's one painter while its tab is current; a render
            # frame arriving here is the second painter the owner's kind exists to rule out.
            return
        path_started = time.perf_counter()
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
        planes_frame = str(getattr(frame, "format", "") or "") in ("yuv420p10le", "yuv420p")
        if hasattr(pane, "set_frame"):
            if planes_frame and hasattr(pane, "set_frame_planes"):
                # The playback path: the producer's YUV planes go to the GPU as they are, the shader converts.
                # No QImage, no `to_ndarray`/`tobytes`, no RGBA conversion - the short path of direction 2.
                pane.set_frame_planes(
                    frame.planes, frame.strides, frame.width, frame.height,
                    index=getattr(frame, "index", None), pts=getattr(frame, "pts", None),
                    format=str(frame.format),
                    color_space=getattr(frame, "color_space", None),
                    color_range=getattr(frame, "color_range", None),
                    gui_ms=(time.perf_counter() - path_started) * 1000.0)
            else:
                image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                               QImage.Format.Format_RGB888).copy()
                pane.set_frame(image, gui_ms=(time.perf_counter() - path_started) * 1000.0,
                               index=getattr(frame, "index", None), media_time=getattr(frame, "pts", None))
            self._preview_playhead = float(getattr(frame, "pts", 0.0) or 0.0)
            if stats is not None:
                stats["painted"] += 1
            # Both halves in one paint: the source frame for this moment is already decoded (the pool runs ahead).
            paint_pair = getattr(self, "_source_paint_pair", None)
            if callable(paint_pair):
                paint_pair()          # the pane's own time: the picture that has just been painted
            # The press is CLOSED by the session (plan P4), not here: `observe(Painted)` decides whether this picture
            # is the one that was asked for (two frames' slack) and calls the page's formatter for the owner's landing
            # line. The page's `_seek_watch` timeout no longer decides anything - a press nothing answers is abandoned
            # by the session's deadline (`check_deadline`, called from the 100 ms tick), and a late frame then only
            # updates the picture. The frame's own index travels with the event - the tail numbers frames, so the
            # session can name the picture the same way the producer does.
            from .preview.session import Painted

            session_of = getattr(self, "_press_session", None)
            if callable(session_of):
                session_of().observe(Painted(media_time=float(getattr(frame, "pts", 0.0) or 0.0),
                                             index=getattr(frame, "index", None)))
            duration = float(getattr(self, "_preview_duration", 0.0) or 0.0)
            live_tail = getattr(self, "_live_tail", None)
            running = self.renderProcess is not None and self.renderProcess.poll() is None
            if live_tail is not None or running:
                # Live: the TINTED BAND is what the RENDERER has produced (`currentFrame/max_value`; it is the played
                # blue at PRODUCED_ALPHA, not a second colour - see `PreviewTimeline`), and the playhead is where the
                # tail stands, which lags the edge after a seek, so the strip must show it. The duration is the
                # estimate the produced share implies; without it the strip would draw no playhead at all and a seek
                # would be invisible until the renderer caught up with the target.
                pane.set_progress(rendered=self.currentFrame, total=self.max_value,
                                  playhead=frame.pts, duration=self._live_duration_estimate())
            else:
                # Finished file: everything exists, so the strip is fully produced; the playhead is the only marker.
                pane.set_progress(rendered=duration, total=duration,
                                  playhead=frame.pts, duration=duration)
        else:
            pixmap = QtGui.QPixmap.fromImage(image)
            pane.setPixmap(self.getRoundedPixmap(pixmap, corner_radius=10))
        # The frame names the playhead; the readout and the scrubber follow the picture, not a timer.
        self._push_preview_state()

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

        It runs ON THE GUI THREAD: the tick is a plain-callable connection (the page is not a `QObject`), and PySide
        queues such a delivery to the thread that owns the `QApplication` - measured, and pinned by
        `harness/tests/test_preview_state_tick.py::test_the_tick_body_runs_on_the_gui_thread`. That is the licence for
        the widget writes below, and `_tick_press_deadline` (plan P4) rides the same clock.

        It used to be called with a 640x360 thumbnail from shared memory. The picture now has two routes, both starting
        here: a preview's live tail decodes the file it follows (`_ensure_live_tail`), an export's display channel reads
        one new `seq` from the block the GUI owns (`_tick_display`). Exactly one of them paints a given pane.
        """

        # The pane's 100 ms clock is also the live tail's start point: it is the only GUI-thread tick during a
        # render, and the tail must be built and wired there (see `_start_live_tail`).
        self._ensure_live_tail()
        # The pane's one press session gets the tick's time: a press whose frame never arrives is abandoned at its
        # deadline (plan P4), and the marker falls back to the picture instead of standing on the request forever.
        self._tick_press_deadline()
        # The block is allocated once per pass, before the job's own `isPreview` is known (the worker sets it), so the
        # pane's seek reach is re-asserted here, on the GUI thread, once the owner is unambiguous.
        self._apply_seek_ceiling()
        # Keep the transport honest for whichever painter owns the pane this tick (the tail or the file decoder): an
        # enabled button that holds nothing, or a disabled one that holds the picture, is worse than no button.
        self._sync_preview_transport()
        self._refresh_held_progress()
        # The bottom bar reads `PreviewState`, and the state used to be pushed ONLY when a frame was painted - so a
        # paused pane froze the produced band and the thumb at the last frame while the renderer kept producing
        # (owner, 2026-11-25: the white band stood at f 316/493 under a "100%" pill, and a press on the scrubber moved
        # nothing). The tick is the only clock a paused pane has, and `Controller._emit` drops an unchanged state, so
        # this costs nothing until a number actually moves.
        self._push_preview_state()
        # The PICTURE path has its own clock (`_picture_tick`): this one is the readout tick at 100 ms, and running the
        # feed and the compare on it capped the preview at 10 pictures a second during a render, whatever the model did
        # (measured 2026-11-26: paints at exactly 100 ms while the render produced ~29 fps).
        self._ensure_picture_timer()


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
            #: The compare's own block, allocated with the pane's so the renderer can publish the source picture of each
            #: composited frame beside the display one (`--source_state_id`).
            source_state_id=SOURCE_STATE_SHARED_MEMORY_ID,
        )
