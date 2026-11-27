import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the widget is only a type here: the model is plain Python and imports without Qt
    from sli_ui_toolkit.widgets import TextListWidget


@dataclass
class RenderOptions:
    def __init__(
        self,
        inputFile: str,
        outputPath: str,
        videoWidth: int,
        videoHeight: int,
        videoFps: int,
        tilingEnabled: bool,
        tilesize: str,
        videoFrameCount: int,
        backend: str,
        modelScale: int,
        interpolateTimes: int,
        benchmarkMode: bool,
        sloMoMode: bool,
        dyanmicScaleOpticalFlow: bool,
        ensemble: bool,
        interpolateModelFile: str | None = None,
        upscaleModelFile: str | None = None,
        upscaleModelArch: str | None = None,
        deblurModelFile: str | None = None,
        denoiseModelFile: str | None = None,
        decompressModelFile: str | None = None,
        sceneChangeModelFile: str | None = None,
        hdrMode: bool = False,
        startTime: float | None = None,
        endTime: float | None = None,
        isPreview: bool = False,
        overrideUpscaleScale: int | None = None,
        encoderCommand: str | None = None,
    ):
        self._inputFile = inputFile
        self._outputPath = outputPath
        self._videoWidth = videoWidth
        self._videoHeight = videoHeight
        self._videoFps = videoFps
        self._tilingEnabled = tilingEnabled
        self._tilesize = tilesize
        self._videoFrameCount = videoFrameCount
        self._backend = backend
        self._interpolateModelFile = interpolateModelFile
        self._deblurModelFile = deblurModelFile
        self._upscaleModelFile = upscaleModelFile
        self._upscaleModelArch = upscaleModelArch
        self._denoiseModelFile = denoiseModelFile
        self._decompressModelFile = decompressModelFile
        self._sceneChangeModelFile = sceneChangeModelFile
        self._modelScale = modelScale
        self._interpolateTimes = interpolateTimes
        self._benchmarkMode = benchmarkMode
        self._sloMoMode = sloMoMode
        self._dyanmicScaleOpticalFlow = dyanmicScaleOpticalFlow
        self._ensemble = ensemble
        self._hdrMode = hdrMode
        self._startTime = startTime
        self._endTime = endTime
        self._isPreview = isPreview
        self._overrideUpscaleScale = overrideUpscaleScale
        self._encoderCommand = encoderCommand

    @property
    def inputFile(self):
        return self._inputFile

    @inputFile.setter
    def inputFile(self, value: str):
        self._inputFile = value

    @property
    def outputPath(self):
        return self._outputPath

    @outputPath.setter
    def outputPath(self, value: str):
        self._outputPath = value

    @property
    def videoWidth(self):
        return self._videoWidth

    @videoWidth.setter
    def videoWidth(self, value: int):
        self._videoWidth = value

    @property
    def videoHeight(self):
        return self._videoHeight

    @videoHeight.setter
    def videoHeight(self, value: int):
        self._videoHeight = value

    @property
    def videoFps(self):
        return self._videoFps

    @videoFps.setter
    def videoFps(self, value: int):
        self._videoFps = value

    @property
    def tilingEnabled(self):
        return self._tilingEnabled

    @tilingEnabled.setter
    def tilingEnabled(self, value: bool):
        self._tilingEnabled = value

    @property
    def tilesize(self):
        return self._tilesize

    @tilesize.setter
    def tilesize(self, value: str):
        self._tilesize = value

    @property
    def videoFrameCount(self):
        return self._videoFrameCount

    @videoFrameCount.setter
    def videoFrameCount(self, value: int):
        self._videoFrameCount = value

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        self._backend = value
    
    @property
    def modelScale(self):
        return self._modelScale
    @modelScale.setter
    def modelScale(self, value: int):
        self._modelScale = value

    @property
    def denoiseModelFile(self):
        return self._denoiseModelFile
    @denoiseModelFile.setter
    def denoiseModelFile(self, value: str):
        self._denoiseModelFile = value
    
    @property
    def decompressModelFile(self):
        return self._decompressModelFile
    @decompressModelFile.setter
    def decompressModelFile(self, value: str):
        self._decompressModelFile = value
    
    @property
    def sceneChangeModelFile(self):
        return self._sceneChangeModelFile
    @sceneChangeModelFile.setter
    def sceneChangeModelFile(self, value: str):
        self._sceneChangeModelFile = value

    @property
    def upscaleModelArch(self):
        return self._upscaleModelArch
    @upscaleModelArch.setter
    def upscaleModelArch(self, value: str):
        self._upscaleModelArch = value

    @property
    def interpolateModelFile(self):
        return self._interpolateModelFile

    @interpolateModelFile.setter
    def interpolateModelFile(self, value: str):
        self._interpolateModelFile = value
    
    @property
    def deblurModelFile(self):
        return self._deblurModelFile
    
    @deblurModelFile.setter
    def deblurModelFile(self, value: str):
        self._deblurModelFile = value

    @property
    def upscaleModel(self):
        return self._upscaleModel

    @upscaleModel.setter
    def upscaleModel(self, value: str):
        self._upscaleModel = value

   
   
    @property
    def interpolateTimes(self):
        return self._interpolateTimes

    @interpolateTimes.setter
    def interpolateTimes(self, value: int):
        self._interpolateTimes = value

    @property
    def benchmarkMode(self):
        return self._benchmarkMode

    @benchmarkMode.setter
    def benchmarkMode(self, value: bool):
        self._benchmarkMode = value

    @property
    def sloMoMode(self):
        return self._sloMoMode

    @sloMoMode.setter
    def sloMoMode(self, value: bool):
        self._sloMoMode = value

    @property
    def dyanmicScaleOpticalFlow(self):
        return self._dyanmicScaleOpticalFlow

    @dyanmicScaleOpticalFlow.setter
    def dyanmicScaleOpticalFlow(self, value: bool):
        self._dyanmicScaleOpticalFlow = value

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, value: bool):
        self._ensemble = value

    @property
    def upscaleModelFile(self):
        return self._upscaleModelFile

    @upscaleModelFile.setter
    def upscaleModelFile(self, value: str):
        self._upscaleModelFile = value
    
    @property
    def hdrMode(self):
        return self._hdrMode
    @hdrMode.setter
    def hdrMode(self, value: bool):
        self._hdrMode = value
    
    @property
    def startTime(self):
        return self._startTime
    @startTime.setter
    def startTime(self, value: float):
        self._startTime = value

    @property
    def endTime(self):
        return self._endTime
    @endTime.setter
    def endTime(self, value: float):
        self._endTime = value

    @property
    def isPreview(self):
        return self._isPreview
    @isPreview.setter
    def isPreview(self, value: bool):
        self._isPreview = value
    
    @property
    def overrideUpscaleScale(self):
        return self._overrideUpscaleScale
    @overrideUpscaleScale.setter
    def overrideUpscaleScale(self, value: int):
        self._overrideUpscaleScale = value
    
    @property
    def encoderCommand(self):
        return self._encoderCommand
    @encoderCommand.setter
    def encoderCommand(self, value: str):
        self._encoderCommand = value
        
        


from dataclasses import dataclass as _dataclass, field as _field
from time import monotonic as _monotonic, time as _wallclock


@_dataclass
class JobState:
    """Where one queued job stands: the queue row's progress bar and its hover line come from here.

    Times are kept twice on purpose: `monotonic` for durations (wall-clock jumps must not produce a negative render time)
    and wall clock for "18hr ago" (a monotonic origin means nothing to a human).
    """

    status: str = "pending"          # pending | rendering | done | failed
    progress: float | None = None    # 0..1 while rendering, 1.0 when done
    added_mono: float | None = None   # when the item entered the queue
    added_wall: float | None = None
    started_mono: float | None = None
    started_wall: float | None = None
    finished_mono: float | None = None
    finished_wall: float | None = None
    source_seconds: float = 0.0      # the input's duration
    error: str = ""

    def elapsed(self) -> float:
        if self.started_mono is None:
            return 0.0
        end = self.finished_mono if self.finished_mono is not None else _monotonic()
        return max(0.0, end - self.started_mono)

    def duration(self) -> float:
        """Render time for a finished job, elapsed time for a running one."""
        return self.elapsed()

    def produced_seconds(self) -> float:
        """Media duration produced so far (progress x source duration)."""
        return max(0.0, (self.progress or 0.0)) * self.source_seconds


class RenderQueue:
    """The render queue: the options, the names, and the list the user picks them from.

    Selection is the manager's business now: the page has no Remove / Move Up / Move Down buttons (owner, 2026-11-22),
    the actions live in the right-click menu and operate on everything selected - which is what makes a rubber-band
    selection worth having.

    Two different locks live here, and they answer different questions:

    * ``locked`` is the RENDER lock. While a pass walks this queue, removing or reordering a row moves the ground under
      the consumer, so ``removeSelected``/``moveSelected`` refuse while it is set. It does **not** close the queue:
      a row added mid-run is the owner's request (2026-11-26, "добавить возможность пополнять очередь во время
      работы"), and an append changes no row the consumer already holds. The pass takes rows by STATUS from the live
      list (``take_next_pending``), so one added while it runs is simply the next one.
    * ``_lock`` is a thread lock. ``queue`` and ``inputNameList`` are written by the GUI thread (add, remove, move,
      clear) and read by the render thread (``take_next_pending``, the job-state writers). Every access goes through
      it: a lock on one side only is the race it exists to prevent. It is re-entrant because a Qt signal fired by
      ``qlistwidget.addItem``/``setItems`` inside a critical section runs on the same thread and asks the queue for
      its rows again.

    A locked queue still SCROLLS - the old `disableProcessPage` disabled the whole page, which froze the list
    mid-render, and a log of finished files is exactly what a user scrolls through then.
    """

    def __init__(self, qlistwidget: "TextListWidget"):
        self.queue = []
        self.inputNameList = []  # this list links up 1:1 with the queue, storing input names allow for index searching
        self.qlistwidget = qlistwidget
        self.locked = False
        self._lock = threading.RLock()

    def set_locked(self, locked: bool) -> None:
        with self._lock:
            self.locked = bool(locked)

    def is_locked(self) -> bool:
        with self._lock:
            return bool(self.locked)

    def add(self, renderable: RenderOptions):
        """Append a row. Allowed while a pass runs: that is what "add to the queue during a render" is."""
        with self._lock:
            self._ensure_state(renderable)
            self._stamp_added(renderable)
            self.queue.append(renderable)
            self.qlistwidget.addItem(renderable.inputFile)
            self.inputNameList.append(renderable.inputFile)

    def addToStart(self, renderable: RenderOptions):
        """Insert a row at the top. Also allowed mid-run: the next ``take_next_pending`` picks it up first."""
        with self._lock:
            self._ensure_state(renderable)
            self._stamp_added(renderable)
            self.queue.insert(0, renderable)
            self.qlistwidget.insertItem(0, renderable.inputFile)
            self.inputNameList.insert(0, renderable.inputFile)

    @staticmethod
    def _ensure_state(renderable: RenderOptions) -> None:
        """Give the row its job state (and the source duration the hover line reads) - once."""
        if getattr(renderable, "state", None) is None:
            renderable.state = JobState(source_seconds=float(getattr(renderable, "videoFrameCount", 0) or 0)
                                        / max(1.0, float(getattr(renderable, "videoFps", 1) or 1)))

    @staticmethod
    def _stamp_added(renderable: RenderOptions) -> None:
        """Record when the item entered the queue - once. `started_*`/`finished_*` already live on `JobState`; this is
        the only moment the queue knows and the job does not, so it is stamped here and never overwritten (a re-add
        of an item that carries a state keeps the arrival it already reported)."""
        state = getattr(renderable, "state", None)
        if state is None or getattr(state, "added_wall", None) is not None:
            return
        state.added_mono = _monotonic()
        state.added_wall = _wallclock()

    def clear(self):
        with self._lock:
            self.queue.clear()
            self.inputNameList.clear()
            self.qlistwidget.clear()

    def getQueue(self) -> list[RenderOptions]:
        """A COPY of the rows, taken under the lock.

        Callers walk it while the render thread may be taking the next row, so handing out the live list would let a
        snapshot be built from a list that changed mid-iteration. The queue's own methods are the only writers.
        """
        with self._lock:
            return list(self.queue)

    def pending_jobs(self) -> list[RenderOptions]:
        """Jobs still to render - a finished row stays in the list as history, it is not run again.

        A one-shot snapshot, kept for callers that want the whole set (counts, tests). The render pass does NOT use
        it: it takes one row at a time with `take_next_pending`, so a row added mid-run is seen.
        """
        with self._lock:
            return [item for item in self.queue
                    if getattr(getattr(item, "state", None), "status", "pending") == "pending"]

    def take_next_pending(self) -> RenderOptions | None:
        """The next row still to render, marked `rendering`, or None when nothing is pending.

        The scan, the check and the mark are ONE critical section: two consumers cannot take the same row, and a row
        appended while the pass runs is seen by the next call (there is no snapshot to go stale). A row removed
        before its turn is simply no longer in the list, so it is not resurrected; a row removed while it is being
        rendered is detached from the list but keeps rendering - its own `mark_finished` then writes to a state no row
        shows, which is harmless.
        """
        with self._lock:
            for renderable in self.queue:
                if getattr(getattr(renderable, "state", None), "status", "pending") == "pending":
                    self._mark_rendering_locked(renderable)
                    return renderable
            return None

    def _mark_rendering_locked(self, renderable: RenderOptions) -> None:
        """`mark_rendering`'s body, for callers already holding `_lock`."""
        state = getattr(renderable, "state", None)
        if state is None:
            return
        state.status = "rendering"
        state.progress = 0.0
        state.started_mono = _monotonic()
        state.started_wall = _wallclock()

    def mark_rendering(self, renderable: RenderOptions) -> None:
        with self._lock:
            self._mark_rendering_locked(renderable)

    def update_progress(self, renderable: RenderOptions, fraction: float) -> None:
        with self._lock:
            state = getattr(renderable, "state", None)
            if state is None or state.status != "rendering":
                return
            state.progress = max(0.0, min(1.0, float(fraction)))

    def mark_finished(self, renderable: RenderOptions, *, ok: bool = True, error: str = "") -> None:
        with self._lock:
            state = getattr(renderable, "state", None)
            if state is None:
                return
            state.status = "done" if ok else "failed"
            state.progress = 1.0 if ok else state.progress
            state.finished_mono = _monotonic()
            state.finished_wall = _wallclock()
            state.error = str(error)

    def selectedRows(self) -> list[int]:
        rows = getattr(self.qlistwidget, "selectedRows", None)
        if callable(rows):
            return list(rows())
        current = self.qlistwidget.currentRow()
        return [current] if current >= 0 else []

    def _refresh(self, selection, current: int | None = None) -> None:
        """Rebuild the visible rows from the model and restore the selection."""
        self.qlistwidget.setItems(self.inputNameList)
        valid = [row for row in selection if 0 <= row < len(self.inputNameList)]
        if valid:
            # Current row FIRST, selection after: `setCurrentRow` selects the one row it moves to (that is what a click
            # does), so doing it second collapsed a moved block to a single row - the owner selected three, pressed Move
            # Up and saw one highlighted (2026-11-22).
            self.qlistwidget.setCurrentRow(valid[0] if current is None else current)
            self.qlistwidget.setSelectedRows(valid)

    def removeSelected(self) -> int:
        """Remove every selected row (descending, so the indices stay valid); returns how many went.

        Refused while a pass runs: the consumer holds one row at a time from the live list, and deleting a row shifts
        every index after it. Adding is allowed (`add`); removing is not.
        """
        with self._lock:
            if self.locked:
                return 0
            rows = sorted({row for row in self.selectedRows() if 0 <= row < len(self.queue)}, reverse=True)
            if not rows:
                return 0
            first = min(rows)
            for row in rows:
                del self.queue[row]
                del self.inputNameList[row]
            self._refresh([], current=max(0, min(first, len(self.inputNameList) - 1)))
            return len(rows)

    def moveSelected(self, direction: str = "up") -> bool:
        """Move the selected rows one place (a block moves as a block); returns True when anything moved."""
        with self._lock:
            if self.locked:
                return False
            rows = sorted({row for row in self.selectedRows() if 0 <= row < len(self.queue)})
            if not rows:
                return False
            selected = set(rows)
            moved = False
            # Ascending for up, descending for down: each row checks the neighbour it is moving towards, so a selected
            # block slides together instead of collapsing into one row.
            order = rows if direction != "down" else list(reversed(rows))
            for row in order:
                target = row + 1 if direction == "down" else row - 1
                if target < 0 or target >= len(self.queue) or target in selected:
                    continue
                self.queue[row], self.queue[target] = self.queue[target], self.queue[row]
                self.inputNameList[row], self.inputNameList[target] = (
                    self.inputNameList[target],
                    self.inputNameList[row],
                )
                selected.discard(row)
                selected.add(target)
                moved = True
            if moved:
                order_after = sorted(selected)
                self._refresh(order_after, current=order_after[0])
            return moved

    # -- the QListWidget-era names this manager answers (the toolkit control keeps the same calls) ------

    def remove(self):
        return self.removeSelected()

    def moveitem(self, direction="up"):
        return self.moveSelected(direction)
