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
        
        


class RenderQueue:
    """The render queue: the options, the names, and the list the user picks them from.

    Selection is the manager's business now: the page has no Remove / Move Up / Move Down buttons (owner, 2026-11-22),
    the actions live in the right-click menu and operate on everything selected - which is what makes a rubber-band
    selection worth having.

    `locked` is the render lock: while a render walks this queue, its items are being consumed by the writer thread, so
    removing or reordering them would change the list under it. A locked queue still SCROLLS - the old
    `disableProcessPage` disabled the whole page, which froze the list mid-render, and a log of finished files is
    exactly what a user scrolls through then.
    """

    def __init__(self, qlistwidget: "TextListWidget"):
        self.queue = []
        self.inputNameList = []  # this list links up 1:1 with the queue, storing input names allow for index searching
        self.qlistwidget = qlistwidget
        self.locked = False

    def set_locked(self, locked: bool) -> None:
        self.locked = bool(locked)

    def is_locked(self) -> bool:
        return bool(self.locked)

    def add(self, renderable: RenderOptions):
        if self.locked:
            return
        self.queue.append(renderable)
        self.qlistwidget.addItem(renderable.inputFile)
        self.inputNameList.append(renderable.inputFile)

    def addToStart(self, renderable: RenderOptions):
        if self.locked:
            return
        self.queue.insert(0, renderable)
        self.qlistwidget.insertItem(0, renderable.inputFile)
        self.inputNameList.insert(0, renderable.inputFile)

    def clear(self):
        self.queue.clear()
        self.inputNameList.clear()
        self.qlistwidget.clear()

    def getQueue(self) -> list[RenderOptions]:
        return self.queue

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
        """Remove every selected row (descending, so the indices stay valid); returns how many went."""
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
