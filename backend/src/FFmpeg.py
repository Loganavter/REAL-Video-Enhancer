from multiprocessing import shared_memory
import sys
import time
import numpy as np

from .utils.PreviewChannel import FMT_RGB888, PreviewChannel

if __name__ != "__main__":
    from .utils.Util import log, subprocess_popen_without_terminal, set_inplace_line
else:
    def log(message):
        print(message)

    def set_inplace_line(open_: bool = True):
        pass




def convertTime(remaining_time):
    """
    Converts seconds to hours, minutes and seconds
    """
    hours = remaining_time // 3600
    remaining_time -= 3600 * hours
    minutes = remaining_time // 60
    remaining_time -= minutes * 60
    seconds = remaining_time
    if minutes < 10:
        minutes = str(f"0{minutes}")
    if seconds < 10:
        seconds = str(f"0{seconds}")
    return hours, minutes, seconds


class PauseManager:
    def __init__(self, paused_shared_memory_id):
        self.isPaused = False
        self.prevState = None
        self.paused_shared_memory_id = paused_shared_memory_id
        if self.paused_shared_memory_id is not None:
            # The segment is named after the GUI process and owned by it: attach through `utils.shm`, which keeps this
            # process's resource tracker out of it (otherwise the tracker unlinks the GUI's flag on the way out, and
            # prints `KeyError: '//paused_state<pid>'` when the GUI got there first - measured 2026-11-23 on a forced
            # kill of the render process).
            from .utils.shm import attach as _attach, create as _create

            try:
                self.pausedSharedMemory = _attach(self.paused_shared_memory_id)
            except FileNotFoundError:
                log(f"FileNotFoundError! Creating new paused shared memory: {self.paused_shared_memory_id}")
                self.pausedSharedMemory = _create(self.paused_shared_memory_id, 1)
    def pause_manager(self):
        if self.paused_shared_memory_id is not None:
            return self.pausedSharedMemory.buf[0] == 1

class InformationWriteOut:
    def __init__(
        self,
        sharedMemoryID,  # preview channel id, None = no viewer
        paused_shared_memory_id,
        outputWidth,
        outputHeight,
        totalOutputFrames,
        preview_size=(640, 360),
    ):
        self.startTime = time.time()
        self.sharedMemoryID = sharedMemoryID
        self.paused_shared_memory_id = paused_shared_memory_id
        self.width = outputWidth
        self.height = outputHeight
        self.totalOutputFrames = totalOutputFrames
        self.last_length = 0
        self.framesRendered = 1
        # The preview is a thumbnail on its own small channel (utils/PreviewChannel.py):
        # no full-size frame, no writer thread holding a copy of the last one.
        # The box fixes the slot stride for the whole render: a 4:3 frame's 640x480
        # thumbnail and a 16:9 frame's 640x360 one must not overlap in the block.
        self.preview = (PreviewChannel(sharedMemoryID,
                                       box=(int(preview_size[0]), int(preview_size[1])))
                        if sharedMemoryID else None)
        self.previewWidth, self.previewHeight = int(preview_size[0]), int(preview_size[1])

        self.pausedManager = PauseManager(paused_shared_memory_id)
        self.isPaused = False
        self.stop = False

    def realTimePrint(self, data):
        data = str(data)
        # Clear the last line
        sys.stdout.write("\r" + " " * self.last_length)
        sys.stdout.flush()

        # Write the new line
        sys.stdout.write("\r" + data)
        sys.stdout.flush()

        # Update the length of the last printed line
        self.last_length = len(data)
        # the line stays open until something writes a whole line (see Util.end_inplace_line);
        # without this the next log record lands on its tail
        set_inplace_line(True)

    def get_is_paused(self):
        return self.isPaused

    def calculateETA(self, framesRendered):
        """
        Calculates ETA

        Gets the time for every frame rendered by taking the
        elapsed time / completed iterations (files)
        remaining time = remaining iterations (files) * time per iteration

        """

        # Estimate the remaining time
        elapsed_time = time.time() - self.startTime
        time_per_iteration = elapsed_time / framesRendered
        remaining_iterations = self.totalOutputFrames - framesRendered
        remaining_time = remaining_iterations * time_per_iteration
        remaining_time = int(remaining_time)
        # convert to hours, minutes, and seconds
        hours, minutes, seconds = convertTime(remaining_time)
        return f"{hours}:{minutes}:{seconds}"

    def previewAttached(self):
        """True when a viewer allocated the block (used to decide whether to warm up)."""
        return self.preview is not None and self.preview.attached

    def previewWanted(self):
        """True when the viewer consumed the last thumbnail and wants the next one."""
        return self.preview is not None and self.preview.wanted()

    def publishPreview(self, payload, width, height, fmt=None, high_bitdepth=False):
        """Hand one thumbnail to the channel; cheap, and only called when wanted."""
        return self.preview.publish(
            payload, width, height,
            fmt=fmt if fmt is not None else FMT_RGB888,
            high_bitdepth=high_bitdepth,
        )

    def setFramesRendered(self, framesRendered: int):
        self.framesRendered = framesRendered

    def stopWriting(self):
        self.stop = True
        if self.preview is not None:
            self.preview.close()

    def writeOutInformation(self):
        """Print FPS/ETA for the GUI's progress line.

        The preview used to ride on this thread as a full-size frame copy every 0.5 s;
        it now lives on its own small channel and is published by the render loop, so
        this loop only formats a line - and `padFrame`/the chunk size are not needed.
        """
        if self.preview is not None and self.preview.attached:
            log(f"Preview channel: {self.preview.name} ({self.previewWidth}x{self.previewHeight})")
        i = 0
        while not self.stop:
            if self.framesRendered > 0:
                fps = round(self.framesRendered / (time.time() - self.startTime))
                eta = self.calculateETA(framesRendered=self.framesRendered)
                message = f"FPS: {fps} Current Frame: {self.framesRendered} ETA: {eta}"
                if i == 0:
                    print("\n", file=sys.stderr)
                    i = 1
                self.realTimePrint(message)
                self.isPaused = self.pausedManager.pause_manager()
            time.sleep(0.5) # setting this to a higher value will reduce the cpu usage, and increase fps
