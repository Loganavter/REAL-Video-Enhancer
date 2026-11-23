from multiprocessing import shared_memory
import sys
import time
import numpy as np


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
        paused_shared_memory_id,
        outputWidth,
        outputHeight,
        totalOutputFrames,
    ):
        self.startTime = time.time()
        self.paused_shared_memory_id = paused_shared_memory_id
        self.width = outputWidth
        self.height = outputHeight
        self.totalOutputFrames = totalOutputFrames
        self.last_length = 0
        self.framesRendered = 1

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


    def setFramesRendered(self, framesRendered: int):
        self.framesRendered = framesRendered

    def stopWriting(self):
        self.stop = True

    def writeOutInformation(self):
        """Print FPS/ETA for the GUI's progress line.

        The GUI's progress bar, the strip's produced band and the live seek all come from this one line ("FPS: ...
        Current Frame: N ETA: ..."), so this loop must keep printing. It used to also announce the preview channel;
        that channel is gone (the pane reads the preview file), and the leftover `self.preview` read took the whole
        loop down with an AttributeError - the progress line stopped, and a seek had no produced band to aim at
        (measured 2026-11-23 on the first run after the channel was deleted).
        """
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
