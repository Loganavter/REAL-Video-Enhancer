"""One-frame-late preview staging: the render loop enqueues a thumbnail, never waits for it.

Measured on this card (RTX 4060 Ti, 118-frame runs, both arms, two thumbnail sizes), the
synchronous path costs 0.5 ms at the median and 1.1-2.2 ms in the tail, and the tail did
not follow the byte count: shrinking 640x360 to 480x270 made the int8 arm's p95 worse
(0.91 -> 1.59 ms) while its payload fell to 56%. What the tail follows is the device, not
the preview - a D2H synchronises the stream and inherits whatever the model, the encoder
and the desktop's own GPU clients are doing at that moment.

So the loop does not synchronise. It gathers the thumbnail into a page-locked host buffer
with `non_blocking=True`, records an event, and publishes *the previous* frame's buffer
once that event has fired. A preview one frame late is invisible at the viewer's 2-5 Hz.

What the loop pays per published frame is a gather (no sync) and a 0.35-0.69 MB copy out
of pinned memory; what it no longer pays is the device's latency.
"""

from . import Frame as frame_module  # Frame owns the lazily-initialised torch module: import
# the module, not `Frame._torch`, because that global is still None at import time


class PreviewStager:
    def __init__(self, thumb_width, thumb_height):
        self.width = int(thumb_width)
        self.height = int(thumb_height)
        self.device = None        # resolved on first use: torch is initialised lazily
        self._pinned = None
        self._out = None          # the gather writes here: no per-call allocation
        self._scratch = None      # fp16 workspace for the rgb quantisation
        self._event = None
        self._pending = False
        self._pending_fmt = None
        self._pending_len = 0
        self._pending_flags = 0

    def busy(self):
        """True while a thumbnail is in flight (only one at a time, so nothing is lost)."""
        return self._pending

    def _buffers(self):
        torch = frame_module._torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        worst = 3 * self.width * self.height
        if self._out is None or self._out.numel() < worst:
            self._out = torch.empty(worst, dtype=torch.uint8, device=self.device)
            self._scratch = torch.empty(worst, dtype=torch.float16, device=self.device)
        return self._out, self._scratch

    def submit(self, frame, high_bitdepth=False):
        """Gather `frame`'s thumbnail into pinned memory and let the device copy it."""
        if self._pending or frame_module._torch is None:
            return False
        out, scratch = self._buffers()
        got = frame.preview_planes(self.width, self.height, out=out, scratch=scratch)
        if got is None:
            return False
        planes, fmt = got
        n = int(planes.numel())
        if self._pinned is None or self._pinned.numel() < n:
            # page-locked: a regular buffer would make `non_blocking` a lie and put the
            # copy back on the caller
            pinned = frame_module._torch.empty(n, dtype=frame_module._torch.uint8,
                                  pin_memory=bool(self.device == "cuda"))
            self._pinned = pinned
            if self.device == "cuda":
                self._event = frame_module._torch.cuda.Event()
        self._pinned[:n].copy_(planes.reshape(-1), non_blocking=True)
        if self._event is not None:
            self._event.record()
        self._pending_fmt = fmt
        self._pending_len = n
        self._pending_flags = 1 if high_bitdepth else 0
        self._pending = True
        return True

    def take(self):
        """`(payload, fmt, high_bitdepth)` of the previous frame if it has arrived, else None."""
        if not self._pending:
            return None
        if self._event is not None and not self._event.query():
            return None
        self._pending = False
        payload = self._pinned[:self._pending_len].numpy().tobytes()
        return payload, self._pending_fmt, bool(self._pending_flags)
