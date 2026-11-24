"""One-frame-late staging for the display channel: the render loop submits, never waits.

The composited frame lives on the GPU, and a synchronous copy back would make every published frame
pay the device's whole latency - measured on this card (RTX 4060 Ti, 118-frame runs) at 0.5 ms median
but 1.1-2.2 ms in the tail, and the tail did not follow the byte count: it follows whatever the model,
the encoder and the desktop's own GPU clients are doing at that moment.

So the loop does not synchronise. `submit()` resizes the frame into the pane's box, copies it into a
page-locked host buffer with `non_blocking=True` and records an event; `take()` hands the PREVIOUS
frame's bytes back once that event has fired. A preview one frame late is invisible at the viewer's
2-5 Hz. The three pieces of the copy contract are the ones the deleted `PreviewStager` established
(`busy()` / `take()` / `submit()`); what changed is the source: the composite goes through the whole
model path, so a pane shows the picture the export is writing rather than a thumbnail of the source.

Import stays torch-free so the harness can import this module on an interpreter without CUDA: the
torch handle is read lazily from `utils.Frame` (which only initialises it when a render builds a
Frame), and `submit` refuses to work when it is not there.
"""

from __future__ import annotations

from . import Frame as frame_module


def fit_box(in_width: int, in_height: int, box_width: int, box_height: int) -> tuple[int, int]:
    """The largest `in_width x in_height` picture that fits the box, aspect preserved.

    The box is the pane's long side: a 1920x1080 frame into a 640x360 box becomes exactly 640x360, and a
    4:3 frame into the same box becomes 480x360 (fitted, never stretched). A frame smaller than the box is
    scaled up to it - the pane asked for that box. Never zero, never larger than the box.
    """
    in_width, in_height = max(1, int(in_width)), max(1, int(in_height))
    box_width, box_height = max(1, int(box_width)), max(1, int(box_height))
    scale = min(box_width / in_width, box_height / in_height)
    width = max(1, min(box_width, int(round(in_width * scale))))
    height = max(1, min(box_height, int(round(in_height * scale))))
    return width, height


class DisplayStager:
    """Resize a composited frame into the pane's box and copy it back without blocking the loop."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.device = None        # resolved on first use: torch is initialised lazily
        self._torch = None
        self._pinned = None       # page-locked host buffer the device copies into
        self._event = None
        self._pending = False
        self._pending_time: float | None = None
        self._pending_fmt = 1
        self._pending_w = 0
        self._pending_h = 0
        self._pending_len = 0

    def busy(self) -> bool:
        """True while a frame is still crossing the link (one in flight, so nothing is lost)."""
        return self._pending

    def set_box(self, width: int, height: int) -> None:
        """Change the box frames are fitted into, without touching anything in flight.

        The next `submit` fits into the new box; a frame already staged keeps the geometry it was staged with, and its
        header (written by `publish`) names that geometry - so a resize is a sequence of whole frames, never a mix. The
        pane announces its box and the renderer applies it at a frame boundary, which is the shape the professional
        sources use for exactly this (mpv refuses to reconfigure while a frame is displaying; VLC only reshapes the
        viewport; OBS resizes the draw target, not the canvas).
        """
        self.width = max(2, int(width))
        self.height = max(2, int(height))

    def _handle(self):
        torch = frame_module._torch
        if torch is None:
            return None
        if self._torch is not torch:
            self._torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch

    def submit(self, frame, width: int | None = None, height: int | None = None,
               fmt: int = 1, time_s: float | None = None) -> bool:
        """Stage `frame` (a `utils.Frame.Frame`) into the box and start the copy back.

        `width`/`height` are the geometry the caller believes the frame has; the tensor's own shape is
        the authority (a header can lag the tensor it holds), so they are kept for the contract only.
        False when the frame cannot be shown (no torch, a raw YUV payload, a frame already in flight).
        """
        if self._pending:
            return False
        torch = self._handle()
        if torch is None or frame is None:
            return False
        if getattr(frame, "_is_raw_payload", None) and frame._is_raw_payload():
            # The int8 tail can hand over the packed yuv420p pipe instead of an RGB image; decoding it
            # back here would cost more than the preview is worth, and the pane needs RGB.
            return False
        tensor = getattr(frame, "_tensor", None)
        if tensor is None:
            getter = getattr(frame, "get_frame_tensor", None)
            tensor = getter() if getter is not None else None
        if tensor is None or getattr(tensor, "dim", lambda: 0)() != 4:
            return False
        channels = int(tensor.shape[1])
        if channels < 1:
            return False
        in_height, in_width = int(tensor.shape[-2]), int(tensor.shape[-1])
        target_w, target_h = fit_box(in_width, in_height, self.width, self.height)
        try:
            image = tensor[:, :3] if channels >= 3 else tensor[:, :1].expand(-1, 3, -1, -1)
            import torch.nn.functional as F
            mode = "area" if target_w < in_width or target_h < in_height else "bilinear"
            small = F.interpolate(image, size=(target_h, target_w), mode=mode)
            uint16 = getattr(torch, "uint16", None)
            if small.dtype == torch.uint8:
                rgb = small
            elif uint16 is not None and small.dtype == uint16:
                rgb = (small >> 8).to(torch.uint8)
            else:
                # NOT in place: the tensor belongs to the frame the writer is about to pack, and a
                # clamp/round into it would change the exported picture to make the preview brighter.
                rgb = (small.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            flat = rgb[0].permute(1, 2, 0).contiguous().reshape(-1)
        except Exception:
            return False
        n = target_w * target_h * 3
        if int(flat.numel()) != n:
            return False
        if self._pinned is None or int(self._pinned.numel()) < n:
            pinned = torch.empty(n, dtype=torch.uint8, pin_memory=bool(self.device == "cuda"))
            self._pinned = pinned
            self._event = torch.cuda.Event() if self.device == "cuda" else None
        self._pinned[:n].copy_(flat, non_blocking=True)
        if self._event is not None:
            self._event.record()
        self._pending_fmt = int(fmt)
        self._pending_w = target_w
        self._pending_h = target_h
        self._pending_len = n
        #: The media time of THIS picture, or None when the caller could not name it. It travels with the staged
        #: payload because a normal render publishes one frame late: the time of the picture being PUBLISHED is the
        #: staged one, not whatever the newest frame carries.
        self._pending_time = time_s
        self._pending = True
        return True

    def take(self, force: bool = False) -> tuple[bytes, int, int, int, float | None] | None:
        """`(payload, fmt, width, height, time_s)` of the previous frame, or None when it is late/absent.

        `force` waits for the in-flight copy instead of answering None. Only the seek path uses it: a
        requested frame has to be on screen now, while a normal render can happily be one frame late.
        """
        if not self._pending:
            return None
        if self._event is not None and not self._event.query():
            if not force:
                return None
            self._event.synchronize()
        self._pending = False
        payload = self._pinned[:self._pending_len].numpy().tobytes()
        return payload, self._pending_fmt, self._pending_w, self._pending_h, self._pending_time
