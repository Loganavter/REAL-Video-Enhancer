import threading
from typing import Any, Optional
try:
    import numpy as np
except ImportError:
    pass
from .Util import resize_image_np, log
_pytorch_device = None
_pytorch_dtype = None
_pytorch_stream = None
_torch_utils = None
_torch = None

def _init_pytorch(device, gpu_id, dtype, width, height, hdr_mode, high_bitdepth=None):
    global _pytorch_device, _pytorch_dtype, _torch_utils, _torch, _pytorch_stream
    if _torch_utils is None:
        import torch
        from ..pytorch.TorchUtils import TorchUtils
        _torch = torch
        _torch_utils = TorchUtils(
            width=width, 
            height=height, 
            device_type=device, 
            hdr_mode=hdr_mode, 
            gpu_id=gpu_id,
            high_bitdepth=high_bitdepth,
            )
        _pytorch_stream = _torch_utils.init_stream(gpu_id=gpu_id)
        _pytorch_device = _torch_utils.handle_device(device, gpu_id)
        _pytorch_dtype = _torch_utils.handle_precision(dtype)
        print("Initialized Frame PyTorch utils")


# ---------------------------------------------------------------------------------------
# Pooled pinned output payload
#
# The output path is: GPU tensor -> tensor_to_frame() (permute/scale/uint8) -> .cpu() (one
# host allocation) -> .numpy() (view) -> get_frame_bytes() -> .tobytes() (a SECOND full host
# copy) -> writeQueue -> ffmpeg.stdin.write. At 4320x7680 that second copy is 99.5 MB of
# memcpy per frame, ~10 ms of a ~41 ms output stage, and it exists only because the queue
# carries `bytes` while the conversion buffer is reused by the next frame.
#
# A pool of `n` pinned host buffers removes it: the conversion (the DMA out of the GPU)
# writes straight into a slot and the writer thread pipes that slot to ffmpeg. A slot must
# not be recycled while its payload is still on the way to the pipe, which is what the
# semaphore is for - acquire() blocks instead of handing out a buffer the writer still owns.
# The pool therefore is the backpressure point, independent of the queue's own maxsize.
#
# Host RAM cost per slot (4320x7680 = 33.2 Mpx):
#   rgb24 / yuv420p10le ....  99.5 MB  (3 bytes/pixel)
#   rgb48le (HDR) ..........  199.1 MB (6 bytes/pixel)
# The default of 4 slots pins ~400 MB (rgb24, yuv420p10le) or ~800 MB (rgb48le). Two slots
# already cover a writer that keeps up; 4-6 cover a slow encoder without stalling the render
# thread. More slots do not make it faster, they only absorb jitter.
#
# The pool lives here, not in pytorch/TorchUtils.py, because this module is importable (and
# the whole mechanism testable) without torch: the allocator is injected.
# ---------------------------------------------------------------------------------------

PINNED_POOL_SLOTS = 4


def _default_pinned_allocator(numel: int, dtype):
    """One host buffer of `numel` elements, page-locked when the build can pin.

    Pinned memory is what makes GPU -> host a single DMA instead of a staging copy through
    a pageable buffer. Only an accelerator build has it; without one we fall back to plain
    host memory (still correct, just not async) rather than failing a render.
    """
    import torch  # local import: keep this module importable on a numpy-only interpreter

    try:
        return torch.empty(numel, dtype=dtype, pin_memory=True)
    except (RuntimeError, AssertionError) as error:
        log(f"Pinned host memory unavailable ({error}); using pageable output buffers.")
        return torch.empty(numel, dtype=dtype)


def _as_byte_view(buffer) -> memoryview:
    """A flat, byte-addressed writable view over a pool slot.

    `len()` is the payload size in bytes - exactly the frame size the writer needs - and
    `Popen.stdin.write` accepts it as a buffer. The view keeps the slot's storage alive.
    """
    if hasattr(buffer, "numpy"):  # torch tensor: shares (possibly pinned) storage
        try:
            buffer = buffer.numpy()
        except (RuntimeError, TypeError):
            # a dtype numpy cannot express (or a tensor that refuses .numpy()): fall back to
            # the raw bytes of the same storage rather than failing the render
            import torch

            buffer = buffer.view(torch.uint8).numpy()
    view = memoryview(buffer)
    return view if view.format == "B" and view.ndim == 1 else view.cast("B")


def _output_buffer_dtype(high_bitdepth: bool):
    """uint16/uint8 for a pooled output buffer, as a torch dtype when torch is loaded."""
    if _torch is not None:
        return _torch.uint16 if high_bitdepth else _torch.uint8
    return np.uint16 if high_bitdepth else np.uint8


class PinnedSlotPool:
    """Rotating pool of `count` equally sized host buffers with acquire/release.

    Contract: the producer acquires a slot, fills it and hands the memoryview on; the
    consumer releases the slot once the payload has been copied out (for the writer: after
    `stdin.write()` returned, i.e. the bytes are in the pipe's own buffer). While a payload
    is live its slot is out of the free list and its semaphore permit is taken, so a second
    producer blocks instead of overwriting a frame that is still being written.
    """

    def __init__(self, count: int, numel: int, dtype, allocator=None):
        self.count = int(count)
        self.numel = int(numel)
        self.dtype = dtype
        if self.count < 1:
            raise ValueError(f"a pinned pool needs at least one slot, got {count}")
        self._allocator = allocator or _default_pinned_allocator
        self._semaphore = threading.Semaphore(self.count)
        self._lock = threading.Lock()
        self._buffers: list = [None] * self.count
        self._views: list = [None] * self.count
        self._free: list = list(range(self.count))
        self._in_use: set = set()
        # Payload size in bytes is whatever the allocator hands back for `numel` elements.
        self.num_bytes: Optional[int] = None
        # Stats a bench needs to see whether the pool ever bound the render thread.
        self.acquires = 0
        self.blocked = 0
        self.max_in_use = 0

    # -- allocation ---------------------------------------------------------------------
    def _allocate(self, slot: int) -> None:
        buffer = self._allocator(self.numel, self.dtype)
        view = _as_byte_view(buffer)
        if self.num_bytes is None:
            self.num_bytes = len(view)
        elif len(view) != self.num_bytes:
            raise RuntimeError(
                f"pool slot {slot} is {len(view)} bytes, expected {self.num_bytes} "
                f"({self.numel} x {self.dtype})"
            )
        self._buffers[slot] = buffer
        self._views[slot] = view

    @property
    def allocated(self) -> int:
        """How many slots have actually been allocated (lazy: only the used ones)."""
        return sum(1 for buffer in self._buffers if buffer is not None)

    @property
    def in_use(self) -> int:
        return len(self._in_use)

    def buffer(self, slot: int):
        """The raw slot storage (a torch tensor in the app) for filling."""
        return self._buffers[slot]

    # -- acquire / release --------------------------------------------------------------
    def acquire(self) -> "tuple[int, memoryview]":
        """Block until a slot is free, then return `(slot_index, byte_view)`."""
        if not self._semaphore.acquire(blocking=False):
            # The writer is behind: wait for it instead of lapping it (that is the whole
            # point of the pool - overwriting a live slot is a torn frame in the output).
            self.blocked += 1
            self._semaphore.acquire()
        with self._lock:
            slot = self._free.pop()
            if self._buffers[slot] is None:
                try:
                    self._allocate(slot)
                except BaseException:
                    self._free.append(slot)
                    self._semaphore.release()
                    raise
            self._in_use.add(slot)
            self.acquires += 1
            if len(self._in_use) > self.max_in_use:
                self.max_in_use = len(self._in_use)
            return slot, self._views[slot]

    def release(self, slot: int) -> bool:
        """Return `slot` to the pool. Idempotent - a second release is refused.

        Refusing matters: a duplicate release would add a permit the pool does not own and
        the next two producers would share one slot.
        """
        with self._lock:
            if slot not in self._in_use:
                return False
            self._in_use.discard(slot)
            self._free.append(slot)
        self._semaphore.release()
        return True

    def stats(self) -> str:
        return (
            f"pinned pool: {len(self._in_use)}/{self.count} slots in use "
            f"(max {self.max_in_use}), {self.allocated} allocated, "
            f"{self.acquires} payloads, {self.blocked} waits on a slow writer, "
            f"{self.numel} elems x {self.dtype}"
        )


class Frame:
    def __init__(self, backend: str, width: int, height: int, device, gpu_id, hdr_mode, dtype,
                 high_bitdepth: bool | None = None):
        self.backend = backend
        self.width = width
        self.height = height
        self.gpu_id = gpu_id
        self.device = device
        self.hdr_mode = hdr_mode
        # 16-bit DATA path (10-bit sources) vs HDR COLOUR tagging - see FFmpegBuffers
        self.high_bitdepth = hdr_mode if high_bitdepth is None else high_bitdepth
        # set by the upscaler: emit yuv420p10le instead of RGB bytes (10-bit SDR path)
        self.yuv10_output = False
        # set by the reader: the incoming bytes are packed yuv420p10le, not RGB
        self.yuv10_input = False
        self.dtype = dtype
        
        self.tensor_conversions = 0
        self._tensor: Optional[Any] = None
        self._np: Optional[np.ndarray] = None
        self._bytes: Optional[bytes] = None
        # When True, get_frame_bytes() hands out a view on a pinned pool slot instead of a
        # freshly converted `bytes` (saves one full-frame host copy, see PinnedSlotPool).
        # The renderer sets this on the frame it is about to queue for the writer.
        self.pooled_output = False
        self.pool_slots = PINNED_POOL_SLOTS
        self._pool: Optional[PinnedSlotPool] = None
        self._pool_slot: Optional[int] = None
        # True once a pooled payload was handed to the writer: it is a one-shot buffer
        self._pooled_payload_spent = False

        if backend in ("pytorch", "tensorrt"):
            _init_pytorch(device, gpu_id, dtype, width, height, hdr_mode, self.high_bitdepth)

    def __del__(self):
        # A pooled slot that was never released (an aborted render, an exception between
        # acquire and the queue) would permanently shrink the pool and eventually hang the
        # render thread on acquire(). Releasing here is the seatbelt for that.
        try:
            self.release_payload()
        except Exception:
            pass

    def _invalidate_cache(self, keep: str):
        """Clear cached representations except the one being set."""
        if keep != "tensor":
            del self._tensor
            self._tensor = None
        if keep != "np":
            del self._np
            self._np = None
        if keep != "bytes":
            # The payload is going away, so its slot may go back to the pool. Keeping the
            # two lifetimes separate would leak one slot per invalidated frame.
            self._release_pooled_slot()
            del self._bytes
            self._bytes = None

    def set_frame_bytes(self, frame: bytes) -> "Frame":
        if not isinstance(frame, bytes):
            raise TypeError(f"Expected bytes, got {type(frame).__name__}")
        self._invalidate_cache("bytes")
        self._bytes = frame
        return self
    
    def set_frame_tensor(self, frame: Any, copy: bool = False) -> "Frame":
        # might need to sync streams here
        if _torch is not None and not isinstance(frame, _torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(frame).__name__}")
        self._invalidate_cache("tensor")
        # No copy by default: callers hand over a freshly allocated tensor (model output)
        # and a full-HR copy costs ~4.6 ms/frame at 4x. Pass copy=True if the caller may
        # keep writing into the same buffer (e.g. preallocated/cudagraph outputs).
        self._tensor = frame.clone() if copy else frame
        _torch_utils.sync_all_streams()
        return self

    def set_frame_np(self, frame: Any) -> "Frame":
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(frame).__name__}")
        self._invalidate_cache("np")
        self._np = frame
        return self
    # --- Lazy Getters ---
    
    def get_frame_tensor(self, clear_cache:bool =False) -> Any:
        """
        Get the frame as a torch tensor in format (1, C, H, W).
        """
        
        if self._tensor is None:
            if self._bytes is not None:
                if getattr(self, "yuv10_input", False):
                    self._tensor = _torch_utils.yuv420p10_to_tensor(
                        self._bytes, _pytorch_stream, _pytorch_device, _pytorch_dtype,
                    )
                else:
                    self._tensor = _torch_utils.frame_to_tensor(
                        self._bytes, _pytorch_stream, _pytorch_device, _pytorch_dtype, 
                    )
            elif self._np is not None:
                self._tensor = _torch_utils.np_to_tensor(
                    self._np, _pytorch_device, _pytorch_dtype
                )
        if clear_cache:
            self._invalidate_cache("tensor")
            
        return self._tensor.clone() # this helps so the frame wont be overwritten? have to test later.

    def get_frame_bytes(self, clear_cache: bool = False) -> Any:
        """The frame as pipe-ready bytes.

        `bytes` normally; when `pooled_output` is set on a pytorch/tensorrt frame it is a
        writable `memoryview` on a pinned pool slot (same payload, different storage) and
        the caller must return the slot with `release_payload()` once it is done with it.
        """
        if self._bytes is None:
            if self._tensor is not None:
                if self._uses_pooled_output():
                    self._bytes = self._fill_pooled_payload()
                elif getattr(self, "yuv10_output", False):
                    # 10-bit SDR: pack YUV420p10 on the GPU so the pipe stays 3 bytes/px
                    self._bytes = _torch_utils.tensor_to_yuv420p10(self._tensor)
                else:
                    self._bytes = _torch_utils.tensor_to_frame(self._tensor)
            elif self._np is not None:
                self._bytes = self._np_to_bytes(self._np)
            elif getattr(self, "_pooled_payload_spent", False):
                # Loud instead of silently re-emitting whatever frame now occupies the slot.
                # The one path that does this in the app is temporal reuse, which remembers
                # the previous frame and emits it again - see INTEGRATION: pooling and
                # temporal reuse are mutually exclusive.
                raise RuntimeError(
                    "this frame's pooled payload was already written and handed back to the "
                    "pool; a pooled frame cannot be emitted twice (keep pooled_output off "
                    "when the frame may be re-emitted, e.g. temporal reuse)"
                )

        if clear_cache:
            self._invalidate_cache("bytes")

        return self._bytes

    # --- pooled output payload ---------------------------------------------------------
    def _uses_pooled_output(self) -> bool:
        return bool(
            getattr(self, "pooled_output", False)
            and _torch_utils is not None
            and self.backend in ("pytorch", "tensorrt")
            and self._tensor is not None
        )

    def _pooled_numel(self) -> int:
        """Element count the conversion is going to produce (from the tensor, not the
        frame header: an upscaled frame's width/height can lag the tensor it holds)."""
        if self._tensor is not None:
            height, width = int(self._tensor.shape[-2]), int(self._tensor.shape[-1])
        else:
            height, width = int(self.height), int(self.width)
        if getattr(self, "yuv10_output", False):
            # packed yuv420p10le: Y plane + two ceil/2 chroma planes (tensor_to_yuv420p10)
            return width * height + 2 * ((height + 1) // 2) * ((width + 1) // 2)
        return width * height * 3

    def _fill_pooled_payload(self) -> memoryview:
        """Acquire a slot and convert the tensor straight into it. Blocks if the writer lags."""
        numel = self._pooled_numel()
        pool = _torch_utils.pinned_slots(
            self.pool_slots, numel, _output_buffer_dtype(self.high_bitdepth)
        )
        slot, view = pool.acquire()
        try:
            dest = pool.buffer(slot)
            if getattr(self, "yuv10_output", False):
                _torch_utils.tensor_to_yuv420p10(self._tensor, dest=dest)
            else:
                _torch_utils.tensor_to_frame(self._tensor, dest=dest)
        except BaseException:
            pool.release(slot)
            raise
        self._pool = pool
        self._pool_slot = slot
        # Single hand-off: the payload belongs to whoever writes it next (see release_payload)
        self._pooled_payload_spent = True
        # The payload now owns the conversion: drop the GPU tensor so a frame waiting in the
        # write queue does not pin VRAM (25 queued frames would be gigabytes at 8K).
        self._tensor = None
        return view

    def _release_pooled_slot(self) -> None:
        pool, slot = getattr(self, "_pool", None), getattr(self, "_pool_slot", None)
        self._pool = None
        self._pool_slot = None
        if pool is not None and slot is not None:
            pool.release(slot)
            return True
        return False

    def release_payload(self) -> None:
        """Give a pooled output slot back to the pool. Idempotent; no-op when not pooled.

        The writer calls this immediately after `stdin.write()` returned. Afterwards the
        payload is stale - its slot may already hold the next frame - so the cached view is
        dropped as well: re-emitting it would write someone else's pixels.
        """
        if self._release_pooled_slot():
            self._bytes = None

    def get_frame_np(self, clear_cache: bool = False) -> Any:
        """
        Get the frame as a numpy array in format (H, W, C).
        """
        if self._np is None:
            if self._tensor is not None:
                self._np = _torch_utils.tensor_to_np(self._tensor)
                
            elif self._bytes is not None:
                self._np = self._bytes_to_np(self._bytes)

        if clear_cache:
            self._invalidate_cache("np")
        return self._np

    # --- Conversion helpers ---
    def _bytes_to_np(self, data: bytes) -> Any:
        # Assuming raw RGB/BGR bytes
        channels = 3
        
        return np.frombuffer(data, dtype=np.uint8 if not self.high_bitdepth else np.uint16).reshape(
            self.height, self.width, int(channels)
        )

    def _np_to_bytes(self, arr: Any) -> bytes:
        return arr.tobytes()

    def resize_frame(self, new_width: int, new_height: int) -> "Frame":
        
        if self._tensor is not None:
            self._tensor = _torch_utils.resize_tensor(
                self._tensor, new_width, new_height
            )
        if self._np is not None:
            self._np = resize_image_np(
                self._np, new_width, new_height
            )
        if self._bytes is not None:
            np_frame = self._bytes_to_np(self._bytes)
            resized_np = resize_image_np(
                np_frame, new_width, new_height
            )
            self._bytes = self._np_to_bytes(resized_np)
        self.width = new_width
        self.height = new_height
        return self
    
    def get_np_sdr(self):
        """
        Get the frame as a numpy array in SDR format (H, W, C) with dtype uint8.
        """
        np_frame = self.get_frame_np()
        if self.high_bitdepth:
            # Convert from HDR (uint16) to SDR (uint8)
            np_frame = (np.clip(np_frame.astype(np.float32) / 65535.0, 0, 1) * 255).astype(np.uint8)
        return np_frame

    
    def clone(self) -> "Frame":
        new_frame = Frame(
            backend=self.backend,
            width=self.width,
            height=self.height,
            device=self.device,
            gpu_id=self.gpu_id,
            hdr_mode=self.hdr_mode,
            dtype=self.dtype,
            high_bitdepth=self.high_bitdepth,
        )
        if self._tensor is not None:
            new_frame.set_frame_tensor(self._tensor.clone())
        if self._np is not None:
            new_frame.set_frame_np(self._np.copy())
        if self._bytes is not None:
            new_frame.set_frame_bytes(self._bytes)
        new_frame.yuv10_output = self.yuv10_output
        return new_frame
    
    def get_dummy_frame(self) -> "Frame":
        return Frame(
            backend=self.backend,
            width=self.width,
            height=self.height,
            device=self.device,
            gpu_id=self.gpu_id,
            hdr_mode=self.hdr_mode,
            dtype=self.dtype,
            high_bitdepth=self.high_bitdepth,
        )
