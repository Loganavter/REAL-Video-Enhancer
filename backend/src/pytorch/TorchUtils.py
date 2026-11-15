import torch
import torch.nn.functional as F
import sys
import threading
from ..utils.BackendDetect import (
    BackendDetect
)
backendDetect = BackendDetect()

from ..utils.Util import (
    log,
    CudaChecker
)
HAS_PYTORCH_CUDA = CudaChecker().HAS_PYTORCH_CUDA
import numpy as np

def dummy_function(*args, **kwargs):
    """
    A dummy function that does nothing.
    This is used as a placeholder for device-specific functions that may not be available.
    """
    pass

def dummy_context_manager(*args, **kwargs):
    """
    A dummy context manager that does nothing.
    This is used as a placeholder for device-specific context managers that may not be available.
    """
    return DummyContextManager()

class DummyContextManager:
    def __call__ (self, *args, **kwargs):
        """
        A dummy callable that returns a DummyContextManager instance.
        This is used as a placeholder for device-specific context managers that may not be available.
        """
        return self
    def __enter__(self):
        return self  # could return any resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"An exception occurred: {exc_type}")
        return False  # re-raise exceptions if any


class TorchUtils:
    # device and precision are in string formats, loaded straight from the command line arguments
    def __init__(self, width, height, device_type:str, hdr_mode=False, padding=None, gpu_id=0,
                 high_bitdepth: bool | None = None):
        self.width = width
        self.height = height
        self.hdr_mode = hdr_mode
        self.high_bitdepth = hdr_mode if high_bitdepth is None else high_bitdepth
        self.gpu_id = gpu_id
        if device_type == "auto":
            self.device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu"    
        else:
            self.device_type = device_type
        try:
            test_tensor = torch.tensor([1.0]).cpu().numpy()
            del test_tensor
            self.use_numpy = True
        except Exception as e:
            log(f"Failed to create a Numpy tensor, This will heavily reduce performance.")
            self.use_numpy = False
        self.__run_stream_func = self.__run_stream_function()
        self.__sync_all_streams_func = self.__sync_all_streams_function()

        # persistent pinned buffer for async GPU uploads
        self._pinned_buffer = None
        self._pinned_buffer_numel = 0
        self._pinned_buffer_dtype = None

        # rotating pool(s) of pinned host buffers for the output payload, keyed by geometry
        # (see utils.Frame.PinnedSlotPool). One pool per (slots, numel, dtype), reused for
        # every frame of a render - the reader creates a Frame per frame, the pool must not.
        self._pinned_pools: dict = {}
        self._pinned_pool_lock = threading.Lock()

    def pinned_slots(self, n: int, numel: int, dtype, allocator=None):
        """The shared pool of `n` pinned host buffers of (numel, dtype).

        `allocator(numel, dtype)` is injected so the pool can be exercised without torch;
        in the app it defaults to pinned torch memory (pageable fallback when the build
        cannot pin). See `utils.Frame.PinnedSlotPool` for the acquire/release contract.
        """
        key = (int(n), int(numel), str(dtype))
        with self._pinned_pool_lock:
            pool = self._pinned_pools.get(key)
            if pool is None:
                # imported here: utils.Frame imports this module lazily, so a module-level
                # import in either direction would be a cycle
                from ..utils.Frame import PinnedSlotPool

                pool = PinnedSlotPool(n, numel, dtype, allocator=allocator)
                self._pinned_pools[key] = pool
        return pool

    def pinned_pool_stats(self) -> str:
        """One line per output pool: the numbers that show whether the writer bound us."""
        with self._pinned_pool_lock:
            pools = list(self._pinned_pools.values())
        if not pools:
            return "pinned pool: unused (no pooled output payloads)"
        return " | ".join(pool.stats() for pool in pools)

    def __sync_all_streams_function(self):
        if self.device_type == "cuda":
            return torch.cuda.synchronize
        if self.device_type == "mps":
            return torch.mps.synchronize
        if self.device_type == "cpu":
            return dummy_function  # CPU does not require explicit synchronization
        if self.device_type == "xpu":
            return torch.xpu.synchronize
        return lambda: log(f"Unknown device type {self.device_type}, skipping stream synchronization.")

    def init_stream(self, gpu_id = 0) -> torch.Stream:
        """
        Initializes the stream based on the device type.
        """
        log(f"Initializing stream for device {self.device_type} (GPU ID: {gpu_id})")
        device = self.handle_device(self.device_type, gpu_id)
        if self.device_type == "cuda":
            return torch.cuda.Stream(device=device)
        elif self.device_type == "xpu":
            return torch.xpu.Stream(device=device)
        else:
            return DummyContextManager()  # For CPU and MPS, we can use a dummy stream

    def __run_stream_function(self) -> callable:
        """
        Runs the stream based on the device type.
        """
        if self.device_type == "cuda":
            return torch.cuda.stream
        elif self.device_type == "xpu":
            return torch.xpu.stream
        else:
            return  dummy_context_manager # For CPU and MPS, we can use a dummy context manager


    def run_stream(self, stream):
        return self.__run_stream_func(stream) 

    def sync_stream(self, stream: torch.Stream):
        match self.device_type:
            case "cuda" | "xpu":
                stream.synchronize()
            case "mps":
                torch.mps.synchronize()
            case "cpu":
                pass  # CPU does not require explicit synchronization
            case _:
                log(f"Unknown device type {self.device_type}, skipping stream synchronization.")
                # For other devices, we assume no synchronization is needed.
        
    def sync_all_streams(self):
        """
        Synchronizes all streams based on the device type.
        """
        self.__sync_all_streams_func()
             
    @staticmethod
    def handle_device(device, gpu_id: int = 0) -> torch.device:
        """
        returns device based on gpu id and device parameter
        """
        log(f"Handling device: {device}, GPU ID: {gpu_id}")
        if device == "auto":
            if torch.cuda.is_available():
                torchdevice = torch.device("cuda", gpu_id)
            else:
                torchdevice = torch.device("mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        
        elif device == "cuda":
            torchdevice = torch.device(
                device, gpu_id
            )  # 0 is the device index, may have to change later
        else:
            torchdevice = torch.device(device)
    
        device = backendDetect.get_gpus_torch()[gpu_id]
        print("Using Device: " + str(device), file=sys.stderr)
        return torchdevice

    @staticmethod
    def handle_precision(precision) -> torch.dtype:
        log(f"Handling precision: {precision}")
        if precision == "auto":
            return torch.float16 if backendDetect.get_half_precision() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16
        if precision == "bfloat16":
            return torch.bfloat16
        return torch.float32

    @torch.inference_mode()
    def copy_tensor(self, tensorToCopy: torch.Tensor, tensorCopiedTo: torch.Tensor, stream: torch.Stream): # stream might be None
        with self.run_stream(stream):  # type: ignore
            tensorToCopy.copy_(tensorCopiedTo, non_blocking=True)
        
            self.sync_stream(stream)

    @torch.inference_mode()
    def yuv420p10_to_tensor(self, frame, stream: torch.Stream, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Packed yuv420p10le bytes -> (1,3,H,W) float on the GPU.

        Doing YUV->RGB here instead of piping rgb48le keeps the input pipe at 3 bytes/pixel
        and removes ffmpeg's own RGB conversion from the loop. BT.709, limited range,
        bilinear chroma upsampling; validated at mean |d| 1.85/255 against ffmpeg.
        """
        with self.run_stream(stream):
            raw = torch.frombuffer(frame, dtype=torch.uint16)
            n = self.width * self.height
            c = (self.height // 2) * (self.width // 2)
            planes = raw.to(device=device, non_blocking=True).half()
            y = planes[:n].view(self.height, self.width)
            u = planes[n:n + c].view(self.height // 2, self.width // 2)
            v = planes[n + c:n + 2 * c].view(self.height // 2, self.width // 2)

            y = y.sub_(64.0).div_(876.0)
            u = u.sub_(512.0).div_(896.0)
            v = v.sub_(512.0).div_(896.0)
            u = torch.nn.functional.interpolate(u[None, None], size=(self.height, self.width),
                                                mode="bilinear", align_corners=False)[0, 0]
            v = torch.nn.functional.interpolate(v[None, None], size=(self.height, self.width),
                                                mode="bilinear", align_corners=False)[0, 0]
            # BT.709
            kr, kb = 0.2126, 0.0722
            r = y + (2.0 * (1.0 - kr)) * v
            b = y + (2.0 * (1.0 - kb)) * u
            g = (y - kr * r - kb * b) / (1.0 - kr - kb)
            rgb = torch.stack((r, g, b)).clamp_(0.0, 1.0).unsqueeze(0).to(dtype=dtype)
            if stream is not None:
                self.sync_stream(stream)
        return rgb

    @torch.inference_mode()
    def frame_to_tensor(self, frame, stream: torch.Stream, device: torch.device, dtype: torch.dtype) -> torch.Tensor: # stream might be None
        with self.run_stream(stream):  # type: ignore
             # ... (tensor creation and manipulation) ...
            frame = torch.frombuffer(
                    frame,
                    dtype=torch.uint16 if self.high_bitdepth else torch.uint8,
                ).to(device=device, non_blocking=True) 
            
            frame = (
                frame
                .div(65535.0 if self.high_bitdepth else 255.0)
                .clamp(0.0, 1.0)
                .reshape(self.height, self.width, 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
                ).to(dtype=dtype, non_blocking=True)
                
            self.sync_stream(stream)

        # No explicit sync for CPU here.
        return frame
    
    @staticmethod
    def clear_cache():
        if HAS_PYTORCH_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
    
    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor, dest: torch.Tensor | None = None):
        """(1,3,H,W) float 0..1 -> HWC uint8/uint16, on the host.

        `dest` (a 1-D host buffer of `dest.numel() == H*W*3` elements, ideally pinned) is
        filled in place: one DMA and no host allocation, which is the whole point of the
        pooled output payload. Without `dest` this is exactly the previous behaviour - a
        fresh host tensor plus the numpy view of it.
        """
        # Prepare the tensor
        tensor = (
            frame.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0., 1.)
            .mul(65535.0 if self.high_bitdepth else 255.0)
            .round()
            .to(torch.uint16 if self.high_bitdepth else torch.uint8)
            .contiguous()
            .detach()
        )
        if dest is not None:
            target = dest if dest.shape == tensor.shape else dest.reshape(tensor.shape)
            target.copy_(tensor, non_blocking=True)
            # the writer thread reads the host buffer next, so the DMA must have landed
            self.sync_all_streams()
            return target
        tensor = tensor.cpu()
        if self.use_numpy:
            # Convert to numpy array if possible
            return tensor.numpy()
        else:
            np_dtype = np.uint16 if self.high_bitdepth else np.uint8
            return np.array(tensor.tolist(), dtype=np_dtype)
    
    @torch.inference_mode()
    def tensor_to_yuv420p10(self, frame: torch.Tensor, pixel_format: str = "yuv420p10le",
                            dest: torch.Tensor | None = None):
        """(1,3,H,W) float 0..1 -> packed yuv420p10le bytes (BT.709, limited range).

        Doing this on the GPU keeps the pipe at 3 bytes/pixel instead of 6 (rgb48le) while
        preserving 10-bit precision: measured 26.6 ms vs 40.8 ms for the old uint8 rgb24
        path at 4320x7680, and vs ~80 ms for piping rgb48le. Validated against ffmpeg's
        own bt709 conversion (mean |d| 1.85/255, the remainder being chroma upsampling).

        With `dest` the packed frame is written into that 1-D uint16 host buffer instead of
        being `.tobytes()`-ed into a new one - same bytes, one less full-frame copy.
        """
        x = frame if frame.dtype == torch.float16 else frame.half()
        r, g, b = x[0, 0], x[0, 1], x[0, 2]
        # BT.709 luma, fp16 to halve the memory traffic at 4K
        y = (0.2126 * r).add_(g, alpha=0.7152).add_(b, alpha=0.0722)
        cb = b.sub(y).mul_(0.5 / (1.0 - 0.0722))
        cr = r.sub(y).mul_(0.5 / (1.0 - 0.2126))
        # 4:2:0 - average 2x2 blocks; odd sizes are padded by replication, because ffmpeg
        # expects ceil(h/2) x ceil(w/2) chroma planes
        if y.shape[0] % 2 or y.shape[1] % 2:
            cb = torch.nn.functional.pad(cb[None, None], (0, y.shape[1] % 2, 0, y.shape[0] % 2), mode="replicate")[0, 0]
            cr = torch.nn.functional.pad(cr[None, None], (0, y.shape[1] % 2, 0, y.shape[0] % 2), mode="replicate")[0, 0]
        cb = torch.nn.functional.avg_pool2d(cb[None, None], 2)[0, 0]
        cr = torch.nn.functional.avg_pool2d(cr[None, None], 2)[0, 0]
        h, w = y.shape
        yq = (y.mul_(876.0).add_(64.0).round_().clamp_(64, 940)).to(torch.uint16).reshape(-1)
        uq = (cb.mul_(896.0).add_(512.0).round_().clamp_(64, 960)).to(torch.uint16).reshape(-1)
        vq = (cr.mul_(896.0).add_(512.0).round_().clamp_(64, 960)).to(torch.uint16).reshape(-1)
        if pixel_format == "p010le":
            # P010LE keeps the 10-bit sample in the MSBs of the 16-bit word (x64); with the
            # sample in the low bits ffmpeg decodes it as garbage (measured: R=B=0). CUDA has
            # no uint16 shift, so shift in int32 and come back.
            uv = torch.stack(((uq.to(torch.int32) << 6).to(torch.uint16),
                              (vq.to(torch.int32) << 6).to(torch.uint16)), dim=-1).reshape(-1)
            packed = torch.cat(((yq.to(torch.int32) << 6).to(torch.uint16), uv))
        else:
            packed = torch.cat((yq, uq, vq))
        if dest is not None:
            target = dest if dest.shape == packed.shape else dest.reshape(packed.shape)
            target.copy_(packed, non_blocking=True)
            self.sync_all_streams()  # the writer thread reads it next
            return target
        # Hand back the numpy array itself: it is freshly allocated per frame, so the writer
        # can put it straight into the pipe and we skip a full host copy of the frame
        # (18.6 MB at 2880x2160, measured ~2-3 ms/frame). bytes are only needed if a caller
        # wants an immutable snapshot.
        return packed.cpu().numpy()

    @staticmethod
    @torch.inference_mode()
    def np_to_tensor(arr: np.ndarray, device, dtype):
        import torch
        return torch.from_numpy(arr).to(device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)

    
    @staticmethod
    @torch.inference_mode()
    def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    @staticmethod
    @torch.inference_mode()
    def resize_tensor(
        tensor: torch.Tensor,
        new_width: int,
        new_height: int,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        return F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode=mode,
            align_corners=False if mode in ["linear", "bilinear", "bicubic", "trilinear"] else None,
        )