import os
import sys
import subprocess
import contextlib
import logging
import math
import traceback
from .Colors import Colors
# non standard python libraries
try:
    import numpy as np
    import cv2
    import shutil
except ImportError:
    pass


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr by redirecting them to /dev/null."""
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

try:
    from ..constants import PLATFORM
except ImportError:
    PLATFORM = sys.platform


def removeFile(file):
    try:
        os.remove(file)
    except Exception:
        log("Failed to remove file! " + file)


def removeFolder(folder):
    try:
        shutil.rmtree(folder)
    except Exception:
        print("Failed to remove file!")


def warnAndLog(message: str):
    print("WARNING: " + message, file=sys.stderr)


def errorAndLog(message: str):
    raise os.error("ERROR: " + message)



def log_error(message: str):
    log(Colors.RED + "ERROR: "+ message + Colors.RESET, show_backend=False)


# --- in-place progress line -------------------------------------------------
# FFmpeg.InformationWriteOut.realTimePrint repaints a single status line with "\r" on stdout
# while a render runs, and never ends it with "\n". Everything else writes whole lines, and the
# front end merges stderr into stdout and reads that pipe line by line
# (src/ui/ProcessTab.py), so a record written while the progress line is open is appended to its
# tail: "ETA: 0:03:19host timeline: upload 0.12 ..." reaches the ETA label and the output box.
# A writer that emits a whole line therefore ends the in-place line first.
_inplace_line_open = False


def set_inplace_line(open_: bool = True):
    """Record that an in-place ("\r") line is on screen, or that it was ended."""
    global _inplace_line_open
    _inplace_line_open = bool(open_)


def end_inplace_line(stream=None):
    """Terminate an open in-place line so the next writer starts a clean one."""
    global _inplace_line_open
    if not _inplace_line_open:
        return
    _inplace_line_open = False
    try:
        out = stream if stream is not None else sys.stdout
        out.write("\n")
        out.flush()
    except Exception:
        pass


class _EndInplaceLineFilter(logging.Filter):
    def filter(self, record):
        end_inplace_line()
        return True


def _install_rve_handler():
    """Print RVE records (all of them warnings) on their own line, never on a progress line."""
    logger = logging.getLogger("RVE")
    for handler in logger.handlers:
        if getattr(handler, "_rve_inplace_aware", False):
            return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._rve_inplace_aware = True
    handler.addFilter(_EndInplaceLineFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    # our handler is the one place RVE records print, so a basicConfig elsewhere cannot double them
    logger.propagate = False


_install_rve_handler()


def log(message: str, show_backend = True):
    """
    Log is now depricated, just using print now.
    """
    end_inplace_line()
    if show_backend:
        print("BACKEND: " + message, file=sys.stderr)
    else:
        print(message, file=sys.stderr)
    #message = message + "\n\n\n\n" + "-" * len(message)
    #print(message, file=sys.stderr)


def guard_render_thread(job, *, path: str = "RenderVideo.render", exit_process=None) -> int:
    """Run the render thread body so a failure ends the JOB, not the process's liveness.

    An exception in the render thread used to leave the backend alive: the writer and reader threads
    block on a queue that will never be fed again, the process keeps its GPU and its stdout, and the
    GUI polls a live process at 0 FPS forever (the 2026-11-22 "render that never ends"). The body is
    wrapped here instead, so the failure is one named line the parent can key on plus the traceback,
    and the process leaves non-zero in the same call - `os._exit` deliberately skips the non-daemon
    threads, which are exactly what kept it alive.

    `exit_process` is injectable so the wrapper is testable without a real process exit.
    """
    leave = os._exit if exit_process is None else exit_process
    try:
        job()
        return 0
    except BaseException as error:  # noqa: BLE001 - a thread body's exception is the process's failure
        # The summary line FIRST: the GUI reads the merged stdout line by line and keys on this
        # prefix (`ProcessTab.renderToPipeThread`) to end the job and show a failed state.
        try:
            log(f"RENDER FAILED: {type(error).__name__}: {error}")
        except Exception:
            pass
        try:
            traceback.print_exc()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            leave(1)
        except Exception:
            pass
        return 1


def bytesToImg(
    image: bytes, width, height, outputWidth: int = None, outputHeight: int = None
):
    channels = len(image) / (height * width) # 3 if RGB24/SDR, 6 if RGB48/HDR
    hdr = channels == 6
    frame = np.frombuffer(image, dtype=np.uint16 if hdr else np.uint8).reshape(height, width, 3).astype(np.uint8) # downgrade to sdr for scenedetect... its good enough.
    if outputHeight and outputWidth:
        frame = cv2.resize(frame, dsize=(100, 100))
    return frame


def get_pytorch_vram() -> int:
    """
    Function that returns the total VRAM amount in MB using PyTorch.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            vram_in_mb = props.total_memory // (1024**2)  # Convert bytes to MB
            return vram_in_mb
        else:
            return 0
    except ImportError as e:
        log(str(e))
        return 0
    except Exception as e:
        log(str(e))
        return 0


def resize_image_np(image, target_width: int, target_height: int):
    """
    Resizes the image to the target resolution.
    
    Args:
        image (np.ndarray): The input image as a numpy array.
        target_width (int): The target width for resizing.
        target_height (int): The target height for resizing.
    
    Returns:
        np.ndarray: The resized image as a numpy array.
    """
    height, width = image.shape[:2]
    if target_width == width and target_height == height:
        return image
    if target_width < width or target_height < height:
        # Best for downscaling
        interpolation = cv2.INTER_AREA
    else:
        # Best for upscaling
        interpolation = cv2.INTER_LANCZOS4

    try:
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    except Exception:
        resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image


#: Bytes per pixel `resize_image_bytes` understands: RGB24 (SDR) or RGB48 (HDR).
_RGB_BYTES_PER_PIXEL = (3, 6)


def resize_image_bytes(image_bytes: bytes, width: int, height: int, target_width: int, target_height: int,
                       path: str = "resize_image_bytes") -> bytes:
    """
    Resizes the image to the target resolution.

    `width`/`height` are the size the CALLER believes the payload has. They are checked against the
    payload's own length before anything is reshaped: a disagreement is a named `ValueError` (both
    geometries and `path`), never numpy's "cannot reshape array of size N into shape (...)" - that
    message names neither the source geometry nor the code path that fed it (measured 2026-11-22,
    `RenderVideo.render` override resize).

    Args:
        image_bytes (bytes): The input image in bytes.
        width (int): The declared width of the payload in pixels.
        height (int): The declared height of the payload in pixels.
        target_width (int): The target width for resizing.
        target_height (int): The target height for resizing.
        path (str): The code path, quoted in the error when the payload does not match.

    Returns:
        bytes: The resized image in bytes.
    """
    payload_size = getattr(image_bytes, "nbytes", None)
    payload_size = len(image_bytes) if payload_size is None else int(payload_size)
    width, height, bytes_per_pixel = payload_geometry(payload_size, width, height, path=path)
    if target_width == width and target_height == height:
        return image_bytes
    dtype = np.uint8 if bytes_per_pixel == 3 else np.uint16
    # Convert bytes to numpy array
    if target_width < width or target_height < height:
        # Best for downscaling
        interpolation = cv2.INTER_AREA
    else:
        # Best for upscaling
        interpolation = cv2.INTER_LANCZOS4
    image_array = np.frombuffer(image_bytes, dtype=dtype)
    image_array = image_array.reshape((height, width, 3))

    # Resize the image
    try:
        resized_image = cv2.resize(image_array, (target_width, target_height), interpolation=interpolation)
    except Exception:
        resized_image = cv2.resize(image_array, (target_width, target_height))
    # Convert the resized image back to bytes
    return resized_image.tobytes()


def resize_frame_bytes(frame, target_width: int, target_height: int,
                       path: str = "RenderVideo.render:override_upscale_scale") -> bytes:
    """Re-scale the frame the upscaler returned, using THAT frame's geometry as the source size.

    `width * modelScale` is not a source geometry: it is what the model descriptor claims, and the
    frame that comes back can be at a different scale (reproduced 2026-11-22 on
    `4xNomos8k_span_otf_weak_no_update_params.pth` with `--override_upscale_scale 2`: the payload was
    320x240 while the caller asserted 640x480). The frame's tensor is the authority when it has one -
    its header can lag the tensor it holds (see `Frame._pooled_numel`) - and the header is the
    fallback for frames that carry bytes only. Either way the payload is measured, so a mismatch is
    a named `ValueError` instead of a reshape error.

    Packed planar YUV payloads (the 8/10-bit SDR pipes) are refused by name: they are not RGB24/RGB48
    and resizing them as if they were would emit a scrambled picture.
    """
    if isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"{path}: a raw {type(frame).__name__} payload has no geometry; pass a Frame or call "
            f"resize_image_bytes() with the source width/height"
        )
    if getattr(frame, "yuv8_output", False) or getattr(frame, "yuv10_output", False):
        raise ValueError(
            f"{path}: this frame is emitted as packed planar YUV (yuv8_output/yuv10_output), not "
            f"RGB24/RGB48; the override resize cannot re-scale it - disable override_upscale_scale "
            f"or render at the model's own scale"
        )
    payload = frame.get_frame_bytes()
    width, height = frame_source_geometry(frame)
    return resize_image_bytes(payload, width, height, target_width, target_height, path=path)


def frame_source_geometry(frame) -> tuple[int, int]:
    """The `(width, height)` the frame's payload really has, as far as the frame can tell.

    The tensor/numpy array wins over `frame.width`/`frame.height` because an upscaled frame's header
    can lag the tensor it holds (`Frame._pooled_numel` says so and reads the tensor for the same
    reason: the override resize was handed a 320x240 payload by a frame whose header said 640x480).
    """
    for attribute in ("_tensor", "_np"):
        value = getattr(frame, attribute, None)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 3:
            return int(shape[-1]), int(shape[-2])
    return int(getattr(frame, "width", 0) or 0), int(getattr(frame, "height", 0) or 0)


def payload_geometry(payload_size: int, width: int, height: int,
                     path: str = "resize_image_bytes") -> tuple[int, int, int]:
    """Validate a byte payload against the geometry its caller declared.

    Returns `(width, height, bytes_per_pixel)` where bytes-per-pixel is 3 (RGB24/SDR) or 6
    (RGB48/HDR). Raises `ValueError` naming the declared geometry, the payload's own size and - when
    the size is a clean multiple of a plausible image - the geometry it actually is, plus `path`.

    cv2-free and numpy-free on purpose: it is the decision half of `resize_image_bytes`, so it is
    testable on an interpreter without cv2 (the backend's own interpreter has it; the harness
    interpreter does not).
    """
    size = int(payload_size)
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"{path}: the declared geometry {width}x{height} is empty, so a {size}-byte payload "
            f"cannot be checked against it"
        )
    if size <= 0:
        raise ValueError(
            f"{path}: the payload is {size} bytes for a declared {width}x{height} image"
        )
    pixels = width * height
    if size % pixels == 0 and size // pixels in _RGB_BYTES_PER_PIXEL:
        return width, height, size // pixels
    raise ValueError(
        f"{path}: the payload does not match the declared geometry {width}x{height} "
        f"({pixels} pixels, i.e. {pixels * 3} bytes at RGB24 or {pixels * 6} at RGB48); "
        f"the payload is {size} bytes{_implied_geometry(size, width, height)}"
    )


def _implied_geometry(size: int, width: int, height: int) -> str:
    """`" (its size is 320x240 at 3 bytes per pixel)"`, or "" when no whole image fits."""
    for bytes_per_pixel in _RGB_BYTES_PER_PIXEL:
        if size % bytes_per_pixel:
            continue
        factor = (size // bytes_per_pixel) / float(width * height)
        side = math.sqrt(factor)
        implied_w = int(round(width * side))
        implied_h = int(round(height * side))
        if implied_w > 0 and implied_h > 0 and implied_w * implied_h * bytes_per_pixel == size:
            return (f" (its size is {implied_w}x{implied_h} at {bytes_per_pixel} bytes per pixel"
                    f" - the frame's own geometry, not the declared one)")
    return " (its size is not a whole image at 3 or 6 bytes per pixel)"

def padFrame(
    frame_bytes: bytes,
    to_width: int,
    to_height: int,
    from_width: int,
    from_height: int,
) -> bytes:
    R = 52
    G = 59
    B = 71
    """
        Pads the frame to the target resolution.
        
        Args:
            frame_bytes (bytes): The input frame in bytes.
            target_width (int): The target width for padding.
            target_height (int): The target height for padding.
        
        Returns:
            bytes: The padded frame in bytes.
        """
    # Convert bytes to numpy array
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_array = frame_array.reshape((from_height, from_width, 3))

    padded_frame = np.full((to_height, to_width, 3), (R, G, B), dtype=np.uint8)

    # Calculate padding offsets
    y_offset = (to_height - from_height) // 2
    x_offset = (to_width - from_width) // 2

    # Place the original frame in the center of the padded frame
    padded_frame[
        y_offset : y_offset + from_height,
        x_offset : x_offset + from_width,
    ] = frame_array

    # Convert the padded frame back to bytes
    return padded_frame.tobytes()

class subprocess_popen_without_terminal(subprocess.Popen):
    """
    A class that allows you to run a subprocess without opening a terminal window.
    """
    def __init__(self, *args, **kwargs):
        if PLATFORM == "win32":
                kwargs["startupinfo"] = subprocess.STARTUPINFO()
                kwargs["startupinfo"].dwFlags |= subprocess.STARTF_USESHOWWINDOW
        super().__init__(*args, **kwargs)

class CudaChecker:
    def __init__(self):
        self.HAS_SYSTEM_CUDA = self.checkForCUDA()
        self.HAS_PYTORCH_CUDA = self.checkForCUDAPytorch()
    @staticmethod
    def checkForCUDA() -> bool:
        try:
            import torch
            import torchvision
            import cupy

            if cupy.cuda.get_cuda_path() == None:
                return False
        except Exception as e:
            return False
        return True

    @staticmethod
    def checkForCUDAPytorch() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False