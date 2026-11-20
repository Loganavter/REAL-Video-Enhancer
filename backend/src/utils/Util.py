import os
import sys
import subprocess
import contextlib
import logging
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


def resize_image_bytes(image_bytes: bytes, width: int, height: int, target_width: int, target_height: int) -> bytes:
    """
    Resizes the image to the target resolution.
    
    Args:
        image_bytes (bytes): The input image in bytes.
        target_width (int): The target width for resizing.
        target_height (int): The target height for resizing.
    
    Returns:
        bytes: The resized image in bytes.
    """
    if target_width == width and target_height == height:
        return image_bytes
    channels = len(bytes(image_bytes)) / (height * width) # 3 if RGB24/SDR, 6 if RGB48/HDR
    dtype = np.uint8 if channels == 3  else np.uint16
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