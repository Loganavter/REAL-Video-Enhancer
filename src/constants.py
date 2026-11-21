import os
import sys
import requests
import platform
from PySide6.QtCore import QDir
def networkCheck(hostname="https://raw.githubusercontent.com", timeout: float = 1.0) -> bool:
    """Is there a usable network right now? A TCP connect, not an HTTP request.

    This runs at IMPORT time (`HAS_NETWORK_ON_STARTUP`), so its cost is paid on every launch before the window exists.
    It used to be `requests.head(url, timeout=1)`: measured 2026-11-22 at **5.3 s** on this host (requests' timeout is
    per socket operation, so DNS + TCP + TLS + a proxy handshake each get their own budget, and a flaky link stacks
    them). A plain connect to the same host answers the only question the flag asks - "is a connection possible" - in
    at most `timeout`.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(hostname if "://" in hostname else f"//{hostname}")
    host = parsed.hostname or "raw.githubusercontent.com"
    port = parsed.port or (443 if parsed.scheme in ("https", "") else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
_NETWORK_CACHE: bool | None = None


def has_network() -> bool:
    """Is a network connection possible? Asked WHEN IT IS NEEDED, never at import.

    This used to run at import (`HAS_NETWORK_ON_STARTUP = networkCheck()`), and the DNS lookup alone measured ~5 s on
    this host - five seconds of a launch with no window, spent on a question only the updater and one confirmation
    dialog ask. Cached after the first real answer; a caller that needs a fresh answer calls `networkCheck()`.
    """
    global _NETWORK_CACHE
    if _NETWORK_CACHE is None:
        _NETWORK_CACHE = networkCheck()
    return _NETWORK_CACHE

LOCKFILE = QDir.tempPath() + "/REAL-Video-Enhancer.lock"



PLATFORM = sys.platform  # win32, darwin, linux
IS_STEAM = "SteamAppId" in os.environ
IS_FLATPAK = "FLATPAK_ID" in os.environ and not IS_STEAM

HOME_PATH = os.path.expanduser("~")

IS_COMPILED_OR_FROZEN = hasattr(sys, "frozen")

# The backend is looked up next to the launcher first, and only then next to the shell's cwd. The old rule was
# cwd-only, so `python3 /path/to/REAL-Video-Enhancer.py` from any other directory fell through to the installed
# copy in ~/.local/share and silently ran a different backend than the one in the tree (found 2026-11-20: the
# repository's updated backend was ignored, and with it the reuse correction).
APP_DIR = (
    os.path.dirname(os.path.abspath(sys.executable)) if IS_COMPILED_OR_FROZEN
    else os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
)


def _find_backend(root: str):
    """A directory is a backend when it carries the entry point, not merely when it is called `backend`."""
    candidate = os.path.join(root, "backend")
    if os.path.isfile(os.path.join(candidate, "rve-backend.py")):
        return candidate
    return None


_LOCAL_BACKEND = _find_backend(APP_DIR) or _find_backend(os.getcwd())
USE_LOCAL_BACKEND = _LOCAL_BACKEND is not None

if not USE_LOCAL_BACKEND:
    
    if PLATFORM == "win32":
        CWD = os.path.join(HOME_PATH, "AppData", "Local", "REAL-Video-Enhancer")
    if PLATFORM == "darwin":
        CWD = os.path.join(HOME_PATH, "Library", "REAL-Video-Enhancer")
    if PLATFORM == "linux":
        CWD = os.path.join(
            HOME_PATH, ".local", "share", "REAL-Video-Enhancer"
        )
else:
    CWD = os.path.dirname(_LOCAL_BACKEND)
    
if IS_FLATPAK:
    CWD = (
        os.path.join(
            HOME_PATH, ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
        )
    )

CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()


EXE_NAME = "REAL-Video-Enhancer.exe" if PLATFORM == "win32" else "REAL-Video-Enhancer"
LIBS_NAME = "_internal" if PLATFORM == "win32" else "lib"
# dirs
MODELS_PATH = os.path.join(CWD, "models")
CUSTOM_MODELS_PATH = os.path.join(CWD, "custom_models")
VIDEOS_PATH = (
    os.path.join(HOME_PATH, "Desktop")
    if PLATFORM == "darwin"
    else os.path.join(HOME_PATH, "Videos")
)
BACKEND_PATH = "/app/bin/backend" if IS_FLATPAK else os.path.join(CWD, "backend")
TEMP_DOWNLOAD_PATH = os.path.join(CWD, "temp")
# exes
FFMPEG_PATH = (
    os.path.join(CWD, "bin", "ffmpeg.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "bin", "ffmpeg")
)
PYTHON_DIRECTORY = os.path.join(CWD, "python")

PYTHON_EXECUTABLE_PATH = (
    os.path.join(CWD, "python", "python", "python.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "python", "python", "bin", "python3")
)
#PYTHON_VERSION = "3.13.2" if PLATFORM != "darwin" else "3.12.9" # sets python version of backend
PYTHON_VERSION = "3.12.9"

EXE_PATH = os.path.join(
    CWD,
    EXE_NAME,
)
LIBS_PATH = os.path.join(
    CWD,
    LIBS_NAME,
)

# is installed
IS_INSTALLED = os.path.isfile(FFMPEG_PATH) and os.path.isfile(PYTHON_EXECUTABLE_PATH)

IMAGE_SHARED_MEMORY_ID = "/image_preview" + str(os.getpid())
# The preview BOX: the largest thumbnail either side may use, not the aspect ratio. The
# backend sizes each thumbnail to the frame it has (long side 640, both axes sampled at the
# same rate), and says the actual w/h in the header, so a 4:3 source is not cropped and not
# stretched - a fixed 640x360 did both. Two slots of 640x640 RGB = 2.5 MB, against a 4K
# frame's 24.9 MB to copy.
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 640
PAUSED_STATE_SHARED_MEMORY_ID = "/paused_state" + str(os.getpid())
INPUT_TEXT_FILE = os.path.join(CWD, f"INPUT{os.getpid()}.txt")
if (
    "--swap-flatpak-checks" in sys.argv
):  # swap check down here as to not interfere with directories
    IS_FLATPAK = not IS_FLATPAK
