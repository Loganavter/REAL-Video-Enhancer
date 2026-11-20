import os
import sys
import requests
import platform
from PySide6.QtCore import QDir
def networkCheck(hostname="https://raw.githubusercontent.com") -> bool:
    """
    checks network availability against a url, default url: raw.githubusercontent.com
    """
    try:
        _ = requests.head(hostname, timeout=1)
        return True
    except Exception as e:
        pass
    return False
HAS_NETWORK_ON_STARTUP = networkCheck()

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
PAUSED_STATE_SHARED_MEMORY_ID = "/paused_state" + str(os.getpid())
INPUT_TEXT_FILE = os.path.join(CWD, f"INPUT{os.getpid()}.txt")
if (
    "--swap-flatpak-checks" in sys.argv
):  # swap check down here as to not interfere with directories
    IS_FLATPAK = not IS_FLATPAK
