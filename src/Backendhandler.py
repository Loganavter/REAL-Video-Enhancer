import json
import os

from .constants import BACKEND_PATH, PYTHON_EXECUTABLE_PATH, PYTHON_DIRECTORY, PLATFORM, IS_INSTALLED, IS_FLATPAK, CWD
from .Util import (
    FileHandler
)
from .version import version


class BackendHandler:
    def __init__(self, parent, settings=None):
        self.parent = parent
        self.settings = settings

    def getAvailableBackends(self):
        """Ask the backend which backends it has - with a plain subprocess, NOT the setup dialog.

        This runs during startup. It used to open `SettingUpBackendPopup`, a modal dialog with an icon, a label and a
        worker thread, for a command that answers in 0.12 s: measured 2026-11-22, the dialog cost 3.08 s of window time
        (and flashed a "Setting Up Backend" window on every launch). The dialog stays for the real setup, where there is
        something to watch.
        """
        import json
        import subprocess

        from .ui.QTcustom import TextOutputPopup

        # NOT under TEMP_DOWNLOAD_PATH: the app wipes that folder on startup (REAL-Video-Enhancer.py), which made the
        # cache cold on every launch - measured as a 3.3 s window for a list that had not changed.
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "REAL-Video-Enhancer")
        cache_path = os.path.join(cache_dir, "backends.json")
        stamp = self._backend_stamp()
        cached = self._read_backends_cache(cache_path, stamp)
        if cached is not None:
            return cached

        process = subprocess.run(
            [
                PYTHON_EXECUTABLE_PATH,
                "-W",
                "ignore",
                os.path.join(BACKEND_PATH, "rve-backend.py"),
                "--list_backends",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return_code = str(process.returncode).strip()
        output: str = (process.stdout or "") + (process.stderr or "")
        if "ERROR" in output or "TRACEBACK" in output or return_code == "1":
            TextOutputPopup(f"ERROR DETECTED IN BACKEND SETUP!\n{output}", title="FATAL ERROR")
            exit(1)
        
        output = output.split(" ")
        # hack to filter out bad find
        new_out = ""
        for word in output:
            if "objc" in word:
                continue
            if "[Torch-TensorRT]" in word:
                continue
            new_out += word + " "

        # Find the part of the output containing the backends list
        output = new_out
        start = output.find("[")
        end = output.find("]") + 1
        backends_str = output[start:end]

        # Convert the string representation of the list to an actual list
        try:
            backends = eval(backends_str)
        except Exception:
            backends = []

        self._write_backends_cache(cache_path, stamp, backends, output)
        return backends, output

    @staticmethod
    def _backend_stamp() -> str:
        """What the cached backend list depends on: the entry point and the interpreter it runs under."""
        parts = []
        for path in (os.path.join(BACKEND_PATH, "rve-backend.py"), PYTHON_EXECUTABLE_PATH):
            try:
                stat = os.stat(path)
                parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append("missing")
        return "|".join(parts)

    @staticmethod
    def _read_backends_cache(path: str, stamp: str):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("stamp") == stamp and isinstance(data.get("backends"), list):
                return data["backends"], data.get("output", "")
        except (OSError, ValueError):
            pass
        return None

    @staticmethod
    def _write_backends_cache(path: str, stamp: str, backends: list, output: str) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"stamp": stamp, "backends": backends, "output": output}, handle)
        except OSError:
            pass
