"""The window's startup IO, as functions that take their instruments instead of owning a window.

P3 of the seam plan moved the launch-time work out of `MainWindow.__init__` (`REAL-Video-Enhancer.py`) into this
package. What stays in the window is policy and presentation: which branch runs when no backend is installed, what a
dialog says, which widgets get disabled. What moves here is the IO - directories, dependency downloads and update
checks (the ffmpeg/ffprobe binary is one of those dependencies), the backend probe, and the report it prints - so each
can be exercised with a fake instrument, without Qt, a display, or a network.

Two rules the functions keep:

* **no swallowing.** None of these adds a `try` the original code did not have: an instrument that raises still
  raises here, because that is how a failed dependency download behaved yesterday (the window never got past it).
* **no hidden instruments.** A caller passes the object that does the work (`create_directory`, the dependency
  classes, the backend handler, the model downloader). The app-level defaults are resolved lazily, inside the
  function, and only when the caller left them out - `src/DownloadDeps` needs PySide6 and the app's dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class BackendReport:
    """What the backend's `--list_backends` answer said about the machine.

    The two GPU fields are COUNTS - how many devices of that backend exist - not the highest id. The settings
    field they feed is a maximum id, so the page derives `count - 1` itself; keeping a count here means "no GPU of
    this backend" (0) and "the only GPU" (1) can never collapse into the same number again.
    """

    ncnn_gpu_count: int
    pytorch_gpu_count: int
    half_precision: bool


def prepare_temp_directory(path: str, create_directory: Callable[[str], object]) -> str:
    """Create the scratch directory the app downloads into, and return it.

    `FileHandler.createDirectory` only creates one level and does nothing when it already exists, which is exactly
    what startup wants: the folder is wiped on launch, never re-created recursively.
    """
    create_directory(path)
    return path


def download_missing_dependencies(dependency_types: Iterable[type] | None = None) -> list[str]:
    """Instantiate each dependency and let it download itself; the class names, in the order tried.

    The list is what makes a launch legible in the log: four classes today (backend, python, ffmpeg, the VC runtime),
    each of which may download a binary. A downloader that raises propagates - installing a broken dependency is not a
    condition to continue past.
    """
    types = tuple(dependency_types) if dependency_types is not None else _dependency_types()
    names: list[str] = []
    for dependency_type in types:
        dependency = dependency_type()
        dependency.download()
        names.append(dependency_type.__name__)
    return names


def update_available_dependencies(dependency_types: Iterable[type] | None = None) -> list[str]:
    """Update every dependency that says an update is available; the class names actually updated.

    A dependency that does not update itself (`VCRedList.updatable = False`) is still asked and still answers; the
    question is the dependency's own, which is why this function does not carry a list.
    """
    types = tuple(dependency_types) if dependency_types is not None else _dependency_types()
    updated: list[str] = []
    for dependency_type in types:
        dependency = dependency_type()
        if dependency.get_if_update_available():
            dependency.update_if_updates_available()
            updated.append(dependency_type.__name__)
    return updated


def available_backends(handler) -> tuple[list[str], str]:
    """The backends the backend reports it can run, plus its full report text.

    `handler` is `src.BackendHandler.BackendHandler`: it owns the subprocess, the cache and the fatal-error popup. This
    function exists so the window says "probe now" without deciding how.
    """
    return handler.getAvailableBackends()


def parse_backend_report(full_output: str) -> BackendReport:
    """Count the devices and read the fp16 flag out of the probe's own text.

    The backend prints `Half precision support: True/False` and one `NCNN GPU n: name` / `PyTorch GPU n: name` line
    per device, so the number of matching lines IS the device count. Each counter starts at 0: the old loop started
    at -1 and clamped, which read one `PyTorch GPU 0` line as 0 - a count pretending to be a maximum id. That is
    exactly what made the GPU-id steppers inert on a one-GPU machine (the field's range became `0..0`).

    Anything else - an empty report, a backend that failed before printing, a line from another module - counts as
    zero devices.
    """
    ncnn_gpu_count = 0
    pytorch_gpu_count = 0
    half_precision = False
    for line in str(full_output).lower().split("\n"):
        if "half precision support:" in line:
            half_precision = "true" in line
        if "ncnn gpu " in line:  # this grabs every device line, whatever its index
            ncnn_gpu_count += 1
        if "pytorch gpu " in line:
            pytorch_gpu_count += 1
    return BackendReport(
        ncnn_gpu_count=ncnn_gpu_count,
        pytorch_gpu_count=pytorch_gpu_count,
        half_precision=half_precision,
    )


def _dependency_types() -> Sequence[type]:
    """The app's dependency classes, in definition order - the order the window used to iterate."""
    from ...DownloadDeps import Dependency

    return tuple(Dependency.__subclasses__())
