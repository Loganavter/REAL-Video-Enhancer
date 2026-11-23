"""The window's startup IO: what each use case does, and what it does when its instrument fails.

P3 of the seam plan moved the launch-time work out of `MainWindow.__init__` into `src/ui/use_cases/`. The point of the
move is testability, so these tests are the contract: every function is driven with a fake instrument, and the failure
path is pinned per function - because the one thing none of them may do is swallow a failure the window used to stop
on. There is no Qt and no network here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.use_cases import startup
from src.ui.use_cases.models import ensure_models


# --- the temp directory -----------------------------------------------------------------------------------

def test_the_temp_directory_is_created_through_the_injected_creator(tmp_path):
    created = []
    target = str(tmp_path / "temp_download")
    assert startup.prepare_temp_directory(target, created.append) == target
    assert created == [target]


def test_a_directory_that_cannot_be_created_propagates():
    """Creating the download folder failing is fatal - it did not become a warning in the move."""
    def failing(_path):
        raise PermissionError("read-only volume")

    with pytest.raises(PermissionError):
        startup.prepare_temp_directory("/somewhere/temp", failing)


# --- dependency downloads and updates ---------------------------------------------------------------------

class Recording:
    def __init__(self, update_available=False, download_error=None, update_error=None):
        self.update_available = update_available
        self.download_error = download_error
        self.update_error = update_error
        self.calls = []

    def download(self):
        self.calls.append("download")
        if self.download_error is not None:
            raise self.download_error

    def get_if_update_available(self):
        self.calls.append("asked")
        return self.update_available

    def update_if_updates_available(self):
        self.calls.append("updated")
        if self.update_error is not None:
            raise self.update_error


def _types(*instances):
    """Dependency classes stand-in: each call of the class hands back the next recorded instance."""
    created = []

    def make(instance):
        class Fake:
            def __new__(cls):
                created.append(instance)
                return instance
        Fake.__name__ = type(instance).__name__
        return Fake

    classes = [make(instance) for instance in instances]
    return classes, created


def test_every_dependency_downloads_in_definition_order():
    first = Recording()
    second = Recording()
    classes, created = _types(first, second)
    assert startup.download_missing_dependencies(classes) == ["Recording", "Recording"]
    assert created == [first, second]
    assert first.calls == ["download"]
    assert second.calls == ["download"]


def test_a_download_that_raises_stops_the_launch():
    """The loop had no try around it; an install that fails must not be reported as done."""
    first = Recording(download_error=OSError("connection reset"))
    second = Recording()
    classes, _ = _types(first, second)
    with pytest.raises(OSError):
        startup.download_missing_dependencies(classes)
    assert second.calls == []


def test_only_dependencies_that_offer_an_update_are_updated():
    current = Recording(update_available=False)
    stale = Recording(update_available=True)
    classes, _ = _types(current, stale)
    assert startup.update_available_dependencies(classes) == ["Recording"]
    assert current.calls == ["asked"]
    assert stale.calls == ["asked", "updated"]


def test_an_update_check_that_raises_propagates():
    broken = Recording(update_error=RuntimeError("no version metadata"))
    broken.update_available = True
    classes, _ = _types(broken)
    with pytest.raises(RuntimeError):
        startup.update_available_dependencies(classes)


# --- the backend probe and its report ---------------------------------------------------------------------

class Handler:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.asked = 0

    def getAvailableBackends(self):
        self.asked += 1
        if self.error is not None:
            raise self.error
        return self.answer


def test_the_backend_list_and_report_come_back_unchanged():
    handler = Handler(answer=(["ncnn Vulkan"], "listing... [ncnn Vulkan]"))
    assert startup.available_backends(handler) == (["ncnn Vulkan"], "listing... [ncnn Vulkan]")
    assert handler.asked == 1


def test_a_probe_that_fails_propagates():
    handler = Handler(error=TimeoutError("backend did not answer in 60 s"))
    with pytest.raises(TimeoutError):
        startup.available_backends(handler)


def test_the_report_counts_gpus_and_reads_the_half_precision_flag():
    """Counts as the pre-move loop counted: the counter starts at -1 and is clamped at zero.

    That sentinel is kept verbatim (P3 moves code, it does not fix it): with one matching line per device, N lines
    read as N-1. The test states it so the number the settings page uses is a known quantity rather than a surprise -
    fixing it is a separate, measured change to the GPU-id ranges.
    """
    report = startup.parse_backend_report(
        "System Information:\n"
        "Half precision support: True\n"
        "NCNN GPU 0\n"
        "NCNN GPU 1\n"
        "NCNN GPU 2\n"
        "Pytorch GPU 0\n"
        "Pytorch GPU 1\n"
    )
    assert report == startup.BackendReport(ncnn_gpus=2, pytorch_gpus=1, half_precision=True)


def test_an_empty_report_is_zero_devices_and_no_half_precision():
    """A backend that failed before printing anything must read as "no GPUs", not as a crash or as -1."""
    assert startup.parse_backend_report("") == startup.BackendReport(0, 0, False)


def test_unrelated_lines_are_not_gpus():
    report = startup.parse_backend_report(
        "Half precision support: False\n"
        "pytorch gpu 0\n"
        "something else entirely\n"
    )
    assert report == startup.BackendReport(ncnn_gpus=0, pytorch_gpus=0, half_precision=False)


# --- the model download -----------------------------------------------------------------------------------

class Downloader:
    def __init__(self, modelFile, downloadModelFile, answer=True, error=None):
        self.modelFile = modelFile
        self.downloadModelFile = downloadModelFile
        self.answer = answer
        self.error = error

    def downloadModel(self):
        if self.error is not None:
            raise self.error
        return self.answer


def test_present_models_answer_true_without_downloading():
    asked = []

    def factory(**kwargs):
        asked.append(kwargs)
        return Downloader(answer=True, **kwargs)

    assert ensure_models([("a.pth", "a-dl.pth"), ("b.pth", "b-dl.pth")], factory) is True
    assert asked == [
        {"modelFile": "a.pth", "downloadModelFile": "a-dl.pth"},
        {"modelFile": "b.pth", "downloadModelFile": "b-dl.pth"},
    ]


def test_a_model_that_cannot_be_downloaded_stops_the_add():
    """False means "stop, say so" in the window; the second pair must not be attempted after it."""
    attempted = []

    def factory(**kwargs):
        attempted.append(kwargs["modelFile"])
        return Downloader(answer=kwargs["modelFile"] != "missing.pth", **kwargs)

    assert ensure_models([("ok.pth", "ok-dl.pth"), ("missing.pth", "m-dl.pth"), ("later.pth", "l-dl.pth")], factory) is False
    assert attempted == ["ok.pth", "missing.pth"]


def test_an_empty_model_list_is_satisfied():
    assert ensure_models([], Downloader) is True


def test_a_downloader_that_raises_propagates():
    def factory(**kwargs):
        return Downloader(error=OSError("disk full"), **kwargs)

    with pytest.raises(OSError):
        ensure_models([("a.pth", "a-dl.pth")], factory)
