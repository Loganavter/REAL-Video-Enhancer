"""Make the suite runnable from any directory: the repo root, the backend package, and the harness.

The tests import three roots: ``harness.*`` (from the repository root), ``backend.src.*`` (from the
root) and ``src.*`` (the backend package itself, i.e. ``backend/`` on the path). Before this file
existed each invocation carried its own ``PYTHONPATH``, which is why the same suite was green in one
shell and an ImportError in another.

It also owns the two guards that would otherwise need plugins this interpreter does not have
(plan-rve-transfer-blinktable-2026-11-21 §3, correction 1):

* **process-wide Qt state** - ``QT_QPA_PLATFORM`` is pinned to ``offscreen`` before any test imports
  Qt, a session ``qapp`` fixture hands out the one ``QApplication``, and every test's teardown sweeps
  the top-level widgets it left behind and flushes their posted ``DeferredDelete`` events (offscreen
  never runs deferred deletes on its own). Qt is imported **inside** the fixtures, never at module
  scope, so a cell whose interpreter has no PySide6 skips instead of failing collection.
* **hangs** - an autouse fixture arms ``faulthandler.dump_traceback_later()`` for ``RVE_TEST_HANG_S``
  seconds (default 300) and cancels it in teardown. That is the stdlib stand-in for
  ``pytest-timeout``: it only prints every thread's stack, it never fails or aborts a test. The dump
  goes to a dup of the real stderr, because pytest's default ``--capture=fd`` would otherwise swallow
  it into the per-test buffer the moment the test finishes.
"""
import faulthandler
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _path in (_ROOT, os.path.join(_ROOT, "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Before anything imports Qt: an unset platform would make a headless run try (and fail) to reach a
# real display. `setdefault` keeps a caller's explicit choice (e.g. a private Xvfb for a capture).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# `src/Util.py` imports py-cpuinfo, which the app's venv carries and this interpreter does not: without a stub,
# `from src.ui.SettingsTab import ...` is a collection ERROR rather than a test. Only `brand_raw` is ever read.
if "cpuinfo" not in sys.modules:
    import types as _types

    _cpuinfo = _types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda: {"brand_raw": "unknown"}
    sys.modules["cpuinfo"] = _cpuinfo

#: ``faulthandler.dump_traceback_later`` default, overridable so the hang probe can use 1 s.
HANG_SECONDS = 300


def _hang_fd(config):
    """A dup of the real stderr, so the dump is visible even though pytest captures fd 2.

    ``--capture=fd`` (pytest's default) points fd 2 at a per-test temp file, so a dump written to
    ``sys.stderr`` is swallowed when the test later passes - measured on this repo: the guard fired,
    the stack went into the capture buffer, and the run printed nothing. Suspending capture for the
    single ``dup`` gives a descriptor that keeps pointing at the terminal, so the stack shows up
    live. ``None`` means "no usable descriptor": faulthandler then falls back to fd 2.
    """
    global _HANG_FD
    if _HANG_FD is None:
        capman = config.pluginmanager.getplugin("capturemanager")
        disabled = getattr(capman, "global_and_fixture_disabled", None)
        try:
            if disabled is None:
                _HANG_FD = os.dup(2)
            else:
                with disabled():
                    _HANG_FD = os.dup(2)
        except (AttributeError, OSError, ValueError):
            _HANG_FD = -1
    return None if _HANG_FD == -1 else _HANG_FD


_HANG_FD = None


def _hang_seconds() -> float:
    """``RVE_TEST_HANG_S`` in seconds, falling back to the default on a value faulthandler would refuse."""
    try:
        return max(1.0, float(os.environ.get("RVE_TEST_HANG_S", HANG_SECONDS)))
    except ValueError:
        return float(HANG_SECONDS)


def _qt_widgets():
    """The QtWidgets module, or ``None`` when this interpreter has no PySide6 - never an import error."""
    try:
        from PySide6 import QtWidgets
    except Exception:  # ImportError, but a broken/absent plugin platform is the same verdict here
        return None
    return QtWidgets


def pytest_configure(config):
    """Name the interpreter pair whose Qt build corrupts the GC - before a run dies with no explanation.

    Measured 2026-11-24 with a forced `gc.collect()` after every test: CPython 3.12 + PySide6 6.8.0.2 (the REAL-2
    bundled interpreter) segfaults inside CPython's own collector (`move_unreachable` -> `visit_reachable` ->
    `_PyObject_IS_GC` on a tuple holding a stack address) at 3 of 3 runs, while CPython 3.14 + PySide6 6.12 (the system
    interpreter) runs the same suite clean at 2 of 2. Nothing in this tree is implicated - but a suite that dies at GC
    is unreadable, and the failure looks like a flake. Run the Qt suite under the system interpreter and the backend
    suite under the bundled one (it carries torch); this warning exists so nobody has to rediscover that.
    """
    import sys

    try:
        import PySide6
    except ImportError:
        return
    if sys.version_info[:2] == (3, 12) and PySide6.__version__ == "6.8.0.2":
        print(
            "\nWARNING conftest: Qt 6.8.0.2 on CPython 3.12 segfaults inside CPython's GC (measured 3/3 runs). "
            "Run this suite under the system python3 (Qt 6.12) and keep the bundled interpreter for backend/tests.\n",
            file=sys.stderr, flush=True)


@pytest.fixture(scope="session")
def qapp():
    """One ``QApplication`` for the session, built on demand.

    Skipped - not failed - where PySide6 is missing: the toolkit-free half of this suite runs on an
    interpreter without it, and only the tests that ask for ``qapp`` need the widgets module.
    """
    QtWidgets = _qt_widgets()
    if QtWidgets is None:
        pytest.skip("PySide6 is not installed for this interpreter")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _qt_toplevel_sweep():
    """Delete the top-level widgets a test leaves behind and flush their deferred deletes.

    Offscreen Qt never processes ``DeferredDelete`` on its own, so a widget that a test called
    ``deleteLater()`` on (or never closed) stays alive past the test and leaks into the next one -
    the classic "green alone, red in the suite" shape. Only runs where Qt is importable, and only
    at teardown, so it changes neither collection nor ordering.
    """
    yield
    QtWidgets = _qt_widgets()
    if QtWidgets is None:
        return
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    from PySide6.QtCore import QEvent

    for widget in app.topLevelWidgets():
        try:
            widget.deleteLater()
        except RuntimeError:  # already deleted by its owner; nothing left to sweep
            pass
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture(autouse=True)
def _hang_guard(request):
    """Print the stack of every thread when a test overruns; never fail it.

    ``repeat=False`` means one dump, then the run continues: this is a diagnostic for a wedged
    run, not a timeout that kills the test (there is no ``pytest-timeout`` here to do that).
    """
    seconds = _hang_seconds()
    fd = _hang_fd(request.config)
    if fd is None:
        faulthandler.dump_traceback_later(seconds, repeat=False)
    else:
        faulthandler.dump_traceback_later(seconds, repeat=False, file=fd)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
