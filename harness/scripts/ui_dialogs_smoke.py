"""Construct every W5 popup once, offscreen, and report its modality.

The seven popups in ``src/ui/QTcustom.py`` block their caller by design: the download window calls ``exec()`` and the
command windows keep a ``QThread`` alive while it runs. That makes them impossible to "just construct" in a test
without entering an event loop or starting a real download, which is why every migrated class accepts the keyword-only
``blocking=False``: it builds the same toolkit window and skips ``exec()``/the worker thread. The window's modality is
set in the constructor, so it is still readable with ``isModal()`` - that is what this script asserts for the six
modal popups, and that ``NotificationOverlay`` (a toast, not a dialog) reports itself non-modal.

Run it from the repository root:

    PYTHONPATH=<toolkit>/src python3 harness/scripts/ui_dialogs_smoke.py

No pages or shell are imported: the pages need the toolkit at build time and the app's window; this script only needs
``src.ui.QTcustom`` and a ``QApplication``.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The supervisor may run this without a display: the toolkit painters and Qt widgets both work offscreen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# `src.Util` imports py-cpuinfo at module level for an unrelated feature; the smoke test is not the place to install a
# dependency for it. Stub only when the dependency is genuinely absent (the app venv has it).
try:  # pragma: no cover - environment dependent
    import cpuinfo  # noqa: F401
except Exception:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda: {}
    sys.modules["cpuinfo"] = _cpuinfo

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from src.ui.QTcustom import (  # noqa: E402
    DisplayCommandOutputPopup,
    DownloadProgressPopup,
    IndependentQTPopup,
    NotificationOverlay,
    RegularQTPopup,
    SettingUpBackendPopup,
    TextOutputPopup,
)

_DOWNLOAD_TARGET = ROOT / "smoke-download.bin"


def _check(name: str, widget, *, modal: bool) -> None:
    assert widget is not None, f"{name} constructed as None"
    is_modal = bool(getattr(widget, "isModal", lambda: False)())
    assert is_modal is modal, f"{name}: expected modal={modal}, got {is_modal}"
    print(f"OK {name}: modal={is_modal}")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    # The toolkit's painters read named tokens. Without the app's palette registration they fall back to their own
    # defaults and log "unknown theme token" for every one; installing RVE's token table keeps the smoke faithful to
    # what the app shows. It is not a requirement of the construction check, so a failure is reported, not raised.
    try:
        from src.ui import toolkit_theme

        if toolkit_theme.enabled():
            toolkit_theme.install(app)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"note: toolkit theme not installed ({exc}); using toolkit defaults")

    host = QWidget()
    host.resize(800, 600)
    host.show()

    built = []
    try:
        overlay = NotificationOverlay("smoke: notification", host, timeout=1500)
        _check("NotificationOverlay", overlay, modal=False)
        built.append(overlay)

        download = DownloadProgressPopup(
            link="https://example.invalid/smoke",
            downloadLocation=str(_DOWNLOAD_TARGET),
            title="Smoke download",
            blocking=False,
        )
        _check("DownloadProgressPopup", download, modal=True)
        built.append(download)

        setup = SettingUpBackendPopup(
            command=[sys.executable, "-c", "print('smoke')"],
            blocking=False,
        )
        _check("SettingUpBackendPopup", setup, modal=True)
        built.append(setup)

        command_output = DisplayCommandOutputPopup(
            command=[sys.executable, "-c", "print('smoke')"],
            progressBarLength=1,
            blocking=False,
        )
        _check("DisplayCommandOutputPopup", command_output, modal=True)
        built.append(command_output)

        text_output = TextOutputPopup(
            "smoke output\n\x1b[31mred line\x1b[0m",
            title="Smoke output",
            blocking=False,
        )
        _check("TextOutputPopup", text_output, modal=True)
        built.append(text_output)

        regular = RegularQTPopup("smoke message", blocking=False)
        _check("RegularQTPopup", regular, modal=True)
        built.append(regular)

        independent = IndependentQTPopup("smoke independent message", blocking=False)
        _check("IndependentQTPopup", independent, modal=True)
        built.append(independent)
    finally:
        for widget in built:
            try:
                widget.close()
            except Exception:
                pass
        app.processEvents()
        try:
            if _DOWNLOAD_TARGET.exists():
                _DOWNLOAD_TARGET.unlink()
        except OSError:
            pass
        host.close()

    print("smoke: all seven popups construct and report the expected modality")
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
