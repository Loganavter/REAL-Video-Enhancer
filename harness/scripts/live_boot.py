"""Boot one real RVE window for live testing (`harness/scripts/live.sh up` calls this).

Why not the app entry point: `REAL-Video-Enhancer.py` probes backends, may open install dialogs and starts the updater -
all of which is noise for "does this widget paint". This builds the same `MainWindow` the window users see (the shell,
the pages, the always-on control channel) and runs the event loop, nothing else.

The environment that makes it work on a headless host is in `live.sh`: xcb + Xvfb + `LIBGL_ALWAYS_SOFTWARE=1` (QRhi's
OpenGL backend needs a real GL context, and llvmpipe provides one), with the GTK appmenu module and the platform theme
emptied (the module aborts the process under xcb on this host).
"""

from __future__ import annotations

import importlib.util
import os
import sys


def main(argv: list[str]) -> int:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "src"))

    from PySide6.QtWidgets import QApplication

    app = QApplication(argv)
    from src.ui import toolkit_boot, toolkit_theme

    toolkit_boot.ensure()
    toolkit_theme.install(app)

    spec = importlib.util.spec_from_file_location("rve_app", os.path.join(repo, "REAL-Video-Enhancer.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    window = module.MainWindow()
    width = int(os.environ.get("RVE_LIVE_WIDTH", "1163"))
    height = int(os.environ.get("RVE_LIVE_HEIGHT", "900"))
    window.resize(width, height)
    window.show()
    print(f"LIVE window {width}x{height} shown on {os.environ.get('DISPLAY', '?')}", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
