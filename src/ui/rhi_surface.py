"""The RHI surface setup, ported from BlinkTable (`ui/canvas_infra/rhi/rhi_backend.py`).

Two pieces, both because of one Qt behaviour:

* **`install_rhi_primer`** - `QRhiWidget::ensureRhi()` takes its QRhi from the *top-level window's own backingstore*,
  which Qt promotes from `RasterSurface` to an RHI-capable surface lazily, on the first real use by a `QRhiWidget`
  descendant. On a real Wayland session that lazy promotion loses a race against the first widget's own first
  paint/resize: Qt prints **`QRhiWidget: No QRhi`** repeatedly, emits `renderFailed()`, and the widget's content can
  stick at a stale size across resizes (BlinkTable measured exactly this; Telegram Desktop's `lib_ui`
  `GL::EnsureWindowRhi` fixes it the same way). The fix is a throwaway **shown** 1x1 `QRhiWidget` child of the
  top-level window, created as early as possible, so the promotion happens once, off the interaction path. It
  self-destructs on its first successful `initialize()`. Shown, not hidden: a hidden widget never gets paint events, so
  `ensureRhi()` would never run there - the earlier `hide()` version was a silent no-op.
* **`configure_rhi_widget`** - pick the backend explicitly (`RVE_UI_RHI_BACKEND=default|opengl|vulkan|d3d11|d3d12|
  metal|null`) instead of leaving it to the platform default, and log `renderFailed` with the remediation, so "the pane
  paints nothing" names its own cause.

Both mirror BlinkTable's module; the names differ only because RVE's env namespace does.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QRhiWidget, QWidget

#: Which backend to ask for. `default` leaves Qt's platform choice alone.
RHI_BACKEND_ENV = "RVE_UI_RHI_BACKEND"

_API_BY_NAME = {
    "default": None,
    "opengl": QRhiWidget.Api.OpenGL,
    "vulkan": QRhiWidget.Api.Vulkan,
    "d3d11": QRhiWidget.Api.Direct3D11,
    "d3d12": QRhiWidget.Api.Direct3D12,
    "metal": QRhiWidget.Api.Metal,
    "null": QRhiWidget.Api.Null,
}


def requested_backend() -> str:
    """The requested backend name, lower-cased; unknown names fall back to `default`."""
    name = (os.environ.get(RHI_BACKEND_ENV, "default") or "default").strip().lower()
    return name if name in _API_BY_NAME else "default"


def configure_rhi_widget(widget: QRhiWidget) -> None:
    """Ask for the configured backend and make a failure name itself."""
    name = requested_backend()
    api = _API_BY_NAME[name]
    if api is not None:
        try:
            widget.setApi(api)
        except Exception:
            pass
    try:
        widget.renderFailed.connect(
            lambda: print(
                f"[preview] {type(widget).__name__} renderFailed (requested={name}); "
                f"set {RHI_BACKEND_ENV}=opengl or vulkan and restart", flush=True))
    except Exception:
        pass


class _RhiSurfacePrimer(QRhiWidget):
    """A 1x1 throwaway that exists only to promote the top-level's backingstore once."""

    def initialize(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        """First successful RHI setup means the promotion happened; leave for good."""
        try:
            self.deleteLater()
        except Exception:
            pass


def install_rhi_primer(window: QWidget) -> None:
    """Install the primer on `window` early - before any real QRhiWidget content exists.

    Call from the main window's constructor (BlinkTable does it at the top of `MainWindow.__init__`, right after the
    frameless setup). Safe to call twice: a second primer would be harmless, but the window keeps one.
    """
    if window is None or not isinstance(window, QWidget):
        return
    if getattr(window, "_rhi_surface_primer", None) is not None:
        return
    try:
        primer = _RhiSurfacePrimer(window)
        primer.setObjectName("rhi_surface_primer")
        primer.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        configure_rhi_widget(primer)
        primer.setGeometry(0, 0, 1, 1)
        window._rhi_surface_primer = primer

        # Shown immediately, as BlinkTable does. Deferring this to the window's Show event was measured to change
        # nothing (a primer under a hidden top-level is never painted, so it cannot promote anything either) and it
        # crashed the app once the window had a real platform window: the primer self-destructs on its first
        # initialize(), and the deferred `show()` then ran on a deleted C++ object. The startup warnings the deferral
        # was meant to cure come from RVE's own pre-show `grab()` pass, not from this widget - see the `window.create()`
        # call in REAL-Video-Enhancer.py's startup.
        primer.show()
        window._rhi_surface_primer = primer
    except Exception as error:  # a primer that cannot be built must not cost the window
        print(f"[preview] rhi primer not installed: {error}", flush=True)

def gpu_preview_available() -> bool:
    """Whether a GPU pane can exist in THIS process at all.

    `QRhiWidget` needs a platform GL/EGL context. On a host whose EGL cannot create a screen (measured 2026-11-23 on
    the owner's Wayland session: `MESA-EGL: driver (null)`, `failed to create dri2 screen`, `eglSwapBuffers failed
    0x300d` after a driver/userland version mismatch) every QRhiWidget prints `No QRhi`, renders nothing and - before the
    guards - crashed the process. Probing one context here lets the page pick the raster pane instead of showing an
    empty one: `QOpenGLContext.create()` is the cheapest question that answers it.

    Overrides: `RVE_UI_RASTER_PREVIEW=1` forces the raster pane (kept for A/B), `RVE_UI_RHI_AVAILABLE=0|1` answers the
    probe by hand.
    """
    if (os.environ.get("RVE_UI_RASTER_PREVIEW", "0") or "0").strip().lower() in ("1", "true", "yes", "on"):
        return False
    override = os.environ.get("RVE_UI_RHI_AVAILABLE")
    if override is not None:
        return str(override).strip().lower() not in ("0", "false", "no", "off")
    try:
        from PySide6.QtGui import QOpenGLContext

        context = QOpenGLContext()
        return bool(context.create())
    except Exception:
        return False
