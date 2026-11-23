"""The window's IO, as plain functions the window calls (P3 of the seam plan).

`startup` is the launch-time work the window's `__init__` used to inline, `models` is the "is this job's model on
disk?" block `getCurrentRenderOptions` repeated per slot. Neither imports Qt, and neither owns a widget: the window
passes its instruments in and keeps its policy and its dialogs.
"""

from __future__ import annotations

from .models import ensure_models
from .startup import (
    BackendReport,
    available_backends,
    download_missing_dependencies,
    parse_backend_report,
    prepare_temp_directory,
    update_available_dependencies,
)

__all__ = [
    "BackendReport",
    "available_backends",
    "download_missing_dependencies",
    "ensure_models",
    "parse_backend_report",
    "prepare_temp_directory",
    "update_available_dependencies",
]
