"""Typed handles for the page widgets RVE's glue reads.

Why this module exists (2026-11-23): the shell exposes every page widget on the window (`expose()` in
`src/ui/shell/window.py`), so glue such as `src/ui/ProcessTab.py` and `src/ui/SettingsTab.py` could reach a
control through `self.parent.<name>`. That surface is a string contract, and a reparenting edit that lost a
widget did not fail anything: the read simply raised `AttributeError` at first click, or - where the glue
guarded it with `getattr(..., None)` - nothing happened at all (`qualityPresetInfoLabel` had been gone for a
while, silently).

These dataclasses are the typed edge instead. Every field is **required** (no defaults): a page that stops
building one cannot construct its handles, so the loss is a `TypeError` at page build rather than a `None`
at first use. The glue reads `self.page_handles.<group>.<name>`, and `harness/tests/test_page_handles.py`
derives the expected names from the glue's own source.

`PageSelector` exists because `QStackedWidget` only speaks indices while the capture walker names stems
`nav{index}`. The selector maps a page NAME to its index, so glue and captures stop hard-coding `2`/`3`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only; this module must import without PySide6
    from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QWidget
    from sli_ui_toolkit.widgets import (
        Button,
        CheckBox,
        ComboBox,
        ConsoleView,
        CustomLineEdit,
        DoubleSpinBox,
        Stepper,
        Text,
        TextListWidget,
        ToastProgressBar,
    )

#: The nav stack's pages, in the order `MainWindowShell.build` adds them. The names are stable identifiers the
#: glue and the capture stems share; `nav{index}` is the capture side of the same mapping.
PAGE_NAMES: tuple[str, ...] = ("home", "more", "proc", "settings", "download")


@dataclass(frozen=True)
class PageSelector:
    """A page NAME over a `QStackedWidget` - never an index at the call site.

    The capture walker still names its stems `nav{index}` (the stack has no stable object name per page), so the
    selector is the one place that translates the name to that index. `show("proc")` is the spelling glue uses;
    `index("proc") == 2` is what a capture stem carries.
    """

    stack: "QStackedWidget"
    names: tuple[str, ...] = PAGE_NAMES

    def __post_init__(self) -> None:
        count = self.stack.count()
        if count != len(self.names):
            raise ValueError(
                f"PageSelector: the stack has {count} pages but {len(self.names)} names were given "
                f"({', '.join(self.names)}); add the page's name in src/ui/pages/handles.py")

    def index(self, name: str) -> int:
        """The stack index for `name`; an unknown name is a `KeyError`, never a silent 0."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown page {name!r}; known pages: {', '.join(self.names)}") from None

    def show(self, name: str) -> int:
        """Make the named page current and return its index (what a capture stem would call it)."""
        index = self.index(name)
        self.stack.setCurrentIndex(index)
        return index

    def name_of(self, index: int | None = None) -> str:
        """The name at `index` (default: the current page) - the inverse of `index()`."""
        if index is None:
            index = self.stack.currentIndex()
        return self.names[index]


@dataclass(frozen=True)
class GeneralSettingsHandles:
    """The General tab's controls the glue reads (`src/ui/pages/general_settings.py`)."""

    inputFileSelectButton: "Button"
    batchSelectButton: "Button"
    inputFileText: "CustomLineEdit"
    outputFileText: "CustomLineEdit"
    outputFileSelectButton: "Button"
    openOutputFolderButton: "Button"
    backendComboBox: "ComboBox"
    interpolateCheckBox: "CheckBox"
    interpolateModelComboBox: "ComboBox"
    interpolationMultiplierSpinBox: "DoubleSpinBox"
    sloMoModeCheckBox: "CheckBox"
    decompressCheckBox: "CheckBox"
    decompressModelComboBox: "ComboBox"
    denoiseCheckBox: "CheckBox"
    denoiseModelComboBox: "ComboBox"
    deblurCheckBox: "CheckBox"
    deblurModelComboBox: "ComboBox"
    upscaleCheckBox: "CheckBox"
    upscaleModelComboBox: "ComboBox"
    upscaleScaleSpinBox: "DoubleSpinBox"
    qualityPresetComboBox: "ComboBox"
    frameReuse: "CheckBox"
    celReuse: "CheckBox"
    addToRenderQueueButton: "Button"


@dataclass(frozen=True)
class ProcPageHandles:
    """The process page's own chrome (`src/ui/pages/proc_page.py`)."""

    startRenderButton: "Button"
    pauseRenderButton: "Button"
    killRenderButton: "Button"
    onRenderButtonsContiainer: "QHBoxLayout"
    previewLabel: "QWidget"
    progressBar: "ToastProgressBar"
    progressDetail: "Text"
    FPS: "Text"
    ETA: "Text"
    STATUS: "Text"


@dataclass(frozen=True)
class RenderQueueHandles:
    """The Render Queue tab (`src/ui/pages/render_queue_tab.py`)."""

    renderQueueListWidget: "TextListWidget"


@dataclass(frozen=True)
class SettingsPageHandles:
    """The settings page's own chrome (`src/ui/pages/settings_page.py`)."""

    resetSettingsBtn: "Button"


@dataclass(frozen=True)
class OutputSettingsHandles:
    """The Output Settings tab (`src/ui/pages/tab_2.py`)."""

    output_folder_location: "CustomLineEdit"
    select_output_folder_location_btn: "Button"
    scene_change_detection_enabled: "CheckBox"
    scene_change_detection_method: "ComboBox"
    scene_change_detection_threshold: "DoubleSpinBox"
    auto_border_cropping: "CheckBox"


@dataclass(frozen=True)
class RenderSettingsHandles:
    """The Render Settings tab (`src/ui/pages/tab_3.py`)."""

    precision: "ComboBox"
    tensorrt_optimization_level: "ComboBox"
    auto_hdr_mode: "CheckBox"
    dynamic_tensorrt_engine: "CheckBox"
    uhd_mode: "CheckBox"
    ncnn_gpu_id: "Stepper"
    pytorch_gpu_id: "Stepper"


@dataclass(frozen=True)
class AppSettingsHandles:
    """The RVE Settings tab (`src/ui/pages/tab_4.py`)."""

    preview_enabled: "CheckBox"
    discord_rich_presence: "CheckBox"
    openRVEFolderBtn: "Button"


@dataclass(frozen=True)
class EncoderHandles:
    """The Encoder Options tab (`src/ui/pages/tab_6.py`)."""

    encoder: "ComboBox"
    EncoderCommand: "CustomLineEdit"
    video_container: "ComboBox"
    video_pixel_format: "ComboBox"
    video_quality: "ComboBox"
    video_encoder_speed: "ComboBox"
    audio_encoder: "ComboBox"
    audio_bitrate: "ComboBox"
    subtitle_encoder: "ComboBox"


@dataclass(frozen=True)
class LogsHandles:
    """The Logs tab (`src/ui/pages/tab_5.py`)."""

    renderOutput: "ConsoleView"


@dataclass(frozen=True)
class AdvancedSettingsHandles:
    """The Advanced tab (`src/ui/pages/advanced_settings.py`)."""

    tilingCheckBox: "CheckBox"
    tileSizeContainer: "QWidget"
    interpolateContainer_2: "QWidget"


@dataclass(frozen=True)
class DownloadPageHandles:
    """The download page controls the glue reads (`src/ui/pages/download_page.py`)."""

    pytorch_version: "ComboBox"
    pytorch_backend: "ComboBox"


@dataclass(frozen=True)
class MenuHandles:
    """The window chrome: the nav stack and its buttons (`src/ui/shell/window.py`)."""

    stackedWidget: "QStackedWidget"
    pages: PageSelector
    homeBtn: "Button"
    processBtn: "Button"
    settingsBtn: "Button"
    downloadBtn: "Button"


@dataclass(frozen=True)
class WindowHandles:
    """Every handle group, reached from the glue as `window.page_handles.<group>.<name>`."""

    general: GeneralSettingsHandles
    proc: ProcPageHandles
    queue: RenderQueueHandles
    settings: SettingsPageHandles
    output: OutputSettingsHandles
    render_settings: RenderSettingsHandles
    app_settings: AppSettingsHandles
    encoder: EncoderHandles
    logs: LogsHandles
    advanced: AdvancedSettingsHandles
    download: DownloadPageHandles
    menu: MenuHandles
