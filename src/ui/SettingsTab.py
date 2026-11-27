from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QMainWindow
from ..constants import PLATFORM
from ..Util import currentDirectory, open_folder, log
# `settings.txt` has its own owner since P3 of the seam plan (`src/ui/settings_store.py`). Re-exported because the
# window's entry point and `ProcessTab` import `Settings` from this module, and that edge is not this move's subject.
from .settings_store import Settings
from ..GenerateFFMpegCommand import FFMpegCommand
from ..VideoInfo import VideoLoader
from ..video_probe import is_probeable_input
from .pages.handles import WindowHandles
from . import i18n
from . import output_path

# the combo box shows user words, the backend wants flag values.
# this is the single place where the two are mapped.
QUALITY_PRESET_LABELS = {
    "Best quality": "best",
    "Fast": "fast",
}


def qualityPresetLabelToValue(label: str) -> str:
    """Maps a combo box label to the value the backend's --quality_preset expects."""
    return QUALITY_PRESET_LABELS.get(label, "best")


def qualityPresetValueToLabel(value: str) -> str:
    """Maps a saved --quality_preset value back to its combo box label."""
    for label, mappedValue in QUALITY_PRESET_LABELS.items():
        if mappedValue == value:
            return label
    return "Best quality"


# --- the controls' side of a stored value (settings audit 2026-11-27, F4) --------------------------------
#
# A stored value is a string, and each control reports it through a different read: a translated combo keeps its
# VALUE in item data (`currentData`), an encoder row shows the value as its text (`currentText`), a check box is
# `isChecked`, a spin box is `value`. A comparator that picked one repr would compare two unrelated things, so each
# row is paired with the read its OWN write handler uses below.

def _text(control) -> str:
    """The read of a combo/line edit whose visible text IS the value (precision, encoder, container...)."""
    return control.currentText()


def _field_text(control) -> str:
    """The read of a line edit that holds a stored string (the user's command line)."""
    return control.text()


def _data(control) -> str:
    """The read of a combo that carries the backend value in item data and a translated label as its text."""
    data = control.currentData()
    return "" if data is None else str(data)


def _data_or_text(control) -> str:
    """The read of a combo that may carry item data; text when it does not (the encoder row's own writer)."""
    data = control.currentData()
    return control.currentText() if data is None else str(data)


def _quality_preset(control) -> str:
    """The quality preset's writer: item data, or the label map for a combo that has none."""
    data = control.currentData()
    return str(data) if data else qualityPresetLabelToValue(control.currentText())


def _checked(control) -> str:
    return "True" if control.isChecked() else "False"


def _coarse(control) -> str:
    """The cel-reuse box is a choice inside the reuse switch: checked means the coarse statistic."""
    return "coarse" if control.isChecked() else "mean_abs"


def _number(control) -> str:
    return str(control.value())


@dataclass(frozen=True)
class ControlFact:
    """A stored key, the control that mirrors it, and the read that control's write handler uses."""

    key: str
    control: object
    read: Callable[[object], str]


def _same_value(stored: str, shown: str) -> bool:
    """Equality with one numeric allowance: `5` and `5.0` are the same setting, `5` and `4` are not."""
    if stored == shown:
        return True
    try:
        return float(stored) == float(shown)
    except ValueError:
        return False


# What the Best/Fast switch actually does since 2026-11-18: it selects the precision of the sixteen body
# convolutions (fp16 engine against the int8 body). The wording is the measurement, not a promise - the
# picture effect depends on the model and the material, which is exactly what the tooltip says.
QUALITY_PRESET_TOOLTIP = (
    "Speed or fidelity.\n"
    "Best quality: full precision, the model's own picture.\n"
    "Fast: about 1.6x faster - 8-bit inner layers, full precision at the ends. Very close to Best; "
    "fine line art can differ once the video is encoded."
)


def applyQualityPresetTooltip(comboBox, extra: str = "") -> None:
    """Attach the explanation to the Best/Fast switch, from one place.

    The combo box and the "?" label next to it used different texts - the label still carried the pre-2026-11-18
    wording about frame/tile reuse, which stopped being true when the switch became the fp16/int8 choice. The "?"
    label no longer exists on the page, and the old read of it was a `getattr(..., None)` that hid that fact; the
    tooltip now has exactly one subject, the combo box it explains.
    """
    text = QUALITY_PRESET_TOOLTIP + (f"\n{extra}" if extra else "")
    comboBox.setToolTip(text)


def hide_int8_preset_if_unsupported(parent, settings) -> None:
    """Remove Fast when this card cannot run the int8 body, and say why in the tooltip.

    Hiding beats disabling: a greyed-out option invites the user to hunt for a way to enable it, and there is none -
    the kernels need int8 tensor cores (compute capability 7.5, Turing and later). The setting is clamped to Best
    so a machine that moves between cards cannot keep a value the combo no longer offers.
    """
    combo = parent.page_handles.general.qualityPresetComboBox
    from .GPUDetect import int8_body_available
    available, reason = int8_body_available()
    if available:
        log(f"quality preset: Fast (int8) available - {reason}")
        return
    # The key, never the label: the item's text is translated ("Быстрое" under `ru`), so `findText("Fast")` missed it
    # on the one translated language - the int8 preset stayed offered, the log said it was hidden, and the stored key
    # was rewritten to `best` while the combo still showed Fast (settings audit, 2026-11-27).
    index = combo.findData("fast")
    removed = index >= 0
    if removed:
        combo.removeItem(index)
    log(f"quality preset: Fast (int8) {'hidden' if removed else 'was not offered'} - {reason}")
    if settings.settings.get("quality_preset") == "fast":
        settings.writeSetting("quality_preset", "best")
        best = combo.findData("best")
        combo.setCurrentIndex(0 if best < 0 else best)
    applyQualityPresetTooltip(
        combo, extra="Not offered on this card: it needs NVIDIA tensor cores (GTX 16-series / RTX or newer).")



class SettingsTab:
    def __init__(
        self,
        parent: QMainWindow,
        halfPrecisionSupport,
        pytorch_gpu_count,
        ncnn_gpu_count,
    ):
        self.parent = parent
        #: The typed page surface (`window.page_handles`). Required dataclass fields: a page that drops a widget
        #: fails while building, instead of handing this glue an `AttributeError` or a silent `getattr(..., None)`.
        self.page_handles: WindowHandles = parent.page_handles
        self.input_file = None
        #: True once the user has typed in the command line: from then on the app does NOT regenerate it (that is what
        #: made it "reset on any little thing" - nine signals rewrote it from the settings on every change).
        self.encoder_command_user = False
        self.ffmpegInfoWrapper = None
        self.color_space = None
        self.color_primaries = None
        self.color_transfer = None
        self.in_pix_fmt = ""
        self.hdr_mode = False
        # Audio and subtitles are ALWAYS copied (owner, 2026-11-24: "аудио и сабы то нахуя настраивать? выкинуть
        # контролы - просто copy делать"), so they are no longer controls this glue mirrors. The settings keys stay in
        # `settings_store` for file compatibility; nothing reads them for the command any more.
        #
        # The three encoder rows used to be collected here so `updateFFMpegCommand` could write them from the widgets
        # on every call; that write is gone (it silently replaced a stored value the combo could not display), and each
        # row now has its own write handler in `connectWriteSettings` like every other row.
        # ONE settings object for the window and this tab. Two instances over one file was a lost-update hazard, and
        # the output path is the fact it would have eaten: the window writes the path the user typed, the tab writes
        # every other row, and whichever wrote last serialised its own stale copy of the path.
        self.settings = getattr(parent, "settings", None) or Settings()
        log("Settings: " + str(self.settings.settings))
        self.connectSettingText() # has to be in this order, otherwise settings wont stick because they get reset.
        self.connectWriteSettings()
        self.restoreEncoderCommand()
        

        # disable half option if its not supported
        if not halfPrecisionSupport:
            self.page_handles.render_settings.precision.removeItem(1)

        # The probe prints one line per device, so the report carries COUNTS - a maximum id is `count - 1`, and 0
        # means "no GPU of this backend", not "the only GPU". The Render Settings page owns its two rows and turns
        # each count into the honest control (disabled row / fixed id / a live `0 .. count - 1` stepper); this glue
        # passes counts and never touches a widget of that page.
        self._render_settings_page().apply_gpu_counts(
            ncnn_gpu_count=ncnn_gpu_count,
            pytorch_gpu_count=pytorch_gpu_count,
        )
        self.page_handles.app_settings.openRVEFolderBtn.clicked.connect(lambda:open_folder(currentDirectory()))
        

        self.updateFFMpegCommand()
        # The restore is over and the probe's rows are clamped: say out loud which controls could not show the value
        # the file holds. This runs LAST on purpose - it reports the state the user actually sees, not an intermediate
        # one (`precision` loses `float16` a few lines above; a GPU stepper is bounded by its count).
        self.report_setting_mirror("startup")

    def _render_settings_page(self):
        """The Render Settings tab's owner object (`Tab3Page`), which owns the two GPU-id rows.

        Pages are not on the typed widget surface (`page_handles` names controls, not pages), so the page is reached
        through the shell exactly as `ProcessTab._general_settings_page` reaches the general page - one path for the
        glue and the capture probes. The page, not this glue, decides what a device count means for its own row.
        """
        return self.parent.shell.settingsPage_page.tab_3_page

    def restoreEncoderCommand(self) -> None:
        """A hand-edited command line comes back after a restart; an empty stored value means "generate it"."""
        saved = str(self.settings.settings.get("encoder_command", "") or "")
        if not saved:
            return
        self.encoder_command_user = True
        self.page_handles.encoder.EncoderCommand.setText(saved)

    def applyLanguageSetting(self) -> None:
        """Persist and apply the language chosen in the row - no restart.

        `emit_language_changed` selects the language and lets the toolkit fan `retranslate(lang)` out to every bound
        page (`src/ui/i18n.py`); the Qt translator reads the same cached value, so each page's
        `QCoreApplication.translate(...)` calls resolve in the new language during that pass. Persisting is separate
        on purpose: the switch must survive a restart, and `settings.txt` owns the value.
        """
        selected = self.page_handles.app_settings.language.currentData()
        if not selected:
            return
        language = i18n.emit_language_changed(selected)
        if self.settings.settings.get("language") != language:
            self.settings.writeSetting("language", language)

    def onEncoderCommandEdited(self) -> None:
        """The user typed in the command line: stop regenerating it, and persist what they typed.

        Clearing the field hands ownership back to the app: the next settings change regenerates it and the stored value
        is dropped.
        """
        text = self.page_handles.encoder.EncoderCommand.text()
        self.encoder_command_user = bool(text.strip())
        self.settings.writeSetting("encoder_command", text if self.encoder_command_user else "")
        # The pane's claim follows the ownership flag, not the next settings change: a typed command line is what
        # runs from this keystroke on.
        self.publishEncoderSource()
        self.parent.updateVideoGUIText()

    def updateFFMpegCommand(self):
        """Build the encoder command line from the settings dict - a READER, not a writer.

        The three encoder rows used to be WRITTEN here (`ffmpeg_settings_dict`): every call - six signals plus the
        launch - copied whatever the combos showed into `encoder`, `video_quality` and `video_pixel_format`. A stored
        value one of those combos cannot display was therefore replaced by the combo's fallback before the user ever
        saw it ("an encoder that changed by itself after an upgrade", settings audit 2026-11-27). Each of the three
        now has its own `currentIndexChanged` handler in `connectWriteSettings`, like every other row, so the dict is
        written by the control the user touched and read here.
        """
        self.out_pixel_fmt = self.settings.settings['video_pixel_format']
        pxfmtDict = {
                "yuv420p": "yuv420p",
                "yuv422p": "yuv422p",
                "yuv444p": "yuv444p",
                "yuv420p (10 bit)": "yuv420p10le",
                "yuv422p (10 bit)": "yuv422p10le",
                "yuv444p (10 bit)": "yuv444p10le",
            }
        # The dict maps the combo's ITEMS; a stored value the combo cannot show is now kept (the restore no longer
        # overwrites it from the widget) and passed through instead of raising `KeyError` at launch. The row's
        # disagreement is named by `setting_mirror_mismatches`, not repaired here.
        self.out_pixel_fmt = pxfmtDict.get(self.out_pixel_fmt, self.out_pixel_fmt)

        input_file = self.page_handles.general.inputFileText.text()
        # Only ask the backend about a REAL file (or a URL): the field carries the batch sentinel `{ MULTIPLE_FILES }`
        # whenever a folder is loaded, and probing that produced no FPS line - which used to end in a ZeroDivisionError
        # on every encoder change (measured 2026-11-22).
        if is_probeable_input(input_file) and self.input_file != input_file:
            # caching is nice: the probe launches a backend for cv2 + ffmpeg
            self.input_file = input_file
            try:
                self.ffmpegInfoWrapper = VideoLoader(self.input_file)
                self.ffmpegInfoWrapper.loadVideo()
                self.ffmpegInfoWrapper.getData()
            except Exception as probe_error:      # a broken file must not take the settings page down
                log(f"encoder settings: could not read {self.input_file} ({probe_error})")
            else:
                self.hdr_mode = (self.ffmpegInfoWrapper.is_hdr) and self.settings.settings['auto_hdr_mode'] == "True"
                self.color_space = self.ffmpegInfoWrapper.color_space
                self.color_primaries = self.ffmpegInfoWrapper.color_primaries
                self.color_transfer = self.ffmpegInfoWrapper.color_transfer
                self.in_pix_fmt = self.ffmpegInfoWrapper.pixel_format


        if self.hdr_mode or ("10" in self.in_pix_fmt and self.settings.settings['auto_hdr_mode'] == "True"):
            pxfmtDict = {
                        "yuv420p": "yuv420p10le",
                        "yuv422p": "yuv422p10le",
                        "yuv444p": "yuv444p10le",
                    }

            if self.out_pixel_fmt in pxfmtDict:
                self.out_pixel_fmt = pxfmtDict[self.out_pixel_fmt]


        command = FFMpegCommand(
            self.settings.settings['encoder'].replace(' (experimental)', '').replace(' (40 series and up)', ''),
            self.settings.settings['video_encoder_speed'],
            self.settings.settings['video_quality'],
            self.out_pixel_fmt,
            self.hdr_mode,
            self.color_space if self.in_pix_fmt != "yuv420p" else None,
            self.color_primaries,
            self.color_transfer,  
        ).build_command()
        if not self.encoder_command_user:
            self.page_handles.encoder.EncoderCommand.setText(" ".join(command))
        self.publishEncoderSource()
        self.parent.updateVideoGUIText()

    def publishEncoderSource(self) -> None:
        """Tell the General page's info pane WHICH encoder fact will run (settings audit 2026-11-27, F5).

        The pane paints its Encoder cell from `settings['encoder']`, which is the truth only while the app generates
        the command. Once the user has typed a command line the FIELD is what executes, and this glue is the one
        place that knows it (`encoder_command_user`). One fact, one owner: the flag and the field decide, the pane
        reports, and the pane names the setting it read (`encoder` or `encoder_command`).
        """
        text = self.page_handles.encoder.EncoderCommand.text() if self.encoder_command_user else ""
        self._general_settings_page().set_encoder_command(text)

    def _general_settings_page(self):
        """The General tab's owner object, whose video-info pane carries the encoder claim (F5).

        Reached through the shell like `_render_settings_page` reaches the Render tab: `page_handles` names controls,
        not pages.
        """
        return self.parent.shell.procPage_page.generalSettings_page
         

    def connectWriteSettings(self):
        

        self.page_handles.render_settings.precision.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "precision", self.page_handles.render_settings.precision.currentText()
            )
        )
        self.page_handles.render_settings.tensorrt_optimization_level.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "tensorrt_optimization_level",
                self.page_handles.render_settings.tensorrt_optimization_level.currentText(),
            )
        )
        self.page_handles.app_settings.preview_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "preview_enabled",
                "True" if self.page_handles.app_settings.preview_enabled.isChecked() else "False",
            )
        )
        self.page_handles.output.scene_change_detection_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_enabled",
                "True"
                if self.page_handles.output.scene_change_detection_enabled.isChecked()
                else "False",
            )
        )
        self.page_handles.app_settings.discord_rich_presence.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "discord_rich_presence",
                "True" if self.page_handles.app_settings.discord_rich_presence.isChecked() else "False",
            )
        )
        self.page_handles.app_settings.language.currentIndexChanged.connect(self.applyLanguageSetting)
        self.page_handles.output.scene_change_detection_method.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_method",
                self.page_handles.output.scene_change_detection_method.currentText(),
            )
        )
        self.page_handles.output.scene_change_detection_threshold.valueChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_threshold",
                str(self.page_handles.output.scene_change_detection_threshold.value()),
            )
        )
        self.page_handles.encoder.video_quality.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_quality",
                str(self.page_handles.encoder.video_quality.currentData()),
            )
        )
        # The Encoder page's "Output folder directory" row is a READ-ONLY mirror of the path field
        # (`src/ui/output_path.py::mirror_directory`), refreshed here because the field is the only owner. It used to be
        # a second, persisted way to set the folder - and the one that silently overwrote the path on every option
        # change, which is the owner's report of 2026-11-26.
        self.page_handles.general.outputFileText.textChanged.connect(self.refreshOutputDirectory)
        # `textEdited`, not `textChanged`: only a real user edit makes the path theirs. A programmatic `setText`
        # (a derived default, a restore, the reset button) must never claim ownership - the same distinction the
        # encoder command line above makes, for the same reason.
        self.page_handles.general.outputFileText.textEdited.connect(self.onOutputPathEdited)

        self.page_handles.settings.resetSettingsBtn.clicked.connect(self.resetSettings)

        self.page_handles.render_settings.uhd_mode.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "uhd_mode",
                "True" if self.page_handles.render_settings.uhd_mode.isChecked() else "False",
            )
        )
        self.page_handles.render_settings.ncnn_gpu_id.textChanged.connect(
            lambda: self.settings.writeSetting(
                "ncnn_gpu_id", self.page_handles.render_settings.ncnn_gpu_id.text()
            )
        )
        self.page_handles.render_settings.pytorch_gpu_id.textChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_gpu_id", self.page_handles.render_settings.pytorch_gpu_id.text()
            )
        )
        self.page_handles.output.auto_border_cropping.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "auto_border_cropping",
                "True" if self.page_handles.output.auto_border_cropping.isChecked() else "False",
            )
        )
        self.page_handles.encoder.video_container.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_container", self.page_handles.encoder.video_container.currentText()
            )
        )
        self.page_handles.encoder.video_pixel_format.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_pixel_format", self.page_handles.encoder.video_pixel_format.currentText()
            )
        )
        self.page_handles.download.pytorch_version.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_version", self.page_handles.download.pytorch_version.currentText()
            )
        )
        self.page_handles.download.pytorch_backend.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_backend", self.page_handles.download.pytorch_backend.currentText()
            )
        )
        
        self.page_handles.render_settings.dynamic_tensorrt_engine.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "dynamic_tensorrt_engine",
                "True" if self.page_handles.render_settings.dynamic_tensorrt_engine.isChecked() else "False"
            )
        )
        self.page_handles.render_settings.auto_hdr_mode.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "auto_hdr_mode",
                "True" if self.page_handles.render_settings.auto_hdr_mode.isChecked() else "False"
            )
        )
        self.page_handles.encoder.video_encoder_speed.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_encoder_speed", self.page_handles.encoder.video_encoder_speed.currentData()
            )
        )
        applyQualityPresetTooltip(self.page_handles.general.qualityPresetComboBox)
        hide_int8_preset_if_unsupported(self.parent, self.settings)
        # The two boxes' stored values are restored by `connectSettingText` (one restore, one owner); here they get
        # their behaviour.
        self.page_handles.general.frameReuse.setToolTip(
            "Skip re-upscaling frames that barely changed and reuse the previous result, corrected for "
            "whatever did change - about 1.9x faster on held animation. The gate still refuses frames it "
            "cannot certify, so this only decides whether it is allowed to try.")
        self.page_handles.general.frameReuse.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "temporal_reuse", "True" if self.page_handles.general.frameReuse.isChecked() else "False"))
        self.page_handles.general.frameReuse.stateChanged.connect(self.onFrameReuseSwitch)
        self.page_handles.general.celReuse.setToolTip(
            "Decide reuse on the drawing instead of on the film: a 3 px low-pass with the scan's sub-pixel "
            "wobble compensated, and the reused frame is moved by that estimate. On a denoised animation this "
            "raises the reusable share from ~12% to ~64% at the same acceptance, because a half-pixel wobble "
            "of a 1-2 px line otherwise reads as a changed cel. It also means grain, dust dots and the scan's "
            "own jitter are NOT preserved on the reused frames - that is the trade this switch makes.")
        self.page_handles.general.celReuse.setEnabled(self.page_handles.general.frameReuse.isChecked())
        self.page_handles.general.celReuse.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "temporal_reuse_statistic",
                "coarse" if self.page_handles.general.celReuse.isChecked() else "mean_abs"))
        self.onFrameReuseSwitch()
        self.page_handles.general.qualityPresetComboBox.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "quality_preset",
                self.page_handles.general.qualityPresetComboBox.currentData()
                or qualityPresetLabelToValue(self.page_handles.general.qualityPresetComboBox.currentText()),
            )
        )
        self.page_handles.general.inputFileText.textChanged.connect(
            self.updateFFMpegCommand
        )
        # The encoder row's own write: it used to be written by `updateFFMpegCommand` from the widget on every call,
        # including at launch, which is how a stored value the combo cannot show got replaced (§`updateFFMpegCommand`).
        # Connected BEFORE the command builder below, so the builder reads the value the user just chose.
        self.page_handles.encoder.encoder.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "encoder", _data_or_text(self.page_handles.encoder.encoder)
            )
        )
        self.page_handles.encoder.encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.encoder.video_pixel_format.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.render_settings.auto_hdr_mode.stateChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.encoder.video_quality.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.encoder.video_encoder_speed.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        # `textEdited`, not `textChanged`: the latter also fires for the programmatic `setText` above, which would mark
        # the field as user-owned the moment the app generated it.
        self.page_handles.encoder.EncoderCommand.textEdited.connect(self.onEncoderCommandEdited)

    def onFrameReuseSwitch(self, *args):
        """The cel switch only means something while reuse is on - an option inside an option.

        Disabled rather than hidden: a hidden checkbox that is checked would still write its setting, and the
        user would have no way to see what the gate is about to compare.
        """
        self.page_handles.general.celReuse.setEnabled(
            self.page_handles.general.frameReuse.isChecked())

    def onOutputPathEdited(self, text: str) -> None:
        """The user typed in the path field: from now on it is theirs, and it is persisted.

        Clearing the field hands the name back to the derivation (`store_user_path("")`) - the next option change
        fills it from the directory default again, which is the only case where a derived path may appear.
        """
        output_path.store_user_path(self.settings, text)

    def refreshOutputDirectory(self, *_args) -> None:
        """Keep the Encoder page's READ-ONLY row equal to the path field's directory.

        The row's own `textChanged` handler is gone, so nothing can be typed or written into it to become a second
        answer; this is the only writer, and the path field is the only source.
        """
        output_path.mirror_directory(
            self.page_handles.general,
            self.page_handles.output.output_folder_location,
            self.settings,
        )

    def resetSettings(self):
        """Reset once, in one defined order: the file, then the dict, then the controls.

        This used to be a `for i in range(10)` loop whose own comment admitted why ("settings wont fully reset until
        like 5 button presses"): a restore that cannot reach a row left that row's handler to write its stale value
        back on the NEXT pass, so passes were added until the visible row happened to settle. Ten passes also meant ten
        whole-file rewrites and ten rounds of every write handler (settings audit 2026-11-27, F6 - and the restore's
        silence is F4).

        One pass now suffices because `connectSettingText` restores with the controls' signals blocked: the file and
        the dict are the defaults and the controls are pushed to them, without any control writing back. What could
        not be reached is returned by the mirror report instead of being retried blindly.
        """
        self.settings.writeDefaultSettings()   # 1. the file holds the defaults (and so does the dict)
        self.settings.readSettings()           # 2. the dict is what the file says
        self.connectSettingText()              # 3. the controls are what the dict says (signals off)
        # The reset dropped the stored path; the field must drop it too, or the page keeps showing a choice the
        # settings file no longer has and the next option change would be refused by a path nobody chose.
        output_path.reset_path(self.page_handles.general, self.settings)
        self.refreshOutputDirectory()
        # The command line is a stored setting too. It used to survive a reset in the FIELD only: the file held "",
        # `encoder_command_user` stayed True and the field kept the old command, so the reset left a third answer.
        self.encoder_command_user = False
        self.updateFFMpegCommand()
        self.parent.switchToSettingsPage()
        return self.report_setting_mirror("reset")

    def _control_facts(self) -> list[ControlFact]:
        """Every row `connectSettingText` restores, paired with the read its own write handler uses.

        `encoder_command` is conditional on purpose: while the app generates the command line the field is a
        DERIVED value (the command the settings imply), not a copy of the stored key - it becomes the stored fact
        only once the user owns it (`onEncoderCommandEdited`).
        """
        ph = self.page_handles
        facts = [
            ControlFact("precision", ph.render_settings.precision, _text),
            ControlFact("tensorrt_optimization_level", ph.render_settings.tensorrt_optimization_level, _text),
            ControlFact("encoder", ph.encoder.encoder, _data_or_text),
            ControlFact("preview_enabled", ph.app_settings.preview_enabled, _checked),
            ControlFact("discord_rich_presence", ph.app_settings.discord_rich_presence, _checked),
            ControlFact("language", ph.app_settings.language, _data),
            ControlFact("scene_change_detection_enabled", ph.output.scene_change_detection_enabled, _checked),
            ControlFact("scene_change_detection_method", ph.output.scene_change_detection_method, _text),
            ControlFact("scene_change_detection_threshold", ph.output.scene_change_detection_threshold, _number),
            ControlFact("video_quality", ph.encoder.video_quality, _data),
            ControlFact("uhd_mode", ph.render_settings.uhd_mode, _checked),
            ControlFact("ncnn_gpu_id", ph.render_settings.ncnn_gpu_id, _number),
            ControlFact("pytorch_gpu_id", ph.render_settings.pytorch_gpu_id, _number),
            ControlFact("auto_border_cropping", ph.output.auto_border_cropping, _checked),
            ControlFact("video_container", ph.encoder.video_container, _text),
            ControlFact("video_pixel_format", ph.encoder.video_pixel_format, _text),
            ControlFact("pytorch_version", ph.download.pytorch_version, _text),
            ControlFact("pytorch_backend", ph.download.pytorch_backend, _text),
            ControlFact("dynamic_tensorrt_engine", ph.render_settings.dynamic_tensorrt_engine, _checked),
            ControlFact("auto_hdr_mode", ph.render_settings.auto_hdr_mode, _checked),
            ControlFact("video_encoder_speed", ph.encoder.video_encoder_speed, _data),
            ControlFact("quality_preset", ph.general.qualityPresetComboBox, _quality_preset),
            ControlFact("temporal_reuse", ph.general.frameReuse, _checked),
            ControlFact("temporal_reuse_statistic", ph.general.celReuse, _coarse),
        ]
        if self.encoder_command_user:
            facts.append(ControlFact("encoder_command", ph.encoder.EncoderCommand, _field_text))
        return facts

    def setting_mirror_mismatches(self) -> list[tuple[str, str, str]]:
        """Every restored row where the control does not report the stored value.

        Each entry is `(key, value in `settings`, value the control reports)`. This is the assertion the settings
        layer never had (settings audit 2026-11-27, F4): a combo without the stored item, a spin box with a lower
        maximum, a check box whose state has no counterpart - each used to keep its own value in silence, so the
        file, the dict a render obeys and the widget a user sees could disagree for a whole session.

        A non-empty result is NOT repaired here: forcing the widget would show a value the control cannot express,
        and rewriting the file from the widget would drop what the user chose. The caller decides; `resetSettings`
        and `__init__` log it.
        """
        mismatches = []
        for fact in self._control_facts():
            stored = str(self.settings.settings.get(fact.key, ""))
            shown = fact.read(fact.control)
            if not _same_value(stored, shown):
                mismatches.append((fact.key, stored, shown))
        return mismatches

    def report_setting_mirror(self, when: str) -> list[tuple[str, str, str]]:
        """Log one named warning per disagreement and return the list (`setting_mirror_mismatches`)."""
        mismatches = self.setting_mirror_mismatches()
        for key, stored, shown in mismatches:
            log(f"settings mirror mismatch after {when}: {key} is {stored!r} in settings.txt but its control "
                f"reports {shown!r}; neither the file nor the control was changed")
        return mismatches

    def connectSettingText(self):
        """Push the stored values into the controls, once, with the controls' signals BLOCKED.

        A restore is not a user edit: every `setCurrentText`/`setChecked` used to emit the row's write handler, which
        wrote the widget's value straight back into the dict and the file - so a row the restore could not reach was
        "corrected" back to the widget's stale value (that is why Reset needed ten passes). Blocking keeps the write
        path for the user's own edits, makes one restore enough, and lets `setting_mirror_mismatches` see the result
        instead of a self-corrected one.
        """
        if PLATFORM == "darwin":
            index = self.page_handles.encoder.encoder.findText("av1")
            self.page_handles.encoder.encoder.removeItem(index)

        blocked = [(fact.control, fact.control.signalsBlocked()) for fact in self._control_facts()]
        for control, _was_blocked in blocked:
            control.blockSignals(True)
        try:
            self._restore_settings_into_controls()
        finally:
            for control, was_blocked in blocked:
                control.blockSignals(was_blocked)
        self.onFrameReuseSwitch()
        self.report_setting_mirror("restore")

    def _restore_settings_into_controls(self) -> None:
        """The restore's body: one statement per row, exactly the rows `_control_facts` compares."""
        self.page_handles.render_settings.precision.setCurrentText(self.settings.settings["precision"])
        self.page_handles.render_settings.tensorrt_optimization_level.setCurrentText(
            self.settings.settings["tensorrt_optimization_level"]
        )
        self.page_handles.encoder.encoder.setCurrentText(self.settings.settings["encoder"])
        self.page_handles.app_settings.preview_enabled.setChecked(
            self.settings.settings["preview_enabled"] == "True"
        )
        self.page_handles.app_settings.discord_rich_presence.setChecked(
            self.settings.settings["discord_rich_presence"] == "True"
        )
        # The language row shows the language actually in force (the validator already turned an unknown stored value
        # into English, `src/ui/settings_store.py::validate_language`).
        self.page_handles.app_settings.language.setCurrentIndex(
            max(0, self.page_handles.app_settings.language.findData(i18n.current_language()))
        )
        self.page_handles.output.scene_change_detection_enabled.setChecked(
            self.settings.settings["scene_change_detection_enabled"] == "True"
        )
        self.page_handles.output.scene_change_detection_method.setCurrentText(
            self.settings.settings["scene_change_detection_method"]
        )
        self.page_handles.output.scene_change_detection_threshold.setValue(
            float(self.settings.settings["scene_change_detection_threshold"])
        )
        self.page_handles.encoder.video_quality.setCurrentIndex(
            max(0, self.page_handles.encoder.video_quality.findData(self.settings.settings["video_quality"]))
        )
        # The Encoder row follows the PATH, never the other way round (`output_folder_location` is only the fallback
        # for a user with no stored path - it is not a control any more).
        self.refreshOutputDirectory()
        self.page_handles.render_settings.uhd_mode.setChecked(self.settings.settings["uhd_mode"] == "True")
        self.page_handles.render_settings.ncnn_gpu_id.setValue(int(self.settings.settings["ncnn_gpu_id"]))
        self.page_handles.render_settings.pytorch_gpu_id.setValue(
            int(self.settings.settings["pytorch_gpu_id"])
        )
        self.page_handles.output.auto_border_cropping.setChecked(
            self.settings.settings["auto_border_cropping"] == "True"
        )
        self.page_handles.encoder.video_container.setCurrentText(
            self.settings.settings["video_container"]
        )
        self.page_handles.encoder.video_pixel_format.setCurrentText(
            self.settings.settings["video_pixel_format"]
        )
        self.page_handles.download.pytorch_version.setCurrentText(
            self.settings.settings["pytorch_version"]
        )
        self.page_handles.download.pytorch_backend.setCurrentText(
            self.settings.settings["pytorch_backend"]
        )
        self.page_handles.render_settings.dynamic_tensorrt_engine.setChecked(
            self.settings.settings["dynamic_tensorrt_engine"] == "True"
        )
        self.page_handles.render_settings.auto_hdr_mode.setChecked(
            self.settings.settings["auto_hdr_mode"] == "True"
        )
        self.page_handles.encoder.video_encoder_speed.setCurrentIndex(
            max(0, self.page_handles.encoder.video_encoder_speed.findData(self.settings.settings["video_encoder_speed"]))
        )
        self.page_handles.general.qualityPresetComboBox.setCurrentIndex(
            max(0, self.page_handles.general.qualityPresetComboBox.findData(
                self.settings.settings["quality_preset"]))
        )
        # The two reuse switches are settings rows too, restored here (they used to be set in `connectWriteSettings`,
        # which a Reset does not re-run - so a reset left them showing the previous session's values).
        self.page_handles.general.frameReuse.setChecked(
            self.settings.settings.get("temporal_reuse", "True") == "True")
        self.page_handles.general.celReuse.setChecked(
            self.settings.settings.get("temporal_reuse_statistic", "mean_abs") == "coarse")
