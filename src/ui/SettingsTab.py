import os

from PySide6.QtWidgets import QMainWindow, QFileDialog
from ..constants import PLATFORM, HOME_PATH
from ..Util import currentDirectory, checkForWritePermissions, open_folder, log
from .QTcustom import RegularQTPopup
# `settings.txt` has its own owner since P3 of the seam plan (`src/ui/settings_store.py`). Re-exported because the
# window's entry point and `ProcessTab` import `Settings` from this module, and that edge is not this move's subject.
from .settings_store import Settings
from ..GenerateFFMpegCommand import FFMpegCommand
from ..VideoInfo import VideoLoader
from ..video_probe import is_probeable_input
from .pages.handles import WindowHandles

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
    index = combo.findText("Fast")
    if index >= 0:
        combo.removeItem(index)
    log(f"quality preset: Fast (int8) hidden - {reason}")
    if settings.settings.get("quality_preset") == "fast":
        settings.writeSetting("quality_preset", "best")
        combo.setCurrentText("Best quality")
    applyQualityPresetTooltip(
        combo, extra="Not offered on this card: it needs NVIDIA tensor cores (GTX 16-series / RTX or newer).")



class SettingsTab:
    def __init__(
        self,
        parent: QMainWindow,
        halfPrecisionSupport,
        total_pytorch_gpus,
        total_ncnn_gpus,
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
        self.ffmpeg_settings_dict = {
            "encoder": self.page_handles.encoder.encoder,
            "audio_encoder": self.page_handles.encoder.audio_encoder,
            "audio_bitrate": self.page_handles.encoder.audio_bitrate,
            "video_pixel_format": self.page_handles.encoder.video_pixel_format,
            "video_quality": self.page_handles.encoder.video_quality,

        }
        self.settings = Settings()
        log("Settings: " + str(self.settings.settings))
        self.connectSettingText() # has to be in this order, otherwise settings wont stick because they get reset.
        self.connectWriteSettings()
        self.restoreEncoderCommand()
        

        # disable half option if its not supported
        if not halfPrecisionSupport:
            self.page_handles.render_settings.precision.removeItem(1)

        # set max gpu id for combo boxs
        self.page_handles.render_settings.pytorch_gpu_id.setMaximum(total_pytorch_gpus)
        self.page_handles.render_settings.ncnn_gpu_id.setMaximum(total_ncnn_gpus)
        self.page_handles.app_settings.openRVEFolderBtn.clicked.connect(lambda:open_folder(currentDirectory()))
        

        self.updateFFMpegCommand()

    def restoreEncoderCommand(self) -> None:
        """A hand-edited command line comes back after a restart; an empty stored value means "generate it"."""
        saved = str(self.settings.settings.get("encoder_command", "") or "")
        if not saved:
            return
        self.encoder_command_user = True
        self.page_handles.encoder.EncoderCommand.setText(saved)

    def onEncoderCommandEdited(self) -> None:
        """The user typed in the command line: stop regenerating it, and persist what they typed.

        Clearing the field hands ownership back to the app: the next settings change regenerates it and the stored value
        is dropped.
        """
        text = self.page_handles.encoder.EncoderCommand.text()
        self.encoder_command_user = bool(text.strip())
        self.settings.writeSetting("encoder_command", text if self.encoder_command_user else "")

    def updateFFMpegCommand(self):
        """
        Updates the FFMpegCommand object with the current settings.
        """
        for key, value in self.ffmpeg_settings_dict.items():
            self.settings.writeSetting(key, value.currentText())
        
        self.out_pixel_fmt = self.settings.settings['video_pixel_format']
        pxfmtDict = {
                "yuv420p": "yuv420p",
                "yuv422p": "yuv422p",
                "yuv444p": "yuv444p",
                "yuv420p (10 bit)": "yuv420p10le",
                "yuv422p (10 bit)": "yuv422p10le",
                "yuv444p (10 bit)": "yuv444p10le",
            }
        self.out_pixel_fmt = pxfmtDict[self.out_pixel_fmt]

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
            self.settings.settings['audio_encoder'],
            self.settings.settings['audio_bitrate'],
            self.settings.settings['subtitle_encoder'],  
            self.hdr_mode,
            self.color_space if self.in_pix_fmt != "yuv420p" else None,
            self.color_primaries,
            self.color_transfer,  
        ).build_command()
        if not self.encoder_command_user:
            self.page_handles.encoder.EncoderCommand.setText(" ".join(command))
        self.parent.updateVideoGUIText()
         

    def connectWriteSettings(self):
        
        self.page_handles.encoder.subtitle_encoder.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "subtitle_encoder", self.page_handles.encoder.subtitle_encoder.currentText()
            )
        )

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
                str(self.page_handles.encoder.video_quality.currentText()),
            )
        )
        self.page_handles.output.output_folder_location.textChanged.connect(
            lambda: self.writeOutputFolder()
        )

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
                "video_encoder_speed", self.page_handles.encoder.video_encoder_speed.currentText()
            )
        )
        applyQualityPresetTooltip(self.page_handles.general.qualityPresetComboBox)
        hide_int8_preset_if_unsupported(self.parent, self.settings)
        self.page_handles.general.frameReuse.setChecked(
            self.settings.settings.get("temporal_reuse", "True") == "True")
        self.page_handles.general.frameReuse.setToolTip(
            "Skip re-upscaling frames that barely changed and reuse the previous result, corrected for "
            "whatever did change - about 1.9x faster on held animation. The gate still refuses frames it "
            "cannot certify, so this only decides whether it is allowed to try.")
        self.page_handles.general.frameReuse.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "temporal_reuse", "True" if self.page_handles.general.frameReuse.isChecked() else "False"))
        self.page_handles.general.frameReuse.stateChanged.connect(self.onFrameReuseSwitch)
        self.page_handles.general.celReuse.setChecked(
            self.settings.settings.get("temporal_reuse_statistic", "mean_abs") == "coarse")
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
                qualityPresetLabelToValue(
                    self.page_handles.general.qualityPresetComboBox.currentText()
                ),
            )
        )
        self.page_handles.general.inputFileText.textChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.encoder.encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.encoder.audio_encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.encoder.video_pixel_format.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.page_handles.encoder.audio_bitrate.currentIndexChanged.connect(
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
        self.page_handles.encoder.subtitle_encoder.currentIndexChanged.connect(
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

    def writeOutputFolder(self):
        outputlocation = self.page_handles.output.output_folder_location.text()
        if not (os.path.exists(outputlocation) and os.path.isdir(outputlocation)):
            RegularQTPopup(
                "No videos folder\nSetting default output folder to home directory."
            )
            outputlocation = HOME_PATH
        if not checkForWritePermissions(outputlocation):
            RegularQTPopup(
                "No permissions to export here!\nSetting default output folder to home directory."
            )
            outputlocation = HOME_PATH

        self.settings.writeSetting(
            "output_folder_location",
            str(outputlocation),
        )

    def resetSettings(self):
        for i in range(10): # idk why, but settings wont fully reset until like 5 button presses.
            self.settings.writeDefaultSettings()
            self.settings.readSettings()
            self.connectSettingText()
            self.parent.switchToSettingsPage()

    def connectSettingText(self):
        if PLATFORM == "darwin":
            index = self.page_handles.encoder.encoder.findText("av1")
            self.page_handles.encoder.encoder.removeItem(index)

        self.page_handles.render_settings.precision.setCurrentText(self.settings.settings["precision"])
        self.page_handles.render_settings.tensorrt_optimization_level.setCurrentText(
            self.settings.settings["tensorrt_optimization_level"]
        )
        self.page_handles.encoder.encoder.setCurrentText(self.settings.settings["encoder"])
        self.page_handles.encoder.audio_encoder.setCurrentText(
            self.settings.settings["audio_encoder"]
        )
        self.page_handles.encoder.audio_bitrate.setCurrentText(
            self.settings.settings["audio_bitrate"]
        )
        self.page_handles.app_settings.preview_enabled.setChecked(
            self.settings.settings["preview_enabled"] == "True"
        )
        self.page_handles.app_settings.discord_rich_presence.setChecked(
            self.settings.settings["discord_rich_presence"] == "True"
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
        self.page_handles.encoder.video_quality.setCurrentText(
            self.settings.settings["video_quality"]
        )
        self.page_handles.output.output_folder_location.setText(
            self.settings.settings["output_folder_location"]
        )
        self.page_handles.output.select_output_folder_location_btn.clicked.connect(
            self.selectOutputFolder
        )
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
        self.page_handles.encoder.video_encoder_speed.setCurrentText(
            self.settings.settings["video_encoder_speed"]
        )
        self.page_handles.encoder.subtitle_encoder.setCurrentText(
            self.settings.settings["subtitle_encoder"]
        )
        self.page_handles.general.qualityPresetComboBox.setCurrentText(
            qualityPresetValueToLabel(self.settings.settings["quality_preset"])
        )

    def selectOutputFolder(self):
        outputFile = QFileDialog.getExistingDirectory(
            parent=self.parent,
            caption="Select Folder",
            dir=os.path.expanduser("~"),
        )
        outputlocation = outputFile
        if os.path.exists(outputlocation) and os.path.isdir(outputlocation):
            if checkForWritePermissions(outputlocation):
                self.settings.writeSetting(
                    "output_folder_location",
                    str(outputlocation),
                )
                self.page_handles.output.output_folder_location.setText(outputlocation)
            else:
                RegularQTPopup("No permissions to export here!")
