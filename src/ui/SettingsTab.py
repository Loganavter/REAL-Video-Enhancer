import os

from PySide6.QtWidgets import QMainWindow, QFileDialog
from ..constants import PLATFORM, HOME_PATH
from ..Util import currentDirectory, checkForWritePermissions, open_folder, log, FileHandler
from .QTcustom import RegularQTPopup
from ..GenerateFFMpegCommand import FFMpegCommand
from ..VideoInfo import VideoLoader
from ..video_probe import is_probeable_input

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


def applyQualityPresetTooltip(comboBox, infoLabel=None, extra: str = "") -> None:
    """Attach the explanation to the Best/Fast switch, from one place.

    The combo box and the "?" label next to it used different texts - the label still carried the pre-2026-11-18
    wording about frame/tile reuse, which stopped being true when the switch became the fp16/int8 choice. Both are
    set here so the pair cannot drift again.
    """
    text = QUALITY_PRESET_TOOLTIP + (f"\n{extra}" if extra else "")
    for widget in (comboBox, infoLabel):
        if widget is not None:
            widget.setToolTip(text)


def hide_int8_preset_if_unsupported(parent, settings) -> None:
    """Remove Fast when this card cannot run the int8 body, and say why in the tooltip.

    Hiding beats disabling: a greyed-out option invites the user to hunt for a way to enable it, and there is none -
    the kernels need int8 tensor cores (compute capability 7.5, Turing and later). The setting is clamped to Best
    so a machine that moves between cards cannot keep a value the combo no longer offers.
    """
    combo = getattr(parent, "qualityPresetComboBox", None)
    if combo is None:
        return
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
        combo, getattr(parent, "qualityPresetInfoLabel", None),
        extra="Not offered on this card: it needs NVIDIA tensor cores (GTX 16-series / RTX or newer).")



class SettingsTab:
    def __init__(
        self,
        parent: QMainWindow,
        halfPrecisionSupport,
        total_pytorch_gpus,
        total_ncnn_gpus,
    ):
        self.parent = parent
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
            "encoder": self.parent.encoder,
            "audio_encoder": self.parent.audio_encoder,
            "audio_bitrate": self.parent.audio_bitrate,
            "video_pixel_format": self.parent.video_pixel_format,
            "video_quality": self.parent.video_quality,

        }
        self.settings = Settings()
        log("Settings: " + str(self.settings.settings))
        self.connectSettingText() # has to be in this order, otherwise settings wont stick because they get reset.
        self.connectWriteSettings()
        self.restoreEncoderCommand()
        

        # disable half option if its not supported
        if not halfPrecisionSupport:
            self.parent.precision.removeItem(1)

        # set max gpu id for combo boxs
        self.parent.pytorch_gpu_id.setMaximum(total_pytorch_gpus)
        self.parent.ncnn_gpu_id.setMaximum(total_ncnn_gpus)
        self.parent.openRVEFolderBtn.clicked.connect(lambda:open_folder(currentDirectory()))
        

        self.updateFFMpegCommand()

    def restoreEncoderCommand(self) -> None:
        """A hand-edited command line comes back after a restart; an empty stored value means "generate it"."""
        saved = str(self.settings.settings.get("encoder_command", "") or "")
        if not saved:
            return
        self.encoder_command_user = True
        self.parent.EncoderCommand.setText(saved)

    def onEncoderCommandEdited(self) -> None:
        """The user typed in the command line: stop regenerating it, and persist what they typed.

        Clearing the field hands ownership back to the app: the next settings change regenerates it and the stored value
        is dropped.
        """
        text = self.parent.EncoderCommand.text()
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

        input_file = self.parent.inputFileText.text()
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
            self.parent.EncoderCommand.setText(" ".join(command))
        self.parent.updateVideoGUIText()
         

    def connectWriteSettings(self):
        
        self.parent.subtitle_encoder.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "subtitle_encoder", self.parent.subtitle_encoder.currentText()
            )
        )

        self.parent.precision.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "precision", self.parent.precision.currentText()
            )
        )
        self.parent.tensorrt_optimization_level.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "tensorrt_optimization_level",
                self.parent.tensorrt_optimization_level.currentText(),
            )
        )
        self.parent.preview_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "preview_enabled",
                "True" if self.parent.preview_enabled.isChecked() else "False",
            )
        )
        self.parent.scene_change_detection_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_enabled",
                "True"
                if self.parent.scene_change_detection_enabled.isChecked()
                else "False",
            )
        )
        self.parent.discord_rich_presence.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "discord_rich_presence",
                "True" if self.parent.discord_rich_presence.isChecked() else "False",
            )
        )
        self.parent.scene_change_detection_method.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_method",
                self.parent.scene_change_detection_method.currentText(),
            )
        )
        self.parent.scene_change_detection_threshold.valueChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_threshold",
                str(self.parent.scene_change_detection_threshold.value()),
            )
        )
        self.parent.video_quality.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_quality",
                str(self.parent.video_quality.currentText()),
            )
        )
        self.parent.output_folder_location.textChanged.connect(
            lambda: self.writeOutputFolder()
        )

        self.parent.resetSettingsBtn.clicked.connect(self.resetSettings)

        self.parent.uhd_mode.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "uhd_mode",
                "True" if self.parent.uhd_mode.isChecked() else "False",
            )
        )
        self.parent.ncnn_gpu_id.textChanged.connect(
            lambda: self.settings.writeSetting(
                "ncnn_gpu_id", self.parent.ncnn_gpu_id.text()
            )
        )
        self.parent.pytorch_gpu_id.textChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_gpu_id", self.parent.pytorch_gpu_id.text()
            )
        )
        self.parent.auto_border_cropping.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "auto_border_cropping",
                "True" if self.parent.auto_border_cropping.isChecked() else "False",
            )
        )
        self.parent.video_container.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_container", self.parent.video_container.currentText()
            )
        )
        self.parent.video_pixel_format.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_pixel_format", self.parent.video_pixel_format.currentText()
            )
        )
        self.parent.pytorch_version.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_version", self.parent.pytorch_version.currentText()
            )
        )
        self.parent.pytorch_backend.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_backend", self.parent.pytorch_backend.currentText()
            )
        )
        
        self.parent.dynamic_tensorrt_engine.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "dynamic_tensorrt_engine",
                "True" if self.parent.dynamic_tensorrt_engine.isChecked() else "False"
            )
        )
        self.parent.auto_hdr_mode.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "auto_hdr_mode",
                "True" if self.parent.auto_hdr_mode.isChecked() else "False"
            )
        )
        self.parent.video_encoder_speed.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_encoder_speed", self.parent.video_encoder_speed.currentText()
            )
        )
        applyQualityPresetTooltip(getattr(self.parent, "qualityPresetComboBox", None),
                                  getattr(self.parent, "qualityPresetInfoLabel", None))
        hide_int8_preset_if_unsupported(self.parent, self.settings)
        if getattr(self.parent, "frameReuse", None) is not None:
            self.parent.frameReuse.setChecked(
                self.settings.settings.get("temporal_reuse", "True") == "True")
            self.parent.frameReuse.setToolTip(
                "Skip re-upscaling frames that barely changed and reuse the previous result, corrected for "
                "whatever did change - about 1.9x faster on held animation. The gate still refuses frames it "
                "cannot certify, so this only decides whether it is allowed to try.")
            self.parent.frameReuse.stateChanged.connect(
                lambda: self.settings.writeSetting(
                    "temporal_reuse", "True" if self.parent.frameReuse.isChecked() else "False"))
            self.parent.frameReuse.stateChanged.connect(self.onFrameReuseSwitch)
        if getattr(self.parent, "celReuse", None) is not None:
            self.parent.celReuse.setChecked(
                self.settings.settings.get("temporal_reuse_statistic", "mean_abs") == "coarse")
            self.parent.celReuse.setToolTip(
                "Decide reuse on the drawing instead of on the film: a 3 px low-pass with the scan's sub-pixel "
                "wobble compensated, and the reused frame is moved by that estimate. On a denoised animation this "
                "raises the reusable share from ~12% to ~64% at the same acceptance, because a half-pixel wobble "
                "of a 1-2 px line otherwise reads as a changed cel. It also means grain, dust dots and the scan's "
                "own jitter are NOT preserved on the reused frames - that is the trade this switch makes.")
            self.parent.celReuse.setEnabled(self.parent.frameReuse.isChecked())
            self.parent.celReuse.stateChanged.connect(
                lambda: self.settings.writeSetting(
                    "temporal_reuse_statistic",
                    "coarse" if self.parent.celReuse.isChecked() else "mean_abs"))
        self.onFrameReuseSwitch()
        self.parent.qualityPresetComboBox.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "quality_preset",
                qualityPresetLabelToValue(
                    self.parent.qualityPresetComboBox.currentText()
                ),
            )
        )
        self.parent.inputFileText.textChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.audio_encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.video_pixel_format.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.audio_bitrate.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.auto_hdr_mode.stateChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.video_quality.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.video_encoder_speed.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.subtitle_encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        # `textEdited`, not `textChanged`: the latter also fires for the programmatic `setText` above, which would mark
        # the field as user-owned the moment the app generated it.
        self.parent.EncoderCommand.textEdited.connect(self.onEncoderCommandEdited)

    def onFrameReuseSwitch(self, *args):
        """The cel switch only means something while reuse is on - an option inside an option.

        Disabled rather than hidden: a hidden checkbox that is checked would still write its setting, and the
        user would have no way to see what the gate is about to compare.
        """
        reuse = getattr(self.parent, "frameReuse", None)
        cel = getattr(self.parent, "celReuse", None)
        if cel is None:
            return
        cel.setEnabled(reuse.isChecked() if reuse is not None else True)

    def writeOutputFolder(self):
        outputlocation = self.parent.output_folder_location.text()
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
            index = self.parent.encoder.findText("av1")
            self.parent.encoder.removeItem(index)

        self.parent.precision.setCurrentText(self.settings.settings["precision"])
        self.parent.tensorrt_optimization_level.setCurrentText(
            self.settings.settings["tensorrt_optimization_level"]
        )
        self.parent.encoder.setCurrentText(self.settings.settings["encoder"])
        self.parent.audio_encoder.setCurrentText(
            self.settings.settings["audio_encoder"]
        )
        self.parent.audio_bitrate.setCurrentText(
            self.settings.settings["audio_bitrate"]
        )
        self.parent.preview_enabled.setChecked(
            self.settings.settings["preview_enabled"] == "True"
        )
        self.parent.discord_rich_presence.setChecked(
            self.settings.settings["discord_rich_presence"] == "True"
        )
        self.parent.scene_change_detection_enabled.setChecked(
            self.settings.settings["scene_change_detection_enabled"] == "True"
        )
        self.parent.scene_change_detection_method.setCurrentText(
            self.settings.settings["scene_change_detection_method"]
        )
        self.parent.scene_change_detection_threshold.setValue(
            float(self.settings.settings["scene_change_detection_threshold"])
        )
        self.parent.video_quality.setCurrentText(
            self.settings.settings["video_quality"]
        )
        self.parent.output_folder_location.setText(
            self.settings.settings["output_folder_location"]
        )
        self.parent.select_output_folder_location_btn.clicked.connect(
            self.selectOutputFolder
        )
        self.parent.uhd_mode.setChecked(self.settings.settings["uhd_mode"] == "True")
        self.parent.ncnn_gpu_id.setValue(int(self.settings.settings["ncnn_gpu_id"]))
        self.parent.pytorch_gpu_id.setValue(
            int(self.settings.settings["pytorch_gpu_id"])
        )
        self.parent.auto_border_cropping.setChecked(
            self.settings.settings["auto_border_cropping"] == "True"
        )
        self.parent.video_container.setCurrentText(
            self.settings.settings["video_container"]
        )
        self.parent.video_pixel_format.setCurrentText(
            self.settings.settings["video_pixel_format"]
        )
        self.parent.pytorch_version.setCurrentText(
            self.settings.settings["pytorch_version"]
        )
        self.parent.pytorch_backend.setCurrentText(
            self.settings.settings["pytorch_backend"]
        )
        self.parent.dynamic_tensorrt_engine.setChecked(
            self.settings.settings["dynamic_tensorrt_engine"] == "True"
        )
        self.parent.auto_hdr_mode.setChecked(
            self.settings.settings["auto_hdr_mode"] == "True"
        )
        self.parent.video_encoder_speed.setCurrentText(
            self.settings.settings["video_encoder_speed"]
        )
        self.parent.subtitle_encoder.setCurrentText(
            self.settings.settings["subtitle_encoder"]
        )
        self.parent.qualityPresetComboBox.setCurrentText(
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
                self.parent.output_folder_location.setText(outputlocation)
            else:
                RegularQTPopup("No permissions to export here!")


class Settings:
    def __init__(self):
        self.settingsFile = os.path.join(currentDirectory(), "settings.txt")

        """
        The default settings are set here, and are overwritten by the settings in the settings file if it exists and the legnth of the settings is the same as the default settings.
        The key is equal to the name of the widget of the setting in the settings tab.
        """
        output_folder_default = FileHandler.getDefaultOutputFolder()
        self.defaultSettings = {
            "precision": "auto",
            "tensorrt_optimization_level": "5",
            "dynamic_tensorrt_engine": "False",
            "encoder": "libx264",
            "video_encoder_speed": "medium",
            "audio_encoder": "copy_audio",
            "subtitle_encoder": "copy_subtitle",
            "audio_bitrate": "192k",
            "preview_enabled": "True",
            "scene_change_detection_method": "sudo_scene_detect",
            "scene_change_detection_enabled": "True",
            "scene_change_detection_threshold": "3.5",
            "discord_rich_presence": "False",
            "video_quality": "High",
            "output_folder_location": output_folder_default,
            "last_input_folder_location": output_folder_default,
            "uhd_mode": "True",
            "ncnn_gpu_id": "0",
            "pytorch_gpu_id": "0",
            "auto_border_cropping": "False",
            "video_container": "mkv",
            "video_pixel_format": "yuv420p",
            "pytorch_version": "2.9.0",
            "pytorch_backend": "CUDA",
            "auto_hdr_mode": "True",
            "quality_preset": "best",
            # The generated ffmpeg command line, persisted so a hand-edited one survives a restart. Empty = "generate
            # from the settings above", which is the state the app starts in.
            "encoder_command": "",
            # reuse of unchanged frames: on by default since 2026-11-20 (the gate refuses material it
            # cannot certify, so the default is safe; this switch is for people who want it off entirely)
            "temporal_reuse": "True",
            # what the gate compares. "mean_abs" is the shipped statistic; "coarse" decides on a 3 px low-pass
            # with the translation compensated, i.e. on the drawing rather than on the film's grain, dust and
            # scan wobble (harness/docs/wsp/verified/reuse-structural-gate.txt). Not the default: it certifies
            # its own acceptance at that same coarse scale, which is a picture judgement the owner makes.
            "temporal_reuse_statistic": "mean_abs",
        }
        self.allowedSettings = {
            "encoder_command": "ANY",
            "precision": ("auto", "float32", "float16"),
            "tensorrt_optimization_level": ("0", "1", "2", "3", "4", "5"),
            "dynamic_tensorrt_engine": ("True", "False"),
            "encoder": (
                "libx264",
                "libx265",
                "vp9",
                "av1",
                "prores",
                "ffv1",
                "utvideo",
                "x264_nvenc",
                "x265_nvenc",
                "av1_nvenc (40 series and up)",
            ),
            "video_encoder_speed": ("placebo","slow", "medium", "fast", "fastest"),
            "audio_encoder": ("aac", "libmp3lame", "opus", "copy_audio"),
            "audio_bitrate": "ANY",
            "subtitle_encoder": ("copy_subtitle","srt","ass","webvtt"),
            "preview_enabled": ("True", "False"),
            "scene_change_detection_method": (
                "mean",
                "mean_segmented",
                "pyscenedetect",
                "sudo_scene_detect",
            ),
            "scene_change_detection_enabled": ("True", "False"),
            "scene_change_detection_threshold": [
                str(num / 10) for num in range(1, 100)
            ],
            "discord_rich_presence": ("True", "False"),
            "video_quality": ("Low", "Medium", "High", "Very_High", "Ultra", "Lossless"),
            "output_folder_location": "ANY",
            "last_input_folder_location": "ANY",
            "uhd_mode": ("True", "False"),
            "ncnn_gpu_id": "ANY",
            "pytorch_gpu_id": "ANY",
            "auto_border_cropping": ("True", "False"),
            "video_container": ("mkv", "mp4", "mov", "webm", "avi"),
            "video_pixel_format": "ANY",
            "pytorch_version": ("2.9.0", "2.8.0", "2.6.0"),
            "pytorch_backend": "ANY",
            "auto_hdr_mode": ("True", "False"),
            "quality_preset": ("best", "fast"),
            "temporal_reuse": ("True", "False"),
            "temporal_reuse_statistic": ("mean_abs", "coarse"),
        }
        self.settings = self.defaultSettings.copy()
        if not os.path.isfile(self.settingsFile):
            self.writeDefaultSettings()
        self.readSettings()
        # check if the settings file is corrupted
        if len(self.defaultSettings) != len(self.settings):
            self.writeDefaultSettings()
        
    def readSettings(self):
        """
        Reads the settings from the 'settings.txt' file and stores them in the 'settings' dictionary.

        Returns:
            None
        """
        with open(self.settingsFile, "r") as file:
            try:
                for line in file:
                    # `partition`, not `split(",")`: the encoder command is free text and legitimately contains commas
                    # (`-vf "scale=…,…"`), and a bare split raised ValueError and rewrote ALL defaults (owner, 2026-11-22:
                    # "the encoder line resets after any little thing").
                    key, _, value = line.strip().partition(",")
                    if not key:
                        continue
                    self.settings[key] = value
            except (
                ValueError
            ):  # writes and reads again if the settings file is corrupted
                self.writeDefaultSettings()
                self.readSettings()

    def writeSetting(self, setting: str, value: str):
        """
        Writes the specified setting with the given value to the settings dictionary.

        Parameters:
        - setting (str): The name of the setting to be written, this will be equal to the widget name in the settings tab if set correctly.
        - value (str): The value to be assigned to the setting.

        Returns:
        None
        """
        self.settings[setting] = value
        self.writeOutCurrentSettings()

    def writeDefaultSettings(self):
        """
        Writes the default settings to the settings file if it doesn't exist.

        Parameters:
            None

        Returns:
            None
        """
        self.settings = self.defaultSettings.copy()
        self.writeOutCurrentSettings()

    def writeOutCurrentSettings(self):
        """
        Writes the current settings to a file.

        Parameters:
            self (SettingsTab): The instance of the SettingsTab class.

        Returns:
            None
        """
        with open(self.settingsFile, "w") as file:
            for key, value in self.settings.items():
                if key in self.defaultSettings:  # check if the key is valid
                    if (
                        value in self.allowedSettings[key]
                        or self.allowedSettings[key] == "ANY"
                    ):  # check if it is in the allowed settings dict
                        file.write(f"{key},{value}\n")
                else:
                    self.writeDefaultSettings()
