from src.constants import CUSTOM_MODELS_PATH, IS_STEAM, MODELS_PATH, CWD,  IS_INSTALLED, TEMP_DOWNLOAD_PATH, USE_LOCAL_BACKEND, PLATFORM
import os
try: 
    os.makedirs(CWD) if not os.path.exists(CWD) else None
    # os.chdir(CWD) # need to actually chdir into the directory to have everything run correctly
except:
    pass
import sys
import os
import time
os.environ["PYTHONNOUSERSITE"] = "1" # Prevents python from installing packages in user site
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP"] = "0"
os.environ["PYTHON_JIT"] = "1" # enable python's experimental JIT for better performance in python 3.13
from PySide6.QtCore import QLockFile, QTimer, Qt
from PySide6.QtWidgets import (QStackedWidget, 
    QApplication,
    QMainWindow,
    QFileDialog,

)
from PySide6.QtGui import QIcon

from PySide6 import QtSvg  # Import the QtSvg module so svg icons can be used on windows
from src.version import version
from src.ModelHandler import getModels, getModelDisplayName

# other imports
from src.Util import (
    getOSInfo,
    getRAMAmount,
    getCPUInfo,
    getAvailableDiskSpace,
    FileHandler,
    log,
    createDirectory
)

createDirectory(os.path.join(CWD, "python"))
createDirectory(os.path.join(CWD, "bin"))


from src.DownloadModels import DownloadModel
from src.DownloadDeps import Python, DownloadDependencies
from src.ui.ProcessTab import ProcessTab
from src.ui.DownloadTab import DownloadTab
from src.ui.SettingsTab import SettingsTab, Settings
from src.ui.use_cases import startup as startup_io
from src.ui.use_cases import models as model_io
from src.ui.HomeTab import HomeTab
from src.Backendhandler import BackendHandler
from src.ModelHandler import totalModels
from src.ui.AnimationHandler import AnimationHandler
from src.ui.QTcustom import RegularQTPopup, NotificationOverlay, TextOutputPopup
from src.ui.RenderQueue import RenderQueue, RenderOptions
from src.VideoInfo import VideoLoader

svg = (
    QtSvg.QSvgRenderer()
)  # utilize the imported QtSvg module to render svg icons on windows


class MainWindow(QMainWindow):
    """Main window class for the REAL Video Enhancer application.

    This class extends QMainWindow; the interface itself is assembled by `src/ui/shell/window.py` (W3 of
    plan-rve-ui-on-toolkit: pages are code modules, the Designer file is gone).
    It sets up the user interface, connects buttons to switch menus, and handles various functionalities such as rendering, file selection, and backend setup.

    Attributes:
        homeDir (str): The home directory path.
        interpolateTimes (int): The number of times to interpolate frames.
        upscaleTimes (int): The number of times to upscale frames.
        pipeInFrames (None): Placeholder for input frames.
        latestPreviewImage (None): Placeholder for the latest preview image.
        aspect_ratio (float): The aspect ratio of the window.

    Methods:
        __init__(): Initializes the MainWindow class.
        QButtonConnect(): Connects buttons to switch menus.
        setupBackendDeps(): Sets up the backend dependencies.
        switchToHomePage(): Switches to the home page.
        switchToProcessingPage(): Switches to the processing page.
        switchToSettingsPage(): Switches to the settings page.
        switchToDownloadPage(): Switches to the download page.
        recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(): Recursively checks if at least one backend is installed.
        startRender(): Starts the rendering process.
        disableProcessPage(): Disables the process page.
        enableProcessPage(): Enables the process page.
        getAvailableBackends(): Retrieves the available backends.
        openInputFile(): Opens an input video file.
        openOutputFolder(): Opens an output folder.
        killRenderProcess(): Terminates the render process.
        closeEvent(event): Handles the close event of the main window."""

    def paintEvent(self, event):
        """Fill the window's own rect, so the surface can be declared opaque.

        `MainWindowShell` sets `WA_OpaquePaintEvent`, which tells Qt (and through it the compositor) that every pixel of
        this window is painted by the app. That claim has to hold: every page and panel inside is
        `WA_TranslucentBackground`, so a widget that paints nothing shows whatever is under it - and without this fill
        the "under it" was the other windows on the desktop (owner's terminal seen through the preview pane,
        2026-11-22).
        """
        from PySide6.QtGui import QPainter

        backdrop = getattr(self, "_backdrop_colour", None)
        if backdrop is None or os.environ.get("RVE_UI_NO_BACKDROP_PAINT"):
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), backdrop)

    def __init__(self, file_to_open=None):
        super().__init__()

        # Promote the top-level's backingstore to an RHI-capable surface BEFORE any real QRhiWidget exists. Qt does
        # that lazily, and on a real Wayland session the promotion races the preview pane's first paint: Qt prints
        # `QRhiWidget: No QRhi` and emits renderFailed (BlinkTable measured the same race; Telegram's lib_ui fixes it
        # with `GL::EnsureWindowRhi`). This is that fix, ported in src/ui/rhi_surface.py.
        from src.ui.rhi_surface import install_rhi_primer

        install_rhi_primer(self)

        # set up base variables
        self.homeDir = os.path.expanduser("~")
        self.pipeInFrames = None
        self.latestPreviewImage = None
        self.videoWidth = None
        self.videoHeight = None
        self.isVideoLoaded = False
        self.anyBackendsInstalled = True
        self.videoLength = 1
        self.batchVideos = []
        #: `path -> probe record` from `src/ui/batch_probe.py`. The queue add reads this instead of launching a
        #: `--print_video_info` interpreter per file a second time.
        self.batchInfo = {}
        self._batchProbe = None


        settings = Settings()
        self.settings = settings


        log(str(os.environ))
        # setup application
        startup_io.prepare_temp_directory(TEMP_DOWNLOAD_PATH, FileHandler.createDirectory)

        # Assemble the interface: the shell builds the chrome and the five nav pages and hands back ONE object,
        # `self.page_handles` (`src/ui/pages/handles.py`). Every page widget this class touches is a required field on
        # it, reached as `self.page_handles.<group>.<name>` - the window copies no page attribute onto itself.
        start_time = time.time()
        from src.ui import toolkit_boot
        toolkit_boot.ensure()               # the pages import toolkit widgets: make that dependency explicit here
        from src.ui.shell.window import MainWindowShell
        self.shell = MainWindowShell()
        self.shell.build(self)
        end_time = time.time()
        log("Setup ui time: " + str(end_time - start_time))
        # P0 of plan-rve-ui-on-toolkit: put toolkit widgets into the live window and look at the result.
        # Off by default; RVE_TOOLKIT_SPIKE=1 is a measurement, not a migration.
        try:
            from src.ui import capture_server, layout_dump, page_capture
            if capture_server.enabled():
                # Long-lived: boot once, answer snapshot requests from disk (harness/scripts/ui_live.py drives it).
                capture_server.install(self)
            elif page_capture.enabled():
                # The capture walker writes the layout dump too when one was asked for (one walk, both artefacts).
                page_capture.install(self)
            elif layout_dump.enabled():
                layout_dump.install(self)
        except Exception as _cap_exc:
            log(f"page capture failed: {_cap_exc}")
        # The toolkit's UI inspector, opt-in (`RVE_UI_INSPECTOR=1`): the tool window plus Shift+click selection. It
        # installs an application-wide event filter, so it is a development instrument, not a default.
        try:
            from src.ui.devtools import ui_inspector

            if ui_inspector.enabled():
                ui_inspector.install(self)
                log("ui inspector: armed (Shift+click a widget; Ctrl+Shift+I raises the tool window)")
        except Exception as _inspector_exc:
            log(f"ui inspector failed: {_inspector_exc}")
        # Debug keys for "there is a rectangle on my screen": Ctrl+Shift+R repaints the window, Ctrl+Shift+H hides the
        # widget under the cursor and logs its chain, Ctrl+Shift+D writes a layout audit (clipped children, stale stacked
        # pages), Ctrl+Shift+P probes the pixel under the cursor (what it renders vs what the screen shows). Armed
        # always: a tool the owner has to remember an env var for is a tool that silently answers nothing.
        try:
            from src.ui.devtools import debug_keys

            if debug_keys.enabled():
                debug_keys.install(self)
                log("debug keys: Ctrl+Shift+R repaint | H hide under cursor | D layout audit | P pixel probe")
            debug_keys.force_composite(self)
            from src.ui.devtools import bisect

            bisect.apply_mode(self)
        except Exception as _debug_keys_exc:
            log(f"debug keys failed: {_debug_keys_exc}")
        # The v2 runtime swap (`toolkit_swap.py`, `toolkit_spike.py`) is gone with W3: pages are built by their own
        # modules now, so a migration is an edit in a page module, not a splice into a live Designer tree.
        #self.page_handles.proc.VideoPreview.setVisible(False)

        # remove false hope
        self.page_handles.download.directMLBackendInstallerContainer.setVisible(False)

        start_time = time.time()
        backendHandler = BackendHandler(self, self.settings)
        end_time = time.time()
        log("Backend handler time: " + str(end_time - start_time))

        start_time = time.time()
        self.renderQueue = RenderQueue(self.page_handles.queue.renderQueueListWidget)
        end_time = time.time()
        log("Render queue time: " + str(end_time - start_time))

        start_time = time.time()
        if not IS_INSTALLED:
            startup_io.download_missing_dependencies()
        end_time = time.time()
        log("Dependency download time: " + str(end_time - start_time))
        
        #popupthread = create_independent_process(IndependentQTPopup, "Checking for dependency updates...")
        #popupthread.start() 

        start_time = time.time()
        startup_io.update_available_dependencies()
        end_time = time.time()
        log("Dependency update time: " + str(end_time - start_time))
        #try:
        #    popupthread.terminate()
        #except Exception:
        #    pass

        start_time = time.time()
        self.backends, self.fullOutput = startup_io.available_backends(backendHandler)
        end_time = time.time()
        

        # set default home page
        self.page_handles.menu.pages.show("home")

        self.anyBackendsInstalled = len(self.backends) > 0
        if not self.anyBackendsInstalled:
            from src.ui.dialogs.toolkit_dialogs import confirm

            automaticInstall = confirm(
                self,
                "Do you want to use simple installation?\n(Recommended for most users)",
            )
            if automaticInstall:
                self.downloadTab = DownloadTab(parent=self, backends=self.backends, skip_info_popup=True)
                self.downloadTab.installRecommended()
                self.backends, self.fullOutput = startup_io.available_backends(backendHandler)
                self.anyBackendsInstalled = len(self.backends) > 0
            else:
                self.page_handles.menu.processBtn.setEnabled(False) # disable process button if no backends are available
                self.page_handles.menu.pages.show("download")
                self.page_handles.menu.homeBtn.setChecked(False)
                self.page_handles.menu.downloadBtn.setChecked(True)
                self.page_handles.menu.processBtn.setToolTip("Please install at least one backend to enable processing.")
                self.page_handles.menu.processBtn.setToolTipDuration(0)



        
        icon_path = ":/icons/icons/logo-v2.svg"
        self.setWindowIcon(QIcon(icon_path))
        QApplication.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("REAL Video Enhancer")
        # Under the toolkit theme this call is the one that put the window subtree back on Fusion's default palette
        # after the theme had already applied RVE's (measured: unmigrated pages shifted, `mainwindow` palette != app
        # palette). The stock build keeps it - that is the look being reproduced.
        try:
            from src.ui import toolkit_theme as _toolkit_theme
            _theme_owns_look = _toolkit_theme.enabled()
        except Exception:
            _theme_owns_look = False
        if not _theme_owns_look:
            self.setPalette(QApplication.style().standardPalette())
        self.setMinimumSize(950, 600)

        self.aspect_ratio = self.width() / self.height()

        self.QConnect()
        # set up tabs
        self.page_handles.general.backendComboBox.addItems(self.backends)
        printOut = (
            "System Information:\n"
            + "OS: "
            + getOSInfo()
            + "\n"
            + "CPU: "
            + getCPUInfo()
            + "\n"
            + "RAM: "
            + getRAMAmount()
            + "\n"
            + "Available Disk Space: "
            + str(round(getAvailableDiskSpace(), 2))
            + "GB"
            + "\n"
            + "\nSoftware Information:\n"
            + f"REAL Video Enhancer Version: {version}\n"
            + f"Python Version: {Python().get_version()}\n"
            + f"Utilizing local backend: {USE_LOCAL_BACKEND}\n"
            + f"Working Directory: {CWD}\n"
            + self.fullOutput
        )
        self.page_handles.home.systemInfoText.setText(printOut)
        log(printOut)

        # process the output
        report = startup_io.parse_backend_report(self.fullOutput)
        halfPrecisionSupport = report.half_precision
        total_ncnn_gpus = report.ncnn_gpus
        total_pytorch_gpus = report.pytorch_gpus

        if self.anyBackendsInstalled:
            self.processTab = ProcessTab(
                parent=self,
                settings=settings,
            )
        self.homeTab = HomeTab(parent=self)
        self.downloadTab = DownloadTab(parent=self, backends=self.backends)
        self.settingsTab = SettingsTab(
            parent=self,
            halfPrecisionSupport=halfPrecisionSupport,
            total_ncnn_gpus=total_ncnn_gpus,
            total_pytorch_gpus=total_pytorch_gpus,
        )
        downloadDeps = DownloadDependencies(False)
        self.downloadTab.hideUninstallButtons()
        self.downloadTab.showUninstallButton(self.backends)

        # No startup fade: the window used to get a QGraphicsOpacityEffect here, BEFORE it was shown. The animation
        # only advances while an event loop runs, and `show()`/`app.exec()` come later - so the first frames were
        # rendered through an offscreen opacity surface that had not been painted yet: a dark window until the first
        # input (owner report 2026-11-22). A window effect is worse than no animation: it changes how EVERYTHING below
        # is painted. Page switches are instant for the same reason (see the switchTo* methods).
        self.animationHandler = AnimationHandler()

        if not self.anyBackendsInstalled:
            RegularQTPopup("Welcome to REAL Video Enhancer!\nPlease install at least one backend to get started.")

        # these additions are not ready for stable rve just yet.
        #self.page_handles.general.denoiseContainer.setVisible(False)
        self.page_handles.general.deblurContainer.setVisible(False)
        #self.denoiseCheckBoxContainer.setVisible(False)
        self.page_handles.general.deblurCheckBoxContainer.setVisible(False)

        #player = QMediaPlayer()
        #player.setSource(QUrl.fromLocalFile(r"C:\Users\tntwi\Downloads\CodeGeass-OP3.webm"))
        #player.setVideoOutput(self.page_handles.proc.VideoPreview)
        #self.page_handles.proc.VideoPreview.show()
        #self.playbutton.clicked.connect(lambda: player.play())
        if (file_to_open is not None and os.path.isfile(file_to_open) and len(self.backends) > 0):
            self.loadVideo(file_to_open)
            self.switchToProcessingPage()
            self.settings.writeSetting("last_input_folder_location", str(os.path.dirname(file_to_open)))

    def QConnect(self):
        # connect buttons to switch menus
        self.page_handles.menu.homeBtn.clicked.connect(self.switchToHomePage)
        self.page_handles.menu.processBtn.clicked.connect(self.switchToProcessingPage)
        self.page_handles.menu.settingsBtn.clicked.connect(self.switchToSettingsPage)
        self.page_handles.menu.downloadBtn.clicked.connect(self.switchToDownloadPage)
        self.page_handles.proc.renderPreviewBtn.clicked.connect(self.renderPreview)
        self.page_handles.proc.VideoPreview.setVisible(False)
        self.page_handles.proc.RenderedPreviewControlsContainer.setVisible(False)
        # connect getting default output file

    def renderPreview(self):
        # The preview renders ONE job, but it used to delete the whole queue to get there - queued work vanished from a
        # button that says "preview" (owner: "items disappear without me deleting them", 2026-11-22). The queue is put
        # aside and restored when the preview finishes (`guiChangesOnRenderCompletion`).
        self._queue_after_preview = list(self.renderQueue.getQueue())
        self.renderQueue.clear()
        renderOptions = self.getCurrentRenderOptions()
        # `getCurrentRenderOptions()` answers `1` when it refuses (no video loaded, no model ticked) and has already
        # shown its own notification. Reading it as an object raised `AttributeError: 'int' object has no attribute
        # 'inputFile'` inside a slot - the button did nothing, the refusal never reached the user, and a control-channel
        # client saw "started" with no render (measured 2026-11-23: a preview job that finished in 0.9 s with no
        # output). One shape for the refusal, checked here.
        if not isinstance(renderOptions, RenderOptions):
            return
        renderOptions.outputPath = os.path.join(TEMP_DOWNLOAD_PATH, f"{os.path.basename(renderOptions.inputFile)}_preview.mkv")
        renderOptions.startTime = self.page_handles.proc.startTimeSpinBox.value()
        FileHandler().removeFile(renderOptions.outputPath)
        renderOptions.endTime = self.page_handles.proc.endTimeSpinBox.value()
        if renderOptions.endTime <= renderOptions.startTime:
            NotificationOverlay("End time must be greater than start time!", self, timeout=1500)
            return


        renderOptions.isPreview = True
        self.renderQueue.add(renderOptions)
        self.startRender(self.renderQueue)


    def setButtonsUnchecked(self, buttonToIgnore):
        buttons = [
            self.page_handles.menu.homeBtn,
            self.page_handles.menu.processBtn,
            self.page_handles.menu.settingsBtn,
            self.page_handles.menu.downloadBtn,
        ]
        for button in buttons:
            if button != buttonToIgnore:
                button.setChecked(False)
            else:
                button.setChecked(True)

    # switch menus
    def switchToHomePage(self):
        self.page_handles.menu.pages.show("home")
        self.setButtonsUnchecked(self.page_handles.menu.homeBtn)

    def switchToProcessingPage(self):
        self.page_handles.menu.pages.show("proc")
        self.setButtonsUnchecked(self.page_handles.menu.processBtn)

    def switchToSettingsPage(self):
        self.page_handles.menu.pages.show("settings")
        self.setButtonsUnchecked(self.page_handles.menu.settingsBtn)

    def switchToDownloadPage(self):
        self.page_handles.menu.pages.show("download")
        self.setButtonsUnchecked(self.page_handles.menu.downloadBtn)

    def updateVideoGUIText(self):
        self.settings.readSettings()
        if self.isVideoLoaded:
            upscaleModelName = self.page_handles.general.upscaleModelComboBox.currentText()
            interpolateModelName = self.page_handles.general.interpolateModelComboBox.currentText()
            interpolateTimes = self.getInterpolationMultiplier(interpolateModelName)
            scale = self.getUpscaleModelScale(upscaleModelName)
            new_bitrate = 8 if "10" not in self.settingsTab.in_pix_fmt else 10
            inputText = (
                f"FPS: {round(self.videoFps, 0)} -> {round(self.videoFps * interpolateTimes, 0) if not self.page_handles.general.sloMoModeCheckBox.isChecked() else round(self.videoFps, 0)}\n"
                + f"Resolution: {self.videoWidth}x{self.videoHeight} -> {self.videoWidth * scale}x{self.videoHeight * scale}\n"
                + f"Frame Count: {self.videoFrameCount} -> {int(round(self.videoFrameCount * interpolateTimes, 0))}\n"
                + f"Encoder: {self.videoEncoder} -> {self.settings.settings['encoder']}\n"
                + f"Container: {self.videoContainer} -> {self.settings.settings['video_container']}\n"
                + f"Color Space: {self.colorSpace} -> {self.colorSpace}\n"
                + f"Pixel Format: {self.settingsTab.in_pix_fmt} -> {self.settingsTab.out_pixel_fmt}\n"
                + f"HDR: {self.videoHDR} -> {self.videoHDR if self.settings.settings['auto_hdr_mode'] == 'True' else 'False'}\n"
                + f"Bit Depth: {self.videoBitDepth} bit -> {new_bitrate if self.settings.settings['auto_hdr_mode'] == 'True' else 8} bit\n"
            )
            
            self.page_handles.general.inputVideoInfoTextEdit.setFontPointSize(10)
            self.page_handles.general.inputVideoInfoTextEdit.setText(inputText)

    def getInterpolationMultiplier(self, interpolateModelName):
        if interpolateModelName == "None" or not self.page_handles.general.interpolateCheckBox.isChecked():
            interpolateTimes = 1
        else:
            interpolateTimes = self.page_handles.general.interpolationMultiplierSpinBox.value()
        return interpolateTimes

    def getUpscaleModelScale(self, upscaleModelName):
        if upscaleModelName == "None" or upscaleModelName == "" or not self.page_handles.general.upscaleCheckBox.isChecked():
            scale = 1
        else:
            scale = int(self.page_handles.general.upscaleScaleSpinBox.value())
        return scale

    def setDefaultOutputFile(self, inputFile, outputDirectory):
        """
        Sets the default output file for the video enhancer.
        Parameters:
        - useDefaultVideoPath (bool): Flag indicating whether to use the default video path for the output file.
        Returns:
        None
        """

        # check if there is a video loaded
        if self.isVideoLoaded:
            if inputFile.strip().replace(" ", "") == "{MULTIPLE_FILES}":
                inputFile = " { FILE_NAME } "

            upscaleModelName = self.page_handles.general.upscaleModelComboBox.currentText()
            decompressModelName = self.page_handles.general.decompressModelComboBox.currentText()
            denoiseModelName = self.page_handles.general.denoiseModelComboBox.currentText()
            interpolateModelName = self.page_handles.general.interpolateModelComboBox.currentText()
            
            interpolateTimes = self.getInterpolationMultiplier(interpolateModelName) if not self.page_handles.general.sloMoModeCheckBox.isChecked() else 1
            scale = self.getUpscaleModelScale(upscaleModelName)
            container = self.settings.settings["video_container"]

            file_name = os.path.splitext(os.path.basename(inputFile))[0]
            base_file_name = (
                f"{file_name}"
                + ("" if not self.page_handles.general.interpolateCheckBox.isChecked() else f"_{getModelDisplayName(interpolateModelName)}")
                + ("" if not self.page_handles.general.decompressCheckBox.isChecked() else f"_{getModelDisplayName(decompressModelName)}")
                + ("" if  not self.page_handles.general.denoiseCheckBox.isChecked() else f"_{getModelDisplayName(denoiseModelName)}")
                + ("" if  not self.page_handles.general.upscaleCheckBox.isChecked() else f"_{getModelDisplayName(upscaleModelName)}")
                + f"_{round(interpolateTimes * self.videoFps, 0)}fps"
                + f"_{scale * self.videoWidth}x{scale * self.videoHeight}"
            )
            output_file = os.path.join(
                outputDirectory,
                f"{base_file_name}.{container}",
            )
            iteration = 0
            while os.path.isfile(output_file):
                output_file = os.path.join(
                    outputDirectory,
                    f"{base_file_name}_({iteration}).{container}",
                )
                iteration += 1
            self.page_handles.general.outputFileText.setText(output_file)
            return output_file

    def updateVideoGUIDetails(self):
        isInterpolate = self.page_handles.general.interpolateCheckBox.isChecked()
        isUpscale = self.page_handles.general.upscaleCheckBox.isChecked()
        isDeblur = self.page_handles.general.deblurCheckBox.isChecked()
        isDenoise = self.page_handles.general.denoiseCheckBox.isChecked()
        isDecompress = self.page_handles.general.decompressCheckBox.isChecked()
        
        self.page_handles.general.interpolationContainer.setVisible(isInterpolate)
        self.page_handles.advanced.interpolateContainer_2.setVisible(isInterpolate)
        self.page_handles.general.deblurContainer.setVisible(isDeblur)
        self.page_handles.general.denoiseContainer.setVisible(isDenoise)
        self.page_handles.general.decompressContainer.setVisible(isDecompress)
        
        # disable decompress for now
        #self.decompressCheckBoxContainer.setVisible(False)
        #self.page_handles.general.decompressContainer.setVisible(False)
        
        # set interpolation container visible if interpolate model is not none
        self.page_handles.advanced.upscaleContainer.setVisible(isUpscale or isDeblur or isDenoise or isDecompress)
        self.page_handles.general.generalUpscaleContainer.setVisible(isUpscale)
        # Quality Preset / "Reuse unchanged frames" / "By cel" need an UPSCALE model specifically - denoise, deblur and
        # decompress alone do not use them:
        #   * `temporal_reuse` is only created when there is an upscale model (backend/src/RenderVideo.py:455:
        #     `if temporal_reuse and self.upscaleModel and not self.interpolateModel ...`);
        #   * `quality_preset` resolves the int8 BODY of the upscale model (RenderVideo.resolve_int8_fit ->
        #     pytorch/UpscaleTorch._int8_body);
        #   * the restoration models run on their own branch (RenderVideo.py:1098-1099).
        # So the container follows `isUpscale` alone, not the whole "upscale body" group (owner's question, 2026-11-22).
        self.page_handles.general.qualityPresetContainer.setVisible(isUpscale)
        self.settings.readSettings()
        self.setDefaultOutputFile(self.page_handles.general.inputFileText.text(), self.settings.settings["output_folder_location"])
        self.updateVideoGUIText()
        self.page_handles.proc.startTimeSpinBox.setMaximum(self.videoLength)
        self.page_handles.proc.endTimeSpinBox.setMaximum(self.videoLength)
        self.page_handles.proc.timeInVideoScrollBar.setMaximum(self.videoLength)
        if isUpscale and (self.page_handles.general.upscaleModelComboBox.currentText() != "" or self.page_handles.general.upscaleModelComboBox.currentText() != "None"):
            try:
                max_scale = totalModels[self.page_handles.general.upscaleModelComboBox.currentText()][2]
                self.page_handles.general.upscaleScaleSpinBox.setMaximum(max_scale if max_scale > 0 else 4)
            except KeyError: # idk why it does this, gui is shit tbh.
                self.page_handles.general.upscaleScaleSpinBox.setMaximum(4)

    def getCurrentRenderOptions(self, input_file=None, output_path=None):
        interpolate = self.page_handles.general.interpolateModelComboBox.currentText()
        upscale = self.page_handles.general.upscaleModelComboBox.currentText()
        deblur = self.page_handles.general.deblurModelComboBox.currentText()
        denoise = self.page_handles.general.denoiseModelComboBox.currentText()
        decompress = self.page_handles.general.decompressModelComboBox.currentText()
        scene_detect_method = self.page_handles.output.scene_change_detection_method.currentText()
        input_file = self.page_handles.general.inputFileText.text() if input_file is None else input_file
        output_path = self.page_handles.general.outputFileText.text() if output_path is None else output_path
        interpolateModelFile = None
        upscaleModelFile = None
        deblurModelFile = None
        denoiseModelFile = None
        decompressModelFile = None
        scene_detect_model_file = None
        if not self.page_handles.general.interpolateCheckBox.isChecked():
            interpolate = None
        if not self.page_handles.general.upscaleCheckBox.isChecked():
            upscale = None
        if not self.page_handles.general.deblurCheckBox.isChecked():
            deblur = None
        if not self.page_handles.general.denoiseCheckBox.isChecked():
            denoise = None
        if not self.page_handles.general.decompressCheckBox.isChecked():
            decompress = None
        if not self.isVideoLoaded:
            NotificationOverlay("Video is not loaded!", self, timeout=1500)
            return 1

        if not interpolate and not upscale and not deblur and not denoise and not decompress:
            NotificationOverlay("Please select at least one model!", self, timeout=1500)
            return 1

        backend = self.page_handles.general.backendComboBox.currentText()
        upscaleModelArch = "custom"
        interpolateModels, upscaleModels, deblurModels, denoiseModels, decompressModels, sceneChangeModels = getModels(backend)

        if interpolate:
            interpolateDownloadFile = interpolateModels[interpolate][1]
            interpolateModelFile = interpolateModels[interpolate][0]

            if "sudo" in scene_detect_method:
                scene_detect_model_file = sceneChangeModels[scene_detect_method][0]
                scene_detect_download_file = sceneChangeModels[scene_detect_method][1]
                if not model_io.ensure_models(
                    [(scene_detect_model_file, scene_detect_download_file)], DownloadModel
                ):
                    NotificationOverlay(
                        "Unable to download scene detection model, please check your network and try again.",
                        self,
                        timeout=1500,
                    )
                    return 1
            if not model_io.ensure_models(
                [(interpolateModelFile, interpolateDownloadFile)], DownloadModel
            ):
                NotificationOverlay(
                    "Unable to download model, please check your network and try again.",
                    self,
                    timeout=1500,
                )
                return 1
        upscaleTimes = 1
        modelScale = 1
        if upscale:
            upscaleModelFile = upscaleModels[upscale][0]
            upscaleDownloadFile = upscaleModels[upscale][1]
            modelScale = upscaleModels[upscale][2]
            upscaleTimes = int(self.page_handles.general.upscaleScaleSpinBox.value())
            upscaleModelArch = upscaleModels[upscale][3]
            if upscaleModelArch != "custom":
                if not model_io.ensure_models(
                    [(upscaleModelFile, upscaleDownloadFile)], DownloadModel
                ):
                    NotificationOverlay(
                        "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                        self,
                        timeout=2500,
                    )
                    return 1
        if deblur:
            deblurModelFile = deblurModels[deblur][0]
            deblurDownloadFile = deblurModels[deblur][1]
            if not model_io.ensure_models(
                [(deblurModelFile, deblurDownloadFile)], DownloadModel
            ):
                NotificationOverlay(
                    "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                    self,
                    timeout=2500,
                )
                return 1
        
        if denoise:
            denoiseModelFile = denoiseModels[denoise][0]
            denoiseDownloadFile = denoiseModels[denoise][1]
            if not model_io.ensure_models(
                [(denoiseModelFile, denoiseDownloadFile)], DownloadModel
            ):
                NotificationOverlay(
                    "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                    self,
                    timeout=2500,
                )
                return 1
        if decompress:
            decompressModelFile = decompressModels[decompress][0]
            decompressDownloadFile = decompressModels[decompress][1]
            if not model_io.ensure_models(
                [(decompressModelFile, decompressDownloadFile)], DownloadModel
            ):
                NotificationOverlay(
                    "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                    self,
                    timeout=2500,
                )
                return 1
        hdrmode = False
        """if "gmfss" in interpolateDownloadFile.lower():
            log("GMFSS model detected, enabling rgb48 proc to have correct colors.")
            hdrmode = True"""
        if self.settings.settings["auto_hdr_mode"] == "True" and (self.videoHDR or self.videoBitDepth > 8):
            hdrmode = True
        return RenderOptions(
            inputFile=input_file,
            outputPath=output_path,
            videoWidth=self.videoWidth,
            videoHeight=self.videoHeight,
            videoFps=self.videoFps,
            tilingEnabled=self.page_handles.advanced.tilingCheckBox.isChecked(),
            tilesize=self.page_handles.advanced.tileSizeComboBox.currentText(),
            videoFrameCount=self.videoFrameCount,
            backend=self.page_handles.general.backendComboBox.currentText(),
            interpolateTimes=self.getInterpolationMultiplier(
                self.page_handles.general.interpolateModelComboBox.currentText()
            ),
            benchmarkMode=self.page_handles.advanced.benchmarkModeCheckBox.isChecked(),
            sloMoMode=self.page_handles.general.sloMoModeCheckBox.isChecked(),
            dyanmicScaleOpticalFlow=self.page_handles.advanced.dynamicScaledOpticalFlowCheckBox.isChecked(),
            ensemble=self.page_handles.advanced.ensembleCheckBox.isChecked(),
            modelScale=modelScale,
            upscaleModelArch=upscaleModelArch,
            upscaleModelFile=upscaleModelFile,
            deblurModelFile=deblurModelFile,
            denoiseModelFile=denoiseModelFile,
            decompressModelFile=decompressModelFile,
            interpolateModelFile=interpolateModelFile,
            sceneChangeModelFile=scene_detect_model_file,
            hdrMode=hdrmode,
            # `--override_upscale_scale` is the "run the model once, then resize to this multiplier" switch. When the
            # request IS the model's own scale there is nothing to override, and passing it anyway made the backend
            # resize every frame to the size it already had (a full 2880x2160 10-bit copy in and out). The backend
            # already falls back to the model's scale when this is None.
            overrideUpscaleScale=(upscaleTimes if upscaleTimes != modelScale else None),
            encoderCommand=self.page_handles.encoder.EncoderCommand.text(),
        )

    def _applyVideoInfo(self, info: dict) -> None:
        """The page's per-video fields, from a probe record (single file or a folder's batch)."""
        self.videoWidth = info.get("width", 0)
        self.videoHeight = info.get("height", 0)
        self.videoFps = info.get("fps", 0)
        self.videoLength = info.get("duration", 0)
        self.videoFrameCount = info.get("total_frames", 0)
        self.videoEncoder = info.get("codec", "")
        self.videoBitrate = info.get("bitrate", "")
        self.colorSpace = info.get("color_space", "")
        self.pixelFMT = info.get("pixel_format", "")
        self.videoHDR = info.get("is_hdr", False)
        self.videoBitDepth = info.get("bit_depth", None)

    def addToRenderQueue(self):
        self.settings.readSettings()
        input_file = self.page_handles.general.inputFileText.text()
        output_path = self.page_handles.general.outputFileText.text()

        if "{MULTIPLE_FILES}" in input_file.strip().replace(" ", ""):
            output_path = os.path.dirname(output_path)
            if not self.batchVideos:
                NotificationOverlay("No videos loaded - pick the folder again.", self, timeout=2000)
                return
            added = 0
            skipped = 0
            for video in list(self.batchVideos):
                # The folder read already probed this file; asking again is what made a batch add feel like a freeze
                # (one backend interpreter per video on the GUI thread).
                info = self.batchInfo.get(video)
                if info:
                    self._applyVideoInfo(info)
                else:
                    videoHandler = VideoLoader(video)
                    videoHandler.loadVideo()
                    videoHandler.getData()
                    self._applyVideoInfo({
                        "width": videoHandler.width, "height": videoHandler.height, "fps": videoHandler.fps,
                        "duration": videoHandler.duration, "total_frames": videoHandler.total_frames,
                        "codec": videoHandler.codec_str, "bitrate": videoHandler.bitrate,
                        "color_space": videoHandler.color_space, "pixel_format": videoHandler.pixel_format,
                        "is_hdr": videoHandler.is_hdr, "bit_depth": videoHandler.bit_depth,
                    })

                # set output_path for checking
                default_output_path = self.setDefaultOutputFile(video, output_path)

                # check if file already exists in renderQueue
                if any(item.outputPath == default_output_path for item in self.renderQueue.getQueue()):
                    skipped += 1
                    continue

                renderOptions = self.getCurrentRenderOptions(
                    input_file=video,
                    output_path=default_output_path,
                )
                if renderOptions == 1:
                    return
                self.renderQueue.add(renderOptions)
                added += 1

            # The batch STAYS loaded: it used to be cleared right here, so a second "Add to queue" press did nothing at
            # all and read as "the list reset itself" (owner, 2026-11-22). Pressing again now adds whatever is new and
            # says how many were already there.
            message = f"Added {added} to the queue"
            if skipped:
                message += f", {skipped} already in it"
            if not added and not skipped:
                message = "Nothing to add"
            NotificationOverlay(message, self, timeout=2000)
            return

        for renderOptions in self.renderQueue.getQueue():
            if output_path == renderOptions.outputPath:
                NotificationOverlay("Output file already in queue!", self, timeout=1500)
                return

        renderOptions = self.getCurrentRenderOptions()
        if renderOptions == 1:
            return
        # alert user that item has been added to queue
        NotificationOverlay(
            f"{self.page_handles.general.inputFileText.text()} Added to queue!", self, timeout=1500
        )
        self.renderQueue.add(renderOptions)

    def startRender(self, renderQueue=None):
        # Remember the model this job runs with: the upscale list's first entry is an accident of the file listing (a 4x
        # model on top), and the next start should open with the model the user actually chose.
        try:
            general = self.page_handles.general
            if general.upscaleCheckBox.isChecked():
                chosen = general.upscaleModelComboBox.currentText()
                if chosen and str(self.settings.settings.get("last_upscale_model", "")) != str(chosen):
                    self.settings.writeSetting("last_upscale_model", str(chosen))
        except Exception:
            pass
        if not renderQueue:
            renderQueue = self.renderQueue
        if len(renderQueue.queue) == 0:
            NotificationOverlay("Render queue is empty!", self, timeout=1500)
            return
        self.page_handles.proc.startRenderButton.setEnabled(False)
        self.page_handles.proc.VideoPreview.setVisible(False)
        self.page_handles.proc.previewLabel.setVisible(True)

        self.disableProcessPage()
        
        self.processTab.run(renderQueue)

    def disableProcessPage(self):
        for child in self.page_handles.general.generalSettings.children():
            child.setEnabled(False)
        for child in self.page_handles.advanced.advancedSettings.children():
            child.setEnabled(False)
        # NOT `self.renderQueueTab.children()`: disabling the page disabled its LIST, and the wheel over a disabled
        # widget does nothing - the owner could not scroll the queue while a render ran (2026-11-22). The tab stays
        # interactive; only mutations are locked, which keeps the queue coherent under the writer thread that is
        # consuming it.
        self.renderQueue.set_locked(True)
        for child in self.page_handles.encoder.encoderSettings.children():
            child.setEnabled(False)
        self.page_handles.proc.RenderedPreviewControlsContainer.setEnabled(False)
        self.page_handles.general.scrollArea_4.setEnabled(True)
        self.page_handles.general.scrollAreaWidgetContents_4.setEnabled(False)
        self.page_handles.general.widget_5.setEnabled(True)
        self.page_handles.general.outputFileContainer.setEnabled(True)
        self.page_handles.general.openOutputFolderButton.setEnabled(True)
        self.page_handles.general.outputFileContainer.setEnabled(True)

    def enableProcessPage(self):
        for child in self.page_handles.general.generalSettings.children():
            child.setEnabled(True)
        for child in self.page_handles.advanced.advancedSettings.children():
            child.setEnabled(True)
        self.renderQueue.set_locked(False)
        for child in self.page_handles.encoder.encoderSettings.children():
            child.setEnabled(True)
        self.page_handles.proc.RenderedPreviewControlsContainer.setEnabled(True)
        self.page_handles.general.scrollAreaWidgetContents_4.setEnabled(True)

    def loadVideo(self, inputFile, multi_file=False):
        if "{MULTIPLE_FILES}" in inputFile.strip().replace(" ", ""):
            return
        if inputFile == "":
            NotificationOverlay("Please select a video file!", self, timeout=1500)
            return
        if multi_file:
            from src.ui.batch_probe import BatchProbeThread
            from src.ui.video_scan import video_files

            self.page_handles.general.outputFileText.setEnabled(False)
            # The batch is REPLACED by a new folder, never appended to: picking a folder twice used to double every
            # entry, and the queue add then reported every file as "already in queue".
            self.batchVideos = []
            self.batchInfo = {}
            files = video_files(inputFile)
            if not files:
                NotificationOverlay("No valid videos found in the selected folder!", self, timeout=1500)
                return
            self.page_handles.general.inputFileText.setText(" { MULTIPLE_FILES } ")
            self.page_handles.general.inputFileText.setEnabled(False)
            self.videoWidth = 0
            self.videoHeight = 0
            self.videoFps = 0
            self.videoLength = 0
            self.videoFrameCount = 0
            self.videoEncoder = "Multi File"
            self.videoBitrate = "Multi File"
            self.videoContainer = "Multi File"
            self.colorSpace = "Multi File"
            self.pixelFMT = "Multi File"
            self.videoHDR = "Multi File"
            self.videoBitDepth = "Multi File"
            self._startBatchProbe(files, total=len(files))
            self.page_handles.general.outputFileSelectButton.setEnabled(True)
            self.page_handles.general.openOutputFolderButton.setEnabled(True)
            self.isVideoLoaded = True
            self.updateVideoGUIDetails()
            return
        else:
                    
            videoHandler = VideoLoader(inputFile)
            videoHandler.loadVideo()
            if (
                not videoHandler.isValidVideo()
            ):  # this handles case for invalid youtube link and invalid video file
                NotificationOverlay("Not a valid input!", self, timeout=1500)
                return
            videoHandler.getData()
            self.videoWidth = videoHandler.width
            self.videoHeight = videoHandler.height
            self.videoFps = videoHandler.fps
            self.videoLength = videoHandler.duration
            self.videoFrameCount = videoHandler.total_frames
            self.videoEncoder = videoHandler.codec_str
            self.videoBitrate = videoHandler.bitrate
            self.videoContainer = videoHandler.videoContainer
            self.colorSpace = videoHandler.color_space
            self.pixelFMT = videoHandler.pixel_format
            self.videoHDR = videoHandler.is_hdr
            self.videoBitDepth = videoHandler.bit_depth

            self.page_handles.general.inputFileText.setText(inputFile)
            self.page_handles.general.outputFileText.setEnabled(True)

        self.page_handles.general.outputFileSelectButton.setEnabled(True)
        self.page_handles.general.openOutputFolderButton.setEnabled(True)    
        self.isVideoLoaded = True
        self.updateVideoGUIDetails()

    def _startBatchProbe(self, files, total: int) -> None:
        """Read the folder on a thread: the window keeps painting and every record shows up as progress.

        This used to be a loop of `VideoLoader` on the GUI thread, one backend interpreter per file (0.16 s each here,
        so half a minute for 200 files), and it re-ran for every file again when the batch was added to the queue.
        """
        from src.ui.batch_probe import BatchProbeThread

        if getattr(self, "_batchProbe", None) is not None:
            self._batchProbe.wait()
        probe = BatchProbeThread(files, self)
        self._batchProbe = probe
        seen = {"count": 0}

        def on_record(record: dict) -> None:
            seen["count"] += 1
            path = record.get("path")
            if not record.get("valid") or not path:
                return
            self.batchInfo[path] = record
            if path not in self.batchVideos:
                self.batchVideos.append(path)
            if seen["count"] % 10 == 0:
                NotificationOverlay(f"Reading videos: {seen['count']} / {total}", self, timeout=600)

        def on_done(records) -> None:
            if not self.batchVideos:
                NotificationOverlay("No valid videos found in the selected folder!", self, timeout=2000)
                self.page_handles.general.inputFileText.setText("")
                self.page_handles.general.inputFileText.setEnabled(True)
                return
            NotificationOverlay(f"Loaded {len(self.batchVideos)} videos.", self, timeout=1500)

        def on_failed(message: str) -> None:
            NotificationOverlay(f"Could not read the folder: {message}", self, timeout=2500)

        probe.record.connect(on_record)
        probe.completed.connect(on_done)
        probe.failed.connect(on_failed)
        probe.start()

    # input file button
    def openInputFile(self):
        """
        Opens a video file and checks if it is valid,

        if it is valid, it will set self.inputFile to the input file, and set the text input field to the input file path.
        if it is not valid, it will give a warning to the user.

        > IMPLEMENT AFTER SELECT AI >  Last, It will enable the output select button, and auto create a default output file

        *NOTE
        This function will set self.videoWidth, self.videoHeight, and self.videoFps

        """

        fileFilter = "Video files (*.mp4 *.mov *.webm *.mkv);;All Files (*)"
        inputFile, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select File",
            dir=self.settings.settings["last_input_folder_location"] if os.path.exists(self.settings.settings["last_input_folder_location"]) else self.homeDir,
            filter=fileFilter,
        )
        self.loadVideo(inputFile)
        self.settings.writeSetting("last_input_folder_location", str(os.path.dirname(inputFile)))
    
    def openBatchFiles(self):
        inputFolder = QFileDialog.getExistingDirectory(
            self,
            caption="Select Input Directory",
            dir=self.homeDir,
        )
        self.loadVideo(inputFolder, multi_file=True)

    def importCustomModel(self, format: str):
        """
        *args
        format: str
            The format of the model to import (pytorch, ncnn)
        """

        if format == "pytorch":
            fileFilter = "PyTorch Model (*.pth *.safetensors)"

            modelFile, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select PyTorch Model",
                dir=self.homeDir,
                filter=fileFilter,
            )
               
            if modelFile == "":
                RegularQTPopup("Please select a model file!")
                return
            
            outputModelPath = os.path.join(
                CUSTOM_MODELS_PATH, os.path.basename(modelFile)
            )
            FileHandler.copyFile(modelFile, CUSTOM_MODELS_PATH)

            if os.path.isfile(outputModelPath):
                NotificationOverlay(
                    "Model imported successfully!\nPlease restart the app for the changes to take effect.",
                    self,
                    timeout=1500,
                )
            else:
                NotificationOverlay(
                    "Failed to import model!\nPlease try again.", self, timeout=1500
                )

        elif format == "ncnn":
            binFileFilter = "NCNN Bin (*.bin)"
            modelBinFile, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select NCNN Bin",
                dir=self.homeDir,
                filter=binFileFilter,
            )
            if modelBinFile == "":
                RegularQTPopup("Please select a bin file!")
                return
            modelParamFile, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select NCNN Param",
                dir=os.path.dirname(modelBinFile),
                filter=os.path.basename(modelBinFile).replace(".bin", ".param"),
            )
            if modelParamFile == "":
                RegularQTPopup("Please select a param file!")
                return
            outputModelFolder = os.path.join(
                CUSTOM_MODELS_PATH,
                os.path.basename(modelBinFile).replace(".bin", ""),
            )
            FileHandler.createDirectory(outputModelFolder)
            outputBinPath = os.path.join(
                outputModelFolder, os.path.basename(modelBinFile)
            )
            FileHandler.copyFile(modelBinFile, outputModelFolder)
            outputParamPath = os.path.join(
                outputModelFolder, os.path.basename(modelParamFile)
            )
            FileHandler.copyFile(modelParamFile, outputModelFolder)

            if os.path.isfile(outputBinPath) and os.path.isfile(outputParamPath):
                NotificationOverlay(
                    "Model imported successfully!\nPlease restart the app for the changes to take effect.",
                    self,
                    timeout=1500,
                )

            else:
                NotificationOverlay(
                    "Failed to import model!\nPlease try again.", self, timeout=1500
                )

       
    # output file button
    def openOutputFolder(self):
        """
        Opens home folder or the same folder as the input file only if the input file is a single file,
        sets the directory that is selected to the self.outputFolder variable
        sets the outputFileText to the output directory

        It will also read the input file name, and generate an output file based on it.
        """
        outputFolder = QFileDialog.getExistingDirectory(
            self,
            caption="Select Output Directory",
            dir=str(os.path.dirname(self.page_handles.general.inputFileText.text())) if (self.isVideoLoaded and len(self.batchVideos) == 0 and os.path.exists(os.path.dirname(self.page_handles.general.inputFileText.text()))) else self.homeDir,
        )
        self.page_handles.general.outputFileText.setText(
            os.path.join(outputFolder, self.setDefaultOutputFile(self.page_handles.general.inputFileText.text(), outputFolder))
        )

    def shutdown(self):
        """Every exit path ends here: `app.quit()` never delivers `closeEvent`.

        A quit (the control surface does it, and so does any non-window exit) left the backend render process ALIVE and
        both shared segments in `/dev/shm` - the orphan then held the GPU and the next render "did nothing" (measured
        2026-11-22: a render was still running after the window was gone). This is wired to `aboutToQuit`, which Qt
        emits however the loop ends.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        if hasattr(self, "processTab"):
            try:
                self.processTab.killRenderProcess()
            except Exception:
                pass
            try:
                self.processTab.closePausedSharedMemory()
            except Exception:
                pass

    def closeEvent(self, event):

        from src.ui.dialogs.toolkit_dialogs import confirm

        if confirm(self, "Are you sure you want to exit?"):
            FileHandler.removeFolder(TEMP_DOWNLOAD_PATH)
            self.shutdown()
            event.accept()
        else:
            event.ignore()



def main(file_to_open=None):
    import time as _time

    _t0 = _time.perf_counter()
    _timing = bool(os.environ.get("RVE_STARTUP_TIMING"))

    def _stamp(label: str) -> None:
        if _timing:
            print(f"startup {_time.perf_counter() - _t0:6.2f}s  {label}", flush=True)

    # A hung window must be dumpable without ptrace (this host runs yama ptrace_scope=1, so py-spy/gdb cannot attach):
    # `kill -USR1 <pid>` writes every thread's Python stack next to log.txt.
    try:
        from src.ui.devtools import fault_stack

        fault_stack.install(os.path.dirname(os.path.abspath(__file__)))
    except Exception as _stack_exc:
        print(f"[stack] dump handler unavailable: {_stack_exc}", flush=True)

    # The surface format stays the platform default, like every other Qt application on this machine: the reference app
    # (Telegram Desktop 7.2.9) never touches `QSurfaceFormat`, and the "desktop shows through the window" artifact turned
    # out to be environment-level (it reproduces in BlinkTable and tdesktop on the same GNOME/NVIDIA session and is
    # invisible in screenshots - projects/rve/docs/sessions/session-2026-11-22.md §20). Asking for an alpha-less surface is kept as
    # an explicit experiment, not as a default deviation.
    if os.environ.get("RVE_UI_ALPHA0"):
        from PySide6.QtGui import QSurfaceFormat

        _surface_format = QSurfaceFormat.defaultFormat()
        _surface_format.setAlphaBufferSize(0)
        QSurfaceFormat.setDefaultFormat(_surface_format)
        print("[window] RVE_UI_ALPHA0: surface format without an alpha channel", flush=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Before the theme: `toolkit_theme.install` imports the toolkit, and without the path it would "fail" into RVE's
    # own palette and log an error nobody reads - the app would then run with toolkit widgets on a stock palette.
    try:
        from src.ui import toolkit_boot
        toolkit_boot.ensure()
    except Exception as _boot_exc:
        log(f"toolkit bootstrap failed: {_boot_exc}")
    # The look has exactly one owner at a time. With a toolkit migration on, `ThemeManager` writes the palette
    # (`src/ui/toolkit_theme.py` maps RVE's colours onto its tokens); otherwise RVE's own palette is applied as
    # before. Installing the theme here, before the window exists, is deliberate: every widget is then built against
    # the palette it will keep - the previous placement (after `setupUi`) lost the race against
    # `MainWindow.setPalette(standardPalette())` and left half the tree on Fusion defaults.
    _stamp("QApplication ready")
    theme_installed = False
    try:
        from src.ui import toolkit_theme
        if toolkit_theme.enabled():
            for line in toolkit_theme.install(app):
                log("toolkit theme: " + line)
            theme_installed = True
    except Exception as _theme_exc:
        # No theme = no tokens = panels that paint nothing (a compositor shows the desktop through them) and stock
        # controls painted black. Fail visibly instead of handing out a ghost window.
        import traceback

        log("toolkit theme failed:\n" + traceback.format_exc())
        raise
    if not theme_installed:
        # The theme is off: apply RVE's own palette, which now lives with the token table it mirrors
        # (`toolkit_theme.qt_palette()`) rather than in the deleted `src/ui/QTstyle.py`.
        from src.ui.toolkit_theme import qt_palette
        app.setPalette(qt_palette())

    """if not "--unlock" in sys.argv:
        lock_file = QLockFile(LOCKFILE)
        if not lock_file.tryLock(10):
            warn(None, "Another instance is already running.", title="Instance Running")
            sys.exit(0)"""

    # setting the pallette
    _stamp("theme installed")
    window = MainWindow(file_to_open=file_to_open)
    _stamp("MainWindow built")

    # Paint the whole tree ONCE offscreen before the window is mapped: this warms the toolkit's text layout, style
    # application and the deferred re-layout, so the frame the window manager presents is already complete. Cheap
    # (one full render) and it does not depend on guessing how many event-loop turns a WM needs.
    window.ensurePolished()

    # Warm EVERY page of every stack, not just the current one. A QStackedWidget lays a page out the first time it is
    # shown, and the toolkit settles its text engine and control style on that same first paint - so each page arrived
    # with one empty frame when the user switched to it (owner report: "every content page, text ones included"). One
    # offscreen render per page, before the window is mapped, pays that cost invisibly: `grab()` renders a hidden widget.
    def _warm_widget(widget) -> None:
        """Render a widget and every page of every stack below it, then recurse into those pages.

        Recursive on purpose: an inner stack's non-current pages are not painted when its parent page is grabbed, so a
        single pass over `findChildren(QStackedWidget)` leaves the nested tabs cold (a settings tab was exactly that).
        """
        if widget is None:
            return
        widget.ensurePolished()
        widget.grab()
        for stack in widget.findChildren(QStackedWidget):
            current = stack.currentIndex()
            for index in range(stack.count()):
                page = stack.widget(index)
                if page is None:
                    continue
                page.ensurePolished()
                page.grab()
                _warm_widget(page)
            stack.setCurrentIndex(current)

    _stamp("pages polished")
    # The warm pass grabs widgets, and a grab PAINTS them. A `QRhiWidget` painted before its top-level has a platform
    # window cannot get an RHI: Qt prints `QRhiWidget: No QRhi`, emits renderFailed and - measured on headless weston
    # 2026-11-23 - draws nothing, so a pane painted in this window is literally transparent afterwards. Creating the
    # window (without showing it) before the warm pass gives the grab a backingstore to paint into.
    window.create()
    _warm_widget(window)
    window.grab()
    _stamp("first frame warmed")
    # One call, one state - `show()` after `showFullScreen()` asks the platform for a NORMAL window again, and
    # QtWayland/GNOME can end up with a state/frame that disagrees with the surface geometry.
    #
    # `--fullscreen` means MAXIMIZED, not `showFullScreen()`: borderless fullscreen is by definition a window with no
    # decorations, which the owner called "уродский, буквально без CSD" (2026-11-22). Maximized keeps the title bar and
    # still covers the screen. The borderless case is its own flag, because it is its own thing.
    if "--borderless-fullscreen" in sys.argv:
        window.showFullScreen()
    elif "--fullscreen" in sys.argv:
        window.showMaximized()
    else:
        window.show()
    # Drain the event loop before handing control to app.exec(). `show()` only QUEUES the map and the first paint, and
    # the toolkit settles its controls on 0-timers (style/scale application, text layout, deferred re-layout): without
    # these turns the first frame the window manager presents is the pre-settle one - measured as a dark page until the
    # first input. The reference host drains four turns for the same reason (its lifecycle paints and flushes frame 1
    # before releasing deferred work).
    for _ in range(4):
        app.processEvents()
    window.repaint()
    app.processEvents()
    _stamp("first frame drained")
    # `aboutToQuit` fires for every way the loop ends; `closeEvent` only for a window close, and the control surface (and
    # any `app.quit()`) bypasses it. Without this the backend render process survived the window and held the GPU.
    app.aboutToQuit.connect(window.shutdown)
    sys.exit(app.exec())


"""
custom command args
--debug: runs the app in debug mode
--fullscreen: starts maximized (decorated); --borderless-fullscreen: real fullscreen, no frame
--swap-flatpak-checks: swaps the flatpak checks, ex if the app is running in flatpak, it will run as if it is not
"""

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if "--debug" in sys.argv:
        import trace

        tracer = trace.Trace(
            ignoredirs=[sys.prefix, sys.exec_prefix], trace=True, count=False
        )
        tracer.run("main()")
    else:
        file_to_open = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
        main(file_to_open)
