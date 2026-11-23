import os
from PySide6.QtWidgets import QMainWindow
from .QTcustom import RegularQTPopup, NetworkCheckPopup, remove_combobox_item_by_text
from ..DownloadDeps import DownloadDependencies
from .Updater import ApplicationUpdater
from ..constants import IS_FLATPAK, PLATFORM, CWD, USE_LOCAL_BACKEND, HOME_PATH, PLATFORM, IS_FLATPAK, CWD, CPU_ARCH
from ..BuiltInTorchVersions import TorchVersion
from .GPUDetect import GPUDetect
from ..Util import FileHandler


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
        backends: list,
        skip_info_popup: bool = False,
    ):
        self.parent = parent
        #: The typed page surface (`window.page_handles`); the window no longer copies page widgets onto itself.
        self.page_handles = parent.page_handles
        self.torch_versions:list[TorchVersion] = [version for version in TorchVersion.__subclasses__()]
        self.downloadDeps = DownloadDependencies()
        self.backends = backends
        self.skip_info_popup = skip_info_popup
        self.applicationUpdater = ApplicationUpdater()

        self.has_enough_space = True
        if IS_FLATPAK:
            minimum_space_required = 7
        else:
            minimum_space_required = 15
        self.page_handles.download.low_storage_label.setVisible(False)
        if FileHandler.getFreeSpace() < minimum_space_required:
            self.page_handles.download.downloadTorchBtn.setEnabled(False)
            self.page_handles.download.downloadTensorRTBtn.setEnabled(False)
            self.page_handles.download.low_storage_label.setVisible(True)
            self.has_enough_space = False
        # disable as it is not complete
        try:
            self.page_handles.download.downloadDirectMLBtn.setEnabled(False)
            if PLATFORM != "win32":
                self.page_handles.download.downloadDirectMLBtn.setEnabled(False)
        except Exception as e:
            print(e)


        # set this all to not visible, as scrapping the idea for now.
        if PLATFORM != "linux":
            remove_combobox_item_by_text(self.page_handles.download.pytorch_backend, "ROCm")
        else:
            if CPU_ARCH == "arm64":
                remove_combobox_item_by_text(self.page_handles.download.pytorch_backend, "XPU")
                remove_combobox_item_by_text(self.page_handles.download.pytorch_backend, "ROCm")
        if IS_FLATPAK:
            remove_combobox_item_by_text(self.page_handles.download.pytorch_backend, "XPU")
            remove_combobox_item_by_text(self.page_handles.download.pytorch_backend, "ROCm")
        if PLATFORM == "darwin":
            if CPU_ARCH == "arm64":
                self.page_handles.download.pytorch_backend.clear()
                self.page_handles.download.pytorch_backend.addItems(
                    ["MPS (Apple Silicon)"]
                )
                # force 2.9.0 as it should include support for uint16
                self.page_handles.download.pytorch_version.setEnabled(False)

                self.page_handles.download.pytorch_backend.setCurrentText("MPS (Apple Silicon)")
                self.page_handles.download.pytorch_backend.setEnabled(False)
                
                self.page_handles.download.downloadTorchBtn.setEnabled(True)
            self.page_handles.download.downloadTensorRTBtn.setEnabled(False)
        if IS_FLATPAK or USE_LOCAL_BACKEND:
            self.page_handles.download.uninstallAppBtn.setDisabled(True)
        else:
            self.page_handles.download.uninstallAppBtn.clicked.connect(self.uninstallApp)

        self.page_handles.download.ApplicationUpdateContainer.setVisible(False)
        self.QButtonConnect()
    
    
    def QButtonConnect(self):
        self.page_handles.download.downloadNCNNBtn.clicked.connect(lambda: self.download("ncnn", True))
        self.page_handles.download.downloadTorchBtn.clicked.connect(
            lambda: self.download("torch", True)
        )
        self.page_handles.download.downloadTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", True)
        )
        self.page_handles.download.downloadDirectMLBtn.clicked.connect(
            lambda: self.download("directml", True)
        )
        self.page_handles.download.downloadRecommendedBtn.clicked.connect(
            self.installRecommended
        )
        
        self.page_handles.download.uninstallNCNNBtn.clicked.connect(
            lambda: self.download("ncnn", False)
        )
        self.page_handles.download.uninstallTorchBtn.clicked.connect(
            lambda: self.download("torch", False)
        )
        self.page_handles.download.uninstallTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", False)
        )
        self.page_handles.download.uninstallDirectMLBtn.clicked.connect(
            lambda: self.download("directml", False)
        )
        self.page_handles.download.selectPytorchCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("pytorch")
        )
        self.page_handles.download.selectNCNNCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("ncnn")
        )
        self.page_handles.download.pytorch_version.addItems(
            [version.torch_version for version in self.torch_versions]
        )

    def uninstallApp(self):
        from src.ui.dialogs.toolkit_dialogs import confirm

        if confirm(self.parent, "Are you sure you want to uninstall?"):
            os.chdir(HOME_PATH) # fix for windows, as you cant delete a directory in use.
            FileHandler().removeFolder(CWD)
            os._exit(0)
        

        
    def hideUninstallButtons(self):
        self.page_handles.download.uninstallTorchBtn.setVisible(False)
        self.page_handles.download.uninstallNCNNBtn.setVisible(False)
        self.page_handles.download.uninstallTensorRTBtn.setVisible(False)
        self.page_handles.download.uninstallDirectMLBtn.setVisible(False)

    def showUninstallButton(self, backends):
        if "pytorch (cuda)" in backends:
            self.page_handles.download.downloadTorchBtn.setVisible(False)
            self.page_handles.download.uninstallTorchBtn.setVisible(True)
        if "pytorch (rocm)" in backends:
            self.page_handles.download.downloadTorchBtn.setVisible(False)
            self.page_handles.download.uninstallTorchBtn.setVisible(True)
        if "pytorch (xpu)" in backends:
            self.page_handles.download.downloadTorchBtn.setVisible(False)
            self.page_handles.download.uninstallTorchBtn.setVisible(True)
        if "pytorch (mps)" in backends:
            self.page_handles.download.downloadTorchBtn.setVisible(False)
            self.page_handles.download.uninstallTorchBtn.setVisible(True)
        if "ncnn" in backends:
            self.page_handles.download.downloadNCNNBtn.setVisible(False)
            self.page_handles.download.uninstallNCNNBtn.setVisible(True)
        if "tensorrt" in backends:
            self.page_handles.download.downloadTensorRTBtn.setVisible(False)
            self.page_handles.download.uninstallTensorRTBtn.setVisible(True)

        # disable as it is not complete
        try:
            self.page_handles.download.downloadDirectMLBtn.setEnabled(False)
            if PLATFORM != "win32":
                self.page_handles.download.downloadDirectMLBtn.setEnabled(False)
        except Exception as e:
            print(e)

    def installRecommended(self):
        pytorch_backend = GPUDetect().getPyTorchFeatures()
        if pytorch_backend and self.has_enough_space:
            self.download("torch", install=True, pytorch_backend=pytorch_backend)
        elif PLATFORM == 'darwin' and CPU_ARCH == "arm64" and self.has_enough_space:
            self.download("torch", install=True, pytorch_backend="mps")
        else:
            self.download("ncnn")

    def download(self, dep, install: bool = True, pytorch_backend:str = None):
        """
        Downloads the specified dependency.
        Parameters:
        - dep (str): The name of the dependency to download.
        Returns:
        - None
        """
        pytorch_ver:TorchVersion|None = None
        current_pytorch_version = self.page_handles.download.pytorch_version.currentText().split()[0]
        current_pytorch_backend = self.page_handles.download.pytorch_backend.currentText().split()[0].lower() if not pytorch_backend else pytorch_backend
        for version in self.torch_versions:
            if version.torch_version == current_pytorch_version:
                pytorch_ver = version
                torchvision_ver = version.torchvision_version

        if not pytorch_ver:
            RegularQTPopup(
                "Please select a valid PyTorch version from the dropdown."
            )
            return
        if current_pytorch_backend == "cuda" or dep.lower() == "tensorrt":
            pytorch_backend = pytorch_ver.cuda_version
        elif current_pytorch_backend == "rocm":
            pytorch_backend = pytorch_ver.rocm_version
        elif current_pytorch_backend == "xpu":
            pytorch_backend = pytorch_ver.xpu_version
        elif current_pytorch_backend == "mps":
            pytorch_backend = pytorch_ver.mps_version
        
        
        if NetworkCheckPopup(
            "https://pypi.org/"
        ):  # check for network before installing
            return_code = self.downloadDeps.downloadPythonDeps(dep, pytorch_ver.torch_version, torchvision_ver, pytorch_backend.lower(), install)
            if return_code == 0 and not self.skip_info_popup:
                RegularQTPopup(
                    "Download Complete\nPlease restart the application to apply changes."
                )
            elif return_code != 0:
                RegularQTPopup("Download Failed!\nPlease check logs for more info.")
