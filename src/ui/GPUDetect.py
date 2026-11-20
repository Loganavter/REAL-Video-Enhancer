try:
    from ..constants import PLATFORM 
    from ..Util import log
except Exception:
    PLATFORM = 'win32'
    def log(msg): print(msg)
import subprocess
import re

class GPUDetect:
    def __init__(self):
        self.gpu_info = self.get_gpu_info()

    def get_gpu_info(self):
        if PLATFORM == "win32":
            try:
                output = subprocess.check_output(
                    "nvidia-smi", shell=True
                ).decode()
                return str(output.strip().split("\n"))
            except Exception:
                return "Unable to retrieve GPU info on Windows"

        elif PLATFORM == "darwin":  # macOS
            try:
                output = subprocess.check_output(
                    "system_profiler SPDisplaysDataType | grep Vendor", shell=True
                ).decode()
                return output.strip().split(":")[1].strip()
            except Exception:
                return "Unable to retrieve GPU info on macOS"

        elif PLATFORM == "linux":
            try:
                # Try lspci command first
                output = subprocess.check_output("lspci | grep -i vga", shell=True).decode()
                return output.strip().split(":")[2].strip()
            except Exception:
                try:
                    # If lspci fails, try reading from /sys/class/graphics
                    with open("/sys/class/graphics/fb0/device/vendor", "r") as f:
                        vendor_id = f.read().strip()
                    return f"Vendor ID: {vendor_id}"
                except Exception:
                    return "Unable to retrieve GPU info on Linux"

        else:
            return "Unsupported operating system"


    def getVendor(self):
        """
        Gets GPU vendor of the system
        vendors = ["Intel", "AMD", "Nvidia"]
        """
        vendors = ["Intel", "AMD", "Nvidia"]
        for vendor in vendors:
            print(self.gpu_info)
            if vendor.lower() in self.gpu_info.lower():
                return vendor
        return None
    
    def getModelOfGPU(self):
        vendor = self.getVendor()
        model = "0"
        if vendor == "Nvidia":
            try:
                model = re.findall(r"RTX \d\d\d\d", self.gpu_info)[0]
                log("GPU Model Found: " + vendor + " " + model)
            except Exception:
                log("Couldnt find gpu model, " + self.gpu_info)
        return model
    
    def getPyTorchFeatures(self) -> str | None:
        try:
            if int(self.getModelOfGPU()[4]) >= 2 and self.getVendor() == "Nvidia":
                return "cuda"
        except Exception:
            return None
        return None
    
if __name__ == '__main__':
    print(GPUDetect().getPyTorchFeatures())


# --- what the int8 body needs from the card --------------------------------------------------------------
#
# The Fast preset swaps the model's body for Triton kernels that accumulate in int32 (`backend/src/pytorch/
# compact/int8_kernels.py`, `tl.dot(..., out_dtype=tl.int32)`). `tl.dot` with int8 lowers to the integer tensor-core
# instruction, which exists from Turing (compute capability 7.5) onward; below that Triton has nothing to lower to,
# so the body is not a slower path there - it is an unavailable one.
#
# Unknown capability does NOT hide the option: on a machine where neither nvidia-smi nor the backend's torch can
# answer, the probe still decides at render time, and hiding a working configuration is worse than offering one
# that may fall back to fp16.

INT8_MIN_CAPABILITY = (7, 5)


def compute_capability():
    """(major, minor) of the first CUDA GPU, or None when it cannot be determined without a full torch import."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        major, minor = out.strip().splitlines()[0].strip().split(".")
        return int(major), int(minor)
    except Exception:
        pass
    try:
        from ..constants import PYTHON_EXECUTABLE_PATH
        result = subprocess.run(
            [PYTHON_EXECUTABLE_PATH, "-c", "import torch;print(*torch.cuda.get_device_capability(0))"],
            capture_output=True, text=True, timeout=40,
        )
        if result.returncode == 0 and result.stdout.strip():
            major, minor = result.stdout.split()[:2]
            return int(major), int(minor)
    except Exception:
        pass
    return None


def int8_decision(vendor, capability):
    """Pure decision, so it can be tested without a GPU: (available, reason)."""
    if vendor is not None and vendor != "Nvidia":
        return False, f"{vendor} GPU: the int8 body is a CUDA/Triton kernel path"
    if capability is None:
        return True, "compute capability unknown; the probe decides at render time"
    if tuple(capability) < INT8_MIN_CAPABILITY:
        return False, (f"compute capability {capability[0]}.{capability[1]} is below "
                       f"{INT8_MIN_CAPABILITY[0]}.{INT8_MIN_CAPABILITY[1]}: no int8 tensor cores")
    return True, f"compute capability {capability[0]}.{capability[1]}"


def int8_body_available():
    """Whether the Fast preset can run here: (available, human-readable reason)."""
    capability = compute_capability()
    vendor = "Nvidia" if capability is not None else GPUDetect().getVendor()
    return int8_decision(vendor, capability)

