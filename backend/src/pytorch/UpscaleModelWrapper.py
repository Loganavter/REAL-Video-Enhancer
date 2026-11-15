import os

import torch
import torch.nn.functional as F
from ..utils.Util import log
from .TorchUtils import TorchUtils
from .VSRArchs.AnimeSR.vsr_inference_helper import AnimeSRInferenceHelper
from .VSRArchs.AnimeSR.animesr_arch import AnimeSR
from .VSRArchs.TSPAN.vsr_inference_helper import TemporalSPANInferenceHelper
from .VSRArchs.TSPAN.tspan import TemporalSPAN
HEAD_PAD_MULTIPLE = 8


class PadInputConv(torch.nn.Module):
    """A convolution that zero-pads its input to the channel count it was widened to.

    The padding has to live *inside* the head, not in front of the model: these networks add an
    upsampled copy of their own input to their output (`SRVGGNetCompact.forward`), so an 8-channel
    input reaches that residual as an 8-channel tensor against a 3-channel output and the forward
    fails. Padding here leaves the model's input - and therefore its residual - at 3 channels while
    the head convolution sees a shape the engine can fuse.

    Measured on the compact 2x model (ledger E19): the head is the only 3-channel convolution, and at
    C=3 TensorRT does not put it into its myelin pattern compiler at all - it builds as a plain
    convolution plus a separate pointwise pass for its PReLU, 57.51 ms for the frame. Padded to 8
    channels (fp16 tensor cores want C % 8 == 0) the whole head block becomes one fused
    `__myl_CorrMulLtSele` kernel and the frame takes 54.15 ms, i.e. -5.8%, with a picture that differs
    by 0.000% of pixels above 8 levels and 78.69 dB. The pad op folds away - the engine has no kernel
    for it - and the engine's input stays 3-channel, so nothing outside the engine changes.
    """

    def __init__(self, conv: torch.nn.Conv2d, extra: int):
        super().__init__()
        self.conv = conv
        self.extra = int(extra)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def in_channels(self):
        return self.conv.in_channels - self.extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.extra:
            # `torch.cat` with a zero tensor, NOT `F.pad`. Both are the same function, but
            # `aten.constant_pad_nd` has no converter in torch_tensorrt's dynamo path, so the node
            # falls back to torch and splits the engine into a TRT sub-graph either side of a torch
            # op: measured on the shipping model that is 26.76 ms/frame of engine time against 16.07
            # unpadded, i.e. the padding's 5.8% turns into a 67% regression. With `cat` (an
            # IConcatenationLayer) the same graph measures 15.43 ms - 4.0% better than unpadded,
            # which is what 5.8% of a 67.6%-of-the-frame engine must give. See ledger E23.
            zeros = torch.zeros((x.shape[0], self.extra, x.shape[2], x.shape[3]),
                                dtype=x.dtype, device=x.device)
            x = torch.cat((x, zeros), dim=1)
        return self.conv(x)


def pad_head_channels(model: torch.nn.Module, multiple: int = HEAD_PAD_MULTIPLE):
    """Widen the first convolution to a multiple of `multiple` input channels and pad behind it.

    Returns ``(model, extra)``; ``extra`` is 0 when nothing was done, which is also the signal the
    engine cache key uses. Only a group-1 convolution with ``0 < in_channels % multiple`` is touched,
    and only in the spandrel path (the other loaders shape their inputs themselves).
    """
    # ON by default, RVE_NO_HEAD_PAD=1 disables it. Measured through this app's own build path,
    # interleaved, three rounds, engines cached (ledger E19/E23):
    #
    #   no padding   11.79 s wall / 16.18 ms engine
    #   padded       11.44 s wall / 15.61 ms engine     (-3.0% wall, -3.5% engine, 3/3 rounds)
    #
    # with the picture differing by 0.012% of pixels above 8 levels at the worst frame. It applies
    # only to a plain 3-channel head, and a 32x32 probe forward reverts it for anything it breaks
    # (the SPAN family fuses its 3x3 and 1x1 weights, so widening one branch breaks the sum).
    if os.environ.get("RVE_NO_HEAD_PAD", "0") == "1":
        return model, 0
    first = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            first = (name, module)
            break
    if first is None:
        return model, 0
    name, conv = first
    channels = conv.in_channels
    # Only the case that was measured: a plain 3-channel head. A 12-channel head (a multi-frame
    # model) or an already aligned one is left alone.
    if channels != 3 or conv.groups != 1:
        return model, 0
    extra = multiple - channels % multiple
    weight = conv.weight.data
    widened = torch.nn.Conv2d(
        channels + extra, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups, bias=conv.bias is not None,
        padding_mode=conv.padding_mode)
    with torch.no_grad():
        widened.weight.zero_()
        widened.weight[:, :channels].copy_(weight)
        if conv.bias is not None:
            widened.bias.copy_(conv.bias)
    widened = widened.to(device=weight.device, dtype=weight.dtype)
    parent_path, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    original = conv
    setattr(parent, attr, PadInputConv(widened, extra))
    # Verify on a tiny probe before anyone renders with it. Two structures break the idea and neither
    # is detectable by inspection: SPAN's head is a Conv3XC that concatenates its 3x3 and 1x1 weights
    # (`torch.cat` along the channel axis), so widening one branch makes the pair 8 against 3, and any
    # model that adds a copy of its input to its output sees 8 channels against 3. Ten of the 78
    # models in the owner's collection are one of those; reverting here costs a 32x32 forward.
    try:
        with torch.inference_mode():
            model(torch.zeros(1, channels, 32, 32, dtype=widened.weight.dtype,
                              device=widened.weight.device))
    except Exception as error:
        setattr(parent, attr, original)
        log(f"head padding skipped for {name}: {type(error).__name__}: "
            f"{str(error).splitlines()[0][:90]}")
        return model, 0
    log(f"head convolution {name}: {channels} input channels -> {channels + extra} "
        f"(zeros, padded inside the head so the model's own residual keeps its 3 channels), "
        f"so the engine can fuse it with its PReLU")
    return model, extra


class UpscaleModelWrapper:
    def __init__(self, model_path: torch.nn.Module, device: torch.device, precision: torch.dtype):
        self.__model_path = model_path
        self.__device = device
        self.__precision = precision
        self.__dummy_input_pre_channels = None
        self.__channels = 3
        self.__head_pad = 0
        self.__inference_mode = None
        self.load_model()
        self.set_precision(self.__precision)
        self.__test_model_precision()

    def set_precision(self, precision: torch.dtype):
        self.__precision = precision
        self.__model.to(self.__device, dtype=precision)

    def get_model(self):
        return self.__model
    
    def get_scale(self):
        return self.__scale

    def get_head_pad(self) -> int:
        """Channels of zero padding applied to the head convolution (0 = untouched)."""
        return int(self.__head_pad or 0)

    def load_state_dict(self, state_dict):
        self.__model.load_state_dict(state_dict)

    def __test_inference(self, test_input:torch.Tensor):
        # inference and get re-load state dict due to issue with span.
        with torch.inference_mode():
            model = self.inference_helper
            model(test_input)
            output = model(test_input)
            self.__model.load_state_dict(model.state_dict()) # reload state dict to fix span
            TorchUtils.clear_cache()
            del model

    def __test_model_precision(self):
        test_input = torch.randn(1, 3, 64, 64).to(self.__device, dtype=self.__precision)
        with torch.inference_mode():
            try:
                self.__test_inference(test_input)
            except Exception as e:
                log(f"Model precision {self.__precision} not supported, falling back to float32: {e}")
                self.set_precision(torch.float32)
                self.__test_inference(test_input)
    
    def get_dummy_input(self, width: int, height: int) -> torch.Tensor:
        assert self.__dummy_input_pre_channels is not None, "Dummy input pre channels not set."
        dummy_input = self.__dummy_input_pre_channels.copy()
        dummy_input.append(self.__channels)
        dummy_input.append(height)
        dummy_input.append(width)
        return torch.zeros(dummy_input, dtype=self.__precision, device=self.__device)
    
    @torch.inference_mode()
    def load_model(self, model=None) -> torch.nn.Module:
        if not model:
            from .spandrel import ModelLoader, ImageModelDescriptor, UnsupportedModelError
            try:
                model = ModelLoader().load_from_file(self.__model_path)
                assert isinstance(model, ImageModelDescriptor)
                self.__scale = model.scale
                model = model.model
                self.__model = model
                self.__model, self.__head_pad = pad_head_channels(self.__model)
                self.inference_helper = self.__model
                self.__dummy_input_pre_channels = [1]
                self.__inference_mode = 'spandrel'

            except (UnsupportedModelError) as e:
                try:
                    model = AnimeSR()
                    # dummy attributes
                    self.__scale = 4
                    state_dict = torch.load(self.__model_path, map_location=self.__device)
                    model.load_state_dict(state_dict=state_dict)
                    self.__model = model.to(self.__device, dtype=self.__precision)
                    self.inference_helper = AnimeSRInferenceHelper(model=self.__model, scale=self.__scale)
                    self.__dummy_input_pre_channels = [3]
                    self.__inference_mode = 'animesr'
                except Exception as e:
                    try:
                        self.__scale = 2
                        self.__dummy_input_pre_channels = [1, 5,]
                        model = TemporalSPAN(upscale=self.__scale)
                        state_dict = torch.load(self.__model_path, map_location=self.__device)
                        model.load_state_dict(state_dict=state_dict['params_ema'], strict=False)
                        self.__model = model.to(self.__device, dtype=self.__precision)
                        self.inference_helper = TemporalSPANInferenceHelper(model=self.__model, scale=self.__scale)
                        self.__inference_mode = 'tspan'
                    except Exception as e:
                        log(f"Model at {self.__model_path} is not supported: {e}")
                        raise e
        else:
            if self.__inference_mode == 'spandrel':
                self.inference_helper = model
            elif self.__inference_mode == 'animesr':
                self.inference_helper = AnimeSRInferenceHelper(model=model, scale=self.__scale)
            elif self.__inference_mode == 'tspan':
                self.inference_helper = TemporalSPANInferenceHelper(model=model, scale=self.__scale)
    def __call__(self, *args, **kwargs):
        assert self.inference_helper is not None, "Inference helper is not initialized."
        # no .clone(): the caller (UpscaleTorch -> Frame.set_frame_tensor) already copies,
        # and a second full-HR copy costs ~4.6 ms/frame at 4x
        return self.inference_helper(*args, **kwargs)