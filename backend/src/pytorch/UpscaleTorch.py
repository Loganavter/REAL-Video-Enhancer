import os
import math

import gc
from .TorchUtils import TorchUtils
from .UpscaleModelWrapper import UpscaleModelWrapper
import torch as torch
import torch.nn.functional as F
import sys
import time
from time import sleep

from ..utils.Util import log, CudaChecker
from ..utils.Frame import Frame

# Diagnostic: with RVE_TRT_SPLIT=1 the model stream is synced before the output Frame is
# built, so `last_model_ms` is attributable to the engine instead of overlapping with
# the Frame's construction. It costs that overlap, so it is never a production path.
_SPLIT_TIMING = os.environ.get("RVE_TRT_SPLIT") == "1"


def pipe_is_yuv8(high_bitdepth: bool, pipe_pixel_format: str) -> bool:
    """Does the packer emit 8-bit planar yuv420p on this pipe?

    The WRITER's answer is `FFmpegBuffers._pipe_input_pixel_format()`, and it reads the same string, so
    this must be derived from the pipe format and nothing else. It used to be derived from the bit depth
    alone (`RVE_YUV8 != 0 and not high_bitdepth`), which diverges on exactly one combination: an 8-bit
    source that DECLARES a non-bt709 matrix. `RenderVideo` then keeps rgb24 (so ffmpeg applies the
    source's own matrix instead of the packer's BT.709) while the packer still packed yuv - 1.5 bytes per
    pixel on a pipe the writer read as 3, i.e. a torn frame with nothing failing loudly (found 2026-11-20,
    row rve-yuv8-flag-divergence-2026-11-20; the same class as the P010/planar mismatch, ledger C4).
    `RVE_YUV8=0` still opts out: it leaves the pipe at `yuv420p10le`, which this reads as "not yuv8".
    """
    return (not high_bitdepth) and pipe_pixel_format == "yuv420p"

HAS_PYTORCH_CUDA = CudaChecker().HAS_PYTORCH_CUDA
import numpy as np
def process_output(output, hdr_mode):
    # Step 1: Squeeze the first dimension
    output = np.squeeze(output, axis=0)

    # Step 2: Permute axes from (C, H, W) to (H, W, C)
    output = np.transpose(output, (1, 2, 0))

    # Step 3: Clamp values to the range [0, 1]
    output = np.clip(output, 0, 1)

    # Step 4: Multiply by scaling factor
    scale_factor = 65535.0 if hdr_mode else 255.0
    output = output * scale_factor

    # Step 5: Convert to appropriate dtype
    dtype = np.uint16 if hdr_mode else np.uint8
    output = output.astype(dtype)

    return output

class UpscalePytorch:
    """A class for upscaling images using PyTorch.

    Args:
        modelPath (str): The path to the model file.
        device (str, optional): The device to use for inference. Defaults to "default".
        tile_pad (int, optional): The padding size for tiles. Defaults to 10.
        precision (str, optional): The precision mode for the model. Defaults to "auto".
        width (int, optional): The width of the input image. Defaults to 1920.
        height (int, optional): The height of the input image. Defaults to 1080.
        backend (str, optional): The backend for inference. Defaults to "pytorch".
        trt_workspace_size (int, optional): The workspace size for TensorRT. Defaults to 0.
        trt_cache_dir (str, optional): The cache directory for TensorRT. Defaults to modelsDirectory().

    Attributes:
        tile_pad (int): The padding size for tiles.
        dtype (torch.dtype): The data type for the model.
        device (torch.device): The device used for inference.
        model (torch.nn.Module): The loaded model.
        width (int): The width of the input image.
        height (int): The height of the input image.
        scale (float): The scale factor of the model.

    Methods:
        handlePrecision(precision): Handles the precision mode for the model.
        loadModel(modelPath, dtype, device): Loads the model from file.
        bytesToFrame(frame): Converts bytes to a torch tensor.
        tensorToNPArray(image): Converts a torch tensor to a NumPy array.
        renderImage(image): Renders an image using the model.
        renderToNPArray(image): Renders an image and returns it as a NumPy array.
        renderImagesInDirectory(dir): Renders all images in a directory.
        getScale(): Returns the scale factor of the model.
        saveImage(image, fullOutputPathLocation): Saves an image to a file.
        renderTiledImage(image, tile_size): Renders a tiled image."""

    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        device="default",
        tile_pad: int = 10,
        precision: str = "auto",
        width: int = 1920,
        height: int = 1080,
        tilesize: int = 0,
        backend: str = "pytorch",
        gpu_id: int = 0,
        hdr_mode: bool = False,
        high_bitdepth: bool | None = None,
        override_upscale_scale: int | None = None,
        # trt options
        trt_workspace_size: int = 0,
        trt_optimization_level: int = 5,
        trt_tiling_level: str = "full",
        trt_timing_iters: int = 8,
        trt_streams: int = 1,
        pipe_pixel_format: str = "yuv420p10le",
        tile_reuse: bool = False,
        tile_lr: int = 512,
        tile_halo: int = 18,
        tile_threshold: float = 0.35,
        tile_max_chain: int = 2,
        trt_max_aux_streams: int | None = None,
        trt_debug: bool = False,
        trt_static_shape: bool = False,

    ):
        self.torchUtils = TorchUtils(width=width, height=height,hdr_mode=hdr_mode,device_type=device,
                                     high_bitdepth=high_bitdepth)  
        device = self.torchUtils.handle_device(device, gpu_id=gpu_id)
        self.dtype = self.torchUtils.handle_precision(precision)
        # the tile path's halo; the experiment that decides whether 18 is enough needs to vary
        # it without a rebuild flag, so an env override is honoured (RVE_TILE_HALO)
        self.tile_pad = int(os.environ.get("RVE_TILE_HALO", tile_pad) or tile_pad)
        self.device = device
        self.videoWidth = width
        self.videoHeight = height
        self.tilesize = tilesize
        self.tile = [self.tilesize, self.tilesize]
        self.modelPath = modelPath
        self.backend = backend
      
        self.trt_workspace_size = trt_workspace_size
        self.trt_optimization_level = trt_optimization_level
        self.trt_tiling_level = trt_tiling_level
        self.trt_timing_iters = trt_timing_iters
        # >1 loads that many independent engine contexts and pipelines frames through them:
        # one execution context cannot be used from two CUDA streams, so the win has to come
        # from separate contexts (the approach vs-mlrt reports as 1.7-2.0x on Ada).
        # MEASURED DEAD END (2026-11-15): with 2 contexts on this app's torchscript engines the
        # render time does not change (4.25 -> 4.27 s for 94 frames) and the output diverges
        # from the single-stream one (Y-PSNR 25 dB, not a frame shift), i.e. two torch.jit.load
        # modules of the same engine do not give independent execution contexts. Kept behind
        # the flag for reference; anything >1 falls back to 1 with a warning.
        self.num_streams = max(1, int(trt_streams))
        if self.num_streams > 1:
            from ..utils.Util import log as _log
            _log("trt_streams > 1 is not supported (shared execution context corrupts output); using 1")
            self.num_streams = 1
        self._slot_models = []
        self._slot_streams = []
        self._slot_busy = []
        self._slot_i = 0
        self._pending = []          # [(tensor, event)] results not yet emitted
        self._slot_models_ready = False
        self.tile_reuse = tile_reuse
        self.tile_threshold = tile_threshold
        self.tile_max_chain = tile_max_chain
        self.tile_upscaler = None
        self._prev_hr = None
        if tile_reuse:
            # the model is fully convolutional: run it on tile windows instead of whole frames
            self.tilesize = tile_lr
            self.tile = [tile_lr, tile_lr]
            self.tile_pad = int(os.environ.get("RVE_TILE_HALO", tile_halo) or tile_halo)
        self.trt_aux_streams = trt_max_aux_streams
        self.trt_debug = trt_debug
        
        self.hdr_mode = hdr_mode         
        self.high_bitdepth = hdr_mode if high_bitdepth is None else high_bitdepth
        #: Luma matrix the 10-bit pipe packer must use; set by the renderer from the SOURCE's colour space (this
        #: app's "HDR mode" is also the 10-bit deep-colour switch, so the flag cannot decide the matrix: a 10-bit
        #: bt709 clip packed with bt2020 coefficients measured mean |d| 3.72 against the rgb48 reference where bt709
        #: measured 1.92).
        self.yuv10_matrix = None
        # 8-bit planar YUV pipe, ON by default since 2026-11-18 (RVE_YUV8=0 reverts to rgb24).
        # Half the host traffic of rgb24 and no ffmpeg scale filter; the A/B that cleared it is in
        # harness/docs/PERFORMANCE.md (two titles: mean |delta| 0.12-0.17 levels, worst frame 1 px over the
        # 8-level bar on one title, none on the other).
        # The decision follows the PIPE FORMAT, not the bit depth: the writer is told the same string and
        # these two must never disagree (see pipe_is_yuv8).
        self.pipe_pixel_format = pipe_pixel_format
        self.yuv8_output = pipe_is_yuv8(self.high_bitdepth, pipe_pixel_format)
        self.gpu_id = gpu_id
        # stage profiling (read by RenderVideo's summary)
        self.last_inference_ms = 0.0
        self.last_convert_ms = 0.0

        self.trt_static_shape = trt_static_shape

        # streams
        self.stream = self.torchUtils.init_stream(gpu_id=gpu_id)
        self.f2tstream = self.torchUtils.init_stream(gpu_id=gpu_id)
        self.prepareStream = self.torchUtils.init_stream(gpu_id=gpu_id)
        self.convertStream = self.torchUtils.init_stream(gpu_id=gpu_id)
        self._load()

    @torch.inference_mode()
    def _load(self):

        self.trt_min_shape = [128, 128]
        self.trt_opt_shape = [1920, 1080]
        self.trt_max_shape = [1920, 1920]
        

        if self.videoWidth > 1920 or self.videoHeight > 1920 and not self.trt_static_shape:
            log("The video resolution is very large for TensorRT dynamic shape and will use a lot of VRAM, falling back to static shape")
            self.trt_static_shape = True

        if self.videoWidth < 128 or self.videoHeight < 128 and not self.trt_static_shape:
            log("The video resolution is too small for TensorRT dynamic shape, falling back to static shape")
            self.trt_static_shape = True

        
        with self.torchUtils.run_stream(self.prepareStream):
            self.upscale_model_wrapper = UpscaleModelWrapper(
                model_path=self.modelPath,
                device=self.device,
                precision=self.dtype
            )
            self.scale = self.upscale_model_wrapper.get_scale()
            
            match self.scale:
                case 1:
                    modulo = 4
                case 2:
                    modulo = 2
                case _:
                    modulo = 1
            if self.tile_reuse:
                from .TiledUpscale import TiledUpscaler, divisor_tile
                # A square core that does not divide the frame pays for a full halo window on a
                # clipped tail: 1920x1080 at 512 px is 12 windows of 548x548, i.e. 1.74x the
                # frame's area for 1.0x of its pixels. Sizing the cores to divide each axis
                # (480x540 here) makes the grid exact: 8 windows, 1.15x.
                self.tile_h = divisor_tile(self.videoHeight, self.tilesize)
                self.tile_w = divisor_tile(self.videoWidth, self.tilesize)
                self.tile_upscaler = TiledUpscaler(
                    tile_lr=self.tile_h,
                    tile_w=self.tile_w,
                    halo_lr=self.tile_pad,
                    threshold=self.tile_threshold,
                    max_chain=self.tile_max_chain,
                    # RVE_TILE_STAT=max judges a tile by its worst block instead of its mean:
                    # a mean lets a moving object inside one tile average itself away, which is
                    # the same blindness the frame gate had (measured: peaks of 98 levels).
                    tile_stat=os.environ.get("RVE_TILE_STAT", "mean"),
                )
                engine_h, engine_w = self.tile_upscaler.tile_shape
                self.pad_w = math.ceil(engine_w / modulo) * modulo
                self.pad_h = math.ceil(engine_h / modulo) * modulo
                tiles = (math.ceil(self.videoHeight / self.tile_h)
                         * math.ceil(self.videoWidth / self.tile_w))
                log(f"Tile reuse enabled: {self.tile_w}x{self.tile_h} LR cores (WxH) + "
                    f"{self.tile_pad}px halo -> engine {self.pad_w}x{self.pad_h} (WxH), "
                    f"{tiles} tiles, "
                    f"{tiles * self.pad_w * self.pad_h / (self.videoWidth * self.videoHeight):.3f}x "
                    f"the frame's area")
            elif all(t > 0 for t in self.tile):
                self.pad_w = (
                    math.ceil(
                        min(self.tile[0] + 2 * self.tile_pad, self.videoWidth) / modulo
                    )
                    * modulo
                )
                self.pad_h = (
                    math.ceil(
                        min(self.tile[1] + 2 * self.tile_pad, self.videoHeight) / modulo
                    )
                    * modulo
                )
            else:
                modulo = 1 if self.videoWidth < 720 or self.videoHeight < 720 else 1
                self.pad_w = math.ceil(self.videoWidth / modulo) * modulo
                self.pad_h = math.ceil(self.videoHeight / modulo) * modulo

            if self.backend == "tensorrt":
                self.tensorrt_example_inputs = (self.upscale_model_wrapper.get_dummy_input(self.pad_w, self.pad_h),) # gotta make a tuple cause im dumb.
                from .TensorRTHandler import TorchTensorRTHandler

                trtHandler = TorchTensorRTHandler(
                    model_parent_path=os.path.dirname(self.modelPath),
                    trt_optimization_level=self.trt_optimization_level,
                    # RVE_TRT_DEBUG=1 makes torch_tensorrt print how it partitioned the graph - which
                    # ops stayed in torch and how many TRT sub-engines it built. That is the only way
                    # to see whether a graph change splits the engine (ledger E23).
                    debug=os.environ.get("RVE_TRT_DEBUG") == "1",

                )
                static_dimensions = f"{self.pad_w}x{self.pad_h}"
                
                for i in range(2):
                    self.trt_min_shape[i] = math.ceil(self.trt_min_shape[i] / modulo) * modulo
                    self.trt_opt_shape[i] = math.ceil(self.trt_opt_shape[i] / modulo) * modulo
                    self.trt_max_shape[i] = math.ceil(self.trt_max_shape[i] / modulo) * modulo

                dynamic_dimensions = (
                    f"min-{self.trt_min_shape[0]}x{self.trt_min_shape[1]}"
                    f"_opt-{self.trt_opt_shape[0]}x{self.trt_opt_shape[1]}"
                    f"_max-{self.trt_max_shape[0]}x{self.trt_max_shape[1]}"
                )
                self.trt_engine_name = os.path.join(
                    (
                        f"{os.path.basename(self.modelPath)}"
                        + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                        + f"_{torch.cuda.get_device_name(self.device)}"
                        + f"_trt-{trtHandler.tensorrt_version}"
                        + f"_torch_tensorrt-{trtHandler.torch_tensorrt_version}"
                        + f"_opt-{self.trt_optimization_level}"
                        # tiling is not part of the plan's own metadata, so it has to be
                        # in the file name or a stale non-tiled engine would be reused
                        + (f"_tiling-{self.trt_tiling_level}" if self.trt_tiling_level != "none" else "")
                        # the head padding is a graph change, not engine metadata: without it in the
                        # name a cached unpadded engine would be served for a padded model (and the
                        # other way round). Ledger E19.
                        + (
                            f"_headpad{self.upscale_model_wrapper.get_head_pad()}"
                            if getattr(self, "upscale_model_wrapper", None) is not None
                            and self.upscale_model_wrapper.get_head_pad()
                            else ""
                        )
                        + (
                            f"_workspace-{self.trt_workspace_size}"
                            if self.trt_workspace_size > 0
                            else ""
                        )
                    ),
                )
                self.trt_engine_static_name = self.trt_engine_name + f"_{static_dimensions}.engine"
                self.trt_engine_dynamic_name = self.trt_engine_name + f"_{dynamic_dimensions}.engine"
                self.trt_engine_name = self.trt_engine_static_name if self.trt_static_shape else self.trt_engine_dynamic_name

                if not trtHandler.check_engine_exists(self.trt_engine_name):
                    
                    if self.trt_static_shape:
                        dynamic_shapes = None
                        
                    else:
                        self.trt_min_shape.reverse()
                        self.trt_opt_shape.reverse()
                        self.trt_max_shape.reverse()

                        _height = torch.export.Dim("height", min=self.trt_min_shape[0] // modulo, max=self.trt_max_shape[0] // modulo)
                        _width = torch.export.Dim("width", min=self.trt_min_shape[1] // modulo, max=self.trt_max_shape[1] // modulo)
                        dim_height = _height * modulo
                        dim_width = _width * modulo
                        dynamic_shapes = {"x": {2: dim_height, 3: dim_width}}

                    try:
                        trt_engine = trtHandler.build_engine(
                            self.upscale_model_wrapper.get_model(),
                            self.dtype,
                            self.device,
                            example_inputs=self.tensorrt_example_inputs,
                            trt_engine_name=self.trt_engine_name,
                            trt_multi_precision_engine=False,
                            num_avg_timing_iters=self.trt_timing_iters,
                            tiling_optimization_level=self.trt_tiling_level,
                            dynamic_shapes=dynamic_shapes,
                        )
                        trtHandler.save_engine(
                            trt_engine,
                            self.trt_engine_name,
                            example_inputs=self.tensorrt_example_inputs,
                        )

                    except Exception as e:
                        if dynamic_shapes is not None:
                            print(f"ERROR: building TensorRT engine with dynamic shapes, trying without.\n", file=sys.stderr)

                            if trtHandler.check_engine_exists(self.trt_engine_static_name):
                                trtHandler.load_engine(
                                    trt_engine_name=self.trt_engine_static_name
                                )

                            else:

                                trt_engine = trtHandler.build_engine(
                                    self.upscale_model_wrapper.get_model(),
                                    self.dtype,
                                    self.device,
                                    example_inputs=self.tensorrt_example_inputs,
                                    trt_engine_name=self.trt_engine_static_name,
                                    trt_multi_precision_engine=False,
                            num_avg_timing_iters=self.trt_timing_iters,
                            tiling_optimization_level=self.trt_tiling_level,
                                )
                                trtHandler.save_engine(
                                    trt_engine,
                                    self.trt_engine_static_name,
                                    example_inputs=self.tensorrt_example_inputs,
                                )
                            self.trt_engine_name = self.trt_engine_static_name
                        else:
                            raise RuntimeError(
                                f"Failed to build TensorRT engine: {e}\n"
                            )
                model = trtHandler.load_engine(trt_engine_name=self.trt_engine_name)
                self.upscale_model_wrapper.load_model(model)
                if self.num_streams > 1:
                    engine_path = os.path.join(
                        os.path.dirname(self.modelPath),
                        self.trt_engine_name + TorchTensorRTHandler.trt_path_appendix,
                    )
                    self._slot_models = [model]
                    for _ in range(self.num_streams - 1):
                        self._slot_models.append(torch.jit.load(engine_path).eval())
                

        self.torchUtils.clear_cache()
        self.torchUtils.sync_all_streams()

    @torch.inference_mode()
    def hotUnload(self):
        self.upscale_model_wrapper = None
        gc.collect()
        self.torchUtils.clear_cache()
        if HAS_PYTORCH_CUDA:
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()

    @torch.inference_mode()
    def hotReload(self):
        self._load()
    
    @torch.inference_mode()
    def __call__(self, image: Frame) -> Frame:
        if self.num_streams > 1 and self.tile_upscaler is None and self.backend == "tensorrt":
            return self._call_pipelined(image)
        return self._call_single(image)

    @torch.inference_mode()
    def _call_pipelined(self, image: Frame) -> Frame:
        """Run frames through N independent engine contexts, emitting N-1 frames late.

        One execution context cannot serve two streams, so each slot owns its own engine.
        Frame t goes to slot t % N; the frame emitted is t - (N-1), i.e. latency grows by
        N-1 frames and the GPU keeps N inferences in flight. The last N-1 frames come out
        through flush(), which RenderVideo calls when the reader ends.
        """
        dtype = image.dtype
        gpu_id = image.gpu_id
        device = image.device
        _t0 = time.perf_counter()
        with self.torchUtils.run_stream(self.f2tstream):  # type: ignore
            image_tensor = image.get_frame_tensor()
        del image
        self.torchUtils.sync_stream(self.f2tstream)
        self.last_convert_ms = (time.perf_counter() - _t0) * 1000.0

        if not self._slot_models_ready:
            self._slot_streams = [self.torchUtils.init_stream(gpu_id=gpu_id)
                                  for _ in range(self.num_streams)]
            self._slot_busy = [False] * self.num_streams
            self._slot_models_ready = True

        slot = self._slot_i
        self._slot_i = (self._slot_i + 1) % self.num_streams
        stream = self._slot_streams[slot]
        if self._slot_busy[slot]:
            # the previous frame on this slot must be done with its input buffer
            torch.cuda.current_stream().wait_stream(stream)
        with self.torchUtils.run_stream(stream):
            out = self._slot_models[slot](image_tensor)
        event = torch.cuda.Event()
        event.record(stream)
        self._slot_busy[slot] = True
        self._pending.append((out, event))

        if len(self._pending) <= self.num_streams - 1:
            # first passes: nothing has completed a full lap yet -> keep latency honest by
            # returning nothing and letting RenderVideo skip the frame
            return None
        result, ev = self._pending.pop(0)
        ev.synchronize()
        retFrame = Frame(self.backend, self.videoWidth * self.scale, self.videoHeight * self.scale,
                         device=device, gpu_id=gpu_id, hdr_mode=self.hdr_mode, dtype=dtype,
                         high_bitdepth=self.high_bitdepth)
        # 10-bit output ships the pipe as yuv420p10le, HDR included: the packer converts with the source's matrix
        # (`bt2020` under HDR) instead of handing ffmpeg rgb48le, which is 6 bytes/pixel for a frame the encoder stores
        # at 3 (measured 2026-11-23: 37.3 vs 18.7 MB/frame, writer 15.5 vs 6.9 ms/frame).
        retFrame.yuv10_output = bool(self.high_bitdepth) and self._pack_yuv()
        retFrame.yuv10_matrix = self._pipe_matrix()
        retFrame.yuv8_output = self.yuv8_output
        retFrame.set_frame_tensor(result)
        return retFrame

    @staticmethod
    def _pack_yuv() -> bool:
        """Whether the pipe carries packed yuv420p10le (`RVE_PIPE=rgb48le` restores the old RGB pipe for an A/B)."""
        return str(os.environ.get("RVE_PIPE", "")).lower() != "rgb48le"

    def _pipe_matrix(self) -> str:
        """The luma coefficients for the 10-bit packer: the source's own, else bt709 (SDR) / bt2020 (HDR)."""
        return str(self.yuv10_matrix or ("bt2020" if self.hdr_mode else "bt709")).lower()

    @torch.inference_mode()
    def flush(self):
        """Emit frames still inside the pipeline. Returns a list of Frame objects."""
        frames = []
        while self._pending:
            result, ev = self._pending.pop(0)
            ev.synchronize()
            f = Frame(self.backend, self.videoWidth * self.scale, self.videoHeight * self.scale,
                      device=self.device, gpu_id=getattr(self, "gpu_id", 0), hdr_mode=self.hdr_mode,
                      dtype=self.dtype, high_bitdepth=self.high_bitdepth)
            f.yuv10_output = bool(self.high_bitdepth) and self._pack_yuv()
            f.yuv10_matrix = self._pipe_matrix()
            f.yuv8_output = self.yuv8_output
            f.set_frame_tensor(result)
            frames.append(f)
        return frames

    def _assert_model_geometry(self, output, *, expected=None, path: str = "") -> None:
        """Refuse an output tensor whose geometry is not the one the writer was told.

        `Frame(...)` is built at `videoWidth * scale` and everything downstream - the override resize,
        the pipe, the writer's `-s` - is told that size, so a model emitting another stride turns into
        a silent crop or a reshape failure in a thread with no window to report it. The load-time probe
        in `UpscaleModelWrapper` is what makes the scale true; this is the check that a tensor which
        contradicts it cannot reach a writer that trusts it (measured live 2026-11-23: a descriptor at
        4x, frames at 2x).
        """
        shape = getattr(output, "shape", None)
        if shape is None or len(shape) != 4:
            return  # the int8 tail hands back a packed uint8 payload, not an image tensor
        want_h, want_w = expected if expected is not None else (
            int(self.videoHeight) * int(self.scale), int(self.videoWidth) * int(self.scale))
        got_h, got_w = int(shape[-2]), int(shape[-1])
        if (got_h, got_w) != (int(want_h), int(want_w)):
            raise ValueError(
                f"model {os.path.basename(self.modelPath)} returned {got_w}x{got_h}, but this render "
                f"sizes its output for scale {self.scale} ({want_w}x{want_h}); the model's own stride "
                f"and the scale the writer will be told disagree ({path or 'render path'})")

    @torch.inference_mode()
    def _call_single(self, image: Frame) -> Frame:
        dtype = image.dtype
        gpu_id = image.gpu_id
        device = image.device
        # host timeline of the whole inference window, no synchronisation of our own
        import logging as _lgt
        _h = {"t0": time.perf_counter()}
        with self.torchUtils.run_stream(self.f2tstream):  # type: ignore
            image_tensor = image.get_frame_tensor()
        import os as _os_dbg
        if _os_dbg.environ.get("RVE_DEBUG_INPUT") == "1" and not getattr(self, "_dbg_input", False):
            self._dbg_input = True
            import logging as _lgd
            _t = image_tensor.float()
            _lgd.getLogger("RVE").warning(
                "fp16 INPUT shape=%s dtype=%s  mean=%.5f min=%.5f max=%.5f  ch-mean=%s",
                tuple(image_tensor.shape), image_tensor.dtype, float(_t.mean()), float(_t.min()),
                float(_t.max()), [round(float(v), 5) for v in _t[0].mean(dim=(1, 2)).tolist()])
        _h["upload"] = time.perf_counter()
        del image
        self.torchUtils.sync_stream(self.f2tstream)
        _h["upload_sync"] = time.perf_counter()

        _t_inf = time.perf_counter()
        with self.torchUtils.run_stream(self.stream):
            while self.upscale_model_wrapper is None:
                sleep(1)
            if self.tile_upscaler is not None:
                rects = self.tile_upscaler.plan(image_tensor)
                if rects is None and self._prev_hr is not None:
                    # nothing changed: re-emit the previous HR frame as-is
                    output = self._prev_hr
                else:
                    output = self.tile_upscaler.compose(
                        self.scale, self.upscale_model_wrapper, self._prev_hr, image_tensor
                    )
                    self._prev_hr = output
            elif self.tilesize == 0:
                body = self._int8_body(image_tensor)
                # GPU time of the fp16 engine, measured the same way as the body's: events, previous
                # frame's pair read on the next, nothing synchronised inside a frame.
                import logging as _lgf, os as _osq
                _quietf = _osq.environ.get("RVE_INT8_NO_EVENTS") == "1"
                _prevf = None if _quietf else getattr(self, "_fp16_ev", None)
                if _prevf is not None:
                    self._fp16_ms = getattr(self, "_fp16_ms", []) + [_prevf[0].elapsed_time(_prevf[1])]
                    if len(self._fp16_ms) % 8 == 0:
                        _v = self._fp16_ms[-8:]
                        _lgf.getLogger("RVE").warning(
                            "fp16 engine GPU: %.2f ms/frame (last 8); alloc_retries %d",
                            sum(_v) / len(_v), torch.cuda.memory_stats().get("num_alloc_retries", -1))
                _fs, _fe = torch.cuda.Event(True), torch.cuda.Event(True)
                _fs.record()
                if body is not None:
                    output = body(image_tensor)
                else:
                    output = self.upscale_model_wrapper(image_tensor)
                _fe.record()
                self._fp16_ev = (_fs, _fe)
            else:
                output = self.renderTiledImage(image_tensor)

            self._assert_model_geometry(output, path="UpscaleTorch._call_single")

            # Split of the timed region: the engine call, the output Frame's own
            # construction, and the handover. `last_inference_ms` measures the whole block,
            # and the block overlaps Frame construction with the model (the ctor runs while
            # the kernels are still going), so it is *not* the engine time - which is exactly
            # how "the app is 23% slower than the isolated benchmark" got misread (ledger P11).
            # RVE_TRT_SPLIT=1 forces a sync here so the parts are attributable; it costs the
            # overlap, so it is a diagnostic, never a production path.
            _h["model"] = time.perf_counter()
            if _SPLIT_TIMING:
                self.torchUtils.sync_stream(self.stream)
            self.last_model_ms = (time.perf_counter() - _t_inf) * 1000.0
            _t_frame_ctor = time.perf_counter()
            retFrame = Frame(self.backend, self.videoWidth * self.scale, self.videoHeight * self.scale, device=device, gpu_id=gpu_id, hdr_mode=self.hdr_mode, dtype=dtype,
                             high_bitdepth=self.high_bitdepth)
            # 10-bit output hands ffmpeg ready-made yuv420p10le instead of RGB (HDR included, with the HDR matrix)
            retFrame.yuv10_output = bool(self.high_bitdepth) and self._pack_yuv()
            retFrame.yuv10_matrix = self._pipe_matrix()
            retFrame.yuv8_output = self.yuv8_output
            retFrame.yuv10_format = self.pipe_pixel_format
            
            retFrame.set_frame_tensor(output, sync=False)
            self.last_frame_ms = (time.perf_counter() - _t_frame_ctor) * 1000.0
            _h["frame"] = time.perf_counter()
        
        self.torchUtils.sync_stream(self.stream)
        # measured after the sync: the model runs async on its stream, so a timer taken at
        # enqueue time only measures the launch (~1.8 ms, not the 35 ms of work)
        self.last_inference_ms = (time.perf_counter() - _t_inf) * 1000.0
        _h["end"] = time.perf_counter()
        self._htl = getattr(self, "_htl", 0) + 1
        if self._htl <= 3 or self._htl % 8 == 0:
            _ms = lambda a, b: (_h[b] - _h[a]) * 1000.0
            _lgt.getLogger("RVE").warning(
                "host timeline: upload %.2f  upload_sync %.2f  model-call %.2f  frame-ctor %.2f  "
                "final-sync %.2f  | total %.2f ms | last_inference_ms %.2f | out dtype %s "
                "shape %s contig %s stride %s",
                _ms("t0", "upload"), _ms("upload", "upload_sync"), _ms("upload_sync", "model"),
                _ms("model", "frame"), _ms("frame", "end"), (_h["end"] - _h["t0"]) * 1000.0,
                getattr(self, "last_inference_ms", -1.0),
                getattr(retFrame, "_tensor", None).dtype if getattr(retFrame, "_tensor", None) is not None else None,
                tuple(getattr(retFrame, "_tensor", None).shape) if getattr(retFrame, "_tensor", None) is not None else None,
                getattr(retFrame, "_tensor", None).is_contiguous() if getattr(retFrame, "_tensor", None) is not None else None,
                getattr(retFrame, "_tensor", None).stride() if getattr(retFrame, "_tensor", None) is not None else None)
        return retFrame
 
    def _int8_body(self, frame=None):
        """The int8 body for this render, or None. Built, calibrated and certified once, then reused."""
        import logging
        log = logging.getLogger("RVE").warning
        if getattr(self, "_int8_done", False):
            return getattr(self, "_int8_live", None)
        self._int8_done = True
        import os
        if str(os.environ.get("RVE_INT8_BODY", "auto")).lower() == "off":
            log("int8: disabled by RVE_INT8_BODY=off - the fp16 engine runs")
            return None
        log("int8: start (tilesize=%s, tile_upscaler=%s, int8Fit=%s)",
            getattr(self, "tilesize", "?"), getattr(self, "tile_upscaler", None) is not None,
            getattr(self, "int8Fit", "<unset>"))
        body = None
        if str(getattr(self, "int8Fit", "")) == "int8":
            wrapper = getattr(self, "upscale_model_wrapper", None)
            model = wrapper.get_model() if wrapper is not None and hasattr(wrapper, "get_model") else None
            log("int8: wrapper=%s model=%s", type(wrapper).__name__ if wrapper else None,
                type(model).__name__ if model is not None else None)
            if model is not None:
                try:
                    from .Int8Body import Int8Body
                    body = Int8Body(model)
                    # The tail may write the pipe payload itself (RVE_TAIL_YUV), but only when the pipe
                    # really carries 8-bit yuv420p. Two conditions, both from the pipe decision and not
                    # from the bit depth alone: `yuv8_output` (the packer's own flag) and the format the
                    # writer was told. On an rgb24 pipe - HDR, 10-bit, or an 8-bit source that declares a
                    # non-bt709 matrix, where the app deliberately keeps rgb24 so ffmpeg uses the source's
                    # matrix - the body must keep returning RGB.
                    body.out_yuv = bool(getattr(body, "tail_yuv", False) and self.yuv8_output
                                        and self.pipe_pixel_format == "yuv420p")
                    # Log the DECISION, not the flag. With RVE_TAIL_YUV=1 by default, a gate that refuses
                    # silently makes an A/B on 10-bit (or non-bt709) material read two identical arms - and
                    # the honest conclusion from that is "the optimisation does nothing", which is wrong.
                    log("int8 tail output: %s (RVE_TAIL_YUV=%s -> tail_yuv=%s, yuv8_output=%s, pipe=%s)",
                        "yuv420p payload" if body.out_yuv else "fp16 RGB tensor",
                        os.environ.get("RVE_TAIL_YUV", "1"), getattr(body, "tail_yuv", None),
                        self.yuv8_output, self.pipe_pixel_format)
                    log("int8: built (supports=%s)", Int8Body.supports(model))
                except Exception as error:  # noqa: BLE001
                    log("int8: build failed (%s)", error)
        else:
            log("int8: verdict is not int8, fp16 engine runs")
        if body is None:
            return None
        self._int8_frame = frame
        try:
            if frame is not None:
                pixels = int(frame.shape[-2]) * int(frame.shape[-1])
                if pixels > 2_500_000:
                    log("int8: refused at %dx%d (measured 0.77x at 4K against 0.96x at 1080p)",
                        frame.shape[-1], frame.shape[-2])
                    return None
        except Exception:  # noqa: BLE001
            pass
        try:
            # CALIBRATION FRAMES OR NOTHING. A body built before the probe hands its spread sample over
            # calibrates on a single frame - measured 20x worse on composition (0.116% against 0.006%) and
            # 0.566% against 0.050% on Violet. That is not hypothetical: the certificate's engine reference
            # calls the upscaler, which builds the body DURING the probe, i.e. before `int8Frames` exists
            # (found 2026-11-19, harness/docs/wsp/verified/int8-calibration-ordering.txt). Refusing to build here is
            # what makes the spread sample mandatory instead of best-effort.
            if getattr(self, "int8Frames", None) is None:
                log("int8: no calibration frames yet - the body is built on a later frame, with the "
                    "probe's spread sample (a single-frame calibration is 20x worse on composition)")
                return None
            _frames = self.int8Frames
            if not getattr(self, "_calib_shape_logged", False):
                self._calib_shape_logged = True
                _f0 = _frames[0] if _frames else None
                log("int8 calibration frames: %d x shape %s dtype %s device %s | render frame shape %s dtype %s "
                    "device %s", len(_frames), tuple(_f0.shape) if _f0 is not None else None,
                    getattr(_f0, "dtype", None), getattr(_f0, "device", None),
                    tuple(frame.shape) if frame is not None else None,
                    getattr(frame, "dtype", None), getattr(frame, "device", None))
            # THE GRID PICK (RVE_INT8_GRID_K=auto): choose K on a slice of the calibration frames and keep the
            # certificate on another slice, because picking on the frames you certify on makes the gate pass by
            # construction. The guard (x2 over K=2) is what stops it chasing noise on a handful of frames.
            # The calibration is split, not repeated: calibrating on all 24 and then again on 18 cost 4.8 s of
            # eager forwards (measured: start-up 39.84 s against 33.08 without the pick).
            if getattr(body, "grid_auto", False) and len(_frames) >= 8:
                hold = list(_frames[3::4])
                body.calibrate([f for i, f in enumerate(_frames) if i % 4 != 3])
            else:
                body.calibrate(_frames)
            if getattr(body, "grid_auto", False) and len(_frames) >= 8:
                try:
                    _k, _scores = body.pick_grid_k(hold)
                    log("int8 grid PICKED K=%g%s (scores %s; guard x2 over K=2, chosen on %d frames, "
                        "certificate on another slice; build %.2f s, score %.2f s)", _k,
                        " (K=2 already clean, sweep skipped)" if getattr(body, "_pick_skipped", False) else "",
                        {f"K={k:g}": round(v, 4) for k, v in sorted(_scores.items())}, len(hold),
                        *getattr(body, "_pick_timing", (0.0, 0.0)))
                except Exception as error:  # noqa: BLE001 - never let the pick cost a render
                    log("int8 grid pick failed (%s); keeping K=%g", error, getattr(body, "grid_k", 2.0))
            share, ok = body.certify(frame)
            # n=1 and a gross gate: this is a coverage sanity check, not a picture verdict. The wording
            # says so, because the old line printed a percentage under the acceptance bar and read as a
            # certificate (rve-body-first-frame-certificate-2026-11-19).
            from .Int8Body import FIRST_FRAME_SANITY
            if ok:
                log("int8 body first-frame sanity: %.5f%% of pixels >8 levels against the fp16 model "
                    "(n=1, gross gate %.1f%%; NOT an acceptance number - the picture is judged on renders)",
                    share, FIRST_FRAME_SANITY)
            else:
                log("int8 body first-frame sanity FAILED: %.5f%% of pixels >8 levels against the fp16 "
                    "model (gross gate %.1f%%) - the quantiser's grid does not cover this frame at this "
                    "resolution; fp16 engine runs", share, FIRST_FRAME_SANITY)
            if not ok:
                return None
            self._int8_live = body
            return body
        except Exception as error:  # noqa: BLE001 - never let this cost a render
            log("int8 body refused to start (%s); fp16 engine runs", error)
            return None

    def getScale(self):
        return self.upscale_model_wrapper.get_scale()

    @torch.inference_mode()
    def renderTiledImage(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.scale
        tile = self.tile
        tile_pad = self.tile_pad

        batch, channel, height, width = img.shape
        output_shape = (batch, channel, height * scale, width * scale)

        # start with black image
        output = img.new_zeros(output_shape).to(device=self.device, dtype=self.dtype)

        tiles_x = math.ceil(width / tile[0])
        tiles_y = math.ceil(height / tile[1])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile[0]
                ofs_y = y * tile[1]

                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile[0], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile[1], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ].to(device=self.device, dtype=self.dtype)

                h, w = input_tile.shape[2:]
                input_tile = F.pad(
                    input_tile, (0, self.pad_w - w, 0, self.pad_h - h), "replicate"
                )

                # process tile
                output_tile = self.upscale_model_wrapper(
                    input_tile
                )
                # the crop below silently shrinks a tile whose stride is not `scale`, and the pasted
                # picture is then a mosaic of one scale on a canvas of another
                self._assert_model_geometry(
                    output_tile,
                    expected=(int(input_tile.shape[-2]) * int(scale),
                              int(input_tile.shape[-1]) * int(scale)),
                    path="UpscaleTorch.renderTiledImage")

                output_tile = output_tile[:, :, : h * scale, : w * scale]

                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                # put tile into output image
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

        return output