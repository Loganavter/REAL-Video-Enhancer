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
        trt_tiling_level: str = "moderate",
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
        self.tile_pad = tile_pad
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
            self.tile_pad = tile_halo
        self.trt_aux_streams = trt_max_aux_streams
        self.trt_debug = trt_debug
        
        self.hdr_mode = hdr_mode         
        self.high_bitdepth = hdr_mode if high_bitdepth is None else high_bitdepth
        self.gpu_id = gpu_id
        self.pipe_pixel_format = pipe_pixel_format
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
                from .TiledUpscale import TiledUpscaler
                self.tile_upscaler = TiledUpscaler(
                    tile_lr=self.tilesize,
                    halo_lr=self.tile_pad,
                    threshold=self.tile_threshold,
                    max_chain=self.tile_max_chain,
                )
                side = self.tile_upscaler.tile
                self.pad_w = math.ceil(side / modulo) * modulo
                self.pad_h = self.pad_w
                log(f"Tile reuse enabled: {self.tilesize}px LR tiles + {self.tile_pad}px halo -> engine {self.pad_w}x{self.pad_h}")
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
        retFrame.yuv10_output = bool(self.high_bitdepth and not self.hdr_mode)
        retFrame.set_frame_tensor(result)
        return retFrame

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
            f.yuv10_output = bool(self.high_bitdepth and not self.hdr_mode)
            f.set_frame_tensor(result)
            frames.append(f)
        return frames

    @torch.inference_mode()
    def _call_single(self, image: Frame) -> Frame:
        dtype = image.dtype
        gpu_id = image.gpu_id
        device = image.device
        with self.torchUtils.run_stream(self.f2tstream):  # type: ignore
            image_tensor = image.get_frame_tensor()
        del image
        self.torchUtils.sync_stream(self.f2tstream)

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
                output = self.upscale_model_wrapper(image_tensor)
            else:
                output = self.renderTiledImage(image_tensor)
            
            retFrame = Frame(self.backend, self.videoWidth * self.scale, self.videoHeight * self.scale, device=device, gpu_id=gpu_id, hdr_mode=self.hdr_mode, dtype=dtype,
                             high_bitdepth=self.high_bitdepth)
            # 10-bit SDR: hand ffmpeg ready-made yuv420p10le instead of RGB
            retFrame.yuv10_output = bool(self.high_bitdepth and not self.hdr_mode)
            retFrame.yuv10_format = self.pipe_pixel_format
            
            retFrame.set_frame_tensor(output, sync=False)
        
        self.torchUtils.sync_stream(self.stream)
        # measured after the sync: the model runs async on its stream, so a timer taken at
        # enqueue time only measures the launch (~1.8 ms, not the 35 ms of work)
        self.last_inference_ms = (time.perf_counter() - _t_inf) * 1000.0
        return retFrame
 
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