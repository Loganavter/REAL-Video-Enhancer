import time as _time
from threading import Thread

# Process-relative clock: the wall-clock "Time to complete render" starts in the writer
# thread, so it hides everything before it (interpreter/import, engine build or deserialize,
# ffmpeg + NVENC init) and the flush at the end. On a 94-frame clip that fixed part is
# ~1.5 s, i.e. 16 ms/frame - large enough to be misread as per-frame overhead, which is
# exactly what happened to the budget table. These marks split it out.
_T0 = _time.time()
import torch

_NO_PREVIEW = object()  # non-None marker: keeps the FPS/ETA printer alive, costs nothing
import os
import math
from time import sleep
from time import perf_counter as _perf_counter, time
from typing import Optional
import sys
from multiprocessing import shared_memory
import cv2

from .FFmpegBuffers import FFmpegRead, FFmpegWrite, MPVOutput
from .FFmpeg import InformationWriteOut
from .utils.Encoders import EncoderSettings
from .utils.SceneDetect import SceneDetect
from .utils.Util import log, resize_image_bytes
from .utils.BorderDetect import BorderDetect
from .utils.VideoInfo import OpenCVInfo
import numpy as np


def remove_shared_memory_block(name):
    try:
        existing_shm = shared_memory.SharedMemory(name=name)
        existing_shm.close()
        existing_shm.unlink()
        print(f"Shared memory block '{name}' removed.")
    except FileNotFoundError:
        print(f"Shared memory block '{name}' does not exist.")
    except Exception as e:
        print(f"Error removing shared memory block '{name}': {e}")


class Render:
    """
    Subclass of FFmpegRender
    FFMpegRender options:
    inputFile: str, The path to the input file.
    outputFile: str, The path to the output file.
    interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
    encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
    pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)

    interpolateOptions:
    interpolationMethod
    upscaleModel
    backend (pytorch,ncnn,tensorrt)
    device (cpu,cuda)
    precision (float16,float32)

    NOTE:
    Everything in here has to happen in a specific order:
    Get the video properties (res,fps,etc)
    set up upscaling/interpolation, this gets the scale for upscaling if upscaling is the current task
    assign framechunksize to a value, as this is needed to catch bytes and set up shared memory
    set up shared memory
    """

    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        # backend settings
        backend="pytorch",
        device="default",
        precision="float16",
        pytorch_gpu_id: int = 0,
        ncnn_gpu_id: int = 0,
        cwd: str = os.getcwd(),
        # model settings
        upscaleModel=None,
        interpolateModel=None,
        interpolateFactor: int = 1,
        extraRestorationModels=None,
        sceneDetectModel: str = None,
        tile_size=None,
        drba=False,
        # ffmpeg settings
        ffmpeg_path: str = "./bin/ffmpeg",
        start_time=None,
        end_time=None,
        custom_encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        crf: str = "18",
        video_encoder_preset: str = "libx264",
        audio_encoder_preset: str = "aac",
        subtitle_encoder_preset: str = "srt",
        audio_bitrate: str = "192k",
        border_detect: bool = False,
        hdr_mode: bool = False,
        merge_subtitles: bool = True,
        # misc
        pause_shared_memory_id=None,
        sceneDetectMethod: str = "pyscenedetect",
        sceneDetectSensitivity: float = 3.0,
        sharedMemoryID: Optional[str] = None,
        trt_optimization_level: int = 5,
        trt_dynamic_shapes: bool = False,
        override_upscale_scale: int | None = None,
        UHD_mode: bool = False,
        slomo_mode: bool = False,
        dynamic_scaled_optical_flow: bool = False,
        ensemble: bool = False,
        output_to_mpv: bool = False,
        trt_timing_iters: int = 8,
        trt_streams: int = 1,
        trt_tiling_level: str = "full",
        tile_reuse: bool = False,
        tile_reuse_size: int = 512,
        tile_reuse_threshold: float = 0.35,
        temporal_reuse: bool = True,
        # TheAnimeScripter calibration: --dedup_sens 35 / 7 = 5.0 luma levels, max span 4
        temporal_reuse_threshold: float = 5.0,
        temporal_reuse_statistic: str = "mean_abs",
        temporal_reuse_max_chain: int = 4,
        # 0/None = off. 40 is the TheAnimeScripter value and is right for clean anime, but on
        # grainy film a single ringing pixel exceeds it: measured 43 vetoes in 94 frames and
        # reuse collapsed to 1.1% (vs 46.8% with the veto off).
        temporal_reuse_pixel_veto: float | None = 0.0,
        temporal_reuse_auto: bool = True,
        quality_preset: str = "best",
    ):
        self.inputFile = inputFile
        self.backend = backend
        self.upscaleModel = upscaleModel
        self.interpolateModel = interpolateModel
        # decided here, before setupUpscale() builds the engine (tile reuse changes its shape)
        self.tileReuseSize = tile_reuse_size
        self.tileReuseThreshold = tile_reuse_threshold
        self.tileReusePad = 18            # LR halo the tile path uses (receptive radius)
        self.tileReuseMaxChain = 2
        self.temporalReuseThreshold = temporal_reuse_threshold
        self.temporalReuseMaxChain = temporal_reuse_max_chain
        self.ffmpegPath = ffmpeg_path
        if quality_preset not in ("best", "fast"):
            raise ValueError(f"quality_preset must be 'best' or 'fast', got {quality_preset!r}")
        self.qualityPreset = quality_preset
        self.tileReuseMode = tile_reuse if isinstance(tile_reuse, str) else (
            "on" if tile_reuse else "off")
        self.tileReuse = bool(
            self.tileReuseMode != "off" and self.upscaleModel and not self.interpolateModel
            and self.backend in ("pytorch", "tensorrt")
        )
        self.tilesize = tile_size
        self.device = device
        self.precision = precision
        self.interpolateFactor = interpolateFactor
        # max timestep is a hack to make sure ncnn cache frames too early, and ncnn breaks if i modify the code at all so ig this is what we are doing
        # also used to help with performace and caching
        self.maxTimestep = (interpolateFactor - 1) / interpolateFactor
        self.ceilInterpolateFactor = math.ceil(self.interpolateFactor)
        
        # self.setupRender = self.returnFrame  # set it to not convert the bytes to array by default, and just pass chunk through
        self.setupFrame0 = None
        self.interpolateOption = None
        self.upscaleOption = None
        self.isPaused = False
        self.drba = drba
        self.sceneDetectModelPath = sceneDetectModel
        self.sceneDetectMethod = sceneDetectMethod
        self.sceneDetectSensitivty = sceneDetectSensitivity
        self.sharedMemoryID = sharedMemoryID
        self.trt_optimization_level = trt_optimization_level
        self.trt_tiling_level = trt_tiling_level
        self.trt_timing_iters = trt_timing_iters
        self.trt_streams = trt_streams

        self.uncacheNextFrame = False
        self.UHD_mode = UHD_mode
        self.dynamic_scaled_optical_flow = dynamic_scaled_optical_flow
        self.ensemble = ensemble
        self.pytorch_gpu_id = pytorch_gpu_id
        self.ncnn_gpu_id = ncnn_gpu_id
        self.outputFrameChunkSize = None
        self.hdr_mode = hdr_mode
        self.override_upscale_scale = override_upscale_scale
        self.trt_dynamic_shapes = trt_dynamic_shapes
        self.extraRestorationModels = []
        
        if cwd:
            log("Working Directory: " + cwd)
        else:
            cwd = os.getcwd()
            log("No Working Directory specified, using current directory: " + cwd)
        videoInfo = OpenCVInfo(input_file=inputFile, start_time=start_time, end_time=end_time, ffmpeg_path=ffmpeg_path)
        
        if not videoInfo.is_valid_video:
            log("Input video is not valid!")
        
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = videoInfo.get_duration_seconds()

        self.width, self.height = videoInfo.get_width_x_height()
        self.originalWidth = self.width
        self.originalHeight = self.height
        input_pix_fmt = videoInfo.get_pixel_format()
        self.borderX = 0
        self.borderY = 0  # set borders for cropping automatically to 0, will be overwritten if borders are detected
        self.totalInputFrames = videoInfo.get_total_frames()
        self.totalOutputFrames = int(
            self.totalInputFrames * self.ceilInterpolateFactor
        )
        self.fps = videoInfo.get_fps()
        color_space = videoInfo.get_color_space()
        color_primaries = videoInfo.get_color_primaries()
        color_transfer = videoInfo.get_color_transfer()
        try:
            color_range = videoInfo.get_color_range()
        except Exception:
            color_range = None
        # 10-bit SDR sources: keep them 16-bit through the pipe instead of quantising to
        # rgb24 (the 8-bit round trip caps the conversion at ~44 dB vs ~79.5 dB).
        # Interpolation feeds synthetic frames through the same buffers, so keep the old
        # 8-bit behaviour there rather than mixing depths.
        try:
            source_bitdepth = videoInfo.get_bit_depth() or 8
        except Exception:
            source_bitdepth = 8
        high_bitdepth = (hdr_mode or source_bitdepth >= 10) and not (self.interpolateModel)
        if high_bitdepth and not hdr_mode:
            log(f"10-bit source detected: keeping the pipeline at 16 bit (was rgb24)")
        self.high_bitdepth = high_bitdepth

        video_encoder = EncoderSettings(video_encoder_preset)
        # NVENC wants P010 and repacks planar internally (~1.5 ms/frame at 2160p), but the
        # P010 pipe is NOT validated end to end here: the packing matches the planar decode
        # bit-exactly in isolation, yet a full render through hevc_nvenc diverges from the
        # planar reference (31 dB). An opt-in, not a default, until that is explained.
        self.pipePixelFormat = (
            "p010le"
            if (os.environ.get("RVE_P010") and video_encoder.getPresetTag() in ("x265_nvenc", "av1_nvenc"))
            else "yuv420p10le"
        )
        audio_encoder = EncoderSettings(audio_encoder_preset, type="audio")
        subtitle_encoder = EncoderSettings(subtitle_encoder_preset, type="subtitle")

        if border_detect:  # border detect has to be put before everything, to overwrite the width and height
            print("Detecting borders", file=sys.stderr)
            borderDetect = BorderDetect(inputFile=self.inputFile, ffmpeg_path=ffmpeg_path)
            self.width, self.height, self.borderX, self.borderY = (
                borderDetect.getBorders()
            )
            log(
                f"Detected borders: Width,Height:{self.width}x{self.height}, X,Y: {self.borderX}x{self.borderY}"
            )

        log("Using backend: " + self.backend)
        # upscale has to be called first to get the scale of the upscale model
        if upscaleModel:
            self.setupUpscale()
            self.upscaleOption.hotUnload()  # unload model to free up memory for trt enging building
            log("Using Upscaling Model: " + self.upscaleModel)
        else:
            self.upscaleTimes = 1  # if no upscaling, it will default to 1
            self.modelScale = 1
            

        
        if extraRestorationModels:
            for model in extraRestorationModels:
                extraRestoration = self.setupExtraRestoration(model)
                if extraRestoration:
                    log("Using Extra Restoration Model: " + model)
                    self.extraRestorationModels.append(extraRestoration)
                    extraRestoration.hotUnload()  # unload model to free up memory for trt enging building
        

        if interpolateModel:
            self.setupInterpolate()
            log("Using Interpolation Model: " + self.interpolateModel)

        if upscaleModel: # load model after interpolation model is loaded, this saves on vram if the user builds 2 separate engines
            self.upscaleOption.hotReload()

        for extraRestoration in self.extraRestorationModels:
            extraRestoration.hotReload()
        
        if self.modelScale and self.override_upscale_scale:
            if int(self.modelScale) == int(self.override_upscale_scale):
                log("Override upscale scale is set to the same value as the model scale, this will not change the output resolution.")
                self.override_upscale_scale = False
        
        log(f"Upscale Times: {self.override_upscale_scale if self.override_upscale_scale else self.upscaleTimes}")
        log(f"Interpolate Factor: {self.interpolateFactor}")
        log(f"Total Output Frames: {self.totalOutputFrames}")
        log("Model Scale: " + str(self.modelScale))
        log("HDR Mode: " + str(hdr_mode))

        self.readBuffer = FFmpegRead(  # input width
            inputFile=inputFile,
            width=self.width,
            height=self.height,
            start_time=start_time,
            end_time=end_time,
            borderX=self.borderX,
            borderY=self.borderY,
            hdr_mode=hdr_mode,
            backend=self.backend,
            device=self.device,
            gpu_id=self.pytorch_gpu_id if self.backend in ["pytorch","tensorrt"] else self.ncnn_gpu_id,
            dtype=self.precision,
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            color_range=color_range,
            high_bitdepth=high_bitdepth,
            input_pixel_format=input_pix_fmt,
            ffmpeg_path=ffmpeg_path,
        )

        from .FFmpegBuffers import resolve_output_pixel_format
        pixelFormat = resolve_output_pixel_format(
            pixelFormat, self.high_bitdepth, bool(self.interpolateModel))
        self.writeBuffer = FFmpegWrite(
            inputFile=inputFile,
            outputFile=outputFile,
            width=self.width,
            height=self.height,
            start_time=start_time,
            end_time=end_time,
            fps=self.fps,
            crf=crf,
            audio_bitrate=audio_bitrate,
            pixelFormat=pixelFormat,
            overwrite=overwrite,
            custom_encoder=custom_encoder,
            benchmark=benchmark,
            slowmo_mode=slomo_mode,
            upscaleTimes=self.upscaleTimes if not self.override_upscale_scale else self.override_upscale_scale,
            interpolateFactor=self.interpolateFactor,
            ceilInterpolateFactor=self.ceilInterpolateFactor,
            video_encoder=video_encoder,
            audio_encoder=audio_encoder,
            subtitle_encoder=subtitle_encoder,
            mpv_output=output_to_mpv,
            hdr_mode=hdr_mode,
            merge_subtitles=merge_subtitles,
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            color_range=color_range,
            high_bitdepth=high_bitdepth,
            pipe_input_pixel_format=getattr(self, "pipePixelFormat", "yuv420p10le"),
            ffmpeg_path=ffmpeg_path,
            ffmpeg_log_file=os.path.join(cwd, "ffmpeg_log.txt"),
        )

        shm_mul = self.override_upscale_scale if self.override_upscale_scale else self.upscaleTimes
        hdr_mul = 6 if hdr_mode else 3

        self.informationHandler = InformationWriteOut(
            sharedMemoryID=sharedMemoryID,
            sharedMemoryChunkSize=self.originalHeight*self.originalWidth*shm_mul*shm_mul*hdr_mul,
            paused_shared_memory_id=pause_shared_memory_id,
            outputWidth=self.originalWidth*shm_mul,
            outputHeight=self.originalHeight*shm_mul,
            croppedOutputWidth=self.width*shm_mul,
            croppedOutputHeight=self.height*shm_mul,
            totalOutputFrames=self.totalOutputFrames,
            border_detect=border_detect,
            hdr_mode=hdr_mode,
            
        )
        

        self.temporalReuse = None
        # always present: the render loop asks about it on every frame, including runs with
        # --no_temporal_reuse where no gate is created (missing it crashed the render thread).
        # This has to come BEFORE the gate block, which assigns the real tuner - placing it
        # after silently disabled auto tuning for every configuration.
        self.autoTuner = None
        # The two detectors are complementary and stack: the gate decides whether this frame
        # needs the model at all, the tile path then decides which tiles of that frame need it.
        # Until now `not self.tileReuse` here made them alternatives, which is why every tile
        # measurement was taken against an ungated render.
        if temporal_reuse and self.upscaleModel and not self.interpolateModel and self.backend in ("pytorch", "tensorrt"):
            from .utils.TemporalReuse import ReferenceFrameGate
            self.temporalReuse = ReferenceFrameGate(
                threshold=temporal_reuse_threshold,
                max_chain=temporal_reuse_max_chain,
                pixel_veto=temporal_reuse_pixel_veto or None,
                statistic=temporal_reuse_statistic,
            )
            log(f"Reference-frame reuse enabled (threshold {temporal_reuse_threshold}/255, "
                f"statistic {temporal_reuse_statistic}, "
                f"max chain {temporal_reuse_max_chain}, pixel veto {temporal_reuse_pixel_veto})")
            self.autoTuner = None
            if temporal_reuse_auto:
                from .utils.TemporalReuse import AutoGateTuner
                # "best" asks the tuner to prove the error budget or refuse to reuse at all,
                # and holds it to a tighter budget; "fast" accepts the least aggressive
                # candidate when nothing proves out. Measured on the film clip the difference
                # is 18.1% reuse with a worst frame at 39.8 dB against no reuse at all.
                # Both presets hold the tuner to the same 2-level budget: 1.0 was too tight
                # and made "best" disable the gate even on clean anime (measured: 18.1 s
                # against 7.65 s, with the gate doing nothing). What separates them is the
                # fallback, i.e. exactly the content that cannot be certified.
                best = getattr(self, "qualityPreset", "fast") == "best"
                self.autoTuner = AutoGateTuner(
                    seed=float(temporal_reuse_threshold),
                    budget=2.0,
                    fallback="off" if best else "strictest",
                    chain_policy="certified" if best else "max",
                )
                log("auto gate tuning: the threshold will be chosen from the first rendered "
                    "frames (LR delta vs the real HR delta they cause)")

        # Pooled pinned output payload: the writer gets a view on a pinned host buffer instead
        # of a fresh per-frame .tobytes() copy. A pooled payload is single-use, so it cannot
        # coexist with temporal reuse (which re-emits the remembered Frame).
        # Pooled pinned output is NOT enabled: wired as the worker specified, a full render
        # comes out 12.6 dB against the reference (slots are rewritten before ffmpeg has the
        # bytes) and it was slower (4.83 s vs 4.21 s). Kept for reference; see the DONE.md of
        # the copy-elim cell for the intended protocol.
        self.pooled_output = False
        self._hr_grid_prev = None
        # the gate stores only its LR reference; the HR frame we may re-emit lives here
        self._lastRenderedFrame = None

        self.renderThread = Thread(target=self.render)
        self.ffmpegReadThread = Thread(target=self.readBuffer.read_frames_into_queue)
        self.ffmpegWriteThread = Thread(target=self.writeBuffer.write_out_frames)
        self.sharedMemoryThread = Thread(
            target=self.informationHandler.writeOutInformation
        )

        self.sharedMemoryThread.start()
        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()

        if output_to_mpv:
            MPVOut = MPVOutput(self.writeBuffer, width=self.width*self.upscaleTimes, height=self.height*self.upscaleTimes,fps=self.fps*self.interpolateFactor, outputFrameChunkSize=self.outputFrameChunkSize)
            MPVoutThread = Thread(target=MPVOut.write_out_frames)
            MPVoutThread.start()

    def write_bytes_to_cv2_frame_debug(self, frame):
        # Convert the byte array to a numpy array
        frame_array = np.frombuffer(frame, dtype=np.uint8)
        # Reshape the array to the correct dimensions
        frame_array = frame_array.reshape((self.height, self.width, 3))
        # Convert the BGR image to RGB
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite("frame.jpg", frame_array)

    def _preview_bytes(self, frame):
        """Payload for the information/preview channel.

        Nothing consumes it in a headless render (no shared memory target), yet building
        it for a 16-bit 4K frame costs a full host-side conversion per frame - that alone
        made the 10-bit path ~2x slower. Without a preview target a sentinel is enough:
        the writer only uses the payload when self.sharedMemoryID is set.
        """
        if self.informationHandler.sharedMemoryID is None:
            return _NO_PREVIEW
        if type(frame) is bytes:
            return frame
        if getattr(self, "high_bitdepth", False) and not getattr(frame, "hdr_mode", False):
            tensor = frame.get_frame_tensor(clear_cache=False)
            with torch.inference_mode():
                small = (tensor[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8)
                return small.cpu().numpy().tobytes()
        return frame.get_frame_bytes()

    def _probe_ffmpeg_path(self) -> str:
        """ffmpeg for the probe: the app's bundled one when it is there, else PATH."""
        import os as _os
        candidates = [getattr(self, "ffmpegPath", None), "./bin/ffmpeg", "bin/ffmpeg"]
        for candidate in candidates:
            if candidate and _os.path.exists(candidate):
                return candidate
        return "ffmpeg"

    def _probe_tile_reuse(self, ffmpeg_path):
        """Ask the source whether tiles pay, before an engine is built for them.

        The engine shape follows the verdict (a tile engine is built for the tile window, not
        for the frame), so this has to happen before the upscaler exists. The probe replays
        the gate's and the tile path's own statistics on a handful of decoded frames and
        returns "would tiles be cheaper than one full-frame call" - see
        `utils/TileProbe.py` for why that question is answerable from a sample.
        """
        try:
            from .utils.TileProbe import probe
            result = probe(
                self.inputFile,
                width=int(self.width),
                height=int(self.height),
                tile_size=int(getattr(self, "tileReuseSize", 512)),
                halo=int(self.tileReusePad),
                ffmpeg_path=ffmpeg_path,
                gate_threshold=float(getattr(self, "temporalReuseThreshold", 5.0)),
                tile_threshold=float(getattr(self, "tileReuseThreshold", 0.35)),
                gate_max_chain=int(getattr(self, "temporalReuseMaxChain", 4)),
                tile_max_chain=int(getattr(self, "tileReuseMaxChain", 2)),
            )
            log(result.stats())
            if not result.use_tiles:
                log(f"  {result.reason}")
            return result
        except Exception as error:  # a probe failure must never cost the render
            log(f"tile probe failed ({error}); leaving tile reuse off")
            return None

    def _probe_int8_fit(self):
        """Ask the source whether the int8 body may serve it, before the first frame.

        The question is composition - what the temporal gate shows when a frame is served from an int8
        reference one frame old - not the engine's own defect: on the Violet pair the own defect passes
        the bar (0.057%) and composition does not (0.170%), so certifying on own-defect would ship them.
        A probe failure, or anything unexpected, leaves int8 off; the fp16 engine is the default path.
        """
        try:
            from .utils.Int8Probe import probe
            result = probe(
                self.inputFile,
                model_path=self.upscaleModel,
                width=int(self.width),
                height=int(self.height),
                bar=float(getattr(self, "int8FitBar", 0.1)),
            )
            log(result.stats())
            if not result.use_int8:
                log(f"  {result.reason}")
            # hand the decoded calibration frames to the body: they are what makes its grids cover the video
            self.int8Frames = getattr(result, "frames", None)
            verdict = "int8" if result.use_int8 else "fp16"
            if verdict == "int8":
                # The probe's answer is not the render's: the body is wired but only into the whole-frame
                # path, while the app renders tiled. And measured against the TILED engine the app actually
                # builds, the body loses - 56.8 ms against 35.85 ms. Say the measured reason, not a promise.
                log("int8 body: the probe admits this video; the body only serves the whole-frame path and "
                    "loses to the app's tiled engine anyway (56.8 vs 35.85 ms at 1080p) - fp16 engine runs")
            return verdict
        except Exception as error:  # a probe failure must never cost the render
            log(f"int8 probe failed ({error}); staying on the fp16 engine")
            return "fp16"

    def _calibrate_gate(self, hr_frame):
        """Feed the auto tuner once per rendered frame: the gate's LR delta is free (it was
        just computed) and the HR delta costs one 1/8 luma pool of the output."""
        import torch.nn.functional as F
        # no clone: the frame already owns the tensor and this runs before it is handed on
        tensor = getattr(hr_frame, "_tensor", None)
        if tensor is None:
            tensor = hr_frame.get_frame_tensor()
        block = 32 * max(1, int(self.modelScale or 1))
        # stride-2 subsample first: the statistic is a block mean, so halving the grid changes
        # it by well under a level while cutting the read by 4x
        luma = tensor[:, :, ::2, ::2].mean(dim=1, keepdim=True).float()
        k = max(1, block // 16)
        grid = F.avg_pool2d(F.avg_pool2d(luma, 8), k, ceil_mode=True) * 255.0
        # The tuner has to score a candidate the way the gate will actually use it: a frame
        # may be served from a reference up to `max_chain` frames old, and that is where the
        # visible error comes from. Measured on the film clip at threshold 5.0 with chain 4,
        # the worst frames were 34.7 dB while a single-step model predicted "p90 error under
        # 2 levels". So keep the recent grids and take the worst delta over every chain depth
        # the gate can reach.
        history = getattr(self, "_hr_history", None)
        if history is None:
            history = self._hr_history = []
        # The gate drops its reference at a scene cut, so a frame in the next shot is never served
        # from the previous one. The tuner's history has to reset at the same frames, or its
        # "two frames back" delta measures reusing across the cut - a configuration the gate does
        # not have. Measured on Chainsaw Man ep01: with the history spanning cuts, calm frames read
        # 0.2 at depth 1 and 16-35 at depths 2-4, so the chain was pinned to 1 on every title whose
        # first 16 frames contain a cut, and forcing chain 4 by hand measured 0.005% (the same as
        # chain 1) at 1.7x. With the reset, the depths describe what the chain can actually do.
        if getattr(self.temporalReuse, "last_scene_cut", False):
            history.clear()
        depth = int(getattr(self.temporalReuse, "max_chain", 0) or 0)
        # The acceptance statistic, on the same calibration frames and for free: the tuner has been
        # certifying a block-mean proxy (2 levels at depth 1, 12 deeper) while the pictures are
        # judged by "share of pixels more than 8 levels off in any RGB channel, worst frame of the
        # sample". Two instruments, so a rule that "proved" the error could still fail the bar -
        # which is exactly the class of mistake this whole investigation is about. Stride-2 RGB at
        # 8 bits is 6 MB per 4K frame and four of them fit in the gate's own budget.
        accept = []
        frames = getattr(self, "_hr_rgb8", None)
        if frames is None:
            frames = self._hr_rgb8 = []
        rgb8 = (tensor[:, :, ::2, ::2].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        for past in reversed(frames[-max(depth, 1):]):
            delta = (rgb8.to(torch.int16) - past.to(torch.int16)).abs().max(dim=1).values
            accept.append(float((delta > 8).float().mean()) * 100.0)
        if history:
            # one delta per chain depth, so the tuner can certify "two frames deep" when
            # "four" is too stale rather than throwing the gate away
            deltas = [float(abs(grid - past).max().item())
                      for past in reversed(history[-max(depth, 1):])]
            lr_delta = float(getattr(self.temporalReuse, "last_delta", 0.0) or 0.0)
            self.autoTuner.add(lr_delta, deltas, accept)
        if os.environ.get("RVE_CALIB_DEBUG") == "1":
            log(f"calib: depth {depth}, hr frames {len(frames)}, accept {len(accept)}, "
                f"lr {float(getattr(self.temporalReuse, 'last_delta', 0.0) or 0.0):.2f}")
        frames.append(rgb8)
        if len(frames) > max(1, depth):
            del frames[:-max(1, depth)]
        history.append(grid)
        if len(history) > max(1, depth):
            del history[:-max(1, depth)]
        self._hr_grid_prev = grid
        if self.autoTuner.ready:
            if self.autoTuner.chosen is None:
                # quality preset "best": the tuner could not certify any threshold on this
                # content, so stop reusing frames entirely rather than pick the least-bad one
                log(self.autoTuner.stats())
                self.temporalReuse = None
                self.autoTuner = None
                return
            self.temporalReuse.threshold = self.autoTuner.chosen
            self.temporalReuse.pixel_veto = self.autoTuner.veto
            if self.autoTuner.chosen_chain:
                self.temporalReuse.max_chain = int(self.autoTuner.chosen_chain)
            log(self.autoTuner.stats())

    def render(self):
        frames_rendered = 0
        """
        from viztracer import VizTracer
        tracer = VizTracer()
        tracer.start()
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()
        """

        while True:
            if not self.informationHandler.get_is_paused():
                _t_read = _perf_counter()
                frame = self.readBuffer.get()
                _read_ms = (_perf_counter() - _t_read) * 1000.0
                #self.write_bytes_to_cv2_frame_debug(frame)
                if frame is None:
                    pending = getattr(self.upscaleOption, "flush", None)
                    if pending is not None:
                        for f in pending():
                            self.informationHandler.setFramesRendered(frames_rendered)
                            self.writeBuffer.writeQueue.put(f.get_frame_bytes())
                            frames_rendered += int(self.ceilInterpolateFactor)
                    self.informationHandler.stopWriting()
                    break

                for extraRestoration in self.extraRestorationModels:
                    frame = extraRestoration(frame)

                if self.interpolateModel:
                    sceneDetect = self.sceneDetect.detect(frame)
                    interpolated_frames = self.interpolateOption(
                        img1=frame,
                        transition=sceneDetect,
                    )
                    if not interpolated_frames:
                        return
                    
                    for interpolated_frame in interpolated_frames:

                        if self.upscaleModel:
                            interpolated_frame = self.upscaleOption(
                                interpolated_frame
                            )
                        if self.override_upscale_scale:
                            interpolated_frame = resize_image_bytes(interpolated_frame.get_frame_bytes(),
                                               width=self.width*self.modelScale,
                                               height=self.height*self.modelScale,
                                               target_width=self.width*self.override_upscale_scale,
                                               target_height=self.height*self.override_upscale_scale,)
                        self.informationHandler.setPreviewFrame(self._preview_bytes(interpolated_frame))
                        self.informationHandler.setFramesRendered(frames_rendered)
                        self.writeBuffer.writeQueue.put(interpolated_frame.get_frame_bytes() if type(interpolated_frame) != bytes else interpolated_frame)
                
                

                _t_frame = _perf_counter()
                if not hasattr(self, "_profile"):
                    self._profile = {"frames": 0, "gate": 0.0, "infer": 0.0, "conv": 0.0,
                                     "frame": 0.0, "to_tensor": 0.0}
                _t_gate = _perf_counter()
                if self.upscaleModel:
                    # during auto calibration every frame is rendered: otherwise the tuner only
                    # ever sees frames that were already "changed", which biases it towards high
                    # thresholds (measured on film: a noise floor of 74 levels out of a calm scene)
                    calibrating = self.autoTuner is not None and not self.autoTuner.ready
                    # The gate is the first consumer of the frame, so `get_frame_tensor()`
                    # here also pays the input conversion (for the 10-bit path: 12.4 MB upload
                    # plus the GPU colour stage). Timing the call separately keeps the gate's
                    # own cost from absorbing it - measured standalone at 0.80 ms, against
                    # 4.3 ms for the whole stage in this profile.
                    _t_conv = _perf_counter()
                    frame_tensor = frame.get_frame_tensor() if self.temporalReuse is not None else None
                    self._profile["to_tensor"] += (_perf_counter() - _t_conv) * 1000.0
                    reuse = (
                        self.temporalReuse is not None
                        and not calibrating
                        and self.temporalReuse.is_duplicate(frame_tensor)
                    )
                    if calibrating:
                        # still advance the gate's own state so it has a reference afterwards
                        self.temporalReuse.is_duplicate(frame_tensor)
                    self._profile["gate"] += (_perf_counter() - _t_gate) * 1000.0
                    if reuse and self._lastRenderedFrame is not None:
                        # re-emit the last rendered frame (the gate only decides *whether*)
                        frame = self._lastRenderedFrame
                    else:
                        frame = self.upscaleOption(
                            frame
                        )
                        if frame is None:
                            # pipelined inference: this frame will come out a call later
                            continue
                        self._lastRenderedFrame = frame
                        if self.autoTuner is not None and not self.autoTuner.ready:
                            self._calibrate_gate(frame)
                
                
                
                if self.override_upscale_scale:
                    frame = resize_image_bytes(frame.get_frame_bytes(),
                                               width=self.width*self.modelScale,
                                               height=self.height*self.modelScale,
                                               target_width=self.width*self.override_upscale_scale,
                                               target_height=self.height*self.override_upscale_scale,)

                
                # opt in before the preview so preview and writer share one payload
                if type(frame) is not bytes and getattr(self, "pooled_output", False):
                    frame.pooled_output = True
                self.informationHandler.setFramesRendered(frames_rendered)
                preview_payload = self._preview_bytes(frame)
                self.informationHandler.setPreviewFrame(preview_payload)
                
                if type(frame) is not bytes and getattr(frame, "pooled_output", False):
                    # acquire the pinned slot and fill it (blocks if the writer is a whole
                    # pool behind - it never overwrites a payload still being written)
                    frame.get_frame_bytes()
                    self.writeBuffer.writeQueue.put(frame)
                else:
                    self.writeBuffer.writeQueue.put(frame.get_frame_bytes() if type(frame) != bytes else frame)
                frames_rendered += int(self.ceilInterpolateFactor)
                if not hasattr(self, "_t_first_frame"):
                    self._t_first_frame = _time.time()
                    log(f"startup: first frame enqueued {self._t_first_frame - _T0:.2f} s "
                        f"after process start (interpreter, engine, ffmpeg/NVENC init)")
                if hasattr(self, "_profile"):
                    self._profile["frames"] += 1
                    self._profile["frame"] += (_perf_counter() - _t_frame) * 1000.0
                    self._profile["read"] = self._profile.get("read", 0.0) + _read_ms
                    up = getattr(self, "upscaleOption", None)
                    # only count frames that actually ran the model: a reused frame leaves
                    # the previous value in place, which used to double-count
                    self._profile["infer"] += getattr(up, "last_inference_ms", 0.0)
                    self._profile["conv"] += getattr(up, "last_convert_ms", 0.0)
                    self._profile["model"] = self._profile.get("model", 0.0) + getattr(
                        up, "last_model_ms", 0.0)
                    self._profile["handover"] = self._profile.get("handover", 0.0) + getattr(
                        up, "last_frame_ms", 0.0)
                    try:
                        up.last_inference_ms = 0.0
                        up.last_convert_ms = 0.0
                        up.last_model_ms = 0.0
                        up.last_frame_ms = 0.0
                    except Exception:
                        pass
            else:
                sleep(1)
        first = getattr(self, "_t_first_frame", None)
        if first is not None:
            log(f"render loop: {_time.time() - first:.2f} s for {frames_rendered} frames "
                f"({1000.0 * (_time.time() - first) / max(1, frames_rendered):.2f} ms/frame), "
                f"total {_time.time() - _T0:.2f} s since process start")
        if self.temporalReuse is not None:
            log(self.temporalReuse.stats())
        if getattr(self, "autoTuner", None) is not None:
            log(self.autoTuner.stats())
        prof = getattr(self, "_profile", None)
        if prof and prof["frames"]:
            n = prof["frames"]
            log(f"profile over {n} frames: gate {prof['gate']/n:.2f} ms "
                f"(input to tensor {prof.get('to_tensor', 0.0)/n:.2f}), "
                f"inference {prof['infer']/n:.2f} ms (engine {prof.get('model', 0.0)/n:.2f}, "
                f"frame handover {prof.get('handover', 0.0)/n:.2f}), "
                f"input convert {prof['conv']/n:.2f} ms, "
                f"read wait {prof.get('read', 0.0)/n:.2f} ms, frame total {prof['frame']/n:.2f} ms")
        if self.upscaleModel and getattr(self, "tileReuse", False):
            up = getattr(self, "upscaleOption", None)
            tu = getattr(up, "tile_upscaler", None)
            if tu is not None:
                log(tu.stats())
        self.writeBuffer.writeQueue.put(None)
        """
        tracer.stop()
        tracer.save()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        """
    
    def upscalePytorchObject(self, modelPath=None):
        from .pytorch.UpscaleTorch import UpscalePytorch
        return UpscalePytorch(
            modelPath,
            device=self.device,
            precision=self.precision,
            width=self.width,
            height=self.height,
            backend=self.backend,
            tilesize=self.tilesize,
            gpu_id=self.pytorch_gpu_id,
            trt_optimization_level=self.trt_optimization_level,
            trt_tiling_level=self.trt_tiling_level,
            trt_timing_iters=self.trt_timing_iters,
            trt_streams=self.trt_streams,
            pipe_pixel_format=getattr(self, "pipePixelFormat", "yuv420p10le"),
            hdr_mode=self.hdr_mode,
            high_bitdepth=getattr(self, "high_bitdepth", self.hdr_mode),
            tile_reuse=getattr(self, "tileReuse", False),
            tile_lr=getattr(self, "tileReuseSize", 512),
            tile_threshold=getattr(self, "tileReuseThreshold", 0.35),
            trt_static_shape= not self.trt_dynamic_shapes,
        )
    
    def upscaleNCNNObject(self, scale=None, modelPath=None):
        from .ncnn.UpscaleNCNN import UpscaleNCNN
        path, last_folder = os.path.split(modelPath)
        modelPath = os.path.join(path, last_folder, last_folder)
        return UpscaleNCNN(
            modelPath=modelPath,
            num_threads=1,
            scale=self.upscaleTimes if scale is None else scale,
            gpuid=self.ncnn_gpu_id,  # might have this be a setting
            width=self.width,
            height=self.height,
            tilesize=self.tilesize,
        )
    def upscaleONNXObject(self, scale=None, modelPath=None):
        from .onnx.UpscaleONNX import UpscaleONNX
        return UpscaleONNX(
            modelPath=modelPath,
            deviceID=self.pytorch_gpu_id,
            precision=self.precision,
            width=self.width,
            height=self.height,
            scale=self.upscaleTimes if scale is None else scale,
            hdr_mode=self.hdr_mode,
        )


    def setupExtraRestoration(self, modelPath):
        log("Setting up Extra Restoration")
        if self.backend == "pytorch" or self.backend == "tensorrt":
            return self.upscalePytorchObject(modelPath)
        
        if self.backend == "ncnn":
            return self.upscaleNCNNObject(scale=1, modelPath=modelPath)

    def setupUpscale(self):
        log("Setting up Upscale")
        if self.backend == "pytorch" or self.backend == "tensorrt":
            # the tile engine is built for the tile window, so the verdict has to be in
            # before upscalePytorchObject() builds anything
            if self.tileReuse and self.tileReuseMode == "auto":
                verdict = self._probe_tile_reuse(
                    getattr(self, "ffmpegPath", None) or self._probe_ffmpeg_path())
                self.tileReuse = bool(verdict and verdict.use_tiles)

            self.upscaleOption = self.upscalePytorchObject(self.upscaleModel)
            self.modelScale = self.upscaleOption.getScale()
            # the int8 body's verdict is taken ONCE, here, before the first frame. It is not a runtime
            # switch: an engine swapped mid-render would mix two error histories inside one gate window.
            if str(getattr(self, "int8Fit", "auto")).lower() == "auto":
                self.int8Fit = self._probe_int8_fit()
                # the verdict has to reach the object that runs frames: gen_render keeps it here, the hook
                # reads it from the upscale option, and until this line existed the body was never built.
                self.upscaleOption.int8Fit = self.int8Fit
                self.upscaleOption.int8Frames = getattr(self, "int8Frames", None)
            

        if self.backend == "ncnn":
            from .ncnn.UpscaleNCNN import getNCNNScale

            self.modelScale = getNCNNScale(modelPath=self.upscaleModel)
            
            self.upscaleOption = self.upscaleNCNNObject(scale=self.modelScale, modelPath=self.upscaleModel)

        if self.backend == "directml":  # i dont want to work with this shit
            from .onnx.UpscaleONNX import UpscaleONNX
            self.modelScale = UpscaleONNX.getModelScale(self.upscaleModel)
            
            self.upscaleOption = UpscaleONNX(
                modelPath=self.upscaleModel,
                precision=self.precision,
                width=self.width,
                height=self.height,
                scale=self.modelScale
            )
        self.upscaleTimes = self.modelScale if not self.override_upscale_scale else self.override_upscale_scale
        


    def setupInterpolate(self):
        log("Setting up Interpolation")
        self.sceneDetect = SceneDetect(
            sceneChangeMethod=self.sceneDetectMethod,
            sceneChangeSensitivity=self.sceneDetectSensitivty,
            width=self.width,
            height=self.height,
            model_path=self.sceneDetectModelPath,
            model_backend=self.backend,
            model_dtype=self.precision,
            model_device=self.device,
            model_gpu_id=self.pytorch_gpu_id if self.backend in ["pytorch","tensorrt"] else self.ncnn_gpu_id,
        )
        if self.sceneDetectMethod != "none":
            log("Scene Detection Enabled")

        else:
            log("Scene Detection Disabled")

        if self.backend == "ncnn":
            from .ncnn.InterpolateNCNN import InterpolateRIFENCNN

            self.interpolateOption = InterpolateRIFENCNN(
                interpolateModelPath=self.interpolateModel,
                width=self.width,
                height=self.height,
                gpuid=self.ncnn_gpu_id,
                max_timestep=self.maxTimestep,
                interpolateFactor=self.ceilInterpolateFactor,
                hdr_mode=self.hdr_mode,
            )

        if self.backend == "pytorch" or self.backend == "tensorrt":
            from .pytorch.InterpolateTorch import InterpolateFactory

            self.interpolateOption = InterpolateFactory.build_interpolation_method(
                self.interpolateModel,
                self.backend,
                self.drba,
            )(
                modelPath=self.interpolateModel,
                ceilInterpolateFactor=self.ceilInterpolateFactor,
                width=self.width,
                height=self.height,
                device=self.device,
                dtype=self.precision,
                backend=self.backend,
                gpu_id=self.pytorch_gpu_id,
                UHDMode=self.UHD_mode,
                drba=self.drba,
                trt_optimization_level=self.trt_optimization_level,
                ensemble=self.ensemble,
                dynamicScaledOpticalFlow=self.dynamic_scaled_optical_flow,
                max_timestep=self.maxTimestep,
                hdr_mode=self.hdr_mode,
                trt_static_shape= not self.trt_dynamic_shapes,  # if dynamic shapes are enabled, we have to set the static shape to false (default is true in the model
            )