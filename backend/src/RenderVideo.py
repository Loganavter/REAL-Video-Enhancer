import time as _time
from threading import Thread

# Process-relative clock: the wall-clock "Time to complete render" starts in the writer
# thread, so it hides everything before it (interpreter/import, engine build or deserialize,
# ffmpeg + NVENC init) and the flush at the end. On a 94-frame clip that fixed part is
# ~1.5 s, i.e. 16 ms/frame - large enough to be misread as per-frame overhead, which is
# exactly what happened to the budget table. These marks split it out.
_T0 = _time.time()
import torch

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
from .utils.Frame import rgb_preview, payload_preview
from .utils.PreviewStager import PreviewStager
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


def resolve_int8_fit(int8_fit: str | None, quality_preset: str) -> str:
    """The Best/Fast switch picks the body's precision; an explicit --int8_fit wins over it.

    `best` runs the fp16 engine (the picture is the model's own output), `fast` runs the int8 body.
    Plain `--int8_fit` is left alone, which is what every measurement run in harness/docs/wsp passes.
    """
    if quality_preset not in ("best", "fast"):
        raise ValueError(f"quality_preset must be 'best' or 'fast', got {quality_preset!r}")
    if int8_fit is None:
        return "fp16" if quality_preset == "best" else "int8"
    if int8_fit not in ("auto", "int8", "fp16"):
        raise ValueError(f"int8_fit must be auto/int8/fp16, got {int8_fit!r}")
    return int8_fit


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
        preview_size: tuple = (640, 360),
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
        temporal_reuse: bool = False,
        # OFF by default since 2026-11-18 (owner decision). On a 1080p source the gate's own noise
        # floor is almost always above its 1.5 refusal limit - the material always carries digital
        # noise - and at low contrast a small delta may be real motion rather than noise. Measured on
        # BD stream copies: Violet refuses (1.67) and renders bit-identically to ungated; the calm
        # Chainsaw shot engages and then exceeds the acceptance bar (0.52379%, 17/539 frames). It is
        # an opt-in measurement tool now, not a default. See harness/docs/wsp/verified/gate-bd.txt.
        # TheAnimeScripter calibration: --dedup_sens 35 / 7 = 5.0 luma levels, max span 4
        temporal_reuse_threshold: float = 5.0,
        temporal_reuse_statistic: str = "mean_abs",
        int8_fit: str | None = None,
        temporal_reuse_max_block_mean: float | None = None,
        temporal_reuse_max_chain: int = 4,
        # 0/None = off. 40 is the TheAnimeScripter value and is right for clean anime, but on
        # grainy film a single ringing pixel exceeds it: measured 43 vetoes in 94 frames and
        # reuse collapsed to 1.1% (vs 46.8% with the veto off).
        temporal_reuse_pixel_veto: float | None = 0.0,
        temporal_reuse_auto: bool = True,
        # the correction the reused frame gets. Without it the gate re-emits the reference's
        # picture verbatim, which loses everything that changed (measured 37/130 over the bar);
        # with it the frequency-split residual takes that to 0/130 on the calibration cut.
        temporal_reuse_correction: bool = True,
        # machine-readable run report: what ran, on what, and what each stage cost. `harness.ingest`
        # turns it into records, so "it got slower" becomes a number with a commit and a card instead
        # of a memory. None = do not write one.
        report_path: str | None = None,
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
        # the probe runs BEFORE the gate object exists (setupUpscale precedes the gate block), so the
        # filter has to come from the configuration, not from the gate
        self.temporalReuseStatistic = temporal_reuse_statistic
        self.temporalReuseMaxChain = temporal_reuse_max_chain
        self.ffmpegPath = ffmpeg_path
        if quality_preset not in ("best", "fast"):
            raise ValueError(f"quality_preset must be 'best' or 'fast', got {quality_preset!r}")
        self.qualityPreset = quality_preset
        # The window's Best/Fast switch is the int8 switch (2026-11-18). The gate it used to control is
        # off by default, so the old meaning was dead. `None` means "follow the preset"; an explicit
        # --int8_fit still wins, which is what every measurement run in harness/docs/wsp uses.
        # "auto" asks the probe; "int8"/"fp16" force it, for measurement only
        self.int8FitPreset = resolve_int8_fit(int8_fit, quality_preset)
        self.int8Fit = self.int8FitPreset
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
        # Thumbnail geometry of the preview channel. Fixed by the GUI (`--preview_size`)
        # so the block is sized once, before the backend starts.
        self.previewWidth, self.previewHeight = int(preview_size[0]), int(preview_size[1])
        self._previewStager = None
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
        # 8-bit planar YUV pipe: opt-in until the A/B against rgb24 says the picture is unchanged.
        # Halves the per-frame D2H (24.9 -> 12.4 MB at 4K) and removes ffmpeg's CPU scale filter.
        if not high_bitdepth and os.environ.get("RVE_YUV8", "1") != "0":
            # The packer's coefficients are BT.709. A source that declares another matrix (bt601 /
            # smpte170m) would be converted with the wrong ones and land in the C3 class, so that
            # case keeps the rgb24 pipe, where ffmpeg is told the source's own matrix explicitly.
            declared = (color_space or "").lower()
            if declared in ("", "bt709"):
                self.pipePixelFormat = "yuv420p"
                log("8-bit pipe: packing yuv420p on the GPU - 1.5 bytes/px instead of rgb24's 3, "
                    "and no ffmpeg scale filter (RVE_YUV8=0 reverts)")
            else:
                log(f"8-bit pipe: NOT packing yuv420p - the source declares {declared}, and the "
                    f"packer's matrix is bt709 (keeping rgb24 so ffmpeg uses the declared one)")
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
            log(f"DEBUG after hotReload: option={id(self.upscaleOption)} int8Fit={getattr(self.upscaleOption,'int8Fit','<unset>')}")

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

        self.informationHandler = InformationWriteOut(
            sharedMemoryID=sharedMemoryID,
            paused_shared_memory_id=pause_shared_memory_id,
            outputWidth=self.originalWidth * (self.override_upscale_scale or self.upscaleTimes),
            outputHeight=self.originalHeight * (self.override_upscale_scale or self.upscaleTimes),
            totalOutputFrames=self.totalOutputFrames,
            preview_size=(self.previewWidth, self.previewHeight),
        )
        if self.informationHandler.previewAttached():
            self._previewStager = PreviewStager(self.previewWidth, self.previewHeight)
            self._warm_preview()
        

        self.reportPath = report_path
        self.temporalReuse = None
        # created together with the gate, below; None means a reused frame is emitted verbatim
        self.temporalCorrection = None
        self.temporalReuseCorrection = bool(temporal_reuse_correction)
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
                max_block_mean=temporal_reuse_max_block_mean,
            )
            if self.temporalReuseCorrection:
                from .utils.TemporalCorrection import ReferenceFrameCorrector
                self.temporalCorrection = ReferenceFrameCorrector()
                log("Reference-frame reuse: the correction is ON (frequency split, recursive base, "
                    "alpha 1.35 on flat static blocks)")
            log(f"Reference-frame reuse enabled (threshold {temporal_reuse_threshold}/255, "
                f"statistic {temporal_reuse_statistic}, "
                f"max chain {temporal_reuse_max_chain}, pixel veto {temporal_reuse_pixel_veto})")
            self.autoTuner = None
            # The tuner works in LEVELS and writes its choice into `threshold`. With a share statistic
            # that write is a unit mismatch, and it fails OPEN: a chosen 3.0 becomes "3% of pixels",
            # which is never exceeded, so every frame is reused and the defect comes back. Measured
            # 2026-11-18: threshold 0.07 overwritten by 3.0, reuse 89/122. Refuse the combination
            # instead of converting silently.
            if temporal_reuse_auto and str(temporal_reuse_statistic) != "mean_abs":
                log(f"auto gate: DISABLED - it tunes a threshold in 8-bit levels and the gate is "
                    f"measuring {temporal_reuse_statistic} (a share); writing one into the other "
                    f"fails open. Pass --no_temporal_reuse_auto and a threshold in the statistic's "
                    f"own unit.")
                temporal_reuse_auto = False
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
        # Pooled pinned output. DISABLED by default. The earlier attempt was wired as the worker
        # specified and came out 12.6 dB against the reference, because slots were rewritten before
        # ffmpeg had the bytes; the pool now carries a semaphore permit per slot and the writer
        # releases after `stdin.write()` returns, so the protocol the old attempt lacked is in place
        # (utils.Frame.PinnedSlotPool, FFmpegBuffers.release_payload). RVE_PACK_RING=1 opts in for the
        # A/B that decides whether it stays: the pack line and the loop are measured beside the
        # byte-identity of the render, because a payload ring that moves the picture is worse than
        # the copy it removes.
        self.pooled_output = str(os.environ.get("RVE_PACK_RING", "0")).lower() in ("1", "true", "yes")
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

    def _publish_preview(self, frame):
        """Publish the thumbnail that has arrived, and enqueue this frame's.

        The loop never waits for the device: `take()` hands over the previous frame's
        buffer once its copy completed, `submit()` starts this frame's gather and D2H.
        Both are cheap and bounded; the device's latency is what used to show up as a
        1.1-2.2 ms tail, and it is now outside the frame.
        """
        stager = self._previewStager
        if stager is None or not self.informationHandler.previewWanted():
            return False
        published = False
        ready = stager.take()
        if ready is not None:
            payload, fmt, high_bitdepth = ready
            published = bool(self.informationHandler.publishPreview(
                payload, self.previewWidth, self.previewHeight,
                fmt=fmt, high_bitdepth=high_bitdepth,
            ))
        try:
            if not stager.busy():
                stager.submit(frame, high_bitdepth=bool(getattr(frame, "hdr_mode", False)))
        except Exception as exc:
            # a broken preview must never cost a render: drop the frame, keep the last one
            log(f"preview: dropping this frame ({exc})")
        return published

    def _warm_preview(self):
        """Pay the thumbnail's cold cost once, at startup, not on the first published frame.

        The first call of this op mix loads the CUDA modules and stages the first transfer:
        measured 8.2 ms once against 0.2-0.5 ms afterwards (`--preview_size` both arms).
        The preview budget is a per-frame *maximum*, so that one-off belongs at init, where
        it is not a frame - and only when a viewer is attached at all.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scale = self.override_upscale_scale or self.upscaleTimes
        out_h = int(self.originalHeight) * int(scale)
        out_w = int(self.originalWidth) * int(scale)
        try:
            # the real output geometry, because the cold cost is the allocator seeing this
            # size for the first time - a warm-up at the input size does not prevent it
            thumb_src = torch.empty((1, 3, out_h, out_w), dtype=torch.float16, device=device)
            rgb_preview(thumb_src[0], self.previewWidth, self.previewHeight)
            payload = torch.empty(out_h * out_w + 2 * (out_h // 2) * (out_w // 2),
                                  dtype=torch.uint8, device=device)
            payload_preview(payload, out_h, out_w, self.previewWidth, self.previewHeight)
            del thumb_src, payload
        except Exception as exc:
            log(f"preview: warm-up skipped ({exc})")

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

    def _probe_int8_fit(self, certify: bool = True):
        """Ask the source for the body's calibration, and (opt-in) for a picture verdict.

        The calibration is the part the body cannot run without: per-layer scales measured on THIS video.
        The verdict is not taken on the shipped path any more - a forced `int8Fit` (which is what the Fast
        preset sets) overwrites it, and its pair set moved with the configuration, so its percentages are
        not evidence (2026-11-19; `harness/docs/wsp/verified/int8-wall-ab.txt` and the queue row
        `rve-int8-probe-population-2026-11-18`). Callers that force the verdict pass `certify=False` and
        save the reference forwards.
        """
        try:
            from .utils.Int8Probe import probe
            # The pair filter must be the gate's own variable and threshold, or the probe certifies a
            # population the render never serves (it used mean(R,G,B) at 5.0 while the gate's tuner
            # clamps to (2.0, 4.0) in levels).
            gate_stat = getattr(self, "temporalReuseStatistic", "mean_abs")
            result = probe(
                self.inputFile,
                model_path=self.upscaleModel,
                width=int(self.width),
                height=int(self.height),
                bar=float(getattr(self, "int8FitBar", 0.1)),
                gate_threshold=float(getattr(self, "temporalReuseThreshold", 5.0)),
                filter_statistic=("channel_max_share" if gate_stat == "channel_max_share"
                                  else "block_mean"),
                # the fp16 side of every composition number must be the engine the render falls back
                # to, not a torch model: measured on Violet the two differ by 0.044% -> 0.081% in the
                # own defect and 0.168% -> 0.216% at depth 2.
                engine_ref=getattr(self, "upscaleOption", None),
                certify=certify,
            )
            log(result.stats())
            if certify and not result.use_int8:
                log(f"  {result.reason}")
            # hand the decoded calibration frames to the body: they are what makes its grids cover the video
            self.int8Frames = getattr(result, "frames", None)
            verdict = "int8" if result.use_int8 else "fp16"
            if certify and verdict == "int8" and not self._int8_body_serves_this_render():
                # true only when the gate is ON: it renders tiles, and the body is wired into the
                # whole-frame path only. With the gate off the body DOES run, and the old message here
                # said "fp16 engine runs" in exactly that case - a log line contradicting the render.
                log("int8 body: the probe admits this video, but the temporal gate renders tiles and the "
                    "body is wired into the whole-frame path only - against the tiled engine the body "
                    "loses (56.8 vs 35.85 ms at 1080p), so the fp16 engine runs")
            return verdict
        except Exception as error:  # a probe failure must never cost the render
            log(f"int8 probe failed ({error}); staying on the fp16 engine")
            return "fp16"

    def _int8_body_serves_this_render(self) -> bool:
        """The body is wired into the whole-frame path; a created gate means the render is tiled."""
        return getattr(self, "temporalReuse", None) is None

    def _calibrate_gate(self, hr_frame):
        """Feed the auto tuner once per rendered frame: the gate's LR delta is free (it was
        just computed) and the HR delta costs one 1/8 luma pool of the output."""
        import torch.nn.functional as F
        # no clone: the frame already owns the tensor and this runs before it is handed on
        tensor = getattr(hr_frame, "_tensor", None)
        block = 32 * max(1, int(self.modelScale or 1))
        # stride-2 subsample first: the statistic is a block mean, so halving the grid changes
        # it by well under a level while cutting the read by 4x
        if tensor is not None and hr_frame._is_raw_payload():
            # the tail wrote the pipe payload itself: the Y plane IS the luma, no decode needed
            h, w = int(hr_frame.height), int(hr_frame.width)
            luma = tensor[: h * w].view(1, 1, h, w).float() / 255.0
            luma = luma[:, :, ::2, ::2]
        else:
            if tensor is None:
                tensor = hr_frame.get_frame_tensor()
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
        # The acceptance estimate needs an RGB tensor. The int8 tail can hand over a PACKED yuv420p payload
        # instead (11.87 MB for a 4K frame, no fp16 RGB intermediate), and slicing that as a 4-D tensor killed
        # the render thread on 2026-11-20 - the first run where int8, reuse-by-default and the auto tuner met.
        # Luma still certifies the threshold: `AutoGateTuner._uses_acceptance` falls back to the proxy
        # instrument when no acceptance samples arrive, so the decision stays, only the second opinion is gone.
        rgb_capable = tensor is not None and tensor.dim() == 4 and tensor.shape[1] >= 3
        rgb8 = None
        if rgb_capable:
            rgb8 = (tensor[:, :, ::2, ::2].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            for past in reversed(frames[-max(depth, 1):]):
                delta = (rgb8.to(torch.int16) - past.to(torch.int16)).abs().max(dim=1).values
                accept.append(float((delta > 8).float().mean()) * 100.0)
        elif not getattr(self, "_calib_rgb_skipped", False):
            log("auto gate: the output frame is a packed yuv420p payload, so the acceptance estimate (share of "
                "pixels >8 levels in RGB) is not computed on it; the luma grids still certify the threshold")
            self._calib_rgb_skipped = True
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
        if rgb8 is not None:
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
                self.temporalCorrection = None
                return
            self.temporalReuse.threshold = self.autoTuner.chosen
            self.temporalReuse.pixel_veto = self.autoTuner.veto
            if self.autoTuner.chosen_chain:
                self.temporalReuse.max_chain = int(self.autoTuner.chosen_chain)
            log(self.autoTuner.stats())

    def _reuse_frame(self, lr_tensor):
        """The frame a reuse emits: the last rendered frame plus the correction for what changed.

        Falls back to the verbatim reference - exactly the old behaviour - whenever the corrector has no pair to
        work from (first frame, seek, a shape change) or raises. A correction must never be able to break a render.
        """
        cached = self._lastRenderedFrame
        if self.temporalCorrection is None or lr_tensor is None:
            return cached
        try:
            corrected = self.temporalCorrection.correct(lr_tensor)
        except Exception as exc:
            log(f"reuse correction failed ({exc}); reuses emit verbatim for the rest of this render")
            self.temporalCorrection = None
            return cached
        if corrected is None:
            return cached
        from .utils.Frame import Frame
        frame = Frame(cached.backend, cached.width, cached.height, device=cached.device,
                      gpu_id=cached.gpu_id, hdr_mode=cached.hdr_mode, dtype=cached.dtype,
                      high_bitdepth=cached.high_bitdepth)
        frame.yuv10_output = cached.yuv10_output
        frame.yuv8_output = cached.yuv8_output
        # same stream as the correction's ops, so the ordering is preserved without a device barrier
        frame.set_frame_tensor(corrected, sync=False)
        return frame

    def _run_report(self) -> dict:
        """What ran, on what, and what each stage cost - the machine-readable twin of the log summary."""
        from .utils.Provenance import stamp
        prof = getattr(self, "_profile", None) or {}
        n = max(1, prof.get("frames", 0))
        gate = self.temporalReuse
        auto = getattr(self, "autoTuner", None)

        def per_frame(key):
            return round(prof.get(key, 0.0) / n, 3) if prof.get("frames") else None

        phases = {k: per_frame(k) for k in ("gate", "to_tensor", "up_wall", "infer", "model", "handover",
                                            "conv", "preview", "pack", "read", "frame")}
        named = sum(prof.get(k, 0.0) for k in ("gate", "up_wall", "read", "preview", "pack")) / n
        phases["unaccounted"] = round(prof.get("frame", 0.0) / n - named, 3) if prof.get("frames") else None
        return {
            "provenance": stamp(".", model=self.upscaleModel),
            "run": {"input": getattr(self, "inputFile", None), "output": getattr(self, "outputFile", None),
                    "backend": getattr(self, "backend", None), "model": self.upscaleModel,
                    "precision": str(getattr(self, "precision", "")), "frames": prof.get("frames")},
            "settings": {
                "temporal_reuse": self.temporalReuse is not None,
                "temporal_reuse_correction": bool(self.temporalReuseCorrection),
                "temporal_reuse_threshold": getattr(self, "temporalReuseThreshold", None),
                "temporal_reuse_max_chain": getattr(self, "temporalReuseMaxChain", None),
                "temporal_reuse_statistic": getattr(self, "temporalReuseStatistic", None),
                "int8_fit": getattr(self, "int8FitPreset", None),
            },
            "phases_ms_per_frame": phases,
            "gate": None if gate is None else {
                "processed": gate.processed, "reused": gate.reused,
                "reuse_pct": round(100.0 * gate.reused / gate.processed, 2) if gate.processed else None,
                "depth_histogram": dict(sorted(gate.depth_counts.items())),
                "vetoes": gate.vetoes, "scene_cuts": gate.scene_cuts,
            },
            "autotuner": None if auto is None else {
                "chosen_threshold": getattr(auto, "chosen", None),
                "chosen_chain": getattr(auto, "chosen_chain", None),
                "stats": auto.stats() if hasattr(auto, "stats") else None,
            },
        }

    def _write_run_report(self) -> None:
        if not self.reportPath:
            return
        try:
            import json
            report = self._run_report()
            with open(self.reportPath, "w") as handle:
                json.dump(report, handle, indent=2)
            log(f"run report: {self.reportPath}")
        except Exception as exc:  # a report must never take a render down
            log(f"run report FAILED ({exc}); the render is unaffected")

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
                        self._publish_preview(interpolated_frame)
                        self.informationHandler.setFramesRendered(frames_rendered)
                        self.writeBuffer.writeQueue.put(interpolated_frame.get_frame_bytes() if type(interpolated_frame) != bytes else interpolated_frame)
                
                

                _t_frame = _perf_counter()
                if not hasattr(self, "_profile"):
                    self._profile = {"frames": 0, "gate": 0.0, "infer": 0.0, "conv": 0.0,
                                     "frame": 0.0, "to_tensor": 0.0,
                                     # the three stages the old line did NOT name, which is why the
                                     # frame total sat above its own parts by 6.2 ms on 2026-11-18
                                     # (rve-frame-accounting-2026-11-19)
                                     "up_wall": 0.0, "preview": 0.0, "pack": 0.0,
                                     # every published thumbnail's cost, so the acceptance question
                                     # ("no frame pays more than 1 ms") is answered by a distribution
                                     # and not by a maximum that could be a single noisy call
                                     "preview_n": 0, "preview_samples": []}
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
                        # The gate only decides *whether*. What is emitted is the last rendered
                        # frame plus the correction for what changed since - verbatim when the
                        # corrector has no pair to work from (first frame, seek, shape change).
                        frame = self._reuse_frame(frame_tensor)
                    else:
                        _t_up = _perf_counter()
                        frame = self.upscaleOption(
                            frame
                        )
                        if frame is None:
                            # pipelined inference: this frame will come out a call later
                            continue
                        self._profile["up_wall"] += (_perf_counter() - _t_up) * 1000.0
                        self._lastRenderedFrame = frame
                        if self.temporalCorrection is not None and frame_tensor is not None:
                            try:
                                # the model output is the pair a later reuse starts from; the
                                # tensor it holds is not copied, the corrector only reads it
                                seed_hr = getattr(frame, "_tensor", None)
                                if seed_hr is None:
                                    seed_hr = frame.get_frame_tensor()
                                self.temporalCorrection.seed(seed_hr, frame_tensor)
                            except Exception as _corr_exc:      # never let the correction break a render
                                log(f"reuse correction: seeding failed ({_corr_exc}); reuses emit verbatim")
                                self.temporalCorrection = None
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
                _t_prev = _perf_counter()
                preview_built = self._publish_preview(frame)
                _preview_ms = (_perf_counter() - _t_prev) * 1000.0
                self._profile["preview"] += _preview_ms
                if preview_built:
                    self._profile["preview_n"] += 1
                    self._profile["preview_samples"].append(_preview_ms)
                
                _t_pack = _perf_counter()
                if type(frame) is not bytes and getattr(frame, "pooled_output", False):
                    # acquire the pinned slot and fill it (blocks if the writer is a whole
                    # pool behind - it never overwrites a payload still being written)
                    frame.get_frame_bytes()
                    self.writeBuffer.writeQueue.put(frame)
                else:
                    self.writeBuffer.writeQueue.put(frame.get_frame_bytes() if type(frame) != bytes else frame)
                self._profile["pack"] += (_perf_counter() - _t_pack) * 1000.0
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
            named = (prof["gate"] + prof.get("up_wall", 0.0) + prof.get("read", 0.0)
                     + prof.get("preview", 0.0) + prof.get("pack", 0.0))
            samples = prof.get("preview_samples", [])
            if samples:
                ordered = sorted(samples)
                p95 = ordered[min(len(ordered) - 1, int(round(len(ordered) * 0.95)))]
                preview_text = (f"preview {prof.get('preview', 0.0)/n:.2f} ms "
                                f"({len(samples)} built: p50 {ordered[len(ordered)//2]:.2f}, "
                                f"p95 {p95:.2f}, max {ordered[-1]:.2f}, first {samples[0]:.2f})")
            else:
                preview_text = f"preview {prof.get('preview', 0.0)/n:.2f} ms (0 built)"
            log(f"profile over {n} frames: gate {prof['gate']/n:.2f} ms "
                f"(input to tensor {prof.get('to_tensor', 0.0)/n:.2f}), "
                f"model wall {prof.get('up_wall', 0.0)/n:.2f} ms "
                f"(self-reported {prof['infer']/n:.2f}, engine {prof.get('model', 0.0)/n:.2f}, "
                f"frame handover {prof.get('handover', 0.0)/n:.2f}), "
                f"input convert {prof['conv']/n:.2f} ms, "
                f"{preview_text}, "
                f"pack+queue {prof.get('pack', 0.0)/n:.2f} ms, "
                f"read wait {prof.get('read', 0.0)/n:.2f} ms, "
                f"named {(named/n):.2f} of frame total {prof['frame']/n:.2f} ms, "
                f"unaccounted {(prof['frame'] - named)/n:.2f} ms")
        if self.upscaleModel and getattr(self, "tileReuse", False):
            up = getattr(self, "upscaleOption", None)
            tu = getattr(up, "tile_upscaler", None)
            if tu is not None:
                log(tu.stats())
        self._write_run_report()
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
            # The render thread can build the body before this method finishes (observed: "int8: start
            # ... int8Fit=<unset>" logged before the probe's assignment), so a FORCED verdict has to
            # reach the option here, at creation, not after the probe.
            if str(getattr(self, "int8Fit", "auto")).lower() in ("int8", "fp16"):
                self.upscaleOption.int8Fit = str(self.int8Fit).lower()
            # the int8 body's calibration is taken ONCE, here, before the first frame. It is not a
            # runtime switch: an engine swapped mid-render would mix two error histories inside one gate
            # window. ALWAYS run the probe: it is also what decodes the spread frames the body calibrates
            # on. A forced verdict used to skip it, which silently left the body on a single-frame
            # calibration (UpscaleTorch falls back to `frame` when int8Frames is None) - on Violet that
            # read 0.566% against the 0.050% the spread calibration gives, i.e. the flag was measuring
            # itself.
            #
            # The verdict is certification, and certification is withdrawn from the evidence chain
            # (2026-11-19): its pair set moved with the configuration, and in this path it never decided
            # anything - a forced int8Fit overwrites it. So a forced caller asks for calibration only and
            # skips the reference forwards; only `auto` still certifies.
            _forced = str(getattr(self, "int8Fit", "auto")).lower()
            # a diagnostic override: certifying allocates the quantised twin (~2 GB) and that changes the
            # process's memory state, which cudnn's algorithm choice can depend on. RVE_INT8_CERTIFY=1
            # re-runs the certificate in a forced run precisely so that hypothesis can be tested.
            _certify = _forced not in ("int8", "fp16") or \
                str(os.environ.get("RVE_INT8_CERTIFY", "")).lower() in ("1", "true", "yes")
            _probe_verdict = self._probe_int8_fit(certify=_certify)
            log(f"int8 calibration frames handed to the body: "
                f"{len(getattr(self, 'int8Frames', None) or [])} "
                f"(forced verdict: {_forced})")
            if _forced in ("int8", "fp16"):
                # in calibration-only mode there is no verdict to quote - saying "the probe said int8"
                # would invent one out of the "admitted by default" return value
                log(f"int8 fit FORCED to {_forced}; " + (f"the probe said {_probe_verdict}"
                                                          if _certify else "the probe calibrated only"))
                self.int8Fit = _forced
            else:
                self.int8Fit = _probe_verdict
            # the verdict has to reach the object that runs frames: gen_render keeps it here, the hook
            # reads it from the upscale option, and until this line existed the body was never built.
            # It must run for a FORCED verdict too, not only for the probe's.
            self.upscaleOption.int8Fit = self.int8Fit
            self.upscaleOption.int8Frames = getattr(self, "int8Frames", None)
            # ---- determinism diagnostics (2026-11-19, rve-render-determinism) ---------------------
            # Removing the probe's certificate moved 0.45% of pixels between two lossless renders, so the
            # render depends on what happened before its first frame. These two arms separate the two
            # candidates: the ~2 GB quantised-twin allocation, and the fp16 engine being warmed as the
            # probe's reference. Env-gated, no effect when unset, and both run at the point the certificate
            # would have run.
            _diag_mb = int(os.environ.get("RVE_DIAG_ALLOC_MB", "0") or 0)
            if _diag_mb:
                _blob = torch.empty(_diag_mb * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)
                del _blob
                log(f"diag: allocated and released {_diag_mb} MB before the first frame (determinism arm)")
            if str(os.environ.get("RVE_DIAG_WARM_ENGINE", "")).lower() in ("1", "true", "yes"):
                try:
                    from .utils.Frame import Frame as _F
                    for _ in range(6):
                        _f = _F("tensorrt", int(self.width), int(self.height), "cuda", 0, False,
                                "float16", high_bitdepth=False)
                        _f.set_frame_tensor(torch.rand(1, 3, int(self.height), int(self.width),
                                                      device="cuda", dtype=torch.float16), copy=True)
                        _ = self.upscaleOption(_f).get_frame_tensor()
                        del _f
                    log("diag: warmed the fp16 engine on 6 frames before the first frame "
                        "(determinism arm)")
                except Exception as _error:  # noqa: BLE001 - a diagnostic must not cost a render
                    log("diag: engine warm-up failed (%s)", _error)
            log(f"DEBUG after assign: option={id(self.upscaleOption)} int8Fit={getattr(self.upscaleOption,'int8Fit','<unset>')}")
            

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