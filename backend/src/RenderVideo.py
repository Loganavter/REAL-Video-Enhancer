import time as _time
from threading import Thread

# Process-relative clock: the wall-clock "Time to complete render" starts in the writer
# thread, so it hides everything before it (interpreter/import, engine build or deserialize,
# ffmpeg + NVENC init) and the flush at the end. On a 94-frame clip that fixed part is
# ~1.5 s, i.e. 16 ms/frame - large enough to be misread as per-frame overhead, which is
# exactly what happened to the budget table. These marks split it out.
_T0 = _time.time()

#: How long the streaming publishes stand aside after a seek's picture went out, seconds. The pane polls at ~10 Hz;
#: 250 ms covers two polls, so the stamped picture is read instead of being overwritten by the next stream frame.
DISPLAY_SEEK_HOLD_S = 0.25
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
from .utils.Util import log, resize_frame_bytes, guard_render_thread, process_uptime
from .utils.LogPolicy import JobPhases, ProgressLimiter, LEVEL_DEBUG
from .utils.BorderDetect import BorderDetect, cache_location
from .utils.VideoInfo import OpenCVInfo
from .utils.DisplayChannel import (
    ETA_UNSET,
    PHASE_BUILDING_ENGINE,
    PHASE_DONE,
    PHASE_FINALISING,
    PHASE_PREPARING,
    PHASE_RENDERING,
    StatusChannel,
    engine_code,
)
import numpy as np


def remove_shared_memory_block(name):
    try:
        from .utils.shm import attach as _attach

        existing_shm = _attach(name)
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


#: The crop record a render carries once, before its first frame, so the front end can decode the LMB
#: "original" with the SAME crop the renderer used instead of guessing from the source's size.
def format_borders_line(width, height, x, y):
    """``BORDERS w=<w> h=<h> x=<x> y=<y>`` - the crop the reader was actually handed, as one record.

    The values are the ones passed to :class:`FFmpegRead` (the detected crop, or the full frame), never the
    source's own size: a reader that crops and a compare that does not show two different pictures. The line
    carries nothing else, so the parser on the other side can require the whole line
    (``src/ui/ProcessTab.py: parse_borders_line``) and treat anything else as unknown instead of half-reading it.
    """
    return f"BORDERS w={int(width)} h={int(height)} x={int(x)} y={int(y)}"


def report_borders(width, height, x, y):
    """Write the one ``BORDERS`` record of this render (see :func:`format_borders_line`).

    Called once, unconditionally, by ``Render.__init__``: a render with detection off reports the full frame,
    and a render whose frame size was never known (an invalid input) logs why there is no record instead of
    inventing a box - the front end reads a missing record as "unknown" and keeps today's uncropped compare.
    """
    box = (width, height, x, y)
    if None in box:
        log("borders: no crop to report (the input frame size was not available)")
        return
    print(format_borders_line(*box), file=sys.stderr, flush=True)


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
        # The pane's composited-frame block and its seek request (both GUI-owned segments, see
        # utils/DisplayChannel.py for the byte layout). None = no viewer, nothing is paid.
        display_state_id: str | None = None,
        display_size: tuple[int, int] | None = None,
        seek_state_id: str | None = None,
        #: The render-status record (see utils/DisplayChannel.py), a GUI-owned pid-scoped segment.
        status_state_id: str | None = None,
        sceneDetectMethod: str = "pyscenedetect",
        sceneDetectSensitivity: float = 3.0,
        trt_optimization_level: int = 5,
        trt_dynamic_shapes: bool = False,
        override_upscale_scale: int | None = None,
        UHD_mode: bool = False,
        slomo_mode: bool = False,
        dynamic_scaled_optical_flow: bool = False,
        ensemble: bool = False,
        output_to_mpv: bool = False,
        # A preview render writes a container a reader can follow while the file grows (fragmented mp4);
        # an export keeps `+faststart`. Passed straight to the writer, never derived here.
        preview_render: bool = False,
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
        # The gate's translation estimate, applied to the cached picture before the residual correction.
        # Without it a reuse keeps the previous frame's geometry, i.e. the drawing jumps by the scan's
        # wobble (0.1-0.5 px here) - small, but the one thing a reuse must not do on line art.
        temporal_reuse_warp: bool = True,
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
        # Phase bookkeeping for this JOB: once-per-job lines and the progress window are per
        # instance, so a second render never inherits the first one's marks or its last tick.
        self._phases = JobPhases()
        self._progress = ProgressLimiter()
        _t_init = _perf_counter()
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
        self.temporalReuseWarp = bool(temporal_reuse_warp)
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
        _t_probe = _perf_counter()
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
        log(f"phase: video probe {(_perf_counter() - _t_probe) * 1000.0:.0f} ms "
            f"({self.width}x{self.height}, {self.totalInputFrames} frames, {self.fps:.3f} fps)")

        # The render-status block (its own pid-scoped segment, see utils/DisplayChannel.py): the numbers the
        # pane used to scrape out of this process's stdout. Created BEFORE the engines are built so the
        # building-engine phase is on the slot, and written beside every log line, never instead of one.
        # None = no viewer; the render pays one `set()` per frame and nothing else.
        self.statusChannel = None
        if status_state_id:
            try:
                self.statusChannel = StatusChannel(status_state_id, engine=engine_code(self.backend))
                if self.statusChannel.attached:
                    log(f"status: publishing to '{status_state_id}' (backend {self.backend})")
                    self._write_status(0, phase=PHASE_PREPARING)
                else:
                    log(f"status: block '{status_state_id}' is not usable; the pane reads no status")
            except Exception as _status_exc:   # a pane must never cost the export
                log(f"status: channel setup failed ({_status_exc}); the render runs without it")
                self.statusChannel = None
        color_space = videoInfo.get_color_space()
        # The pipe packer's luma matrix follows the SOURCE's colour space, not the bit depth: "HDR mode" here is also
        # the 10-bit switch, and a 10-bit bt709 clip packed with bt2020 coefficients measured mean |d| 3.72 against the
        # rgb48 reference where bt709 measured 1.92 (2026-11-23, 1440x1080, 24 frames).
        self.yuv10Matrix = "bt2020" if str(color_space or "").lower().startswith("bt2020") else "bt709"
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
            # Cache beside the report (or the output): the input can be a read-only source, so the
            # decision never goes into its directory. A second render of the same input re-reads it.
            borderDetect = BorderDetect(
                inputFile=self.inputFile,
                ffmpeg_path=ffmpeg_path,
                cache_path=cache_location(output_path=outputFile, report_path=report_path),
            )
            self.width, self.height, self.borderX, self.borderY = (
                borderDetect.getBorders()
            )
            log(
                f"Detected borders: Width,Height:{self.width}x{self.height}, X,Y: {self.borderX}x{self.borderY}"
            )

        # The one machine-readable record of the crop this render used, for the LMB compare: the front end
        # parses it out of this process's output and decodes the SOURCE with the same box, so the two pictures
        # differ by the upscaler's own work rather than by a cropped-versus-uncropped framing. Emitted on EVERY
        # render (also with border detection off, where the box is the full frame), before the reader is built
        # and long before the first frame is written.
        report_borders(self.width, self.height, self.borderX, self.borderY)

        log("Using backend: " + self.backend)
        # The engine build below is the long silent stretch the pane used to learn from the stdout words
        # "This may take a while": the slot says it in the phase byte (the log line is untouched).
        self._write_status(0, phase=PHASE_BUILDING_ENGINE)
        # upscale has to be called first to get the scale of the upscale model
        if upscaleModel:
            _t_model = _perf_counter()
            self.setupUpscale()
            self.upscaleOption.hotUnload()  # unload model to free up memory for trt enging building
            log(f"phase: model load upscale '{os.path.basename(str(self.upscaleModel))}' "
                f"{(_perf_counter() - _t_model) * 1000.0:.0f} ms")
            log("Using Upscaling Model: " + self.upscaleModel)
        else:
            self.upscaleTimes = 1  # if no upscaling, it will default to 1
            self.modelScale = 1
            

        
        if extraRestorationModels:
            for model in extraRestorationModels:
                _t_model = _perf_counter()
                extraRestoration = self.setupExtraRestoration(model)
                if extraRestoration:
                    log("Using Extra Restoration Model: " + model)
                    self.extraRestorationModels.append(extraRestoration)
                    extraRestoration.hotUnload()  # unload model to free up memory for trt enging building
                    log(f"phase: model load restoration '{os.path.basename(str(model))}' "
                        f"{(_perf_counter() - _t_model) * 1000.0:.0f} ms")
        

        if interpolateModel:
            _t_model = _perf_counter()
            self.setupInterpolate()
            log(f"phase: model load interpolation '{os.path.basename(str(self.interpolateModel))}' "
                f"{(_perf_counter() - _t_model) * 1000.0:.0f} ms")
            log("Using Interpolation Model: " + self.interpolateModel)

        if upscaleModel: # load model after interpolation model is loaded, this saves on vram if the user builds 2 separate engines
            self.upscaleOption.hotReload()
            log(f"debug: after hotReload option={id(self.upscaleOption)} int8Fit={getattr(self.upscaleOption,'int8Fit','<unset>')}",
                level=LEVEL_DEBUG)

        for extraRestoration in self.extraRestorationModels:
            extraRestoration.hotReload()

        # Every engine this job needs is built; the reader/encoder and the first frame are still ahead.
        self._write_status(0, phase=PHASE_PREPARING)

        if self.modelScale and self.override_upscale_scale:
            if int(self.modelScale) == int(self.override_upscale_scale):
                log("Override upscale scale is set to the same value as the model scale, this will not change the output resolution.")
                self.override_upscale_scale = False
        
        log(f"Upscale Times: {self.override_upscale_scale if self.override_upscale_scale else self.upscaleTimes}")
        log(f"Interpolate Factor: {self.interpolateFactor}")
        log(f"Total Output Frames: {self.totalOutputFrames}")
        log("Model Scale: " + str(self.modelScale))
        log("HDR Mode: " + str(hdr_mode))

        _t_reader = _perf_counter()
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
        log(f"phase: reader open {(_perf_counter() - _t_reader) * 1000.0:.0f} ms")

        from .FFmpegBuffers import resolve_output_pixel_format
        pixelFormat = resolve_output_pixel_format(
            pixelFormat, self.high_bitdepth, bool(self.interpolateModel))
        _t_writer = _perf_counter()
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
            preview=preview_render,
        )
        log(f"phase: writer/encoder init {(_perf_counter() - _t_writer) * 1000.0:.0f} ms")

        self.informationHandler = InformationWriteOut(
            paused_shared_memory_id=pause_shared_memory_id,
            outputWidth=self.originalWidth * (self.override_upscale_scale or self.upscaleTimes),
            outputHeight=self.originalHeight * (self.override_upscale_scale or self.upscaleTimes),
            totalOutputFrames=self.totalOutputFrames,
        )

        self.reportPath = report_path
        # The pane's block: sized by the GUI, written only while the pane asks. Building it is cheap -
        # no GPU work happens until the first frame is actually published (utils/DisplayStager.py).
        self.displayEvery = max(1, int(os.environ.get("RVE_DISPLAY_EVERY", "1") or 1))
        self.displayChannel = None
        self.displayStager = None
        self._displayCounter = 0
        if display_state_id:
            try:
                from .utils.DisplayChannel import DisplayChannel
                from .utils.DisplayStager import DisplayStager

                box = tuple(display_size) if display_size else (640, 360)
                self.displayChannel = DisplayChannel(display_state_id, box=box)
                if self.displayChannel.attached:
                    self.displayStager = DisplayStager(box[0], box[1])
                    log(f"display: publishing to '{display_state_id}' at most every "
                        f"{self.displayEvery} rendered frame(s), box {box[0]}x{box[1]}")
                else:
                    log(f"display: block '{display_state_id}' is not usable; the pane stays empty")
            except Exception as _display_exc:   # a pane must never cost the export
                log(f"display: channel setup failed ({_display_exc}); the render runs without it")
                self.displayChannel = None
                self.displayStager = None
        # Newest-wins seek request. One slot: a newer request overwrites an older one, and an iteration
        # that finds nothing pays a 12-byte read before it blocks on the reader.
        self.seekState = None
        self.lastSeekSeconds = None
        #: The resident preview decoder (`utils/SeekReader.py`), built by the first seek request and kept
        #: for the rest of the render; closed with the other resources at the end of `render()`.
        self._seekReader = None
        if seek_state_id:
            try:
                from .utils.DisplayChannel import SeekState

                self.seekState = SeekState(seek_state_id)
                if self.seekState.attached:
                    log(f"seek: watching '{seek_state_id}' (newest request wins, "
                        f"generation {self.seekState.last_generation})")
                else:
                    log(f"seek: block '{seek_state_id}' is not usable; the pane cannot seek")
            except Exception as _seek_exc:
                log(f"seek: request block setup failed ({_seek_exc}); the render runs without it")
                self.seekState = None
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
            # `coarse` is in the same unit as `mean_abs` (8-bit levels on a block grid), so the tuner's
            # write is legal there; `channel_max_share` is in percent and is the only ratio statistic.
            if temporal_reuse_auto and str(temporal_reuse_statistic) not in ("mean_abs", "coarse"):
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
                    # Certify with the statistic the pictures are judged by. The tuner collects
                    # acceptance samples from the same forced-render calibration frames
                    # (`_calibrate_gate` -> `autoTuner.add(lr, hr, accept)`), and with the default
                    # "proxy" it then decided on the block-mean budget instead: measured
                    # 2026-11-20 on a stabilised source, the proxy certified threshold 4 / chain 4
                    # and the render came out 0.81% of pixels over the 0.1% bar. Every false
                    # "price of quality" in the ledger is that mismatch; when no samples arrive
                    # (int8 packed payload) `_uses_acceptance` still falls back to the proxy.
                    certify_on="acceptance",
                    # The noise-floor guard's premise - a 32 px block mean cannot separate grain from
                    # motion - is what the coarse statistic removes by construction. With it on, the
                    # coarse statistic's floor (a low-pass level, measured ~3 on the denoised Pink
                    # Panther) would read as "grainy material" against a limit measured at the pixel
                    # scale, and refuse material the acceptance certifier bounds at the same scale the
                    # decision is taken at. The guard stays for mean_abs, where it was calibrated.
                    noise_floor_refuse=(None if str(temporal_reuse_statistic) == "coarse" else 1.5),
                    # The coarse statistic decides on the drawing, so a frame that did not change at all says
                    # nothing about which threshold is right - and a film that opens on black used to spend its
                    # whole calibration there. 2.0 is the narrowest candidate: anything below it is reused whatever
                    # the tuner picks. The `mean_abs` path keeps the old, unfiltered sample: its noise-floor guard
                    # was calibrated on it.
                    min_informative=(2.0 if str(temporal_reuse_statistic) == "coarse" else None),
                )
                if str(temporal_reuse_statistic) == "coarse":
                    log("auto gate: the noise-floor guard is off for the coarse statistic (its premise is "
                        "the grain the low-pass removes); the acceptance certifier still bounds the picture, "
                        "at the same scale as the decision")
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

        # The render body is wrapped (utils.Util.guard_render_thread): an exception here used to kill
        # only this thread, leaving the reader/writer threads blocked on a queue nobody feeds - a live
        # process at 0 FPS, `Status: Rendering` forever and no error anywhere (2026-11-22). The
        # wrapper logs one `RENDER FAILED: <type>: <msg>` line plus the traceback and exits non-zero
        # at once, so the GUI's stdout loop ends the job and shows the failure.
        self.renderThread = Thread(
            target=guard_render_thread,
            args=(self.render,),
            kwargs={"path": "RenderVideo.render"},
            name="render",
        )
        self.ffmpegReadThread = Thread(target=self.readBuffer.read_frames_into_queue)
        self.ffmpegWriteThread = Thread(target=self.writeBuffer.write_out_frames)
        self.sharedMemoryThread = Thread(
            target=self.informationHandler.writeOutInformation
        )

        self.sharedMemoryThread.start()
        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()
        log(f"phase: backend init {(_perf_counter() - _t_init) * 1000.0:.0f} ms "
            f"(backend {self.backend}, device {self.device}, precision {self.precision}); "
            f"{process_uptime():.2f} s since process start")

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

    def _accept_lowpass(self, rgb8, sigma):
        """The acceptance statistic's own coarse scale: the same low-pass the gate's decision is taken at.

        ``rgb8`` is stride-2 of the HR output, so ``sigma`` arrives already divided for that grid. The kernel is
        cached per (sigma, device): it is rebuilt at most once per calibration, but a 4K frame at 3 channels is
        12 MB and the padding is the allocation that matters.
        """
        import torch.nn.functional as F
        key = (round(float(sigma), 3), str(rgb8.device))
        cached = getattr(self, "_accept_kernels", None)
        if cached is None:
            cached = self._accept_kernels = {}
        if key not in cached:
            radius = max(1, int(3.0 * sigma + 0.5))
            x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=rgb8.device)
            k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
            cached[key] = ((k / k.sum()), radius)
        kernel, radius = cached[key]
        channels = int(rgb8.shape[1])
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        out = rgb8.float()
        out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
        out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)
        return out.round().clamp_(0, 255)

    def _lr_history(self, lr_tensor, depth):
        """The last ``depth`` low-resolution frames, for pricing a reuse's residual (calibration only).

        Bounded on purpose: a 4K LR frame is 50 MB in fp16 and five of them would sit on the card beside the
        engine, so the budget decides how deep the acceptance certifier can look. With fewer frames than the
        chain allows, the deeper depths simply get no samples and the tuner certifies only what it could price -
        the honest degradation, and it is logged.
        """
        budget = int(getattr(self, "_lr_history_budget", 64 * 1024 * 1024))
        history = getattr(self, "_lr_hist", None)
        if history is None:
            history = self._lr_hist = []
        if lr_tensor is not None and lr_tensor.dim() == 4:
            # a clone: the reader's input buffer is reused for the next frame
            history.insert(0, lr_tensor.detach().clone())
        keep = 1
        used = 0
        for frame in history:
            used += frame.numel() * frame.element_size()
            if used > budget:
                break
            keep += 1
        keep = min(keep, max(1, int(depth) + 1), len(history))
        if len(history) > keep:
            del history[keep:]
            if not getattr(self, "_lr_history_trimmed", False):
                self._lr_history_trimmed = True
                log(f"auto gate: the acceptance certifier keeps {keep - 1} LR frames "
                    f"({budget // (1024 * 1024)} MB budget), so it can price reuses up to "
                    f"{keep - 1} frames deep; deeper chains are not certified on this resolution")
        return history

    def _accept_shift(self, rgb8, dx, dy):
        """Move a sample by ``(dx, dy)`` grid pixels: the geometry a reuse would apply.

        Takes either an 8-bit levels tensor (the HR output) or an LR frame in [0, 1], so it must NOT round: an
        LR frame rounded to integers is a binary picture, and the residual taken against it read as a 99.6%
        difference on every depth the first time this ran.
        """
        import torch.nn.functional as F
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return rgb8
        h, w = int(rgb8.shape[-2]), int(rgb8.shape[-1])
        ys = torch.arange(h, dtype=torch.float32, device=rgb8.device) - dy
        xs = torch.arange(w, dtype=torch.float32, device=rgb8.device) - dx
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack(((gx / max(1, w - 1)) * 2.0 - 1.0,
                            (gy / max(1, h - 1)) * 2.0 - 1.0), dim=-1).unsqueeze(0)
        return F.grid_sample(rgb8.float(), grid, mode="bilinear", padding_mode="border",
                             align_corners=True)

    def _calibrate_gate(self, hr_frame, lr_tensor=None):
        """Feed the auto tuner once per rendered frame: the gate's LR delta is free (it was
        just computed) and the HR delta costs one 1/8 luma pool of the output."""
        import torch.nn.functional as F        # no clone: the frame already owns the tensor and this runs before it is handed on
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
        if not hasattr(self, "_accept_pixel"):
            self._accept_pixel = []
            self._accept_emitted = []
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
            coarse = str(getattr(self, "temporalReuseStatistic", "")) == "coarse"
            # At the coarse statistic the acceptance statistic is taken at the SAME scale as the decision.
            # Otherwise the certifier counts the scan's registration and the dust dots as damage - the very
            # difference the decision was built to ignore - and no threshold can ever pass: measured on the
            # denoised Pink Panther, a reuse the gate accepted at the coarse scale reads 1.5% of pixels at the
            # pixel scale of which 1.1% is dust dots, against a 0.1% bar.
            sigma = (getattr(self.temporalReuse, "coarse_sigma", 3.0) * float(self.modelScale or 1)) / 2.0
            # What a reuse at distance d actually EMITS, not what the stale frame costs: the cached output
            # moved by the translation the gate measured for that distance, plus the LR residual that the
            # corrector adds (upscaled the way it does it). Both halves matter and they are separable:
            # measured on the denoised Pink Panther against the fresh model output at the coarse scale, the
            # stale frame reads 0.343%, the moved one 0.181%, and moved+residual 0.000% (5 consecutive pairs
            # of a real render pair). Without this the certifier prices the verbatim fallback and refuses
            # material the emission handles - 0.28-0.40% against a 0.1% bar at depth 1.
            shifts = getattr(self, "_hr_shift_steps", None)
            if shifts is None:
                shifts = self._hr_shift_steps = []
            shift = tuple(getattr(self.temporalReuse, "last_shift_step", (0.0, 0.0)) or (0.0, 0.0))
            shifts.append((float(shift[0]), float(shift[1])))
            if len(shifts) > max(1, depth) + 1:
                del shifts[:-(max(1, depth) + 1)]
            lr_window = self._lr_history(lr_tensor, depth)
            # LR pixels -> the stride-2 acceptance grid of the HR output
            to_grid = float(self.modelScale or 1) / 2.0
            for distance, past in enumerate(reversed(frames[-max(depth, 1):]), start=1):
                pixel_delta = (rgb8.to(torch.int16) - past.to(torch.int16)).abs().max(dim=1).values
                pixel_share = float((pixel_delta > 8).float().mean()) * 100.0
                self._accept_pixel.append(pixel_share)
                del self._accept_pixel[:-512]
                if coarse:
                    accumulated = (sum(s[0] for s in shifts[-distance:]),
                                   sum(s[1] for s in shifts[-distance:]))
                    moved = self._accept_shift(past, accumulated[0] * to_grid, accumulated[1] * to_grid)
                    base_lr = lr_window[distance] if len(lr_window) > distance else None
                    if base_lr is not None:
                        aligned = self._accept_shift(base_lr, accumulated[0], accumulated[1])
                        residual = (lr_tensor - aligned).float()
                        if abs(to_grid - 1.0) > 1e-6:
                            residual = torch.nn.functional.interpolate(
                                residual, scale_factor=to_grid, mode="nearest")
                        moved = moved + residual * 255.0
                    delta = (self._accept_lowpass(rgb8, sigma).to(torch.int16)
                             - self._accept_lowpass(moved.clamp(0, 255), sigma).to(torch.int16)).abs().max(dim=1).values
                else:
                    delta = pixel_delta
                accept.append(float((delta > 8).float().mean()) * 100.0)
                if coarse and moved is not None:
                    # the number the OLD bar would judge, on the emission rather than on the stale frame:
                    # dust dots plus whatever registration the warp left. Reported, never certified.
                    emitted = float(((moved - rgb8).abs().max(dim=1).values > 8).float().mean()) * 100.0
                    self._accept_emitted.append(emitted)
                    del self._accept_emitted[:-512]
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
            # Row (c) of rve-reuse-structural-gate: the pixel-scale number is reported as what it is - dust
            # dots and the scan's registration - beside the coarse one that decided. It is not a bar.
            if self._accept_pixel and str(getattr(self, "temporalReuseStatistic", "")) == "coarse":
                stale = sorted(self._accept_pixel)
                total = len(stale)
                emitted = sorted(getattr(self, "_accept_emitted", []))
                tail = ""
                if emitted:
                    tail = (f"; the EMISSION itself reads {emitted[len(emitted) // 2]:.3f}% median, "
                            f"{emitted[-1]:.3f}% worst")
                log(f"auto gate: at the PIXEL scale the stale frame reads {stale[total // 2]:.3f}% of pixels "
                    f"above 8 levels (worst {stale[-1]:.3f}% over {total} samples){tail} - dust dots and the "
                    f"scan's registration, which the coarse certification is built to ignore")

    def _reuse_frame(self, lr_tensor):
        """The frame a reuse emits: the last rendered frame plus the correction for what changed.

        Falls back to the verbatim reference - exactly the old behaviour - whenever the corrector has no pair to
        work from (first frame, seek, a shape change) or raises. A correction must never be able to break a render.
        """
        cached = self._lastRenderedFrame
        if self.temporalCorrection is None or lr_tensor is None:
            return cached
        # the gate's geometry, applied before the photometric residual: the cached picture is moved onto this
        # frame first, so the residual is not asked to undo a sub-pixel wobble it can only approximate (and
        # so the drawing does not stay where the previous frame put it).
        shift = getattr(self.temporalReuse, "last_shift_step", (0.0, 0.0))
        if self.temporalReuseWarp and shift and (abs(shift[0]) + abs(shift[1]) > 0.0):
            try:
                self.temporalCorrection.align(float(shift[0]), float(shift[1]))
            except Exception as exc:      # geometry must never break a render either
                log(f"reuse alignment failed ({exc}); this reuse keeps the previous geometry")
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
                                            "conv", "pack", "read", "frame")}
        named = sum(prof.get(k, 0.0) for k in ("gate", "up_wall", "read", "pack")) / n
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
                "temporal_reuse_warp": bool(getattr(self, "temporalReuseWarp", False)),
                # the configured value is not what ran: the tuner overwrites the gate's threshold
                "temporal_reuse_threshold_effective": getattr(gate, "threshold", None) if gate else None,
                "int8_fit": getattr(self, "int8FitPreset", None),
            },
            "phases_ms_per_frame": phases,
            "gate": None if gate is None else {
                "processed": gate.processed, "reused": gate.reused,
                "reuse_pct": round(100.0 * gate.reused / gate.processed, 2) if gate.processed else None,
                "depth_histogram": dict(sorted(gate.depth_counts.items())),
                "vetoes": gate.vetoes, "scene_cuts": gate.scene_cuts,
                "registration_refusals": getattr(gate, "registration_refusals", None),
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

    def _stream_media_time(self, frames_rendered: int) -> float | None:
        """The media time of the frame the loop is drawing, derived from the export's own counter.

        `frames_rendered` counts OUTPUT frames (it advances by the interpolation factor per input frame), so the input
        media time is `frames_rendered / (fps * interpolateFactor)`. None when the numbers are unusable - a wrong time
        is worse than a missing one, because the pane compares this against a seek request.
        """
        try:
            fps = float(self.fps)
            factor = float(getattr(self, "interpolateFactor", 1) or 1)
        except (TypeError, ValueError):
            return None
        if fps <= 0:
            return None
        return max(0.0, float(frames_rendered) / (fps * factor))

    def _write_status(self, frames_rendered, phase: int | None = None) -> None:
        """Write the render-status record the pane reads (`utils/DisplayChannel.StatusChannel`).

        Called once per rendered frame and on every phase change. The ETA/FPS here are the SAME arithmetic
        the log line uses (`InformationWriteOut.remaining_seconds` / the same elapsed division), so the pane
        and the log cannot disagree about how much is left. A detached channel is a no-op, and a failure in
        the slot must never take a render down - the log line is already written by its own path.
        """
        channel = getattr(self, "statusChannel", None)
        if channel is None:
            return
        info = getattr(self, "informationHandler", None)
        frames_total = getattr(info, "totalOutputFrames", None)
        if frames_total is None:
            frames_total = getattr(self, "totalOutputFrames", 0)
        eta = ETA_UNSET
        fps = 0.0
        if info is not None:
            eta = info.remaining_seconds(frames_rendered)
            if frames_rendered > 0:
                elapsed = time() - info.startTime
                if elapsed > 0:
                    fps = float(round(frames_rendered / elapsed))
        changes = {
            "frames_rendered": int(frames_rendered),
            "frames_total": int(frames_total or 0),
            "eta_seconds": float(eta),
            "fps": fps,
        }
        if phase is not None:
            changes["phase"] = int(phase)
        try:
            channel.set(**changes)
        except Exception:
            pass

    def _publish_display(self, frame, force: bool = False, time_s: float | None = None,
                         seek_generation: int | None = None) -> bool:
        """Offer this composited frame to the pane; returns whether a publish actually happened.

        Bounded on purpose: at most one staged frame is in flight, and a normal render publishes at most
        every `RVE_DISPLAY_EVERY`-th rendered frame (default 1). `force` is the seek path - a requested
        frame is what the pane asked for, so it goes out even between the N-th frames.
        """
        channel, stager = self.displayChannel, self.displayStager
        if channel is None or stager is None or not channel.attached:
            return False
        if not channel.wanted():
            # Nobody is watching: this is the whole cost of the channel when it is off.
            return False
        if not force:
            # A seek's picture must reach the pane: the pane polls at ~10 Hz while the renderer keeps publishing, so
            # the NEXT streaming frame overwrote the seek's stamped picture before the poll - measured 2026-11-24
            # (four backend "displayed the frame" lines, not one landed time in the pane's log).
            if getattr(self, "_displayHoldUntil", 0.0) > _time.monotonic():
                return False
            self._displayCounter += 1
            if self._displayCounter % self.displayEvery:
                return False
        if force:
            # A requested frame is shown NOW, not one frame late: the pane is waiting on exactly this
            # picture, and there may be no further publish for seconds. Drain the in-flight frame (it is
            # superseded), stage the seek frame and wait for its copy - one D2H, on the seek path only.
            stager.take(force=True)
            if not stager.submit(frame, getattr(frame, "width", None), getattr(frame, "height", None),
                                 time_s=time_s):
                return False
            previous = stager.take(force=True)
            if previous is None:
                return False
            payload, fmt, width, height, stamp = previous
            # The generation this picture answers rides in the header: only the seek path has one, and the pane closes
            # the request on that identity, not on "the first picture with a media time".
            return channel.publish(payload, width, height, fmt=fmt, time_s=stamp,
                                   seek_generation=seek_generation)
        published = False
        previous = stager.take()
        if previous is not None:
            payload, fmt, width, height, stamp = previous
            published = channel.publish(payload, width, height, fmt=fmt, time_s=stamp)
        if not stager.busy():
            stager.submit(frame, getattr(frame, "width", None), getattr(frame, "height", None), time_s=time_s)
        return published

    def _take_seek_request(self):
        """The newest requested time when the generation moved since the last one acted on, else None."""
        state = self.seekState
        if state is None:
            return None
        try:
            return state.pending()
        except Exception as exc:
            log(f"seek: request read failed ({exc}); ignoring seeks for this render")
            self.seekState = None
            return None

    def _seek_reader(self, seconds: float):
        """The resident preview decoder, built on the first seek and reused for the rest of the render.

        A seek cannot reuse the export's reader: its ffmpeg is already streaming the sequential range
        down a pipe, and restarting it would either drop frames the writer is still waiting for or put
        the output out of order. So the seek path owns a decoder of its own - but ONE, not one per
        request: the pane's request block is newest-wins, and a scrub used to be served by a chain of
        ffmpeg spawns (start-up + demux seek + decoder init each). `utils/SeekReader.py` walks the
        resident decoder forward instead and re-opens it only for a request behind it or further ahead
        than a walk is worth. The options are the export's own, so a seek frame is the export's frame.
        """
        if getattr(self, "_seekReader", None) is None:
            from .utils.SeekReader import SeekReader

            reader = self.readBuffer
            self._seekReader = SeekReader(
                inputFile=self.inputFile,
                width=reader.width,
                height=reader.height,
                start_time=float(seconds),
                end_time=None,
                borderX=reader.borderX,
                borderY=reader.borderY,
                hdr_mode=reader.hdr_mode,
                high_bitdepth=reader.high_bitdepth,
                backend=reader.backend,
                device=reader.device,
                gpu_id=reader.gpu_id,
                dtype=reader.dtype,
                color_space=reader.color_space,
                color_primaries=reader.color_primaries,
                color_transfer=reader.color_transfer,
                color_range=reader.color_range,
                input_pixel_format=reader.input_pixel_format,
                ffmpeg_path=reader.ffmpeg_path,
            )
        return self._seekReader

    def _seek_preview(self, seconds: float, generation: int | None = None) -> None:
        """Decode one frame at `seconds`, run it through the model path and show it - nothing else.

        Not written to the writer, not counted in `frames_rendered`, no gate/tuner state touched: the
        export's stream must stay sequential and valid, so this is a side branch that reads the source
        again and publishes only to the pane.
        """
        up = getattr(self, "upscaleOption", None)
        channel = self.displayChannel
        if channel is None or not channel.attached:
            # Nothing to show it on: do not spend an ffmpeg spawn on a pane that is not wired.
            return
        pipelined = (getattr(up, "num_streams", 1) or 1) > 1 and getattr(up, "tile_upscaler", None) is None \
            and getattr(up, "backend", self.backend) == "tensorrt"
        if pipelined:
            # `_call_pipelined` keeps N-1 frames inside the engine and emits them a call later; pushing a
            # seek frame through it would shift that queue and swap one exported frame for the seek's.
            if not getattr(self, "_seekPipelinedWarned", False):
                self._seekPipelinedWarned = True
                log("seek: preview disabled - this upscaler emits frames N-1 late (tensorrt, "
                    "trt_streams > 1), and a seek frame through it would corrupt the export")
            return
        _t_pick = _perf_counter()
        # Wall clock at the moment the loop ACTED on the request. `monotonic` cannot cross into the pane's process;
        # `time.time()` can (one host), and the pane stamps its press with the same clock - the difference is the
        # queue/renderer-loop latency that no other number in either log covers.
        _wall_pick = _time.time()
        _t_reader = _perf_counter()
        try:
            # Split on purpose: `_seek_reader` may BUILD a decoder (a spawn) and that is a different cost from
            # decoding a frame with one that already exists. Folded together, a cold seek looked like a slow decode.
            reader = self._seek_reader(seconds)
            _t_decoding = _perf_counter()
            chunk = reader.frame_at(seconds)
        except Exception as exc:
            log(f"seek: could not decode a frame at {seconds:.3f}s ({exc})")
            return
        _t_decoded = _perf_counter()
        if chunk is None:
            return
        try:
            from .utils.Frame import Frame

            source = self.readBuffer
            frame = Frame(source.backend, source.width, source.height, source.device, source.gpu_id,
                          source.hdr_mode, source.dtype, high_bitdepth=source.high_bitdepth)
            frame.yuv10_input = bool(source.high_bitdepth and not source.hdr_mode)
            frame.set_frame_bytes(chunk)
            _t_bytes = _perf_counter()
            # The MODEL is the cost people mean by "the seek is slow", and it used to be buried inside `prepare=`
            # together with the byte copy and the resize.
            for extraRestoration in self.extraRestorationModels:
                frame = extraRestoration(frame)
            if self.upscaleModel:
                frame = self.upscaleOption(frame)
            if frame is None:
                return
            # The seek path KNOWS the media time of this picture - it decoded that exact frame - so this is the one
            # publish whose content can be checked against the request. The streaming publishes hold off briefly so
            # this picture is the one the pane's next poll reads.
            #
            # The generation goes into the header too: the pane can then close a seek on the picture that ANSWERS it
            # instead of on the first picture that happens to carry a media time - the streaming path names every
            # frame it draws, so "named" stopped meaning "asked for" (measured 2026-11-24: a press at 1.888 s was
            # closed by the streaming frame at 4.046 s, +2.158 s off).
            _t_prepared = _perf_counter()
            published = self._publish_display(frame, force=True, time_s=float(seconds),
                                              seek_generation=generation)
            _t_published = _perf_counter()
            if published:
                self._displayHoldUntil = _time.monotonic() + DISPLAY_SEEK_HOLD_S
            self.lastSeekSeconds = float(seconds)
            # The stages, so a turnaround can be read instead of guessed: decode is whatever the resident reader paid,
            # prepare is restoration/upscale, publish is the channel (staging copy + header). The pane's own poll is
            # the fourth stage and only the app side can see it.
            log(f"seek: displayed the frame at {seconds:.3f}s (generation {generation}, not written to the export); "
                f"stages "
                f"reader={(_t_decoding - _t_reader) * 1000.0:.1f}ms "
                f"decode={(_t_decoded - _t_decoding) * 1000.0:.1f}ms "
                f"bytes={(_t_bytes - _t_decoded) * 1000.0:.1f}ms "
                f"model={(_t_prepared - _t_bytes) * 1000.0:.1f}ms "
                f"publish={(_t_published - _t_prepared) * 1000.0:.1f}ms "
                f"backend_total={(_t_published - _t_pick) * 1000.0:.1f}ms "
                f"pickup_wall={_wall_pick:.3f}")
        except Exception as exc:
            log(f"seek: frame at {seconds:.3f}s could not be displayed ({exc})")

    def render(self):
        frames_rendered = 0
        # The engines are built and the reader/writer are up: this thread only moves frames now.
        self._write_status(0, phase=PHASE_RENDERING)
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
                # A seek is acted on between frames, before the next read blocks: the request is a single
                # slot, so a newer one simply replaces this one - the loop never waits for the pane.
                requested = self._take_seek_request()
                if requested is not None:
                    self._seek_preview(requested, getattr(self.seekState, "last_generation", None))
                    continue
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
                    # The reader is drained; the writer may still be flushing. Say so instead of leaving
                    # "Rendering" on the pane until the process exits.
                    self._write_status(frames_rendered, phase=PHASE_FINALISING)
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
                            interpolated_frame = resize_frame_bytes(
                                interpolated_frame,
                                target_width=self.width * self.override_upscale_scale,
                                target_height=self.height * self.override_upscale_scale,
                                path="RenderVideo.render:interpolated-frame override_upscale_scale",
                            )
                        self.informationHandler.setFramesRendered(frames_rendered)
                        self.writeBuffer.writeQueue.put(interpolated_frame.get_frame_bytes() if type(interpolated_frame) != bytes else interpolated_frame)
                
                

                _t_frame = _perf_counter()
                if not hasattr(self, "_profile"):
                    self._profile = {"frames": 0, "gate": 0.0, "infer": 0.0, "conv": 0.0,
                                     "frame": 0.0, "to_tensor": 0.0,
                                     # the three stages the old line did NOT name, which is why the
                                     # frame total sat above its own parts by 6.2 ms on 2026-11-18
                                     # (rve-frame-accounting-2026-11-19)
                                     "up_wall": 0.0, "pack": 0.0}
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
                            try:
                                self._calibrate_gate(frame, frame_tensor)
                            except Exception as _calib_exc:
                                # a calibration failure costs the tuning, never the render: the thread died here
                                # once and the writer waited for frames that would never come
                                log(f"auto gate: calibration failed ({_calib_exc}); keeping the configured "
                                    f"threshold for the rest of this render")
                                self.autoTuner = None
                
                
                
                # The composited frame, offered to the pane BEFORE the override resize: that call returns
                # raw pipe bytes, and the display channel speaks RGB888 only. The media time is DERIVED from the
                # export's own frame counter (`frames_rendered` counts output frames, so the interpolation factor
                # divides it out): an unnamed picture cannot be checked against a seek, and leaving it unnamed hid
                # which frame the render actually drew (owner, 2026-11-24).
                self._publish_display(frame, time_s=self._stream_media_time(frames_rendered))

                if self.override_upscale_scale:
                    # Source geometry comes from the frame the upscaler returned, never from
                    # `width * modelScale`: on `4xNomos8k_span_otf_weak_no_update_params.pth` the
                    # frame came back at 2x (320x240) while the descriptor claimed 4x (640x480), and
                    # the old call fed that payload to a 640x480 reshape - an opaque reshape error,
                    # then a dead render thread (2026-11-22). A frame whose payload disagrees with
                    # its own geometry now raises a named ValueError instead.
                    #
                    # The SDR pipes emit packed planar YUV (`yuv8_output` / `yuv10_output`), so the
                    # resize is plane-wise there (`resize_yuv420_bytes`): Y to the override size and
                    # each chroma plane to its half, in the packing the writer's `-pix_fmt` names.
                    # `RVE_OVERRIDE_YUV=0` restores the old refusal for an A/B on the merge.
                    frame = resize_frame_bytes(
                        frame,
                        target_width=self.width * self.override_upscale_scale,
                        target_height=self.height * self.override_upscale_scale,
                        path="RenderVideo.render:override_upscale_scale",
                    )

                
                if type(frame) is not bytes and getattr(self, "pooled_output", False):
                    frame.pooled_output = True
                self.informationHandler.setFramesRendered(frames_rendered)
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
                # Once per rendered frame: the counters, ETA and FPS the pane shows, beside the log line
                # the information thread keeps printing (the slot never replaces it).
                self._write_status(frames_rendered)
                if not hasattr(self, "_t_first_frame"):
                    self._t_first_frame = _time.time()
                    if self._phases.first("first-frame"):
                        log(f"startup: first frame enqueued {self._t_first_frame - _T0:.2f} s "
                            f"after process start (interpreter, engine, ffmpeg/NVENC init)")
                if (self.temporalReuse is not None and hasattr(self, "_profile")
                        and self._profile["frames"] % 250 == 0):
                    # visible during the render, not only in the summary: a cancelled encode used to say nothing.
                    # Bounded to one line per second of wall clock - the old every-250-frames cadence printed
                    # ~4 lines/s on a fast clip and nothing for a minute on a slow one.
                    if self._progress.allow():
                        log("reuse: " + self.temporalReuse.progress())
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
        self._write_status(frames_rendered, phase=PHASE_DONE)
        first = getattr(self, "_t_first_frame", None)
        if first is not None and self._phases.first("render-loop"):
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
                     + prof.get("pack", 0.0))
            log(f"profile over {n} frames: gate {prof['gate']/n:.2f} ms "
                f"(input to tensor {prof.get('to_tensor', 0.0)/n:.2f}), "
                f"model wall {prof.get('up_wall', 0.0)/n:.2f} ms "
                f"(self-reported {prof['infer']/n:.2f}, engine {prof.get('model', 0.0)/n:.2f}, "
                f"frame handover {prof.get('handover', 0.0)/n:.2f}), "
                f"input convert {prof['conv']/n:.2f} ms, "
                f"pack+queue {prof.get('pack', 0.0)/n:.2f} ms, "
                f"read wait {prof.get('read', 0.0)/n:.2f} ms, "
                f"named {(named/n):.2f} of frame total {prof['frame']/n:.2f} ms, "
                f"unaccounted {(prof['frame'] - named)/n:.2f} ms", level=LEVEL_DEBUG)
        if self.upscaleModel and getattr(self, "tileReuse", False):
            up = getattr(self, "upscaleOption", None)
            tu = getattr(up, "tile_upscaler", None)
            if tu is not None:
                log(tu.stats())
        self._write_run_report()
        self.writeBuffer.writeQueue.put(None)
        # Detach the GUI-owned blocks: closing never unlinks a segment we did not create. A block the
        # backend created as a fallback (the window had not attached yet) is removed by close().
        for _block in (self.displayChannel, self.seekState, getattr(self, "statusChannel", None)):
            if _block is not None:
                try:
                    _block.close()
                except Exception:
                    pass
        # The preview decoder is this render's own ffmpeg, not the export's: it is closed here, and its
        # line says what the seek path actually cost (one spawn vs one per request).
        _seek_reader = getattr(self, "_seekReader", None)
        if _seek_reader is not None:
            try:
                log(_seek_reader.stats())
                _seek_reader.close()
            except Exception:
                pass
            self._seekReader = None
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
            if hasattr(self.upscaleOption, "yuv10_matrix"):
                self.upscaleOption.yuv10_matrix = getattr(self, "yuv10Matrix", "bt709")
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
            log(f"debug: after assign option={id(self.upscaleOption)} int8Fit={getattr(self.upscaleOption,'int8Fit','<unset>')}",
                level=LEVEL_DEBUG)
            

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