import queue
import sys
from abc import ABC, abstractmethod
import os
import subprocess
import queue
import time
# Guarded like `utils/Util.py` does it: these two are used only by the reader's yuv420p path, and the argument
# builder must stay importable where they are absent (the harness unit tests, a verification cell).
try:
    import cv2
    import numpy as np
except ImportError:
    pass

from .utils.Util import (
    log,
    subprocess_popen_without_terminal,
)
from .utils.Encoders import  EncoderSettings
from .utils.Frame import Frame
from .utils.Segment import audio_trim_filter, segment_args

class Buffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        pass


class FFmpegRead(Buffer):
    def __init__(
            self, 
            inputFile, 
            width, 
            height, 
            start_time, 
            end_time, 
            borderX, 
            borderY, 
            hdr_mode,
            high_bitdepth: bool | None = None,
            backend: str = "pytorch",
            device: str = "cuda",
            gpu_id: int = 0,
            dtype: str = "float16",
            color_space=None, 
            color_primaries=None, 
            color_transfer=None, 
            color_range=None,
            input_pixel_format: str | None = None,
            ffmpeg_path: str = "./bin/ffmpeg",
        ):
        
        self.inputFile = inputFile
        self.width = width
        self.height = height
        self.start_time = start_time
        self.end_time = end_time
        self.borderX = borderX
        self.borderY = borderY
        self.hdr_mode = hdr_mode
        # "high_bitdepth" is the DATA path (16-bit frames/pipe); "hdr_mode" is the COLOUR
        # tagging (bt2020/PQ). A 10-bit SDR source needs the former without the latter,
        # otherwise its precision is thrown away by an 8-bit rgb24 round trip (measured:
        # the conversion caps at ~44 dB vs ~79.5 dB for the 16-bit path).
        self.high_bitdepth = hdr_mode if high_bitdepth is None else high_bitdepth
        self.backend = backend
        self.device = device
        self.gpu_id = gpu_id
        self.dtype = dtype
        self.color_space = color_space
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.color_range = color_range
        self.input_pixel_format = input_pixel_format
        self.yuv420pMOD = self.input_pixel_format == "yuv420p" and not self.high_bitdepth
        self.ffmpeg_path = ffmpeg_path
        #self.yuv420pMOD = False
        if self.high_bitdepth and not self.hdr_mode:
            # 3 bytes/pixel: the GPU does YUV->RGB itself (see TorchUtils.yuv420p10_to_tensor)
            self.inputFrameChunkSize = width * height * 3
        elif self.high_bitdepth:
            self.inputFrameChunkSize = width * height * 6
        else:
            if self.yuv420pMOD:
                self.inputFrameChunkSize = width * height * 3 // 2
            else:
                self.inputFrameChunkSize = width * height * 3
        command = self.command()
        log("FFMPEG READ COMMAND: " + str(command))
        self.readProcess = subprocess_popen_without_terminal(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.readQueue = queue.Queue(maxsize=25)

    def command(self):

        command = [f"{self.ffmpeg_path}"]
        # the segment bound, before the input so the seek does not decode what precedes it (see segment_args)
        command += segment_args(self.start_time, self.end_time)
        command += [
            "-i",
            f"{self.inputFile}",
        ]

        filter_string = f"crop={self.width}:{self.height}:{self.borderX}:{self.borderY},scale=w=iw*sar:h=ih"
        if self.high_bitdepth and not self.hdr_mode:
            # YUV in, YUV out: only the SAR fix applies, no colour matrix involved
            pass
        elif not self.hdr_mode:
            # swscale defaults to bt601 when nothing says otherwise, which shifts chroma on
            # a bt709 source (measured |dU| ~6.6 levels without these two options).
            if self.color_space and not self.yuv420pMOD:
                filter_string += f":in_color_matrix={self.color_space}"
            # RGB output is full range; a YUV stream must stay in its own range.
            filter_string += (
                f":in_range={self.color_range or 'tv'}"
                f":out_range={'tv' if self.yuv420pMOD else 'pc'}"
            ) #+ ":in_range=limited:out_range=full,format=yuv420p" if self.yuv420pMOD == "yuv420p" else "" # fix dar != sar
        #if not self.hdr_mode:
        #    if self.input_pixel_format == "yuv420p":
        #        filter_string += ":in_range=tv:out_range=pc" # color shifts a smidgen but helps with artifacts when converting yuv to raw
        command += [
            "-vf",
            filter_string,
            "-f",
            "image2pipe",
            "-pix_fmt",
            ("yuv420p10le" if not self.hdr_mode else "rgb48le") if self.high_bitdepth
            else (self.input_pixel_format if self.yuv420pMOD else "rgb24"),
            #"rgb48le" if self.hdr_mode else "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-"
            
        ]

        log("FFMPEG READ COMMAND: " + str(command))
        return command

    def read_frame(self):
        chunk = self.readProcess.stdout.read(self.inputFrameChunkSize)
        if len(chunk) < self.inputFrameChunkSize:
            return None

        if self.yuv420pMOD:
            # Convert raw YUV420p data to RGB
            # The data is Y plane, then U plane, then V plane, concatenated.
            # cv2.COLOR_YUV420P2RGB expects a single channel image of shape (height * 3 // 2, width)
            np_frame = np.frombuffer(chunk, dtype=np.uint8)
            # Ensure height is an integer for reshape, Python 3 // operator already does this.
            yuv_image_height = self.height * 3 // 2
            yuv_image = np_frame.reshape((yuv_image_height, self.width))
            rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_I420)
            # cv2.imwrite("temp_rgb_image.png", rgb_image)  # Debugging line, can be removed
            chunk = rgb_image.tobytes()
        
        return chunk

    def read_frames_into_queue(self):
        while True:
            chunk = self.read_frame()
            if chunk is None:
                break
            frame = Frame(self.backend, self.width, self.height, self.device, self.gpu_id, self.hdr_mode, self.dtype,
                          high_bitdepth=self.high_bitdepth)
            frame.yuv10_input = bool(self.high_bitdepth and not self.hdr_mode)
            frame.set_frame_bytes(chunk)
            self.readQueue.put(frame)
        self.readQueue.put(None)

    def get(self) -> Frame:
        return self.readQueue.get()

    def close(self):
        self.readProcess.stdout.close()
        self.readProcess.terminate()


#: Containers whose index (`moov`) is written last: with `+faststart` ffmpeg relocates it at close.
_MP4_FAMILY = ("mp4", "mov", "m4v")


def resolve_movflags(output_file: str | None, extension: str | None, preview: bool) -> list[str]:
    """The `-movflags` value for this output, or `[]` when the container takes none.

    An export keeps `+faststart`: a finished file with its index at the front, seekable immediately.
    A *preview* render instead writes fragmented mp4 (`+frag_keyframe+empty_moov` - one fragment per
    keyframe, an empty index up front), because `+faststart` moves the index to the end **at close**:
    until ffmpeg exits the growing file has no index, so a reader following it cannot demux it at all.
    That reader is the preview surface's tail reader (plan-rve-video-preview-widget-2026-11-23).
    Matroska/TS (and everything else) are readable as they grow and stay untouched.
    """
    if not output_file:
        return []
    if str(extension or "").lower() not in _MP4_FAMILY:
        return []
    return ["+frag_keyframe+empty_moov"] if preview else ["+faststart"]


def resolve_output_pixel_format(requested: str | None, high_bitdepth: bool,
                                interpolating: bool = False) -> str:
    """Output pixel format when the caller did not choose one.

    A 10-bit source keeps its depth. That used to be a speed trade ("10-bit costs 2x NVENC"),
    and it was not: the encoder costs 1.47x (9.4 against 6.4 ms/frame at 2160p), and the 2x
    was swscale, inserted because the colour tags were declared on the *output* side - see
    `FFmpegWrite._color_tags`. With the tags on the input side the same 94-frame 2160p render
    takes 4.67 s at 10-bit against 4.64 s at 8-bit, so the ~6 dB are free.
    An explicit request always wins, and interpolated frames keep 8-bit because the
    interpolator works in 8 bits.
    """
    if requested:
        return requested
    return "yuv420p10le" if (high_bitdepth and not interpolating) else "yuv420p"


class FFmpegWrite(Buffer):
    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        width: int,
        height: int,
        start_time: float,
        end_time: float,
        fps: float,
        crf: str,
        audio_bitrate: str,
        pixelFormat: str,
        overwrite: bool,
        custom_encoder: str,
        benchmark: bool,
        slowmo_mode: bool,
        upscaleTimes: int,
        interpolateFactor:int,
        ceilInterpolateFactor: int,
        video_encoder: EncoderSettings,
        audio_encoder: EncoderSettings,
        subtitle_encoder: EncoderSettings,
        hdr_mode: bool,
        mpv_output: bool,
        merge_subtitles: bool,
        high_bitdepth: bool | None = None,
        pipe_input_pixel_format: str | None = None,
        color_space: str = None,
        color_primaries: str = None,
        color_transfer: str = None,
        color_range: str = None,
        ffmpeg_path: str = "./bin/ffmpeg",
        ffmpeg_log_file: str = "ffmpeg_log.txt",
        preview: bool = False,
    ):
        self.inputFile = inputFile
        self.outputFile = outputFile
        if self.outputFile:
            self.outputFileExtension = os.path.split(self.outputFile)[-1].split(".")[-1]
        self.width = width
        self.height = height
        self.start_time = start_time
        self.end_time = end_time
        self.outputWidth = width * upscaleTimes
        self.outputHeight = height * upscaleTimes
        self.fps = fps
        self.crf = crf
        self.audio_bitrate = audio_bitrate
        self.pixelFormat = pixelFormat
        self.overwrite = overwrite
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.slowmo_mode = slowmo_mode
        self.upscaleTimes = upscaleTimes
        self.interpolateFactor = interpolateFactor
        self.ceilInterpolateFactor = ceilInterpolateFactor
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.subtitle_encoder = subtitle_encoder
        self.mpv_output = mpv_output
        self.hdr_mode = hdr_mode
        self.high_bitdepth = hdr_mode if high_bitdepth is None else high_bitdepth
        self.pipe_input_pixel_format = pipe_input_pixel_format
        self.merge_subtitles = merge_subtitles
        self.writeQueue = queue.Queue(maxsize=25)
        # Where the frame time actually goes on the writer side. `write_ms` is the part where
        # the pipe blocks us (ffmpeg draining); `wait_ms` is the writer idling because the
        # render loop has not produced anything. Comparing their sums against the render
        # loop's own total answers "is the encoder serialized behind the model?" without
        # guessing from a wall-clock difference that also contains startup and flush.
        self.write_stats = {"frames": 0, "bytes": 0, "write_ms": 0.0, "wait_ms": 0.0,
                            "max_write_ms": 0.0, "queue_max": 0}
        self.framesRendered: int = 1
        self.writeProcess = None
        self.color_space = color_space
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.color_range = color_range
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_log_file = ffmpeg_log_file
        # A preview render must be readable while it is still being written (fragmented mp4); an export must
        # not be (it keeps `+faststart`). Explicit, never inferred from the output path or its name.
        self.preview = preview
        self.outputFPS = (
            (self.fps * self.interpolateFactor)
            if not self.slowmo_mode
            else self.fps
        )
        self.ffmpeg_log = open(self.ffmpeg_log_file, "w", encoding='utf-8')
        try:
            command = self.command()
            log("\nFFMPEG WRITE COMMAND: " + str(command) + "\n")
            self.writeProcess = subprocess_popen_without_terminal(
                command,
                stdin=subprocess.PIPE,
                stderr=self.ffmpeg_log,
                stdout=subprocess.PIPE if self.mpv_output else self.ffmpeg_log,
                text=True,
                universal_newlines=True,
            )
        except Exception as e:
            self.onErroredExit()



    def _pipe_input_pixel_format(self) -> str:
        """What the raw pipe carries from the renderer to ffmpeg.

        This MUST be what the renderer packed, not what the encoder would prefer. It used to
        be derived from the encoder tag ("NVENC wants P010"), while the packer emits planar
        `yuv420p10le` unless `RVE_P010` is set - so every 10-bit source rendered with NVENC fed
        ffmpeg planar samples labelled P010 and came out at 12.7 dB against the reference.
        A format disagreement here cannot fail loudly: both are 16-bit words at 3 bytes/pixel.
        """
        if self.high_bitdepth and not self.hdr_mode:
            return self.pipe_input_pixel_format or "yuv420p10le"
        if not self.high_bitdepth and self.pipe_input_pixel_format == "yuv420p":
            # 8-bit planar pipe (RVE_YUV8): what the renderer packed, 1.5 bytes/pixel
            return "yuv420p"
        return "rgb48le" if self.high_bitdepth else "rgb24"

    def _color_tags(self) -> list[str]:
        """The colour description of the raw pipe, for the INPUT side of ffmpeg.

        These are input options, not output options, and that is not cosmetic: `-colorspace`
        placed after the input tells ffmpeg the *output* must be that space, so it inserts a
        swscale matrix conversion of every frame - measured 79.0 ms/frame against 16.6 ms/frame
        for the same stream at 2880x2160 10-bit (a 4.8x tax for a metadata declaration, and the
        real explanation of what looked like "10-bit output costs 2x NVENC": NVENC itself is
        9.4 against 6.4 ms/frame). On the input side the same tags produce the same container
        tags and no conversion at all.
        """
        if self.hdr_mode:
            return []
        args: list[str] = []
        if self.color_space:
            args += ["-colorspace", self.color_space]
        if self.color_range:
            args += ["-color_range", self.color_range]
        if self.color_primaries:
            args += ["-color_primaries", self.color_primaries]
        if self.color_transfer:
            args += ["-color_trc", self.color_transfer]
        # The 8-bit YUV pipe carries samples the PACKER made (BT.709, limited), so its tags must
        # describe that even when the source declares nothing. The rgb24 branch gets bt709 stamped
        # for free by `scale=out_color_matrix=bt709`; measured without this, the YUV pipe's output
        # came out `color_space=unknown` - the same "undeclared matrix" class that cost 5 dB in C3.
        if not self.hdr_mode and self._pipe_input_pixel_format() == "yuv420p":
            if not (self.color_space or self.color_primaries):
                args += ["-colorspace", "bt709"]
            if not self.color_range:
                args += ["-color_range", "tv"]
        return args

    def _color_convert_args(self) -> list[str]:
        """RGB pipe -> YUV with an explicit matrix.

        When ffmpeg converts RGB rawvideo to YUV it picks its own default matrix (bt601),
        which shifts chroma on a bt709/bt2020 source - measured |dU| ~6.6 levels on a
        bt709 clip. Passing the source's own matrix/range removes that shift entirely.

        Only the RGB pipe needs a filter here; a YUV pipe is described by `_color_tags()` on the
        input side and must not be touched again. P010 used to fall through to this branch and got
        `scale=in_range=pc:...`: a full-range reinterpretation of a limited-range stream, which is
        what cost that pipe 31 dB against the planar reference (ledger C4).
        """
        if self.hdr_mode or self.custom_encoder is not None:
            return []
        if self._pipe_input_pixel_format() in ("yuv420p10le", "p010le", "yuv420p"):
            return []
        cs = self.color_space or "bt709"
        rng = self.color_range or "tv"
        args = ["-vf", f"scale=in_range=pc:out_color_matrix={cs}:out_range={rng}"]
        return args


    def command(self):
        if self.mpv_output:
            command = [
                f"{self.ffmpeg_path}",
                "-loglevel",
                "error",
                "-framerate",
                f"{self.outputFPS}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                self._pipe_input_pixel_format(),
                *self._color_tags(),
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.outputWidth}x{self.outputHeight}",
                "-i",
                "-",
                "-r",
                f"{self.outputFPS}",
                "-f",
                "matroska",
                "-b:v",
                "15000k",
                "-crf",
                "0",
                *self._color_convert_args(),
                "-af",
                audio_trim_filter(self.start_time, self.end_time),

            ]
            
            
            if self.hdr_mode:

                # override pixel format
                pxfmtdict = {
                    "yuv420p": "yuv420p10le",
                    "yuv422": "yuv422p10le",
                    "yuv444": "yuv444p10le",
                }

                if self.pixelFormat in pxfmtdict:
                    self.pixelFormat = pxfmtdict[self.pixelFormat]
                
                command += [
                    "-pix_fmt",
                    self.pixelFormat,
                ]
                
            command += [
                "-",
            ]
            return command


        if not self.benchmark:
            # maybe i can split this so i can just use ffmpeg normally like with vspipe
            command = [
                f"{self.ffmpeg_path}",
                "-loglevel",
                "error",
            ]

            if self.custom_encoder is None:
                pre_in_set = self.video_encoder.getPreInputSettings()
                if pre_in_set is not None:
                    command += pre_in_set.split()

            command += [
                "-framerate",
                f"{self.outputFPS}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                self._pipe_input_pixel_format(),
                *self._color_tags(),
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.outputWidth}x{self.outputHeight}",
                "-i",
                "-",
                "-r",
                f"{self.outputFPS}",
            ]

            if not self.slowmo_mode:
                command += [
                    # Input 1: original file for audio/subtitles.
                    # Put timestamp hygiene flags *before* the input they apply to.
                    "-fflags",
                    "+genpts",
                ]
                # The same segment bound on the audio input. `-af atrim` cannot do this job on the shipped path:
                # the audio is usually COPIED (`-c:a copy`), and a filter does not apply to a copied stream - a
                # 3-second segment render came out 6:25 long, with the source's remaining audio after the last
                # video frame. Bounding the input works for copy and for re-encode alike.
                command += segment_args(self.start_time, self.end_time)
                command += [
                    "-i",
                    f"{self.inputFile}",
                    "-map",
                    "0:v",  # Map video stream from input 0
                    "-map",
                    "1:a?",
                    "-map",
                    "1:s?",
                ]

                # Output timestamp/interleave hygiene.
                command += [
                    "-avoid_negative_ts",
                    "make_zero",
                    "-max_interleave_delta",
                    "0",
                    "-muxpreload",
                    "0",
                    "-muxdelay",
                    "0",
                ]

                

            if self.custom_encoder is not None:

                for i in self.custom_encoder.split():
                    command.append(i)
            else:
                if not self.audio_encoder.getPresetTag() == "copy_audio":
                    command += [
                        "-b:a",
                        self.audio_bitrate,
                    ]
                command += self.video_encoder.getPostInputSettings().split()
                # NVENC defaults to p4; p1 is ~36% faster at a slightly worse quality/bit
                # (measured 3.46x -> 4.71x realtime at 2880x2160 10-bit). The encoder is on the
                # critical path again once the model and the gate are in place, so this is
                # worth ~8% of the frame. Override with RVE_NVENC_PRESET=p4/p7.
                try:
                    _tag = self.video_encoder.getPresetTag()
                except Exception:
                    _tag = ""
                if _tag in ("x265_nvenc", "x264_nvenc", "av1_nvenc"):
                    command += ["-preset", os.environ.get("RVE_NVENC_PRESET", "p1")]
                command += [self.video_encoder.getQualityControlMode(), str(self.crf)]
                command += self.audio_encoder.getPostInputSettings().split()
                command += self.subtitle_encoder.getPostInputSettings().split()

                if self.hdr_mode:
                    
                    

                    # override pixel format
                    pxfmtdict = {
                        "yuv420p": "yuv420p10le",
                        "yuv422": "yuv422p10le",
                        "yuv444": "yuv444p10le",
                    }

                    if self.pixelFormat in pxfmtdict:
                        self.pixelFormat = pxfmtdict[self.pixelFormat]

                    if self.video_encoder.getPresetTag() == "libx265" or self.video_encoder.getPresetTag() == "x265_nvenc":
                        command += [
                            "-x265-params",
                                "hdr-opt=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
                        ]
                    elif self.video_encoder.getPresetTag() == "prores":
                        command += [
                            "-profile:v",
                            "4",
                            "-vendor",
                            "ap10",
                            "-color_range",
                            "full",
                        ]

                command += self._color_convert_args()
                command += [
                    "-pix_fmt",
                    self.pixelFormat,

                ]

                # MP4/MOV: an export moves the moov atom to the front; a preview render fragments instead so the
                # written part is a readable stream while the file is still growing (see resolve_movflags).
                movflags = resolve_movflags(
                    self.outputFile, getattr(self, "outputFileExtension", None), self.preview)
                if movflags:
                    command += ["-movflags", *movflags]
                # A preview render also gets a SHORT GOP (~one second of output frames). Two measured reasons
                # (projects/rve/docs/measurements/measure-video-preview-2026-11-23.md): with the encoder's default GOP a seek costs
                # ~0.5-1 s of decode (480 ms mean, 932 ms p95 on a 4K clip), and `+frag_keyframe` emits a fragment per
                # keyframe - so a 250-frame GOP would hand the tail reader new data once per ~8 s instead of ~1 Hz.
                # `-g` is AVCodecContext's gop_size: generic for every encoder this app exposes. Exports keep theirs.
                if self.preview:
                    command += ["-g", str(max(1, int(round(float(self.outputFPS or 30)))))]
            if str(self.outputFile) == "null":
                # Fixed-cost measurement path: rawvideo encoding is a memcpy and the null muxer
                # discards the packets, so the wall measures read + gate + inference only. Without
                # it the encoder's cost depends on the content's entropy, which is what made a
                # quantised output look 1.55x faster when the engine was not.
                command += ["-c:v", "rawvideo", "-f", "null", "-"]
            else:
                command +=[
                    f"{self.outputFile}",
                ]

            if self.overwrite:
                command.append("-y")

            if self.slowmo_mode:
                log("Slowmo mode enabled, will not merge audio or subtitles.")

        else: # Benchmark mode

            command = [
                f"{self.ffmpeg_path}",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stats",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-video_size",
                f"{self.width * self.upscaleTimes}x{self.upscaleTimes * self.height}",
                "-pix_fmt",
                self._pipe_input_pixel_format(),
                *self._color_tags(),
                "-r",
                str(self.outputFPS),
                "-i",
                "-",
                "-benchmark",
                "-f",
                "null",
                "-",
            ]

        
        return command

    def get_num_frames_rendered(self):
        return self.framesRendered

    def put_frame_in_write_queue(self, frame):
        
        self.writeQueue.put(frame)

    def write_out_frames(self):
        log("Rendering")
        self.startTime = time.time()
        started_at = time.time()

        exit_code: int = 0
        try:
            while True:
                _t_wait = time.perf_counter()
                frame = self.writeQueue.get()
                _wait_ms = (time.perf_counter() - _t_wait) * 1000.0
                if frame is None:
                    break
                
                # A pooled-output item is the Frame itself (its payload is a live view on a
                # pinned host slot); plain bytes still go through unchanged.
                payload = frame.get_frame_bytes() if hasattr(frame, "get_frame_bytes") else frame
                # a payload copied on the side stream is only readable once its event has fired
                _ev = getattr(frame, "_payload_event", None)
                if _ev is not None:
                    _ev.synchronize()
                _t_write = time.perf_counter()
                self.writeProcess.stdin.buffer.write(payload)
                _write_ms = (time.perf_counter() - _t_write) * 1000.0
                if hasattr(frame, "release_payload"):
                    # the bytes are in the pipe's own buffer now: the slot may be rewritten
                    frame.release_payload()

                st = self.write_stats
                st["frames"] += 1
                st["bytes"] += (getattr(payload, "nbytes", None)
                                or (len(payload) if payload is not None else 0))
                st["write_ms"] += _write_ms
                st["wait_ms"] += _wait_ms
                st["max_write_ms"] = max(st["max_write_ms"], _write_ms)
                try:
                    st["queue_max"] = max(st["queue_max"], self.writeQueue.qsize())
                except Exception:
                    pass

            self.writeProcess.stdin.close()
            self.writeProcess.wait()
            exit_code = self.writeProcess.returncode

            renderTime = time.time() - self.startTime
            log(f"\nTime to complete render: {round(renderTime, 2)}")
            st = self.write_stats
            n = st["frames"] or 1
            log(f"writer: {st['frames']} frames, write {st['write_ms']/n:.2f} ms/frame "
                f"(max {st['max_write_ms']:.1f}), wait for frames "
                f"{st['wait_ms']/n:.2f} ms/frame, {st['bytes']/1e6:.0f} MB total, "
                f"busy {st['write_ms']/1000.0:.2f} s of {time.time()-started_at:.2f} s, "
                f"queue max {st['queue_max']}")
            
        except Exception as e:
            log(str(e))
            self.onErroredExit()

        if exit_code != 0:
            self.onErroredExit()
            return
        
        

    def onErroredExit(self):
        log("FFmpeg failed to render the video.")
        try:
            with open(self.ffmpeg_log_file, "r") as f:
                log("FULL FFMPEG LOG:")
                for line in f.readlines():
                    log(line)

            with open(self.ffmpeg_log_file, "r") as f:
                for line in f.readlines():
                    if f"[{self.outputFileExtension}" in line:
                        log(line)

            if self.video_encoder.getPresetTag() == "x264_vulkan":
                log("Vulkan encode failed, try restarting the render.")
                log(
                    "Make sure you have the latest drivers installed and your GPU supports vulkan encoding."
                )
        except Exception as e:
            print("Failed to read FFmpeg log file.")
            print(str(e))

        time.sleep(1)
        os._exit(1)

    def __del__(self):
        self.ffmpeg_log.close()

class MPVOutput:
    def __init__(self, FFMpegWrite: FFmpegWrite, width, height,fps, outputFrameChunkSize):
        self.proc = None
        self.startTime = time.time()
        self.FFMPegWrite = FFMpegWrite
        self.outputFrameChunkSize = outputFrameChunkSize
        self.width = width
        self.height = height
        self.fps = fps


    def command(self):
        command = [
        "mpv",
        f"--audio-file={self.FFMPegWrite.inputFile}",
        "--no-config",
        "--cache=yes",
        "--cache-secs=5",                    # Cache 30 seconds of video
        "--demuxer-max-bytes=500Mib",         # Increase max bytes
        "--demuxer-readahead-secs=5",        # Read ahead 30 seconds
        "--demuxer-seekable-cache=yes",       # Enable seekable cache
        "--stream-buffer-size=500MiB",        # Increase buffer size
        "--hr-seek-framedrop=no",            # Prevent frame dropping during seeks
        "-"
        ]
        return command

    def write_out_frames(self):
        with open('mpv_log.txt', "w") as f:
            while not self.FFMPegWrite.writeProcess:
                time.sleep(1)
            self.proc = subprocess_popen_without_terminal(
                self.command(),
                stdin=self.FFMPegWrite.writeProcess.stdout,
                stderr=f,
                stdout=f,
            )
            self.FFMPegWrite.writeProcess.stdout.close()
            self.proc.wait()
            self.stop()
            os._exit(0) # force exit

    def stop(self):
        """
        Stop mpv by closing stdin.
        """
        if self.proc:
            self.proc.terminate()
            self.proc.wait()