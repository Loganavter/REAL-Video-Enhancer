"""The render command line is a function of the options - pinned byte for byte against the pre-move method.

P3 of the seam plan moved `ProcessTab.build_command` into `src/ui/render_command.py`. The risk is not that the move
fails loudly; it is that the order of a flag or one of the fifteen settings it reads drifts while the render still
starts. So the two lists below are a capture of the OLD method's output, taken before the first edit, by executing
the source at commit eb966ae1 (`git show HEAD:src/ui/ProcessTab.py`, the `build_command` function body) against fixed
options, fixed paths and a fixed settings mapping. They were not produced by the new module - a golden computed by the
code under test proves nothing.

No Qt and no display: `RenderOptions` is the real dataclass (`src/ui/RenderQueue.py` imports Qt only for typing), the
paths are a `RenderPaths` and the settings are a mapping with the two attributes the builder reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.RenderQueue import RenderOptions
from src.ui.render_command import RenderPaths, build_command


PATHS = RenderPaths(
    python="/opt/rve/python/bin/python",
    backend_path="/opt/rve/backend",
    cwd="/opt/rve",
    ffmpeg_path="/opt/rve/bin/ffmpeg",
    models_path="/opt/rve/models",
    custom_models_path="/opt/rve/models/custom",
    paused_shared_memory_id="paused_state_shm",
    preview_shared_memory_id="preview_image_shm",
    preview_width=1280,
    preview_height=720,
)

BASE_SETTINGS = {
    "precision": "float16",
    "tensorrt_optimization_level": "3",
    "ncnn_gpu_id": "0",
    "pytorch_gpu_id": "1",
    "auto_border_cropping": "False",
    "dynamic_tensorrt_engine": "False",
    "quality_preset": "best",
    "temporal_reuse": "True",
    "temporal_reuse_statistic": "mean_abs",
    "preview_enabled": "True",
    "scene_change_detection_enabled": "True",
    "scene_change_detection_method": "sudo_scene_detect",
    "scene_change_detection_threshold": "3.5",
    "uhd_mode": "True",
}


class FakeSettings:
    """The two attributes `build_command` reads, and nothing else."""

    def __init__(self, **overrides):
        self.settings = dict(BASE_SETTINGS, **overrides)
        self.allowedSettings = {"quality_preset": ("best", "fast")}


def options(**overrides):
    base = dict(
        inputFile="/in/clip.mkv", outputPath="/out/clip.mkv", videoWidth=1920, videoHeight=1080,
        videoFps=24, tilingEnabled=False, tilesize="256", videoFrameCount=120, backend="ncnn",
        modelScale=2, interpolateTimes=2, benchmarkMode=False, sloMoMode=False,
        dyanmicScaleOpticalFlow=False, ensemble=False,
    )
    base.update(overrides)
    return RenderOptions(**base)


MINIMAL = ['/opt/rve/python/bin/python',
 '-W',
 'ignore',
 '/opt/rve/backend/rve-backend.py',
 '-i',
 '/in/clip.mkv',
 '-o',
 '/out/clip.mkv',
 '-b',
 'ncnn',
 '--precision',
 'float16',
 '--report',
 '/out/clip.report.json',
 '--custom_encoder',
 ' None ',
 '--tensorrt_opt_profile',
 '3',
 '--pause_shared_memory_id',
 'paused_state_shm',
 '--ncnn_gpu_id',
 '0',
 '--pytorch_gpu_id',
 '1',
 '--cwd',
 '/opt/rve',
 '--ffmpeg_path',
 '/opt/rve/bin/ffmpeg',
 '--quality_preset',
 'best',
 '--preview_shared_memory_id',
 'preview_image_shm',
 '--preview_size',
 '1280x720',
 '--scene_detect_method',
 'sudo_scene_detect',
 '--scene_detect_threshold',
 '3.5']

EVERYTHING = ['/opt/rve/python/bin/python',
 '-W',
 'ignore',
 '/opt/rve/backend/rve-backend.py',
 '-i',
 '/in/clip.mkv',
 '-o',
 '/out/clip.mkv',
 '-b',
 'pytorch',
 '--precision',
 'float16',
 '--report',
 '/out/clip.report.json',
 '--custom_encoder',
 ' None ',
 '--tensorrt_opt_profile',
 '3',
 '--pause_shared_memory_id',
 'paused_state_shm',
 '--ncnn_gpu_id',
 '0',
 '--pytorch_gpu_id',
 '1',
 '--cwd',
 '/opt/rve',
 '--ffmpeg_path',
 '/opt/rve/bin/ffmpeg',
 '--upscale_model',
 '/opt/rve/models/custom/upscale.pth',
 '--override_upscale_scale',
 '2',
 '--tilesize',
 '256',
 '--interpolate_model',
 '/opt/rve/models/interp.pth',
 '--interpolate_factor',
 '2',
 '--slomo_mode',
 '--dynamic_scaled_optical_flow',
 '--ensemble',
 '--extra_restoration_models',
 '/opt/rve/models/deblur.pth',
 '--extra_restoration_models',
 '/opt/rve/models/denoise.pth',
 '--extra_restoration_models',
 '/opt/rve/models/decompress.pth',
 '--preview_render',
 '--border_detect',
 '--tensorrt_dynamic_shapes',
 '--quality_preset',
 'best',
 '--no_temporal_reuse',
 '--preview_shared_memory_id',
 'preview_image_shm',
 '--preview_size',
 '1280x720',
 '--scene_detect_method',
 'pyscenedetect',
 '--scene_detect_threshold',
 '3.5',
 '--scene_detect_model',
 '/opt/rve/models/scene.pth',
 '--benchmark',
 '--UHD_mode',
 '--overwrite',
 '--hdr_mode']

SETTINGS_BYTES = ('precision,auto\n'
 'tensorrt_optimization_level,5\n'
 'dynamic_tensorrt_engine,False\n'
 'encoder,libx264\n'
 'video_encoder_speed,medium\n'
 'audio_encoder,copy_audio\n'
 'subtitle_encoder,copy_subtitle\n'
 'audio_bitrate,192k\n'
 'preview_enabled,True\n'
 'scene_change_detection_method,sudo_scene_detect\n'
 'scene_change_detection_enabled,True\n'
 'scene_change_detection_threshold,3.5\n'
 'discord_rich_presence,False\n'
 'video_quality,High\n'
 'output_folder_location,/fixed/Videos\n'
 'last_input_folder_location,/fixed/Videos\n'
 'uhd_mode,True\n'
 'ncnn_gpu_id,0\n'
 'pytorch_gpu_id,0\n'
 'auto_border_cropping,False\n'
 'video_container,mkv\n'
 'video_pixel_format,yuv420p\n'
 'pytorch_version,2.9.0\n'
 'pytorch_backend,CUDA\n'
 'auto_hdr_mode,True\n'
 'quality_preset,best\n'
 'encoder_command,\n'
 'temporal_reuse,True\n'
 'temporal_reuse_statistic,mean_abs\n')


def test_minimal_command_is_byte_identical():
    """The plain ncnn job: no model, no preview, reuse on. This is the list the old method returned."""
    assert build_command(options(), FakeSettings(), paths=PATHS, notice=lambda _m: None) == MINIMAL


def test_every_branch_command_is_byte_identical():
    """Every optional flag on at once: this is what pins the ORDER, not only the presence, of the flags."""
    settings = FakeSettings(
        auto_border_cropping="True", dynamic_tensorrt_engine="True",
        temporal_reuse="False", scene_change_detection_method="pyscenedetect",
    )
    command = build_command(
        options(
            backend="pytorch (cuda)", tilingEnabled=True, benchmarkMode=True, sloMoMode=True,
            dyanmicScaleOpticalFlow=True, ensemble=True, hdrMode=True, isPreview=True,
            upscaleModelFile="upscale.pth", upscaleModelArch="custom", overrideUpscaleScale=2,
            interpolateModelFile="interp.pth", deblurModelFile="deblur.pth",
            denoiseModelFile="denoise.pth", decompressModelFile="decompress.pth",
            sceneChangeModelFile="scene.pth", videoWidth=3840, videoHeight=2160,
        ),
        settings, is_overwrite=True, paths=PATHS, notice=lambda _m: None,
    )
    assert command == EVERYTHING


def test_the_pytorch_variants_are_folded_on_the_options():
    """The one mutation the old method made, kept: the queue row reads the backend back after the command."""
    job = options(backend="pytorch (rocm)")
    build_command(job, FakeSettings(), paths=PATHS, notice=lambda _m: None)
    assert job.backend == "pytorch"


def test_a_custom_upscale_arch_reads_the_custom_model_folder():
    command = build_command(
        options(upscaleModelFile="mine.pth", upscaleModelArch="custom", overrideUpscaleScale=4),
        FakeSettings(), paths=PATHS, notice=lambda _m: None,
    )
    assert command[command.index("--upscale_model") + 1] == "/opt/rve/models/custom/mine.pth"


def test_a_registered_upscale_arch_reads_the_shipped_model_folder():
    command = build_command(
        options(upscaleModelFile="shipped.pth", upscaleModelArch="swinir", overrideUpscaleScale=2),
        FakeSettings(), paths=PATHS, notice=lambda _m: None,
    )
    assert command[command.index("--upscale_model") + 1] == "/opt/rve/models/shipped.pth"


def test_an_unknown_quality_preset_falls_back_to_best():
    """An old settings.txt can hold a value the current build no longer offers; the backend must still get one."""
    command = build_command(options(), FakeSettings(quality_preset="turbo"), paths=PATHS, notice=lambda _m: None)
    assert command[command.index("--quality_preset") + 1] == "best"


def test_scene_detection_off_is_one_flag_and_no_threshold():
    command = build_command(
        options(), FakeSettings(scene_change_detection_enabled="False"), paths=PATHS, notice=lambda _m: None
    )
    assert command[command.index("--scene_detect_method") + 1] == "none"
    assert "--scene_detect_threshold" not in command


def test_coarse_reuse_is_passed_only_when_chosen():
    coarse = build_command(
        options(), FakeSettings(temporal_reuse="True", temporal_reuse_statistic="coarse"),
        paths=PATHS, notice=lambda _m: None,
    )
    assert coarse[coarse.index("--temporal_reuse_statistic") + 1] == "coarse"
    mean_abs = build_command(options(), FakeSettings(), paths=PATHS, notice=lambda _m: None)
    assert "--temporal_reuse_statistic" not in mean_abs


def test_uhd_mode_is_noticed_exactly_once_and_only_over_1080p():
    """The old method logged "UHD mode enabled" from inside the assembly; the notice is injected now."""
    notices = []
    build_command(options(videoWidth=3840), FakeSettings(), paths=PATHS, notice=notices.append)
    assert notices == ["UHD mode enabled"]

    notices.clear()
    build_command(options(videoWidth=1920), FakeSettings(), paths=PATHS, notice=notices.append)
    assert notices == []
