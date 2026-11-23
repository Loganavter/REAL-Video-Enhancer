"""The render command line, assembled from the render options and nothing else.

`ProcessTab.build_command` was 185 lines of pure string assembly that happened to live on a widget object, so the only
way to ask "what command does this job produce?" was to build a window. This module is that assembly with the widget
removed (P3 of the seam plan): `ProcessTab.build_command` now calls `build_command` here, and the command list is a
function of its arguments.

Three things are deliberately injected instead of imported:

* **the paths** - the app's interpreters, backend script, model directories, shared-memory names and preview size come
  from `src.constants`, which imports PySide6. `RenderPaths` lets a test state them, and `app_paths()` reads the real
  ones, lazily, for the app. Without that indirection this module could not be exercised on the portable interpreter
  (`harness/tests/test_render_command.py` pins the list against a capture of the pre-move method).
* **the settings** - anything with a `settings` mapping and an `allowedSettings` mapping (the real `Settings` from
  `src/ui/settings_store.py` is one). The command reads fifteen values from it and writes none.
* **the one notice** - UHD mode logs "UHD mode enabled" when it turns on. That is the only side effect the old method
  had besides mutating the options, so it is a callable: the app passes `src.Util.log`, a test passes a list append,
  and the assembly itself touches nothing.

The command must stay byte-identical for a fixed input: it IS the backend's invocation, and a reordered flag is a
different render. Nothing here may add, drop or reorder a flag without a measurement that says why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RenderPaths:
    """Everything in the command that is a property of the installation rather than of the job."""

    python: str
    backend_path: str
    cwd: str
    ffmpeg_path: str
    models_path: str
    custom_models_path: str
    paused_shared_memory_id: str
    preview_shared_memory_id: str
    preview_width: int
    preview_height: int


def app_paths() -> RenderPaths:
    """The real installation's paths. Imports `src.constants` here, not at module scope: it needs PySide6."""
    from ..constants import (
        BACKEND_PATH,
        CWD,
        FFMPEG_PATH,
        IMAGE_SHARED_MEMORY_ID,
        MODELS_PATH,
        CUSTOM_MODELS_PATH,
        PAUSED_STATE_SHARED_MEMORY_ID,
        PREVIEW_HEIGHT,
        PREVIEW_WIDTH,
        PYTHON_EXECUTABLE_PATH,
    )

    return RenderPaths(
        python=PYTHON_EXECUTABLE_PATH,
        backend_path=BACKEND_PATH,
        cwd=CWD,
        ffmpeg_path=FFMPEG_PATH,
        models_path=MODELS_PATH,
        custom_models_path=CUSTOM_MODELS_PATH,
        paused_shared_memory_id=PAUSED_STATE_SHARED_MEMORY_ID,
        preview_shared_memory_id=IMAGE_SHARED_MEMORY_ID,
        preview_width=PREVIEW_WIDTH,
        preview_height=PREVIEW_HEIGHT,
    )


def _app_notice(message: str) -> None:
    """The app's log line, resolved lazily (`src.Util` is not importable without the app's dependencies)."""
    from ..Util import log

    log(message)


def build_command(render_options, settings, is_overwrite: bool = False, paths: RenderPaths | None = None,
                  notice=None) -> list[str]:
    """The backend invocation for one render job, in the order the backend parses it.

    `render_options` is the tab's `RenderOptions` (nothing is imported from it: the fields are read by name);
    `settings` is the settings store; `is_overwrite` is the tab's answer to "the output file already exists, replace
    it"; `paths` defaults to `app_paths()` and `notice` to `src.Util.log`.

    One mutation is kept from the original, on purpose: the four pytorch collection variants are folded to
    ``"pytorch"`` ON the options object, because the tab's own state and the queue rows read it back afterwards.
    """
    if paths is None:
        paths = app_paths()
    if notice is None:
        notice = _app_notice

    if (
        render_options.backend == "pytorch (cuda)"
        or render_options.backend == "pytorch (rocm)"
        or render_options.backend == "pytorch (xpu)"
        or render_options.backend == "pytorch (mps)"
    ):
        render_options.backend = (
            "pytorch"  # pytorch is the same for both cuda and rocm
        )

    command = [
        f"{paths.python}",
        "-W",
        "ignore",
        os.path.join(paths.backend_path, "rve-backend.py"),
        "-i",
        render_options.inputFile,
        "-o",
        render_options.outputPath,
        "-b",
        f"{render_options.backend}",
        "--precision",
        f"{settings.settings['precision']}",
        # the run report is written next to the output, always: it is what makes "it got slower" a
        # number with a commit and a card (python3 -m harness.ingest <report>)
        "--report",
        f"{os.path.splitext(render_options.outputPath)[0]}.report.json",
        "--custom_encoder",
        f" {render_options.encoderCommand} ",
        "--tensorrt_opt_profile",
        f"{settings.settings['tensorrt_optimization_level']}",
        "--pause_shared_memory_id",
        f"{paths.paused_shared_memory_id}",
        "--ncnn_gpu_id",
        f"{settings.settings['ncnn_gpu_id']}",
        "--pytorch_gpu_id",
        f"{settings.settings['pytorch_gpu_id']}",
        "--cwd",
        f"{paths.cwd}",
        "--ffmpeg_path",
        f"{paths.ffmpeg_path}",

    ]

    if render_options.upscaleModelFile:
        modelPath = os.path.join(paths.models_path, render_options.upscaleModelFile)
        if render_options.upscaleModelArch == "custom":
            modelPath = os.path.join(
                paths.custom_models_path, render_options.upscaleModelFile
            )
        command += [
            "--upscale_model",
            modelPath,
        ]
        command += ["--override_upscale_scale", f"{render_options.overrideUpscaleScale}"]

    if render_options.tilingEnabled:
        command += [
            "--tilesize",
            f"{render_options.tilesize}",
        ]


    if render_options.interpolateModelFile:
        command += [
            "--interpolate_model",
            os.path.join(
                paths.models_path,
                render_options.interpolateModelFile,
            ),
            "--interpolate_factor",
            f"{render_options.interpolateTimes}",
        ]
        if render_options.sloMoMode:
            command += [
                "--slomo_mode",
            ]
        if render_options.dyanmicScaleOpticalFlow:
            command += [
                "--dynamic_scaled_optical_flow",
            ]
        if render_options.ensemble:
            command += [
                "--ensemble",
            ]

    if render_options.deblurModelFile:
        command += [
            "--extra_restoration_models",
            os.path.join(
                paths.models_path,
                render_options.deblurModelFile,
            ),
        ]
    if render_options.denoiseModelFile:
        command += [
            "--extra_restoration_models",
            os.path.join(
                paths.models_path,
                render_options.denoiseModelFile,
            ),
        ]
    if render_options.decompressModelFile:
        command += [
            "--extra_restoration_models",
            os.path.join(
                paths.models_path,
                render_options.decompressModelFile,
            ),
        ]

    if render_options.isPreview:
        # The preview file is read WHILE it is written (the pane tail-follows it): the backend writes a fragmented
        # container with a short GOP for exactly this job, and keeps `+faststart` for real exports.
        command += ["--preview_render"]

    if settings.settings["auto_border_cropping"] == "True":
        command += [
            "--border_detect",
        ]

    if settings.settings["dynamic_tensorrt_engine"] == "True":
        command += [
            "--tensorrt_dynamic_shapes",
        ]

    # always explicit, even though "best" is the backend default, and tolerant of an
    # older settings.txt that predates the key.
    quality_preset = settings.settings.get("quality_preset", "best")
    if quality_preset not in settings.allowedSettings["quality_preset"]:
        quality_preset = "best"
    command += [
        "--quality_preset",
        f"{quality_preset}",
    ]

    # the backend defaults to reuse ON; the switch is therefore an opt-out, and it is passed explicitly so
    # an old settings.txt cannot silently mean something else than what the checkbox shows
    if settings.settings.get("temporal_reuse", "True") == "False":
        command += ["--no_temporal_reuse"]
    # "by cel": compare the drawing, not the film (3 px low-pass + the scan's translation compensated, and
    # the reused frame is moved by that estimate). Passed only when chosen - the backend's default is the
    # shipped statistic, so an old settings.txt that predates the key cannot mean something else.
    elif settings.settings.get("temporal_reuse_statistic", "mean_abs") == "coarse":
        command += ["--temporal_reuse_statistic", "coarse"]

    if settings.settings["preview_enabled"] == "True":
        command += [
            "--preview_shared_memory_id",
            f"{paths.preview_shared_memory_id}",
            "--preview_size",
            f"{paths.preview_width}x{paths.preview_height}",
        ]

    if settings.settings["scene_change_detection_enabled"] == "False":
        command += ["--scene_detect_method", "none"]
    else:
        command += [
            "--scene_detect_method",
            settings.settings["scene_change_detection_method"],
            "--scene_detect_threshold",
            settings.settings["scene_change_detection_threshold"],
        ]
        if render_options.sceneChangeModelFile:
            command += [
                "--scene_detect_model",
                os.path.join(
                    paths.models_path,
                    render_options.sceneChangeModelFile,
                ),
            ]

    if render_options.benchmarkMode:
        command += ["--benchmark"]

    if settings.settings["uhd_mode"] == "True":
        if render_options.videoWidth > 1920 or render_options.videoHeight > 1080:
            command += ["--UHD_mode"]
            notice("UHD mode enabled")

    if is_overwrite:
        command += ["--overwrite"]

    if render_options.hdrMode:
        command += ["--hdr_mode"]

    return command
