"""Tests for ``harness/tests/tools/bench_engine.py`` - the half that needs no GPU.

The harness is split on purpose: ``parse_args``, ``derive_tflops``,
``tflops_per_second``, ``format_summary``, ``format_tactic_table`` and the engine-file-name
helpers take numbers and strings; only ``load_engine_module``,
``build_engine_module``, ``time_module`` and ``dump_tactics`` touch a device. That split
is what lets this file exist in a cell with no torch, no CUDA and no ffmpeg - the same
reason ``harness/tests/test_tiled_upscale.py`` keeps its geometry layer torch-free.

Nothing in this file starts a process that touches the GPU: importing the tool must not
even import torch (asserted in a clean interpreter, below).
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parent / "tools" / "bench_engine.py"


def _load_tool():
    """Import the CLI as a module (``harness/tests/tools`` is not a package)."""
    spec = importlib.util.spec_from_file_location("bench_engine_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_tool()

# An engine name exactly as the app writes it (UpscaleTorch.py builds the static name from
# model + precision + device + versions + opt level + tiling + resolution).
APP_ENGINE_NAME = (
    "2x_Futsuu_Anime.pth_fp16_NVIDIA GeForce RTX 4060 Ti_trt-10.12.0.36"
    "_torch_tensorrt-2.7.0_opt-5_tiling-full_1920x1080.engine"
)


def _row(index, ms, previous=None, *, label="quiet", flop_basis=None, note=""):
    return bench.Measurement(
        index=index,
        label=label,
        source="engine models/x.engine",
        width=1920,
        height=1080,
        scale=2,
        dtype="fp16",
        dtype_basis="from the engine file name",
        tiling="full",
        tiling_basis="build flag",
        iters=25,
        warm=6,
        ms_per_frame=ms,
        loop_ms=ms * 25,
        flops_tflop=2.4816,
        flop_basis=flop_basis or "derived: 2.482 TFLOP (MODEL.md §1)",
        previous_ms=previous,
        note=note,
    )


# --------------------------------------------------------------------------------------
# importability without a GPU stack
# --------------------------------------------------------------------------------------
def test_module_imports_without_torch():
    """Importing the harness must not drag torch in: it has to load where no CUDA exists."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('bench_probe', {str(TOOL_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['bench_probe'] = module\n"
        "spec.loader.exec_module(module)\n"
        "print('torch' in sys.modules)\n"
        "print('torch_tensorrt' in sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False", "False"], done.stdout


def test_tool_file_exists():
    assert TOOL_PATH.is_file()


# --------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------
def test_accepts_the_documented_flags():
    args = bench.parse_args([
        "--engine", "models/x.engine",
        "--width", "1440", "--height", "1080", "--scale", "4",
        "--tiling", "moderate", "--opt-level", "5", "--timing-iters", "8",
        "--iters", "10", "--warm", "2", "--repeat", "3",
        "--tflop", "2.57", "--sample-clocks", "clocks.csv",
        "--with-nvenc", "--nvenc-size", "3840x2160", "--dump-tactics", "--json",
    ])
    assert args.engine == "models/x.engine"
    assert (args.width, args.height, args.scale) == (1440, 1080, 4)
    assert args.tiling == "moderate"
    assert (args.opt_level, args.timing_iters) == (5, 8)
    assert (args.iters, args.warm, args.repeat) == (10, 2, 3)
    assert args.tflop == pytest.approx(2.57)
    assert args.with_nvenc and args.dump_tactics and args.json


def test_measurement_defaults_match_the_documented_protocol():
    """harness/docs/MODEL.md §8: 25 timed calls after 6 warmup, opt 5, 8 timing iters."""
    args = bench.parse_args(["--model", "out/model.ts"])
    assert (args.iters, args.warm) == (25, 6)
    assert (args.opt_level, args.timing_iters) == (5, 8)
    assert args.tiling == "full"
    assert args.repeat == 1
    assert args.tflop is None


def test_rejects_an_unknown_tiling_level():
    with pytest.raises(SystemExit):
        bench.parse_args(["--model", "m.ts", "--tiling", "medium"])


def test_rejects_an_engine_and_a_model_together():
    with pytest.raises(SystemExit):
        bench.parse_args(["--engine", "e.engine", "--model", "m.ts"])


def test_requires_a_source():
    with pytest.raises(SystemExit):
        bench.parse_args([])


def test_rejects_out_of_range_measurement_settings():
    with pytest.raises(SystemExit):
        bench.parse_args(["--model", "m.ts", "--iters", "0"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--model", "m.ts", "--warm", "-1"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--model", "m.ts", "--opt-level", "6"])


def test_help_mentions_every_flag():
    buffer = io.StringIO()
    with pytest.raises(SystemExit) as exited:
        with redirect_stdout(buffer):
            bench.parse_args(["--help"])
    assert exited.value.code == 0
    text = buffer.getvalue()
    for flag in (
        "--engine", "--model", "--width", "--height", "--scale", "--tiling",
        "--opt-level", "--timing-iters", "--dtype", "--device", "--iters", "--warm",
        "--repeat", "--tflop", "--sample-clocks", "--json", "--with-nvenc",
        "--nvenc-size", "--nvenc-ffmpeg", "--dump-tactics",
    ):
        assert flag in text, flag


# --------------------------------------------------------------------------------------
# FLOP derivation (harness/docs/MODEL.md §1)
# --------------------------------------------------------------------------------------
def test_derive_tflops_reproduces_the_documented_table():
    assert bench.derive_tflops(1920, 1080, 2) == pytest.approx(2.48, abs=0.01)
    assert bench.derive_tflops(1920, 1080, 4) == pytest.approx(2.57, abs=0.01)
    # the ledger rounds the 1440x1080 x2 figure to 1.85 TFLOP; the arithmetic gives 1.861
    assert bench.derive_tflops(1440, 1080, 2) == pytest.approx(1.86, abs=0.01)


def test_derive_tflops_scales_with_area_and_is_dominated_by_the_body():
    assert bench.derive_tflops(3840, 2160, 2) == pytest.approx(
        4 * bench.derive_tflops(1920, 1080, 2), rel=1e-9)
    # scale changes only the tail (3*scale^2 output channels): x4 is ~3.5% above x2
    gain = bench.derive_tflops(1920, 1080, 4) / bench.derive_tflops(1920, 1080, 2)
    assert 1.03 < gain < 1.04


def test_derive_tflops_rejects_nonsense():
    with pytest.raises(bench.BenchError):
        bench.derive_tflops(0, 1080, 2)
    with pytest.raises(bench.BenchError):
        bench.derive_tflops(1920, 1080, 0)


def test_tflops_per_second_matches_the_measured_row():
    """51.37 ms at 2.48 TFLOP is the 48.3 TFLOP/s row of harness/docs/MODEL.md §2."""
    tflops = bench.derive_tflops(1920, 1080, 2)
    assert bench.tflops_per_second(tflops, 51.37) == pytest.approx(48.3, abs=0.1)
    with pytest.raises(bench.BenchError):
        bench.tflops_per_second(2.48, 0.0)


# --------------------------------------------------------------------------------------
# engine file name metadata (the plan does not carry the build flags)
# --------------------------------------------------------------------------------------
def test_engine_name_metadata_is_read_back():
    assert bench.tiling_from_engine_name(APP_ENGINE_NAME) == "full"
    assert bench.dtype_from_engine_name(APP_ENGINE_NAME) == "fp16"
    assert bench.tiling_from_engine_name("x_trt_tiling-moderate_1920x1080.engine") == "moderate"
    assert bench.dtype_from_engine_name("x_fp32_1920x1080.engine") == "fp32"


def test_engine_name_without_flags_says_nothing():
    assert bench.tiling_from_engine_name("custom.engine") is None
    assert bench.dtype_from_engine_name("custom.engine") is None


# --------------------------------------------------------------------------------------
# summary formatting: every number names its basis
# --------------------------------------------------------------------------------------
def test_summary_states_shape_dtype_tiling_and_bases():
    text = bench.format_summary([_row(1, 51.37)])
    assert "shape 1920x1080 @x2" in text
    assert "dtype fp16 (from the engine file name)" in text
    assert "tiling full (build flag)" in text
    assert "ms/frame 51.370" in text
    assert "measured:" in text
    assert "TFLOP/s  48.31" in text
    assert "derived" in text
    assert "baseline" in text


def test_summary_reports_the_ratio_to_the_previous_measurement():
    text = bench.format_summary([
        _row(1, 51.37),
        _row(2, 55.99, previous=51.37, label="+nvenc 3840x2160 hevc_nvenc"),
    ])
    # 55.99 / 51.37 = 1.090x, the NVENC contention row of harness/docs/MODEL.md §2
    assert "1.090x vs #1" in text
    assert "+nvenc" in text


def test_summary_calls_a_given_tflop_given():
    row = _row(1, 51.37, flop_basis="TFLOP from --tflop (2.000 TFLOP, given)")
    text = bench.format_summary([row])
    assert "given" in text
    assert "derived" not in text


def test_summary_without_measurements_does_not_crash():
    assert "no measurements" in bench.format_summary([])


def test_json_summary_round_trips():
    payload = bench.summary_json([_row(1, 51.37), _row(2, 55.99, previous=51.37)])
    text = json.dumps(payload)
    restored = json.loads(text)
    assert restored["tool"] == "bench_engine"
    first, second = restored["measurements"]
    assert first["shape"] == [1920, 1080]
    assert first["ratio_to_previous"] is None
    assert second["ratio_to_previous"] == pytest.approx(1.0899, abs=1e-3)
    assert second["tflops"] == pytest.approx(44.33, abs=0.05)


# --------------------------------------------------------------------------------------
# tactic dump parsing (harness/docs/MODEL.md §5)
# --------------------------------------------------------------------------------------
def test_tactic_table_groups_the_sm75_kernels():
    payload = json.dumps({"Layers": [
        {"Name": "conv1", "LayerType": "Convolution",
         "TacticName": "sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32"},
        {"Name": "conv2", "LayerType": "Convolution",
         "TacticName": "sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32"},
        {"Name": "conv_last", "LayerType": "Convolution",
         "TacticName": "sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16"},
        {"Name": "reshape_0", "LayerType": "Shuffle"},
    ]})
    text = bench.format_tactic_table(payload)
    assert "4 layers, 3 with a tactic name" in text
    assert "2x sm75_xmma" in text
    assert "1x (no tactic name)" in text
    assert "conv_last [Convolution] -> sm80_xmma" in text
    # a shuffle layer has no tactic by nature: that must not be reported as a build fault
    assert "non-compute layers" in text
    assert "not built with DETAILED" not in text


def test_tactic_table_without_tactic_names_says_so():
    payload = json.dumps({"Layers": [{"Name": "a", "LayerType": "Convolution"}]})
    text = bench.format_tactic_table(payload)
    assert "0 with a tactic name" in text
    assert "not built with DETAILED" in text


def test_tactic_table_rejects_non_json():
    with pytest.raises(bench.BenchError):
        bench.format_tactic_table("Engines with no layers")


def test_size_parsing():
    assert bench.parse_size("3840x2160") == (3840, 2160)
    with pytest.raises(bench.BenchError):
        bench.parse_size("3840")
