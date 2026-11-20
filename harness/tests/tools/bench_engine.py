#!/usr/bin/env python3
"""Reproducible benchmark harness for the TensorRT engines this repo builds.

Every number in ``harness/docs/MODEL.md`` and ``harness/docs/RESEARCH-LEDGER.md`` was taken with a
throwaway script in /tmp. This file is the same measurement, kept in the repository, so a
number can be re-taken instead of re-derived from prose::

    python3 harness/tests/tools/bench_engine.py --engine models/<name>.engine \\
        --width 1920 --height 1080 --tiling full --repeat 3

Nothing here imports ``backend/src`` on purpose: a machine with a TensorRT card and no app
installed must still be able to re-take the numbers. ``torch``/``torch_tensorrt``/
``tensorrt`` are imported inside the functions that need them, so a missing one is a
one-line error rather than a traceback (and so the geometry/formatting half is importable
and testable where no CUDA exists - see ``harness/tests/test_bench_tool.py``).

Measurement protocol, as recorded in ``harness/docs/MODEL.md`` §8 and the "Measurement protocol"
section of the research ledger:

* one GPU job at a time - a second render or engine build on the same card invalidates
  every number, and ``nvidia-smi`` utilisation is not an idle check on this host;
* time the loop, not the call: cuda events around ``--iters`` calls after ``--warm``
  warmup calls, ms/frame = loop time / iters;
* ratios are only meaningful inside one session (the spread between sessions on this host
  reaches 26%), which is why ``--repeat`` prints every run instead of one average.

The split between layers: the geometry/formatting functions take numbers and strings and
return numbers and strings; only ``load_engine_module``, ``build_engine_module``,
``time_module`` and ``dump_tactics`` touch a device.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BenchError",
    "Measurement",
    "TILING_LEVELS",
    "build_parser",
    "parse_args",
    "parse_size",
    "derive_tflops",
    "tflops_per_second",
    "tiling_from_engine_name",
    "dtype_from_engine_name",
    "format_summary",
    "summary_json",
    "format_tactic_table",
    "load_engine_module",
    "build_engine_module",
    "time_module",
    "dump_tactics",
    "main",
]

# torch-tensorrt 2.7 accepts these; "fast" is the level between "none" and "moderate"
# (torch_tensorrt.dynamo.CompilationSettings.tiling_optimization_level).
TILING_LEVELS: Tuple[str, ...] = ("none", "fast", "moderate", "full")

# The x2 compact net of harness/docs/MODEL.md §1, in the form its FLOP table is written:
#   head  conv3x3(3  -> 64)          every convolution at the LOW-resolution size
#   body  16 x conv3x3(64 -> 64)
#   tail  conv3x3(64 -> 3*scale^2)   then PixelShuffle
HEAD_IN_CHANNELS = 3
BODY_CHANNELS = 64
BODY_CONVS = 16
KERNEL = 3


class BenchError(RuntimeError):
    """A condition the operator can fix (missing module, missing engine, bad ffmpeg)."""


# --------------------------------------------------------------------------------------
# pure geometry: FLOPs, engine-name metadata, sizes
# --------------------------------------------------------------------------------------
def parse_size(text: str) -> Tuple[int, int]:
    """``"3840x2160"`` -> ``(3840, 2160)``. Raises BenchError on anything else."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise BenchError(f"expected WIDTHxHEIGHT, got {text!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise BenchError(f"expected WIDTHxHEIGHT, got {text!r}") from exc
    if width <= 0 or height <= 0:
        raise BenchError(f"expected positive WIDTHxHEIGHT, got {text!r}")
    return width, height


def _conv_flops(planes: int, channels_in: int, channels_out: int, kernel: int = KERNEL) -> float:
    """FLOPs of one k x k convolution over ``planes`` (H*W) positions, MAC = 2 FLOPs."""
    return 2.0 * planes * channels_in * channels_out * kernel * kernel


def derive_tflops(width: int, height: int, scale: int = 2) -> float:
    """TFLOP of one frame through the compact SR net, from its own layer arithmetic.

    Reproduces the table in ``harness/docs/MODEL.md`` §1: 2.48 TFLOP at 1920x1080 for x2 and
    ~2.57 for x4, i.e. the x4 tail adds 0.09 TFLOP while all the work stays at the input
    resolution. This is arithmetic, not a measurement: the summary prints it as derived.
    """
    if width <= 0 or height <= 0:
        raise BenchError(f"width and height must be positive, got {width}x{height}")
    if scale < 1:
        raise BenchError(f"scale must be >= 1, got {scale}")
    planes = width * height
    head = _conv_flops(planes, HEAD_IN_CHANNELS, BODY_CHANNELS)
    body = BODY_CONVS * _conv_flops(planes, BODY_CHANNELS, BODY_CHANNELS)
    tail = _conv_flops(planes, BODY_CHANNELS, 3 * scale * scale)
    return (head + body + tail) / 1e12


def tflops_per_second(flops_tflop: float, ms_per_frame: float) -> float:
    """TFLOP/s from a per-frame FLOP count and a measured ms/frame."""
    if ms_per_frame <= 0:
        raise BenchError(f"ms/frame must be positive, got {ms_per_frame}")
    return flops_tflop / (ms_per_frame / 1000.0)


def tiling_from_engine_name(path: str) -> Optional[str]:
    """Tiling level recorded in an engine file name, or None.

    The plan itself does not carry the flag, which is why the app puts it in the file
    name (``backend/src/pytorch/UpscaleTorch.py``, the ``_tiling-<level>`` suffix).
    """
    name = Path(path).name
    for level in ("full", "moderate", "fast"):
        if f"_tiling-{level}" in name:
            return level
    return None


def dtype_from_engine_name(path: str) -> Optional[str]:
    """``fp16``/``fp32`` as the app writes it into the engine file name, or None."""
    name = Path(path).name
    if "_fp16" in name:
        return "fp16"
    if "_fp32" in name:
        return "fp32"
    return None


def torch_dtype_name(dtype_name: str) -> str:
    """Normalise the ``--dtype`` value for printing."""
    return "fp16" if dtype_name in ("fp16", "half", "float16") else "fp32"


# --------------------------------------------------------------------------------------
# pure formatting
# --------------------------------------------------------------------------------------
@dataclass
class Measurement:
    """One row of the summary. Every number carries the basis it was obtained from."""

    index: int
    label: str
    source: str
    width: int
    height: int
    scale: int
    dtype: str
    dtype_basis: str
    tiling: str
    tiling_basis: str
    iters: int
    warm: int
    ms_per_frame: float
    loop_ms: float
    flops_tflop: float
    flop_basis: str
    previous_ms: Optional[float] = None
    note: str = ""

    @property
    def ratio(self) -> Optional[float]:
        """ms/frame relative to the previous measurement, or None for the first."""
        if not self.previous_ms:
            return None
        return self.ms_per_frame / self.previous_ms

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "source": self.source,
            "shape": [self.width, self.height],
            "scale": self.scale,
            "dtype": self.dtype,
            "dtype_basis": self.dtype_basis,
            "tiling": self.tiling,
            "tiling_basis": self.tiling_basis,
            "iters": self.iters,
            "warm": self.warm,
            "ms_per_frame": round(self.ms_per_frame, 3),
            "loop_ms": round(self.loop_ms, 3),
            "tflops": round(tflops_per_second(self.flops_tflop, self.ms_per_frame), 2),
            "flops_tflop": round(self.flops_tflop, 4),
            "flop_basis": self.flop_basis,
            "ratio_to_previous": None if self.ratio is None else round(self.ratio, 4),
            "note": self.note,
        }


def _labelled(value: str, basis: str) -> str:
    return f"{value} ({basis})" if basis else value


def format_summary(rows: Sequence[Measurement]) -> str:
    """The human block: one block per measurement, each number naming its basis."""
    if not rows:
        return "bench_engine: no measurements"
    out: List[str] = [
        f"bench_engine: {len(rows)} measurement(s), ms/frame = cuda events around the "
        f"whole loop after warmup",
    ]
    for row in rows:
        out.append(f"  #{row.index} {row.label} - {row.source}")
        out.append(
            "      "
            + "  ".join(
                (
                    f"shape {row.width}x{row.height} @x{row.scale} "
                    "(LR input, from --width/--height)",
                    f"dtype {_labelled(row.dtype, row.dtype_basis)}",
                    f"tiling {_labelled(row.tiling, row.tiling_basis)}",
                )
            )
        )
        out.append(
            f"      ms/frame {row.ms_per_frame:.3f}   measured: {row.loop_ms:.2f} ms over "
            f"{row.iters} calls after {row.warm} warmup"
        )
        out.append(
            f"      TFLOP/s  {tflops_per_second(row.flops_tflop, row.ms_per_frame):.2f}   "
            f"{row.flop_basis} / {row.ms_per_frame:.3f} ms"
        )
        if row.ratio is None:
            ratio = "baseline (no previous measurement)"
        else:
            ratio = f"{row.ratio:.3f}x vs #{row.index - 1} (measured ms/frame pair, same session)"
        out.append(f"      ratio    {ratio}")
        if row.note:
            out.append(f"      note     {row.note}")
    out.append(
        "  ratios compare measurements taken in one process only; one GPU job at a time "
        "(harness/docs/RESEARCH-LEDGER.md)"
    )
    return "\n".join(out)


def summary_json(rows: Sequence[Measurement]) -> Dict[str, Any]:
    """The same block as JSON, for a machine-readable record."""
    return {
        "tool": "bench_engine",
        "protocol": "one GPU job at a time; cuda events around the whole loop",
        "measurements": [row.as_dict() for row in rows],
    }


def format_tactic_table(engine_info: str) -> str:
    """Compact tactic table from ``IEngineInspector.get_engine_information(JSON)``.

    The raw JSON is one object per layer and hundreds of lines long; what the sm75 lead in
    ``harness/docs/MODEL.md`` §5 needed was the grouped tactic names and then the per-layer list.
    """
    try:
        payload = json.loads(engine_info)
    except (TypeError, ValueError) as exc:
        raise BenchError(f"IEngineInspector did not return JSON: {exc}") from exc
    layers = payload.get("Layers") if isinstance(payload, dict) else None
    if not layers:
        raise BenchError("engine info JSON has no 'Layers' array")

    rows: List[Tuple[str, str, str]] = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        rows.append(
            (
                str(layer.get("Name", "?")),
                str(layer.get("LayerType", "?")),
                str(layer.get("TacticName") or ""),
            )
        )

    counts: Dict[str, int] = {}
    for _, _, tactic in rows:
        key = tactic or "(no tactic name)"
        counts[key] = counts.get(key, 0) + 1

    without = sum(1 for _, _, tactic in rows if not tactic)
    with_tactic = len(rows) - without
    lines = [f"tactic dump: {len(rows)} layers, {with_tactic} with a tactic name"]
    for tactic, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count:>3}x {tactic}")
    lines.append("  per layer:")
    for name, layer_type, tactic in rows:
        lines.append(f"    {name} [{layer_type}] -> {tactic or '-'}")
    if without:
        lines.append(
            f"  {without} layer(s) report no tactic name; non-compute layers (shuffle, "
            "reformat, constants) never carry one"
        )
    if with_tactic == 0:
        lines.append(
            "  none of the layers carries a tactic name: the plan was not built with "
            "DETAILED profiling verbosity, which is the only setting TRT reports tactics for"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_engine.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Time a torch-tensorrt engine (or build one with the flags the app uses) and "
            "print a summary where every number names its basis."
        ),
        epilog=(
            "examples:\n"
            "  bench_engine.py --engine models/animevideov3.engine --width 1920 --height 1080\n"
            "  bench_engine.py --model out/model.ts --width 1920 --height 1080 \\\n"
            "      --tiling full --opt-level 5 --timing-iters 8 --repeat 3\n"
            "  bench_engine.py --engine e.engine --with-nvenc --sample-clocks clocks.csv\n"
        ),
    )
    source = parser.add_argument_group("engine source (one of)")
    source.add_argument("--engine", metavar="PATH",
                        help="existing torch-tensorrt engine file, loaded with torch.jit.load")
    source.add_argument("--model", metavar="PATH",
                        help="model to build from (TorchScript or a torch.export .pt2 file)")

    build = parser.add_argument_group("build (used when --engine is absent)")
    build.add_argument("--width", type=int, default=1920,
                       help="LR input width in pixels (default 1920)")
    build.add_argument("--height", type=int, default=1080,
                       help="LR input height in pixels (default 1080)")
    build.add_argument("--scale", type=int, default=2,
                       help="model scale, for the FLOP derivation (default 2)")
    build.add_argument("--tiling", choices=TILING_LEVELS, default="full",
                       help="TRT tiling_optimization_level (default full, the app default)")
    build.add_argument("--opt-level", type=int, default=5,
                       help="TRT builder optimization level 0-5 (default 5)")
    build.add_argument("--timing-iters", type=int, default=8,
                       help="TRT num_avg_timing_iters (default 8)")
    build.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16",
                       help="engine precision / input dtype (default fp16)")
    build.add_argument("--device", default="cuda:0", help="torch device (default cuda:0)")

    measure = parser.add_argument_group("measurement")
    measure.add_argument("--iters", type=int, default=25,
                         help="timed calls per measurement (default 25)")
    measure.add_argument("--warm", type=int, default=6,
                         help="warmup calls before timing (default 6)")
    measure.add_argument("--repeat", type=int, default=1,
                         help="repeat the whole measurement N times (default 1)")
    measure.add_argument("--tflop", type=float, default=None,
                         help="override the per-frame TFLOP used for TFLOP/s (else derived "
                              "from --width/--height/--scale)")
    measure.add_argument("--sample-clocks", metavar="FILE", default=None,
                         help="append timestamp,clocks.sm,power.draw,utilization.gpu,"
                              "temperature.gpu every ~0.5 s while measuring (nvidia-smi)")
    measure.add_argument("--json", action="store_true",
                         help="print the summary as JSON instead of the human block")

    extra = parser.add_argument_group("optional extras")
    extra.add_argument("--with-nvenc", action="store_true",
                       help="also measure with an ffmpeg hevc_nvenc encode of a synthetic "
                            "10-bit 3840x2160 raw stream running alongside")
    extra.add_argument("--nvenc-size", default="3840x2160", metavar="WxH",
                       help="synthetic encoder frame size (default 3840x2160)")
    extra.add_argument("--nvenc-ffmpeg", default="ffmpeg", metavar="PATH",
                       help="ffmpeg binary for --with-nvenc (default ffmpeg)")
    extra.add_argument("--dump-tactics", action="store_true",
                       help="rebuild with DETAILED profiling verbosity and print the "
                            "per-layer tactic table via IEngineInspector")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.engine and args.model:
        parser.error("--engine and --model are mutually exclusive (pass one)")
    if not args.engine and not args.model:
        parser.error("one of --engine or --model is required")
    if not 0 <= args.opt_level <= 5:
        parser.error(f"--opt-level must be within 0..5, got {args.opt_level}")
    if args.width <= 0 or args.height <= 0:
        parser.error(f"--width/--height must be positive, got {args.width}x{args.height}")
    if args.scale < 1:
        parser.error(f"--scale must be >= 1, got {args.scale}")
    if args.timing_iters < 1:
        parser.error(f"--timing-iters must be >= 1, got {args.timing_iters}")
    if args.iters < 1:
        parser.error(f"--iters must be >= 1, got {args.iters}")
    if args.warm < 0:
        parser.error(f"--warm must be >= 0, got {args.warm}")
    if args.repeat < 1:
        parser.error(f"--repeat must be >= 1, got {args.repeat}")
    if args.tflop is not None and args.tflop <= 0:
        parser.error(f"--tflop must be positive, got {args.tflop}")
    try:
        parse_size(args.nvenc_size)
    except BenchError as exc:
        parser.error(f"--nvenc-size: {exc}")
    return args


# --------------------------------------------------------------------------------------
# runtime layer: torch / torch_tensorrt / tensorrt / ffmpeg / nvidia-smi
# --------------------------------------------------------------------------------------
def _import_or_fail(module_name: str, hint: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise BenchError(f"{module_name} is not installed ({exc}); {hint}") from exc


def require_torch() -> Any:
    return _import_or_fail("torch", "install the app's torch build or run the GPU-free tests only")


def require_torch_tensorrt() -> Any:
    # importing it also registers the torchscript ops an app engine file needs at load time
    return _import_or_fail("torch_tensorrt", "pip install torch_tensorrt on the machine with the card")


def require_tensorrt() -> Any:
    return _import_or_fail("tensorrt", "torch_tensorrt ships tensorrt; check the install")


def _torch_device(spec: str, torch: Any) -> Any:
    device = torch.device(spec)
    if device.type != "cuda":
        raise BenchError(f"a TensorRT engine needs a CUDA device, got {spec!r}")
    if not torch.cuda.is_available():
        raise BenchError("torch reports no CUDA device; this harness cannot run here")
    return device


def _torch_dtype(name: str, torch: Any) -> Any:
    return torch.float16 if torch_dtype_name(name) == "fp16" else torch.float32


def _load_model_object(path: str, torch: Any) -> Any:
    """Load ``--model``: a TorchScript module first, a torch.export program second."""
    try:
        return torch.jit.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - the message is the point
        try:
            return torch.export.load(path).module()
        except Exception:
            raise BenchError(
                f"cannot load {path} as a model: torch.jit.load failed ({exc}) and "
                f"torch.export.load failed too. Pass a TorchScript module or a "
                f"torch.export .pt2 file; an app spandrel .pth is a state dict, not a module."
            ) from exc


def _prepare_model(model: Any, sample: Any) -> Any:
    """Move the module to the sample's device and dtype, as the app does before building.

    The app builds from a wrapper whose model already sits on the card in the engine's
    precision (UpscaleModelWrapper.set_precision); exporting a CPU module against a CUDA
    sample fails on the device mismatch instead of on anything interesting.
    """
    try:
        return model.eval().to(device=sample.device, dtype=sample.dtype)
    except Exception as exc:  # noqa: BLE001
        raise BenchError(
            f"cannot move the model to {sample.device}/{sample.dtype}: {exc}"
        ) from exc


def _exported_program(model: Any, sample: Any, torch: Any) -> Any:
    """TorchScript -> ExportedProgram the way the app does it (TensorRTHandler)."""
    if isinstance(model, torch.jit.ScriptModule):
        from torch._export.converter import TS2EPConverter

        return TS2EPConverter(model, sample_args=(sample,), sample_kwargs=None).convert()
    return torch.export.export(model, (sample,))


def _exported_program_from_file(model_path: str, sample: Any, torch: Any) -> Any:
    model = _prepare_model(_load_model_object(model_path, torch), sample)
    return _exported_program(model, sample, torch)


def _compile_settings(args: argparse.Namespace, torch: Any,
                      debug: bool = False) -> Dict[str, Any]:
    """The compile kwargs the app passes, read out of TensorRTHandler.build_engine."""
    return dict(
        device=torch.device(args.device),
        enabled_precisions={_torch_dtype(args.dtype, torch)},
        use_explicit_typing=False,   # app: trt_multi_precision_engine=False
        debug=debug,
        num_avg_timing_iters=args.timing_iters,
        workspace_size=0,            # app default: trt_workspace_size=0 ("let TRT pick")
        min_block_size=1,
        max_aux_streams=None,
        optimization_level=args.opt_level,
    )


def load_engine_module(path: str, device: Any, torch: Any) -> Any:
    """Load an engine the way the app does: ``torch.jit.load(...).eval()``."""
    if not Path(path).is_file():
        raise BenchError(f"engine not found: {path}")
    try:
        return torch.jit.load(path, map_location=device).eval()
    except Exception as exc:  # noqa: BLE001
        raise BenchError(
            f"cannot load {path} as a torch-tensorrt engine ({exc}). The file has to be the "
            f"torchscript export torch_tensorrt.save wrote, not a raw .plan."
        ) from exc


def build_engine_module(model_path: str, sample: Any, args: argparse.Namespace,
                        torch: Any, torch_tensorrt: Any, debug: bool = False) -> Any:
    """Build an engine with the app's flags; same tiling fallback as the app."""
    exported = _exported_program_from_file(model_path, sample, torch)
    settings = _compile_settings(args, torch, debug=debug)

    if args.tiling != "none":
        try:
            print(
                f"bench_engine: building TensorRT engine (tiling={args.tiling}, "
                f"opt-level={args.opt_level}, timing-iters={args.timing_iters}, debug={debug})",
                file=sys.stderr,
            )
            return torch_tensorrt.dynamo.compile(
                exported, (sample,), tiling_optimization_level=args.tiling, **settings
            )
        except Exception as exc:  # noqa: BLE001 - the app falls back here too
            print(
                f"bench_engine: tiling '{args.tiling}' failed ({exc}), rebuilding without it",
                file=sys.stderr,
            )
    print("bench_engine: building TensorRT engine (no tiling optimization)", file=sys.stderr)
    return torch_tensorrt.dynamo.compile(exported, (sample,), **settings)


def time_module(module: Any, sample: Any, iters: int, warm: int) -> Tuple[float, float]:
    """Return ``(ms_per_frame, loop_ms)``: cuda events around the whole loop."""
    torch = require_torch()
    with torch.inference_mode():
        for _ in range(warm):
            module(sample)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module(sample)
        end.record()
        torch.cuda.synchronize()
    loop_ms = float(start.elapsed_time(end))
    return loop_ms / float(iters), loop_ms


class ClockSampler:
    """Append one nvidia-smi row every ``interval`` seconds for the life of the block."""

    QUERY = ("timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu")

    def __init__(self, path: str, interval: float = 0.5, binary: str = "nvidia-smi") -> None:
        self.path = path
        self.interval = interval
        self.binary = binary
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file: Optional[Any] = None

    def __enter__(self) -> "ClockSampler":
        try:
            self._file = open(self.path, "a", encoding="utf-8")
        except OSError as exc:
            raise BenchError(f"cannot open --sample-clocks file {self.path}: {exc}") from exc
        if self._file.tell() == 0:
            self._file.write(f"# {self.QUERY} (nvidia-smi csv, noheader, nounits)\n")
            self._file.flush()
        if shutil.which(self.binary) is None:
            self._note(f"# {self.binary} not found; no samples recorded")
            print(f"bench_engine: {self.binary} not found, no clock samples recorded",
                  file=sys.stderr)
            return self
        self._thread = threading.Thread(target=self._loop, name="clock-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._file is not None:
            self._file.close()

    def _note(self, line: str) -> None:
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        try:
            done = subprocess.run(
                [self.binary, f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._note(f"# nvidia-smi failed: {exc}")
            return
        if done.returncode != 0:
            self._note(f"# nvidia-smi exited {done.returncode}: {done.stderr.strip()[:200]}")
            return
        for line in done.stdout.splitlines():
            if line.strip():
                self._note(line.strip())


class NvencAlongside:
    """Run an ``ffmpeg hevc_nvenc`` encode of synthetic 10-bit 4K raw for the block.

    This is the encoder-contention measurement of ``harness/docs/MODEL.md`` §2 (51.37 -> 55.99 ms
    for one engine at 1920x1080): the encoder is a second GPU job on the same card, which
    the "one GPU job at a time" rule normally forbids - so it is opt-in and the summary
    says which row had it.
    """

    def __init__(self, ffmpeg: str = "ffmpeg", size: Tuple[int, int] = (3840, 2160),
                 pix_fmt: str = "yuv420p10le", seconds: int = 600) -> None:
        self.ffmpeg = ffmpeg
        self.size = size
        self.pix_fmt = pix_fmt
        self.seconds = seconds
        self.proc: Optional[subprocess.Popen] = None
        self._stderr = ""

    def __enter__(self) -> "NvencAlongside":
        if shutil.which(self.ffmpeg) is None:
            raise BenchError(f"--with-nvenc needs ffmpeg, {self.ffmpeg!r} is not on PATH")
        width, height = self.size
        command = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", self.pix_fmt,
            "-s", f"{width}x{height}", "-r", "24", "-i", "/dev/zero",
            "-t", str(self.seconds),
            "-c:v", "hevc_nvenc", "-preset", "p5", "-f", "null", "-",
        ]
        try:
            self.proc = subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
        except OSError as exc:
            raise BenchError(f"cannot start ffmpeg: {exc}") from exc
        # NVENC takes a moment to open the session; timing it while it is still initialising
        # measures the ramp, not the contention.
        time.sleep(1.5)
        if self.proc.poll() is not None:
            raise BenchError(
                "ffmpeg exited before the measurement started "
                f"(exit {self.proc.returncode}); stderr: {self._drain_stderr()}"
            )
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
        # communicate() drains the pipe: reading after wait() can block on a full buffer
        try:
            _, err = self.proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            _, err = self.proc.communicate()
        self._stderr = (err or "").strip()

    def _drain_stderr(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return (self.proc.stderr.read() or "").strip()[:400]
        except (OSError, ValueError):
            return ""

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def stderr(self) -> str:
        return self._stderr


def dump_tactics(model_path: str, sample: Any, args: argparse.Namespace) -> str:
    """Rebuild the engine with DETAILED profiling verbosity and dump the tactic table.

    ``harness/docs/MODEL.md`` §8 records this exact path: ``cfg.profiling_verbosity =
    trt.ProfilingVerbosity.DETAILED``, then
    ``IEngineInspector().get_engine_information(LayerInformationFormat.JSON)``. The
    high-level dynamo API does not expose the verbosity field, so the only knob available
    here is ``debug=True``, which is what raises it in the torch-tensorrt builds this repo
    measured on. If the plan still reports no tactic names, the dump says so instead of
    pretending - the reader then has to build through the raw TRT builder.
    """
    torch = require_torch()
    torch_tensorrt = require_torch_tensorrt()
    trt = require_tensorrt()

    settings = _compile_settings(args, torch, debug=True)
    exported = _exported_program_from_file(model_path, sample, torch)

    converter = getattr(torch_tensorrt.dynamo,
                        "convert_exported_program_to_serialized_trt_engine", None)
    if converter is None:
        raise BenchError(
            "this torch_tensorrt has no dynamo.convert_exported_program_to_serialized_trt_engine, "
            "so the built plan cannot be handed to IEngineInspector"
        )
    try:
        engine_bytes = converter(exported, arg_inputs=[sample],
                                 tiling_optimization_level=args.tiling, **settings)
    except TypeError as exc:
        # torch_tensorrt changed this signature across 2.x: an older build takes the
        # inputs positionally and knows no tiling kwarg.
        if "unexpected keyword argument" not in str(exc):
            raise
        engine_bytes = converter(exported, [sample], **settings)

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise BenchError("TensorRT could not deserialize the freshly built plan")
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    return format_tactic_table(info)


def _measure(args: argparse.Namespace, label: str, module: Any, sample: Any,
             previous_ms: Optional[float], index: int, *, source: str, tiling: str,
             tiling_basis: str, dtype_label: str, dtype_basis: str,
             flops: float, flop_basis: str, note: str = "") -> Measurement:
    ms, loop_ms = time_module(module, sample, args.iters, args.warm)
    return Measurement(
        index=index,
        label=label,
        source=source,
        width=args.width,
        height=args.height,
        scale=args.scale,
        dtype=dtype_label,
        dtype_basis=dtype_basis,
        tiling=tiling,
        tiling_basis=tiling_basis,
        iters=args.iters,
        warm=args.warm,
        ms_per_frame=ms,
        loop_ms=loop_ms,
        flops_tflop=flops,
        flop_basis=flop_basis,
        previous_ms=previous_ms,
        note=note,
    )


def run(args: argparse.Namespace) -> Tuple[List[Measurement], Optional[str]]:
    torch = require_torch()
    torch_tensorrt = require_torch_tensorrt()
    device = _torch_device(args.device, torch)

    if args.tflop is not None:
        flops = float(args.tflop)
        flop_basis = f"TFLOP from --tflop ({flops:.3f} TFLOP, given)"
    else:
        flops = derive_tflops(args.width, args.height, args.scale)
        flop_basis = (
            f"derived: {flops:.3f} TFLOP from {args.width}x{args.height} @x{args.scale} "
            f"(head 3->{BODY_CHANNELS} + {BODY_CONVS}x {BODY_CHANNELS}->{BODY_CHANNELS} + "
            f"tail ->{3 * args.scale * args.scale}, MODEL.md §1)"
        )

    # An engine is built for one precision; when its file name says which (the app writes
    # _fp16/_fp32 there), that decides the input dtype rather than --dtype, so a mismatch
    # is a printed warning instead of an op-level error from the engine.
    name_dtype = dtype_from_engine_name(args.engine) if args.engine else None
    dtype_label = name_dtype or torch_dtype_name(args.dtype)
    if name_dtype:
        dtype_basis = "from the engine file name"
        if name_dtype != torch_dtype_name(args.dtype):
            print(
                f"bench_engine: {args.engine} is named {name_dtype}; feeding {name_dtype} "
                f"inputs (--dtype {args.dtype} ignored for this engine)",
                file=sys.stderr,
            )
    else:
        dtype_basis = "from --dtype (fed to the module)" if args.engine else "build flag"

    sample = torch.zeros(
        (1, 3, args.height, args.width),
        dtype=_torch_dtype(dtype_label, torch),
        device=device,
    )

    if args.engine:
        module = load_engine_module(args.engine, device, torch)
        source = f"engine {args.engine}"
        tiling = tiling_from_engine_name(args.engine) or "unknown (loaded plan)"
        tiling_basis = "from the engine file name" if tiling_from_engine_name(args.engine) else ""
    else:
        module = build_engine_module(args.model, sample, args, torch, torch_tensorrt)
        source = f"built from {args.model}"
        tiling = args.tiling
        tiling_basis = "build flag"

    rows: List[Measurement] = []
    previous_ms: Optional[float] = None
    sampler: Any = ClockSampler(args.sample_clocks) if args.sample_clocks else contextlib.nullcontext()
    with sampler:
        for repeat in range(args.repeat):
            label = "quiet" if args.repeat == 1 else f"quiet #{repeat + 1}/{args.repeat}"
            row = _measure(
                args, label, module, sample, previous_ms, len(rows) + 1,
                source=source, tiling=tiling, tiling_basis=tiling_basis,
                dtype_label=dtype_label, dtype_basis=dtype_basis,
                flops=flops, flop_basis=flop_basis,
            )
            rows.append(row)
            previous_ms = row.ms_per_frame

        if args.with_nvenc:
            width, height = parse_size(args.nvenc_size)
            quiet_ms = previous_ms
            with NvencAlongside(args.nvenc_ffmpeg, (width, height)) as encoder:
                row = _measure(
                    args, f"+nvenc {width}x{height} hevc_nvenc", module, sample,
                    quiet_ms, len(rows) + 1,
                    source=source, tiling=tiling, tiling_basis=tiling_basis,
                    dtype_label=dtype_label, dtype_basis=dtype_basis,
                    flops=flops, flop_basis=flop_basis,
                    note=(
                        f"an ffmpeg hevc_nvenc encode of a synthetic {width}x{height} "
                        "10-bit raw stream from /dev/zero ran alongside; compare with the "
                        "last quiet row"
                    ),
                )
                rows.append(row)
                ran_alongside = encoder.alive
            if not ran_alongside:
                rows[-1].note += " - WARNING: ffmpeg exited during the window, ratio invalid"
                if encoder.stderr:
                    print(f"bench_engine: ffmpeg stderr: {encoder.stderr}", file=sys.stderr)

    dump_text: Optional[str] = None
    if args.dump_tactics:
        if not args.model:
            print(
                "bench_engine: --dump-tactics needs --model: profiling verbosity is a "
                "build-time flag, a loaded plan cannot be re-inspected at DETAILED",
                file=sys.stderr,
            )
        else:
            dump_text = dump_tactics(args.model, sample, args)
    return rows, dump_text


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        rows, dump_text = run(args)
    except BenchError as exc:
        print(f"bench_engine: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary_json(rows), indent=2))
    else:
        print(format_summary(rows))
    if dump_text:
        print(dump_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
