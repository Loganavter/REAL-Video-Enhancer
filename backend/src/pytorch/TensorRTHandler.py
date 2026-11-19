"""
MIT License

Copyright (c) 2024 TNTwise

cPermission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
from ..utils.Util import suppress_stdout_stderr, warnAndLog, log
from ..version import __version__
import time
with suppress_stdout_stderr():
    import torch
    import torch_tensorrt
    import tensorrt as trt
    from torch._export.converter import TS2EPConverter
    from torch.export.exported_program import ExportedProgram
    from .TorchUtils import TorchUtils

def _normalize_example_inputs(example_inputs):
    # Accept torch.Tensor or sequence of tensors; always return a tuple of args
    if isinstance(example_inputs, torch.Tensor):
        return (example_inputs,)
    # If already a list/tuple
    if len(example_inputs) == 1 and isinstance(example_inputs[0], torch.Tensor):
        return (example_inputs[0],)
    return tuple(example_inputs)

def torchscript_to_dynamo(
    model: torch.ScriptModule, example_inputs
) -> ExportedProgram:
    """Converts a TorchScript module to a Dynamo program."""
    sample_args = _normalize_example_inputs(example_inputs)
    traced = torch.jit.trace(model, sample_args)
    return TS2EPConverter(traced, sample_args=sample_args, sample_kwargs=None).convert()

def nnmodule_to_dynamo(
    model: torch.nn.Module, example_inputs: list[torch.Tensor], dynamic_shapes=None
) -> ExportedProgram:
    """Converts a nn.Module to a Dynamo program."""
    return torch.export.export(
        model, tuple(example_inputs), dynamic_shapes=dynamic_shapes
    )

"""onnx_support = True
try:
    import onnx
except ImportError:
    onnx_support = False"""


class TorchTensorRTHandler: 
    """
    Args:
        trt_workspace_size (int): The workspace size to use when compiling models using TensorRT. Defaults to 0.
        max_aux_streams (int | None): The maximum number of auxiliary streams to use when compiling models using TensorRT. Defaults to None.
        trt_optimization_level (int): The optimization level to use when compiling models using TensorRT. Defaults to 5.
        debug (bool): Whether to enable debugging when compiling models using TensorRT. Defaults to False.

        Level 5 is not a taste: the default level 3 leaves every PReLU as its own pass, which measured
        78.33 ms per frame against 47.50 at level 5 on the same ONNX and the same card
        (docs/wsp/verified/fp16-engine-layers.txt). Every caller passes this value today, so this default
        is a fallback - but a fallback that silently costs a third of the frame is the wrong one.


        multi precision engines seem to not like torchscript2exportedprogram,
        or maybe its just the model not playing nice with explicit_typing,
        either way, forcing one precision helps with speed in some cases.
    """

    trt_path_appendix = (f"_{__version__}.engine") # this is used to identify the models that were exported with this version of RVE

    def __init__(
        self,
        model_parent_path: str,
        max_aux_streams: int | None = None,
        debug: bool = False,
        trt_optimization_level: int = 5,
        trt_workspace_size: int = 0,
        
    ):
        self.tensorrt_version = trt.__version__  # can just grab version from here instead of importing trt and torch trt in all related files
        self.torch_tensorrt_version = torch_tensorrt.__version__

        self.trt_workspace_size = trt_workspace_size
        self.max_aux_streams = max_aux_streams
        self.optimization_level = trt_optimization_level
        self.debug = debug
        self.model_parent_path = model_parent_path
        # clear previous tensorrt models
        cleared_models = False
        if os.path.exists(self.model_parent_path):
            for model in os.listdir(self.model_parent_path):
                
                if not self.trt_path_appendix.lower() in model.lower() and "tensorrt" in model.lower():
                    model_path = os.path.join(self.model_parent_path, model)
                    try:
                        os.remove(model_path)
                        cleared_models = True
                        log(f"Removed {model_path}")
                    except Exception as e:
                        log(f"Failed to remove {model_path}: {e}")
            if cleared_models:
                print("Cleared old TensorRT models...", file=sys.stderr)
    
    

    def grid_sample_decomp(self, exported_program):
        from torch_tensorrt.dynamo.conversion.impl.grid import GridSamplerInterpolationMode
        GridSamplerInterpolationMode.update(
            {
                0: trt.InterpolationMode.LINEAR,
                1: trt.InterpolationMode.NEAREST,
                2: trt.InterpolationMode.CUBIC,
            }
        )
        return exported_program

    def check_engine_exists(self, trt_engine_name: str) -> bool:
        """Checks if a TensorRT engine exists at the specified path."""
        trt_engine_name += self.trt_path_appendix
        trt_engine_path = os.path.join(self.model_parent_path, trt_engine_name)
        if os.environ.get("RVE_RAW_TRT", "0") == "1":
            return os.path.exists(trt_engine_path + ".plan")
        return os.path.exists(trt_engine_path)

    def raw_trt_enabled(self) -> bool:
        """Build with the raw TRT builder instead of torch_tensorrt (ledger E19)."""
        return os.environ.get("RVE_RAW_TRT", "0") == "1"

    def build_engine(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        example_inputs: list[torch.Tensor],
        trt_engine_name: str,
        trt_multi_precision_engine: bool = False,
        dynamic_shapes: dict | None = None,
        num_avg_timing_iters: int = 8,
        tiling_optimization_level: str = "moderate",
    ):
        """
        Returns a Torch TensorRT engine built from the provided model.
        """
        start_time = time.time()
        trt_engine_name += self.trt_path_appendix
        TorchUtils.clear_cache()
        # The raw builder is faster on this graph (ledger E22: 11.01 s against 11.92 s on the same
        # clip, picture bit-identical). Engines are static per resolution by default (tensorrt_
        # dynamic_shapes is off), which is exactly the shape this path exports; the opt-in dynamic
        # mode is skipped, and any failure falls through to torch_tensorrt rather than costing a
        # render.
        if self.raw_trt_enabled() and dynamic_shapes is None:
            try:
                from .RawTRTEngine import RawTRTEngine, build_plan, export_onnx
                engine_path = os.path.join(self.model_parent_path, trt_engine_name)
                onnx_path = engine_path + ".onnx"
                export_onnx(model, example_inputs, onnx_path)
                build_plan(onnx_path, engine_path + ".plan", fp16=(dtype == torch.float16),
                           tiling=tiling_optimization_level,
                           optimization_level=self.optimization_level)
                print(f"TensorRT engine built in {time.time() - start_time:.2f} seconds "
                      f"(raw builder).", file=sys.stderr)
                TorchUtils.clear_cache()
                return RawTRTEngine(engine_path + ".plan")
            except Exception as error:
                log(f"raw TensorRT path failed ({error}); falling back to torch_tensorrt")
                for stale in (engine_path + ".plan", engine_path + ".onnx"):
                    if os.path.exists(stale):
                        os.remove(stale)
        elif self.raw_trt_enabled():
            log("raw TensorRT path exports one static shape; this engine asked for dynamic "
                "(tensorrt_dynamic_shapes), so torch_tensorrt builds it")
        """Builds a TensorRT engine from the provided model."""
        print(
            f"Building TensorRT engine {os.path.basename(trt_engine_name)}. This may take a while...",
            file=sys.stderr,
        )
        
        with suppress_stdout_stderr():
            exported_program = nnmodule_to_dynamo(model, example_inputs, dynamic_shapes=dynamic_shapes)
            TorchUtils.clear_cache()

            exported_program = self.grid_sample_decomp(exported_program)

            compile_kwargs = dict(
                device=device,
                enabled_precisions={dtype} if not trt_multi_precision_engine else {torch.float},
                use_explicit_typing=trt_multi_precision_engine,
                debug=self.debug,
                num_avg_timing_iters=num_avg_timing_iters,
                workspace_size=self.trt_workspace_size,
                min_block_size=1,
                max_aux_streams=self.max_aux_streams,
                optimization_level=self.optimization_level,
            )

            # Tiling keeps per-layer intermediates in L2 between consecutive kernels.
            # On convolution-heavy SR nets this is worth ~10% (measured on RTX 4060 Ti:
            # 57.5 ms -> 52.7 ms at 1080p for realesr-animevideov3). Some graphs do not
            # support it (the builder fails), so fall back to the plain build.
            model_trt = None
            if tiling_optimization_level and tiling_optimization_level != "none":
                try:
                    model_trt = torch_tensorrt.dynamo.compile(
                        exported_program,
                        tuple(example_inputs),
                        tiling_optimization_level=tiling_optimization_level,
                        **compile_kwargs,
                    )
                except Exception as e:
                    log(
                        f"TensorRT tiling optimization '{tiling_optimization_level}' failed "
                        f"({e}), rebuilding without it"
                    )
                    model_trt = None
            if model_trt is None:
                model_trt = torch_tensorrt.dynamo.compile(
                    exported_program,
                    tuple(example_inputs),
                    **compile_kwargs,
                )
        
        # How many TRT sub-engines the partitioner produced. One means the whole graph is in TRT;
        # more than one means something stayed in torch between them, which costs far more than the
        # ops themselves (ledger E23: the padded head measured 71% slower and this is the number that
        # says whether that is the graph or the partition).
        engines = getattr(model_trt, "_trt_engines", None)
        if engines is not None:
            log(f"TensorRT partitioning: {len(engines)} sub-engine(s)")
        print(
            f"TensorRT engine built in {time.time() - start_time:.2f} seconds.",
            file=sys.stderr,
        )
        TorchUtils.clear_cache()
        return model_trt

    def save_engine(self, trt_engine: torch.jit.ScriptModule, trt_engine_name: str, example_inputs: list[torch.Tensor]):
        """Saves a TensorRT engine to the specified path."""
        trt_engine_name += self.trt_path_appendix
        trt_engine_path = os.path.join(self.model_parent_path, trt_engine_name)

        from .RawTRTEngine import RawTRTEngine
        if isinstance(trt_engine, RawTRTEngine):
            # the plan is already on disk next to the engine name; nothing to serialise
            log(f"raw TensorRT plan kept at {os.path.basename(trt_engine.plan_path)}")
            return
        torch_tensorrt.save(
                trt_engine,
                trt_engine_path,
                output_format="torchscript",
                inputs=tuple(example_inputs),
        )
        TorchUtils.clear_cache()

    def load_engine(self, trt_engine_name: str) -> torch.jit.ScriptModule:
        """Loads a TensorRT engine from the specified path."""
        
        trt_engine_name += self.trt_path_appendix
        trt_engine_path = os.path.join(self.model_parent_path, trt_engine_name)
        if self.raw_trt_enabled() and os.path.exists(trt_engine_path + ".plan"):
            from .RawTRTEngine import RawTRTEngine
            print(f"Loading raw TensorRT plan from {trt_engine_path}.plan.", file=sys.stderr)
            return RawTRTEngine(trt_engine_path + ".plan")
        print(f"Loading TensorRT engine from {trt_engine_path}.", file=sys.stderr)
        loaded = torch.jit.load(trt_engine_path).eval()
        engines = getattr(loaded, "_trt_engines", None)
        if engines is not None:
            log(f"TensorRT partitioning (loaded): {len(engines)} sub-engine(s)")
        return loaded

