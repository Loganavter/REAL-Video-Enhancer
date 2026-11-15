"""A TensorRT plan built by the raw builder, wrapped as a module.

Why this exists: torch_tensorrt.dynamo.compile does not preserve the myelin fusion that the head
padding unlocks (ledger E19). The same graph built by `trt.Builder` from ONNX is 5.8% faster
(57.51 -> 54.15 ms) and keeps 17 fused subgraphs where the app's path turns the padding into a 71%
regression. This module is the seam that lets the app build and run that plan instead.

It is deliberately narrow: one input, one output, shapes fixed per resolution, the stream is torch's
current stream, and buffers are cached per shape so a render does not allocate a 4K output every
frame.
"""
import collections

import torch
import tensorrt as trt

from ..utils.Util import log

_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}

_logger = trt.Logger(trt.Logger.ERROR)


class RawTRTEngine(torch.nn.Module):
    """Wraps a serialized TensorRT plan in the module interface the pipeline expects."""

    def __init__(self, plan_path: str):
        super().__init__()
        self.plan_path = plan_path
        self._engine = None
        self._context = None
        self._input_name = None
        self._output_name = None
        self._input_dtype = None
        self._output_dtype = None
        self._output_shape = None
        self._output = None

    # -- lazy, so building the module in one process and running it in another both work
    def _ensure(self):
        if self._engine is not None:
            return
        with open(self.plan_path, "rb") as handle:
            engine = trt.Runtime(_logger).deserialize_cuda_engine(handle.read())
        if engine is None:
            raise RuntimeError(f"could not deserialize {self.plan_path}")
        self._engine = engine
        self._context = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self._input_name = next(n for n in names
                                if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self._output_name = next(n for n in names
                                 if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self._input_dtype = _DTYPES[engine.get_tensor_dtype(self._input_name)]
        self._output_dtype = _DTYPES[engine.get_tensor_dtype(self._output_name)]
        self._output_shape = tuple(engine.get_tensor_shape(self._output_name))
        log(f"raw TensorRT engine: {self._input_dtype} {tuple(engine.get_tensor_shape(self._input_name))}"
            f" -> {self._output_dtype} {self._output_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure()
        if x.dtype != self._input_dtype:
            x = x.to(self._input_dtype)
        x = x.contiguous()
        if self._output is None or tuple(self._output.shape) != self._output_shape:
            self._output = torch.empty(self._output_shape, device=x.device,
                                       dtype=self._output_dtype)
        self._context.set_tensor_address(self._input_name, x.data_ptr())
        self._context.set_tensor_address(self._output_name, self._output.data_ptr())
        stream = torch.cuda.current_stream(x.device).cuda_stream
        self._context.execute_async_v3(stream)
        return self._output


def build_plan(onnx_path: str, plan_path: str, fp16: bool = True, tiling: str = "full",
               optimization_level: int = 5) -> str:
    """Build a plan from an ONNX file with the raw builder - the path that keeps the fusion."""
    levels = {"none": trt.TilingOptimizationLevel.NONE,
              "moderate": trt.TilingOptimizationLevel.MODERATE,
              "full": trt.TilingOptimizationLevel.FULL}
    builder = trt.Builder(_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, _logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            for index in range(parser.num_errors):
                log(f"onnx parse: {parser.get_error(index)}")
            raise RuntimeError(f"raw TensorRT could not parse {onnx_path}")
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.tiling_optimization_level = levels.get(tiling, trt.TilingOptimizationLevel.FULL)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("raw TensorRT build failed")
    with open(plan_path, "wb") as handle:
        handle.write(bytes(blob))
    return plan_path


def export_onnx(model: torch.nn.Module, example_inputs, onnx_path: str) -> str:
    """Export the module at one static shape - the shape the engine is for."""
    example = example_inputs[0] if isinstance(example_inputs, (tuple, list)) else example_inputs
    with torch.inference_mode():
        torch.onnx.export(model, example, onnx_path, opset_version=17,
                          input_names=["input"], output_names=["output"],
                          do_constant_folding=True, dynamo=False)
    return onnx_path
