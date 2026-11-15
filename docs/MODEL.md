# Model report: Real-ESRGAN compact on this machine

Scope: how fast the network itself runs, why it cannot be made much faster per frame, and
what the measurements actually say. Pipeline stages (colour packing, encoder) are only cited
where they explain a number.

Rig: RTX 4060 Ti 16 GB (AD106, 34 SM), Ryzen 5 5600, TensorRT 10.12.0.36, torch 2.9.0+cu130,
torch_tensorrt 2.9.0. Engines: fp16, static shape per resolution, `optimization_level=5`,
`num_avg_timing_iters=8`, `tiling_optimization_level=moderate`.

## 1. What the model computes

`SRVGGNetCompact` ("Real-ESRGAN anime video v3" and friends): a plain VGG-style stack, all
convolutions at the LOW-resolution size, upsampling only in the tail.

```
x -> clamp(0,1) -> conv3x3(3 -> 64) -> PReLU
  -> 16 x [ conv3x3(64 -> 64) -> PReLU ]
  -> conv3x3(64 -> 3*scale^2) -> PixelShuffle(scale)
  -> + nearest_upsample(x, scale)
```

Cost per frame (2.07 Mpx, i.e. 1920x1080):

| part | GFLOP | share |
|---|---|---|
| head 3 -> 64 | 7.2 | 0.3% |
| 16 body convs 64 -> 64 | 2440 | 98.3% |
| tail 64 -> 3*scale^2 (48 for x4, 12 for x2) | 28.7 / 114 | 1.2% / 4.5% |
| **total** | **~2.48 TFLOP (x2), ~2.56 (x4)** | |

Consequences worth knowing before optimising anything:

- **Scale is almost free.** The x2 and x4 variants cost the same, because all the work happens
  at the input resolution; only the last conv and the upsampler differ. A 4x model used at a
  2x output does ~1 for the body and throws the extra tail away - that is what `FusedTail.py`
  in this fork is for.
- **The body is 98% of the work**, so any per-frame optimisation has to attack 16 identical
  `64 -> 64, 3x3` convolutions. There is nothing else to tune.
- Receptive radius is 1 + 16 + 1 = 18 LR pixels. That number is why tile reuse needs an 18 px
  halo and why a 512 px tile really runs on 548 px.

## 2. Measured speed

| model | input | engine | ms/frame | fps |
|---|---|---|---|---|
| animevideov3 x4 | 1920x1080 | opt3, iters4, no tiling (old defaults) | **94.3** | 10.6 |
| animevideov3 x4 | 1920x1080 | opt5, iters8, no tiling | 57.5 | 17.4 |
| animevideov3 x4 | 1920x1080 | opt5, iters8, tiling moderate | **52.7** | 19.0 |
| 2x Futsuu Anime | 1920x1080 | opt5, iters8, tiling moderate | **45.7** | 21.9 |
| 2x Futsuu Anime | 1440x1080 | same | **35.1** | 28.5 |

Notes:

- Build flags alone give **1.79x** on the same weights (94.3 -> 52.7). Nothing about the model
  changed: TRT was simply picking bad tactics at level 3 and not keeping layer tiles in L2.
- Run-to-run spread is about +-15% (clocks, power limit at 165 W). Quote ratios, not absolute
  milliseconds, unless the two numbers come from the same session.
- The x2 model is faster than the x4 one only in the tail (45.7 vs 52.7 ms = 13%), which matches
  the FLOP split above.

## 3. Where those milliseconds go

2.48 TFLOP in 45.7 ms = **54 TFLOP/s**.

The card's fp16 tensor throughput is ~44 TFLOPS with fp32 accumulation and ~88 TFLOPS with
fp16 accumulation, so the engine is at 61% of the fp16-accumulate peak (and above the
fp32-accumulate peak, i.e. it already uses fp16 accumulation somewhere). A perfect kernel could
therefore be ~1.6x faster in theory. Everything tried to reach that 1.6x measures at ~1.0x:

| technique | why it does not help | evidence |
|---|---|---|
| FP8 (E4M3) | Ada FP8 conv kernels are not faster than fp16 for these shapes; TRT needs Q/DQ nodes, not a flag | NVIDIA TensorRT issue #4081: FP8 ConvNet on a 4090 ~1.0x, same kernels on L40S 1.7-1.9x |
| INT8 PTQ | dequant/reformat layers cost more than the int8 convs save at C=64 | mmdeploy, same 16-conv class: fp16 12.42 ms vs int8 12.45 ms |
| 2:4 structured sparsity | needs pruned weights (retraining) and TRT rarely selects sparse tactics at C=64 | TRT docs + NVIDIA engineer note: <10% for conv nets under 256 channels |
| Winograd | fp16 transforms overflow the range (ncnn disabled fp16 Winograd for exactly this) and the fp32 transforms eat the 2.25x | ncnn source, 2020 benchmarks |
| hand-written kernels | TRT's implicit-GEMM tactic beats tuned plugins | CORE-TAG: 39.8 TFLOP/s plugin vs 186.7 engine on a 5090 |
| CUDA graphs | only removes launch overhead | ~0.3% at 45 ms/frame |
| batching / multiple streams | the GPU is already 96-100% busy | 0%, and separate contexts corrupted the output |

### Why this architecture is a hard shape

- **Arithmetic intensity sits on the ridge.** One body layer at 1080p reads+writes ~530 MB of
  fp16 activations for ~288 FLOP per byte, and the card's ridge point is ~305 FLOP/byte. The
  kernel therefore alternates between compute-bound and bandwidth-bound; a fix for one side
  loses on the other.
- **C = 64 is small.** NVIDIA's own convolution guide asks for channels divisible by 64 and
  "ideally 256" to keep tensor-core tiling efficient. With C=K=64 the implicit-GEMM K dimension
  is only 576 and the tile grid is short, so tail effects are a permanent tax.
- **18 kernels back to back**, each with its own launch and its own full-tensor round trip.
  Fusing across layers is not possible because PReLU sits between them (element-wise, non-linear).

## 4. The only lever that works: compute fewer frames

Because the per-frame cost is at its practical floor, the wins came from not calling the model:

| scenario | frames run through the model | measured |
|---|---|---|
| ungated, 1080p -> 2160p, anime clip (502 frames) | 100% | 16.3 fps |
| duplicate gate, same clip | **27.3%** | **43.7 fps** (1.79x), output median 53 dB / worst 48.1 dB vs ungated |
| ungated, film clip (94 frames) | 100% | 19-22 fps |
| duplicate gate, film clip | 53.2% | 23.7 fps, 71 dB vs ungated |

So the model does 45.7 ms of work on the frames it sees, but the *pipeline* pays
45.7 x 0.273 = **12.5 ms per output frame** on anime. Further gains must come from the
remaining 27% of frames (tile-level reuse, which needs mostly-static content to beat its
18 px halo overhead), not from making the 16 convolutions faster.

## 5. How to reproduce

```bash
# isolated model speed, no pipeline
python - <<'PY'
import sys, time, torch
sys.path.insert(0, "<fork>/backend")
from src.pytorch.UpscaleTorch import UpscalePytorch
up = UpscalePytorch(modelPath="<model>.pth", device="cuda", precision="auto",
                    width=1920, height=1080, backend="tensorrt",
                    trt_optimization_level=5, trt_static_shape=True)
x = torch.rand(1, 3, 1080, 1920, device="cuda", dtype=torch.float16)
with torch.inference_mode():
    for _ in range(5): up.upscale_model_wrapper(x)
    torch.cuda.synchronize(); t = time.time(); n = 0
    while time.time() - t < 10: up.upscale_model_wrapper(x); n += 1
    torch.cuda.synchronize()
print(f"{n/10:.1f} fps  {10000/n:.1f} ms/frame")
PY

# engine tactic audit (the one experiment still worth running)
trtexec --loadEngine=<engine file> --dumpProfile --dumpLayerInfo
```

## 6. Open question

The 1.6x theoretical gap has never been *localised*. `trtexec --dumpProfile` would show which
tactic each convolution got and how much time it takes; if any layer is running a non-tensor-
core tactic or paying a reformat, that is the only place a per-frame win could still hide. Until
that is measured, the honest statement is: 61% of the fp16-accumulate peak, every known
technique measured at ~1.0x, and the productive direction is fewer frames, not faster ones.
