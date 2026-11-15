# Model report: Real-ESRGAN compact on this machine

Scope: how fast the network itself runs, what is actually measured versus read from elsewhere,
and what remains unverified. Pipeline stages (colour packing, encoder) appear only where they
explain a number.

Rig: RTX 4060 Ti 16 GB (AD106, 34 SM, 165 W cap), Ryzen 5 5600, TensorRT 10.12.0.36,
torch 2.9.0+cu130. Engines: fp16, static shape per resolution, `optimization_level=5`,
`num_avg_timing_iters=8`, `tiling_optimization_level=moderate` unless stated.

> Status of the conclusion in this file: **the per-frame cost is not proven to be at a floor.**
> What is proven is narrower - none of the techniques *we ran locally* helped, and several
> widely-cited techniques were never reproduced here. Section 5 lists the measurements that
> would settle it; Section 6 lists what is read-not-measured.

## 1. What the model computes

`SRVGGNetCompact` ("Real-ESRGAN anime video v3" and friends): a plain VGG-style stack, every
convolution at the LOW-resolution size, upsampling only in the tail.

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
| 16 body convs 64 -> 64 | 2446 | 98.3% |
| tail 64 -> 3*scale^2 (12 for x2, 48 for x4) | 28.7 / 114.7 | 1.2% / 4.5% |
| **total** | **~2.48 TFLOP (x2), ~2.57 (x4)** | |

Consequences worth knowing before optimising anything:

- **Scale is cheap in arithmetic, not necessarily in bandwidth.** The x2 and x4 variants cost
  the same in FLOPs (3.5% apart) because all the work happens at the input resolution; but the
  x4 tail writes a 199 MB fp16 output against 49.8 MB for x2, so a tail that is cheap on paper
  can still be memory-bound in practice.
- **The body is 98% of the arithmetic**, so a per-frame optimisation has to attack 16
  identical `64 -> 64, 3x3` convolutions.
- Receptive radius is 1 + 16 + 1 = 18 LR pixels; that is why tile reuse needs an 18 px halo and
  why a 512 px tile really runs on 548 px of work.

## 2. Measured speed (this machine)

| model | input | engine | ms/frame | fps |
|---|---|---|---|---|
| animevideov3 x4 | 1920x1080 | opt3, iters4, no tiling (old defaults) | **94.3** | 10.6 |
| animevideov3 x4 | 1920x1080 | opt5, iters8, no tiling | 57.5 | 17.4 |
| animevideov3 x4 | 1920x1080 | opt5, iters8, tiling moderate | **52.7** | 19.0 |
| 2x Futsuu Anime | 1920x1080 | opt5, iters8, tiling moderate | **45.7** | 21.9 |
| 2x Futsuu Anime | 1440x1080 | same | **35.1** | 28.5 |

- Build flags alone give **1.79x** on identical weights (94.3 -> 52.7). This is the single
  strongest evidence that "the model has no headroom" is the wrong prior: the same model had
  44% more headroom until TRT was told to search tactics harder and keep tiles in L2.
- Run-to-run spread is roughly +-15% (clocks, 165 W cap). Compare ratios from the same session,
  not absolute milliseconds across sessions.
- The x2-vs-x4 difference is **7 ms while the FLOP difference is only 3.5% (~1.6 ms)**. That is
  either inside the noise band or the x4 tail is memory-bound (199 MB vs 49.8 MB of output).
  It has not been separated, so "only the 16 convs matter" is an assumption, not a result.

## 3. What the milliseconds correspond to

2.48 TFLOP in 45.7 ms = **54 TFLOP/s**.

Peaks on this card are ~44 TFLOPS for fp16 with fp32 accumulation and ~88 TFLOPS with fp16
accumulation, but those are boost-clock whitepaper numbers. Two corrections matter:

1. **Sustained, power-limited**: at the 165 W cap an independent measurement of this card gives
   ~46.9 TF/s for fp16->fp32 accumulate and ~93.8 TF/s for reduced precision; a realistic
   sustained fp16-acc figure is ~74 TF/s.
2. **The tiling engine does more work than the frame**: a 512 px tile runs on 548 px (+14.5%),
   so the effective rate is higher than the nominal one.

With both corrections the model sits at roughly **73-84% of what is achievable**, i.e. the
remaining per-frame headroom is about **1.2-1.35x**, not the 1.6x a whitepaper ratio suggests.

## 4. Techniques: measured here vs read elsewhere

The distinction matters. Rows marked *read* are other people's benchmarks on other hardware and
other shapes; they are hypotheses about our case, not results.

| technique | we ran it here? | result | source / caveat |
|---|---|---|---|
| builder opt level 5 + timing iters 8 | **yes** | 94.3 -> 57.5 ms (1.64x) | our run |
| L2 tiling (`moderate`) | **yes** | 57.5 -> 52.7 ms (1.09x) | our run; this is cross-kernel fusion and it works, contrary to an earlier claim in this file |
| CUDA graphs | **yes** | ~0.3% | our run |
| batching / parallel engine contexts | **yes** | 0%, and our two-context wiring produced corrupt output | our run; the corruption was an implementation fault (two `torch.jit.load` modules are not independent contexts), not a GPU property |
| INT8 PTQ on this model at 1080p | **no** | unknown | the commonly cited 12.42 vs 12.45 ms is ESRGAN/RRDB at **32x32 on a T4 with TRT 7.2** - launch-overhead bound and a different architecture, not transferable |
| FP8 conv | **no** | unknown for us | NVIDIA issue #4081 reports ~1.0x for a ConvNet on a 4090, but the same thread reports **2x for int8 on the 4090** and 2x for both on L40S; GeForce parts are capped in firmware |
| 2:4 structured sparsity | **no** | needs retrained sparse weights | TRT docs; NVIDIA engineers report small gains for conv nets under 256 channels |
| Winograd | **no** | expected loss here | F(2,3) cuts MACs 2.25x but inflates tensors ~4x, dropping arithmetic intensity below the ridge; the fp16-overflow argument is from ncnn's Vulkan backend, not from this GPU |
| hand-written kernels | **no** | slower in the one public attempt | CORE-TAG: 39.8 TFLOP/s plugin vs 186.7 engine on a 5090 |
| accumulator precision (fp32acc vs fp16acc share) | **no** | unknown - the most likely remaining lever | kernel names expose it (`...fp16mul_fp32acc` vs `...fp16mul_fp16acc`); never inspected here |
| `tiling=full` + `l2_limit_for_tiling` sweep | **no** | unknown | `moderate` already gave 9%, and the L2 of this part is 32 MB |
| distillation / channel pruning (16 -> 8 convs, 64 -> 48) | **no** | 1.5-2x in the literature | this is retraining and was declared out of scope by the owner, which is a scope decision, not a hardware limit |

## 5. Audit log (what has been measured so far)

| item | result |
|---|---|
| our own GEMM peak, fp16 with fp32 accumulate | **42.1 TF/s** (torch matmul, 8192^2); the `allow_fp16_reduced_precision_reduction` flag does not change it (42.5), because it controls split-k reduction, not the accumulator |
| model rate | **49.4 TF/s** with tiling off, i.e. **above** the fp32-accumulate GEMM peak - so fp16-accumulate kernels are already in use for part of the network |
| engine built straight from ONNX (no torch-tensorrt), tiling off | 50.27 ms |
| layer structure with tiling on | only 25 top-level layers, all `__myl_*`: TRT fuses the network into myelin subgraphs, i.e. cross-kernel fusion is happening (this is the mechanism that gave 57.5 -> 52.7 ms) |
| per-layer tactic names | read with `profiling_verbosity=DETAILED` at build time (see below) |
| `tiling=full` vs `moderate`, same build path | **full is faster**, 3.8% in the app (two runs each: 11.01 vs 10.60/10.59 s on 502 anime frames) and 2-6% in the ONNX audit path |
| INT8 PTQ (entropy calibration on 8 real frames, INT8+FP16 fallback) | **slower than fp16 in every run**: 73.2 and 94.3 ms against 48-51 ms. Dequant/reformat at C=64 costs more than int8 saves - the "2x for int8 on a 4090" note from issue #4081 does not transfer to this shape |
| `l2_limit_for_tiling = 24 MiB` | **harmful**: 67.4 and 103.2 ms against 48.6. TRT's own default budget is better; leave the knob alone |

### The accumulator question is settled

A DETAILED build (tiling full, fp16) reports the tactic for every convolution:

```
16x  sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32
 1x  sm75_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16
 1x  sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16
```

`f16f16_f16f16_f16` = fp16 inputs, **fp16 accumulator**. So all 16 body convolutions already run
with fp16 accumulation: there is no fp32-accumulate share to convert, and the "switch the
accumulator for ~1.4x" hypothesis is closed. What remains is efficiency against a sustained
~74 TF/s (we are at ~51-54), and the only measured lever in that space is `tiling=full`
(+3.8%), not precision.

Also visible: the body uses `tilesize256x64x32`, i.e. a K tile of 32 for C=64 - two K steps per
convolution, which is where the C=64 penalty shows up numerically.

Caveat on the audit's absolute milliseconds: those engine builds overlapped each other on the
card (see the measurement protocol in RESEARCH-LEDGER.md), so only the *ratios* and the
direction of each effect are trustworthy; the app A/B above was measured on a quiet GPU.

Consequence for the accumulator hypothesis: switching "everything" to fp16 accumulation is
already partly done, so the realistic gap is from ~49 TF/s to a sustained ~74 TF/s, i.e.
**~1.3-1.5x**, and it can only be claimed or refuted by a DETAILED dump plus an INT8 A/B.

## 6. Measurements that would settle it (hours, not weeks)

1. **Per-layer tactic dump** (`trtexec --loadEngine --dumpProfile --dumpLayerInfo`, or
   `IEngineInspector` from Python). Look for three things: `fp32acc` in kernel names (a 50/50
   mix would explain the observed 54 TFLOP/s exactly), `Reformat` layers, and whether PReLU is
   fused into the convolution epilogue.
2. **Our own GEMM peaks** for fp16->fp32 and fp16->fp16 accumulate, so "% of peak" is measured
   rather than quoted.
3. **INT8 Q/DQ engine on this model at 1080p**, with PSNR against fp16 on ~20 real frames.
4. **`tiling=full` and an `l2_limit_for_tiling` sweep.**
5. `nsys` tensor-active % and DRAM throughput with clocks pinned (`nvidia-smi -lgc`).

If none of those yields more than 5%, the floor is proven rather than assumed.

## 7. The lever that does work today: fewer frames

Independent of the above, the largest measured win is not calling the model:

| scenario | frames through the model | measured |
|---|---|---|
| ungated, 1080p -> 2160p, anime (502 frames) | 100% | 16.3 fps |
| duplicate gate | **27.3%** | **43.7 fps** (1.79x), median 53 dB / worst frame 48.1 dB vs ungated |
| ungated, film (94 frames) | 100% | 19-22 fps |
| duplicate gate, film | 53.2% | 23.7 fps, 71 dB vs ungated |

So the model does 45.7 ms on the frames it sees, but the pipeline pays 45.7 x 0.273 =
**12.5 ms per output frame** on anime. The remaining 27% of frames is where the next
"do less work" win lives (tile-level reuse), not in the 16 convolutions.

## 8. Reproduce

See `docs/PERFORMANCE.md` for the pipeline-level commands. Isolated model timing: build the
engine through `UpscalePytorch` with `trt_static_shape=True` and time
`upscale_model_wrapper(x)` over 25 iterations after 6 warmup calls.
