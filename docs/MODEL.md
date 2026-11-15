# Model report: Real-ESRGAN compact on this machine

Scope: how fast the network itself runs, what was measured here versus read elsewhere, and what
is still open. Pipeline stages appear only where they explain a number.

Rig: RTX 4060 Ti 16 GB (AD106, 34 SM, 165 W cap), Ryzen 5 5600, TensorRT 10.12.0.36,
torch 2.9.0+cu130. Unless stated: fp16 engines, static shape, `optimization_level=5`,
`num_avg_timing_iters=8`, `tiling_optimization_level=full` (the new default).

**Conclusion up front**: the per-frame path has no large wins left that we could measure, and
the one real win of this round is `tiling=full` (~6.6% on the model, 3.8% end-to-end). The
remaining gap to a theoretical ceiling is small and **not localised**; one concrete lead from
the tactic dump (Turing-generation kernels selected on an Ada part) is listed in §6 and has not
been chased.

## 1. What the model computes

`SRVGGNetCompact` ("Real-ESRGAN anime video v3" and friends): a plain VGG stack, every
convolution at the LOW-resolution size, upsampling only in the tail.

```
x -> clamp(0,1) -> conv3x3(3 -> 64) -> PReLU
  -> 16 x [ conv3x3(64 -> 64) -> PReLU ]
  -> conv3x3(64 -> 3*scale^2) -> PixelShuffle(scale)
  -> + nearest_upsample(x, scale)
```

Per frame at 1920x1080 (2.07 Mpx):

| part | GFLOP | share |
|---|---|---|
| head 3 -> 64 | 7.2 | 0.3% |
| 16 body convs 64 -> 64 | 2446 | 98.3% |
| tail 64 -> 3*scale^2 (12 for x2, 48 for x4) | 28.7 / 114.7 | 1.2% / 4.5% |
| **total** | **~2.48 TFLOP (x2), ~2.57 (x4)** | |

- **Scale is cheap in arithmetic, not necessarily in bandwidth.** x2 and x4 are 3.5% apart in
  FLOPs (all work is at input resolution) but the x4 tail writes a 199 MB fp16 tensor against
  49.8 MB. This difference has **not** been separated from the measured 7 ms gap - see §2.
- **The body is 98% of the arithmetic.** A per-frame optimisation has to attack 16 identical
  `64 -> 64, 3x3` convolutions.
- Receptive radius is 1 + 16 + 1 = 18 LR pixels (why tile reuse needs an 18 px halo).

## 2. Measured speed

All rows: static engine, opt level 5, iters 8. Session-relative comparisons only.

| model | input | tiling | ms/frame | TFLOP/s |
|---|---|---|---|---|
| animevideov3 x4 | 1920x1080 | none (opt3/iters4 - the old app default) | 94.3 | 27.2 |
| animevideov3 x4 | 1920x1080 | none (opt5/iters8) | 57.5 | 44.6 |
| animevideov3 x4 | 1920x1080 | moderate | 52.7 | 48.7 |
| 2x Futsuu Anime | 1920x1080 | moderate | 45.7 | 54.3 |
| 2x Futsuu Anime | 1440x1080 | moderate | 35.1 | 53.7 |
| 2x Futsuu Anime | 1920x1080 | full | ~42.7 (derived, see below) | ~58 |
| same, built from ONNX instead of torch-tensorrt | 1920x1080 | none | 50.3 | 49.4 |

- **Build flags alone are 1.79x** (94.3 -> 52.7) on identical weights. This is why "the model has
  no headroom" was the wrong prior: tactics and L2 tiling were leaving 44% on the table.
- **`tiling=full` vs `moderate` (2x model, 1920x1080):** the A/B is end-to-end on 502 gated anime
  frames - 11.01 s -> 10.595 s, i.e. **3.8% of wall clock**. Per *processed* frame the model
  share is 45.7 x 0.273 = 12.5 ms of a 21.1 ms frame, so if the delta is entirely model-side it
  is **~6.6% of the model (45.7 -> ~42.7 ms)**. Both numbers are stated because the A/B mixes
  bases; the per-model figure is derived, not directly measured.
- The x2-vs-x4 gap is 7 ms against a 3.5% FLOP difference (~1.6 ms), i.e. **not explained**.
  Either noise (+-15% between sessions) or the x4 tail is memory-bound.
- Run-to-run spread is +-15%; the audit runs that overlapped each other on the card spread up to
  50%, so the protocol in RESEARCH-LEDGER.md applies: one GPU job at a time.

## 3. What the achieved rate means

2.48 TFLOP in 45.7 ms = 54.3 TFLOP/s (tiling moderate), or ~58 TFLOP/s at the derived `full`
figure.

Peaks are fuzzy on a power-capped consumer part:

| base | value | how obtained |
|---|---|---|
| fp16 with fp32 accumulate, our own GEMM (8192^2, torch matmul) | **42.1 TF/s** | measured here. `allow_fp16_reduced_precision_reduction` does not change it (42.5) - that flag controls split-k reduction, not the accumulator |
| fp16 with fp16 accumulate, sustained | ~74 TF/s | borrowed estimate for this part at 165 W (not measured here) |
| fp16 with fp16 accumulate, whitepaper boost | 88 TF/s | spec sheet, not sustainable |

So: 54.3 / 74 = **73%** of the borrowed sustained fp16-acc figure, or 54.3 / 88 = 62% of boost.
The earlier claim of "73-84% of achievable" **added a +14.5% halo correction that belongs to our
* pipeline* tiling (512 -> 548 px), not to TRT's internal spatial tiling, which is unknown**. It
is therefore not a solid number and is removed; the honest statement is "somewhere between 60%
and 73% of a ceiling that is itself borrowed or unmeasured".

### A conv-shaped ceiling, measured here

Single 3x3 convolutions at 1080p, built through the same stack (tiling full, opt 5, fp16):

| conv | ms | TFLOP/s |
|---|---|---|
| 64 -> 64 (the shape of the body) | 7.90 | 19.4 |
| 64 -> 256 | 22.75 | 26.9 |
| 256 -> 256 | 49.89 | 49.0 |

The model runs the whole 18-conv stack at 54 TFLOP/s, i.e. **faster than any of its own layers
measured standalone**. Eighteen unpacked 64->64 convs would cost 18 x 7.9 = 142 ms against the
45.7 ms the fused engine needs, so cross-kernel fusion is worth more than 3x here - which is
also the explanation for why per-layer reasoning kept misleading us.

Caveat: the standalone builds logged "CUDA error 2 allocating 4246732800-byte buffer" during
autotuning, so the best tactics were skipped for some of them. These are **lower bounds**, not a
ceiling; a fair conv ceiling needs a quieter tactic search (smaller workspace, editable timing
cache).

So the honest summary of §3 is: we measure 54 TFLOP/s, we do not have a trustworthy ceiling for
this shape, and the two numbers we used to compare (42.1 GEMM with fp32 accumulate, 74 borrowed
sustained fp16-acc) bracket it loosely. A per-layer time breakdown would replace all of this -
see §6.

## 4. Techniques: what we ran here vs what we read

| technique | ran here? | result on this machine | elsewhere / caveat |
|---|---|---|---|
| builder opt level 5 + iters 8 | **yes** | 94.3 -> 57.5 ms (1.64x) | - |
| L2 tiling `moderate` | **yes** | 57.5 -> 52.7 ms (1.09x) | this IS cross-kernel fusion; the earlier claim in this file that "fusing across layers is impossible because of PReLU" was wrong (PReLU is fused into the conv epilogue) |
| L2 tiling `full` | **yes** | 11.01 -> 10.60 s end-to-end on 502 anime frames; ~6.6% of the model (derived) | adopted as default |
| `l2_limit_for_tiling = 24 MiB` | **yes** | 67.4 and 103.2 ms against 48.6 (two overlapping runs) - harmful | leave the knob alone |
| CUDA graphs | **yes** | ~0.3% | - |
| parallel engine contexts | **yes** | 0%, and our wiring produced corrupt output | the corruption was an implementation fault (two `torch.jit.load` modules are not independent contexts), not a GPU property |
| duplicate-frame gate (pipeline level) | **yes** | 46.8% of film frames / **72.7% of anime frames** skipped; anime 16.3 -> 47.4 fps; 52.6 dB vs ungated | see §7 |
| INT8, implicit + fp16 fallback, entropy calibration on 8 real frames | **yes** | **slower**: 73.2 and 94.3 ms against 48-51; the two runs also overlapped on the card (29% spread), so treat the magnitude as indicative | this is the implicit-with-fallback configuration, i.e. a reformat-heavy graph. Explicit Q/DQ, fully-int8 NHWC, per-channel: **not tried**. The "2x for int8 on a 4090" note in issue #4081 does not transfer to C=64 implicit |
| accumulator precision | **yes (closed)** | all 16 body convs run `f16f16_f16f16` = fp16 accumulator, per DETAILED tactic dump | no fp32-accumulate share exists to convert |
| hand-written conv kernels | **no** | - | one public attempt was 4.7x slower than TRT, but there is a **counter-precedent**: for a 3x3 conv cuDNN chose Winograd at 74 ms while TRT's best tactic was 156 ms, i.e. TRT's tactic set is not the set of all algorithms. Recorded as untested, not as "slower" |
| 2:4 structured sparsity | **no** | - | needs retrained sparse weights; NVIDIA reports small gains under 256 channels |
| Winograd | **no** | - | F(2,3) cuts MACs 2.25x but inflates tensors ~4x, dropping intensity below the ridge; the fp16-overflow argument comes from ncnn's Vulkan backend, not this GPU |
| FP8 | **no** | - | issue #4081 reports ~1.0x for a ConvNet on a 4090 and 2x on L40S; GeForce parts are firmware-capped. Not tested here |
| distillation / pruning (16 -> 8 convs, 64 -> 48) | **no** | - | 1.5-2x in the literature; **out of scope by the owner's decision**, not by hardware |

## 5. What the tactic dump actually showed

DETAILED profiling verbosity at build time, tiling full, fp16:

```
16x  sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32
 1x  sm75_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16
 1x  sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16
```

Three things, in order of importance:

1. **`f16f16_f16f16_f16`** = fp16 inputs with fp16 accumulation, NHWC. There is no fp32-acc share
   to convert; the accumulator hypothesis is closed.
2. **`sm75_` kernels on an sm89 part.** 16 of 18 convolutions get Turing-generation kernels; the
   names carry no `stage*` / `warpsize*` / `tensor16x8x16` suffixes typical of Ampere+ xmma.
   This is not proof of a loss, but it is the one concrete lead in the dump and it is testable
   with an editable timing cache (§6, item 4).
3. **GEMM geometry, not a C=64 penalty.** The body runs `tilesize256x64x32`, so for fprop
   implicit GEMM K = C*R*S = 576 and a K-tile of 32 means **18 mainloop steps** - a normal
   length, not a defect. The task is M = 2,073,600, N = 64, K = 576: with a 256x64 tile that is
   ~8100 blocks and ~238 waves over 34 SMs, i.e. plenty of parallelism. (An earlier revision
   claimed "two K steps" and a "short tile grid"; both were wrong.)

Also: the tiled engine has **25 top-level layers, all `__myl_*`** - TRT fuses the network into
myelin subgraphs. So the old picture "18 separate kernels, each with a full 530 MB round trip"
is wrong. How much DRAM traffic actually remains after fusion is **unknown**; the
arithmetic-intensity/ridge argument therefore stays a hypothesis, not a result, until the
counter is read (§6, item 2).

## 6. What is left to measure (short list)

1. ~~Per-layer time~~ **done without trtexec, via `trt.IProfiler`**: 23 layers, 51.95 ms total,
   the top layer at 3.91 ms and the 18 fused conv subgraphs spread evenly at 2.2-3.9 ms each.
   There is **no outlier** - so no single layer to fix, and the `sm75_` lead is a uniform tactic
   choice rather than one bad kernel. What is still missing is the split of that time into
   compute versus DRAM, which needs `nsys` (item 2).
2. **nsys** tensor-active % and DRAM bytes with clocks pinned (`nvidia-smi -lgc`): that is the
   actual localisation of the gap; everything in §3 is inference.
3. **A conv ceiling instead of a GEMM ceiling**: time one 3x3 conv at the same area
   (64->64 and 64->256 at 2.07 Mpx) through the same stack, so 51-54 TF/s is compared against a
   conv-shaped achievable number rather than matmul.
4. **Alternative tactics for the 16 body convs**: editable timing cache, check whether an
   sm80/86/89 variant exists for this shape, how it timings, and A/B it - the `sm75_` lead.
5. **Explicit Q/DQ int8 in NHWC** if the format (rather than the implementation) is to be closed.

Each of these is hours, not days. Until at least (1) and (2) are done, "per-frame is at its
floor" is a statement about *what we tried*, not about the hardware.

## 7. The lever that works today: fewer frames

| scenario | frames through the model | measured |
|---|---|---|
| anime, ungated (502 frames, 1080p -> 2160p) | 100% | 16.3 fps |
| anime, duplicate gate | **27.3%** | **47.4 fps** (10.54 s), median 53 dB / worst frame 48.1 dB vs ungated |
| film, ungated (94 frames, 1440p -> 2880p) | 100% | 19-22 fps |
| film, duplicate gate | 53.2% | 23.7 fps, 71 dB vs ungated |
| tile-level reuse (anime, 512/1024 px) | 65.8% / 66.0% of tiles | **slower**: 21.4 / 26.1 s against 10.6 - the tile path costs ~42.5 ms/frame at 100% of frames, no better than the 45.7 ms full-frame call |

So the model costs 45.7 ms on the frames it sees, and the pipeline pays 45.7 x 0.273 = 12.5 ms
per output frame on anime. Tile-level reuse was the last "do less work" candidate and it is a
measured loss on both materials.

## 8. Reproduce

Isolated model timing: build the engine through `UpscalePytorch` with `trt_static_shape=True`,
then time `upscale_model_wrapper(x)` over 25 iterations after 6 warmup calls. Tactic dump:
rebuild with `cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED` and read
`IEngineInspector().get_engine_information(LayerInformationFormat.JSON)` (this TRT build's
ONELINE format lists layer names only).
