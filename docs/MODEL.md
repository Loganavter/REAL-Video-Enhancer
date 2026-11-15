# Model report: Real-ESRGAN compact on this machine

Scope: how fast the network itself runs, what was measured here versus read elsewhere, and what
is still open. Pipeline stages appear only where they explain a number.

Rig: RTX 4060 Ti 16 GB (AD106, 34 SM, 165 W cap), Ryzen 5 5600, TensorRT 10.12.0.36,
torch 2.9.0+cu130. Unless stated: fp16 engines, static shape, `optimization_level=5`,
`num_avg_timing_iters=8`, `tiling_optimization_level=full` (the new default).

**Conclusion up front**: no kernel-level technique we could run locally beat a well-built
engine, and the one model-side win of this round is `tiling=full` (3.8% end-to-end). We do
**not** have a trustworthy ceiling for this shape, so the size of the remaining gap is unknown:
the model is at 65-73% of a *borrowed* sustained figure, which is not the same as "finished".
The pipeline-side win of the round is bigger than any model work: the frame gate plus a tile
grid fitted to the frame is 1.4x, and the tile grid was previously written off as a dead end
(§7). One concrete lead from the tactic dump (Turing-generation kernels selected on an Ada
part) is listed in §6 and has not been chased.

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
| 16 body convs 64 -> 64 | 2446 | 98.5% |
| tail 64 -> 3*scale^2 (12 for x2, 48 for x4) | 28.7 / 114.7 | 1.2% / 4.5% |
| **total** | **~2.48 TFLOP (x2), ~2.57 (x4)** | |

- **Scale is cheap in arithmetic, not necessarily in bandwidth.** x2 and x4 are 3.5% apart in
  FLOPs (all work is at input resolution) but the x4 tail writes a 199 MB fp16 tensor against
  49.8 MB. This difference has **not** been separated from the measured 7 ms gap - see §2.
- **The body is 98.5% of the arithmetic.** A per-frame optimisation has to attack 16 identical
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
| 2x Futsuu Anime | 1440x1080 | moderate | 35.1 | 53.0 |
| 2x Futsuu Anime | 1920x1080 | full | **51.37 (measured, same session as the app runs below)** | 48.3 |
| 2x Futsuu Anime | 1920x1080 | full, while an NVENC 2160p encode runs alongside | 55.99 | 44.3 |
| same, built from ONNX instead of torch-tensorrt | 1920x1080 | none | 50.3 | 49.4 |

- **Build flags alone are 1.79x** (94.3 -> 52.7) on identical weights. This is why "the model has
  no headroom" was the wrong prior: tactics and L2 tiling were leaving 44% on the table.
- **`tiling=full` vs `moderate` (2x model, 1920x1080):** the A/B is end-to-end on 502 gated anime
  frames - 11.01 s -> 10.595 s, i.e. **3.8% of wall clock**. That is the trustworthy number.
  The earlier "45.7 -> ~42.7 ms, i.e. ~58 TFLOP/s" was derived from it by assuming the whole
  delta is model-side, and a direct measurement of the `full` engine at 1920x1080 (51.37 ms,
  see the table) does **not** support it: 51.37 ms is 48.3 TFLOP/s, and the `moderate` figure
  of 45.7 ms came from a different session in which this machine was ~26% faster overall (the
  same ungated anime render took 30.8 s then against 38.85 s now). So the derived 58 TFLOP/s is
  withdrawn; what stands is the same-session ratio (3.8% of wall clock) and the same-session
  absolute (51.4 ms, 48.3 TFLOP/s at today's clocks).
- **NVENC alongside the engine costs 9%** of the engine (51.37 -> 55.99 ms in a controlled
  A/B with an ffmpeg encode feeding from /dev/zero on the same card; the app measures 54.3 ms
  per rendered frame, which is this effect plus 5.7% over the quiet standalone run). The
  earlier "the app is 23-43% slower than isolated" was that cross-session drift, not a
  wrapper cost.
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
| fp16 with fp32 accumulate, our own GEMM (8192^3, torch matmul) | **42.1 TF/s** | measured here. `allow_fp16_reduced_precision_reduction` does not change it (42.5) - that flag controls split-k reduction, not the accumulator |
| fp16 with fp16 accumulate, sustained | ~74 TF/s | borrowed estimate for this part at 165 W (not measured here) |
| fp16 with fp16 accumulate, whitepaper boost | 88 TF/s | spec sheet, not sustainable |

So, with bases stated: 54.3 (tiling moderate, the session in which this machine was faster)
/ 74 = **73%**, 54.3 / 88 = 62%; and at the `tiling=full` default the direct measurement of
48.3 TFLOP/s gives **65% / 55%**. The report previously mixed these two bases without saying
which was which, and quoted a derived 58 TFLOP/s that a direct measurement does not support.
The earlier claim of "73-84% of achievable" **added a +14.5% halo correction that belongs to our
* pipeline* tiling (512 -> 548 px), not to TRT's internal spatial tiling, which is unknown**. It
is therefore not a solid number and is removed; the honest statement is "somewhere between 60%
and 73% of a ceiling that is itself borrowed or unmeasured".

### Conv-shaped numbers, and what they do and do not support

Single 3x3 convolutions at 1080p, built through the same stack (tiling full, opt 5, fp16):

| conv | ms | TFLOP/s | compute floor | DRAM floor |
|---|---|---|---|---|
| 64 -> 64 | 7.90 | 19.4 | 2.07 ms @74 TF/s | 1.84 ms @288 GB/s |
| 64 -> 256 | 22.75 | 26.9 | - | - |
| 256 -> 256 | 49.89 | 49.0 | 33 ms | 7.4 ms |

**The 64->64 number is not usable as a ceiling.** 7.9 ms is twice a fully serial compute+DRAM
floor, and the build logged "CUDA error 2 allocating 4246732800-byte buffer" during autotuning,
i.e. tactics were skipped. The likely second cause is that these standalone engines take NCHW
input and return NCHW output, so each one pays two `Reformat` layers over ~530 MB (about 3.7 ms)
that the fused stack pays only at the network borders. An earlier revision of this file built a
"cross-kernel fusion is worth >3x" claim on 16 x 7.9 = 126 ms against 45.7 ms; that was wrong on
three counts - the measurement is invalid, the body has 16 convs not 18, and §4 measures
cross-kernel tiling at 1.09x + 1.04x, so the same mechanism cannot also be worth 3x.

What the numbers *do* support, and it is stronger than the cautious line this section used to
end on:

- The per-layer profile (§6) puts the 18 subgraphs at 2.2-3.9 ms, about **2.66 ms per body conv**
  on average; 152.9 GFLOP / 2.66 ms = **~57 TF/s, i.e. ~78% of the borrowed 74 TF/s**. That
  figure was taken in the faster session (51.95 ms under `IProfiler` against 45.7 ms wall
  clock); at today's clocks the same engine is 51.4 ms and the per-conv figure would be ~3 ms,
  i.e. ~51 TF/s, and the profiler serialises on top of that.
- The only standalone conv that is compute-dominated, 256->256, reaches **49 TF/s - below the
  model's 54.3**. So the model already runs faster than any of its own convolutions measured
  standalone, and the borrowed GEMM ceiling is probably the wrong reference for a conv shape with
  C=64.

Honest summary: 48-54 TFLOP/s depending on the session's clock state (the spread is the
machine, not the kernel); no trustworthy conv ceiling; the model is at 65-73% of a *borrowed*
sustained figure and above every standalone conv we could build. That is close to a floor, but
it is not a proof, and the difference matters: 25-35% of a borrowed ceiling is not "finished".

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
2. **`sm75_` kernels on an sm89 part.** **17 of 18** convolutions get Turing-generation kernels
   (the head conv included; only the last conv is sm80); the
   names carry no `stage*` / `warpsize*` / `tensor16x8x16` suffixes typical of Ampere+ xmma.
   This is not proof of a loss, but it is the one concrete lead in the dump and it is testable
   with an editable timing cache (§6, item 4).
3. **GEMM geometry, not a C=64 penalty.** The body runs `tilesize256x64x32`, so for fprop
   implicit GEMM K = C*R*S = 576 and a K-tile of 32 means **18 mainloop steps** - a normal
   length, not a defect. The task is M = 2,073,600, N = 64, K = 576: with a 256x64 tile that is
   ~8100 blocks and ~238 waves over 34 SMs, i.e. plenty of parallelism. (An earlier revision
   claimed "two K steps" and a "short tile grid"; both were wrong.)

Also: the tiled engine has **25 top-level layers, all `__myl_*`** (23 in the `IProfiler` run of
§6 - a different build, same shape) - TRT fuses the network into myelin subgraphs. So the old picture "18 separate kernels, each with a full 530 MB round trip"
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
3. ~~A conv ceiling instead of a GEMM ceiling~~ **done, but the result is unusable for 64->64**
   (§3): the tactic search hit OOM and each standalone engine pays border reformats. It would
   need re-running with a constrained workspace and, ideally, NHWC input/output so the comparison
   is not dominated by layout conversion.
4. **Alternative tactics for the 16 body convs**: editable timing cache, check whether an
   sm80/86/89 variant exists for this shape, how it timings, and A/B it - the `sm75_` lead.
5. **Explicit Q/DQ int8 in NHWC** if the format (rather than the implementation) is to be closed.

Each of these is hours, not days. Item (2) is the one that would turn §3 from inference into
measurement; until it is done, "per-frame is at its floor" remains a statement about *what we
tried*.

## 7. The lever that works today: fewer frames

| scenario | frames through the model | measured |
|---|---|---|
| anime, ungated (502 frames, 1080p -> 2160p) | 100% | 16.3 fps |
| anime, duplicate gate | 29.1% | 51.71 dB average / 48.64 dB worst frame against the ungated render, 70.9% reuse |
| anime, gate only (default) | 29.1% of frames | **11.3 s against 38.85 s ungated (3.4x)**, 0.045% of pixels above 8 levels (RGB) on the worst of 19 sampled frames, peak 13 |
| anime, gate + tile reuse | 57.7% of tiles | 9.2 s, but **0.287%** above 8 levels and peaks of 98: the tile path has its own staleness, so it is off by default (see the investigation linked below) |
| film, ungated (94 frames, 1440p -> 2880p) | 100% | 4.66 s |
| film, duplicate gate | 81.9% | 4.33 s (1.08x), 50.34 dB average / 39.83 dB worst frame against ungated. The 71 dB this row used to claim came from comparing two renders that were both corrupted by the pipe-format bug in ledger C4 |
| tile-level reuse, square 512 px cores (anime) | 65.8% of tiles | 1.44x faster than ungated (30.8 -> 21.4 s) but slower than the frame gate (10.6 s). The loss was **geometry**: 512 does not divide 1080, so 12 windows of 548x548 cover a frame of 1.55 Mpx = 1.74x its area |
| tile-level reuse, cores fitted to the frame (anime, 480x540) | 65.6% of tiles | **18.14 s** ungated against 21.37 s for the 512 grid: the area model `area x (1 - reuse)` predicted 21.49 s for the 512 grid (measured 21.37, 0.6%) and 16.9 s for the fitted one (measured 18.14), so there is no per-tile launch penalty worth modelling |
| gate **and** fitted tiles (anime) | 57.4% of tiles inside the 27% of frames the gate renders | **8.45 s** against 11.74 s for the gate alone = **1.39x**; 51.6 dB average / 47.5 dB worst frame against the ungated render (gate alone 52.7 / 48.2). The tile share falls from 65.6% to 57.4% because the frames the gate leaves are exactly the ones that changed, so the two levers compose at ~1.4x rather than multiplicatively |

So the model costs 45.7 ms on the frames it sees at 1440x1080, and the pipeline pays
45.7 x 0.273 = 12.5 ms per output frame on anime at that size. Tile-level reuse is not a dead
end after all: with a grid that divides the frame it stacks on the gate for 1.39x (see the
table above). What it cannot do is replace the gate, because the reuse it finds inside a frame
is concentrated in frames the gate already skips.

Un-gated anime numbers in this table were taken in different sessions and drift: the same
ungated full-frame configuration measured 30.8 s in one session and 38.85 s in another on the
same machine (the desktop's own GPU clients keep the card at 17-47% "utilisation", and the
power budget moves with them). Ratios within one session are the only safe comparison.

## 8. Reproduce

Isolated model timing: build the engine through `UpscalePytorch` with `trt_static_shape=True`,
then time `upscale_model_wrapper(x)` over 25 iterations after 6 warmup calls. Tactic dump:
rebuild with `cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED` and read
`IEngineInspector().get_engine_information(LayerInformationFormat.JSON)` (this TRT build's
ONELINE format lists layer names only).
