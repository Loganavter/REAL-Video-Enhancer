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
The pipeline-side win of the round is bigger than any model work: the frame gate alone is
**3.3x** (38.85 -> 11.7 s on 502 anime frames) at 0.009% of pixels above 8 levels. The tile grid
fitted to the frame adds 1.39x on top *inside* the frames the gate renders, but it carries its
own artifact (0.287% at 9.2 s), so it is **opt-in, not shipped** - §7 has the trade. The lead from
the tactic dump (Turing-generation kernels selected on an Ada part) is closed in §6: restricting
the tactic sources to JIT convolutions picks the *same* `sm75_` kernel, so the name is the name
of the winner.

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

2.48 TFLOP in 45.7 ms = 54.3 TFLOP/s at `tiling=moderate`; the same measurement at the
`tiling=full` default reads 48.3 TFLOP/s. (An earlier revision quoted "~58 TFLOP/s at the derived
`full` figure"; that derivation assumed the whole 45.7 -> 42.7 ms step was the model, and direct
measurement does not support it - see the withdrawn-derivation note below.)

### The card is at its power cap, and that is most of the answer

An isolated engine loop (490 runs in 28.1 s = 57.37 ms/frame, no host pipeline, no encoder) sampled
once a second with `nvidia-smi`:

| what | measured | note |
|---|---|---|
| SM utilisation | **100%** in all 23 samples | not a stall, not idle lanes |
| memory-controller utilisation | 85% (max 88) | busy, not saturated |
| SM clock | **2130 MHz** (max 2295) | 7% below the part's maximum |
| memory clock | 8751 MHz | at its rating |
| power draw | **162 W** (max 163) | against the card's **165 W** cap |
| temperature | 60 C | not thermal |

It also moves **9.37 GiB of tensor traffic per frame**, which at the card's 288 GB/s is 36.5 ms of
pure DRAM time (65% of peak while the engine runs) - so the traffic floor is 20 ms below the 57 ms
it actually takes (§6 item 2 has the table). The engine saturates the SMs, keeps the memory system
85% busy, and does both at **162 of 165 W** with the SM clock held below maximum. So the binding constraint is the power budget, not a kernel -
which is consistent with everything in §4 and §6: no kernel-level lever moved the number, and the
only two things that did were fewer DRAM round trips (`tiling=full`) and fewer frames (the gate).
It is also why the int8 ceiling below is the only place in this report where a 1.5x exists: halving
the FLOPs is the only lever that does not fight the cap.

What the sampling does **not** give is the DRAM-byte and tensor-active-% split - that is what `nsys`
would add. The 85% memory-controller figure says the memory system is working hard; it does not say
how much of the 57 ms is spent waiting on it.

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
| L2 tiling `full` | **yes** | 11.01 -> 10.60 s end-to-end on 502 anime frames = **3.8%**, measured | adopted as default |
| `l2_limit_for_tiling = 24 MiB` | **yes** | 67.4 and 103.2 ms against 48.6 (two overlapping runs) - harmful | leave the knob alone |
| CUDA graphs | **yes** | ~0.3% | - |
| parallel engine contexts | **yes** | 0%, and our wiring produced corrupt output | the corruption was an implementation fault (two `torch.jit.load` modules are not independent contexts), not a GPU property |
| duplicate-frame gate (pipeline level) | **yes** | **70.9% of anime frames** / 18.1% of the film clip skipped; anime 16.3 -> 29.1% of frames rendered; 52.7 dB average vs ungated | see §7. The 46.8% / 72.7% pair this row used to quote came from two renders that were both corrupted by the pipe-format bug in ledger C4, and the film clip now trips the noise-floor guard rather than reusing |
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

1. ~~Per-layer time~~ **done without trtexec, via `trt.IProfiler`**: the fp16 engine has 25 layers,
   23 of which report time (16 of them are the fused conv+PReLU myelin blocks, 2 the head and tail
   convolutions). The top layer is 3.15 ms and the 18 conv subgraphs spread evenly at 1.3-3.1 ms;
   there is **no outlier** - so no single layer to fix, and the `sm75_` lead is a uniform tactic
   choice rather than one bad kernel. The session total reads 51.95 ms here and 55.3 in a later
   one, so only the shape is portable. What is still missing is the split of that time into
   compute versus DRAM, which needs `nsys` (item 2).
2. ~~**nsys** tensor-active % and DRAM bytes with clocks pinned~~ **the byte count is computable
   from the engine's own layer list, and that is most of the answer** (sum of input+output tensor
   sizes over the 25 layers, `Format/Datatype`-aware):

   | | fp16 engine | int8 engine (per-channel PReLU) |
   |---|---|---|
   | tensor traffic per frame | **9.37 GiB** (10.1 GB) | **14.48 GiB** (15.5 GB) |
   | at the card's 288 GB/s that traffic alone takes | 36.5 ms | 56.0 ms |
   | measured | 55-57 ms | 68-71 ms |
   | fraction of peak DRAM it implies | 65% | 64% |

   Two results, and the second is the int8 verdict in one number:

   - The fp16 engine sits at **65% of the card's DRAM peak** while the SMs are 100% busy and the
     draw is 162 of 165 W: a balanced, power-limited engine about 20 ms above its own traffic floor.
   - The int8 build moves **55% more bytes than fp16** - 14.5 against 9.4 GiB - because every block
     round-trips through a half-precision tensor for its PReLU. "Fewer bytes and fewer FLOPs" is
     inverted by the boundary, which is the same fact the profile shows as 17 separate passes.
   - Arithmetic intensity is **236 FLOP/byte**: under the 306 that the whitepaper peak (88 TF/s)
     implies, over the 146 that our own measured GEMM (42.1 TF/s) implies. So "compute- or
     memory-bound" has no single answer here: 65% of DRAM, a full SM pipe and an exhausted power
     budget at the same time.

   **`nsys` runs here and confirms the shape of that time** (its timeline needs no counters).
   25 frames, 25 kernel launches per frame (16 of them the same GEMM), 56.8 ms of kernel time
   against a 57.4 ms loop - i.e. the
   GPU is busy for the whole frame and there is no launch gap to reclaim (which is also why CUDA
   graphs measured ~0.3%):

   | ms/frame | launches | kernel |
   |---|---|---|
   | **48.72** | 16 | `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_nhwckrsc_nhwc_tilesize128x64x32_stage1` |
   | 2.40 | 1 | `__myl_MulLtSele` (the one PReLU not fused into a GEMM) |
   | 1.92 | 1 | `sm80_xmma_..._indexed_wo_smem_...` (tail conv) |
   | 1.45 | 1 | `sm75_xmma_..._indexed_wo_smem_...` (head conv) |
   | 1.06 | 1 | `__myl_TranReshMoveAdd` (residual + reshape) |
   | **0.61** | 1 | `cuInt8::nchwTonchw<__half, float, 4>` - a layout conversion |
   | 0.37 | 1 | `ResizeNearest2DKernel` |
   | **0.20** | 1 | `CUTENSOR permutationKernelPLC3` - another permute |
   | 0.10 | 2 | `__myl_Move`, `__myl_MaxMin` |

   Two things this adds to the `IProfiler` view. First, the fp16 path already **fuses 16 of its 17
   PReLUs into the convolution kernel** (the myelin subgraph named `CorrMulLtSele` compiles to one
   xmma GEMM per block, and only the last PReLU arrives as a separate `MulLtSele` pass at 2.40 ms) -
   that is precisely what the int8 build cannot do, so its 31.2 ms of PReLU passes is the whole
   difference. Second, **0.81 ms/frame of pure layout conversion** (`nchwTonchw` + the cuTENSOR
   permute) that no layer in the profile owns; an NHWC input/output would remove it, and it is the
   only clean ~1.5% left in the engine.

   `ncu` would still add the byte count and tensor-pipe activity. It also **installs without root**,
   and it runs - but this host's driver is loaded with **`RmProfilingAdminOnly: 1`**
   (`/proc/driver/nvidia/params`), so counter collection is refused:
   `ERR_NVGPUCTRPERM`. Unlocking it is one line in `/etc/modprobe.d/` -
   `options nvidia NVreg_RestrictProfilingToAdminUsers=0` - plus a reboot, because the module cannot
   be unloaded under a running desktop. Nothing else is missing.
3. ~~A conv ceiling instead of a GEMM ceiling~~ **done, but the result is unusable for 64->64**
   (§3): the tactic search hit OOM and each standalone engine pays border reformats. It would
   need re-running with a constrained workspace and, ideally, NHWC input/output so the comparison
   is not dominated by layout conversion.
4. ~~Alternative tactics for the 16 body convs~~ **done, and the lead is closed**: four builds of
   the same ONNX (baseline, 8 GB explicit workspace, cuDNN-only, JIT-only) with three interleaved
   timing rounds each. A workspace changes the tactic mix (one tail conv moves from sm80 to sm75)
   and buys 0.5%, i.e. noise; a JIT-only source set picks the *same* `sm75_` tactic, so there is no
   sm80/86/89 variant being skipped for this shape at K=576; cuDNN-only is 10.7% *slower*. The name
   `sm75_` is the name of the winner, not a symptom.
5. ~~Explicit Q/DQ int8~~ **done, and the earlier verdict on it was wrong in its mechanism**:
   calibrated on our own frames it is essentially lossless (0.000% of pixels above 8 levels against
   the FP16 engine, peak 4 levels, PSNR 54.1 dB) and **slower** - 68.11 ms against 55.54. The first
   explanation here was "TRT picked an NCHW int8 kernel where the FP16 path is NHWC, and the 60
   reformat layers cost more than int8 saves". The layer dump says otherwise on both counts: the
   tactic is `sm80_xmma_fprop_implicit_gemm_interleaved_i8f16_i8i32_f32_nchw_vect_c_...`, i.e. the
   **interleaved NC/32HWC32 int8 tensor-core layout**, and the engine holds **21 reformat layers of
   78** - "60" was the number of non-convolution layers. The real cost is the boundary. Per-layer,
   with `trt.IProfiler`:

   | group | count | ms/frame each | total |
   |---|---|---|---|
   | int8 `CaskConvolution` (int8 in, half out) | 16 | 1.56-1.88 (152.9 GFLOP / 1.68 ms = **91 TFLOP/s**) | 26.8 |
   | `PointWiseV2` = the PReLU, a separate pass | 17 | 1.71-2.37 | 31.2 |
   | **reformats** - the layers the old verdict blamed | 21 | 0.05-2.41 | **5.6** |
   | head, tail, upsampler, Add, Clip | 6 | - | 4.9 |

   A second profile of the same engine minutes later read 71.24 ms with the same shares
   (PReLU 32.74, convs 27.77, reformats 5.59, the rest 4.87), so the ±4% between runs is the
   machine, not the graph - and it also makes the point: the reformats are 8% of the engine, the
   PReLU passes are 46%.

   In FP16 the same 16 blocks are **one fused myelin kernel each** (`__myl_CorrMulLtSele` = conv +
   mul + lt + select = conv + PReLU) at 2.93 ms, i.e. 52 TFLOP/s for the same FLOPs. Quantised,
   the conv alone is 1.74x faster (1.68 ms) and
   the PReLU becomes a pass of its own (1.80 ms), so the block costs 3.48 against 2.93. It is not
   fused for a measurable reason: the int8 conv reports `HasReLU: 0, Activation: NONE`, and the
   PReLU is a separate `PointWiseV2` that reads two tensors - the frame and the per-channel slope -
   with 21 reformat layers for the slope broadcasts. A scalar slope would fit a per-tensor int8
   epilogue, and there is none to be had here: the 1088 slopes span **-1.39..1.70** with a
   per-channel coefficient of variation above 1. That is the whole gap. Quantising the conv output as
   well, so the PReLU reads int8: **78.36 ms**, with 36 reformat layers instead of 21 and 16 `DQout`
   passes at 1.64 ms - charging the boundary per hop makes it worse, not better. So the precision
   path is closed as a TRT configuration.

   **How big the prize would be if the PReLU were fusion-friendly**, same graph, same calibration,
   only `PRelu` replaced by a scalar `LeakyRelu` (alpha = that node's mean slope):

   | build | ms/frame | vs fp16 | reformat layers | PReLU passes | picture against fp16 |
   |---|---|---|---|---|---|
   | fp16 (fused conv+PReLU) | 56.28 | 1.00x | 2 | - | reference |
   | int8, per-channel PReLU (the real function) | 71.37 | 0.79x | 21 | 17 | **0.000%** above 8, peak 4, 54.05 dB |
   | int8, scalar LeakyRelu (the form TRT fuses) | **40.24** | **1.40x** | 4 | 1 | **99.323%** above 8, peak 255, **-2.02 dB** |

   So the ceiling is real - 40 ms instead of 56, because the tensor chain stays int8 end to end and
   the PReLU stops being a pass - and the only form that reaches it is **not this function**: a
   scalar slope destroys the picture (99.3% of pixels, negative PSNR). The model's per-channel
   slopes span -1.39..1.70, so there is no scalar stand-in. That leaves exactly one route to the
   1.4x: a plugin that implements an int8 convolution with a per-channel PReLU epilogue - i.e. its
   own int8 GEMM, against a counter-precedent (a hand-written conv kernel once measured 39.8
   TFLOP/s where the engine holds 91 on this shape). Recorded as the honest boundary of this
   investigation, not as a recommendation. The verdict is therefore "the quantised block boundary costs more than int8
   saves on this architecture", not "no int8 kernel exists".

What is left is item (2) alone: `nsys` tensor-active-% and DRAM bytes with clocks pinned would turn
§3 from inference into measurement. Until then, "per-frame is at its floor" remains a statement
about *what we tried*.

## 7. The lever that works today: fewer frames

| scenario | frames through the model | measured |
|---|---|---|
| anime, ungated (502 frames, 1080p -> 2160p) | 100% | 16.3 fps |
| anime, duplicate gate | 29.1% | 51.71 dB average / 48.64 dB worst frame against the ungated render, 70.9% reuse |
| anime, gate only (default) | 29.1% of frames | **11.7 s against 38.85 s ungated (3.3x)**, 0.009% of pixels above 8 levels (RGB) on the worst of 19 sampled frames, peak 14. Re-taken after the chain-reachability fix: chain 4 still certified, the same **362/498** reuses, 11.93 s, 0.017% on a 13-frame sample (the sample changed, the configuration did not) |
| anime, gate + tile reuse | 57.7% of tiles | 9.2 s, but **0.287%** above 8 levels and peaks of 98: the tile path has its own staleness, so it is off by default (see the investigation linked below) |
| Chainsaw Man ep01 (1080p -> 2160p, 8 s) | 66.4% | **6.56 s against 13.50 s ungated (2.06x)**, 0.005% of pixels above 8 levels on the worst of 15 sampled frames, peak 14, chain 2 certified |
| Violet Evergarden ep01 (1080p -> 2160p, 8 s) | 64.8% | **6.92 s against 13.31 s ungated (1.92x)**, 0.096%, peak 18, chain 2 - 4% of margin under the bar |
| film, ungated (94 frames, 1440p -> 2880p) | 100% | 4.66 s |
| film, duplicate gate | 100% rendered at the shipping preset | **the gate refuses** (noise floor 2.27, above the 1.5 limit), so the render is byte-identical to the ungated one: 4.67 s against 4.63 s for 94 frames. Before the noise-floor guard it reused 18.1% of frames for 4.33 s (1.08x) at 39.83 dB worst frame; the 71 dB this row used to claim came from comparing two renders that were both corrupted by the pipe-format bug in ledger C4 |
| tile-level reuse, square 512 px cores (anime) | 65.8% of tiles | 1.44x faster than ungated (30.8 -> 21.4 s) but slower than the frame gate (10.6 s). The loss was **geometry**: 512 does not divide 1080, so 12 windows of 548x548 cover a frame of 1.55 Mpx = 1.74x its area |
| tile-level reuse, cores fitted to the frame (anime, 480x540) | 65.6% of tiles | **18.14 s** ungated against 21.37 s for the 512 grid: the area model `area x (1 - reuse)` predicted 21.49 s for the 512 grid (measured 21.37, 0.6%) and 16.9 s for the fitted one (measured 18.14), so there is no per-tile launch penalty worth modelling |
| gate **and** fitted tiles (anime) | 57.4% of tiles inside the 27% of frames the gate renders | **8.45 s** against 11.74 s for the gate alone = **1.39x**; 51.6 dB average / 47.5 dB worst frame against the ungated render (gate alone 52.7 / 48.2). The tile share falls from 65.6% to 57.4% because the frames the gate leaves are exactly the ones that changed, so the two levers compose at ~1.4x rather than multiplicatively |

So the model costs **45.7 ms at 1920x1080 and 35.1 ms at 1440x1080** (the two sizes in the table
above; an earlier revision wrote 45.7 for the 1440x1080 clip). On the anime clip, which is
1920x1080, the pipeline pays 45.7 x 0.273 = 12.5 ms per output frame. Tile-level reuse is not a dead
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
