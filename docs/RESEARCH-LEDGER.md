# Research ledger

Everything below was either measured on this machine or read from a named source. Numbers
without a source are ours. If an idea is not in this file, it was not considered.

Measurement rig: RTX 4060 Ti 16 GB, Ryzen 5 5600 (6c/12t), TensorRT 10.12, torch 2.9,
PyAV 19.0.2.

**Measurement protocol - one GPU job at a time.** Two engine builds running next to a render
inflate every number: in this session a `full_l2` engine measured 67.4 ms and 103.2 ms in two
"identical" runs, and an app render measured 20.4 s against 11.5 s for the same configuration,
purely because a second process was building engines on the same card. Check for a running render/benchmark process before quoting anything, and repeat the pair
(same-session ratios only - the same ungated configuration measured 30.8 s in one session and
38.85 s in another). Note that `nvidia-smi` utilisation is **not** a usable idle check on this
host: the desktop's own GPU clients (gnome-remote-desktop, sunshine, VS Code) submit small
copies and keep the counter at 17-47% while nothing of ours runs. They do not take meaningful
throughput - idle draw is 23-26 W against 164 W under load - so the rule "one GPU job at a
time" means no other render or benchmark, not "utilisation 0". Two
clips: a 1440x1080 10-bit HEVC film excerpt (94 frames, "the clip") and a 40 s window of the
same film ("the window"); anime work uses a 502-frame 1920x1080 clip ("anime"). Unless stated,
wall-clock for the whole render with `x265_nvenc` (so the encoder is not the bottleneck).

Status: **adopted** (on by default), **opt-in** (code exists, off by default), **rejected**
(measured, do not retry *in that configuration* - the reason says which configuration),
**open** (not tested, or tested in a way that does not settle it).

## The model itself: what is left

Per frame the network is 1.85 TFLOP at 1440x1080 (2.48 at 1920x1080). Three same-configuration
absolutes have been measured - 45.7, 51.37 and 57.37 ms - which is **43-54 TFLOP/s**, and the spread
is the host (clock state, no pinning available), not the kernel. That is **58-73%** of a *borrowed*
74 TF/s sustained fp16-accumulate figure. We do **not** have a trustworthy ceiling for this conv
shape - MODEL.md §3 explains why the standalone-conv experiments cannot supply one. So "kernel-level tuning is finished" is not
something the numbers say. What the numbers say is narrower, and still useful:

- the model runs faster than any of its own convolutions measured standalone, and the only
  compute-dominated standalone conv (256->256, 49 TF/s) is *slower* than the whole network;
- eight ideas are closed by measurement (E5, E9', E12, E13, E14, E17, E19 with its tiling
  hypothesis, E24 with its tail hypothesis); what is left of the non-GEMM 8.11 ms of the frame is
  **2.68 ms, 4.7%**: the layout kernels (E18, 0.81), the tail convolution's 0.83 against its 1.09
  traffic floor (E20's tail half - its head half closed with E19), and the upsampler plus myelin's
  reshape/add (E21, 1.04). E16 Winograd and E10 hand-written kernels are untested, and E22 is a build
  path rather than a lever;
- every per-layer time is spread evenly (1.3-3.1 ms over 18 conv subgraphs, top layer 3.15 ms), so
  no single layer is the lever - and `nsys` has since confirmed the shape: 25 launches, 48.7 of
  56.8 ms in the 16 body GEMMs, 8.1 ms in everything else, no launch gap.

The ways to make a frame cheaper are therefore: do fewer frames, do fewer pixels per frame,
or run the same pixels in a cheaper numeric format.

| lever | status | measured |
|---|---|---|
| skip frames that are duplicates of the last rendered one | **adopted** | anime **362/498 frames (72.7%)** reused, 11.7-11.9 s against 38.85 s ungated, 0.009-0.017% of pixels above 8 levels; two digital BDs 146/220 and 142/219 (2.06x, 1.92x); the film clip and 35 mm now **refuse** via the noise floor (byte-identical to ungated). The "46.8% / 71.1 dB" this row used to carry came from two renders both corrupted by C4 |
| recompute only changed tiles, grid fitted to the frame | **open** (was "opt-in, not recommended") | the 512 px grid was a loss because 512 does not divide 1080 (T5); with a fitted 480x540 grid the same mechanism is a win - see T5/T5' |
| batching 2-4 frames per enqueue | rejected | there is no launch gap to fill (`nsys`: 56.8 ms of kernel time in a 57.4 ms loop, 25 launches), so batching can only be linear, and the measured dead end was *concurrent contexts*, not batching: no gain and corrupt output (E13). `nvidia-smi` "util 96-100%" was used as the argument for "already saturated" and does not mean that - it means at least one kernel is alive |
| lower precision (INT8/FP8) | **closed by measurement** | implicit and FP8 were dead ends (E9, E8); explicit Q/DQ is lossless (0.000% above 8, peak 4) and slower - 68.11 against 55.54 ms - because the per-channel PReLU cannot join an int8 epilogue (E9'). The ceiling with a fusible scalar slope is 40.24 ms (1.40x), and that form destroys the picture (-2.02 dB), so the only route left is a plugin with its own int8 convolution |
| a cheaper published model | not applicable | the owner keeps his ESRGAN model set |

## Engine / kernels

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| E1 | builder optimization level 5 instead of 3 | TRT docs; NVIDIA benchmarking page | 1.3-1.6x | isolated engine 94.3 -> 57.5 ms | **adopted** |
| E2 | `num_avg_timing_iters` 8 instead of 4 | torch-tensorrt defaults | few % | included in E1 | **adopted** |
| E3 | `tiling_optimization_level=moderate` | TRT "tiling optimization" docs | 5-20% | 57.5 -> 52.7 ms | **adopted** |
| E4 | clamp the engine to a static shape per resolution | torch-tensorrt docs | removes dynamic-shape tactic loss | used everywhere | **adopted** |
| E5 | `workspace_size` as the reason the body convs got `sm75_` tactics | our tactic dump + the OOM in the standalone builds | - | **measured: it changes the tactic mix and not the time.** Four builds of the same ONNX (8 GB explicit workspace, baseline, cuDNN-only, JIT-only), then three interleaved timing rounds per plan. 8 GB shifts one conv from sm80 to sm75 (17+1 instead of 16+2) and measures 49.84 ms against the baseline's 50.09 (0.995x, inside noise). So the default workspace does truncate the search, and the truncation is worth nothing measurable - the conv that moves is the tail one at 1% of the FLOPs | **closed** (measured) |
| E5' | explicit workspace around the build (the standalone-conv builds hit `CUDA error 2` while autotuning) | our measurement | - | covered by E5 above: the search does change, the speed does not. The OOM was real in the *standalone* engines, where the workspace is shared with the reformat buffers; the full engine has room | **closed** (measured) |
| E6 | CUDA-graph replay of the engine | TAS `upscale/tensorrt.py`; vs-mlrt `trt_utils.h` | 2-4 ms claimed | ~0.3% at 35-45 ms/frame | **rejected** |
| E7 | 2:4 structured sparsity | TRT docs; NVIDIA engineer on GitHub | 2x theoretical | needs pruned weights; TRT picks sparse tactics for 22/53 layers; up to 5% slower in fp16 | **rejected** |
| E8 | FP8 on consumer Ada | TRT issue #4081; OwLite report | 1.4-1.8x | no gain vs FP16 on 4090/4060 Ti; L40S gets 1.7-1.9x. Consumer parts are firmware-capped | **rejected** |
| E9 | INT8 PTQ, implicit with fp16 fallback | mmdeploy's ESRGAN benchmark | 2x ceiling | **slower**: 73.2 and 94.3 ms against 48-51 (the two runs overlapped on the card, so the magnitude is indicative) | **rejected as implemented** - this configuration is a reformat-heavy graph. Explicit Q/DQ NHWC per-channel is E9' |
| E9' | INT8 with explicit Q/DQ, per-channel weights, calibrated on real frames | TRT PTQ docs | 2x of the arithmetic | **measured: accurate, slower, and the mechanism is the boundary, not the kernel.** Calibrated per-tensor activation scales from 8 real frames, per-channel int8 weights, 16 body convs quantised, head and tail in FP16: **0.000% of pixels above 8 levels against the FP16 engine, peak 4, PSNR 54.1 dB** - the format is essentially lossless here. Time **68.11 ms against 55.54 (0.82x)**. Per-layer (`trt.IProfiler`): the 16 int8 convs are `sm80_xmma_..._interleaved_i8f16_i8i32_f32_nchw_vect_c_...` - the **interleaved NC/32HWC32 tensor-core layout, not plain NCHW** - at 1.56-1.88 ms each, i.e. 152.9 GFLOP / 1.68 ms = **91 TOPS against 52 TFLOP/s for the fused fp16 block**; the PReLU runs as **17 separate `PointWiseV2` passes** of 1.71-2.37 ms (31.2 ms, 46% of the engine) because a per-channel slope (1088 values, -1.39..1.70, per-channel Cv > 1) does not fit an int8 epilogue. In fp16 those 16 blocks are one fused myelin kernel each (`__myl_CorrMulLtSele`, 2.93 ms). The engine holds **21 reformat layers of 78**, worth **5.6 ms in total** (8% of the engine) against **32.7 ms for the 17 PReLU passes** (46%) - the "60 reformats" in earlier revisions was 78 minus 18 convs. Second build, quantising the conv output too so the PReLU reads int8: **78.36 ms**, 36 reformats, 16 `DQout` passes at 1.64 ms - the boundary pass is charged per hop, so adding a hop makes it worse | **rejected as quantised** (measured, with the mechanism): the per-channel PReLU forces one int8->fp16->int8 hop per conv, and each hop costs about what int8 saves on the conv. Precision path closed, pending a plugin that fuses conv+PReLU in int8 |
| E10 | hand-written conv kernel / plugin | CORE-TAG/rtx50-realesrgan-video | - | 39.8 TFLOP/s vs the engine's 186.7 on a 5090. Counter-precedent: for a 3x3 conv cuDNN chose Winograd at 74 ms where TRT's best tactic was 156 ms, so TRT's tactic set is not the set of all algorithms | **untested** (was "rejected": the source is one attempt on another card, and it does not measure this GPU) |
| E11 | L2 persistence window | blog measurement (H100/3090) | 1.02-1.20x | nothing to pin: 1.2 MB of weights. The old justification ("265 MB activation stream") predates the myelin fusion finding - the engine is 25 top-level layers, not 18 separate kernels with a full round trip each - so how much traffic actually remains is unknown; the verdict rests on the weights being 1.2 MB against **32 MB** of L2 (AD106; 24 MB is the 4060 without Ti), not on the traffic count | **rejected** (justification corrected) |
| E12 | per-layer precision constraints (force fp16 accumulate) | TRT docs | unknown | closed by measurement, not by an API limit: the DETAILED tactic dump shows all 16 body convs already running `f16f16_f16f16` (fp16 accumulation, NHWC). There is no fp32-accumulate share to convert | **closed (already in effect)** |
| E19 | make the one unfused PReLU fold into the convolution before it, by padding the head to 8 input channels | `nsys` per-kernel trace, then the engine's own subgraph metadata | 2.40 ms (4.2% of the frame) | **closed and shipping: -3.5% of engine time through this app's build path** (11.79 -> 11.44 s wall, 16.18 -> 15.61 ms engine, interleaved three rounds, 3/3 on both numbers; picture 0.003% median / 0.012% worst / peak 15). The mechanism was measured on a raw TRT build first: the head is the only 3-channel convolution and at C=3 it does not enter myelin at all (a minimal Conv(3->64)->PReLU->Conv(64->64) builds as 2 `CaskConvolution` + a separate `PointWiseV2`), while at C=8 the same graph becomes one `fusion __myl_CorrMulLtSele` covering the conv *and* its PReLU (11.21 -> 9.59 ms, -14.5%); on the real graph the three head layers collapse into that one fusion and the pad folds away. The route to the product was blocked for two rounds by the operation rather than the graph - `F.pad` has no dynamo converter and split the engine (+67%), `torch.cat` does not (E23). A probe forward reverts the padding for structures it cannot widen, and the 78-model sweep below says which those are. | **closed (shipped on by default; `RVE_NO_HEAD_PAD=1` disables)** |
**BUT it does not survive this app's build path, and the difference is the builder, not the idea.** The same padded graph compiled
the way the app compiles (`torch.export` + `torch_tensorrt.dynamo.compile`, the padding expressed as a `PadInputConv` wrapper inside the head so the
model's own residual keeps its 3 channels) measures **15.42 -> 26.40 ms of engine time (+71%)** and **11.40 -> 16.75 s** of wall clock, three runs
each, interleaved, with different engines confirmed by the `_headpad5` cache marker. So the 5.8% is a property of the raw TRT build and not of the
model: shipping the padding would need this graph built by the raw builder (or the partitioner fixed), which is a change of build architecture, not
a flag. The code ships **off** (`RVE_HEAD_PAD=1` opts in) with four unit tests, and the picture through the app differs by 0.011% of pixels at the
worst frame - rounding, not the padding. | **engine-level win measured, not shippable through the app's build path** |
| E20 | head and tail convolution tactics | `nsys` | ~1.31 ms | both run `indexed_wo_smem` at 1.45 and 1.92 ms against a traffic floor of about 2.06 - the gap is 1.31 ms across the two, and it is the same tactic question as E19 | open |
| E21 | express the nearest upsampler as a shuffle instead of a gather | `nsys`: `ResizeNearest2D` 0.37 + `__myl_TranReshMoveAdd` 1.06 | ~1.04 ms | 1.43 ms against a ~0.39 floor. Nearest upsample is a pure gather, and `Reshape (1,C,H,1,W,1) -> Expand (...,2,...,2) -> Reshape (1,C,2H,2W)` is the same function in shuffle form, which TRT's shuffle fusion folds into the consumer's indexing - the 0.37 ms kernel then disappears without a new kernel. That is separate from the DepthToSpace idea, which is about PixelShuffle and not about the residual path | open |
| E18 | NHWC input/output for the engine (removes the layout kernels) - **and see E19: this and the head padding are one operation** | nsys per-kernel trace of the fp16 plan | ~1.5% (0.81 ms/frame) | measured cost, not yet removed: `cuInt8::nchwTonchw<__half, float, 4>` 0.61 ms and `CUTENSOR permutationKernelPLC3` 0.20 ms run once per frame, and no layer in `IProfiler` owns them. It is the only clean ~1.5% left in the engine | open (small, well-defined) |
| E13 | concurrent engine contexts (batch/streams) | vs-mlrt wiki: 1.65 -> 2.25x on a 2nd stream | up to 2x | no gain (4.25 -> 4.27 s) and output corrupt (25 dB) - because two `torch.jit.load` modules of one engine file are not independent contexts | **rejected as wired** (a wiring fault, not a GPU property) |
| E14 | alternative tactics for the 16 body convs | our tactic dump: 17 of 18 convs get `sm75_` kernels on an sm89 part | unknown | **measured and closed.** With the tactic sources restricted to JIT convolutions only (no cuDNN, no cuBLAS) the winner is the *same* `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32`, at 49.97 ms against the baseline's 50.09 (0.998x). So there is no sm80/86/89 alternative being skipped for this shape at K=576 in this TRT build; the `sm75_` name is the name of the winner, not a symptom. Restricting to cuDNN alone is 10.7% **slower** (55.43 ms, stable across three rounds), so TRT's own kernels beat cuDNN here | **closed** (measured) |
| E15 | crop the active picture area (letterbox bars) | - | up to 1.33x if 25% of the frame is black | **n/a on our material**: the film clip (1440x1080) and the anime clip (1920x1080) both have zero rows darker than 6/255, minimum row mean 59.9 and 124.8. Worth one row-histogram check per new source before rendering | open (source-dependent) |
| E16 | Winograd F(2,3)/F(4,3) | cuDNN/CORE-TAG counter-precedent | F(2,3) cuts MACs 2.25x, inflates tensors ~4x | not measured. The ridge is **128 MAC/byte** at the borrowed 74 TF/s and 288 GB/s (257 FLOP/byte), not 42: 42 corresponds to a ~24 TF/s fp32-shader peak, so the earlier figure was off by 3x in the wrong direction. **The intensity objection was withdrawn 2026-11-15**: Winograd's transform stages are memory-bound at the shared-memory/L2 level, and that is a cache-blocking problem the literature fixes (PPoPP'20: b_k 32->64 raises intensity 8 -> 10.67 ops/byte; PKF fuses the stages by memory level), not a property of the algorithm on this GPU. The cuDNN-only build being 10.7% slower is E14's measurement of cuDNN's *default* algorithm, not of a forced Winograd kernel, so it does not close this row either. Cheapest test and the numbers behind the reclassification: §What the literature closes, and the one item it reopens. **Measured 2026-11-15 (E26): no cuDNN-selected algorithm reaches our kernel** - full autotune on the exact shape gives 3.67 ms against the engine's 3.05, flat from 1080p down to 135x240. Closed as negative through cuDNN/TRT; what remains open is a hand-written kernel, i.e. E10's cost class | **closed (negative, measured)** |
| E17 | rewrite PReLU as `a*x + (1-a)*Relu(x)` | TRT supported-op patterns | maybe changes tactic eligibility | **measured, as the unlock E9' was missing - and it is not one.** The rewrite does not remove the boundary: the `a*x` term still needs the pre-ReLU frame, and the `(1-a)*Relu(x)` term gives a second full-frame pass, so it trades one 1.8 ms PReLU pass for two. What would remove the boundary is a fused int8 conv+PReLU epilogue, and this build has none for a per-channel slope (measured: the int8 conv reports `HasReLU: 0, Activation: NONE` and the PReLU appears as its own `PointWiseV2` layer). The earlier guess that it was "cheap and near zero" was right about the cost and wrong about the mechanism | **closed together with E9'** |

## Precision / colour (10-bit)

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| C1 | pass the source's colour matrix/range to ffmpeg | ffmpeg/swscale docs | removes chroma shift | \|dU\| 6.64 -> 0.2 codes; 8-bit path 40.25 -> 44.06 dB | **adopted** |
| C2 | keep 10 bits end to end (pack YUV420p10 on the GPU) | waifu2x postproc shader uses a GPU colour stage | parity with ffmpeg | 49.69/55.34/54.84 dB vs the ffmpeg path | **adopted** |
| C3 | pipe `rgb48le` (the app's old "16-bit" path) | - | better than 8-bit | **worse**: 39.0 dB and a -5.9 level shift, because of the missing matrix | **rejected** |
| C4 | pipe P010 for NVENC (avoids a 1.54 ms/frame repack) | ffmpeg encoder caps | ~12% of the NVENC path | **two separate faults, both found and fixed.** (a) P010 was not in the "already YUV" branch, so it got `scale=in_range=pc:out_color_matrix=...`: a full-range reinterpretation of a limited-range stream plus a second matrix. (b) worse, `_pipe_input_pixel_format()` derived the format from the *encoder* tag ("NVENC wants P010") while the packer emits planar `yuv420p10le` unless `RVE_P010` is set, so **every 10-bit source rendered with NVENC fed planar samples labelled P010 and came out at 12.68 dB**. With the command following the packer and P010 opt-in: 52.01 dB against a libx265 render at 10-bit output, 46.11 dB at 8-bit output. A format disagreement cannot fail loudly here, both are 16-bit words at 3 bytes/pixel - which is why it survived a session. (c) the colour tags were declared on the *output* side, which makes ffmpeg convert every frame to the declared space: 79.0 ms/frame against 16.6 ms/frame with the tags on the input, and a chroma shift of 5.3 levels in U because already-bt709 samples were converted from bt601. Same tags in the container either way. See the investigation for the bisection | **fixed** (measured); P010 itself is **off, measured**: against the corrected planar path a full render is byte-identical in picture (0.000% above 8 levels) and **0.5 ms/frame slower** (28.16 against 27.65 ms on 220 frames of 10-bit), because the repack it was meant to avoid no longer exists |
| C5 | convert YUV->RGB on the GPU on the input side | same as C2 | halves the input pipe | works, colour matches; costs ~1.2 ms/frame | **adopted** |

## Temporal / dedup

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| T1 | frame-level reuse vs the previous frame, 0.35 threshold | our first attempt | 20-40% | 24.7% reused, 0.5 s saved | superseded by T3 |
| T2 | exact-duplicate (bit-identical) gating | video2x issue #270 discussion | large | 0% bit-identical frames on encoded video | **rejected** |
| T3 | reuse vs the **last kept** frame, 32 px block means, threshold chosen automatically, chain 4 | TAS `dedup/dedup.py` + `PARAMETERS.MD` (sens 35 / 7) | 20-40% frames | **anime**: 70.9% reused, worst frame 48.64 dB against ungated. **film**: 18.1% reused, average 50.34 dB and worst 39.83 dB against ungated, 4.33 s against 4.66 s (1.08x). The earlier "46.8% reused, 71 dB" was measured against a *broken* pipe (see C4): both compared renders were equally corrupted, so the pair agreed with itself | **adopted** |
| T4 | full-res per-pixel veto at 40 levels | TAS `PixelDiffGuard` | rescues mouth flaps | on grainy film a single pixel exceeds it: 43 vetoes, reuse 1.1%, render 5.08 s | opt-in (`--temporal_reuse_pixel_veto 40` for anime) |
| T5 | tile-level reuse with an 18 px halo, **square 512 px cores** | TAS/waifu2x tiling; our halo measurement (radius 18 LR px) | 2-4x on static content | 65.8% of tiles reused on anime and *still* slower than the full-frame call (21.37 s vs 30.8 ungated, but 10.6 s for the frame gate); film 27.7% reused, 6.57 s vs 4.27 | superseded by T5' - the loss was the grid, see below |
| T5' | same mechanism, but the core grid **divides the frame** | this ledger's area model, checked against T5's own timings | 2.55x on the model at 65.8% reuse, instead of 1.68x | **18.14 s** ungated against 21.37 s for the 512 grid: the geometry fix alone is 1.18x on the tile path. On top of the frame gate it reads 7.68-9.15 s against 11.7 s for the gate alone = up to **1.4x**, but every one of those runs has tile reuse enabled, which is what the artifact section shows is visible (0.287% of pixels above 8 levels, peaks of 98). Tile reuse falls from 65.6% to 57.4% inside the gate, because the frames the gate leaves are exactly the ones that changed. Those 7.68-9.15 s figures predate the mask **dilation**, which is what actually ships in `--tile_reuse_mode auto`: dilated, the same path reads **10.0 s at 0.071%** (peak 23) | **opt-in, off by default.** With the frame gate out of the way the trade is explicit: undilated 9.2 s against 11.7 s on 502 anime frames, at **0.287%** of pixels above 8 levels (RGB, worst frame, peaks of 98) against 0.009%; dilated - the shipping opt-in - **10.0 s at 0.071%**, still eight times the defect area for 14% of the clock. Its per-tile mean cannot be tightened into safety - at threshold 0.10 reuse falls from 57.7% to 3.0% and the render ends up slower than not using the tile path at all - and switching the judge to the worst block made the clip worse (1.174%). `--tile_reuse_mode auto` keeps the pre-render probe for material where the trade is wanted |
| T6 | blend or discard the tile overlap | vs-mlrt discards, chaiNNer blends sinusoidally | - | we crop (discard), which is exact thanks to the halo | **adopted** |
| T7 | per-slot HR memo: copies of one held frame share one upscale | TAS `pipeline/executor.py` | - | our gate already skips the whole model for a duplicate | equivalent |

**Why T5 lost, and what T5' measured.** The cost of the tile path is `area x (1 - reuse)`,
with `area` the halo-inflated grid area over the frame area:

| grid | tiles | area with halo 18 | break-even reuse |
|---|---|---|---|
| 512x512 @1920x1080 | 4x3 = 12 | **1.738x** | 42.5% |
| 1024x1024 @1920x1080 | 2x2 = 4 | 2.167x | 53.9% |
| 1024x1024 @1440x1080 | 2x2 = 4 | 2.890x | 65.4% |
| **480x540 @1920x1080** | 4x2 = 8 | **1.147x** | **12.8%** |
| **480x540 @1440x1080** | 3x2 = 6 | **1.147x** | **12.8%** |

That model, with no free parameters, reproduces the measured tile renders: the 512 px grid
predicts 21.49 s and the render took 21.37 s (0.6%), the 1024 px grid predicts 24.76 s against
26.14 s (5%), and the fitted grid predicts 16.9 s against 18.14 s measured. So there is **no
per-tile launch penalty worth modelling** - the entire cost of T5 was the clipped tail of a
grid that does not divide the frame (512 does not divide 1080: the third row is a 56 px core
inside a full 548 px window). `divisor_tile()` in `backend/src/pytorch/TiledUpscale.py` picks
the core that divides each axis.

Two reasons the combination is 1.5x rather than the multiplicative 1.5-2x: the tile share of
reuse drops from 65.6% to 57.4% once the gate runs in front (the frames left to render are the
ones that changed), and 36.9% of those frames still come out fully reused by the tile path. Those
figures are for a chain of 4, which is what `best` certifies on this clip; on the two digital BDs it
certifies chain 2 and the artifact section below explains why the third one (35 mm) refuses outright.
The measured composition, 502 anime frames, same session:

| configuration | wall | loop | frames rendered | tiles reused |
|---|---|---|---|---|
| ungated, full frame | 38.85 s | - | 100% | - |
| tiles only, fitted grid | 18.14 s | 35.06 ms | 100% (56% of them fully reused) | 65.6% |
| gate only | 11.66 / 11.74 / 11.84 s | 22.6-22.9 ms | 27% | - |
| gate + fitted tiles, chain 4 | **7.64 / 8.42 / 8.45 / 8.48 / 8.65 s** | 14.6-15.4 ms | 27% | 57.4% |
| gate only, no tile reuse (the default now) | 11.7 s | 22.8 ms | 27% | - |

Median against median is 11.74 -> 8.45 s = **1.39x**; the best pair in the same window is
7.64 against 11.66 = 1.54x. Those runs all used tile reuse, which is now off by default because
of the artifact in the investigation: the configuration that ships is the gate alone at **11.7 s**
(3.3x over ungated) with 0.009% of pixels above 8 levels. The spread inside one configuration is 10%, which is this
machine's run-to-run noise, so the honest statement is "1.4x, up to 1.5x".

On the moving film clip (94 frames, 1440x1080) the same combination is **neutral**: 4.07 s for
the gate against 4.14 s with tiles, tile reuse only 9.5%, quality 68.46 dB average against
69.29 (worst frame 65.2 against 65.3). That is the selection effect in its purest form: the
frames the gate renders on this clip have essentially no unchanged area left.

That is why the default is a measurement rather than a flag: `TileProbe` decodes 16 frames at
4 spread positions, replays the gate's and the tile path's statistics on them, and reports the
predicted cost of the tile path as a fraction of one full-frame call. Two things make it
dependable. The tile statistic has to be the mean of the *absolute* block deltas, not the
difference of the tile's two means: the latter lets local changes cancel and predicted 38%
reuse where the render then achieved 9.5%. And equality is not a safe threshold, because a
16-frame sample mispredicts by a few points in both directions (anime 33% predicted / 58%
achieved, film 15% / 9.5%), so tiles must clear `1 / area_tax` - a saving at least as large as
the halo tax - instead of the bare break-even.

## Pipeline / IO

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| P1 | remove the double clone of the output (`UpscaleModelWrapper` + `Frame`) | our measurement | 4.6 ms/frame at 4x | 1.5 ms at 2880p, bit-identical output | **adopted** |
| P2 | stop building previews nobody consumes | our measurement | large on 16-bit frames | 10 s of 21 s on the naive 10-bit path | **adopted** |
| P3 | drop the extra `numpy -> bytes` copy of the frame | TAS keeps frames as CUDA tensors | 2-3 ms | 0.79 ms at 2880p; output bit-identical | **adopted** |
| P4 | pooled pinned buffers with a semaphore, writer releases the slot | TAS writer; our design | removes a full copy | 12.6 dB against the reference (slots rewritten before ffmpeg consumed them) and slower (4.83 vs 4.21 s) | **rejected as wired** - and the overlap it was meant to buy already exists, see P4' |
| P4' | overlap the encode with the model (writer thread + bounded queue) | our writer instrumentation | 1.44x if the encoder serialised behind the loop | **already overlapped**: on the 502-frame anime render the writer spends 12.34 ms/frame blocked in `write()` (ffmpeg draining) and **10.63 ms/frame waiting for frames** - it is idle half the time, queue depth max 4, and its 6.20 s of busy time sit inside the 11.74 s render. The render loop is 22.72 ms/frame and the writer has slack | **not a lever** - the "13 ms not covered by the render loop" in the budget table is fixed startup on a 94-frame clip, not serialisation |
| P5 | remove the ffmpeg subprocess, encode in-process | TAS uses in-process NeLux | 45 -> 20-25 ms | PyAV 19 wheel has no `av.CudaContext`; `hevc_nvenc` will not open; libx265 in-process is only +2% (10.35 vs 10.58 s) | **rejected** (the +2% bound is the whole upside, since the pipe already overlaps) |
| P6 | drop the device-wide `torch.cuda.synchronize()` per frame | TAS orders with `wait_stream` only | 2-4 ms | within noise here (not model-bound) | **adopted** (harmless) |
| P7 | one full-res pass per frame instead of frame-on-disk | video2x 4.8.1 does 4 passes | large | we already pipe raw frames | n/a |
| P8 | reflect-pad to a bucket instead of one engine per resolution | TAS `upscale/tensorrt.py`; chaiNNer `remove_padding` | avoids rebuilds | not measured | open |
| P9 | write tiles straight into the destination plane | waifu2x `postproc.comp` | saves a pass | we composite on GPU already; per-plane write untested | open |
| P10 | `F_SETPIPE_SZ` 1 MB, one `os.write` per frame | pipe throughput reports | 2-5 ms claimed | with ffmpeg as the reader: +6% (noise) | **rejected** |
| P11 | in-app model time vs the isolated benchmark | our profile | - | **answered** (see the open question below: 51.37 ms standalone against 54.3 in-app). **split measured**: with `RVE_TRT_SPLIT=1` the engine alone costs **14.67 ms/frame** averaged over all frames = **54.3 ms on the 27% that run**, against 42.7-45.7 ms isolated for the same shape (+19-27%). Without the forced sync the same block measures 16.10 ms/frame = 59.6 ms, i.e. the output Frame's construction and handover add ~5 ms and **overlap with the kernels**, which is why the reported "inference" was never the engine time. `last_inference_ms` = enqueue + ctor + sync (engine 0.30 ms at enqueue); `last_model_ms`/`last_frame_ms` now separate them | **closed** - the 19-27% "gap" was this measurement's own bookkeeping (enqueue + construction + sync, not the engine); with NVENC alongside, 51.37 ms standalone against 54.3 in-app at the same shape and the encode costing 9% in a controlled A/B (51.37 -> 55.99) |
| P12 | cheaper gate: no clone, block means without a full-frame fp32 luma | our profile | 1-2 ms/frame | **rejected as a lever.** The stage reads 4.37 ms/frame, but 2.80 ms of that is `get_frame_tensor()` (the 10-bit upload + colour stage), which every configuration pays on its first read of the frame; the gate's own statistic measures **0.80 ms** standalone (clone 0.017, luma fp32 0.43, pool 0.40, `max().item()` 0.03). So the clone is worth ~0.02 ms and the whole optimisation ~0.7 ms of a 22.6 ms frame = 3%. The number that looked like a 19% block was mostly the input pipe | **rejected** (measured) |

## Encoder

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| N1 | use NVENC instead of `libx265 medium` | our measurement | removes the bottleneck | **11.18 -> 4.27 s** on the clip (encode alone 9.6 -> 1.2 s) | **adopted** (advice) |
| N2 | `--frame-threads 4..6` for x265 | x265 threading docs | -4% | measured -4% | open (only if x265 is used) |
| N3 | anime flag set instead of `--tune animation` (`aq-mode 3`, `psy-rd 1.0`, `bframes 8`, `deblock -1:-1`) | JET guide, kokomins | same time, less banding | `--tune animation` measured +8% time, -6.5% bits and is wrong for flat gradients | open |
| N4 | `av1_nvenc` | IEEE Access 2025 BD-rate table | quality/size, not speed | -34% BD-rate vs hevc_nvenc | open (quality, not time) |
| N5 | p010 for NVENC to skip the repack | ffmpeg caps | 1.54 ms/frame | see C4 | blocked on C4 |

## Current per-frame budget (measured by the built-in profiler)

Two configurations, because they answer different questions.

**Anime, 502 frames, 1920x1080 -> 3840x2160, gate at 73.5% reuse, `x265_nvenc`:**

| stage | ms/frame | note |
|---|---|---|
| duplicate gate | 4.37 | of which **2.80 is `get_frame_tensor()`**: the 10-bit upload + colour stage, paid once per frame by whoever reads it first. The gate's own statistic is 0.80 ms standalone (P12) |
| model + stream sync | 16.31 | over all frames; of that the **engine alone is 14.67 (P11), i.e. 54.3 ms on the running share** - the remaining 1.64 is stream sync and the output Frame's construction, and it overlaps with the kernels |
| read wait | 0.26 | the decoder is not a factor |
| render loop total | **22.72** | |
| writer blocked in `write()` | 12.34 | concurrent with the loop, 10.63 ms/frame idle - not on the critical path |
| process start -> first frame | ~? | measured by the `startup:` line now in the log |

**Film, 94 frames, 1440x1080 -> 2880x2160, gate at 46.8% reuse:** gate 3.5, model 22.7
(43 ms on the 53% that run), pack + D2H + queue ~3, read wait 0.5.

**Where the 22.7 ms of "model" actually goes (anime, gate only, `RVE_TRT_SPLIT=1`):** engine
14.67 ms/frame over all frames, handover 0.01, and the difference to the 16.10 ms of the
unsplit run is the output Frame's construction, which overlaps the kernels. Per rendered frame
that is 54.3 ms of engine against 42.7-45.7 ms isolated.

The old table added a "~13 ms not covered by the render loop" row and read it as encoder
serialisation. It is not: `3.5 + 22.7 + 3 + 0.5 = 29.7 ms` of loop against 45.4 ms/frame of
wall clock on a 94-frame clip is 94 x 29.7 = 2.79 s of loop inside a 4.27 s render, and the
difference is fixed cost (engine build/deserialize, ffmpeg + NVENC init, flush) amortised over
only 94 frames. The 502-frame run shows it directly: 22.72 x 502 = 11.40 s of loop inside an
11.74 s render.

## Backlog, ranked by expected gain on our pipeline

Nothing in the model is a measured loss any more, and the last precision row closed on measurement
rather than on a hunch.

1. ~~E9' - explicit Q/DQ INT8~~ **measured, and it is not 2x away: it is one epilogue away.** The
   format is essentially lossless (0.000% of pixels above 8 levels, peak 4, 54.1 dB), the int8 conv
   is fast (91 TOPS against 52 for the fused fp16 block), and the build is still slower - 68.11
   against 55.54 ms - because the per-channel PReLU cannot be fused into the int8 epilogue and
   becomes a 1.8 ms pass between two int8 convs. Quantising the conv output as well (so the PReLU
   reads int8) makes it worse, not better: 78.36 ms with 36 reformats and 16 `DQout` passes. The
   arithmetic in this row was right and the conclusion was wrong: the 2x is in the convolution and
   the gate is the *boundary*, which TRT charges per hop. Reachable only through a plugin that fuses
   conv+PReLU in int8 (or a scalar-slope approximation, which the measured slopes rule out:
   -1.39..1.70 with per-channel Cv > 1). So: closed as a TRT-path idea, open only as plugin work,
   which is a different kind of project from the ones in this ledger.
2. ~~The input conversion~~ **measured: it is PCIe-bound, not kernel-bound.** The 10-bit
   conversion is 1.42 ms standalone at 1080p for 6.2 MB of packed planes, and the H2D of those
   planes at ~6 GB/s is about 1 ms of that. Rewriting it to one preallocated buffer with `g`
   computed from `y,u,v` (instead of through `r` and `b`, which materialised two extra full
   frames), both chroma planes upsampled in one call and no `stack`, is correct to 0.02 levels
   mean (max 0.37, fp16 rounding) and takes **1.26 ms against 1.42 ms** - 11%, not the 10x that
   the earlier "30x above the bandwidth floor" estimate implied. That estimate counted GPU-side
   bandwidth and forgot the frame has to cross PCIe first. The `gbrp` idea (planar decoder, no
   permute) is the same order and does not justify reworking the preview and cv2 consumers.
3. **E16 Winograd - reclassified 2026-11-15: the largest untested lever, not "expected near zero."**
   The old wording rested on one measurement: restricting the build to cuDNN alone made the engine
   10.7% *slower*. That measures **cuDNN's default algorithm choice**, not a forced Winograd kernel,
   and cuDNN's default for this shape is its own implicit GEMM - so the measurement is silent about
   the algorithm it was taken to rule out. The external picture points the other way: Welder
   (OSDI'23) loses to TensorRT precisely on conv-dominated stacks (UNet 9.251 against 4.429 ms,
   VGG16 4.123 against 2.584 ms) and its authors attribute the loss to TensorRT using Winograd while
   Welder has none; PPoPP'20 measures cuDNN Winograd 1.54-1.59x over GEMM on a `[64,3x3,64]` layer,
   which is our exact body shape. Our tactic dump shows all 16 body convolutions on `implicit_gemm`
   and no Winograd, which is consistent with either "TRT has no Winograd tactic for this shape" or
   "it has one and the search never picked it" - two different findings.
   The cost is one hour: force `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD` on a single 3x3 64->64 conv at
   1920x1080 fp16 NHWC, with fp32 NCHW as a control, and print the algorithm the v7 query returns.
   Both outcomes are informative and neither is currently known. Queue entry:
   `docs/backlog/queue.yaml` `rve-winograd-tactics-2026-11-15`.
   (E17 used to be listed here as a micro-optimisation; it is now part of E9' above, where it turned
   out not to unlock anything.)
4. **P8 / P9 / N2 / N3 / N4** - quality-of-life and encoder-side, only if x265 or per-plane tile
   writes are kept.

## Which metric answers "is it visible" (2026-11-15)

PSNR alone was the wrong instrument for a reuse decision, and the numbers show why. The same
three pairs, compared with `libvmaf` (model v0.6.1neg, both inputs scaled to 1080p because that
is what the model is calibrated for):

| pair | PSNR avg | PSNR worst | VMAF mean | VMAF p5 | VMAF min |
|---|---|---|---|---|---|
| anime, gate + fitted tiles | 51.71 | 48.64 | 95.50 | 94.18 | 93.65 |
| film, gate | 50.34 | 39.83 | 96.70 | 88.81 | 83.79 |
| P010 after the fix vs a libx265 render | 52.01 | 50.40 | 96.24 | 95.65 | 95.63 |

The film row is the lesson: its *mean* is the best of the three (96.7) while its minimum is the
worst by 10 points, and the minimum is what a viewer sees on the frame where a reused edge landed
in the wrong place. The pixel-level peak agrees: the worst block of that film frame is **196
levels** off (mean 1.74), against 100 levels for the anime frame (mean 0.98).

VMAF is **necessary but not sufficient**, and this is measured, not assumed: the anime
configuration a viewer rejected (a moved fist, chain 4 with tile reuse) scores **min 93.65 / p5
94.18**, i.e. it passes a `min >= 90` floor. Whole-frame metrics are blind here because the defect
is a fraction of a percent of a 4K frame, and VMAF at 1080p downscale makes it smaller still.

The deciding number is local and in RGB: **share of pixels more than 8 levels off in any channel, on
the worst frame of a spread sample, must stay under 0.1%**. Measured: **0.009%** for the shipping
configuration and 0.287% for the rejected tile configuration. VMAF min/p5 and PSNR are
reported next to it as second opinions, and the worst frame's PNGs are kept for eyeballing
(`remastering/compare_frames/`).

## What the reuse artifact actually is (2026-11-15)

Chased from a viewer's report: on the worst frame of the gated anime render a fist sits in a
different pose than in the ungated one. Five measurements, in the order that narrowed it:

1. **A/A control.** Two runs, same input, same flags: **502 of 502 frames byte identical**. The
   pipeline is deterministic, so every difference between configurations is real.
2. **Global alignment.** Frame i against frames i-2..i+2 of the other render: the peak is at
   offset 0 in both clips (anime 55.2 dB at 0 against 52.5 at -1). The pair is not time-shifted.
3. **Patch-level probe** at the reported region: nearly identical PSNR at every offset from -5 to
   +5, i.e. the content matches no neighbouring reference frame - so it is not "the frame is one
   late" either.
4. **Tiles against full frame**, same short clip, halo 18 / 24 / 32: 51.34 / 51.36 / 51.36 dB,
   worst frame 49.84 / 49.94 / 49.94, a 2D shift search peaking at (0,0) with 49.53 dB, and the
   error at tile boundaries (0.496 levels) indistinguishable from inside the tiles (0.477). So the
   tile path is geometrically exact; its difference from the full-frame render is fp16-level noise
   with no seams, and the halo is not the limit.
5. **Chain depth**, same frame 383, share of pixels by error magnitude:

| threshold | chain 4, tile reuse **on** | chain 1, tile reuse **on** |
|---|---|---|
| > 2 levels | 10.87% | 8.91% |
| > 4 | 1.13% | 0.39% |
| > 8 | 0.29% | **0.00%** |
| > 16 | 0.12% | **0.00%** |
| peak in the worst block | 100 levels | **12 levels** |

So inside the tile configuration the artifact is the **chain depth**: a reference four frames old
moves a fist, and this is the only place it shows at 0.29% of pixels, because the tile path's own
staleness masks it everywhere else (0.288% against 0.287% across a tenfold threshold change).
`best` certifies the depth itself (chain 4 once the chain budget is expressed on the proxy's own
scale, chain 1 when it is held to the one-step bar) and `fast` keeps the full chain.

Two honest limits on that paragraph, both found by re-reading the table against the shipping
configuration:

- The table's absolute column is **tiles on**. Shipped is tiles off, where chain 4 measures
  **0.009%** of pixels above 8 on the worst of 19 sampled frames - so what the table isolates is
  the depth's *contribution inside the tile path*, not its price on its own.
- **The clean measurement exists now, and it changes the reading.** Chain 1 with tiles off, same
  clip: **212/498 reused (42.6%), 17.9 s, 0.016% of pixels above 8 (peak 15)** against the shipped
  chain 4 at **362/498 (72.7%), 9.97-12.03 s, 0.017% (peak 17)**. So with the tile path out of the
  way the depth costs **nothing measurable** and is worth 1.5-1.8x in time (chain 1 reuses half as
  much, so it renders nearly twice as many frames); with tiles on the same pair measured 0.29%
  against 0.00%. The two sources do not only mask each other - they **compound**: a four-frame-old
  reference served into a stale tile is what moves the fist, and neither alone does. (The
  11.7-vs-10.0 pair that used to be here mixed a tiles-off number with a tiles-on-and-dilated one.)

- **On real material the depth was not free because the rule was wrong, and fixing it paid 1.5x.**
  Forcing chain 4 by hand measured **148/220 (67.3%) in 5.82 s at 0.005% (peak 14)** on Chainsaw Man
  - the same defect as its chain-1 run to the last digit - while Violet Evergarden measured
  165/219 in 4.97 s at **0.124%**, past the bar. So the depth was worth about 1.7x on one title and
  unsafe on the other, and the tuner certified chain 1 on both.

  The cause was not the budget (budget 4 and chain budget 30 changed nothing) but **reachability**:
  to serve a reference `k` frames old the gate must have reused the frames in between, and the tuner
  scored depth-`k` deltas even when the intermediate frames would never have passed the threshold.
  The per-frame debug shows it plainly - on Chainsaw Man, calm frames read `0.2` at depth 1 and
  `16-35` at depths 2-4 because each sits next to a one-frame flash, and the p90 of those
  unreachable reuses exceeded every budget. A second, older bug sat behind it: the scene-cut test
  (`|mean - ref_mean| > 32`) lived inside an `elif` that stopped being taken when the statistic
  became `mean_abs`, so it had not run since that change.

  Both are fixed (`AutoGateTuner._decide` filters unreachable depths; the cut test moved out of the
  branch; the gate exposes `last_scene_cut` and the app resets the tuner's HR history there). With
  16-frame calibration frames per title, on the same clips and the same preset:

  | title | before | after | defect before -> after |
  |---|---|---|---|
  | Chainsaw Man ep01 | chain 1, 83/220, 9.94 s | **chain 2, 146/220, 6.56 s** | 0.005% -> **0.005%** (peak 15 -> 14) |
  | Violet Evergarden ep01 | chain 1, 107/219, 8.58 s | **chain 2, 142/219, 6.92 s** | 0.097% -> **0.096%** |
  | Cowboy Bebop ep01 | refused | refused | 0% both |

  So 1.5x on the first title and 1.24x on the second, with the defect unchanged (Violet's is a
  hair lower). Those two rows are the **acceptance-certified** run (the shipping rule); the
  proxy-certified run of the same code read 6.56 s and 6.92 s with identical counts, i.e. the two
  instruments agreed here. The tuner stops at chain 2 because deeper depths run out of *reachable samples* in a
  16-frame window, and that turns out not to matter much: forcing chain 4 by hand buys only 6.56 ->
  5.82 s on Chainsaw and makes Violet worse (0.124%). Chain 2 is the sweet spot it now finds by
  itself.

Acceptance for this family of defects is therefore local and absolute, not a mean: **under 0.1%
of pixels more than 8 levels off in any channel, on the worst frame of a spread sample, with a
peak under 20 levels**. Whole-frame PSNR and VMAF both pass a frame that fails this, because a
fist is a fraction of a percent of the frame. (Two stricter-sounding bars used to live here - "no
pixel above 8" and "under 0.5% above 4" - and the first is contradicted by every render we have:
the shipping configuration measures 0.009% of pixels above 8 on its worst frame, which is small but
not zero, and 0.105% on material with mild grain. Both bars were written from an eyeball rather
than from a measurement of the config that ships.)

## Three real Blu-rays, and the rule that did not transfer (2026-11-15)

The acceptance rule and the presets were derived from two clips. Three actual BDs, 8 s cuts at
5:00, `--quality_preset best` against an ungated render of the same cut:

| title | noise floor | decision, reuse as a count | time | RGB > 8 levels, worst of 15 frames |
|---|---|---|---|---|
| Chainsaw Man ep01 (digital, added grain) | 0.95 | thr 4.0, chain 2 - **146/220** (66.4%) | 6.61 s against 13.50 s (**2.04x**) | **0.005%** (peak 14) |
| Violet Evergarden ep01 (clean digital) | 1.32 | thr 4.0, chain 2 - **142/219** (64.8%) | 6.90 s against 13.31 s (**1.93x**) | **0.096%** (peak 18) - 4% of margin |
| Cowboy Bebop ep01 (35 mm TV, deinterlaced) | 2.27 | **refused** - 0/240 | 14.79 s against 14.93 s, **byte-identical 240/240** | 0.000% now; **0.470%** at threshold 4 and **0.469%** at 2 when it was still allowed |
| clean anime clip + synthetic grain 4 (sweep below) | 1.84 | **refused** at the 1.5 limit - 0/498 | 31.61 s against 31.11 s, **byte-identical 498/498** | 0.000% now; **0.105%** when it was still allowed |

Every row above was re-derived from the renders on disk while auditing this section, which moved two
ratios by a few percent against their first run (Violet 1.60 -> 1.55, Chainsaw Man 1.38 -> 1.36):
the defect numbers reproduced to the last digit, the times did not, which is the rule at the top of
this file applied to itself.

What this settled:

- **The threshold is not the knob on grainy material.** Bebop measures the same defect at
  threshold 4 and at threshold 2 (0.470% against 0.469%) while reuse halves, i.e. a 32 px block
  mean cannot tell "the grain changed" from "something moved one pixel". Lowering the threshold
  buys nothing and costs the reuse.
- **The noise floor is the signal, and its limit sits where the bar is crossed.** Four points, floor
  against worst-frame defect: **0.95 -> 0.005%, 1.32 -> 0.097%, 1.84 -> 0.105%, 2.27 -> 0.470%**.
  So the 0.1% bar is crossed between 1.32 and 1.84, and the limit is **1.5**, not the 2.0 this row
  first shipped with - 2.0 let the 1.84 clip through at 0.105%, i.e. past the bar by 5%. On Bebop
  the refusal gives a render that is **byte-identical to the ungated one** (240/240 frames) at
  14.79 s against 14.93 s - an honest 1% instead of a visible 0.470% for a nominal 1.15x.
- **The chain is the adaptive knob, not the threshold.** The clamp at 4 was hit on all three
  titles while the chain was cut to 1 on all three; that is the opposite of what the two-clip
  derivation suggested.
- Violet at 0.097% sits just inside the 0.1% bar, so the margin on ordinary good material is
  thin - 3% of margin on one of three titles. The "0.18 floor" that made this look like a
  comfort margin was the **retired statistic's** scale: the same family of clean 1080p anime
  measures **0.77** under `box(|diff|)`, i.e. just below Chainsaw Man's 0.95, not an outlier.

## The sample bar cannot be calibrated yet (measured, and one earlier claim withdrawn)

The acceptance-certified tuner reads its calibration sample lower than the clip, so the obvious fix
was a longer window or a tighter sample bar. Both were tried, and neither is established:

| material | calibration window | tuner's worst share on the sample | decision | reuse | clip, worst frame |
|---|---|---|---|---|---|
| Chainsaw Man ep01 | 16 | 0.00 / 0.00 | chain 2 | 146/220 | **0.005%** |
| Violet Evergarden ep01 | 16 | 0.01 / 0.02 | chain 2 | 142/219 | **0.096%** |
| anime clip | 16 | 0.00 / 0.00 | chain 4 | 362/498 | **0.017%** |
| Cowboy Bebop ep01 | 16 | - (floor refuses) | refused | 0 | **0.000%** |
| Violet Evergarden ep01 | 48 | 1.95 / 99.69 | chain 1 | 117/219 | **0.097%** |

Three things this table says, and one it cannot:

- **The sample reads 5x low where it is non-zero at all.** Violet is the only material with a
  non-zero sample (0.02% against 0.096%), and the other three read 0.00 on 16 frames while their
  clips measure 0.005-0.017%. So the factor is ~5 on one title and undefined on the rest - **not a
  calibration, one point**.
- **The bar cannot be set from passing cases.** A 0.05% sample bar would leave Violet exactly where
  it is (0.02 < 0.05) and change nothing for any measured configuration. To calibrate it, a
  *failing* configuration measured on the fixed code is needed, with its sample number.
- **The 48-frame window is not a fix, and the claim that it made things worse is withdrawn.** The
  chain 3 / 0.121% result that suggested "longer window certifies deeper and fails the bar" was
  measured while the tuner's HR history spanned scene cuts (the bug below); with that fixed the same
  window reads chain 1 at 0.097%, i.e. *more* conservative, not less. Its sample numbers are still
  contaminated at depth >= 2 (99.69% of pixels differ, which is a cut, not a reuse), so the 48-frame
  row cannot calibrate anything either. (Its wall time is not quoted: that run collided with another
  GPU job and measured 16.1 s against 6.8 s for the same decision class.)
- **The 16-frame window reproduces** across runs (chainsaw 146/220 0.005% and violet 142/219 0.096%
  on every repetition), which is why it stays the default.

Two side results, both useful:

- chain 1 on Violet measures **0.097%** against chain 2's 0.096% (9.56 against 6.90 s): the cheap
  depth is not buying picture on this clip either, the same conclusion the anime clip reached with
  tiles off (0.016% against 0.017%).
- the depth-2 sample numbers of 99.69% are the honest limit of the acceptance statistic on a moving
  shot: per-pixel differences above 8 levels stay huge under a pan even when the block-mean
  statistic says the frame barely moved. That is the statistic doing its job (it is why the tuner
  refuses depth on this material), and it is also why the sample cannot be read as a smooth
  continuous quantity.

## E22: is the raw builder faster? Not established - the drift is bigger than the effect

The head padding only reached the product through a question nobody had asked: does the way this app
compiles an engine matter next to how the graph looks? One pair of single runs said yes, loudly -
app 11.92 s against raw `trt.Builder` 11.01 s on the same clip, with a **bit-identical picture**
(0.000% of pixels above 8 levels, peak 0) - and a second pair said the opposite (gate off: 33.12 s
app against 36.02 s raw). So it went to the interleaved protocol, three rounds each:

| round | app (`torch_tensorrt.dynamo.compile`) | raw (`trt.Builder`) |
|---|---|---|
| 1 | 13.00 s / 15.44 ms engine | 12.81 s / 16.52 ms |
| 2 | 14.65 s / 18.27 ms | 12.95 s / 17.33 ms |
| 3 | 14.84 s / 20.01 ms | 14.55 s / 19.97 ms |

Raw wins the wall clock in all three rounds and loses the engine number in one of them, while **both
series climb 13-30% across the experiment** - the machine heats up and there is no thermal control on
this host. An effect smaller than the drift inside its own measurement is not measured; the honest
status is **no established difference**, and the 7.6% this row used to claim is withdrawn.

What survives is the *structural* result, which is worth keeping: the two builders produce a
bit-identical picture on the shipping model, and the raw path covers **78 of 78** models in the
owner's collection at the load/forward/ONNX-export stages (SPAN's 11 files are the only padding
casualties, and they revert). So `RVE_RAW_TRT` stays as an independent build path with a fallback, not
as a lever.

Two method notes, because this row cost two wrong headlines:

- a single pair is not a measurement for anything under ~10% on this host, and I quoted one;
- cross-run drift over a session this long is now the dominant error term, not the configuration.

## E24: the tail convolution does not fuse, and padding it makes things worse (measured)

The x2 compact model's tail emits **12** channels (3 x scale^2) and 12 is not a multiple of 8 - the
same alignment rule that kept the head out of myelin - so the obvious next move was to pad it to 16,
trim the extra four before the `PixelShuffle`, and collect the same kind of win. Measured on the same
model at 256x256, two raw TRT builds:

| build | time | layers | fusion | correlation |
|---|---|---|---|---|
| head padding only (what ships) | **1.32 ms** | 25 | 16 | 2 |
| head **and** tail padding | 1.35 ms (+2.1%) | 25 | **still 16** | 2 |

So the tail never enters myelin even when it is aligned: the fusion count does not move, no new slice
or shuffle layer appears, and the time gets worse rather than better. The alignment reason that
explains the head does not transfer - the tail's consumers (`PixelShuffle` and the residual add) are
themselves not myelin-friendly, so there is nothing for the epilogue to attach to.

Recorded as a **negative result with the mechanism**: the hypothesis was "the same defect as the
head", and the graph says the tail's defect is different. A 1080p confirmation was not run, because the
indicator that refuted it (the fusion count) is a property of the graph, not of the pixel count.

The tail's 1.92 ms against a 1.09 ms traffic floor is therefore **not** a padding problem, and the
0.83 ms gap has no owner yet - which is why E20's "tail tactics" half stays open while its head half
closed with E19.

## E23: the 71% regression was a graph break, and the fix is one operation

`F.pad` in front of the head measured **26.76 ms** of engine time against 16.07 unpadded - a 67%
regression from adding five zero channels to one convolution, which no arithmetic explains. The
mechanism is the partitioner: `aten.constant_pad_nd` has no converter in torch_tensorrt's dynamo path,
so the node falls back to torch and the engine is **cut into two TRT sub-graphs with a torch op between
them**. The cost of that seam, not of the padding.

Replacing the operation with `torch.cat` (an `IConcatenationLayer`, which has a converter) keeps the
graph whole and gives the win that was predicted all along:

| head input, app build path | wall | engine |
|---|---|---|
| no padding | 11.79 s | 16.18 ms |
| `F.pad` (broken) | 16.87 s | 26.76 ms |
| **`torch.cat` + zeros** | **11.44 s** | **15.61 ms** |

Interleaved, three rounds, engines cached, clocks logged per run (no pinning on this host: `-lgc` needs
root): padding wins **3/3 on both numbers**, medians **-3.0% wall and -3.5% engine**, with a 0.3% spread
inside the padded runs. That is what Amdahl demands from a 5.8% engine win at a 67.6% engine share
(3.9% wall predicted, 3.0% measured), and it retires the "0.9% on top of the raw builder" figure that
was holding the flag off - a single pair, below this ledger's own noise threshold, and wrong.

Picture on the same clip: `cat` against unpadded **0.003% median, 0.012% worst, peak 15** - the fp16
rounding of summing eight terms instead of three, inside the 0.1% bar.

## E25: certifying the depth on the acceptance statistic fails its own verification

The review's argument was that the tuner certifies a block-mean proxy while the pictures are judged by
"share of pixels above 8 levels on the worst frame", and that the calibration sample should therefore
be adversarial - the 16 frames with the largest LR delta rather than the first 16 that happen to
arrive. Implemented, with a 64-frame observation window, and then measured. It is a **negative result,
and an instructive one**:

| what | tuner's worst acceptance share | actual render, frames 1-64 |
|---|---|---|
| Violet, certifying on acceptance | depth 2 reuse read **99.69%** of pixels | **0.011%** worst frame, none above 0.1% |
| Violet, chain 1 - the configuration it chose instead | - | **0.129%** worst frame, one above 0.1% |

So the acceptance path refused chain 2 because of a reuse that **never happens**, and what it picked
instead is worse than the bar. The proxy had chosen chain 2 all along, and that configuration measures
0.011%.

The mechanism is the depth model, not the statistic: `accept[k]` compares a frame against the k-th
previous *rendered* frame, while the gate serves the last *kept* frame no older than `max_chain`, and
whether a k-deep reference exists depends on the chain state and the pixel veto, not only on whether
the intermediate frames passed the LR test. The reachability filter added in E19 checks the LR test
only, so the simulation over-predicts reuses the gate never makes. On a pan, a frame two back can
differ by 99% of its pixels while its block mean barely moves - which is exactly the blindness this
statistic was introduced to catch, turned against the depth decision.

Three defects were fixed on the way, and they outlived the experiment:

- the **noise floor** was being computed over the adversarial subset, whose calmest third is not calm:
  Chainsaw Man read 0.95 -> 9.38 and refused material that reuses at 0.005%;
- the selection by LR delta picked **cut-adjacent frames first**, where the HR history had just been
  cleared, so every kept frame had an empty acceptance array and the tuner silently fell back to the
  proxy;
- `any(accept)` tests **truthiness, not presence**, and a measured 0.0 is the common answer on clean
  material - so every measured sample was discarded. This one had been live since the statistic was
  introduced.

What ships: `certify_on="proxy"` (the depth is certified on the block-mean p90 as before),
`calibration_window` back to the frame count it always used, and the acceptance statistic kept as a
**guard and a log line** with its per-depth values. The `acceptance` mode exists, is tested, and is the
thing to revisit once the depth simulation matches the gate - that is the open item, and it is a
modelling problem rather than a statistics one.

Both titles are back to their previous decisions with the same reuse counts and defects (Chainsaw
146/220 at 0.004%, Violet 142/219 at 0.097%).

## The head padding across the whole collection (78 models, 2026-11-15)

Load -> pad -> forward -> ONNX export, CPU, no engine builds (a build is minutes per model and tactic
choice is a property of the graph): **0 load failures, 0 forward failures, 0 ONNX export failures**.
The padding applies to 60 models (5 zero channels each), reverts on **11 SPAN files** because their
head is a `Conv3XC` that adds its 1x1 weight to its 3x3 one - widening a single branch makes that sum
8 against 3 - and is skipped by rule on the 12-channel RRDB and on the already-aligned heads.

That is the gate for any future "make it the default": the export prerequisite holds across six
architectures, and the structures that break are caught by a 32x32 probe forward rather than by a
render.

The raw path itself was then checked for **correctness** on the two architectures that build at
256x256 - a compact SRVGG and an RRDB - by comparing the plan's output against eager torch:
**0.000% of pixels above 8 levels, peak 2**, on both. So the path computes the same function, not just
the same shapes.

The third representative, DAT (`4xLSDIRDAT`), cannot even run in fp16 through this codebase -
`RuntimeError: expected scalar type Float but found Half` in its attention - which is why the app has a
precision fallback and is not a property of the raw path. The fourth, SPAN, was not reached because
the probe stopped at DAT. Both are stated rather than glossed: "two of the four representatives" is
what was checked.

A note on that probe's timing column, because it is easy to misread: it compared the plan against
**eager torch**, not against torch_tensorrt, so its ratios (2.92 against 4.41 ms for the compact,
14.51 against 71.90 for the RRDB) say the plan is much faster than eager inference and nothing about
the two builders. The builder comparison is the interleaved one above, and it established no
difference.

## Is `tiling=none` faster than what ships? No - and the probe that said so measured a different engine

A three-way engine-level probe of raw-TRT plans (round-robin, 7 rounds, 2250 MHz / 161 W) put `none`
5% ahead: **53.64** against 56.67 (`moderate`) and 56.57 (`full`). That looked like a free 5% behind a
flag, so it was taken to the product: five rounds per level through the app, every engine cached,
interleaved, reading the app's own reported per-frame engine time (medians of rounds 2-5; the first
round is cold):

| level | wall | app's `inference` | vs full |
|---|---|---|---|
| `full` (shipped) | **11.32 s** | **15.37 ms** | - |
| `none` | 11.35 s | 15.45 ms | +0.2% / +0.5% |
| `moderate` | 11.75 s | 16.29 ms | +3.7% / +6.0% |

So the default is the best of the three, `none` is indistinguishable from it, and the flag's real
content is "not moderate" - which reproduces the original 3.8% A/B as 3.7%. The isolated probe was
not wrong about its own plans: the app builds through torch-tensorrt, and its `none` engine measures
56.6 ms per rendered frame against `full`'s 56.3. **A hand-built plan is not the shipped engine**, and
engine-level numbers from one build path do not transfer to another.

Method note, because the first attempt at this was wasted: that A/B was run while the sample/clip
renders were still going (`ab_moderate_1` at 16:27:04 and `sp_vi48` at 16:27:06), which produced 18.6
and 23.9 s for `moderate` and a false "no difference" verdict. "One GPU job at a time" is a rule
about *this* measurement too.

## Provenance of the rows in this file

Five things changed after most of these numbers were taken, and it is worth knowing which rows
predate which change:

- **the `mean_abs` statistic** - every row below is post-change unless it says `|box - box|`;
- **the noise-floor limit (2.0 -> 1.5)** - rows measured at 2.0 say so (the grain-4 row above does);
- **the tuner's chain reachability** - subtle: the gate's own reuse decisions did not change with it,
  so counts and floors measured before stay valid; what changed is *which depth the tuner certifies*.
  The three-BD table and the anime clip were therefore **re-measured after** the fix (chain 2 and
  chain 4 respectively), while the grain sweep and the film rows predate it and are unaffected
  because those clips either certify the same depth or refuse;
- the dead scene-cut guard (it had not run since the statistic changed) did **not** alter any reuse
  decision either - a cut frame misses the threshold by 255 levels regardless - but it fed the tuner
  a history spanning the cut, which is part of what the reachability fix addresses;
- **the acceptance certification** (2026-11-15, last) changed which instrument decides. Rows measured
  before it were certified on the block-mean proxy; on the three Blu-rays both instruments agreed
  (same chain, same counts), so those rows are comparable, and the shipping numbers quoted above are
  from the acceptance-certified run.

## Grain sweep, re-measured on the statistic that ships (2026-11-15)

The first version of this section predates the change from `max |box - box|` to
`max box(|I - I_ref|)`, and its central sentence - "grain does not stop frame reuse because a
32 px block mean averages grain out" - is a statement about the **retired** statistic. `box(|diff|)`
does not average: `E|d| = 0.8 * sigma * sqrt(2)` at any block size, so grain has to show. The same
five variants, re-cut at 500 frames, `--quality_preset best` against an ungated render of the same
clip, one GPU job at a time:

| variant | noise floor | decision | reuse (integer) | gated | ungated | RGB > 8 levels, worst |
|---|---|---|---|---|---|---|
| clean | 0.77 | thr 4.0, chain 4 | 362/498 (72.7%) | 12.03 s | 30.93 s | 0.017% (peak 17) |
| grain 4 | 1.84 | thr 4.0, chain 4 | 359/498 (72.1%) | 12.18 s | 31.11 s | **0.105% (peak 21)** - past the bar |
| grain 8 | 3.49 | reuse OFF | 0 | 30.75 s | 30.97 s | 0.000% |
| grain 12 | 5.17 | reuse OFF | 0 | 30.94 s | 30.94 s | 0.000% |
| grain 16 | 6.89 | reuse OFF | 0 | 30.95 s | 30.71 s | 0.000% |

- **The share is not the indicator; the defect is.** Reuse barely moves (72.7% -> 72.1% at grain 4)
  while the worst frame goes from 0.017% to **0.105%**, i.e. past the 0.1% bar, for the same 2.55x.
  So "grain does not stop frame reuse" survives as a statement about the share and is false as a
  statement about safety, which is the only thing it was ever used to argue.
- **The floor tracks grain linearly** - 0.77, 1.84, 3.49, 5.17, 6.89 for grain 0/4/8/12/16, about
  0.3 per grain unit - and the crossing of the bar sits between grain 0 and grain 4, which is the
  four-point calibration recorded above (limit 1.5). Re-run with that limit, the grain-4 clip is
  refused and the render is **byte-identical to its ungated pair (498/498 frames)**, as are grain 8,
  12 and 16 - the rows above were produced under the old 2.0 limit, which is why grain 4 still
  shows a reuse count there.
- **"Grain stops tile reuse between 0.40 and 0.51" is on the retired scale** and is not comparable
  with anything in this table. Tile reuse is off by default for its own reason (0.287% against
  0.009%), so nothing here depends on that row.

`best` no longer refuses on the clean clip: it certifies chain 4 at threshold 4.0. An earlier
revision of this section recorded that `best` refused on *every* variant including the clean one
and called that "a bug in the preset wiring". That was true of that build - the two presets
disagreed at the same budget - and the disagreement is what the chain-policy split fixed; the
clean row above is the same clip certifying depth 4 today.

**Timing spread within one session, on identical work.** The clean row (auto, threshold 4.0,
chain 4, 362/498 reused) measured 12.03 s here and 9.97 s in the integer-count table below. Part of
that is the tuner's 16 forced calibration frames (~0.7 s, which only the auto run pays); the rest is
the machine's clock state. So an A/B below ~10% needs interleaved runs even inside one session -
"ratios only within a session" is necessary, not sufficient.

## Integer reuse counts at a fixed chain (why "70.2% at three thresholds" was unreadable)

Same clip, tiles off, chain fixed at 4, threshold forced with `--no_temporal_reuse_auto`:

| threshold | reuse | time | RGB > 8 levels, worst |
|---|---|---|---|
| 2 | 359/498 (72.1%) | 10.52 s | 0.016% |
| 4 | 362/498 (72.7%) | 9.97 s | 0.016% |
| 8 | 370/498 (74.3%) | 9.51 s | 0.015% |

The published table reported 70.2% at thresholds 2, 4 and 8 while the worst-frame defect changed
115x (0.004% -> 0.460%). If the same frames are reused the output is identical, so a share quoted to
one decimal was hiding the only thing that mattered: 0.1% of 502 frames is half a frame, and under a
worst-frame metric **one frame is the entire result**. Published as counts, the three settings differ
by 11 frames and the defect is flat at 0.015-0.016% - on `box(|diff|)` the threshold really is not
the lever, and that now shows in the counts as well as in the defect.

## E26: Winograd is not reachable through cuDNN on this card, and the margin is 20%

The question the literature raised was whether our body convolution leaves a Winograd win unclaimed.
Answered by measurement, and the answer is no - at least not through any algorithm cuDNN will select.

Shape is the model's own body layer: 3x3, 64 -> 64, 1920x1080, fp16 NHWC, 152.9 GFLOP. The engine
runs 16 of these in 48.72 ms, i.e. **3.05 ms and 50.2 TFLOP/s** each.

| run | ms | TFLOP/s |
|---|---|---|
| cuDNN 9.13, heuristic (`benchmark=False`) | 4.06-4.12 | 37.2-37.7 |
| cuDNN 9.13, full autotune (`benchmark=True`, every engine timed) | **3.67** | **41.7** |
| cuDNN 9.13, fp32 NCHW control | 10.4 | 14.6 |
| our engine (TRT, `implicit_gemm` myelin kernel) | **3.05** | **50.2** |

Three rounds, interleaved, one process per mode so no benchmark result survives between them; the
spread across rounds was under 1%. Autotune beats the heuristic by 10%, and still loses to our kernel
by 20%. The fp32 control shows no win at all (14.6 TFLOP/s, autotune does not improve it), which is
what a missing algorithmic shortcut looks like: Winograd's saving is largest exactly where it would
have shown up there.

Shape sweep, to rule out the possibility that Winograd exists but is only viable at small spatial
sizes (`tiling=moderate` is a shipped option): 1080x1920 -> 41.7, 540x960 -> 41.4, 270x480 -> 38.8,
135x240 -> 28.1 TFLOP/s. Efficiency is flat or worse, so there is no tile size at which a different
algorithm takes over.

**What this does not establish, stated plainly: neither the kernel NAME nor that Winograd was among
the candidates.** cuDNN's autotune picks the best of what it enumerates; a forced
`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD` would settle whether the algorithm exists for this shape and what
it costs, and PyTorch exposes no algorithm selector, so that test needs the cuDNN C API. "cuDNN's best
is slower" is a fact about selection, not about Winograd, and the two have been confused once already
in this ledger (E14 restricted sources, it never enumerated algorithms within a source).** Trying to name it failed twice -
`torch.profiler` returns zero CUDA events under this CUDA 13 / torch 2.9 pairing, and the nsys
archive (2025.1.3) refuses to run against CUDA 13 (`hasn't been installed with CUDA Toolkit
#CUDA_MAJOR#...`, its own template unexpanded). So we know that *whatever* cuDNN selects after timing
every engine it has for this shape is 20% slower than our kernel, and not which engine that is.
For the decision that is enough: either Winograd was among the timed engines and lost, or it was
never offered as an engine for this shape. Both readings close the "cheap lever" framing. What they
leave open is a hand-written Winograd F(2,3) with fused transforms, which is E10's cost class and
carries E10's counter-precedent (a hand-written conv kernel measured 39.8 TFLOP/s where this engine
holds 91 on a narrower precision).

So the literature's claim does not transfer, and the reason is visible in the numbers: Welder's
authors compared against TensorRT 8.x on an V100/3090 where an fp16 Winograd kernel existed and won.
On an Ada part running TRT with an sm89 tensor-core GEMM, the convolution is already at 68% of the
74 TFLOP/s peak the ledger borrows, and a 2.25x MAC reduction does not survive the transform traffic
at that efficiency.

## The distance to 2x, in measured numbers (2026-11-15)

The target for this round is 2x against the current engine without skipping frames, so 56.28 -> 28.1 ms
of engine time. This is where each measured lever sits against that, and it is worth writing down now
because two of the plan's items cannot reach it and one intermediate number is easy to misread.

| configuration | ms/frame | against 56.28 | what it is |
|---|---|---|---|
| fused fp16 (ships) | 56.28 | 1.00x | 16 myelin conv+PReLU blocks at 2.93 ms |
| int8 with a scalar activation fused (E9' ceiling test) | **40.24** | **1.40x** | the epilogue is free, the picture is meaningless (a scalar slope is not the model) |
| int8 two-pass, as built (E9') | 68.11 | 0.83x | conv writes half, PReLU is its own pass, 21 reformats |
| target | 28.1 | 2.00x | 16 blocks at ~1.25 ms plus the 8.1 ms of non-GEMM |

Three consequences, and one correction to an earlier reading of the same numbers.

**The int8 ceiling is 1.40x, not 2x, and that number is a measurement** - not the plug-in estimate.
The arithmetic in the review put the two-pass block at 2.14 ms and the fused one lower; the engine
actually builds at 1.95 ms per block with the epilogue fused, which leaves 40.24 ms for the frame. So
even with a plugin that fuses conv and PReLU perfectly, the remaining distance to 28.1 ms is another
1.4x that has to come from somewhere else.

**That somewhere is GEMM efficiency at int8, not another algorithm.** The 16 body convolutions are
2.45 TFLOP of work; at the 91 TOPS the int8 conv actually reports that is 26.9 ms, and the measured
block is 1.95 ms including its epilogue, i.e. **31.2 ms for the body alone**. The int8 tensor-core
peak on this part is around 148 TFLOP/s if the borrowed 74 TFLOP/s fp16 figure is right, so the body
runs at ~55-60% of it. Fusion does not raise that: fusing layers reduces DRAM round trips, and the
body at int8 is compute-bound (that is why int8 conv is 1.7x faster than the fp16 one while moving
the same tensors). So the honest ordering is: make int8 available at 1.4x, then chase GEMM efficiency
on the int8 build, where the headroom to 28 ms lives.

**The reformats are not the difference, and an earlier revision of this section said they were.**
E9' splits the two-pass engine's overhead precisely: **21 reformats worth 5.6 ms (8% of the engine)**
and **17 PReLU passes worth 32.7 ms (46%)**. So removing every reformat lands at 62.5 ms, still behind
the 56.28 ms that ships; the 42.7 ms figure was arithmetic that assumed the PReLU pass could run in
int8 at 0.92 ms, and the build that tried it measured **78.36 ms** because TensorRT charges the hop
per layer (36 reformats, 16 `DQout` passes at 1.64 ms each). The PReLU pass is already at its own
traffic floor: 530.8 MB at 288 GB/s is 1.84 ms against 1.71-2.37 measured. There is nothing to win
there - the pass has to disappear into the convolution, which is the plugin.

**Low-rank is the same mechanism as int8 and we have already measured its answer.** A 64 -> r -> 64
factorisation of one body layer at r=32 is 640r MAC/px against 36864, i.e. 1.8x fewer, and then the
intermediate r=32 tensor costs a write and a read of 133 MB each, 0.92 ms at 288 GB/s, against a layer
that costs 2.94 ms. So the layer goes 2.94 -> 1.63 + 0.92 = 2.55 ms, **1.15x, not 1.8x** - and that is
the int8 story retold, because int8's 1.74x conv gain was also eaten by a per-hop boundary. The
transferable lesson from the LLM literature is not the factorisation but the reason it works there:
decoding is weight-bandwidth-bound (weights 1.2 MB here, activations 265 MB, so the regime is the
opposite) and architectures such as MLA are learned, not applied afterwards. What survives is the
activation-aware reconstruction, which is E30's method, and it survives because it changes the
*model*, not because it removes a boundary.

What is genuinely blocked, and for how long: an in-kernel epilogue needs an int8 GEMM we control.
The TRT python wheel ships no `NvInfer.h`, so a C++ plugin cannot be compiled here without fetching
TensorRT's headers, and cuDNN cannot fuse a per-channel PReLU into its int8 convolution (its fused
activation takes a scalar slope, which the measured slopes -1.39..1.70 rule out). That leaves a
hand-written or CUTLASS-generated int8 conv with a per-channel epilogue: E10's cost class, weeks, and
the item that actually stands between 1.4x and 2x.

## What the card can actually do, measured (2026-11-15)

Three dense GEMMs, because the 2x question turned out to be a peak question and the peak in this
ledger has been a borrowed number until now: `torch._int_mm` and `torch.matmul` on 8192^3, warm, 20
iterations.

| operation | TFLOP/s or TOPS |
|---|---|
| fp16 GEMM, fp32 accumulate (cuBLAS) | 42.1 |
| fp16 GEMM, **fp16 accumulate** | **71.6** at 4096^3, 68.9 at 8192^3 |
| int8 GEMM, B in the K-major layout | **147.4** |
| int8 GEMM, B in the transposed layout | 39.3 |

Two things follow, and the second one is the useful one.

**The borrowed 74 TFLOP/s peak was right, but only for fp16 accumulate** - which is what our kernel
does, and nobody had measured it here. cuBLAS with `allow_fp16_accumulation` reaches **71.6 TFLOP/s**
against 42.1 with an fp32 accumulator, so the myelin kernel's 50.2 is **70% of its own class's peak**,
not 68% of a borrowed number and not 57% of a guess. That closes the "the ceiling is unknown" question
and it makes fp16 arithmetic final: the body is 2.446 TFLOP, so even at a perfect 71.6 the floor is
**34.2 ms**, i.e. 42 ms for the frame, i.e. **1.34x is the entire fp16 headroom**. 2x is unreachable
without leaving fp16.

**int8 has a real 147 TOPS, so 2x is arithmetically reachable - at a price.** The body is 2.446 TFLOP,
i.e. 16.6 ms at peak and 26.9 ms at the 91 TOPS our int8 convolution actually reports (62% of the K-major peak).
Adding the 8.1 ms of non-GEMM work:

| body efficiency | body | + non-GEMM | frame | against 56.28 |
|---|---|---|---|---|
| 147 TOPS (peak, unreachable in practice) | 16.6 | 8.1 | 24.7 | 2.28x |
| 120 TOPS | 20.4 | 8.1 | 28.5 | 1.97x |
| 109 TOPS | 22.4 | 8.1 | 30.5 | 1.84x |
| 91 TOPS (measured today) | 26.9 | 8.1 | 35.0 | 1.61x |

So 2x is not blocked by arithmetic, it is blocked by the epilogue that costs 32.7 ms, and behind that
by GEMM efficiency: no configuration is 2x below 82% of the int8 peak. An in-kernel epilogue gets us
to the 1.61x row (35 ms, and the measured ceiling test says 40.24 ms - the difference between 35 and 40
is the epilogue's own write traffic and the non-GEMM part being larger than 8.1 in that build). The
1.61x row is what a plugin buys; the rows above it need the GEMM itself to improve.

**Scope change: grain-heavy cel samples are out of the acceptance sample set** (owner, 2026-11-15).
Nobody upscales celluloid animation with its grain preserved, so the 35 mm TV clip is no longer a
quality target and the question "what should the bar mean on grain" is withdrawn as posed: at sigma=16
two independent grain fields differ by more than 8 levels on 72.4% of pixels, which made the bar
unreachable by construction rather than by a defect. The measured numbers on that clip stay in this
ledger as history (the gate's floor of 2.27, the refusal, the 1.85x), and the 2-4x headline for anime
material is unaffected.

## The correction is 0.9 levels: what that licenses (2026-11-15)

The model is residual by construction, and nobody in this ledger had measured the size of the residual.
Measured on a real frame (Chainsaw Man, 1920x1080, x2 model `futsuu`, fp16 engine weights in torch):
`correction = out - nearest_upsample(x)`, reduced with the acceptance rule's own "worst channel", in
0..255 levels.

| statistic | levels |
|---|---|
| median | **0.9** |
| p90 | 2.1 |
| p99 | 10.1 |
| p99.9 | 19.4 |
| max | 44.5 |
| share below 8 levels | **98.43%** |
| share below 2 levels | 88.69% |

So for 88.7% of pixels the network adds less than two levels on top of a pixel repeat, and for 98.4%
it adds less than the acceptance bar itself.

**The licence this looks like it grants, and why the first reading of it was wrong.** The obvious
model is `error = e * |correction|`: a pixel crosses the 8-level bar only where `|correction| > 8/e`,
0.1% of pixels sit above 19.4 levels, so `e ~ 0.4`. That model is **refuted by our own int8
measurement**, and the refutation is measurable in one minute:

- the int8 engine's per-channel weight quantisation perturbs the weights by **2.0-2.3% RMS** per body
  layer (measured on this checkpoint, 16 values between 0.014 and 0.023);
- the model then predicts a peak error of `0.021 * 44.5 = 0.93` levels and a mean of `0.021 * 1.4 =
  0.03`;
- the int8 picture actually measured **peak 4 levels, mean 0.66, PSNR 54.1 dB**. The mean is off by
  **20x**.

So the error has a component that is not proportional to the correction, and it dominates. The
mechanism is structural: on a flat input patch the network computes `g(c) ~ c`, i.e. the correction is
**zero by construction** - and that is 88.7% of the frame. Any change in the weight space moves `g` to
`g'` and leaves a DC shift there, exactly where the proportional model predicts no error at all. That
is what a tint or banding on a background is made of.

What survives is narrower and still useful: the residual measurement licenses approximations that are
**multiplicative in the output** - truncating the correction where it is small, dropping the tail's
precision, quantising the correction in flat regions. It does not license changing the function, and
therefore it does not license a structural cut of the body. Two further reasons the 0.4 was never a
working point even inside its own model: `8/e = 19.4` puts the decision exactly **on** the bar with no
margin, on a single frame, with a known ~5x spread between frames (f383 vs f360 on the reuse
artefact), and with channel-max against luma differing 8x (0.29% against 0.036%).

**What the int8 falsification does give: a calibration point.** A 2.1% perturbation of the weights
costs a 0.66-level mean and a 4-level peak, i.e. half the bar. Whatever the admissible perturbation
is, it is of that order, not forty times larger.

**And the MAC arithmetic for a channel cut was wrong twice over.** E19 already refuted FLOP
proportionality on this engine: widening the head from C=3 to C=8 raised FLOPs by 0.48% and moved
time by 3.36 ms, because the kernel class changed. For C=48 the arithmetic is worse than
proportionality suggests: the shipped tactic is `tilesize128x64x32`, so an N tile of 64 is 75% filled
by 48 channels and K = 432 is 14 steps of 32 against 18. The expectation is nearer **1.29x than
1.78x**, and C=32 is the width that fills both 64 and 32 exactly (MAC x0.25, K = 288 = 9 steps). All
of it is a time question, and it is measurable without asking anyone about the picture - which is what
`scripts/probe_channel_width.py` does.

## E30a: a channel cut is worth 1.23x, not 1.78x - measured

Same architecture, same builder settings as the app (fp16, `tiling=full`, optimization level 5),
synthesized weights (tactics depend on shapes, not values), 1920x1080, per-layer times from
`trt.IProfiler`.

| width | builds | fusion | MAC ratio | **time, same harness** | **ratio** |
|---|---|---|---|---|---|
| C=64 | 52.99, 51.38 | 16 fused, 0 separate PReLU (1.03x spread) | 1.00x | 51.4-53.0 ms | 1.00x |
| C=48 | 44.26, 41.84 | 16 fused, 0 (1.06x) | 1.77x | 41.8-44.3 ms | **1.16-1.27x** |
| C=40 | 48.54, 48.48 | 16 fused, 0 (1.00x) | 2.56x | 48.5 ms | **1.06-1.09x** |
| C=32 | 24.88, 24.06, 22.41 | 16 fused, 0 (1.11x) | 3.94x | 22.4-24.9 ms | **2.07-2.36x** |

**The curve is not monotonic, and that is the finding.** C=48 is faster than C=64 by 1.2x, C=40 is
*slower* than C=48 by 16% despite removing another 17% of the MACs, and C=32 is 1.9x faster than C=40.
Widths do not buy time proportionally; they buy a different kernel class, and only 32 lands on one that
fuses and also fits the tiling. The earlier readings for 48 and 32 in this table (46.02 and 46.10 ms
with the PReLU split out) were one build each of a choice that mostly does fuse - so the "fusion break"
row is retired, and the honest statement is that a build can land in either family, while the fused one
reproduces 3/3 for C=32 in this run.

Two results, and the second one is the interesting one.

**A 25% cut buys 1.23x, not 1.78x.** The body's per-block time goes 3.0-3.5 ms to 2.5-2.7 ms, i.e.
exactly the 1.25 the N-tile arithmetic predicted: `tilesize128x64x32` means 48 channels fill 75% of a
64-wide tile, so a quarter of the tensor-core work is spent on nothing. This is the second time on this
engine that a change in FLOPs did not survive contact with the kernel class - E19 was the first (C=3 to
8: +0.48% FLOPs, -3.36 ms).

**C=32 removes 4x the MACs and buys nothing at all.** The 16 convolutions do become 2.78x cheaper -
20.09 against 55.83 ms - and then myelin stops fusing the PReLU at that width: sixteen separate
`PWN(/body.N/PRelu)` passes at 1.4-1.5 ms each put the 22 ms straight back, and the frame lands at
46.10 ms, i.e. the same 1.23x as the 48-channel build. This is the **third** appearance of the same
mechanism in one investigation: the fp16 build pays it unless myelin fuses, int8 pays it per hop
(measured, E9'), and a narrower channel count pays it too. The binding constraint on this network is
not the arithmetic and not the peak - it is the boundary between operators, and it is charged whether
the saving comes from precision, from width, or from a different algorithm.

**C=32 is not a ceiling of the direction, it is a broken build, and the engine inspector says why.**
Its 16 convolutions run on `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_...` - the same family the
head and tail convolutions use, and the one that does not take an epilogue. That is why myelin stops
fusing: not the width, the tactic. E19 is the precedent: the head at C=3 refused myelin, the same head
at C=8 accepted it, and the fix was to pad the head's channels rather than to abandon the idea. The
same move is untested here.

The arithmetic if the epilogue comes back at C=32: 16 convolutions at 20.09 ms, head and tail scaled
to ~3.6, layout 0.85 - a frame near **24.5 ms, i.e. 2.31x**, and even with half of a 64-wide N tile
empty, ~30.8 ms, i.e. 1.84x. Both ends are above every other row in the table above, which is why the
direction is open rather than closed.

The accounting, exactly (engine inspector, sum of layer times against the measured total): C=64 ->
53.14 ms total, 52.82 attributed, 22 fused myelin layers, **zero** separate PReLU passes. C=32 ->
45.69 total, 45.09 attributed, 16 fused convs, and 52 pointwise layers (each PReLU is split into
several segments). The earlier "body" column mixed 22 fused layers with 16 convs and was not
comparable between rows; the 2 ms that did not add up was the head and tail, not a mystery.

Every lever measured so far therefore tops out between 1.23x and 1.40x:

| lever | number | what that number actually is |
|---|---|---|
| fp16, any configuration | 1.34x | a **headroom to a peak**, with no mechanism attached - and the peak is a square GEMM's, while the one compute-dominated conv measured here ran at 49 TFLOP/s |
| int8 with a fused epilogue | 1.40x | a **measured ceiling of the wrong function**: the epilogue that made it cheap was a scalar slope, -2.02 dB |
| channel cut C=48 | 1.23x | **measured real time of a working build** |
| channel cut C=32 | 1.23x | the same, but **of a build whose fusion broke** - not a ceiling of the direction (see below) |
| Winograd | not available | **not established**: cuDNN's best configuration is 20% slower than our kernel, which is a statement about selection, not about Winograd |
| low-rank | 1.15x | arithmetic on the boundary mechanism, never built |

2x is not reachable through any of these, and the reason is now consistent across four independent
measurements: the boundary is charged per hop and the arithmetic saving does not survive it. What
remains is a kernel we control, where the epilogue lives in registers and the tiling is ours - and its
ceiling is the int8 rate, 147 TOPS, i.e. 16.6 ms of body plus non-GEMM = 24.7 ms = 2.28x at peak,
1.97x at 120 TOPS, 1.84x at 109. That is E27 extended into a real kernel project (CUTLASS or
hand-written), weeks of work, with 2x at the very top of its range.

## The int8 block with a per-channel epilogue: 1.91 ms and 80 TOPS, measured, not projected

The whole int8 case rested on a number measured with a SCALAR activation fused into the epilogue (40.24
ms) and on an accuracy figure from a build whose PReLU ran in fp16. Both were borrowed. This is the
configuration the plugin would actually be, written as a kernel and measured:

one 3x3 conv 64->64, int8 in, int8 out, in a single Triton kernel - nine taps accumulating into int32,
then dequantise, per-channel PReLU, per-channel requant, clamp, store. NHWC activations, K-major weights.

| block | warps | time | TOPS |
|---|---|---|---|
| 64 | 2 | 2.450 ms | 62.4 |
| **128** | **2** | **1.925 ms** | **79.4** |
| 256 | 4 | 1.910 ms | 80.0 |
| 512 | 8 | 2.264 ms | 67.5 |
| 256 | 2 | 80.1 ms | 1.9 (a cliff, not a trend) |

Correctness first: the kernel output matches a shift-plus-`_int_mm` reference **exactly** (max |diff| 0
on 96x96 and 32x32, all taps and center-only). That check also caught a bad reference - the first version
mapped the weights with `permute(1,0,2)` and made a correct kernel look 40% wrong.

**So the budget is met**: <=1.9-2.0 ms and >=76-80 TOPS, achieved at 1.91-1.93 ms and 79-80 TOPS. Against
the fp16 fused block's 2.93 ms, a block is 1.53x cheaper while carrying the per-channel slope and a
per-channel output scale - the two things TensorRT's epilogue could not take.

Projection through the frame: 16 blocks at 1.91 ms = 30.6 ms, plus the measured 8.1 ms of non-GEMM work
= **38.7 ms against 56.28, i.e. 1.45x of engine time and 1.25x of wall**. That is the best
content-independent lever on the board, and it is now a kernel rather than an arithmetic hope.

**Accuracy, simulated with real scales: it holds the bar.** Weights per-output-channel symmetric int8,
each conv input per-tensor int8, epilogue with per-channel PReLU and a **per-channel** output scale -
the configuration E9' could not express. Measured on a real frame against the fp16 model:

| statistic | value |
|---|---|
| share of pixels above 8 levels | **0.000%** (bar 0.1) |
| peak | 7 levels |
| median | 1.15 levels |

**But it does not survive a held-out title, and the leak was mine.** Calibrated on one frame of one
title and evaluated on a frame of another (Violet Evergarden):

| held-out | share above 8 levels | peak | median |
|---|---|---|---|
| same shot, 1/24 s apart | 0.000% | 7 | 1.15 |
| **different title** | **3.573%** | **120** | 1.34 |

Clip rate on the held-out title, by layer: **body.0 23.8%**, body.1 2.9%, body.3 0.46%. So the failure is
not diffuse - it is the head conv's input, which is the image itself and whose range a single frame
under-samples; the per-tensor scale from one title's maximum then clips a fifth of another title's
pixels, and clipping is what produces a 120-level peak.

Two corrections follow, and one of them also corrects an earlier reading:

- **The peak bar is 20 levels, not 8.** The acceptance rule has two numbers - share above 8 levels under
  0.1%, and peak under ~20 - so "peak 7 against 8" was comparing the peak against the share's threshold.
  Against its own bar the same-frame result has 13 levels of margin, and the real defect was the sample,
  not the margin.
- **The head's input range is known** (0..1 after the clamp), so its scale is a domain constant and not a
  calibration result; the layers after it need calibration over several titles, or a percentile, or the
  asymmetric form below.

**And the mechanism behind "per-channel was worse" is the format, not the granularity.** E9's boundary was
`int8 -> half -> PReLU in fp16 -> requant`, i.e. 11 bits at the hop; the kernel's is `int32 -> PReLU ->
int8`, i.e. 8. At an activation of 55 the fp16 step is ~0.031 and the per-channel int8 step is ~0.435, so
fp16 is ~14x finer there and per-channel scales buy back about a bit of the three the format change
spends. That is why a design that should have been more accurate measured worse - and it is the reason
the next lever is the **asymmetric** boundary (scale + zero-point), where a one-sided channel such as a
negative PReLU slope spends half its codes on values it never produces: 60/256 = 0.234 against
55/127 = 0.433 on a [-55, +5] channel, i.e. ~0.9 of a bit, folded into the bias as `z_a * sum(w)`.

**The head's domain bound is the one-line fix, and it removes 87% of the failure.**

| held-out | share above 8 levels | peak | worst clip rate |
|---|---|---|---|
| cross-title, head scale calibrated on one frame | 3.573% | 120 | body.0 23.8% |
| cross-title, head scale = 1.0 (the clamp's own range) | **0.471%** | **72** | body.1 2.87% |
| same title, disjoint frames (3 s -> 7 s), head pinned | **0.066%** | **40** | body.1 0.25% |

So the third cell - which separates the format's price from the distribution shift - reads 0.066%, inside
the share bar, with the peak at 40 against a bar of 20. The remaining defect is not the format either: it
is `body.1`'s output clipping, the layer right after the head.

**The analytic interval bound for that layer is rejected on measurement.** For a conv whose input is in
[0,1] the output interval is exact - `[sum(min(W,0))+b, sum(max(W,0))+b]`, mapped through the PReLU - but
it assumes all 27 taps are simultaneously extreme. Against the empirical per-channel maxima it is
**10.3x wider at the median channel and up to 17000x on individual channels**, i.e. 3.4 bits spent to
remove 2.87% of clipping. It is fine for the head's INPUT (that bound is the data's own [0,1]) and
useless one layer later, which is exactly the distinction to carry forward.

What is left is calibration coverage, and the app already renders the material for it: the gate observes
the first frames of every video unconditionally, so the int8 scales can be calibrated per video from
frames the pipeline is already producing. That is the honest form of "the scales must cover the material"
- per collection is not enough, per video is what the data supports.

Also checked before any of this could mislead a later step: the requant rounds to nearest
(`floor(v/s + 0.5)` in both kernel and reference), so there is no coherent -0.5 LSB bias across the 16
layers; and the model clips to [0,1] *before* the residual add, so the tail kernel reads one buffer, not
two.

What it does not settle: the seam structure when the block is inserted into the graph (target: the
40.24 ms build's four reformats and one pointwise), and whether the 8.1 ms of non-GEMM survives the
insertion - head-to-int8 and int8-to-tail conversions are new passes worth roughly 0.8 ms, which turns
1.45x into 1.42x before anything else is won.

The Triton-to-TensorRT path is not a risk worth carrying: sixteen Triton kernels between two small
TensorRT engines on one stream under a CUDA graph need no plugin, no QDP and no headers. And once the
body is ours, the head and tail stop being TensorRT's problem and become small memory-bound kernels of
our own - the shapes that cost 3.90 and 3.35 ms in the fp16 engine are 145 MB and 195 MB of traffic,
i.e. 0.50 and 0.68 ms at the floor, and the 0.81 ms of layout work disappears if everything stays NHWC.
That is 8.11 ms of non-GEMM down to roughly 1.7, and a frame at 30.6 + 1.7 = 32.3 ms = 1.74x of engine
time, 1.38x of wall - better than the plugin route and with less risk, at the price of owning the
execution of the whole model instead of sixteen blocks.

## What bounds the routing lane, before it is measured further

Four constraints, each of which decides whether a number is worth chasing.

**The lane's own ceiling is the thin model, and it is already at 72% of the cost.** In the 1.59x row,
`c = 0.4545` of `0.6295` is the thin pass itself; as the change rate goes to zero the cost goes to `c`,
i.e. **2.2x of engine time is the absolute limit** of this scheme. Another 1.38x at most, and only at
zero change.

**The wall is much lower than the engine number.** The engine is 64.6% of the render cycle
(14.67/22.72), so 1.40x of engine time is **1.23x** of wall clock and 1.59x is **1.32x**. For scale: the
existing tile path already measures 1.17x of wall at 0.071%.

**The change rate has to be conditional on the gate rendering the frame.** The measured 1.2-2.3% is an
average over all pairs, and ~73% of pairs are reuse. If those are near-zero, the rate on the frames that
actually reach this mechanism is ~4-5%, and with the measured 17x dilation factor (1.5% -> 26.1%) the
recomputed region approaches the whole frame. Break-even is at `0.4545 + k*delta = 1` with `k ~ 12-17`,
i.e. **delta ~ 3-4%**. This is the third time a "do less work" lane is decided by a selection effect
(tile reuse fell from 65.6% to 57.4% inside the gate for the same reason), so it belongs in the ledger
as a rule rather than as a coincidence: *a saving measured on all frames is not a saving on the frames
the gate leaves*.

**`delta` must be a content difference, not a mask difference.** The mask is a gradient threshold; the
pixels under it can drift without moving it (slow brightness, grain, dithering), and a cached full answer
then ages silently. It is the same defect that moved the gate from `|box-box|` to `box(|diff|)`, and it
returns here in regional form. On grain the content changes everywhere while the mask stands still, so
the grain sweep is a required part of the measurement, not an optional one.

Two consequences that follow from the geometry rather than from tuning:

- **The seam creeps.** The boundary of F moves with the change mask, so the composite's step (thin's own
  error, up to 8 levels) travels across the frame - and a moving contour on a flat gradient is far more
  visible than a static one. The safe fix is free: feather the composite across the halo, where the error
  lies between 0 and thin's error and thin reads 0.060% above the bar on flat material. One blend, no
  extra pass, and the step stops existing.
- **This halo is not the int8 boundary tax.** The int8 boundary is an implementation artefact and fusion
  removes it (E19, -5.8%). The 18-pixel halo is the network's receptive field: it cannot be removed
  without fewer layers or a **cone instead of a cylinder** (E36). For a linear change of width 2 px and
  length L, the cylinder costs 16*38*L and the cone L*sum(2+2k) = 304L - exactly 2x less work, i.e.
  26.1% -> ~13% and 1.40x -> 1.71x. That is the only lever on this lane's numerator.

## Spatial routing survives the oracle: 1.6-1.95x on a mask a router can actually compute

The pruning post-mortem ended on a split worth more than the pruning attempt: at C=32 the error sits on
edges (26.1% of edge pixels above 8 levels) and flat regions are at the bar (0.060%). That is a licence
for conditional compute rather than for a global cut, and it is testable on outputs that already exist:

    out = where(mask, full, thin)

Inside the mask the composite *is* the full model, so the only error it can show is the thin model's
error outside the mask. Measured against `full` with the acceptance rule, on a real frame:

| mask | area | covers error pixels | >8 levels | time |
|---|---|---|---|---|
| oracle (where thin is actually wrong) | 1.6% | 100% | 0.000% | 2.16x |
| grad>4, dilate 2 LR px | 14.9% | 100% | **0.000%** | **1.79x** |
| grad>4, dilate 4 | 21.4% | 100% | 0.000% | 1.66x |
| grad>8, dilate 2 | 8.3% | 94.1% | 0.094% | 1.95x |
| grad>8, dilate 4 | 12.6% | 95.2% | 0.077% | 1.84x |
| grad>2, dilate 18 (the first guess) | 75.9% | 100% | 0.000% | 1.02-1.15x |
| grad>16, dilate 18 | 10.0% | 55.6% | 0.703% | 1.91x |

Two things this settles. **The pixel bound is real**: a perfect router would give 2.16x, because the
thin model fails on 1.6% of pixels. **And the mask does not need to be smart**: a 2-pixel dilation
around pixels whose LR gradient exceeds 4 levels covers every error pixel on this frame.

**The cost column above is wrong, and correcting it kills the per-frame version.** The full engine has
to compute on the *dilated* mask, not the mask: to produce a correct answer at a pixel, its input must
cover `mask + 18`. Measured areas, with `c = 1/2.2`:

| mask | mask | F = mask+18 | cost `c + |F|` | cost `|F| + c*|T+18|` |
|---|---|---|---|---|
| thr 2, r2 | 22.5% | 78.8% | 0.80x | 0.98x |
| thr 4, r2 | 14.9% | **52.1%** | 1.02x | 1.18x |
| thr 8, r2 | 8.3% | 33.8% | **1.26x** | 1.40x |
| thr 8, r18 | 32.0% | 45.6% | 1.10x | 1.33x |

So per-frame spatial routing is worth 1.0-1.3x, not 1.8x, and the 18-pixel receptive radius is the
reason - the same tax that ate every other algorithmic saving in this ledger.

**And the gated statistic, measured properly, is 1.31x of wall clock at its best.** The first attempt at
it used `max|pixel difference|` as a stand-in for the gate and declared 97.7% of the clip rendered where
the real gate renders 33.6%; the gate pools luma into 32x32 blocks at 1080p and thresholds the max over
blocks. Reimplemented that way it renders 73 of 219 frames (33.3%, against the app's 146/220 reused),
which is the check that matters, and then the 2x2 grid separates the two mask parameters:

| mask | F | change on rendered frames | cost (engine) | wall |
|---|---|---|---|---|
| thr 2, r2 | 78.8% | 46.7% | 1.09x | 1.05x |
| thr 2, r18 | 93.4% | 18.5% | 1.56x | 1.30x |
| thr 8, r2 | 33.8% | 25.4% | 1.41x | 1.23x |
| **thr 8, r18** | 45.6% | **18.0%** | **1.58x** | **1.31x** |

The surprise is that the *larger* mask is the stabler one: a thin mask thresholded on gradient magnitude
flickers pixel by pixel at the threshold, so its frames differ by 25%; a mask dilated to 18 becomes a
smooth blob that moves as a whole, and the recomputed area falls to 18%. The composite quality is
unchanged by that choice - thr 8, r18 covers 97.1% of the thin model's error pixels and measures 0.046%
against the bar.

**And that 1.58x describes a cached region the composite cannot afford.** The 0.046% was measured on a
*fresh* composite - the full answer recomputed for that frame - while `c + delta` assumes 82% of F comes
from a cache. Those are different configurations, and the ranking flips between them:

| configuration | quality (worst frame) | engine | wall |
|---|---|---|---|
| fresh, thr 8 r18 | 0.026-0.034% | 1.10x | 1.06x |
| fresh, thr 8 r2 | 0.026-0.034% | 1.26x | 1.15x |
| **cached, thr 8 r18** | **0.423-0.648%** | *1.58x* | *1.31x* |
| **cached, thr 8 r18, stride 3** (gate spacing) | **0.485-0.648%** | *1.58x* | *1.31x* |

Simulated offline over 20 frames, cache updated only where `delta` covers, truth is the full render of
the same frame. The cached composite is **4-6x over the bar**, and it is over by the same class of
mechanism that made the tile path measure 0.287% before dilation: parts of the reused region have
different ages. Dilation fixes the mask's spatial coherence and says nothing about the cache's age.

So the lane closes numerically, not by prior: **the only passing configuration is the fresh one, and at
1.15x of wall clock it is slightly slower than the tile path at 1.17x** while measuring 0.030% of pixels
against 0.071% - 2.4x better defect for 1.7% less speed, against 0.009% for the shipping default. That
trade is real, and the reason to refuse it is the cost of the mechanism (two engines, a mask, a new code
path) for an option that is only a little better than one that already exists; it is not that nothing
new was found.

Three things from the closure worth keeping, because they will be proposed again:

- **Fragmentation is why the cache failed, and it is the general case.** The 0.423-0.648% came from a
  reused area fragmented across thousands of independent cell ages; the tile path's 0.287% came from
  eight coarse cells. The number of age boundaries grows with the number of units, so the tile result
  was the *coarse* case and not the typical one. Any future "cache the expensive part" idea inherits
  this: count the boundaries, not the area.
- **Caching only on the frames the gate renders makes it worse, not better** (0.485 against 0.423%).
  That closes the loophole "update the cache only where the gate is drawing anyway" - the frames the
  gate leaves are exactly the changed ones, the same selection effect that dropped tile reuse from
  65.6% to 57.4% inside the gate.
- **F in the tables above is a compute area, not a mask area**: the full region is computed with the
  18-pixel halo on its input and cropped on output, which is why thr 8 r2 and r18 reach the same
  quality at 33.8% and 45.6%. Reading F as the mask would make the fresh variant slower, not faster,
  so the closure holds either way - but the surviving number has to carry its label.

For reference, int8's measured ceiling is 1.40x engine / 1.23x wall, content-independent, 0.000% of
pixels, no masks, no cache, no seam.

**What survives is the temporal version.** The LR gradient mask is mostly scene contours, and contours
are nearly static between frames: over a 1/24 s pair the mask+18 region changes by 1.2-2.3%. With its
own halo on the changed part, the cost is `c + |delta + 18|`:

| mask | F | F changes per frame | with halo | cost |
|---|---|---|---|---|
| thr 8, r2 | 33.8% | 1.5% | 26.1% | **1.40x** |
| thr 2, r18 | 93.4% | 2.3% | 17.5% | **1.59x** |

That is a per-frame path that keeps the edges on the full engine and pays the thin engine for
everything else, cashing the full answer on the static region once. It is also the same class of bet as
the existing gate - content-dependent, and a pan destroys it - so it belongs next to the gate, not
instead of it, and it needs the gate's certification rather than its own.

What is still unmeasured: the step at the mask boundary. The composite is the full answer inside the
mask and the thin one outside, so along the contour it steps by exactly the thin model's error, up to 8
levels - inside the bar's per-pixel rule and visible as an outline on a flat gradient. That is the kind
of defect the bar does not see, and it has to be looked at before any of this ships.


## E30b closed: 32 channels cannot hold the stack's features, and that is a bound, not a method failure

Three diagnostics and then the refit that settles it.

**The error is on edges, not in flat regions, and it is not a DC shift.** Measured with the acceptance
reduction on a real frame at C=32: flat pixels read **0.060%** above 8 levels - i.e. at the bar - while
edge pixels read **26.1%**; the signed mean is **0.01 levels** and a per-channel affine refit moves the
total from 1.573% to 1.553%. So the plateau is not the stack losing its balance; the cut loses
structure exactly where structure lives.

**The control rules out wiring.** Zeroing K channels in place (width unchanged) costs about what
removing them costs - 0.87% against 1.57% at eight, 1.64% against 1.59% at thirty-two - so the slicing,
the channel propagation and the PReLU indexing are doing what they claim to.

**The refit answers the only question left: can this width represent the teacher at all?** Sequential
layer-wise least squares (each layer fitted to the teacher's pre-activation, restricted to the kept
channels, on the already-refitted student's input, ridge 0.05) leaves a per-layer fit residual of
**14-37% of the target**, worst in the middle of the stack:

| layer | fit residual |
|---|---|
| body.2 | 14.3% |
| body.10 | 21.8% |
| body.14 | 37.0% |
| body.22 | 19.8% |
| body.32 | 32.6% |

A perfect fit is available in principle: the layer has 289 free parameters and 260k samples, so 14% is
not underdetermination - it is the part of the teacher's activation this width cannot express. And it
composes as it must: the output statistic stays at 1.58% no matter how many layers are refitted
(1.584% after body.2, 1.585% after body.16, 1.580% after body.32).

That bound applies to ASVD, SNOWS and K-step reconstruction alike, because they all end at the same
width. Widening the method does not widen the layer.

So the pruning lane closes on a measurement rather than on a disappointment: **C=32 buys 2.07-2.36x of
time and cannot pay for it in features; C=48 buys 1.2x and has the same picture as C=32** (the error
does not scale with the cut, and 48 is the same cut on this metric). Nothing between them is worth the
attempt.

What is left for 2x is the int8 kernel: **1.40x measured ceiling for a SCALAR fused epilogue** - a
per-channel epilogue that loads a slope vector and an output-scale vector per tile, and requantises,
has never been measured here, so 1.40x is an upper bound and not an expectation. Judge the probe by
block time (<=1.9-2.0 ms for 64->64 at 2.07 Mpx including the epilogue), not by TOPS.

**And the int8 accuracy figure does not transfer to the plugin.** The 0.000% / peak 4 / 54.1 dB was
measured on a build where the PReLU ran in fp16 with a dequantise/quantise hop either side, and with
**no requant after the PReLU**. The plugin does something that build never did: int32 -> per-channel
PReLU -> requant to int8. The one build that attempted it is invalid as an accuracy reference (its
output contained NaN from mis-calibrated output scales). So the accuracy of the design is **unmeasured,
not good**. Two quantitative reasons to expect pressure: slopes span -1.39..1.70, so the per-channel
range after PReLU varies by factors and flips sign; a per-tensor output scale spanning that union costs
the narrow channels on the order of a bit, and the measured headroom was peak 4 against a bar of 8.
Per-channel output scales are therefore probably mandatory rather than a checkbox.

Fourth criterion for the same probe: **do not break the fusion at the seams.** E19's head padding works
inside an fp16 graph; a plugin changes partitioning. The target structure is the one the 40.24 ms build
had - four reformat layers and one pointwise - and any more means the boundaries are eating the win even
if the block itself is fast. Read layer names and reformat counts, not just TFLOP/s.


## E30b first look: a naive channel cut is 16x over the bar, and the error does not shrink with width

Real frame, real x2 model, worst-channel reduction (the acceptance rule), reference is the uncut model:

| keep | share above 8 levels (bar 0.1) | peak | median |
|---|---|---|---|
| 64 (control) | **0.000%** | 0 | 0.00 |
| 56 | 1.573% | 59 | 0.90 |
| 48 | 1.607% | 59 | 0.98 |
| 40 | 1.580% | 59 | 0.93 |
| 32 | 1.585% | 59 | 0.93 |

The identity control at 64 reading exactly zero is what makes the rest readable: the slicing, the
selection propagation through the stack and the statistic are correct, so this is the model's answer,
not the instrument's.

**The error does not scale with the cut.** Removing 8 of 64 channels costs the same as removing 32, and
the peak is 59 levels in all four cases. That is the flat-region mechanism again, and it means the
naive cut destroys a *different* thing than it appears to: not gradually more detail, but a DC-type
shift that saturates as soon as the stack's balance is broken. It also explains why 48 - the cheap width
that costs only 1.2x of time - is no better than 32 on the picture: on this metric they are the same
cut.

So the picture half stands at **16x over the bar before any reconstruction**, at every width worth
considering. ASVD/SNOWS reconstruction is designed to minimise the output error rather than the weight
error, and the next rung is the cheap version of exactly that: a per-layer least-squares refit on
calibration activations, which is half an hour and tells whether the gap is closable before a day is
spent on the full method.

## C=32 fuses and runs at 22.4-24.9 ms: the "fusion break" was the builder (2026-11-15)

The same synthesized C=32 model that measured 45.69 ms with the PReLU split out rebuilt three times
with an explicit 8 GB workspace pool: **24.88, 24.06, 22.41 ms, sixteen fused myelin layers and zero
separate PReLU passes in all three.** The body convolutions come in at 1.17-1.42 ms each, i.e. at the
1.3 ms bar that makes the frame 2x.

Against 56.68 ms at C=64 that is **2.28-2.53x**, and the spread across three identical builds (1.11x)
is smaller than the effect.

Two candidate causes and they are not yet separated: the **workspace pool** (myelin needs scratch for
its fused kernels; without it, the body falls back to `indexed_wo_smem` plus separate pointwise passes,
which is exactly the 45.69 ms configuration) and the **tactic-selection instability** measured earlier
on a single block, where two builds of one file chose `scudnn_winograd` and `sm80_xmma` in turn. Both
produce the same symptom, and the separation matters beyond this row: `build_plan` in
`RawTRTEngine.py` never sets a workspace limit, so if the pool is the cause, the **shipped** engine is
also leaving tactics on the table, and that would be a speedup with no width cut and no picture risk.

Honest status of the number: it is a synthesized model at C=32 with random weights, so it measures time
and nothing else. What it establishes is that the fused build exists, reproduces, and lands where the
plan needs it. What it does not establish is the picture at 32 channels - a half-width body is a far
larger perturbation than the 2.1% weight noise that int8 showed to cost a 4-level peak against an
8-level bar.

## Winograd exists on this card and TRT selects it - at C=32, not at C=64 (2026-11-15)

A single Conv(3x3)+PReLU block exported at 1920x1080 and built with the app's settings is enough to see
which family the builder reaches for, because the engine names the tactic it chose:

| width | default build | fused? | time |
|---|---|---|---|
| C=64 | `__myl_CorrMulLtSele_myl1_0` -> `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_...` | **yes** | 2.67 ms |
| C=32 | `/conv/Conv` -> `ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1` | **no** | 3.12 ms + 2.46 ms PReLU pass |

So E16's stronger claim ("Winograd is not available here") is **wrong and now retracted**: a Winograd
kernel exists for sm89 fp16 in this TensorRT build and is chosen by the builder as soon as it wins on a
shape. What the earlier measurement established stands and is narrower - for the 64-channel body shape,
cuDNN's best is 3.67 ms against our kernel's 3.05, so no cuDNN-selected algorithm beats what ships
there. Two different statements, and the ledger had them merged.

The second result explains the fusion break at C=32 better than "the width is too narrow": the tactic
family changes, and both families that win at narrow widths - `scudnn_winograd` here,
`sm80_xmma_..._indexed_wo_smem_...` in the full model - take no epilogue, so myelin leaves the PReLU
outside. Whatever restores the epilogue at C=32 has to beat those kernels while staying in a family
that fuses, which is exactly what `scripts/force_tactic.py` scans by forcing one tactic index at a time
and reading the resulting layer names.

## What the literature closes, and the one item it reopens (2026-11-15)

Written down because these are structural closures - true by the shape of our network, not measured
by us - and because the reopened item was previously dismissed on a measurement that does not answer
the question it was cited for. Nothing in this section is our measurement; every number attributed
to a paper is the paper's, about its own hardware. The queue that turns the open ones into
experiments is `docs/backlog/queue.yaml`; who runs what is `docs/backlog/DELEGATION.md`.

**Closed by construction, so nobody re-enters them:**

- **Activation sparsity / dynamic skipping** (Deja Vu, PowerInfer and the rest of that family) assumes
  roughly half the post-activation values are zero. Our activation is PReLU, whose output is
  non-zero almost everywhere: there is nothing to skip. This is inapplicable rather than untested.
- **Collapsing linear parts** (folding two adjacent 3x3 layers into one 5x5) costs 25 taps against 18,
  i.e. **+39% MAC**, even with a fixed PReLU sign mask. Pre-composition is counterproductive at any
  precision on a stack of 3x3s.
- **Strassen and fast matrix multiplication**: N=64 leaves no recursion depth, the intermediate sums
  add exactly the traffic we are already short of, and fp16 stability is unproven. Theoretically
  attractive on large square GEMMs, dead on this shape.

**The fp16-range objection to Winograd does not hold at our activations.** We measured the activation
range at 1.0..55.2. For F(2x2,3x3) the transform matrix B^T has +-1 entries with row L1-norm 2, so the
2-D gain is at most 4 and the worst intermediate is about **221** against an fp16 maximum of 65504.
Even F(4x4,3x3), whose gain is roughly two orders larger, lands near 5e3. The ncnn/Vulkan overflow
story is about half-precision paths without a wide accumulate, which is not this GPU - the report
itself already said so.

**The E16 ridge argument was half right, and the half that was wrong is the actionable one.** The
literature agrees that Winograd's arithmetic intensity is 2.25x lower than GEMM convolution, and that
its transform stages are memory-bound. It does not conclude "do not use Winograd": PPoPP'20 raises
intensity by growing the cache block (b_k 32 -> 64 takes it from 8 to 10.67 ops/byte, +33%), and PKF
(NPC'22) reports 8.22-13.69x over cuDNN by fusing the transform stages by memory level. The intensity
drop is a shared-memory/L2 problem, which is the layer we can influence.

**The pruning family exists and is training-free, which is a question before it is a task.** The SVD
test measured weight energy and layer output error; ASVD scales weights by activation statistics so
the decomposition minimises output error, SparseGPT prunes one-shot with layer-wise reconstruction,
and SNOWS does the vision-network version without retraining (ResNet20 2:4 at 1.97x FLOPs, -0.64%).
Two caveats travel with it: 2:4 needs sparse hardware and TensorRT already selects sparse tactics
with no measured win here (E7), so the useful form is structural channel pruning on dense kernels;
and reconstruction changes the weights, which the brief forbids. That is an owner's decision, filed
as such rather than absorbed into a technical plan.

**The film loss is not in the model.** On 35 mm the gate refuses (floor 2.27) and the render stays at
1.85x against the 2-4x the same engine reaches on anime. Two standard techniques from video coding
apply to the SR output rather than to the network: motion-compensated prediction (warp the previous HR
frame, recompute only uncovered areas - about 1-2 ms at 1080p against 45 ms of model) and AV1-style
film grain synthesis (strip grain before, synthesise after, which is the only way the floor falls
below 1.5). Both change what the acceptance metric sees, and at sigma=16 the mean absolute deviation
is already about 18 levels, so an 8-level bar on grain is unreachable by construction. This is a
metric conversation with the owner, not an optimisation.

Queue ids keep the reserved numbers: E26 Winograd tactics, E27 int8 PReLU plugin, E28 Welder on our
ONNX, E29 L2-resident pyramid, E30 pruning bounds (owner decision), E31 film gate via codec-side
techniques. They become ledger rows when they have a number.

## Open questions

- ~~Is the film gate worth its quality cost?~~ **No, and the question is closed by the noise-floor
  guard.** Measured before it: 18.1% reuse, 4.33 s against 4.66 s (1.08x), average 50.34 dB and
  **worst frame 39.83 dB** - not transparent for 7%. The clip's floor is 2.27, i.e. above the 1.5
  limit, so the gate now refuses and the render is byte-identical to ungated (94/94 frames).
- **Does the tile probe generalise?** It reads 16 frames at 4 positions. On our two clips it
  classified both correctly after the margin rule, but the prediction is off in both directions
  (anime 33% predicted against 58% achieved, film 15% against 9.5%) and the sample is two titles.
  Three more anime clips with different grain would settle whether "cost must clear 1/area_tax"
  is the right bar or luck.
- ~~The GUI never passes any reuse flag.~~ Now it passes one: `--quality_preset`, exposed as a
  "Quality preset" combo in the render options and defaulting to `best`. The tile and threshold
  knobs stay CLI-only on purpose - they are the ones that need measurement to use well.
- **The GUI's output depth** comes from its own `--custom_encoder` string within
  `GenerateFFMpegCommand`, not from `--video_pixel_format`; the backend's depth resolution is
  bypassed there, and `auto_hdr_mode` promotes a 10-bit source on its own.
- **The pooled payload corruption is explained but unverified.** The writer instrumentation
  shows the queue never fills (depth 4 of 25), so the mechanism must be a slot released before
  its last byte was copied. The feature is off; re-verifying it would buy 0.79 ms/frame.
- ~~Does the chain-aware tuner over-tighten?~~ **Yes, and it was a correctness bug, not calibration.**
  With the one-step budget at every depth it certified chain 1 on anime (11.5 s); with the chain
  budget on the proxy's own scale it certifies chain 4 (11.7 s). That history turned out to be the
  same over-strictness in a different place: the tuner also scored depths the gate cannot reach, and
  fixing that (see the chain section above) moved the real Blu-rays from chain 1 to chain 2 and the
  time from 9.94 s to 6.56 s with the same defect.
- ~~Why does the P010 pipe diverge (31 dB)?~~ Answered: two faults, see C4. The packing was never
  the problem; the command declared a format the packer did not produce, and P010 additionally
  fell through to the RGB branch.
- ~~Is the app's per-frame model time really higher than the isolated benchmark?~~ Answered: no,
  or almost. 51.37 ms standalone against 54.3 in-app at the same shape, with an NVENC encode
  alongside costing 9% in a controlled A/B (51.37 -> 55.99). The earlier 23-43% "gap" was a
  cross-session comparison against a session in which this machine was 26% faster.
