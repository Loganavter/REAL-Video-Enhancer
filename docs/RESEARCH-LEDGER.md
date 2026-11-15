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

Per frame the network is 1.85 TFLOP at 1440x1080 (2.48 at 1920x1080) and runs at 35.1 /
45.7 ms, i.e. **53-54 TFLOP/s**, or ~58 TFLOP/s at the `tiling=full` default (derived, see
MODEL.md §2). That is ~78% of a *borrowed* 74 TF/s sustained fp16-accumulate figure, and we
do **not** have a trustworthy ceiling for this conv shape - MODEL.md §3 explains why the
standalone-conv experiments cannot supply one. So "kernel-level tuning is finished" is not
something the numbers say. What the numbers say is narrower, and still useful:

- the model runs faster than any of its own convolutions measured standalone, and the only
  compute-dominated standalone conv (256->256, 49 TF/s) is *slower* than the whole network;
- seven kernel-level ideas are closed by measurement (below: E5, E9', E12, E13, E14, E17, and E19's
  tiling hypothesis), five are open and all five are quantified: E16 Winograd, E18 layout kernels
  (0.81 ms), E19 the head convolution's 3-channel shape (2.40 ms), E20 head/tail tactics (1.31 ms),
  E21 upsampler/reshape fusion (1.04 ms) - the last four are 5.56 ms, 9.8% of the frame, and E10
  (hand-written kernels) is untested;
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
| E9' | INT8 with explicit Q/DQ, per-channel weights, calibrated on real frames | TRT PTQ docs | 2x of the arithmetic | **measured: accurate, slower, and the mechanism is the boundary, not the kernel.** Calibrated per-tensor activation scales from 8 real frames, per-channel int8 weights, 16 body convs quantised, head and tail in FP16: **0.000% of pixels above 8 levels against the FP16 engine, peak 4, PSNR 54.1 dB** - the format is essentially lossless here. Time **68.11 ms against 55.54 (0.82x)**. Per-layer (`trt.IProfiler`): the 16 int8 convs are `sm80_xmma_..._interleaved_i8f16_i8i32_f32_nchw_vect_c_...` - the **interleaved NC/32HWC32 tensor-core layout, not plain NCHW** - at 1.56-1.88 ms each, i.e. 152.9 GFLOP / 1.68 ms = **91 TFLOP/s against 52 TFLOP/s for the fused fp16 block**; the PReLU runs as **17 separate `PointWiseV2` passes** of 1.71-2.37 ms (31.2 ms, 46% of the engine) because a per-channel slope (1088 values, -1.39..1.70, per-channel Cv > 1) does not fit an int8 epilogue. In fp16 those 16 blocks are one fused myelin kernel each (`__myl_CorrMulLtSele`, 2.93 ms). The engine holds **21 reformat layers of 78**, worth **5.6 ms in total** (8% of the engine) against **32.7 ms for the 17 PReLU passes** (46%) - the "60 reformats" in earlier revisions was 78 minus 18 convs. Second build, quantising the conv output too so the PReLU reads int8: **78.36 ms**, 36 reformats, 16 `DQout` passes at 1.64 ms - the boundary pass is charged per hop, so adding a hop makes it worse | **rejected as quantised** (measured, with the mechanism): the per-channel PReLU forces one int8->fp16->int8 hop per conv, and each hop costs about what int8 saves on the conv. Precision path closed, pending a plugin that fuses conv+PReLU in int8 |
| E10 | hand-written conv kernel / plugin | CORE-TAG/rtx50-realesrgan-video | - | 39.8 TFLOP/s vs the engine's 186.7 on a 5090. Counter-precedent: for a 3x3 conv cuDNN chose Winograd at 74 ms where TRT's best tactic was 156 ms, so TRT's tactic set is not the set of all algorithms | **untested** (was "rejected": the source is one attempt on another card, and it does not measure this GPU) |
| E11 | L2 persistence window | blog measurement (H100/3090) | 1.02-1.20x | nothing to pin: 1.2 MB of weights. The old justification ("265 MB activation stream") predates the myelin fusion finding - the engine is 25 top-level layers, not 18 separate kernels with a full round trip each - so how much traffic actually remains is unknown; the verdict rests on the weights being 1.2 MB against **32 MB** of L2 (AD106; 24 MB is the 4060 without Ti), not on the traffic count | **rejected** (justification corrected) |
| E12 | per-layer precision constraints (force fp16 accumulate) | TRT docs | unknown | closed by measurement, not by an API limit: the DETAILED tactic dump shows all 16 body convs already running `f16f16_f16f16` (fp16 accumulation, NHWC). There is no fp32-accumulate share to convert | **closed (already in effect)** |
| E19 | make the one unfused PReLU fold into the convolution before it | `nsys` per-kernel trace + the engine's subgraph metadata | 2.40 ms (4.2% of the frame) | **located, and the first fix hypothesis is measured and refuted.** The engine fuses 16 of its 17 PReLUs into the body GEMMs (`__myl_CorrMulLtSele`, body.2..body.32) and leaves exactly one separate (`__myl_MulLtSele_myl3_2`), which the metadata attributes to **body.1/PRelu** - directly behind the head convolution. Hypothesis was "the head got `indexed_wo_smem` instead of an xmma with an epilogue, so force another tactic"; **measured false**: rebuilding the same ONNX at `tiling_optimization_level` none, moderate and full gives the same 16 fusion layers and the same single separate kernel every time (moderate even changes the layer mix to 29 with 3 correlations and still leaves it). So the fusion is not a tiling accident. The head is a **3-channel** convolution (RGB in) while the fused pattern is what the 64->64 body convs get - a different GEMM class, K=27 against K=576. The next test is graph surgery, not tactics, and **it is one operation with E18, not two items**: pad the input and the head's weights to **8 channels** (fp16 dense tensor cores want C % 8 == 0; 4 would leave it unaligned and probably reproduce the same unfused path) with zeros, so the head becomes a class the epilogue supports - **+0.48% of the model's FLOPs** (the head is 7.17 GFLOP, 0.29% of the model, and 8/3 of it is +11.9 GFLOP). Order matters more than the padding: padding on the host adds 20.7 MB to the H2D (+3.5 ms at 6 GB/s - a regression), padding on the GPU before the layout kernel grows E18's 0.61 ms to ~1.6 and eats 40% of the prize, while **padding inside the colour kernel straight into NHWC-8** removes E18 (-0.81), folds the PReLU (-2.40), and costs the head +0.07 (narrower input) and +0.25 (3 K-steps instead of 1) - net **about -2.9 ms, 5.1%**. Arithmetic of the head itself: 2.40 - 1.45 = **0.95 ms** if the head doubled, not 1.35 as an earlier revision of this row said. The prize is real: the PReLU pass moves 530.8 MB, whose floor is 1.84 ms, against 2.40 measured = 1.30x | **open** (located; the tiling hypothesis is refuted; next is the padded head, together with E18) |
| E20 | head and tail convolution tactics | `nsys` | ~1.31 ms | both run `indexed_wo_smem` at 1.45 and 1.92 ms against a traffic floor of about 2.06 - the gap is 1.31 ms across the two, and it is the same tactic question as E19 | open |
| E21 | express the nearest upsampler as a shuffle instead of a gather | `nsys`: `ResizeNearest2D` 0.37 + `__myl_TranReshMoveAdd` 1.06 | ~1.04 ms | 1.43 ms against a ~0.39 floor. Nearest upsample is a pure gather, and `Reshape (1,C,H,1,W,1) -> Expand (...,2,...,2) -> Reshape (1,C,2H,2W)` is the same function in shuffle form, which TRT's shuffle fusion folds into the consumer's indexing - the 0.37 ms kernel then disappears without a new kernel. That is separate from the DepthToSpace idea, which is about PixelShuffle and not about the residual path | open |
| E18 | NHWC input/output for the engine (removes the layout kernels) - **and see E19: this and the head padding are one operation** | nsys per-kernel trace of the fp16 plan | ~1.5% (0.81 ms/frame) | measured cost, not yet removed: `cuInt8::nchwTonchw<__half, float, 4>` 0.61 ms and `CUTENSOR permutationKernelPLC3` 0.20 ms run once per frame, and no layer in `IProfiler` owns them. It is the only clean ~1.5% left in the engine | open (small, well-defined) |
| E13 | concurrent engine contexts (batch/streams) | vs-mlrt wiki: 1.65 -> 2.25x on a 2nd stream | up to 2x | no gain (4.25 -> 4.27 s) and output corrupt (25 dB) - because two `torch.jit.load` modules of one engine file are not independent contexts | **rejected as wired** (a wiring fault, not a GPU property) |
| E14 | alternative tactics for the 16 body convs | our tactic dump: 17 of 18 convs get `sm75_` kernels on an sm89 part | unknown | **measured and closed.** With the tactic sources restricted to JIT convolutions only (no cuDNN, no cuBLAS) the winner is the *same* `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32`, at 49.97 ms against the baseline's 50.09 (0.998x). So there is no sm80/86/89 alternative being skipped for this shape at K=576 in this TRT build; the `sm75_` name is the name of the winner, not a symptom. Restricting to cuDNN alone is 10.7% **slower** (55.43 ms, stable across three rounds), so TRT's own kernels beat cuDNN here | **closed** (measured) |
| E15 | crop the active picture area (letterbox bars) | - | up to 1.33x if 25% of the frame is black | **n/a on our material**: the film clip (1440x1080) and the anime clip (1920x1080) both have zero rows darker than 6/255, minimum row mean 59.9 and 124.8. Worth one row-histogram check per new source before rendering | open (source-dependent) |
| E16 | Winograd F(2,3)/F(4,3) | cuDNN/CORE-TAG counter-precedent | F(2,3) cuts MACs 2.25x, inflates tensors ~4x | not measured. The ridge is **128 MAC/byte** at the borrowed 74 TF/s and 288 GB/s (257 FLOP/byte), not 42: 42 corresponds to a ~24 TF/s fp32-shader peak, so the earlier figure was off by 3x in the wrong direction. The conclusion is unchanged - Winograd inflates tensors ~4x, which drops intensity below the ridge - but it is a weaker argument than it looked, and cuDNN-only builds measuring 10.7% *slower* also weakens the counter-precedent | open (untested) |
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
   is fast (91 TFLOP/s against 52 for the fused fp16 block), and the build is still slower - 68.11
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
3. **E16 Winograd.** Cheap, expected near zero, and its counter-precedent is weaker than it looked:
   restricting the build to cuDNN alone made the engine 10.7% *slower*, so this TRT build's own
   kernels beat cuDNN's for this shape, whatever way cuDNN wins a single-conv race elsewhere. (E17
   used to be listed here as a micro-optimisation; it is now part of E9' above, where it turned out
   not to unlock anything.)
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
