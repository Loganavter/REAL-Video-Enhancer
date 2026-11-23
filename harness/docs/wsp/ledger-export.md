# RVE evidence index - export

Generated from `harness/docs/wsp/ledger.db` (93 records, 213 claims, 23 verdicts: 17 current, 5 superseded, 1 refuted).

This is a snapshot. The source of truth is the files - `harness/docs/wsp/verdicts.yaml`, `harness/docs/wsp/verified/records/*.jsonl`, `harness/docs/claims-*.txt` - and the index is rebuilt from them with `python -m harness.cli db build`. A verdict marked *superseded* is kept only so that the replacement can be followed; do not cite one.

## Current state, by lane

### engine

**b3-prelu-forms** (2026-11-17) - *synthesized 16-block graph, C=32 and C=64, 1920x1080 fp16, opt 5, 8 GiB workspace*

> At C=32 the PReLU fuses (19 myelin, 23.51 ms); the a*x+(1-a)*relu rewrite fuses too and is 6% slower (25.06 ms); scalar-slope LeakyReLU breaks fusion (26.89 ms). The C=32 build is bimodal - 39.23 / 39.29 / 20.95 ms from one ONNX - while C=64 is stable at 46.80 ms (probe) and 48.4-51.4 ms (real checkpoint, three builds).

Evidence: [`harness/docs/wsp/verified/prelu-forms.txt`](#evidence-docs-wsp-verified-prelu-forms-txt) - full text below

**streams-no-overlap** (2026-11-17) - *two independent TRT contexts, two streams, 1920x1080 fp16*

> A second stream buys 1.001x of throughput on both a 87.9 ms split plan and a 54.0 ms fused one, with SM at 100% and 155-162 W of a 165 W cap: no overlap, so the strips proposal cannot pay its halo tax.

Evidence: [`harness/docs/wsp/verified/streams.txt`](#evidence-docs-wsp-verified-streams-txt) - full text below

**pyramid-budget** (2026-11-17) - *whole 16-boundary body, 1080p and 4K, CPU arithmetic*

> The L2 reuse buffer is 7.5 MB at 1080p and 15.0 MB at 4K, both inside 32 MiB, but the fp16 compute floor (33.1 / 132.2 ms) sits above the unabsorbed DRAM time (29.5 / 118.0 ms), so absorbing the traffic cannot raise the roofline: 1080p is 56.8 ms measured against a 33.1 ms floor = 1.72x whatever L2 does.

Evidence: [`harness/docs/wsp/verified/pyramid-budget.txt`](#evidence-docs-wsp-verified-pyramid-budget-txt) - full text below

### int8

**int8-graph-launcher** (2026-11-17) - *CUDA graph over 16 int8 launches, head and tail leading out of the graph*

> The 16-launch chain is bit-exact under one CUDA graph capture, with the fifth contract (the capture runs on torch's stream) satisfied.

Evidence: [`harness/docs/wsp/verified/int8-block.txt`](#evidence-docs-wsp-verified-int8-block-txt) - full text below

**trt-int8-rate-79** (2026-11-17) - *TRT plan c2x_int8_fp16heads.plan, 84 layers, 22 reformat, 19 pointwise*

> The TRT int8 plan's convolutions run at a median 79.2 TOPS (min 76.2, max 119.7 over 18 conv layers), and the plan takes 67.67 ms/frame. This is the TRT two-pass build, NOT our kernel.

Evidence: [`harness/docs/wsp/verified/int8-tactics.txt`](#evidence-docs-wsp-verified-int8-tactics-txt) - full text below

**int8-body-hop** (2026-11-17) - *body_kernel (our Triton int8 conv + per-channel PReLU + per-channel requant), 1920x1080, swept BLOCK_M and warps*

> One hop is 1.82 ms at BLOCK_M 256 with 4 warps (84 TOPS), 2.38 ms at 128/4 and 4.06 at 64/4; BLOCK_M 512 with 4 warps collapses to 79 ms/hop (register pressure). The 16-hop chain with the per-layer tensors precomputed is 29.07-29.64 ms.

Evidence: [`harness/docs/wsp/verified/int8-window.txt`](#evidence-docs-wsp-verified-int8-window-txt) - full text below

**int8-split-calibration** (2026-11-17) - *calibration coverage, per-title composition*

> Calibration must be split per title: single-set calibration mis-states composition by 20x, and by 7.5x on the second title. This is a correctness condition, not a tuning knob.

Evidence: [`harness/docs/REPORT-int8-lane-2026-11-16.md`](#evidence-docs-report-int8-lane-2026-11-16-md) - full text below

**int8-probe-f-total** (2026-11-17) - *certification probe, per-title, full price on video*

> The probe is built and measured at F_total 65.4 s, which sets break-even at 16.2 min of video (5.6 min once head and tail are fused).

Evidence: [`harness/docs/REPORT-int8-lane-2026-11-16.md`](#evidence-docs-report-int8-lane-2026-11-16-md) - full text below

**int8-composition-d2** (2026-11-17) - *composition at the depth the gate serves, two titles, 6 pairs each*

> At depth 2 the passing composition reads 0.390% (Chainsaw) and 0.166% (Violet) against a 0.1% bar, so both titles refuse. At depth 1 the same filter reads 0.002% and 0.104%.

Evidence: [`harness/docs/claims-int8-stage4.txt`](#evidence-docs-claims-int8-stage4-txt) - full text below

**OPEN:** Chainsaw is 195x its own depth-1 and 78x the fp16 gate's 0.005% at the same depth, while Violet is 1.6x / 1.73x - inside the measured range. One of the two numbers is probably not the population it is labelled with. The discriminating control needs no int8: fp16(t-2) against fp16(t) on the same pairs, at the gate's depth.

**int8-window-measured** (2026-11-17) - *our Triton int8 chain (6 of 16 hops on the three-accumulator kernel, 10 on the epilogue one, channels_last head/tail), 1920x1080, gate off, no encoder, calibration on 9 frames, accuracy on 4*

> The window is 36.19 ms = 27.63 fps against a same-process fp16 TensorRT plan at 48.94 ms = 20.43 fps (1.35x this draw, 1.41x on an earlier one - the fp16 plan read 48.94-55.61 ms across today's runs). The 16-hop chain is 30.48 ms; head+quantise 1.60; tail+residual 5.78. Accuracy: worst of 4 frames 0.09169% against the 0.1% bar, all four held out of the calibration set.

Evidence: [`harness/docs/wsp/verified/int8-window.txt`](#evidence-docs-wsp-verified-int8-window-txt) - full text below

**OPEN:** The peak is 55 levels against the proposed bar of 20, and every configuration measured landed between 29 and 55; the share bar passes, the peak bar does not. The sample is four frames (the bar's minimum is two) - the 13-19 frame spread the protocol asks for is not measured here.

**int8-asymmetric** (2026-11-17) - *our Triton kernel, asymmetric quantisation with the zero-point folded into the bias*

> Asymmetric (uint8) quantisation is implemented in our kernel - zero-point folded into the bias - and measured: 0.102% at peak 32 against 0.058% at peak 37.

Evidence: [`harness/docs/wsp/verified/int8-block.txt`](#evidence-docs-wsp-verified-int8-block-txt) - full text below

**int8-triton-block** (2026-11-17) - *our Triton kernel, one conv+per-channel-PReLU+requant block, 1920x1080*

> 1.91 ms and 79-80 TOPS for one 64->64 3x3 block with the per-channel slope AND a per-channel output scale in the epilogue, bit-exact against the int32 reference.

Evidence: [`harness/docs/wsp/verified/int8-block.txt`](#evidence-docs-wsp-verified-int8-block-txt) - full text below

**int8-restart-under-gate** (2026-11-17) - *shipped configuration, temporal gate on, tilesize=0*

> With tilesize=0 the gate creates no tile_upscaler, the body branch is reachable, and int8 starts under the gate (int8Fit=int8). The old "--no_temporal_reuse is a precondition" is refuted.

Evidence: [`harness/docs/REPORT-int8-lane-2026-11-16.md`](#evidence-docs-report-int8-lane-2026-11-16-md) - full text below

**int8-live-verdicts** (2026-11-17) - *the app's own int8 fit probe (backend/src/utils/Int8Probe.py), three clips, 1920x1080, gate on*

> The product accepts int8 where the material is easy (smoke clip: 0.003% share, peak 13 -> int8, body 37.9 ms) and refuses it on BOTH real titles on the DEPTH-2 composition: Violet d1 0.058% / d2 0.122%, Chainsaw d1 0.000% / d2 0.218%, against a 0.1% bar - the verdict is max(d1, d2) while the log prints d1 only, and the peak (37 and 11) is reported but does not decide.

Evidence: [`harness/docs/wsp/verified/int8-live.txt`](#evidence-docs-wsp-verified-int8-live-txt) - full text below

**OPEN:** The blocker to fix is the depth-2 composition (0.122% / 0.218% against 0.1%), i.e. the gap between a frame and a reference two steps back, not the peak: bar_int8_peak (20) is also unreachable (29-55 across twelve configurations) but the product does not refuse on it.

**int8-peak-unmet** (2026-11-17) - *the measured int8 chain family (epilogue and three-accumulator kernels), 1920x1080, 4 held-out frames*

> Twelve configurations were measured for the peak, and every one lands between 29 and 55 levels against bar_int8_peak = 20: margin 1.2 with per-tensor grids reads 29 levels at 0.124% share, while the hybrid that passes the share bar (0.09169%) reads 55. The lowest peak and the passing share are not available in the same configuration of this kernel family.

Evidence: [`harness/docs/wsp/verified/int8-window.txt`](#evidence-docs-wsp-verified-int8-window-txt) - full text below

**OPEN:** bar_int8_peak (20) is still marked "PENDING OWNER DECISION" in harness/docs/claims-bars.json, and the product's own probe does not refuse on it - the refusals are compositional (d2). So this row is accepted debt with a measured range (29-55) rather than an active blocker; the active blocker is int8-composition-d2.

### reuse

**cone-budget** (2026-11-17) - *16-layer body, changed square r=5..40 px, cone halo (17-L) per layer*

> The cone is 2.54x less work than the cylinder at r=5 and 2.12x at r=10 (1-D control reproduces the ledger's 2.00x), but sixteen launches cost 0.08 ms while the work at r<=20 is below that: the prize is 0.08-0.27 ms instead of a 56.8 ms re-render, not a ratio on the engine.

Evidence: [`harness/docs/wsp/verified/cone-budget.txt`](#evidence-docs-wsp-verified-cone-budget-txt) - full text below

**OPEN:** Phase 2 needs a region-level certificate, which does not exist: the gate's veto is frame-level, and the seams between a recomputed cone and a reused neighbourhood are the artifact class that made tile reuse read 0.287%.

**e37-reachability** (2026-11-17) - *app's temporal gate, Violet 20 s and Chainsaw Man 20 s, 1920x1080, chain certified per clip*

> The premise of rve-hr-reference-cache ("the gate stores ONE reference") is stale: the gate keeps a chain and certified depth 4 on Violet (p90 1.85/4.66/6.98/9.61 against a limit of 12, monotonically worse) and depth 3 on Chainsaw (0.73/0.69/0.66, better by 0.07 units out of 12). On the clip where more references help, the gain is 0.6% of the certification budget, and the reuse there (100/481 = 20.8%) is limited by the threshold rather than by reachability.

Evidence: [`harness/docs/wsp/verified/int8-live.txt`](#evidence-docs-wsp-verified-int8-live-txt) - full text below

## Superseded and refuted - do not cite

| verdict | status | replaced by | scope |
|---|---|---|---|
| `e17-prelu-rewrite` | superseded | `b3-prelu-forms` | TRT graph, C=64, 2026-11-15 |
| `int8-window` | superseded | `int8-window-measured` | model window (engine, no encoder), one process, 1920x1080, session stage4 |
| `model-exactly-one-route` | superseded | `int8-graph-launcher` | MODEL.md section 6, "distance to 2x" |
| `report8-no-temporal-reuse` | refuted | `int8-restart-under-gate` | REPORT-int8-lane-2026-11-16 section 8, launch precondition |
| `summary-lower-precision-closed` | superseded | `int8-triton-block` | ledger summary table, row "lower precision (INT8/FP8)" |
| `trt-int8-two-pass` | superseded | `int8-triton-block` | TRT explicit Q/DQ build, per-tensor activation scales, 2026-11-15 |

## Records

Each row is a measurement tuple: value, unit, statistic, and the population it was taken on. `of` is what the number is *of*, which is the field the old prose was missing.

### `harness/docs/wsp/verified/records/cone-budget.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 18 | count | single | receptive_radius | assumed | the 16-body-conv receptive path |
| R002 | 35344 | count | single | cylinder_px | assumed | changed square r=5 (side 11 px), 1920x1080, cone halo 1 px/layer |
| R003 | 13904 | count | single | cone_px | assumed | changed square r=5 (side 11 px), 1920x1080, cone halo 1 px/layer |
| R004 | 2.542 | x | single | work_ratio | derived | changed square r=5 (side 11 px), 1920x1080, cone halo 1 px/layer |
| R005 | 0.08 | ms | single | cone_time_bound | assumed | changed square r=5 (side 11 px), 1920x1080, cone halo 1 px/layer |
| R006 | 51984 | count | single | cylinder_px | assumed | changed square r=10 (side 21 px), 1920x1080, cone halo 1 px/layer |
| R007 | 24464 | count | single | cone_px | assumed | changed square r=10 (side 21 px), 1920x1080, cone halo 1 px/layer |
| R008 | 2.12492 | x | single | work_ratio | derived | changed square r=10 (side 21 px), 1920x1080, cone halo 1 px/layer |
| R009 | 0.08 | ms | single | cone_time_bound | assumed | changed square r=10 (side 21 px), 1920x1080, cone halo 1 px/layer |
| R010 | 94864 | count | single | cylinder_px | assumed | changed square r=20 (side 41 px), 1920x1080, cone halo 1 px/layer |
| R011 | 55184 | count | single | cone_px | assumed | changed square r=20 (side 41 px), 1920x1080, cone halo 1 px/layer |
| R012 | 1.71905 | x | single | work_ratio | derived | changed square r=20 (side 41 px), 1920x1080, cone halo 1 px/layer |
| R013 | 0.094 | ms | single | cone_time_bound | assumed | changed square r=20 (side 41 px), 1920x1080, cone halo 1 px/layer |
| R014 | 219024 | count | single | cylinder_px | assumed | changed square r=40 (side 81 px), 1920x1080, cone halo 1 px/layer |
| R015 | 155024 | count | single | cone_px | assumed | changed square r=40 (side 81 px), 1920x1080, cone halo 1 px/layer |
| R016 | 1.41284 | x | single | work_ratio | derived | changed square r=40 (side 81 px), 1920x1080, cone halo 1 px/layer |
| R017 | 0.265 | ms | single | cone_time_bound | assumed | changed square r=40 (side 81 px), 1920x1080, cone halo 1 px/layer |

### `harness/docs/wsp/verified/records/engine-fp16.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 56.26 | ms | median | frame_time | measured | /tmp/opencode/compact2x_1080p_static.plan: 1920x1080, fp16, one proces |
| R002 | 56.45 | ms | max | frame_time | measured | /tmp/opencode/compact2x_1080p_static.plan: 1920x1080, fp16, one proces |
| R003 | 55.08 | ms | median | frame_time | measured | /tmp/opencode/compact2x_1080p_static.plan: 1920x1080, fp16, one proces |
| R004 | 55.95 | ms | max | frame_time | measured | /tmp/opencode/compact2x_1080p_static.plan: 1920x1080, fp16, one proces |
| R005 | 56.3 | ms | median | frame_time | measured | /tmp/opencode/compact2x_1080p_static.plan: 1920x1080, fp16, one proces |
| R006 | 56.34 | ms | max | frame_time | measured | /tmp/opencode/compact2x_1080p_static.plan: 1920x1080, fp16, one proces |

### `harness/docs/wsp/verified/records/engine-int8.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 70.85 | ms | median | frame_time | measured | /tmp/opencode/c2x_int8_fp16heads.plan: 1920x1080, fp16, one process, n |
| R002 | 71.17 | ms | max | frame_time | measured | /tmp/opencode/c2x_int8_fp16heads.plan: 1920x1080, fp16, one process, n |
| R003 | 0.0097 | % | single | share | measured | heldout_f100.raw: compact2x_1080p_int8.plan vs compact2x_1080p_static. |
| R004 | 18.87 | levels | max | peak | measured | heldout_f100.raw: compact2x_1080p_int8.plan vs compact2x_1080p_static. |
| R005 | 44.36 | dB | single | quality | measured | heldout_f100.raw: compact2x_1080p_int8.plan vs compact2x_1080p_static. |
| R006 | 0.2908 | % | single | share | measured | heldout_f100.raw: compact2x_1080p_int8_p9999.plan vs compact2x_1080p_s |
| R007 | 65.11 | levels | max | peak | measured | heldout_f100.raw: compact2x_1080p_int8_p9999.plan vs compact2x_1080p_s |
| R008 | 47.72 | dB | single | quality | measured | heldout_f100.raw: compact2x_1080p_int8_p9999.plan vs compact2x_1080p_s |
| R009 | 4.3885 | % | single | share | measured | heldout_f100.raw: c2x_int8_fp16heads.plan vs compact2x_1080p_static.pl |
| R010 | 89.48 | levels | max | peak | measured | heldout_f100.raw: c2x_int8_fp16heads.plan vs compact2x_1080p_static.pl |
| R011 | 37.95 | dB | single | quality | measured | heldout_f100.raw: c2x_int8_fp16heads.plan vs compact2x_1080p_static.pl |

### `harness/docs/wsp/verified/records/int8-block-probe.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 1.848 | ms | median | engine_time | measured | one 3x3 conv 64->64 NHWC int8 + per-channel PReLU + requant, 1920x1080 |
| R002 | 82.8 | TOPS | median | rate | measured | one 3x3 conv 64->64 NHWC int8 + per-channel PReLU + requant, 1920x1080 |

### `harness/docs/wsp/verified/records/int8-block.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 1.84 | ms | median | engine_time | measured | one 3x3 conv 64->64 NHWC int8 + per-channel PReLU + requant, 1920x1080 |
| R002 | 83.1 | TOPS | median | rate | measured | one 3x3 conv 64->64 NHWC int8 + per-channel PReLU + requant, 1920x1080 |

### `harness/docs/wsp/verified/records/int8-body-ve.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 0.0062 | % | single | share | measured | int8_ve_out.raw vs ref_ve_out.raw at 3840x2160 |
| R002 | 11.95 | levels | max | peak | measured | int8_ve_out.raw vs ref_ve_out.raw at 3840x2160 |
| R003 | 53.15 | dB | single | quality | measured | int8_ve_out.raw vs ref_ve_out.raw at 3840x2160 |

### `harness/docs/wsp/verified/records/int8-live.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 0.003 | % | worst_of | share | measured | smoke clip 1 s, 1920x1080 -> 2160p, gate on, int8 ACCEPTED |
| R002 | 13 | levels | max | peak | measured | smoke clip 1 s, 1920x1080 -> 2160p, gate on, int8 ACCEPTED |
| R003 | 37.9 | ms | mean | body_time | measured | smoke clip 1 s, 1920x1080 -> 2160p, gate on, int8 ACCEPTED |
| R004 | 37.5 | ms | mean | frame_time | measured | smoke clip 1 s, 1920x1080 -> 2160p, gate on, int8 ACCEPTED |
| R005 | 0.058 | % | worst_of | share | measured | Violet Evergarden 20 s, 1920x1080 -> 2160p, gate on, int8 refused |
| R006 | 37 | levels | max | peak | measured | Violet Evergarden 20 s, 1920x1080 -> 2160p, gate on, int8 refused |
| R007 | 69.8 | % | single | reuse_share | measured | Violet Evergarden 20 s, 1920x1080 -> 2160p, gate on, int8 refused |
| R008 | 35.12 | ms | mean | frame_time | measured | Violet Evergarden 20 s, 1920x1080 -> 2160p, gate on, int8 refused |
| R009 | 9.61 | levels | p90 | reference_delta | measured | Violet Evergarden 20 s, 1920x1080 -> 2160p, gate on, int8 refused |
| R010 | 20.8 | % | single | reuse_share | measured | Chainsaw Man 20 s, 1920x1080 -> 2160p, gate on, int8 refused |
| R011 | 47.58 | ms | mean | frame_time | measured | Chainsaw Man 20 s, 1920x1080 -> 2160p, gate on, int8 refused |
| R012 | 0.66 | levels | p90 | reference_delta | measured | Chainsaw Man 20 s, 1920x1080 -> 2160p, gate on, int8 refused |

### `harness/docs/wsp/verified/records/int8-pipeline-ve.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 0.00644 | % | single | share | measured | ve60.raw 1920x1080, int8 chain vs fp16 model, 2x-Futsuu-Anime(1).pth |
| R002 | 12 | levels | max | peak | measured | ve60.raw 1920x1080, int8 chain vs fp16 model, 2x-Futsuu-Anime(1).pth |
| R003 | 53.94 | ms | mean | engine_time | measured | 16 int8 hops at 1920x1080 |

### `harness/docs/wsp/verified/records/int8-tactics.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 67.67 | ms | median | frame_time | measured | int8 plan as-built: 1920x1080, one process, no encoder |
| R002 | 79.2 | TOPS | median | per_conv_rate | measured | per-conv rate, as-built |

### `harness/docs/wsp/verified/records/int8-window.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 47.81 | ms | mean | frame_time | measured | whole model window at 1920x1080, int8 body, no gate, no encoder |
| R002 | 36.19 | ms | mean | frame_time | measured | whole model window at 1920x1080, int8 body, no gate, no encoder |
| R003 | 48.94 | ms | mean | frame_time | measured | whole model window at 1920x1080, no gate, no encoder |
| R004 | 0.09169 | % | worst_of | share | measured | 4 frames, 1920x1080, int8 epilogue chain vs fp16, calibration on frame |
| R005 | 55 | levels | max | peak | measured | 4 frames, 1920x1080, int8 epilogue chain vs fp16, calibration on frame |

### `harness/docs/wsp/verified/records/model-prelu.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 1.69556 | x | max | slope | measured | all 17 PReLU tensors of the 2x Futsuu model |
| R002 | -1.38866 | x | min | slope | measured | all 17 PReLU tensors of the 2x Futsuu model |
| R003 | 1.62437 | x | median | slope | measured | all 17 PReLU tensors of the 2x Futsuu model |

### `harness/docs/wsp/verified/records/prelu-forms.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 23.51 | ms | median | frame_time | measured | C=32 prelu: whole 16-block body, 1920x1080, synthesized random weights |
| R002 | 25.06 | ms | median | frame_time | measured | C=32 affine: whole 16-block body, 1920x1080, synthesized random weight |
| R003 | 26.89 | ms | median | frame_time | measured | C=32 leaky: whole 16-block body, 1920x1080, synthesized random weights |
| R004 | 46.8 | ms | median | frame_time | measured | C=64 prelu: whole 16-block body, 1920x1080, synthesized random weights |

### `harness/docs/wsp/verified/records/pyramid-budget.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 7.5 | MB | single | reuse_buffer_total | assumed | 1920x1080, 64 ch fp16, 2-row reuse buffer per layer boundary |
| R002 | 29.5 | ms | single | dram_time_at_zero_absorption | assumed | 1920x1080, 64 ch fp16, 2-row reuse buffer per layer boundary |
| R003 | 33.1 | ms | single | compute_floor_fp16 | assumed | 1920x1080, 64 ch fp16, 2-row reuse buffer per layer boundary |
| R004 | 56.8 | ms | median | frame_time | assumed | 1920x1080, 64 ch fp16, 2-row reuse buffer per layer boundary |
| R005 | 1.71601 | x | single | ceiling_vs_measured | derived | 1920x1080, 64 ch fp16, 2-row reuse buffer per layer boundary |
| R006 | 15 | MB | single | reuse_buffer_total | assumed | 3840x2160, 64 ch fp16, 2-row reuse buffer per layer boundary |
| R007 | 118 | ms | single | dram_time_at_zero_absorption | assumed | 3840x2160, 64 ch fp16, 2-row reuse buffer per layer boundary |
| R008 | 132.2 | ms | single | compute_floor_fp16 | assumed | 3840x2160, 64 ch fp16, 2-row reuse buffer per layer boundary |

### `harness/docs/wsp/verified/records/residual-scale-ve.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 0.561807 | levels | median | levels | measured | ve60.raw 1920x1080, correction |out-nearest| |
| R002 | 1.88003 | levels | p90 | levels | measured | ve60.raw 1920x1080, correction |out-nearest| |
| R003 | 4.35137 | levels | p99 | levels | measured | ve60.raw 1920x1080, correction |out-nearest| |
| R004 | 8.24693 | levels | p99.9 | levels | measured | ve60.raw 1920x1080, correction |out-nearest| |
| R005 | 33.3861 | levels | max | levels | measured | ve60.raw 1920x1080, correction |out-nearest| |
| R006 | 99.8865 | % | single | levels | measured | ve60.raw 1920x1080, correction |out-nearest| |

### `harness/docs/wsp/verified/records/residual-scale.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 1.16189 | levels | median | levels | measured | frame1080p.raw 1440x1080, correction |out-nearest| |
| R002 | 4.82577 | levels | p90 | levels | measured | frame1080p.raw 1440x1080, correction |out-nearest| |
| R003 | 14.3499 | levels | p99 | levels | measured | frame1080p.raw 1440x1080, correction |out-nearest| |
| R004 | 44.6719 | levels | p99.9 | levels | measured | frame1080p.raw 1440x1080, correction |out-nearest| |
| R005 | 109.031 | levels | max | levels | measured | frame1080p.raw 1440x1080, correction |out-nearest| |
| R006 | 96.5152 | % | single | levels | measured | frame1080p.raw 1440x1080, correction |out-nearest| |

### `harness/docs/wsp/verified/records/shot-cascade.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 76.5169 | % | worst_of | share | measured | 2x_ModernSpanimationV1.pth against 2x-Futsuu-Anime(1).pth, 1920x1080 |

### `harness/docs/wsp/verified/records/streams.jsonl`

| id | value | unit | statistic | of | tier | population |
|---|---|---|---|---|---|---|
| R001 | 1.001 | x | median | concurrency_ratio | measured | 2 independent contexts of one plan, 1920x1080, no encoder |
| R002 | 1.001 | x | median | concurrency_ratio | measured | 2 independent contexts of one plan, 1920x1080, no encoder |

## Claims

The legacy corpus (`harness/docs/claims-*.txt`): names, values and the session that measured them. These are the lines the migrate gate reports on, not record-backed claims.

| name | value | unit | session | source:line |
|---|---|---|---|---|
| `F_total_s` | 65.4 | s | stage4 | `harness/docs/claims-int8-stage4.txt:264` |
| `acceptance_bar` | 0.1 | % | example | `harness/docs/claims-example.txt:32` |
| `adv_chainsaw_comp` | 0.022 | % | stage4 | `harness/docs/claims-int8-stage4.txt:10` |
| `adv_violet_comp` | 0.165 | % | stage4 | `harness/docs/claims-int8-stage4.txt:15` |
| `adv_violet_engine` | 0.058 | % | stage4 | `harness/docs/claims-int8-stage4.txt:14` |
| `app_engine_ms` | 46.4 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:47` |
| `app_engine_ms` | 46.4 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:110` |
| `app_engine_tiled` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:146` |
| `app_frame_ms` | 45.36 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:148` |
| `app_gate_overspend` | 23.42 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:120` |
| `app_head_fused` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:186` |
| `app_inference_1080` | 35.85 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:147` |
| `app_kill_overspend` | 16.2 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:118` |
| `app_supports` | 0 | x | stage4 | `harness/docs/claims-int8-stage4.txt:184` |
| `archivist_convs` | 18 | x | stage5 | `harness/docs/claims-int8-stage5.txt:22` |
| `archivist_feat` | 48 | ch | stage5 | `harness/docs/claims-int8-stage5.txt:21` |
| `bias_mean_body` | 2.11 | % | stage1 | `harness/docs/claims-int8-stage1.txt:21` |
| `body_4k_idle` | 240.51 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:131` |
| `body_4k_nvenc` | 259.74 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:135` |
| `body_4k_share` | 0.11579 | % | stage4 | `harness/docs/claims-int8-stage4.txt:138` |
| `body_app_share` | 0 | % | stage4 | `harness/docs/claims-int8-stage4.txt:122` |
| `body_before` | 2.876 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:27` |
| `body_before_1080` | 3.53381 | % | stage4 | `harness/docs/claims-int8-stage4.txt:34` |
| `body_before_512` | 1.27525 | % | stage4 | `harness/docs/claims-int8-stage4.txt:35` |
| `body_built` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:191` |
| `body_cert_fused` | 0.0082 | % | stage4 | `harness/docs/claims-int8-stage4.txt:233` |
| `body_cert_share` | 0.00832 | % | stage4 | `harness/docs/claims-int8-stage4.txt:212` |
| `body_idle_ms` | 60.41 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:53` |
| `body_ms` | 56.56 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:48` |
| `body_ms_app_base` | 48.55 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:114` |
| `body_nvenc_ms` | 69.03 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:57` |
| `body_per_launch` | 2.771 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:26` |
| `body_rolled_ms` | 48.55 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:159` |
| `body_share_1080` | 0 | % | stage4 | `harness/docs/claims-int8-stage4.txt:41` |
| `body_share_256` | 0 | % | stage4 | `harness/docs/claims-int8-stage4.txt:39` |
| `body_share_512` | 0 | % | stage4 | `harness/docs/claims-int8-stage4.txt:40` |
| `body_vs_app` | 0.82 | x | stage2 | `harness/docs/claims-int8-stage2.txt:49` |
| `body_vs_app` | 0.953 | x | stage4 | `harness/docs/claims-int8-stage4.txt:115` |
| `body_vs_app_median_ratio` | 0.962 | x | stage4 | `harness/docs/claims-int8-stage4.txt:117` |
| `body_vs_app_unrolled` | 0.82 | x | stage4 | `harness/docs/claims-int8-stage4.txt:112` |
| `body_vs_engine_share` | 1e-05 | % | stage4 | `harness/docs/claims-int8-stage4.txt:42` |
| `body_vs_image_scale` | 28.3 | x | stage1 | `harness/docs/claims-int8-stage1.txt:23` |
| `breakeven_min_ht` | 5.6 | min | stage4 | `harness/docs/claims-int8-stage4.txt:269` |
| `breakeven_min_now` | 16.2 | min | stage4 | `harness/docs/claims-int8-stage4.txt:267` |
| `breakeven_rendered` | 47.8 | % | example | `harness/docs/claims-example.txt:5` |
| `build_cost_s` | 0.85 | s | stage4 | `harness/docs/claims-int8-stage4.txt:263` |
| `build_error` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:185` |
| `c10_comp` | 0.011 | % | stage4 | `harness/docs/claims-int8-stage4.txt:11` |
| `calib_frames_prod` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:253` |
| `calibration_gap` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:250` |
| `ceiling_body_ms` | 30.4 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:164` |
| `ceiling_cycle_ms` | 43.7 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:166` |
| `ceiling_ratio` | 1.062 | x | stage4 | `harness/docs/claims-int8-stage4.txt:168` |
| `certify_line` | 0.00823 | % | stage4 | `harness/docs/claims-int8-stage4.txt:192` |
| `chainsaw_composition` | 0.01 | % | stage4 | `harness/docs/claims-int8-stage4.txt:9` |
| `chainsaw_engine` | 0.009 | % | stage4 | `harness/docs/claims-int8-stage4.txt:8` |
| `comp_d1_chainsaw` | 0.002 | % | stage4 | `harness/docs/claims-int8-stage4.txt:277` |
| `comp_d1_violet` | 0.104 | % | stage4 | `harness/docs/claims-int8-stage4.txt:279` |
| `comp_d2_chainsaw` | 0.39 | % | stage4 | `harness/docs/claims-int8-stage4.txt:278` |
| `comp_d2_violet` | 0.166 | % | stage4 | `harness/docs/claims-int8-stage4.txt:280` |
| `comp_single_frame` | 0.116 | % | stage4 | `harness/docs/claims-int8-stage4.txt:254` |
| `comp_single_violet` | 0.377 | % | stage4 | `harness/docs/claims-int8-stage4.txt:256` |
| `comp_spread_frame` | 0.006 | % | stage4 | `harness/docs/claims-int8-stage4.txt:255` |
| `comp_spread_violet` | 0.05 | % | stage4 | `harness/docs/claims-int8-stage4.txt:257` |
| `composition_share` | 0.017 | % | example | `harness/docs/claims-example.txt:25` |
| `contig_head_gain` | -0.3 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:86` |
| `cross_instrument_vs` | 0.631 | x | stage2 | `harness/docs/claims-int8-stage2.txt:51` |
| `cycle_idle_ms` | 56.82 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:92` |
| `cycle_nvenc_ms` | 64.24 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:95` |
| `delivered_fp16_ms` | 46.92 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:195` |
| `delivered_frames` | 24 | x | stage4 | `harness/docs/claims-int8-stage4.txt:193` |
| `delivered_int8_ms` | 96.88 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:194` |
| `delivered_median` | 1 | levels | stage4 | `harness/docs/claims-int8-stage4.txt:200` |
| `delivered_peak` | 24 | levels | stage4 | `harness/docs/claims-int8-stage4.txt:199` |
| `delivered_ratio` | 0.484 | x | stage4 | `harness/docs/claims-int8-stage4.txt:196` |
| `delivered_share` | 0.07592 | % | stage4 | `harness/docs/claims-int8-stage4.txt:198` |
| `depth_ratio_chainsaw` | 195 | x | stage4 | `harness/docs/claims-int8-stage4.txt:281` |
| `depth_ratio_violet` | 1.6 | x | stage4 | `harness/docs/claims-int8-stage4.txt:283` |
| `dequant_block256` | 1.57 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:91` |
| `dequant_fused` | 4 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:29` |
| `design_floor_ms` | 29.9 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:174` |
| `design_ratio` | 1.548 | x | stage4 | `harness/docs/claims-int8-stage4.txt:176` |
| `dot_floor_ms` | 1.9 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:157` |
| `dram_write_bytes` | 184.6 | MB | example | `harness/docs/claims-example.txt:46` |
| `dram_write_bytes_b` | 184.6 | MB | example | `harness/docs/claims-example.txt:47` |
| `engine_4k_idle` | 186.14 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:130` |
| `engine_4k_nvenc` | 198.68 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:134` |
| `engine_idle_ms` | 80.47 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:8` |
| `engine_idle_ms` | 79.86 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:52` |
| `engine_ms_other` | 56.3 | ms | example | `harness/docs/claims-example.txt:52` |
| `engine_ms_violet` | 51.4 | ms | example | `harness/docs/claims-example.txt:51` |
| `engine_nvenc_ms` | 91.91 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:12` |
| `engine_nvenc_ms` | 91.65 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:56` |
| `engine_share` | 0.006 | % | example | `harness/docs/claims-example.txt:24` |
| `engine_vs_torch` | 0 | % | stage2 | `harness/docs/claims-int8-stage2.txt:24` |
| `fixed_costs_ms` | 16.1 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:66` |
| `four_x_out_of_scope` | 1 | x | stage5 | `harness/docs/claims-int8-stage5.txt:41` |
| `fp16_window_ms` | 51.06 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:208` |
| `fused_head_ms` | 1.71 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:230` |
| `futsuu_convs` | 18 | x | stage5 | `harness/docs/claims-int8-stage5.txt:9` |
| `futsuu_feat` | 64 | ch | stage5 | `harness/docs/claims-int8-stage5.txt:8` |
| `futsuu_ratio` | 1.489 | x | stage5 | `harness/docs/claims-int8-stage5.txt:11` |
| `futsuu_scale` | 2 | x | stage5 | `harness/docs/claims-int8-stage5.txt:10` |
| `futsuu_short` | -3.45 | ms | stage5 | `harness/docs/claims-int8-stage5.txt:28` |
| `gate_excess` | 3.34 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:64` |
| `gate_margin` | -0.78 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:98` |
| `gate_margin_nvenc` | -1.28 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:103` |
| `gate_share` | 0.005 | % | example | `harness/docs/claims-example.txt:23` |
| `gather_ms` | 1.06 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:158` |
| `head_codes_differ` | 98.34 | % | stage4 | `harness/docs/claims-int8-stage4.txt:36` |
| `hop_floor_ms` | 1.037 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:172` |
| `hop_ms` | 1.753 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:213` |
| `hop_tops` | 87.2 | TOPS | stage4 | `harness/docs/claims-int8-stage4.txt:214` |
| `int8_idle_ms` | 54.03 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:9` |
| `int8_nvenc_ms` | 61.06 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:13` |
| `int8_vs_engine` | 4e-05 | % | stage2 | `harness/docs/claims-int8-stage2.txt:22` |
| `int8_vs_torch` | 1e-05 | % | stage2 | `harness/docs/claims-int8-stage2.txt:23` |
| `int8_window_ms` | 46.52 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:209` |
| `int8_window_ms2` | 42.75 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:232` |
| `kernel_regs` | 241 | regs | stage1 | `harness/docs/claims-int8-stage1.txt:22` |
| `kernel_vs_sim_chainsaw` | 1.71 | x | stage4 | `harness/docs/claims-int8-stage4.txt:246` |
| `kernel_vs_sim_violet` | 1.69 | x | stage4 | `harness/docs/claims-int8-stage4.txt:248` |
| `kill_excess` | -9.13 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:62` |
| `kill_overspend2` | 8.3 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:162` |
| `margin_chainsaw` | 18.6 | pp | example | `harness/docs/claims-example.txt:6` |
| `modernspan_convs` | 0 | x | stage5 | `harness/docs/claims-int8-stage5.txt:23` |
| `net_24min_ht` | 213.7 | s | stage4 | `harness/docs/claims-int8-stage4.txt:273` |
| `net_24min_now` | 31.8 | s | stage4 | `harness/docs/claims-int8-stage4.txt:271` |
| `nvenc_cost_engine` | 14.22 | % | stage2 | `harness/docs/claims-int8-stage2.txt:17` |
| `nvenc_cost_int8` | 13.01 | % | stage2 | `harness/docs/claims-int8-stage2.txt:19` |
| `pass_rate` | 0.6 | x | stage4 | `harness/docs/claims-int8-stage4.txt:16` |
| `peak_tops` | 147.4 | TOPS | stage4 | `harness/docs/claims-int8-stage4.txt:171` |
| `peak_value` | 7 | levels | example | `harness/docs/claims-example.txt:42` |
| `pipeline_median` | 1.08 | levels | stage1 | `harness/docs/claims-int8-stage1.txt:12` |
| `pipeline_peak` | 7 | levels | stage1 | `harness/docs/claims-int8-stage1.txt:11` |
| `pipeline_share` | 1.87 | % | example | `harness/docs/claims-example.txt:30` |
| `pipeline_share` | 0 | % | stage1 | `harness/docs/claims-int8-stage1.txt:9` |
| `pipeline_violet_4k` | 0.03065 | % | stage1 | `harness/docs/claims-int8-stage1.txt:16` |
| `prelu_fuse_fp16_ms` | 1.522 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:89` |
| `prelu_fuse_fp32_ms` | 12.818 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:90` |
| `prelu_fuse_loss` | 9 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:84` |
| `prelu_passes_ms` | 3.632 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:85` |
| `probe_admitted` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:150` |
| `probe_bug_c2` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:293` |
| `probe_cost_s` | 64.6 | s | stage4 | `harness/docs/claims-int8-stage4.txt:262` |
| `probe_independent` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:227` |
| `probe_oom` | 1 | x | stage4 | `harness/docs/claims-int8-stage4.txt:142` |
| `probe_real_chainsaw` | 0.116 | % | stage4 | `harness/docs/claims-int8-stage4.txt:239` |
| `probe_real_same_cal_chainsaw` | 0.006 | % | stage4 | `harness/docs/claims-int8-stage4.txt:244` |
| `probe_real_same_cal_violet` | 0.05 | % | stage4 | `harness/docs/claims-int8-stage4.txt:245` |
| `probe_real_violet` | 0.377 | % | stage4 | `harness/docs/claims-int8-stage4.txt:241` |
| `probe_served` | 0 | x | stage4 | `harness/docs/claims-int8-stage4.txt:151` |
| `probe_sim_chainsaw` | 0.01 | % | stage4 | `harness/docs/claims-int8-stage4.txt:238` |
| `probe_sim_violet` | 0.084 | % | stage4 | `harness/docs/claims-int8-stage4.txt:240` |
| `quant_phase_ms` | 4.01 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:210` |
| `quant_phase_where_ms` | 9.91 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:211` |
| `quantise_fused` | 0.49 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:28` |
| `ratio_4k_idle` | 0.774 | x | stage4 | `harness/docs/claims-int8-stage4.txt:132` |
| `ratio_4k_nvenc` | 0.765 | x | stage4 | `harness/docs/claims-int8-stage4.txt:136` |
| `ratio_idle` | 1.489 | x | stage2 | `harness/docs/claims-int8-stage2.txt:10` |
| `ratio_idle` | 1.322 | x | stage4 | `harness/docs/claims-int8-stage4.txt:54` |
| `ratio_idle2` | 1.411 | x | stage4 | `harness/docs/claims-int8-stage4.txt:93` |
| `ratio_nvenc` | 1.505 | x | stage2 | `harness/docs/claims-int8-stage2.txt:14` |
| `ratio_nvenc` | 1.328 | x | stage4 | `harness/docs/claims-int8-stage4.txt:58` |
| `ratio_nvenc2` | 1.428 | x | stage4 | `harness/docs/claims-int8-stage4.txt:96` |
| `ratio_rolled` | 0.962 | x | stage4 | `harness/docs/claims-int8-stage4.txt:160` |
| `ratio_violet_4k` | 0.552 | x | stage1 | `harness/docs/claims-int8-stage1.txt:18` |
| `realesr_feat` | 64 | ch | stage5 | `harness/docs/claims-int8-stage5.txt:14` |
| `realesr_ratio` | 1.363 | x | stage5 | `harness/docs/claims-int8-stage5.txt:16` |
| `realesr_scale` | 4 | x | stage5 | `harness/docs/claims-int8-stage5.txt:15` |
| `realesr_share` | 0.00058 | % | stage5 | `harness/docs/claims-int8-stage5.txt:18` |
| `realesr_short` | 1.7 | ms | stage5 | `harness/docs/claims-int8-stage5.txt:30` |
| `render_engine_ms` | 46.47 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:187` |
| `render_frames` | 220 | x | stage4 | `harness/docs/claims-int8-stage4.txt:139` |
| `render_inference` | 70.7 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:141` |
| `render_ms` | 120.18 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:140` |
| `render_peak_epilogue` | 26 | levels | stage4 | `harness/docs/claims-int8-stage4.txt:223` |
| `render_share_epilogue` | 0.07024 | % | stage4 | `harness/docs/claims-int8-stage4.txt:222` |
| `render_share_prologue` | 0.07592 | % | stage4 | `harness/docs/claims-int8-stage4.txt:224` |
| `reuse_chainsaw` | 66.4 | % | example | `harness/docs/claims-example.txt:4` |
| `routing_margin` | 14.13 | pp | example | `harness/docs/claims-example.txt:18` |
| `save_per_frame_ms` | 8.45 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:266` |
| `seg_body_ms` | 43.851 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:77` |
| `seg_dequant_ms` | 2.32 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:78` |
| `seg_head_ms` | 5.272 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:75` |
| `seg_quant_ms` | 1.603 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:76` |
| `seg_residual_ms` | 1.708 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:81` |
| `seg_sum_ms` | 57.754 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:82` |
| `seg_tailconv_ms` | 2.258 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:79` |
| `seg_upsampler_ms` | 0.742 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:80` |
| `separate_head_ms` | 4.01 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:231` |
| `shortfall_idle` | -3.45 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:32` |
| `shortfall_nvenc` | -4.59 | ms | stage2 | `harness/docs/claims-int8-stage2.txt:34` |
| `sim_median` | 1.23 | levels | stage1 | `harness/docs/claims-int8-stage1.txt:14` |
| `sim_peak` | 8 | levels | stage1 | `harness/docs/claims-int8-stage1.txt:13` |
| `sim_violet_4k` | 0.0555 | % | stage1 | `harness/docs/claims-int8-stage1.txt:17` |
| `simulation_share` | 0 | % | example | `harness/docs/claims-example.txt:31` |
| `simulation_share` | 0 | % | stage1 | `harness/docs/claims-int8-stage1.txt:10` |
| `superseded_ratio` | 0.631 | x | stage4 | `harness/docs/claims-int8-stage4.txt:123` |
| `tail_nhwc_gain` | -0.2 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:83` |
| `tap_rolled_ms` | 2.198 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:155` |
| `tap_stages2_ms` | 2.497 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:156` |
| `tap_static_ms` | 2.919 | ms | stage4 | `harness/docs/claims-int8-stage4.txt:154` |
| `tiles_share` | 0.071 | % | example | `harness/docs/claims-example.txt:17` |
| `tops_bm128_capped` | 67.6 | TOPS | example | `harness/docs/claims-example.txt:11` |
| `tops_bm256` | 77.6 | TOPS | example | `harness/docs/claims-example.txt:12` |
| `triangle_bound` | 0.067 | % | example | `harness/docs/claims-example.txt:56` |
| `verdict_chainsaw` | 0 | x | stage4 | `harness/docs/claims-int8-stage4.txt:291` |
| `verdict_depth_worst` | 2 | x | stage4 | `harness/docs/claims-int8-stage4.txt:286` |
| `verdict_violet` | 0 | x | stage4 | `harness/docs/claims-int8-stage4.txt:292` |
| `violet_comp` | 0.17 | % | stage4 | `harness/docs/claims-int8-stage4.txt:13` |
| `violet_engine` | 0.057 | % | stage4 | `harness/docs/claims-int8-stage4.txt:12` |
| `wrapper_model` | 0 | x | stage4 | `harness/docs/claims-int8-stage4.txt:183` |
## Evidence, in full

Every file the verdicts and records rest on, embedded so this document stands alone. The claims lines (`@claim ...`) at the end of each one are the machine-checkable form; the prose above them is what the checks cannot verify.

<a id="evidence-docs-wsp-verified-engine-int8-txt"></a>

### `harness/docs/wsp/verified/engine-int8.txt`

#### The int8 Q/DQ engine (E9'): time reproduces, picture does not, and the record does not say why

Rebuilt 2026-11-17 on the same card, same TensorRT **10.12.0.36** the ledger used: compact 2x ->
ONNX (static 1920x1080) -> Q/DQ on the 16 body convs, per-channel int8 weights, head and tail in
fp16, calibrated on 8 named frames (sha256 in the manifest).

##### What reproduces

| aspect | ledger E9' | here |
|---|---|---|
| graph | 78 layers, **21 reformat**, **17 PointWiseV2** passes | 84 layers, **19-22 reformat**, **19 PointWise** |
| time | **68.11 ms** against 55.54 fp16, i.e. **0.82x** | **64.3-70.4 ms** in two sessions against 50-56 fp16, i.e. **0.79-0.86x** |
| mechanism | the per-channel PReLU cannot join an int8 epilogue, so every conv pays a boundary hop | same: zero myelin fusion, one pointwise pass per conv |

The mechanism is the part the claim is *about*, and it reproduces exactly - including the layer and
pass counts. The absolute time lands inside the ledger's own session spread.

##### What does not, and the size of the disagreement

Accuracy against the fp16 engine on a **held-out** frame (sha256 `ee4ad425`, never calibrated on):

| build | share above 8 levels | peak | PSNR |
|---|---|---|---|
| ledger E9' | **0.000%** | **4 levels** | **54.1 dB** |
| min/max calibration, fp32 head/tail | 0.010% | 18.87 | 44.36 dB |
| p99.99 calibration, fp32 head/tail | 0.291% | 65.11 | 47.72 dB |
| p99.99 calibration, fp16 flag on | 4.389% | 89.48 | 37.95 dB |

Three configurations of the *same* recipe differ by **three orders of magnitude** in the share, and
none reaches the recorded picture. The record names "per-tensor activation scales calibrated on 8
frames" and "head and tail in FP16"; it does not name the calibration percentile, the Q/DQ
placement, or the frame the comparison ran on. Those three unrecorded knobs move the answer more than
the quantisation itself does.

##### One thing the hardware settles

Asymmetric (uint8) activation quantisation - the PTQ default in most guides, and the natural choice
for a PReLU's one-sided output - is **not available here**: TRT rejects a non-zero zero-point with
`kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA`, a DLA-only switch. Every explicit-quant build on
this GPU is symmetric, which costs precision the ledger's 54.1 dB may or may not have had.

##### Verdict

The ledger's conclusion ("the quantised block boundary costs more than int8 saves on this
architecture") stands and is now reproduced with its own numbers. The *accuracy* half of the claim is
**not reproducible from the record**: same recipe, same TRT, same card, and the picture differs by
10-16 dB depending on knobs nobody wrote down.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/engine-int8.txt \
       --records harness/docs/wsp/verified/records/engine-int8.jsonl

@claim R001 70.85 ms "plan repeats [202f9e052701404e] n=3" measured
@claim R002 71.17 ms "plan repeats [202f9e052701404e] n=3" measured
@claim R003 0.0097 % "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R004 18.87 levels "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R005 44.36 dB "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R006 0.2908 % "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R007 65.11 levels "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R008 47.72 dB "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R009 4.3885 % "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R010 89.48 levels "pixels of one frame [a3a47682984c18d3] n=1" measured
@claim R011 37.95 dB "pixels of one frame [a3a47682984c18d3] n=1" measured

<a id="evidence-docs-research-ledger-md"></a>

### `harness/docs/RESEARCH-LEDGER.md`

#### Research ledger


<!-- BEGIN CURRENT STATE (generated by `python -m harness.cli db write-block`) -->
##### CURRENT STATE

Index: 93 records, 213 claims, 23 verdicts (17 current, 5 superseded, 1 refuted). The files are truth (`harness/docs/wsp/verdicts.yaml`, `harness/docs/wsp/verified/records/*.jsonl`); this section is a view. Everything below it is history and may be superseded - follow `superseded_by` in the verdict registry, never the position in this file.

###### engine
- **b3-prelu-forms** (synthesized 16-block graph, C=32 and C=64, 1920x1080 fp16, opt 5, 8 GiB workspace) - At C=32 the PReLU fuses (19 myelin, 23.51 ms); the a*x+(1-a)*relu rewrite fuses too and is 6% slower (25.06 ms); scalar-slope LeakyReLU breaks fusion (26.89 ms). The C=32 build is bimodal - 39.23 / 39.29 / 20.95 ms from one ONNX - while C=64 is stable at 46.80 ms (probe) and 48.4-51.4 ms (real checkpoint, three builds).
- **streams-no-overlap** (two independent TRT contexts, two streams, 1920x1080 fp16) - A second stream buys 1.001x of throughput on both a 87.9 ms split plan and a 54.0 ms fused one, with SM at 100% and 155-162 W of a 165 W cap: no overlap, so the strips proposal cannot pay its halo tax.
- **pyramid-budget** (whole 16-boundary body, 1080p and 4K, CPU arithmetic) - The L2 reuse buffer is 7.5 MB at 1080p and 15.0 MB at 4K, both inside 32 MiB, but the fp16 compute floor (33.1 / 132.2 ms) sits above the unabsorbed DRAM time (29.5 / 118.0 ms), so absorbing the traffic cannot raise the roofline: 1080p is 56.8 ms measured against a 33.1 ms floor = 1.72x whatever L2 does.

###### int8
- **int8-graph-launcher** (CUDA graph over 16 int8 launches, head and tail leading out of the graph) - The 16-launch chain is bit-exact under one CUDA graph capture, with the fifth contract (the capture runs on torch's stream) satisfied.
- **trt-int8-rate-79** (TRT plan c2x_int8_fp16heads.plan, 84 layers, 22 reformat, 19 pointwise) - The TRT int8 plan's convolutions run at a median 79.2 TOPS (min 76.2, max 119.7 over 18 conv layers), and the plan takes 67.67 ms/frame. This is the TRT two-pass build, NOT our kernel.
- **int8-body-hop** (body_kernel (our Triton int8 conv + per-channel PReLU + per-channel requant), 1920x1080, swept BLOCK_M and warps) - One hop is 1.82 ms at BLOCK_M 256 with 4 warps (84 TOPS), 2.38 ms at 128/4 and 4.06 at 64/4; BLOCK_M 512 with 4 warps collapses to 79 ms/hop (register pressure). The 16-hop chain with the per-layer tensors precomputed is 29.07-29.64 ms.
- **int8-split-calibration** (calibration coverage, per-title composition) - Calibration must be split per title: single-set calibration mis-states composition by 20x, and by 7.5x on the second title. This is a correctness condition, not a tuning knob.
- **int8-probe-f-total** (certification probe, per-title, full price on video) - The probe is built and measured at F_total 65.4 s, which sets break-even at 16.2 min of video (5.6 min once head and tail are fused).
- **int8-composition-d2** (composition at the depth the gate serves, two titles, 6 pairs each) - At depth 2 the passing composition reads 0.390% (Chainsaw) and 0.166% (Violet) against a 0.1% bar, so both titles refuse. At depth 1 the same filter reads 0.002% and 0.104%.
  - OPEN: Chainsaw is 195x its own depth-1 and 78x the fp16 gate's 0.005% at the same depth, while Violet is 1.6x / 1.73x - inside the measured range. One of the two numbers is probably not the population it is labelled with. The discriminating control needs no int8: fp16(t-2) against fp16(t) on the same pairs, at the gate's depth.
- **int8-window-measured** (our Triton int8 chain (6 of 16 hops on the three-accumulator kernel, 10 on the epilogue one, channels_last head/tail), 1920x1080, gate off, no encoder, calibration on 9 frames, accuracy on 4) - The window is 36.19 ms = 27.63 fps against a same-process fp16 TensorRT plan at 48.94 ms = 20.43 fps (1.35x this draw, 1.41x on an earlier one - the fp16 plan read 48.94-55.61 ms across today's runs). The 16-hop chain is 30.48 ms; head+quantise 1.60; tail+residual 5.78. Accuracy: worst of 4 frames 0.09169% against the 0.1% bar, all four held out of the calibration set.
  - OPEN: The peak is 55 levels against the proposed bar of 20, and every configuration measured landed between 29 and 55; the share bar passes, the peak bar does not. The sample is four frames (the bar's minimum is two) - the 13-19 frame spread the protocol asks for is not measured here.
- **int8-asymmetric** (our Triton kernel, asymmetric quantisation with the zero-point folded into the bias) - Asymmetric (uint8) quantisation is implemented in our kernel - zero-point folded into the bias - and measured: 0.102% at peak 32 against 0.058% at peak 37.
- **int8-triton-block** (our Triton kernel, one conv+per-channel-PReLU+requant block, 1920x1080) - 1.91 ms and 79-80 TOPS for one 64->64 3x3 block with the per-channel slope AND a per-channel output scale in the epilogue, bit-exact against the int32 reference.
- **int8-restart-under-gate** (shipped configuration, temporal gate on, tilesize=0) - With tilesize=0 the gate creates no tile_upscaler, the body branch is reachable, and int8 starts under the gate (int8Fit=int8). The old "--no_temporal_reuse is a precondition" is refuted.
- **int8-live-verdicts** (the app's own int8 fit probe (backend/src/utils/Int8Probe.py), three clips, 1920x1080, gate on) - The product accepts int8 where the material is easy (smoke clip: 0.003% share, peak 13 -> int8, body 37.9 ms) and refuses it on BOTH real titles on the DEPTH-2 composition: Violet d1 0.058% / d2 0.122%, Chainsaw d1 0.000% / d2 0.218%, against a 0.1% bar - the verdict is max(d1, d2) while the log prints d1 only, and the peak (37 and 11) is reported but does not decide.
  - OPEN: The blocker to fix is the depth-2 composition (0.122% / 0.218% against 0.1%), i.e. the gap between a frame and a reference two steps back, not the peak: bar_int8_peak (20) is also unreachable (29-55 across twelve configurations) but the product does not refuse on it.
- **int8-peak-unmet** (the measured int8 chain family (epilogue and three-accumulator kernels), 1920x1080, 4 held-out frames) - Twelve configurations were measured for the peak, and every one lands between 29 and 55 levels against bar_int8_peak = 20: margin 1.2 with per-tensor grids reads 29 levels at 0.124% share, while the hybrid that passes the share bar (0.09169%) reads 55. The lowest peak and the passing share are not available in the same configuration of this kernel family.
  - OPEN: bar_int8_peak (20) is still marked "PENDING OWNER DECISION" in harness/docs/claims-bars.json, and the product's own probe does not refuse on it - the refusals are compositional (d2). So this row is accepted debt with a measured range (29-55) rather than an active blocker; the active blocker is int8-composition-d2.

###### reuse
- **cone-budget** (16-layer body, changed square r=5..40 px, cone halo (17-L) per layer) - The cone is 2.54x less work than the cylinder at r=5 and 2.12x at r=10 (1-D control reproduces the ledger's 2.00x), but sixteen launches cost 0.08 ms while the work at r<=20 is below that: the prize is 0.08-0.27 ms instead of a 56.8 ms re-render, not a ratio on the engine.
  - OPEN: Phase 2 needs a region-level certificate, which does not exist: the gate's veto is frame-level, and the seams between a recomputed cone and a reused neighbourhood are the artifact class that made tile reuse read 0.287%.
- **e37-reachability** (app's temporal gate, Violet 20 s and Chainsaw Man 20 s, 1920x1080, chain certified per clip) - The premise of rve-hr-reference-cache ("the gate stores ONE reference") is stale: the gate keeps a chain and certified depth 4 on Violet (p90 1.85/4.66/6.98/9.61 against a limit of 12, monotonically worse) and depth 3 on Chainsaw (0.73/0.69/0.66, better by 0.07 units out of 12). On the clip where more references help, the gain is 0.6% of the certification budget, and the reuse there (100/481 = 20.8%) is limited by the threshold rather than by reachability.

###### superseded / refuted - do not cite
- e17-prelu-rewrite -> b3-prelu-forms  (TRT graph, C=64, 2026-11-15)
- int8-window -> int8-window-measured  (model window (engine, no encoder), one process, 1920x1080, session stage4)
- model-exactly-one-route -> int8-graph-launcher  (MODEL.md section 6, "distance to 2x")
- report8-no-temporal-reuse -> int8-restart-under-gate  (REPORT-int8-lane-2026-11-16 section 8, launch precondition)
- summary-lower-precision-closed -> int8-triton-block  (ledger summary table, row "lower precision (INT8/FP8)")
- trt-int8-two-pass -> int8-triton-block  (TRT explicit Q/DQ build, per-tensor activation scales, 2026-11-15)
<!-- END CURRENT STATE -->
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

##### The model itself: what is left

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
| lower precision (INT8/FP8) | **closed by measurement** | implicit and FP8 were dead ends (E9, E8); explicit Q/DQ is lossless (0.000% above 8, peak 4) and slower - 68.11 against 55.54 ms - because the per-channel PReLU cannot join an int8 epilogue (E9'). The ceiling with a fusible scalar slope is 40.24 ms (1.40x), and that form destroys the picture (-2.02 dB), so the only route left is a plugin with its own int8 convolution. **SUPERSEDED 2026-11-17 - see `int8-triton-block`**: the plugin was written and measures 1.91 ms / 79-80 TOPS; this row describes the TRT path |
| a cheaper published model | not applicable | the owner keeps his ESRGAN model set |

##### Engine / kernels

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
| E9' | INT8 with explicit Q/DQ, per-channel weights, calibrated on real frames | TRT PTQ docs | 2x of the arithmetic | **measured: accurate, slower, and the mechanism is the boundary, not the kernel.** Calibrated per-tensor activation scales from 8 real frames, per-channel int8 weights, 16 body convs quantised, head and tail in FP16: **0.000% of pixels above 8 levels against the FP16 engine, peak 4, PSNR 54.1 dB** - the format is essentially lossless here. Time **68.11 ms against 55.54 (0.82x)**. Per-layer (`trt.IProfiler`): the 16 int8 convs are `sm80_xmma_..._interleaved_i8f16_i8i32_f32_nchw_vect_c_...` - the **interleaved NC/32HWC32 tensor-core layout, not plain NCHW** - at 1.56-1.88 ms each, i.e. 152.9 GFLOP / 1.68 ms = **91 TOPS against 52 TFLOP/s for the fused fp16 block**; the PReLU runs as **17 separate `PointWiseV2` passes** of 1.71-2.37 ms (31.2 ms, 46% of the engine) because a per-channel slope (1088 values, -1.39..1.70, per-channel Cv > 1) does not fit an int8 epilogue. In fp16 those 16 blocks are one fused myelin kernel each (`__myl_CorrMulLtSele`, 2.93 ms). The engine holds **21 reformat layers of 78**, worth **5.6 ms in total** (8% of the engine) against **32.7 ms for the 17 PReLU passes** (46%) - the "60 reformats" in earlier revisions was 78 minus 18 convs. Second build, quantising the conv output too so the PReLU reads int8: **78.36 ms**, 36 reformats, 16 `DQout` passes at 1.64 ms - the boundary pass is charged per hop, so adding a hop makes it worse | **rejected as quantised** (measured, with the mechanism): the per-channel PReLU forces one int8->fp16->int8 hop per conv, and each hop costs about what int8 saves on the conv. Scope: **TRT explicit Q/DQ build**. **SUPERSEDED 2026-11-17 by `int8-triton-block`** - the hop is a property of TRT's builder, not of int8 or of PReLU |
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
| E17 | rewrite PReLU as `a*x + (1-a)*Relu(x)` | TRT supported-op patterns | maybe changes tactic eligibility | **measured, as the unlock E9' was missing - and it is not one.** The rewrite does not remove the boundary: the `a*x` term still needs the pre-ReLU frame, and the `(1-a)*Relu(x)` term gives a second full-frame pass, so it trades one 1.8 ms PReLU pass for two. What would remove the boundary is a fused int8 conv+PReLU epilogue, and this build has none for a per-channel slope (measured: the int8 conv reports `HasReLU: 0, Activation: NONE` and the PReLU appears as its own `PointWiseV2` layer). The earlier guess that it was "cheap and near zero" was right about the cost and wrong about the mechanism | **SUPERSEDED 2026-11-17 by `b3-prelu-forms`** - the conclusion holds (the rewrite buys nothing) but the mechanism does not: at C=32 the PReLU fuses on its own and the rewrite is 6% slower. Scope: TRT graph at C=64 |

##### Precision / colour (10-bit)

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| C1 | pass the source's colour matrix/range to ffmpeg | ffmpeg/swscale docs | removes chroma shift | \|dU\| 6.64 -> 0.2 codes; 8-bit path 40.25 -> 44.06 dB | **adopted** |
| C2 | keep 10 bits end to end (pack YUV420p10 on the GPU) | waifu2x postproc shader uses a GPU colour stage | parity with ffmpeg | 49.69/55.34/54.84 dB vs the ffmpeg path | **adopted** |
| C3 | pipe `rgb48le` (the app's old "16-bit" path) | - | better than 8-bit | **worse**: 39.0 dB and a -5.9 level shift, because of the missing matrix | **rejected** |
| C4 | pipe P010 for NVENC (avoids a 1.54 ms/frame repack) | ffmpeg encoder caps | ~12% of the NVENC path | **two separate faults, both found and fixed.** (a) P010 was not in the "already YUV" branch, so it got `scale=in_range=pc:out_color_matrix=...`: a full-range reinterpretation of a limited-range stream plus a second matrix. (b) worse, `_pipe_input_pixel_format()` derived the format from the *encoder* tag ("NVENC wants P010") while the packer emits planar `yuv420p10le` unless `RVE_P010` is set, so **every 10-bit source rendered with NVENC fed planar samples labelled P010 and came out at 12.68 dB**. With the command following the packer and P010 opt-in: 52.01 dB against a libx265 render at 10-bit output, 46.11 dB at 8-bit output. A format disagreement cannot fail loudly here, both are 16-bit words at 3 bytes/pixel - which is why it survived a session. (c) the colour tags were declared on the *output* side, which makes ffmpeg convert every frame to the declared space: 79.0 ms/frame against 16.6 ms/frame with the tags on the input, and a chroma shift of 5.3 levels in U because already-bt709 samples were converted from bt601. Same tags in the container either way. See the investigation for the bisection | **fixed** (measured); P010 itself is **off, measured**: against the corrected planar path a full render is byte-identical in picture (0.000% above 8 levels) and **0.5 ms/frame slower** (28.16 against 27.65 ms on 220 frames of 10-bit), because the repack it was meant to avoid no longer exists |
| C5 | convert YUV->RGB on the GPU on the input side | same as C2 | halves the input pipe | works, colour matches; costs ~1.2 ms/frame | **adopted** |

##### Temporal / dedup

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

##### Pipeline / IO

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

##### Encoder

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| N1 | use NVENC instead of `libx265 medium` | our measurement | removes the bottleneck | **11.18 -> 4.27 s** on the clip (encode alone 9.6 -> 1.2 s) | **adopted** (advice) |
| N2 | `--frame-threads 4..6` for x265 | x265 threading docs | -4% | measured -4% | open (only if x265 is used) |
| N3 | anime flag set instead of `--tune animation` (`aq-mode 3`, `psy-rd 1.0`, `bframes 8`, `deblock -1:-1`) | JET guide, kokomins | same time, less banding | `--tune animation` measured +8% time, -6.5% bits and is wrong for flat gradients | open |
| N4 | `av1_nvenc` | IEEE Access 2025 BD-rate table | quality/size, not speed | -34% BD-rate vs hevc_nvenc | open (quality, not time) |
| N5 | p010 for NVENC to skip the repack | ffmpeg caps | 1.54 ms/frame | see C4 | blocked on C4 |

##### Current per-frame budget (measured by the built-in profiler)

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

##### Backlog, ranked by expected gain on our pipeline

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
   `projects/rve/docs/backlog/queue.yaml` `rve-winograd-tactics-2026-11-15`.
   (E17 used to be listed here as a micro-optimisation; it is now part of E9' above, where it turned
   out not to unlock anything.)
4. **P8 / P9 / N2 / N3 / N4** - quality-of-life and encoder-side, only if x265 or per-plane tile
   writes are kept.

##### Which metric answers "is it visible" (2026-11-15)

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

##### What the reuse artifact actually is (2026-11-15)

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

##### Three real Blu-rays, and the rule that did not transfer (2026-11-15)

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

##### The sample bar cannot be calibrated yet (measured, and one earlier claim withdrawn)

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

##### E22: is the raw builder faster? Not established - the drift is bigger than the effect

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

##### E24: the tail convolution does not fuse, and padding it makes things worse (measured)

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

##### E23: the 71% regression was a graph break, and the fix is one operation

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

##### E25: certifying the depth on the acceptance statistic fails its own verification

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

##### The head padding across the whole collection (78 models, 2026-11-15)

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

##### Is `tiling=none` faster than what ships? No - and the probe that said so measured a different engine

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

##### Provenance of the rows in this file

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

##### Grain sweep, re-measured on the statistic that ships (2026-11-15)

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

##### Integer reuse counts at a fixed chain (why "70.2% at three thresholds" was unreadable)

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

##### E26: Winograd is not reachable through cuDNN on this card, and the margin is 20%

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

##### The distance to 2x, in measured numbers (2026-11-15)

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

##### What the card can actually do, measured (2026-11-15)

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

##### The correction is 0.9 levels: what that licenses (2026-11-15)

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
`harness/scripts/probe_channel_width.py` does.

##### E30a: a channel cut is worth 1.23x, not 1.78x - measured

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

##### The int8 block with a per-channel epilogue: 1.91 ms and 80 TOPS, measured, not projected

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

**The peak is distributed clipping, not the format, and one number proves it.** Same-title held-out
pair, head pinned to its domain bound, everything else as calibrated:

| knob | share above 8 | peak | median |
|---|---|---|---|
| calibrated maxima (margin 1.0) | 0.066% | **40** | 1.25 |
| margin 1.5 on **body.1 only** | 0.071% | 42 | 1.23 |
| margin 2.0 on body.1 only | 0.073% | 43 | 1.25 |
| fp16 boundary on the last 2 layers | 0.062% | 36 | 1.00 |
| fp16 boundary on the last 8 layers | 0.047% | 35 | 0.81 |
| **margin 1.5 on every layer** | **0.019%** | **21** | 1.74 |
| **margin 2.0 on every layer** | 0.030% | **21** | 1.99 |
| margin 4.0 on every layer | 5.351% | 27 | 3.77 |

A per-layer margin does nothing, the tail's precision buys 5 levels, and a margin on *everything* takes
the peak from 40 to 21 with the share falling to 0.019%. So the 40 was not one bad layer and not the int8
hop: it was a little clipping spread across many layers, each small, adding up at the outliers. And the
curve turns over - at 4x the coarser step everywhere costs more than the clipping it removes (5.351%).

The operating point is therefore a global margin of 1.5-2x on calibrated scales. At 1.5 the share is
0.019% and the peak 21; at 2.0 the share is 0.030% and the peak 21, so **1.5 strictly dominates**: same
peak, better share, and the curve is steep enough on the other side that 4.0 costs 5.351%.

**The residual 21 is localized, and it is not what a step-size story predicts.** Three pixels above 20
levels, 1537 above 8, on a frame of 8.3 Mpx. The worst one sits at (449, 2349) channel 2, where the
reference reads 0.048 and the simulation reads -0.034: a dark, low-contrast patch (neighbourhood range
-0.021..0.199) where an offset of 0.08 in 0..1 units is 21 levels. And the peak is **invariant between
margin 1.5 and 2.0** - 21 in both - although the rounding sigma grows 33% between those points. A peak
that does not move when the noise does is not the noise: it is a residual offset of the quantised stack
in the darkest, flattest places, which is where a step of ~0.43 levels across 16 layers shows up as a
level shift rather than as noise.

**The bar this is judged against was being imported, not applied.** The shipping tile path is documented
at 0.071% with **peak 23** and called passing; the acceptance rule's peak guidance is not a gate the
project applies. Against the same standard, int8 at margin 1.5 is 0.019% / peak 21 - better on both axes
than an option already in the product.

**Asymmetric scales are worth about 0.7 of a bit, measured.** Per-PReLU `(max-min)/2 / max(|.|)` has a
median of **0.616** across 17 layers (range 0.500-0.727, six layers below 0.6), so a 256-code asymmetric
range buys a ~1.6x finer step on the average channel - real, less than the 1.85x a one-sided channel
would give, and enough to pay for the margin rather than replacing it.

**The spread sample kills it, and it is the third time a single pair was optimistic.** Calibrated on two
frames of one title (1 s and 5 s) and evaluated on five unseen frames including another title, margin 1.5:

| held-out frame | share above 8 | peak | median |
|---|---|---|---|
| Chainsaw 1 s | 0.094% | 17 | 2.43 |
| Chainsaw 3 s | 0.093% | 14 | 2.43 |
| Chainsaw 7 s | 0.126% | 16 | 2.46 |
| Violet 1 s | 0.078% | 17 | 2.43 |
| Violet 5 s | **0.174%** | **35** | 2.49 |
| Violet 8 s | **0.162%** | 20 | 2.43 |

Three of six frames exceed the 0.1% share bar, and the same-pair result that motivated margin 1.5
(0.019%) reappears nowhere. Worse, the optimum moved: on the Violet 5 s frame the margin is not monotone
either -

| margin | share | peak |
|---|---|---|
| 1.0 | 0.078% | 44 |
| 1.25 | **0.063%** | 38 |
| 1.5 | 0.174% | 35 |
| 2.0 | 1.768% | 36 |

so 1.25 is the best point on that frame, 1.5 makes the share *worse*, and the peak sits between 35 and 44
whatever the margin is. That peak does not respond to step size at all, which is the same invariance the
localization found - but on Violet it is 2x the bar rather than 1 level over it.

**Recalibrating in-domain answers where that came from, and the answer is "both".** Same Violet frames,
calibrated on Violet instead of Chainsaw:

| held-out | cross-title calib (1.25) | in-domain calib (1.25) |
|---|---|---|
| Violet 1 s | 0.078% / peak 17 | **0.006% / peak 12** |
| Violet 5 s | 0.174% / peak 35 (at 1.5) | **0.058% / peak 37** |
| Violet 8 s | 0.162% / peak 20 | **0.059% / peak 23** |

So the share is a calibration-coverage effect and it largely goes away: at margin 1.25 in-domain the
share is 0.058%, **better than the 0.071% the shipping tile path documents**. The peak on the 5 s frame
is not a coverage effect - it sits at 35-37 across margins 1.0-1.25 and only falls to 28 at 1.5, where
the share climbs to 0.141%.

**And that peak is on a bright edge, not a dark flat patch.** Localized at (1228, 1027) channel 2:
reference 0.714, simulation 0.858 - +37 levels on a bright pixel in a neighbourhood of 0.476..0.911.
Eight pixels above 20, 4771 above 8. The earlier localization found the opposite corner of the range (a
dark patch at 0.048), so the two are not one mechanism: one is a level shift where the signal is small,
the other is a bright-region excursion consistent with a top-end clipping event that the calibration set
missed.

**The ill-conditioning ratio is spread, not concentrated, so targeted migration is not the fix.** For
every conv, per output channel: `sum_k |W[c,k]| * max|x_k| / max|out_c|` - how much weight-quantisation
error that channel amplifies relative to what it produces.

| layer | median ratio | worst channel |
|---|---|---|
| 0 (head) | 4.4 | 59.3 |
| 30 | 18.3 | 43.5 |
| 28 | 18.4 | 42.3 |
| 26 | 19.3 | 39.3 |
| **all 18 layers** | **17.9** | 37-59, only 1 above 50 |

A handful of pathological channels would justify SmoothQuant-style migration targeted at them. The ratio
is instead flat across the stack - median ~18 everywhere, worst channels all in the same 40-60 band - so
the error is spread over all channels and the remaining levers are the ones that act on every channel:
asymmetric scales (~0.7 of a bit, measured) and the fp16 boundary on the early layers (untested, and the
tail experiment showed the last layers are not where it enters).

**Three cheap checks localize the peak, and the null test moved the blame.**

| test | share | peak | median |
|---|---|---|---|
| own-calibration clip rate at margin 1.25 | **0.00% on every layer** | - | - |
| fp32 reference against the fp16 one, whole frame | 0.000% | **0** | 0.00 |
| weights quantised, activations NOT (all 16 boundaries fp16) | 0.005% | **28** | 0.75 |
| weights and activations quantised, margin 1.25 | 0.058% | 37 | 1.93 |

So: clipping is **excluded** in-domain (zero on every layer), and the reference is **excluded** - the fp32
and fp16 models agree to the level (max 0 over the whole frame), so the 37 is not the measuring stick.
What is left is the quantisation, and the null test splits it: **weight quantisation alone accounts for 28
of the 37 levels**; adding activation quantisation takes it to 37 and the median from 0.75 to 1.93.

**And the attribution flips depending on which metric is read, which decides the order.**

| configuration | share above 8 | peak | median |
|---|---|---|---|
| weights only | 0.005% | 28 | 0.75 |
| weights + activations | 0.058% | 37 | 1.93 |
| **activation contribution** | **+0.053 pp = 91% of the share** | +9 = 24% of the peak | x2.6 |

By the peak the weights own 76%; by the **share** - the statistic the project actually applies, the one
the shipping opt-in passes at peak 23 - the **activations own 91%**. So asymmetric activation scales are
the *first* lever, not the remainder: they are free at runtime (the zero point folds into the bias), act
on the dominant term of the binding metric, and are exact on the head's input where the clamp fixes the
range to [0, 1].

Two levers are now measured rather than argued:

- **Per-tap weight scales: 2.47x finer at the median** (range 1.66-3.72 over 18 convs). One scale per
  output channel currently covers all 576 entries of that channel, and the taps differ by more than 2x, so
  the small taps lose bits. This is exact rather than approximate *in our own kernel*: dequantise inside
  the tap loop, accumulate per tap in int32 over 64 channels (64*127*127 ~ 1.03M, no overflow) and sum the
  taps in fp32. Cost is eight extra int32->fp32 conversions and FMAs per output element, ~0.1 ms on a
  1.91 ms block, and it hits the 28 levels the weights own. TensorRT cannot express this - it may only
  dequantise after the whole int32 accumulation - so the kernel probe is what opened it.
- **SmoothQuant migration is a trade, not a free win, on this material**: the correlation between an input
  channel's activation range and its weight magnitude is **+0.45** at the median across 17 body convs.
  Correlated means shrinking the activation range inflates the weights, so alpha sets how much peak is
  paid for share.

**Per-tap weight scales are the first lever that improves both metrics at once.** Same held-out frame,
same calibration, margin 1.25, symmetric activations:

| weight scales | share above 8 | peak | median |
|---|---|---|---|
| per output channel (what TensorRT can do) | 0.058% | 37 | 1.93 |
| **per (output channel, tap)** | **0.021%** | **32** | 1.99 |

**The best combination is per-tap WITHOUT the asymmetric grid and without a margin.** Violet 5 s, two
Violet frames for calibration:

| configuration | share above 8 | peak | median |
|---|---|---|---|
| per-channel, symmetric, margin 1.25 | 0.058% | 37 | 1.93 |
| per-tap, symmetric, margin 1.0 | **0.008%** | 33 | 1.62 |
| per-tap, asymmetric, margin 1.0 | 0.019% | 37 | 1.99 |
| per-tap, asymmetric, margin 1.25 | 0.049% | 33 | 2.46 |
| per-tap, asymmetric, margin 1.5 | 0.246% | **28** | 2.99 |

So the margin and the asymmetric grid both cost share for peak, and neither gets the peak below 28; the
headroom is not needed once the weights are per-tap. At 0.008% the share is 12x inside the bar and 9x
better than the tile path's 0.071%, with the median at 1.62 levels - and the peak sits at 33 whatever is
done to it.

**And it costs 2x the block time, not the 0.1 ms the arithmetic predicted.** The same kernel with the
dequant inside the tap loop:

| config | single scale | per-tap |
|---|---|---|
| BM 128, warps 4 | 2.355 ms (64.9 TOPS) | **3.117 ms (49.0)** |
| BM 256, warps 4 | 1.970 ms (77.6 TOPS) | **4.093 ms (37.4)** |
| BM 128, warps 2 | 2.358 ms | 4.419 ms |

The prediction counted eight extra int32->fp32 conversions and FMAs per output element. What actually
changed is the code path: a float accumulator replaces the int32 one, so every tap materialises a
converted tile and the MMA sequence stops being the cheap thing it was. A 1.5-2x penalty buys 7x on the
share, and at 3.1-4.1 ms per block the body would cost more than the fp16 engine it is meant to beat.

So the lever is real for accuracy and unaffordable as written. **And the middle is closed too: grouping
the taps does not help.**

| variant | BM 128, warps 4 | BM 256, warps 4 |
|---|---|---|
| one scale per channel | 2.355 ms | 1.970 ms |
| three groups (one per kernel row) | **3.050 ms** | 3.478 ms |
| nine taps | 3.117 ms | 4.093 ms |

Three conversions instead of nine costs the same as nine, and the slowdown locates the price better than
the word "conversions" does: 1.58-1.77x at BM 256 and 1.30x at BM 128, against the measured **2.06x**
separating int8 from fp16 tensor throughput (147.4 TOPS against 71.6 TFLOP/s). So a float accumulator
inside the loop does not merely add work - it **takes the dot out of int8 mode**, which is the only
reason the kernel was fast. The per-tap build at 3.1-4.1 ms is an fp16 kernel with extra steps, and its
likeness to the 2.93 ms fp16 block is not a coincidence.

The one untried variant - several int32 accumulators, converted only after the loop - keeps int8 MMA but
costs 128 registers per accumulator per thread at BM 256 x BN 64; three of them is 384 against a 255
limit, so it spills, and the tile would have to shrink to BM 128 which is already 20% slower. ~2.2-2.4 ms
for a statistic that already passes with 1.7x margin is a bad trade, and it was not measured on purpose.

**Where the lane actually stands, stated against the right alternative.** int8 does not replace the tile
path; it goes *under the gate*. The shipped default is gate-only: **0.009%, peak 14**. int8 changes the
engine beneath it, and the composition 1.42x of engine time / 1.24x of wall (16 Triton blocks plus the
TRT head and tail) at a share better than the shipped opt-in's 0.071%.

Two consequences of that composition are unmeasured:

- **The errors add.** The output is `HR_int8(t-k)` against an `HR_fp16(t)` reference, so by the triangle
  inequality the bound is 0.058% + 0.009% = 0.067% when the pixel sets are disjoint - inside the bar, as a
  bound on one frame, not as a spread measurement.
- **The tuner's certification identity breaks.** Today the tuner measures acceptance between two frames
  of *its own* engine, and that is legitimate because the engine is the reference. With int8 it would
  measure int8 against int8 while the bar judges int8 against fp16 - the same instrument gap that this
  investigation closed at the cost of three false "quality walls", returning through a change of precision.
  The cheap fix is to run both engines on the 16 calibration frames and certify against fp16 there (16 x
  (56 + 40) ms ~ 1.5 s per video), leaving the other frames on int8 alone.

  **Measured on an adjacent pair** (Chainsaw 3.000 s and 3.042 s, calibration on two other frames):

  | what is compared | share above 8 | peak | median |
  |---|---|---|---|
  | bar: int8(t) against fp16(t) - engine error alone | 0.000% | 10 | 1.65 |
  | tuner: int8(t) against int8(t-1) - what it would see | **0.024%** | 17 | 2.21 |
  | bar: int8(t-1) against fp16(t) - reuse plus engine | 0.006% | 13 | 1.88 |
  | fp16 against fp16, no int8 at all | 0.000% | 10 | 1.03 |

  So the gap is 4x on anime - and **Violet inverts the conclusion**:

  | comparison (Violet, adjacent pair 5.000 / 5.042 s) | share | peak | median |
  |---|---|---|---|
  | bar: int8(t) against fp16(t) - engine alone | 0.028% | 37 | 1.62 |
  | tuner: int8(t) against int8(t-1) | **0.242%** | 19 | 2.61 |
  | bar: int8(t-1) against fp16(t) - reuse plus engine | **0.146%** | 34 | 2.24 |
  | fp16 against fp16, no int8 | 0.009% | 14 | 1.59 |

  On the heavy material the composition **crosses the bar** (0.146% against 0.1%) while the fp16 pipeline
  reads 0.009% on the same pair. So int8 does not merely tighten the margin there - it removes reuse the
  shipped engine is allowed: under fp16 the gate certifies the pair, under int8 both the tuner (0.242%) and
  the bar (0.146%) refuse it. The cost is speed on exactly the material that has the least of it, and it is
  the first place in this lane where int8 is worse than what ships on a statistic that matters.

  Two mechanisms, and neither is the one I first wrote:
  - the composition is **not** additive in proportion; two sub-threshold errors cross the threshold
    together (anime: 0.000% + 0.000% -> 0.006%), because what sums is the error *distribution* below the
    bar, not the sets of failing pixels. The triangle-inequality bound of 0.067% was never a bound on a
    share, and it also mixed Violet's numbers with anime's to get there;
  - in the body of the distribution the errors add **in quadrature** (median sqrt(1.03^2 + 1.65^2) = 1.94
    against 1.88 measured), but the tail is not Gaussian: at sigma 2.88 a Gaussian would put 0.55% above
    8 levels where 0.006% is measured, 90x less. Quantisation error is bounded at half a step, and a
    bounded distribution's tail is exponentially sensitive to where its edge sits - which is why the share
    moved 11.6x while the median moved 2.6x.

  And the blind spot is the point, not the conservatism: on a **static** pair int8(t) equals int8(t-1)
  exactly, so the tuner certifies zero while the bar measures the engine's own error. The tuner is blind to
  the engine's error precisely on the frames it certifies. What keeps that safe is not the sign of the gap
  but that the engine's own error sits far under the bar everywhere measured (0.000-0.005% anime, 0.028%
  Violet). The fix is therefore a **floor**, not a comparison: certify the engine against fp16 on the
  calibration frames and treat that as the lower bound of what the bar will see.

**But the reuse loss costs more than the engine wins.** Violet ships at 142/219 reused (64.8%) and 6.90 s,
with 51.4 ms of engine on a drawn frame, so the engine averages 18.1 ms and everything else is 13.4 ms.
At int8's 36.2 ms per drawn frame:

| reuse under int8 | per frame | 219 frames | against 6.90 s |
|---|---|---|---|
| 64.8% (kept) | 26.1 ms | 5.72 s | 1.21x faster |
| **50%** | 31.5 ms | 6.90 s | **break-even** |
| 0% (gate refuses) | 49.6 ms | 10.86 s | **1.57x slower** |

So int8 can afford to lose **15 points of reuse, not more** - and Violet already sits at 0.096% against a
0.1% bar, i.e. 4% of margin, with the composition measuring 5.2x the engine's own error there. If that pair
is representative of the title, the gate refuses and int8 is 1.57x *slower* than what ships on the material
that is already the slowest (1.93x against anime's 3.3x). The first place in this lane where int8 loses on
the metric that matters is therefore time, not the share.

**But Violet is not uniformly over the bar, which turns the verdict from title-level into per-video.** A
second adjacent pair per title, each with in-domain calibration:

| pair | engine | composition | fp16 only |
|---|---|---|---|
| Chainsaw 3.000 s | 0.000% / 10 | **0.006% / 13** | 0.000% / 10 |
| Chainsaw 1.000 s | 0.001% / 12 | **0.006% / 13** | 0.000% / 11 |
| Violet 1.000 s | 0.001% / 12 | **0.062% / 16** | 0.006% / 13 |
| Violet 5.000 s | 0.028% / 37 | **0.146% / 34** | 0.009% / 14 |

So one pair in four crosses the bar, and it is the one whose own frame is the engine's worst (0.028% and
peak 37 in isolation). The other three pass with 1.6-17x margin. That is a probe's job description rather
than a verdict on a title: the decision is per video (or per shot), the instrument is 16 frames on both
engines at the start, and the fallback is the fp16 engine that already works.

**A floor is not sufficient, and Violet is the proof.** The engine's own error there is 0.028% while the
composition is 0.146% - 5.2x - so certifying the engine against fp16 would have approved a depth that
fails the bar. The correction is that the composition does not decompose: certify
`|int8(t-k) - fp16(t)|` for every reachable depth on the calibration frames, with both engines. Same cost
(16 x (56 + 40) ms ~ 1.5 s per video), different justification. The static-pair blind spot then closes by
construction - on a static pair `int8(t-k) == int8(t)` and the composition degenerates to the engine's own
error, which is exactly what the bar sees - and the engine error survives as a cheap pre-filter, like the
noise floor, rather than as an estimate.

**And "our own executor" is transport for int8, not a lever of its own.** Our int8 kernel reaches 77.6 TOPS
of a 147.4 peak - **53%** - while TensorRT's fp16 block reaches 52.2 TFLOP/s of a 71.6 peak - **73%**. The
same kernel structure ported to fp16 would give 0.53 x 71.6 = 38 TFLOP/s, i.e. 4.0 ms per block against
TRT's 2.93, a 1.38x regression. The win came from doubling the rate, not from a better kernel. An executor
with an int8 body is 1.69x engine / 1.36x wall; an executor with an fp16 body is at best parity and
probably slower. Removing the build path as a hidden variable is hygiene worth having, but it is not 0.12x
of wall and it does not justify a kernel project on its own.

**And architecture weighs more than the remaining accuracy.** At 1.97 ms per block, keeping the TRT head
and tail gives 39.6 ms = 1.42x engine / 1.24x wall; doing head and tail as Triton memory-bound kernels
(145 MB and 195 MB of traffic, floors 0.50 and 0.68 ms against the current 3.90 and 3.35) gives 33.2 ms =
1.69x / 1.36x. That 0.12x of wall is larger than everything left in the accuracy lane.

**The spread, on the final configuration, inverts the margin recommendation and beats the shipped
default on anime.** Per-channel weights, symmetric activations, calibrated per title:

| frame | margin 1.0 | margin 1.25 |
|---|---|---|
| Chainsaw 3 s | **0.000% / peak 10** | 0.005% / 13 |
| Chainsaw 7 s | **0.005% / 13** | 0.018% / 16 |
| Violet 5 s | **0.029% / 35** | 0.058% / 37 |

Margin 1.0 dominates on both titles - better share *and* lower peak - so the headroom was never needed
once the head's input is pinned to its domain bound. And on anime, the material every shipped number was
measured on, int8 reads **0.000-0.005% with peaks 10-13 against the gate-only default's 0.009% and peak
14**: better on both statistics, not merely inside the bar.

Violet remains the heavy case at 0.029% / peak 35, and it is the material whose gate-only defect is
already 0.096% against anime's 0.009%. So the honest summary of the picture is a range, not a number:
0.000-0.029% of pixels and peaks 10-35 depending on the title, against bars of 0.1% and ~20.

Finally, there is no tradable point: inside the bar *and* faster than fp16 there is exactly one
configuration - per-channel weights, symmetric activations, 1.97 ms, 0.058%, peak 37. Everything that
lowers the peak is slower than the fp16 engine it replaces. So the decision is binary, and the peak is a
constant rather than a dial.

2.8x better on the share, five levels off the peak, and the median flat - so the gain is not a
redistribution between the two statistics, it is a genuine reduction. This is exact rather than
approximate in our own kernel: dequantise inside the tap loop, accumulate each tap in int32 over 64
channels (64*127*127 ~ 1.03M, no overflow), sum the taps in fp32. Cost measured separately at ~0.1 ms on a
1.91 ms block, and TensorRT cannot express it because its dequant must follow the whole accumulation.

**Asymmetric activation scales were expected to take 91% of the binding metric and instead trade one
against the other.** Same held-out frame, same calibration (two Violet frames), margin 1.25:

| scales | share above 8 | peak | median |
|---|---|---|---|
| symmetric (per-tensor / per-channel max) | **0.058%** | 37 | 1.93 |
| **asymmetric (256 codes over [min, max])** | 0.102% | **32** | 2.68 |

The peak improves as predicted and the share gets worse, so it does not dominate - it is a trade, and the
trade runs against the metric the project applies. The median moving from 1.93 to 2.68 is the part that
does not fit the theory: over a one-sided channel a 256-code asymmetric grid is strictly finer than a
127-code symmetric one, so the median should have improved with everything else. Either the
implementation has a wrinkle to find (zero-point handling on the two-sided channels, where [lo, hi] with
both signs gives a step of (hi-lo)/255 that equals max/127.5 - no gain - while the margin widens the span
it is fitted to), or the activation distributions are two-sided enough that the theoretical gain never
materialises. **Unresolved, and the number is recorded as measured rather than as expected.**

The conditioning ratio being flat said "no bad channels to target", not "nothing to migrate": migration
acts on the input axis, where no scaling exists at all today.

A note for whoever repeats this: `margin` multiplies the calibration maxima, which **coarsens** the grid -
it buys clipping headroom and pays in rounding. There is no "identity quantisation" direction in that
knob, so the null test has to be run by disabling quantisation (`--fp16-tail 16`), not by a large margin
(a 1000x margin overflows fp16 in the requant divide and returns NaN, which is how the first attempt
failed).

Status, honestly: **with per-video calibration and margin 1.25 the int8 picture passes the statistic the
project actually applies** - 0.058% against 0.1%, and better than the shipped opt-in's 0.071% - while
reading peak 37 against that option's 23. Two frames of one title are still a narrow calibration set;
the peak is the remaining open question and it is now a named bright-pixel event rather than an
invariant. Chainsaw reads 0.093-0.126% at peaks 14-17; Violet reads 0.078-0.174% at peaks
17-35 depending on the frame, with no margin recovering the peak. The speed is real (1.91 ms, 79-80
TOPS); the picture is not established on a spread sample, and per this ledger's own rule a configuration
that looks good on one pair is not measured.

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

##### What bounds the routing lane, before it is measured further

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

##### Spatial routing survives the oracle: 1.6-1.95x on a mask a router can actually compute

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


##### E30b closed: 32 channels cannot hold the stack's features, and that is a bound, not a method failure

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


##### E30b first look: a naive channel cut is 16x over the bar, and the error does not shrink with width

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

##### C=32 fuses and runs at 22.4-24.9 ms: the "fusion break" was the builder (2026-11-15)

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

##### Winograd exists on this card and TRT selects it - at C=32, not at C=64 (2026-11-15)

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
that fuses, which is exactly what `harness/scripts/force_tactic.py` scans by forcing one tactic index at a time
and reading the resulting layer names.

##### The counters are unblocked, and the int8 kernel is not at any limit (2026-11-15)

`RmProfilingAdminOnly` is 0 after the owner's modprobe line and reboot, so `ncu` runs. First measurement
on the int8 block kernel (BM 256, warps 4), per launch at 1920x1080:

| counter | value |
|---|---|
| duration | 1.88 ms |
| DRAM read | 141.6 MB |
| DRAM write | 170.1 MB |
| DRAM throughput | **59.4% of peak** |
| tensor pipe active | **51.2%** |
| L2 sector hit rate | 90.0% |

So the block sits at **59% of DRAM and 51% of the tensor pipe at the same time** - neither resource is
saturated. Traffic is 312 MB against a 265 MB floor for one 64-channel int8 plane in and out (1.18x, which
is the halo and some imperfect reuse), and at 288 GB/s the floor is 0.92 ms against 1.88 measured. The
tensor pipe at 51% means the MMA is idle half the time, so this is not a compute limit either, and the
`num_stages` sweep confirms it: 1, 3 and 5 stages all land within 1% (1.855-1.871 ms), i.e. there is nothing
to overlap.

**And the limiter is registers, not memory access patterns.** One run with the occupancy and stall
sections says it outright:

| counter | value |
|---|---|
| **registers per thread** | **255** (the architectural maximum) |
| block limit by registers | **2 blocks per SM** |
| **achieved occupancy** | **16.5%** (7.92 active warps per SM) |
| warp cycles per issued instruction | 6.23 |
| compute (SM) throughput | 69.2% |
| memory throughput | 69.2% / DRAM 62.6% |

So the kernel runs at 16.5% occupancy because 255 registers cap it at two blocks per SM, and with eight
warps in flight neither pipe can be fed past ~69%. The earlier reading of "59% DRAM, 51% tensor, so at
neither limit" was right about the symptom and wrong about the cause: both are held down by occupancy.

The traffic figure corrects too: the excess is 47 MB (312 against 265), worth 0.16 ms of the 0.73 ms gap,
and almost all of it is on the **write** side (170.1 MB written against 132.7 produced, 1.28x) while the
read side is 141.6 against 132.7 (1.07x, i.e. the halo is fully absorbed by L2 at a 90% hit rate). So
eliminating every byte of amplification buys 1.88 -> 1.72 ms, not 1.3.

The sweep already pointed at this and I read it as "128 values per thread is good": the configs with 128
accumulator values per thread (BM 128/warps 2, BM 256/warps 4) run at 1.91-1.925 ms, the ones with 64
(BM 64/warps 2, BM 128/warps 4) at 2.36-2.45. What it actually shows is a frontier - 128 int32
accumulators plus operands saturate the register file, so occupancy cannot be bought with more warps
without handing back the per-thread ILP that the MMA shape needs. `num_stages` is flat for the same
reason: staged buffers need registers that do not exist.

**And the occupancy fix is closed by measurement before it was built.** Three configurations, same kernel:

| config | registers/thread | blocks/SM | occupancy | warps/SM | time |
|---|---|---|---|---|---|
| BM 256, warps 4 | 255 | 2 | 16.5% | 7.92 | **1.910 ms** |
| BM 128, warps 2 | 255 | 4 | 16.5% | 7.92 | 1.925 ms |
| **BM 128, warps 4** | **138** | 3 | **24.8%** | **11.89** | **2.355 ms** |

The one configuration that actually trades registers for occupancy - 138 registers, 24.8% occupancy, 12
warps instead of 8 - is **23% slower**. So occupancy was already doubled once, and the price was the
per-thread ILP that the MMA shape needs. Splitting the output channels into two blocks of 32 would land on
exactly that configuration by the invariants (64 accumulators per thread, 8192 outputs per block, 16200
blocks in the grid), so it is the same experiment with a different name, and it is already measured as a
loss.

Hence the honest number for this kernel as written: **1.49x of engine and 1.27x of wall** (block 1.97 ms,
frame 37.7 ms), not 1.93x - that would need 1.43 ms, which needs 107 TOPS, which needs the pipeline to run
at ~90% of the counter's "compute throughput" while the same counters say that buying occupancy at the cost
of ILP is a net loss here.

**The write amplification is not the kernel's tile, and the counter says so in one number.** Store sectors
at the LSU are 4,147,200 x 32 B = **132.7 MB, exactly the bytes produced** (1.000x). So the kernel writes
the core and only the core - no halo on the output, which refutes the 18/16 hypothesis that predicted
1.266x. The amplification appears one level down:

| level | traffic | against produced |
|---|---|---|
| LSU store sectors | 132.7 MB | **1.000x** |
| L2 write sectors | 323.6 MB | **2.44x** |
| DRAM writes | 184.6 MB | 1.39x |

So each 32-byte sector store turns into ~2.4 L2 sector operations and ~1.4 bytes at DRAM. That is
write-allocate behaviour, not layout and not the tile: the stores are already exactly the right size, and
1.39x of DRAM traffic is what the memory system does with them. Worth 0.13-0.2 ms, and **not fixable in
the kernel** without changing how the output is consumed (e.g. writing straight into a buffer the next
op reads, so the L2 lines are already owned).

Where that leaves the lane, honestly: **int8 stops at 1.49x of engine time and 1.27x of wall** (block 1.97
ms, frame 37.7 ms), and that is a measured local optimum in the tile space, not an assumption. Occupancy
was bought and cost 23%; 255 registers is not a compiler accident but the state in which per-thread ILP is
maximal; the traffic excess is 0.16 of the 0.73 ms gap; and the last 0.13 ms belongs to L2 write-allocate,
which is outside the kernel.

**The register cap is reachable after all, and the spill is tiny.** `maxnreg` as a Triton launch option is
ignored, and `ptxas -maxrregcount` is ignored too while `.reqntid` is present in the PTX - ptxas keeps 138
both ways. Strip the launch bounds and the cap applies:

| PTX variant | cap | registers | spill stores / loads |
|---|---|---|---|
| as emitted | none | 138 | 0 / 0 |
| as emitted | 128 | 138 (cap ignored) | 0 / 0 |
| **`.reqntid` stripped** | **128** | **128** | **8 B / 16 B** |
| `.reqntid` stripped | 168 | 168 | 0 / 0 |

So the risk that killed the idea in the abstract - "forcing 128 spills the mainloop" - measures at **8 bytes
of stack**, i.e. a handful of values, not a hot loop. At 128 registers and 128 threads a block needs 16,384
registers, so **four blocks and 16 warps fit** where three blocks and 12 warps fit at 138. The total in-flight
accumulator comes back to 16 x 64 = 1024, i.e. the same as the current best configuration, which is why the
prediction for this point is **parity, not a win** - it buys warp-level latency hiding at constant ILP,
which is exactly the unnamed 6.24 cycles.

And the third check needed correcting before it could be read. **`ptxas -v` reports static shared memory;
Triton uses dynamic**, declared as `.extern .shared .b8 global_smem[]` and sized at launch - so the zero in
that line was expected and proved nothing. The number that matters is `metadata.shared`:

    metadata.shared = 12288 B  (12 KB per block)

With 128 registers and 128 threads: four blocks need 49 152 B of smem (of 102 400) and exactly 65 536
registers (of 65 536) - **four blocks and 16 warps fit on both axes**, and the fifth would exceed the
register file. So the point is viable, and the one barrier the kernel declares is the staging that made the
"zero smem" reading wrong in the first place.

And it does not scale with the flag: `metadata.shared` is **12 288 B at `num_stages` 1, 2, 3 and 4
alike**, with the same 128 registers and the same 8-byte spill after the cap. So Triton does not deepen the
staging here at all - the flat timing sweep is not "the registers would not allow more", it is "the flag
changes nothing". The cross term I wanted to test (128 registers **and** a deeper staging) therefore does
not exist as a configuration: there is no deeper staging to combine with.

Which leaves exactly one point to run - but it is not "the same ILP with more warps", because the register
cut comes out of the operand budget, not the accumulator:

| config | registers | accumulator per thread | left for operands and addresses |
|---|---|---|---|
| BM 256, 255 regs | 255 | 128 | 127 |
| BM 128, 128 regs | 128 | 64 | **64** |

And 64 is barely enough, which is why the cap spills 8 bytes at all. So the experiment asks whether warp
switching replaces per-thread load hoisting - a real question, not a free lunch.

Three distinguishable predictions, each against its own base:

| model | prediction | base |
|---|---|---|
| total in-flight accumulator binds | **1.97 ms** | parity with today's best |
| warps hide latency | **2.355 x 12/16 = 1.766 ms** | 10% better than today's best |
| load hoisting beats warps | **> 2.355 ms** | worse than the same shape at 138 registers |

The second base has to be 2.355 - the same BM/BN/warps at 138 registers and three blocks - because
comparing only against 1.97 mixes a tile-shape change with a register-budget change.

**And the ceiling at this budget is 1.33x of wall, not 1.48x.** Saturating the pipeline from the 2.355 base
would need 2.355 x 12/1.363 = 20.7 warps, and four blocks of four warps gives 16; five blocks would need
102 registers or fewer (38 left for everything but the 64-value accumulator) and six would need 85. So 16
warps is the practical maximum at BN=64, and the "warps hide latency" model gives 1.766 ms = 34.46 ms per
frame = 1.633x engine = **1.334x of wall**, i.e. **+5% over today's 1.271x**. 1.48x is a boundary, not an
expectation.

A fourth silent contract for the launcher, and the only one whose stale value is still a valid number:
BM 128 doubles the **grid** to 16 200 blocks while leaving the block size at 128. A launcher that keeps
8 100 computes half a frame - no fault, just half an image.

Given that the prize is 0 to +5% of wall, the launcher is worth building as the **delivery path for the whole
int8 lane** (sixteen Triton kernels under a CUDA graph between two small engines, no plugin and no QDP),
with the register experiment as a by-product, rather than as its own project.

That number is also a launcher requirement, not a detail: `cuLaunchKernel` must be given
`sharedMemBytes = 12 288`, or the kernel reads and writes outside its staging buffer - a silent corruption
rather than a fault, which is exactly what the bit-exact comparison is for.

Three checks on data already collected, and all three favour the point:

| check | result | consequence |
|---|---|---|
| shared memory per block | **0 bytes** (ptxas reports 1 barrier, no smem) | nothing caps the fourth block; smem was never the constraint, the 3-block limit was registers alone |
| `.reqntid` stripped, **no cap** | **138 registers** | removing the directive is neutral on its own, so the 128 variant isolates the register count and nothing else |
| where the spill sits | **6 LDL/STL in the whole kernel**, scattered at five addresses, none in a hot loop | cold spill, ~50 MB of traffic at worst, 0.17 ms - not a mainloop disaster |

And the prize is bounded by the mechanism that actually binds: with 16 warps the figure is linear in warps,
`1.97 x 12/16 = 1.48 ms` (pipeline 92.3% of the 69.2% it runs at today), not the 1.43 ms that saturation
would need, because saturation requires more than 16 warps. In frames: 16 x 1.478 + 6.2 = 29.85 ms = 1.885x
of engine, **1.435x of wall** against today's 1.49x / 1.27x - +13% of wall, not more.

Three distinguishable outcomes are recorded in advance, so the result reads unambiguously: ~1.97 ms means the
register-frontier model holds and occupancy does not matter (branch closed with its mechanism); ~1.48 ms
means latency was the binder and 12 warps were not enough; worse than 1.97 means the hypothesis was not
tested (a hot spill, since smem is now known to be zero).

What remains is not a question of measurement but of plumbing: load the cubin through `cuModuleLoadData` and
launch with the same 128-thread grid, bypassing Triton's launcher. That is the last untried point in this
lane, and its prize is bounded by the lane's own ceiling: at best ~1.43 ms per block = 1.93x engine / 1.45x
of wall against the 1.49x / 1.27x measured today.

**The one flag that would have tested the register frontier is not reachable from Triton 3.5.** Passing
`maxnreg=128` at BM 128/warps 4 changes nothing measurable (2.478 -> 2.459 ms) and, more to the point, the
registers stay at **138** with occupancy at **24.77%** and the block limit still "3 by registers". So the
compiler ignores the request here; testing the frontier needs the PTX `.maxnreg` directive patched before
cubin assembly, which is a different kind of work from a launch option.

That leaves the frontier described but not crossed: **the register file holds the SM's total in-flight
accumulator, and warps are bought in whole blocks.** The two measured configurations say it in one line -
BM 256/w4 has 8 warps x 128 values = 1024 in flight and runs at 1.910 ms; BM 128/w4 has 12 warps x 64 = 768
and runs at 2.355 ms. So the slower one lost 25% of the total in-flight work, not "ILP traded for
occupancy", and the loss is block quantisation (138 registers give 3.71, i.e. 3 blocks, not 4).

Corrections to the two counters I read too quickly: `lts__t_sectors_op_write` at 2.44x the produced bytes
almost certainly counts stores **and writebacks** (4,147,200 + 5,768,750 = 9,915,950 against 10,111,279
measured, a 2% match) rather than amplification per sector, and **write-allocate is refuted by the read
counter**: a read-for-ownership over the output would add 132.7 MB of DRAM reads, and the measurement shows
141.6 MB against an input floor of 132.7, i.e. +8.9. So DRAM writes at 1.39x are worth 0.18 ms, sit outside
the kernel, and their mechanism is **not attributed** - which is the honest label, not "write-allocate".

What is still unnamed is *why* the pipeline idles 6.24 cycles per issued instruction at 16.5% occupancy.
The stall breakdown section did not produce the per-reason counters in that run, so the ledger records the
surrounding evidence - not occupancy (measured), not DRAM bandwidth (62.6%), not conversions (num_stages
flat), not traffic (0.16 of 0.73 ms) - and leaves the mechanism open rather than guessed. At 90% of the
measured traffic the block is 1.20 ms, at 90% of the tensor pipe 1.15 ms - both land near **1.2 ms**,
i.e. 16 x 1.2 + non-GEMM ~ 25.4 ms = 2.22x of engine, which is **1.55x of wall** on gated anime (engine
share 0.646) and 1.82x ungated. The owner will see 1.55x, not 2x.

Extrapolating one block to sixteen is legitimate here for a specific reason: the output is 132.7 MB
against 32 MB of L2, so it must reach DRAM between blocks and the single-block measurement is
representative.

Best measured so far, for the record: 1.854 ms and **82.5 TOPS** (BM 256, warps 4, one stage), against the
147.4 TOPS K-major peak - 56%.

The likely limiter, given the counters: the nine *masked* loads (a predicate per tap per program) plus
store amplification (170 MB written against 132.7 MB produced, 1.28x). The standard fix is a border-free
fast path - compute the interior with unmasked, vectorised loads and handle the frame edges in a separate
kernel or a separate branch - which is a kernel change, not a modelling one.

##### The direct launcher runs, and fails its own bit-exact check (2026-11-15)

`harness/scripts/launch_cubin.py` loads the capped cubin with `cuModuleLoadData`, launches 32 x 128 threads with
`sharedMemBytes` from the metadata, and compares against the shift-plus-`_int_mm` reference. Two things were
needed to get that far, both of the "silent contract" class:

- `cuLaunchKernel` returned **400 (invalid value)** until the argument types were pinned. Without `argtypes`
  ctypes passes Python ints as 32-bit C ints and the driver rejects the call.
- It returned 400 again with a context I created myself: **torch already owns the context**, and a module
  loaded into a second context cannot see tensors allocated in the first. Sharing torch's context
  (`cuCtxGetCurrent`) fixed it.

And the 0.66% turned out to be the **reference's** bug, not the kernel's: it declared
`wf = w * 0.01` and then accumulated the raw weights, so almost every output saturated at +-127 and only the
handful of pixels near zero disagreed - "126 on 0.66%" is exactly what that looks like. With the scale
applied the comparison reads **max |diff| 0, share 0.00%**: the capped cubin, loaded and launched by hand,
reproduces the reference bit for bit. That is the second time this reference has caught a mistake (the first
was `permute(1,0,2)`), which is why it exists.

**Then the timing at the same frame, and the register point does not win.** 1920x1080, grid 16 200:

| configuration | registers | blocks | warps | TOPS |
|---|---|---|---|---|
| BM 128, capped, hand-launched | 128 | 4 | **16** | **68.5** (2.232 ms) |
| BM 128, uncapped | 138 | 3 | 12 | 64.9 (2.355 ms) |
| **BM 256, uncapped** | 255 | 2 | 8 | **77.6** (1.970 ms) |

So 16 warps buy **+5.5% over their own base** and stay **11.7% behind the best configuration**, which has
half the warps and twice the per-thread work. (The first pass of this was measured at 1080x1080, where BM 128
also straddles row boundaries - 1080/128 = 8.4375 against 1920/128 = 15 - so the same-frame rerun was needed;
it moved 67.6 to 68.5 and changed nothing.)

And the mechanism is not "not enough registers" but **ILP against TLP at fixed register-file capacity**:

| config | threads/SM | accumulator | operands | total |
|---|---|---|---|---|
| BM 256, 255 regs | 256 | 32 768 | 32 512 | 65 280 |
| BM 128, 128 regs | 512 | 32 768 | 32 768 | 65 536 |

Both spend the whole file and split it identically; only the per-thread cut differs, 128/127 against 64/64
at twice the threads. **And the last candidate point - 168 registers on BM 256, which would give three blocks and 12 warps while
keeping 128 accumulator values - is dead on the compiler.** `ptxas -maxrregcount` on the same PTX:

| cap | registers | stack frame | spill stores / loads |
|---|---|---|---|
| none | 255 | 176 B | 184 B / 196 B |
| 170 | 170 | 1 336 B | 1 640 B / 2 128 B |
| **168** | **168** | **1 496 B** | **1 788 B / 2 200 B** |
| 160 | 160 | 1 720 B | 2 468 B / 2 896 B |

A 1.5 KB stack frame is not the 8 bytes the BM 128 cap cost: this configuration genuinely needs the register
file and spills ~1.8 KB per thread when it does not get it. So the window the arithmetic suggested is not a
window - and the native 184 B of spill at 255 registers says the same thing more quietly: BM 256 already
borrows a little against the file to hold its ILP.

Both effects are real - warps helped by 5.5% - and ILP was worth 11.7%, which is why
the branch closes. It closes because 16 warps is the ceiling at BN=64 (five blocks would need 102 registers,
six 85), not because warp switching does nothing.

**And what closed is this codegen's frontier, not the hardware's.** The accumulator-to-overhead ratio is about
1:1 at 128 values per thread, while the MMA shape needs roughly 1:4 (four A fragments, two B, double
buffering, ~32 registers of addresses and predicates). A kernel holding overhead near 32 would fit 49 152
registers of accumulator per SM at the same warp count, and the frontier would move. Same class of statement
as E14's "the name is the name of the winner, not a symptom": **77.6 TOPS is the ceiling of what the compiler
chose, not of what the shape allows** - and a hand-written kernel would hit its own ceiling, not this one.

That is the third of the three predictions: not parity, not the 1.766 ms the "warps hide latency" model
wanted, but a small gain over the wrong base. **Warp switching does not replace per-thread load hoisting at
this budget**, and the 64 registers left for operands are the reason. The branch is closed with its mechanism
and with the launcher that outlives it. So the
launcher works and the kernel it loads does not yet agree with the reference. By the plan this stops the
work here: the timing is meaningless until the picture is bit-exact, and the two candidate causes are both
invisible to a smoke test - the two trailing scratch pointers (Triton sizes them from
`metadata.global_scratch_size`, and small stand-in buffers would corrupt exactly this way) and the frame
border, where the kernel's masked loads and the reference's zero padding must agree.

What is established: the register-capped cubin loads and runs, and the four contracts (grid 16 200 at BM
128, block 128, `sharedMemBytes` 12 288, scratch) are all real - the first two produced hard failures, which
is the lucky half; the scratch and the border are the quiet half, and the comparison caught them on the
first try, which is why it exists.

##### A, step 1: the graph is bit-exact, and the fifth contract is subtler than "pass torch's stream"

Sixteen launches captured into a CUDA graph and replayed, compared against the same sixteen run eagerly:

    graph vs eager: max |diff| 0, share 0.000%

Getting there needed one correction that is worth recording, because it fails **silently**. The stream has to
be read *inside* the capture:

    with torch.cuda.graph(g):
        live = torch.cuda.current_stream().cuda_stream    # the capture side-stream
        for _ in range(16):
            launch(live)

Reading it before the `with` and passing that handle produces an **empty graph** - torch warns
"The CUDA Graph is empty", the capture still reports OK, and the replay does nothing at all, so the frame
simply does not update. That is the third instance in this lane of a wrong picture with no error (after the
reference ignoring its weight scale and the gate statistic in the wrong units), and the first one that would
have shipped as "the graph made it faster" if nobody had compared the output.

Capture works with raw `cuLaunchKernel` on the side-stream, so the 16 Triton kernels can live under a graph
without a plugin, a QDP or a TensorRT header - the delivery path the lane needed.

##### Film is the better target, and its accuracy question is one number (2026-11-15)

Every composition measurement so far was taken on material the gate reuses, which is the hard case: the
engine's error and a stale reference add, and the amplification is 1.55-2.03x over the sum of the parts. On
material where the gate **refuses** there is no reuse, so there is no composition - the defect is the
engine's own, measured directly.

And the gate refuses on exactly the material whose engine share is largest:

| material | gate | engine share of the frame | wall gain at 1.49x engine |
|---|---|---|---|
| anime 1080p | reuse 72.7% | 0.646 | 1.27x |
| film 1440x1080 | **refuses** (noise floor) | 0.866 | **1.40x** |
| Bebop 1080p | **refuses** | 0.904 | **1.42x** |

**Measured on Bebop, five scattered frames, int8 against fp16, no gate filter: worst share 0.016%, worst
peak 16** (0.007% and 0.012% on the individual frames - 0.007, 0.016, and one below). Against a bar of 0.1%
that is a 6x margin, and peak 16 sits under the "~20" guidance.

So the film path asks one question and answers it: no reuse, no staleness, no break-even in points of reuse.
The prize there is 1.40-1.42x of wall against 1.27x on anime, and in absolute terms a two-hour film goes
2.99 h -> 2.10 h.

What remains for film is delivery, and none of it needs new knowledge: sixteen launches under a CUDA graph
on torch's stream (the launcher is bit-exact and knows its four contracts), the head and tail either left on
TRT (non-GEMM 6.2 ms, 1.49x engine) or written as their own memory-bound kernels (non-GEMM ~1.7 ms, 1.69x
engine), scales calibrated per video from frames the probe already runs, and the probe in the render path
where it degenerates to "own defect against fp16 on N frames".

2x on film would need 104 TOPS after the head and tail become ours (71% of the peak), against 77.6 measured
and a codegen that cannot give more at this register allocation - so 2x stays a hand-written-kernel project,
not the next step.

##### The int8 fit probe (2026-11-15)

`harness/scripts/probe_int8_fit.py` answers the only product question left: does int8 fit *this* video? It takes N
adjacent pairs spread over the clip, calibrates on a disjoint half, applies the gate's own statistic so
only pairs it would actually reuse enter the composition, and returns one verdict.

| clip | gated pairs | engine | composition | verdict |
|---|---|---|---|---|
| Chainsaw | 4 of 6 | 0.006% / 14 | **0.017% / 18** | **int8** |
| Violet | 5 of 6 | 0.031% / 20 | **0.258% / 26** | **fp16 fallback** |

Three bugs it had to lose before it could answer, all of the class this ledger keeps recording: it paired
frames spread over the clip instead of adjacent ones (measuring scene changes, an "80% error"), then it
compared every pair rather than the pairs the gate would reuse (0.45% and 7.5%, peaks 178 and 187 - a cut
is not a reuse), and its gate statistic was computed in 0..1 instead of levels, which passed every pair
trivially. Each of the three looked like a result.

Cost: ~1.5 s of forwards per video, against 0.7 s for the existing tile probe. At 1.42x of engine time the
win saves 0.19*T, so the probe pays for itself when the render is longer than ~4.2 s.

##### What int8 costs in reuse: a formula, not a table (2026-11-15)

An engine that is `k` times faster can afford to lose this much reuse, and no derivation is needed beyond
`T = N + E*r` against `T' = N + (E/k)*r'`:

    margin (in reuse points) = (k - 1) * r      where r is the RENDERED fraction
    applicability:  r <= 1/k  (0.704 at k = 1.42)

| title | rendered | break-even reuse | margin | gate-only defect |
|---|---|---|---|---|
| Chainsaw | 33.64% | 52.24% | **14.13 pp** | 0.005% |
| Violet | 35.16% | 50.07% | **14.77 pp** | 0.096% |

Three things follow, and all three contradict the intuition the table was built on:

- the margin depends **only** on the engine's speed-up and the rendered share - neither the engine's share
  of the frame nor the non-engine time enters it;
- it is **larger for titles that render more**, i.e. reuse less. "More reuse means more headroom" is false
  here;
- it is computed from one number the application already logs (reuse counts) - no probe, no int8 forwards.

The margins are therefore **equal within 0.6 pp**, and the two verdicts cannot be explained by different
reuse headroom. The discriminator is the picture: the fp16 gate alone reads 0.005% on Chainsaw (20x of
margin) and 0.096% on Violet (1.04x), and the probe reproduces exactly that -
`(0.005 + 0.006) * 1.55 = 0.017%` and `(0.096 + 0.031) * 2.03 = 0.258%`, both matching the measurement.

Which gives a pre-filter that needs no int8 work at all: since the composition is roughly
`(T_gate + E_engine) * amplification`, requiring it under the bar at an amplification of 2 gives
**`T_gate <= 0.019%`**. Chainsaw qualifies at 0.005%; Violet is rejected at 0.096% without a single int8
forward; the anime clip at 0.017% sits on the line with a measured composition of 0.006%, i.e. the estimate
is conservative where the pixel sets overlap. As with the sample bar, one point does not calibrate the rule
- it is a filter to decide whom to probe, not a verdict.

**This is the ninth time in this file that a number was computed on one population (or scale, or column)
and applied to another**: `mask` instead of `mask + 18`, the noise floor on the adversarial subset, the
16 marginal frames, `lr` without chain state, the 1440x1080 tile ruler, frames spread instead of adjacent,
all pairs instead of gated ones, 0..1 instead of levels, and now rendered subtracted from reuse. Hence the
formula: a formula cannot be subtracted from the wrong column.

##### The rule this investigation earned (2026-11-15)

Eight separate conclusions in this file were wrong in the same way, and it is worth naming once:

> **Before a number is called a result, name the population it was computed on and the population the
> decision applies to. If they differ, the number is about something else.**

The eight, for the record: pruning cost computed on the mask instead of `mask + 18`; the noise floor
computed on the adversarial subset instead of the observed window; acceptance certified on a distribution
of 16 marginal frames instead of the population the threshold serves; reachability checked on `lr` alone
while the gate also depends on chain state and the pixel veto; the tile ruler applied to the wrong
configuration (45.7 ms at 1440x1080); the probe pairing frames spread over a clip instead of adjacent ones;
the probe comparing every pair instead of the pairs the gate would reuse; and the probe's gate statistic
computed in 0..1 while the decision is in levels. The last three happened in one afternoon, in one script,
each looking like a result.

Two corollaries that this file already applies elsewhere and should apply here:

- **A ratio is not a constant.** The composition's amplification over the sum of its parts measured 1.55x
  on Chainsaw and 2.03x on Violet, against 3.9x on an earlier single Violet pair. That variability is the
  argument for keeping the probe rather than deriving a formula.
- **An enrichment is per-title too.** The reuse break-even is ~50% on Violet and ~52% on Chainsaw, from
  their own 6.90 s / 142-of-219 and 6.61 s / 146-of-220 - not a universal number.

And the amortisation, correctly based on the incremental cost: the probe adds ~0.8 s over the tuner's
existing 0.7 s, and only passing titles earn the 1.42x, so `T > 4.2 / P(pass)`. On a 870 s episode that
holds for any P; on a 6.9 s test clip it needs P > 0.61.

What is still missing is `P(pass)` itself: two titles locally, one passing, i.e. 0.5 with an interval that
means nothing. Ten titles is ~15 s of GPU and the probe is ready for it; the clips available here are
three of the same two titles, and the 16-frame excerpts are too short to hold ten adjacent pairs.

##### What the literature closes, and the one item it reopens (2026-11-15)

Written down because these are structural closures - true by the shape of our network, not measured
by us - and because the reopened item was previously dismissed on a measurement that does not answer
the question it was cited for. Nothing in this section is our measurement; every number attributed
to a paper is the paper's, about its own hardware. The queue that turns the open ones into
experiments is `projects/rve/docs/backlog/queue.yaml`; who runs what is `harness/docs/backlog/DELEGATION.md`.

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

##### 2026-11-17: where these measurements live now

This file is history and is cited by line, so it is not rewritten. Measurements from 2026-11-17 onward
live in dated per-lane files, and every conclusion carries a lifetime in the registry:

| what | file | verdicts |
|---|---|---|
| L2 pyramid and cone budgets (E29, E36) | `harness/docs/wsp/history/2026-11-17-budgets.md` | `pyramid-budget`, `cone-budget` |
| C=32 activation forms, bimodal build, streams (B3, C) | `harness/docs/wsp/history/2026-11-17-engine.md` | `b3-prelu-forms`, `streams-no-overlap` |
| the TRT int8 plan's per-conv rate (E33) | `harness/docs/wsp/history/2026-11-17-int8.md` | `trt-int8-rate-79` |

Entry point for "what is true now": `python -m harness.cli db current`.

##### Open questions

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

<a id="evidence-docs-wsp-verified-prelu-forms-txt"></a>

### `harness/docs/wsp/verified/prelu-forms.txt`

#### Three activation forms at C=32, and the C=64 control (B3)

Measured 2026-11-17, one session, interleaved rounds, `harness/scripts/probe_prelu_forms.py`, 1920x1080 fp16
opt 5 avg_timing_iters 8 workspace 8 GiB, 16-block body, synthesized random weights (tactics depend on
shapes, not values - the same trick E30a used).

| form | fused `__myl_` | reformat | pointwise | ms/frame |
|---|---|---|---|---|
| PReLU (what ships) | 19 | 2 | 0 | 23.51 |
| `a*x + (1-a)*relu(x)` | 20 | 1 | 0 | 25.06 |
| LeakyReLU, scalar slope | 1 | 3 | 5 | 26.89 |

The item's premise ("the PReLU does not fuse at 32") is false in this graph: it fuses, and the rewrite
that was supposed to fix it is 6% slower. The scalar-slope form is the one that breaks fusion, which is
a result about int8 epilogues, not about fp16.

The C=64 control is the same probe at the same settings in the same session: **46.80 ms**, 19 fused
layers, one tactic, three rounds within 0.11 ms of each other. The same-session ratio is therefore
**46.80 / 23.51 = 1.99x**, which is the number to quote - dividing by the ledger's recorded 56.68 ms
would be a cross-session quotient.

The C=32 build is **bimodal**: three rebuilds of one ONNX gave 39.23 / 39.29 / 20.95 ms with 0/0/18
fused layers (ledger B3 section; reproducible with `harness/scripts/probe_build_stability.py --trials 3`). The
19-fused census does not discriminate - the conv tactic tile does, and the fast draw is the one with
`..._tilesize256x32x32_stage1_...`.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/prelu-forms.txt \
       --records harness/docs/wsp/verified/records/prelu-forms.jsonl --bars harness/docs/claims-bars.json

@claim R001 23.51 ms "bench rounds [ccf46a14a09fc984] n=3" measured
@claim R002 25.06 ms "bench rounds [ccf46a14a09fc984] n=3" measured
@claim R003 26.89 ms "bench rounds [ccf46a14a09fc984] n=3" measured
@claim R004 46.8 ms "bench rounds [ccf46a14a09fc984] n=3" measured

<a id="evidence-docs-report-int8-lane-2026-11-16-md"></a>

### `harness/docs/REPORT-int8-lane-2026-11-16.md`

#### Отчёт: int8-полоса (RVE, ветка `perf/inference`)

Дата: 2026-11-16, редакция 2 — после разбора с калькулятором. Изменения против редакции 1 перечислены в §9;
два из них меняют вывод, а не формулировку.

---

##### 1. Главное: полоса не имеет продуктового смысла, и не из-за скорости

**ОПРОВЕРГНУТО ниже, в этом же отчёте (см. `int8-restart-under-gate`, harness/docs/wsp/verdicts.yaml):** при `tilesize=0` гейт не создаёт `tile_upscaler`, и ветка тела достижима под гейтом. Абзац оставлен как история. **Тело на int8 по умолчанию не запускается вообще.** `--no_temporal_reuse` не «рекомендация», а условие
работы: гейт временного переиспользования включён по умолчанию (`help: "on by default; skip only, never
invents pixels"`), он создаёт `tile_upscaler`, и ветка тела в `_call_single` не выполняется. В логе
дефолтного прогона нет ни строки сертификации, ни слова `serving`.

Значит «0.484x доставлено» — это **цена в конфигурации, которую продукт не поставляет**. Если включить тело,
придётся выключить гейт, а гейт стоит больше, чем тело может принести:

| конфигурация | кадр целиком | что это |
|---|---|---|
| **гейт ВКЛ + fp16 (дефолт)** | **45.36 мс** | то, что поставляется |
| гейт ВЫКЛ + fp16 | 54.56 мс | цена отключения гейта |
| гейт ВЫКЛ + int8-тело | **104.44 мс** | цена int8 вместе с отключением гейта |

По кадру это **2.30x медленнее дефолта**, а по стене клипа — хуже, потому что гейт ещё и пропускает ~66%
инференсов: по MODEL.md §7 дефолт даёт 11.7 с там, где без гейта 38.85 с (3.3x). Отсюда порядок величины
из разбора: **≈6.9x медленнее дефолта**.

Отсюда вывод, который надо было ставить первым: **полоса не окупается ни в одной конфигурации.** Она
проигрывает по кадру (0.484x), и её включение требует отказаться от выигрыша, который больше её собственного.
Это не вопрос оптимизации ядра — это вопрос, есть ли предмет оптимизации.

---

##### 2. Точность — доставлена и проходит. Предыдущее «недостижимо» отменено

Сертификация на первом кадре рендера: **0.00823%** пикселей >8 уровней (планка 0.1%).

Сравнение рендеров на кадре 10 клипа: **0.07592%** (пик 24, медиана 1.00).

**Популяция названа честно: это ОДИН кадр (№10) одного клипа, не худший из разнесённой выборки.** Правило
проекта требует худший кадр разнесённой выборки; здесь его нет, и 0.07592% нельзя читать как «худший». Пик 24
против ориентира ~20: поставляемый тайловый opt-in живёт с 23 и назван проходящим, так что 24 сравнимо, но это
надо было сказать, а не опустить.

**И это отменяет прежний структурный вывод.** Баг №7 из §4 — эталон считался **после** `calibrate`, то есть
против модели с квантованными весами — означает, что числа прошлого раунда (1.693% / 3.589% и вывод «цепочная
int8-схема не держит бар», «стадия 1 недостижима») измерены против неверного эталона. **Полоса по точности не
закрыта — она проходит.**

---

##### 3. Конфигурации и числа доставки

| с чем сравниваем | fp16 | int8-тело | отношение |
|---|---|---|---|
| движок приложения (tiled, opt-5, голова слита), 1920×1080, **один процесс**, 3 раунда | 46.29 мс | 48.55 мс | **0.953x** (отношение медиан); медиана трёх по-раундовых отношений **0.962x** |
| тот же движок, **рендер приложения** (гейт ВЫКЛ) | 46.92 мс | 96.88 мс | **0.484x** |
| монолитный TRT-движок (мой харнесс) | 80 мс | 48.55 мс | 1.65x |
| torch fp16 (eager, channels_last) | 126.9 мс | 48.55 мс | 2.6x |
| движок приложения, 3840×2160 | 186.1 мс | 240.5 мс | 0.774x |

Две агрегации в первой строке дают 0.953 и 0.962 — это разные вещи (отношение медиан против медианы
отношений), и в редакции 1 стояло 0.962 без пояснения. **Победа над torch fp16 (2.6x) не является целью**:
eager-путь никто не поставляет, приложение использует TensorRT. Её не надо предъявлять как достижение.

---

##### 4. Оба пути измерены одним инструментом: GPU 0.703x, стена 0.472x

Разбор был прав: 0.66x сравнивал GPU одной стороны со стеной другой. Оба пути измерены событиями в одном
процессе, на одних кадрах:

| | GPU (события) | стена (`inference`) | накладные |
|---|---|---|---|
| fp16-движок | **49.80 мс** | 51.43 мс | **1.63 мс** |
| int8-тело | **70.8 мс** | 109.07 мс | **38.3 мс** |

Отсюда два отношения, и оба честные, потому что на одном инструменте:

- **GPU к GPU: 49.80 / 70.8 = 0.703x** — тело медленнее на 42%;
- **стена к стене: 51.43 / 109.07 = 0.472x** — вдвое, потому что сверху 38.3 мс накладных.

**И 38.3 мс — наши, не приложения.** У fp16-пути накладные 1.63 мс: один enqueue. У int8 — 16 Python-запусков
Triton с упаковкой аргументов, три конверсии, чтение событий предыдущего кадра, обслуживание пробы. Утверждение
«38 мс не собственность int8» из прошлой редакции **отменено**.

Заодно закрывается открытый пункт про разрыв хопа: `38.2 / 16 = 2.3875` против `2.198` в харнессе — **1.086**, а
MODEL.md документирует **9%** от NVENC рядом (51.37 → 55.99, контролируемый A/B). Совпадение в 0.3%: разрыв —
конкурентный энкодер, а не синхронизация и не поток. Новый замер не нужен.

##### 4b. Найден баг №13: мёртвая строка перезаписывала буфер — и 18 мс снялись

Проверка «какой из трёх путей PReLU врёт» дала неожиданное: в изоляции **`F.prelu` и `torch.where(out=)`
бит-в-бит одинаковы** (max |diff| 0.000000 на полном кадре 512×512), а fused-ядро расходится. Значит виноваты
не они. Причина оказалась в файле:

```python
quantise_nchw_to_nhwc[qgrid](self._prelu_buf, ..., buf, ...)
buf = (torch.round(head / s_act...).clamp(...).to(int8).permute(...))   # <-- остаток старого патча
```

Строка от предыдущей правки **перезаписывала `buf`** квантованием **до-PReLU** головы — то есть PReLU
отбрасывался, и оба «виноватых» варианта давали одну и ту же долю 2.15278%, потому что до их результата дело не
доходило. Это тот же класс, что баги №4 и №6: правка не убрала то, что заменяла.

После удаления и двух безопасных правок (`torch.where` с переиспользуемым `out=`, плюс предвыделенный
`_prelu_tmp`, чтобы `head * slope` тоже не аллоцировал):

| | было | стало |
|---|---|---|
| сертификация | 2.15278% (отказ) | **0.00846%** (serving) |
| фаза квантования | 20.2 мс | **8.9 мс** |
| тело, GPU | 70.8 мс | **62.3 мс** |
| тело, стена | 109.07 мс | **97.4 мс** |
| **GPU к GPU** | 0.703x | **0.799x** |
| **стена к стене** | 0.472x | **0.528x** |

Накладные int8 по-прежнему 35.1 мс против 1.6 у fp16 — это 16 Python-запусков и три конверсии, и они остаются
главным оставшимся куском. Из 8.9 мс квантования 1.6 — само ядро; остальное ещё не объяснено.

##### 4c6. Переквант перенесён в эпилог: −12.2 мс, и полоса вышла в паритет

Самое крупное одиночное изменение за всю полосу, и оно из диагноза ревью: переквант per-channel → per-tensor
делался **в прологе, на каждой из девяти загрузок** (9× на элемент), хотя по смыслу он нужен **один раз на
элемент**. Перенос в эпилог предыдущего хопа:

| вариант | мс/хоп | 16 хопов |
|---|---|---|
| переквант в прологе (было) | 2.518 | 40.3 |
| **переквант в эпилоге (стало)** | **1.753** | **28.0** |

`ptxas` на прологе: **REG 165, STACK 0, LOCAL 0** — spill нет, так что дело не в регистровом давлении, а в
одной и той же арифметике, применённой девять раз вместо одного.

Результат в рендере:

| | было | стало |
|---|---|---|
| окно вызова | 61.7 мс | **51.54 мс** |
| против fp16 (51.06) | 0.828x | **0.991x** |
| сертификация | 0.00846% | **0.00832%** |

То есть полоса **вышла в паритет с движком** и это ещё до двух названных фиксов (квантование 8.9 → 3.2 и
голова+хвост 12.4 → 2.6). С ними проекция `51.54 − 15.5 = 36.0` мс → **1.42x** в продуктовых числах, то есть
выше ворот 1.40 ещё без выигрыша от интеграции под гейт.

##### 4c17. Вердикты на обслужебной глубине получены — оба fp16

После починки (`c2` считался после вердикта → `NameError` → оба пробы рапортовали `probe failed`):

| тайтл | d1 | d2 | худшая | вердикт |
|---|---|---|---|---|
| a5_chainsaw | 0.002% | 0.390% | **0.390%** | **fp16** |
| a5_violet | 0.104% | 0.166% | **0.166%** | **fp16** |

**Честная заметка о процессе:** первый прогон я объявил как «идёт в фоне», не проверив; он уже завершился с
ошибкой, и оба пробы вернули `probe failed`. Ошибка была моя и в моём же патче — `c2` использовался в вердикте до
присваивания. Вердикты выше получены после исправления и проверены полем `error: none`.

##### 4c16. На глубине, которой служит гейт, бар не держится ни на одном тайтле

Проба мерила композицию на **соседних** парах (`int8(t−1)` против `fp16(t)`), а гейт отдаёт ссылку возрастом до
k кадров (сертифицировано 2). С правильным фильтром — глубину 2 считаю только там, где **оба** шага под порогом
гейта, иначе мерится движение, а не квантование:

| тайтл | d1 (соседи) | **d2 (глубина гейта)** | отношение |
|---|---|---|---|
| a5_chainsaw | 0.002% | **0.390%** | **195x** |
| a5_violet | 0.104% | **0.166%** | 1.6x |

**На глубине 1 проходят оба, на глубине 2 — ни один.** И рост в 195 раз на Chainsaw показывает, что ломает бар
не квантование, а **устаревание ссылки**: на статичном материале int8 почти идеален (0.002%), но как только
гейт отдаёт кадр двумя шагами раньше, вклад движения перекрывает всё.

Вердикт пробы переведён на **худшую обслуживаемую глубину** (`max(d1, d2)`) — «сертифицируй той глубиной, которой
служишь». На этих двух тайтлах это означает **fp16**.

##### 4c15. F_total измерен: 65.4 с, и он в 77 раз больше того, что я считал

| компонент | время |
|---|---|
| проба (`Int8Probe`, 1080p, 8 позиций × 3 кадра) | **64.6 с** |
| сборка тела + калибровка + сертификация | 0.85 с |
| **F_total** | **65.4 с** |

Я считал 850 мс — это только model-call первого кадра. Проба идёт в `setupUpscale`, до первого кадра, и я её в
F не включал. Отсюда порог безубыточности:

| состояние | экономия на отрисованном кадре | порог |
|---|---|---|
| сейчас | 8.45 мс | **23260 кадров ≈ 16.2 мин видео** |
| после головы+хвоста (окно 26.8) | 24.26 мс | **8102 кадра ≈ 5.6 мин** |

И что это значит вживую (24 fps, 33.3% кадров отрисовывается):

| видео | сейчас | после головы+хвоста |
|---|---|---|
| 1 минута | **−61.4 с** | −53.8 с |
| 24 минуты (серия) | **+31.8 с** | **+213.7 с** |

То есть **на коротких клипах полоса в минусе**, на 24-минутной серии даёт от 32 с до 3.5 минут. Проба —
доминирующая стоимость, и она сокращаема (меньше позиций, меньше кадров, меньшее разрешение), но каждое
сокращение меняет инструмент и требует валидации вердиктов, как и предупреждалось про кроп.

##### 4c14. Разнесённая калибровка внедрена как рабочий путь

Проба теперь **отдаёт свои декодированные кадры**, и тело калибруется на них — один набор форваров на обоих
потребителей, как и требовалось. В рендере:

```
int8 body self-certification on the first frame: 0.00821% -> serving
host timeline: ... | total 42.61 ms
```

Окно не изменилось (42.61 против 42.75) — калибровка разовая, а не покадровая. Изменилась композиция:

| тайтл | однокадровая (было) | разнесённая (стало) | планка |
|---|---|---|---|
| a5_chainsaw | 0.116% | **0.006%** | 0.1% ✓ |
| a5_violet | 0.377% | **0.050%** | 0.1% ✓ |

То есть 20x и 7.5x улучшение композиции, и **Violet проходит**. Это не оптимизация, а условие корректности:
однокадровая калибровка отказывала видео, которые обслуживаются.

##### 4c13. Зазор был в КАЛИБРОВКЕ, а не в ядрах — и проба не врёт

Мой тест сравнивал симуляцию, откалиброванную на **16 кадрах**, с телом, откалиброванным на **одном**. Это была
не та популяция. С одинаковой калибровкой:

| тайтл | симуляция | **реальные ядра (16 кадров)** | зазор | вердикт |
|---|---|---|---|---|
| a5_chainsaw | 0.010% | **0.006%** | ядра **в 1.71x лучше** | оба → int8 |
| a5_violet | 0.084% | **0.050%** | ядра **в 1.69x лучше** | оба → **int8** |

Три следствия, и все три важнее предыдущего вывода:

1. **Проба не ошибается.** Она консервативна в **1.7x**, что сходится с измеренным ранее 1.8x на 4K. Прокси
   валиден, чинить его не нужно.
2. **Дефект — в теле: калибровка на одном кадре.** Именно она давала 0.116% и 0.377%. План это и требовал с
   самого начала («разнесённые кадры плюс намеренно яркие»), а я калибровал на первом кадре.
3. **Violet проходит** с реальными ядрами: **0.050%** против планки 0.1%. То есть обслуживаемая популяция
   шире, чем следовало из прологовой схемы и однокадровой калибровки.

Предыдущий вывод («проба оптимистична в 10x и допускает видео, которые ядра проваливают») **отменён**: это было
сравнение двух разных калибровок, а не двух математик.

##### 4c12. Проба против реальных ядер (отменено)

Прогнал одни и те же пары двумя путями — симуляцией пробы и настоящим `Int8Body`:

| тайтл | симуляция | **реальные ядра** | зазор | вердикт пробы | вердикт ядер |
|---|---|---|---|---|---|
| a5_chainsaw | 0.010% | **0.116%** | **11.6x** | int8 | **fp16** |
| a5_violet | 0.084% | **0.377%** | **4.5x** | int8 | **fp16** |

**Оба тайтла проба пропускает, а ядра проваливают.** То есть прокси не консервативен — он оптимистичен в
10 раз, и цена этого — не ложные отказы (как предполагалось), а **ложные допуски**: проба отдаёт под
обслуживание видео, которое бар не проходит.

Причина видна из того, что мерилось раньше: sim и сборка соглашались на **движковой** ошибке (int8(t) против
fp16(t)), а композиция добавляет устаревание ссылки — и именно на ней пути расходятся. Плюс теперь симуляция не
моделирует ни переквант в эпилоге, ни слитый PReLU+квант.

**Следствие:** проба обязана звать путь, который обслуживает, а не `_quantised_forward`. Иначе решение
per-title принимается по математике, которой в рендере нет — ровно тот зазор инструментов, который в этой
работе уже трижды давал ложные «пределы».

##### 4c11. Точность переснята на эпилоговой схеме — и она не хуже

Рендер в **дефолтной** конфигурации (гейт включён), кадр 10, 3840×2160:

| схема | >8 уровней | пик | медиана |
|---|---|---|---|
| прологовая (прежняя) | 0.07592% | 24 | 1.00 |
| **эпилоговая (сейчас)** | **0.07024%** | 26 | 1.00 |

Доля **улучшилась** (одно округление вместо двух), пик на два уровня выше — обе величины под планкой 0.1%.
Прогноз ревью, что «Violet останется fp16» и что снятие округления не убирает член устаревания ссылки,
согласуется: 0.0702% — это ВИДЕО-уровень, а композиционный член задаётся гейтом, не движком.

**И вердикты пробы от этого изменения не зависят:** `Int8Probe` считает через `_quantised_forward` —
симуляцию, а не через мои ядра, так что перенос перекванта её вердикты не трогает. Переснимать пробу не нужно,
и это стоит было проверить прежде, чем ставить на неё день работы.

##### 4c10. Тело уже обслуживает под гейтом — «инертность под дефолтом» была артефактом

Прогон **без** `--no_temporal_reuse`, то есть в поставляемой конфигурации, с активным гейтом
(`reference reuse: 16/24 frames reused (66.7%)`):

```
int8: start (tilesize=0, tile_upscaler=False, int8Fit=int8)
int8: wrapper=UpscaleModelWrapper model=SRVGGNetCompact
int8: built (supports=True)
int8 body self-certification on the first frame: 0.00832% of pixels >8 levels (bar 0.1%) -> serving
```

**Тело работает под дефолтом.** Моё прежнее «под дефолтом тело инертно, включение требует отказаться от гейта в
3.3x» — **неверно**, и вот почему: `TiledUpscaler` создаётся только при `tile_reuse`, а это opt-in от пробы;
гейт (`temporalReuse`) — другой механизм и `tile_upscaler` не создаёт. Ветка тела достижима и под гейтом.
Прошлые прогоны не показывали сертификации потому, что тело тогда **не собиралось** (`supports=False` из-за
`PadInputConv`), а не потому, что ветка не бралась.

**Следствие для продуктовой арифметики.** Гейт и тело не конкурируют: гейт пропускает 66.7% кадров, тело
обслуживает остальные 33.3%. По Amdahl при доле движка в цикле 0.646:

| S (движок) | стена под гейтом (доля движка 0.679 для этого клипа) |
|---|---|
| 1.098x (сейчас) | **1.065x** |
| 1.391x (с последним фиксом) | **1.235x** |

То есть **6.5% стены** — на поставляемой конфигурации, без компромисса с гейтом, и 23.5% после фикса
головы/хвоста. **Но это проекция через Amdahl, а не измерение на этих клипах:** per-video фикс 850 мс против
экономии 4.54 мс на отрисованном кадре даёт порог **187 отрисованных кадров ≈ 23 с видео**, и на 24-кадровом
прогоне полоса будет в 2.35 раза медленнее. Порог падает до ~6.4 с после фикса головы/хвоста. Раздел §1 отчёта («полоса не имеет продуктового смысла») на этом основании **отменяется**.

##### 4c9. Моя «безопасная правка» была регрессом в 2.5 раза — A/B подтвердил трафик

Проверка предсказания по трафику, три варианта одной фазы на реальной форме (1080p, 64 канала):

| вариант | время | коды против эталона |
|---|---|---|
| **A: `F.prelu` + квантующее ядро** | **4.01 мс** | **идентичны (0.000%)** |
| B: `torch.where(out=)` + квантование | **9.91 мс** | идентичны |
| C: слитое Triton (PReLU+квант) | 1.71 мс | **неверны на 34.7% (max 205)** |

**B — это моя «безопасная правка»**, и она в 2.5 раза дороже слитого ATen-ядра: `F.prelu` — один проход
(530 МБ, 1.84 мс), `torch.where` — три eager-прохода (1856 МБ, 6.44 мс). Предсказание 8.04 против измеренных 9.91
сошлось в 19%. Убраны были аллокации, которые, как показал `num_alloc_retries = 0`, и так ничего не стоили.

**Слитое ядро по-прежнему неверно** — 34.7% кодов, хотя мёртвая строка удалена. Значит его прошлый «отказ» был
двойным: и мёртвая строка, и собственная ошибка. Оно осталось бы выигрышем в 2.3 мс, но сначала его надо
починить, и это отдельная работа.

После возврата A:

| | было | стало |
|---|---|---|
| фаза квантования | 9.91 мс | **4.01 мс** |
| окно вызова | 51.54 мс | **46.52 мс** |
| против fp16 (51.06) | 0.991x | **1.098x** |
| сертификация | 0.00832% | 0.00832% (serving) |

**Полоса теперь быстрее движка на 9.8%** в продуктовом окне. С последним названным фиксом (голова+хвост
12.4 → 2.6) проекция `51.06 / 36.72 = 1.391x`, а со слитым ядром `51.06 / 34.42 = 1.483x`.

##### 4c7. Разведка интеграции: путь гейта не заблокирован, это инъекция callable

Час разведки, как и предлагалось, — и ответ короткий.

```python
TiledUpscale.compose(self, scale, upscale_fn: Callable, prev_hr, lr_tensor, reuse_prev=True)
UpscaleTorch.py:502   self.tile_upscaler.compose(self.scale, self.upscale_model_wrapper, ...)
```

`upscale_fn` — **параметр**, который вызывающий передаёт; гейт/тайловый путь дергает его с фиксированной формой
`(1, 3, T, T)`, где `T = tile_lr + 2*halo`. Значит интеграция тела под гейт — это **передать другой callable**, одна
строка, без рефактора. Тело уже shape-generic: буферы кэшируются по форме.

**Настоящая работа не в хуке, а в двух вопросах рядом с ним:**

1. **Калибровка и сертификация.** Сейчас они идут на первом кадре **полного кадра**. Под гейтом первый вызов —
   это тайл `(1,3,T,T)`, и сетки, снятые на тайле, — другие. Значит нужен явный порядок: полнокадровая
   калибровка/сертификация до первого тайла, тайлы обслуживаются уже готовыми сетками.
2. **Три потребителя калибровочных форваров** (масштабы, проба, сертификация тюнера) — слить в один набор, иначе
   F = 2.65 с и порог безубыточности 43 с вместо 14.

##### 4c8. Уточнения к числам перекванта

- **28.0 мс — харнесс-число.** В приложении оба числа на ~9% выше (NVENC): `1.753 × 1.09 = 1.911 мс/хоп`, то есть
  `16 × 1.911 = 30.6` против 40.3, экономия **9.7 мс**, а не 12.2. Окно сэкономило 10.16 — сходится.
- **Сертификация улучшилась не от шума:** 0.00846 → 0.00832%. В прологе значение квантовалось **дважды**
  (per-channel на выходе, per-tensor при загрузке), в эпилоге — **один раз**. Меньше округлений — чуть точнее.
  Это изменение численности, а не эквивалентная замена.
- **Хоп догнал движок по эффективности:** 1.753 мс = **87.2 TOPS** против TRT conv **91 TOPS** (4.3%). Разрыв,
  который в прошлом раунде оценивался в 13.4 мс, сжался до **1.17 мс**. Регистровый фронтир обойдён не борьбой
  с регистрами, а устранением девятикратной избыточности.
- **Проекция: 1.379x при реалистичных 1.3× пола для PReLU** (фаза 4.2 мс, не 3.2), а не 1.42x. Ворота 1.40
  лежат **внутри** диапазона 1.36–1.42, а не под ним. Под гейтом это **1.216–1.236x**.

##### 4c4. dtype в хопах: проверено, промоушена нет — отсюда 1.11x, а не 1.28x

Самый дорогой открытый вопрос закрыт измерением. Эпилог пролога переведён в fp16 (тот же `int32`-аккумулятор,
де-квантование и PReLU в half) и измерен на реальной форме:

| эпилог | мс на хоп | 16 хопов |
|---|---|---|
| fp32 (как есть) | **2.519** | 40.3 |
| fp16 | 2.611 | 41.8 |

**fp16 не быстрее — на 3.6% медленнее**, и коды расходятся на 0.061% (1 код). То есть «6.4 мс dtype-промоушена в
хопах» **не существует**: 0.32 мс из квантования головы — единственная реальная часть, и она уже учтена.
Отсюда проекция не 1.28x, а **1.105x**.

##### 4c5. Пересчёт от устойчивого состояния

Фиксированная часть окна: `upload 1.86 + model-call 1.0 = 2.86 мс`.

| сценарий | GPU | окно | против fp16 51.06 |
|---|---|---|---|
| сейчас | 63.7 | 61.7 | **0.828x** |
| измеренные GPU-фиксы (квантование → пол трафика, голова+хвост → пол) | 48.2 | 46.2 | **1.105x** |
| потолок на базе TRT int8 conv (1.68 мс/хоп, продемонстрировано на этой форме) | — | 31.8 | **1.60x** |
| потолок на базе GEMM-пика (1.037 мс/хоп, другая форма) | — | 21.6 | **2.37x** |

**Впервые полоса проектируется выше 1.0 в продуктовых числах** (1.105x только на измеренных фиксах), а потолок —
**1.6–2.4x**, не «1.06x» из §5. Прежний 1.06x считался из времени синтетического стенда (1.900 мс/хоп),
принятого за пол; это остаётся неверным.

##### 4c3. 33 мс — это JIT первого кадра, размазанный по среднему

Разбор был прав, что 33 мс устойчивы и лежат вне окна вызова. Владелец найден — **первый кадр**:

```
frame 1: upload 4.27  model-call 846.30  final-sync 63.30  | total 91x.x ms
frame 2: upload 1.94  model-call   0.96  final-sync 61.62  | total 64.5
frame 3: upload 1.86  model-call   0.94  final-sync 58.77  | total 61.6
frame 4: upload 1.86  model-call   1.06  final-sync 58.86  | total 61.7
```

`model-call` первого кадра — **846 мс**. **И это не JIT: два прогона в свежих процессах дают одну цифру.**

| прогон | model-call первого кадра |
|---|---|
| 1 | 844.65 мс |
| 2 (кэш Triton тёплый, процесс новый) | **850.38 мс** |

Разница 5.7 мс. Значит 850 мс — **per-video работа**: сборка тела, калибровка на кадрах, сертификация (полный
fp16-форвард + полный int8-форвард + сравнение) и первый запуск каждого ядра в новом процессе. Кэш cubin её не
убирает. Дальше 0.94–1.06 мс на кадр.

**Моё прежнее «на втором прогоне — ноль» отменено:** кэшируется только компиляция, а не работа с видео. Профиль приложения печатает **среднее по 24 кадрам**:

```
(846 + 23 × 61.7) / 24 = 94.7 мс   ← профиль печатает 97.04
```

То есть «33 мс накладных» — это 846 мс компиляции, размазанные по 24 кадрам. На рендере в 1000 кадров они дают
**0.85 мс на кадр**, а на втором прогоне на той же машине — ноль: кэш Triton (`~/.triton/cache`) переживает
запуск.

**Что это меняет в продуктовой таблице разбора:** устойчивое состояние — int8 **61.7** против fp16 **51.06**, то
есть **0.828x**, а не 0.530x. Отношение 0.530x получалось именно из среднего, отравленного первым кадром.
Прогнозы 0.684 / 0.799 / 1.210 тоже считались от 97.04 и подлежат пересчёту от 61.7.

**И выходной тензор ни при чём** — формат совпадает с fp16-путём:

```
out dtype torch.float16  shape (1, 3, 2160, 3840)  contig True  stride (24883200, 8294400, 3840, 1)
```

гипотеза «fp32/NHWC на границе» не подтверждается.

##### 4c2. Таймлайн: 35 мс не в вызове, отношение 0.798x

Host-таймстемпы внутри `_call_single`, без синхронизаций, оба пути в одном и том же коде:

| | model-call (хост) | final-sync (дренаж GPU) | **total** |
|---|---|---|---|
| fp16-движок | **0.36 мс** | 48.85–52.35 | **51.06–54.68 мс** |
| int8-тело | **1.05 мс** | 61.10–61.20 | **64.02–64.13 мс** |

Отсюда:

- **хост-время вызова: 1.05 мс против 0.36**, то есть +0.7 мс, а не +35;
- **стена к стене: 51.06 / 64.02 = 0.797x**;
- **GPU к GPU: 49.80 / 63.70 = 0.782x**.

Два независимых инструмента дают одно и то же. Значит **«35 мс накладных» не существует**: это артефакт сравнения
поля `inference` из профиля приложения (97.04 мс) с окном, которое реально занимает вызов (64.02). Два инструмента
согласны, поле профиля — выброс, и оно шире на обеих сторонах (у fp16: 51.43 в профиле против 51.06 в таймлайне —
там разница мала; у int8: 97.04 против 64.02).

**Что это отменяет:**

- «38 мс — наши» из §4 — отменено;
- аргумент «хост — шлюз, только хостовые фиксы дают 0.800x, все GPU-фиксы 0.681x» — отменён вместе с исходным
  числом: хостовой надбавки в 33 мс нет;
- прогноз «оба фикса дают 1.213x» — пересчитывается только по GPU-части.

**Что остаётся верным:** тело медленнее на **1.25x** на обоих инструментах, и весь разрыв — GPU-работа (63.7 против
49.8), а не обвязка. Оптимизировать надо ядра, а не вызов.

##### 4c. Три инструмента на один вопрос: 35 мс — не запуски и не мои события

**Учёт сходится.** Когда сумма фаз и полное время снимаются в одном кадре, разрыв **0.0**
(`SUM 63.7  BODY-TOTAL 63.7`, три кадра подряд). Прежние «2.8 мс» — сравнение разных прогонов, а не сегментов.

**16 запусков Triton стоят 0.37 мс хоста**, а не 35:

```
int8 host: 16 launches cost 0.37 ms of HOST (0.023 ms each); whole hop loop 0.38 ms
int8 host: 16 launches cost 0.33 ms of HOST (0.021 ms each); whole hop loop 0.34 ms
```

Это ровно то, что предсказывал разбор: 2.19 мс на запуск было бы в 40–400 раз выше нормы для скомпилированного
лаунчера. Значит «16 Python-запусков» — имя на числе, которое его не выделяет, и оно снято.

**И мои события тоже не виноваты.** С `RVE_INT8_NO_EVENTS=1` (все события заменены null-объектом, ни одного
`elapsed_time`, который ждёт предыдущий кадр) инференс **96.63 мс** против 97.42 с инструментацией. Стоимость
инструментации — 0.8 мс, а не 33.

Итого три независимых ответа: тело на GPU **63.7 мс**, полное окно инференса **96.6 мс**, и разница **33 мс**
не является ни ядрами тела, ни запусками, ни инструментацией. `set_sync_debug_mode("warn")` не выдал ни одного
предупреждения о неявной синхронизации внутри тела.

**Следующий замер назван:** host-таймстемпы по этапам внутри `_call_single` (загрузка кадра → `get_frame_tensor`
→ вызов тела → построение `Frame`) без синхронизаций, чтобы увидеть, что занимает 33 мс внутри окна. Пока это
не измерено, «накладные int8 35 мс» остаётся числом без владельца.

##### 4d. Механизм 18 мс: `cudaMalloc` опровергнут

**`num_alloc_retries = 0` по кадрам** (`reserved 2194 МБ` при `peak 1779` — в пуле 415 МБ готового резерва).
Значит аллокатор ни разу не вытеснял кэш и не звал `cudaMalloc`: история про синхронизацию при аллокации
**мертва**, и 18 мс ожидания надо объяснять иначе.

Дамп PReLU по трём путям:

```
in dtype=float16  weight dtype=float16  shape=(64,)
slope_half[:4] = [-0.3728, 0.9473, 0.9238, 0.1449]
weight[:4]     = [-0.3728, 0.9473, 0.9238, 0.1449]   <- идентичны
out dtype=float16 contiguous=True
```

То есть **входы всех трёх путей одинаковы** (те же значения, тот же dtype, та же ось). Гипотеза «оба новых пути
берут не тот offset» не подтверждается. Значит расхождение 2.15278% сидит не в наклонах и не в конверсии, а в
остальной части того варианта кода (в переиспользуемом буфере), и причина не установлена.

**Что это меняет для плана:** −18 мс в квантовании не доказаны как аллокация; их владелец по-прежнему
неизвестен, и «70.8 → ~53» остаётся проекцией.

##### 4c. Голова и хвост — не фиксированные расходы, и это отменяет §5

| сегмент | трафик | пол @288 ГБ/с | измерено | во сколько раз |
|---|---|---|---|---|
| голова (read 12.4 + write 265 МБ) | 277 МБ | 0.96 мс | 4.6 | **4.8x** |
| хвост (read 265 + 12.4 + write 49.8 МБ) | 327 МБ | 1.14 мс | 7.8 | **6.8x** |

Это 12.4 мс против 2.1 мс пола — **10.3 мс лежат вне тела**, ровно там, куда указывал пункт «свои голова/хвост».

Сумма названных фиксов: `70.8 − 18.6 (квантование) − 6.4 (dtype в хопах, если он там) − 9.8 (голова+хвост) =
36.0 мс` → **1.30x**. А пол по вычислениям `16 × 1.037 + 1.6 + 2.6 = 20.8 мс` → **2.26x**. Ядро идёт на 43%
измеренного int8-пика, движок fp16 — на 75% своего: запас в относительной эффективности **вдвое**, и он в
перекрытии, а не в dot'е.

##### 4d. Первая редакция этого раздела (для истории)

Разрыв атрибутирован событиями CUDA **без `synchronize()` внутри кадра** (события предыдущего кадра читаются
на следующем, так что окно не ждёт чужую работу). Стабильное состояние, 24 кадра:

| фаза | приложение (GPU, события) | харнесс | разница |
|---|---|---|---|
| голова | 4.6 мс | 3.24 | +1.4 |
| **квантование** | **20.3 мс** | ~3.6 (prelu 2.0 + ядро 1.6) | **+16.7** |
| 16 хопов | 38.3 мс | 35.2 (2.198 × 16) | +3.1 |
| хвост | 7.8 мс | 4.7 | +3.1 |
| **тело, GPU** | **71.0 мс** | 48.55 | **+22.4** |
| стена вокруг вызова | **108.77 − 71.0 = 37.8 мс** | — | — |

То есть: **тело на GPU стоит 71 мс, а не 96.9**; ещё **37.8 мс** — это стена вокруг вызова (передача кадра,
синхронизация, работа писателя), которая к телу не относится. И из 22.4 мс превышения над харнессом **16.7
приходится на одно квантование**.

**Что проверено и опровергнуто для этого сегмента:**

1. **артефакт `synchronize()`** — нет, события без sync дают те же 20.3;
2. **аллокация 133 МБ внутри кадра** (`cudaMalloc` синхронизирует) — предвыделение буферов не изменило ничего
   (20.3 → 20.3);
3. **`dtype`-промоушен в ядре** — реален, но стоит 17% (1.92 → 1.60 мс при сетке в fp16), не 20 мс;
4. **раскладка/транспонирование** — в изоляции то же ядро на том же тензоре (1×64×1080×1920, непрерывный
   fp16, те же `strides=(132710400, 2073600, 1920, 1)`, тот же `BLOCK=256`) стоит **1.92 мс**.

Значит квантование в приложении медленнее **в 10 раз** при идентичных входе, форме, раскладке и параметрах
запуска. Это не математика и не layout. Остаётся окружение: поток/контекст приложения, состояние памяти или
взаимодействие с движком, который держит 115 МБ и свой execution context. **Следующий замер назван:** вызвать
это же ядро в процессе приложения на том же тензоре, обернув событиями только его (без prelu) и напечатав
`torch.cuda.memory_allocated()`, чтобы отделить «ядро медленное» от «ядро ждёт».

**Найдено, где именно ждёт квантование.** То же ядро, вызванное **отдельно в процессе приложения** на том же
тензоре, стоит **1.60 мс** (5 вызовов, события, `allocated 893 MB / reserved 2194 MB / peak 1779 MB`). Значит
ядро не медленное — оно **ждёт**. В фазе между событиями стоит `F.prelu`, который аллоцирует **265 МБ на кадр**;
`cudaMalloc` синхронизирует устройство, и ожидание приписывается следующему ядру. Это и есть те ~18 мс.

**Две попытки убрать эту аллокацию — обе отклонены по точности:**

| вариант | картинка | вердикт |
|---|---|---|
| `F.prelu` + конверсия (как есть) | **0.00823%** | принято |
| PReLU, влитый в Triton-ядро конверсии | **2.15278%** | отказ |
| `torch.where(..., out=переиспользуемый буфер)` | **2.15278%** | отказ |

Оба варианта дают **одно и то же** 2.15278% — значит расходится не конверсия и не аллокация, а сам способ
применения PReLU, и причина не установлена. В изоляции fused-ядро совпадает с двухпроходным на 99.6% кодов
(расхождение в 1 код), так что эффект тонкий и требует отдельного разбора. **Правильность важнее:** оставлен
путь, который сертифицируется на 0.00823%.

Потенциал, если аллокацию убрать корректно: тело 70.8 → ~53 мс GPU, стена 109 → ~91 мс, то есть 0.52x против
движка. Вердикт не меняется, но это ещё −18 мс из уже измеренных.

**Что это меняет:** 37.8 мс стены вокруг вызова — не собственность тела, и вычитать их из «вины int8» нельзя.
Тело на GPU: **71.0 мс против 46.92 мс движка = 0.66x**, а не 0.484x. Вывод (не окупается) не меняется, но
величина проигрыша была завышена.

##### 4a. Прежняя (неверная) разбивка — сохранена для истории

Редакция 1 утверждала: квантование головы 27.38 мс, накладные 30 мс. **Оба числа — артефакт моей
инструментации.**

В изоляции то же NCHW-ядро на том же тензоре (1×64×1080×1920, непрерывный fp16) стоит:

| вариант | BLOCK 128 | BLOCK 256 | BLOCK 512 |
|---|---|---|---|
| сетка fp32 (как сейчас) | 1.93 мс | 1.92 мс | 1.71 мс |
| сетка fp16 | 1.72 мс | **1.60 мс** | 1.60 мс |
| только трафик, без арифметики | 1.92 мс | 1.92 мс | 1.93 мс |

То есть квантование — **1.6–1.9 мс, это трафик** (398 МБ / 288 ГБ/с = 1.38 мс), а не 27.4 и не «20x пола».
Механизм, который я назвал (транспонирование), **не подтверждён**. `dtype`-промоушен существует и стоит ~17%
(1.92 → 1.60 при сетке в fp16), но это не 27 мс.

Почему 27.4: я мерил сегменты через `torch.cuda.synchronize()` **внутри** кадра, когда рядом работает поток
писателя и NVENC — каждый sync ждал чужую работу, и я приписал её квантованию. Отсюда же и «30 мс без
владельца»: это, скорее всего, тот же эффект.

**Честная формулировка:** разрыв между 96.88 (рендер) и 48.55 (харнесс) **не атрибутирован**. Он реален
(2.0x), но чей — тела, синхронизаций приложения или его собственных накладных — этим методом не измерено.
Никакие «−26 мс дают ~71» из этого не следуют, и рекомендация §6 редакции 1 ставилась на неизвестной базе.

---

##### 5. Потолок: 1.06x — это предел текущего кодгена, а не конструкции

Редакция 1 считала потолок как `16 × 1.900 + 13.3 = 43.7 → 1.06x`. Ошибка логическая: **1.900 — время
текущего ядра на синтетическом стенде, а не пол.** Реальное ядро в харнессе — 2.198 мс, в приложении — 2.57.

Пол надо считать из пика:

```
на хоп:  2·1920·1080·64·64·9 = 152.9 GFLOP
пик int8 (измеренный):        147.4 TOPS
пол на хоп:                   152.9 / 147.4 = 1.037 мс
16 хопов:                     16.6 мс
+ нетелесная работа 13.3 мс:  29.9 мс  →  46.29 / 29.9 = 1.55x
```

И это согласуется с собственными измерениями: сбор без dot = 1.060 мс, с dot = 1.900 — сумма 2.097 > 1.900,
значит перекрытие уже частичное; при полном перекрытии `max(1.060, 1.037) = 1.060` мс на хоп:

```
16 × 1.060 + 13.3 = 30.3 мс  →  46.29 / 30.3 = 1.53x
```

Где сейчас ядро: 2.198 мс = **69.6 TOPS = 47% пика**; движок приложения на 46.29 мс = **53.6 TF/s = 75%**
своего пика. Значит int8-ядро относительно пика **вдвое хуже**, чем TRT относительно своего, и это **запас, а
не потолок**.

**Исправленный вывод:** доступно **≈1.5x** против движка приложения, но для этого нужен другой кодген
(перекрытие памяти и dot'а, регистровое давление). «1.06x» — предел того, что даёт текущий набор плиток, и
именно так его надо называть. Это ровно то, что было записано про регистровый фронтитр двумя раундами раньше.

Три популяции времени на хоп, которые нельзя смешивать:

| где | мс/хоп | TOPS | % пика |
|---|---|---|---|
| синтетический стенд | 1.900 | 80.5 | 55% |
| харнесс | 2.198 | 69.6 | 47% |
| приложение | 2.57 | 59.5 | 40% |

Разрыв 1.900 / 2.198 не объяснён; 1.900 в расчёт потолка попадать не должен был.

---

##### 6. Что построено

| файл | что делает |
|---|---|
| `backend/src/pytorch/Int8Kernels.py` | пять Triton-ядер: пролог, три конверсии, де-квантование |
| `backend/src/pytorch/Int8Body.py` | clamp → голова fp16 → 16 int8-хопов → де-квантование → хвост модели → остаток |
| `backend/src/utils/Int8Probe.py` | проба: решает по **композиции**, до первого кадра |
| `backend/src/RenderVideo.py` | проба один раз в `setupUpscale`, вердикт пробрасывается в апскейлер |
| `backend/src/pytorch/UpscaleTorch.py` | `_int8_body()`: строит, калибрует, сертифицирует, обслуживает или отказывает |
| `scripts/int8_{pipeline,timing,app_gate,graph}.py` | офлайн-конвейер, сегменты, ворота, CUDA-граф |
| `harness/docs/claims-int8-stage{1,2,4,5}.txt` | утверждения, все проходят `claim_check.py` |

Покрытие: 2x-Futsuu (64 кан., 2x) — работает; realesr-animevideov3 (64, 4x) — работает, но медленнее;
Archivist_AntiLines (48 кан.) и 2x_ModernSpanimationV1 (не Compact) — **не покрыты структурно**.

**4x-модель — в области, а не вне её.** В редакции 1 она выведена «потому что её дефицит мал на фоне общего
проигрыша». Это перевёрнутая логика: если проигрыш не зависит от модели, то и починка его починит 4x-модель.
Её дефицит (+1.70 мс) мал именно потому, что общая причина общая. Сужать область тут не за что.

---

##### 7. Одиннадцать неполадок (самая ценная часть — и она честная)

| # | что случилось | чем поймали | чем кончилось |
|---|---|---|---|
| 1 | проба читала искажённые кадры: клипы 4K, ридер резал первые `w*h*3` байт | 4.1% «ошибки движка» на всех тайтлах против 0.0086% при нормальном чтении | `-vf scale=W:H`; P(pass)=0.6 |
| 2 | фильтр гейта не перенесён: в композицию попадали склейки | 99.7% композиции при пике 255 | фильтр перенесён, вердикты совпали |
| 3 | сетки калибровки в fp16 → переполнение множителя в INT_MAX | картинка 99.58% при нормальном времени | `.float()` на сетках (трижды) |
| 4 | округление отрицательных «от нуля» | сравнил три сдвига на одних числах | `+half` для обоих знаков: 1.569% → 0.00043% |
| 5 | отсутствовал bias свёрток (все 18, \|bias\| 2–4 при активациях ~0.5) | `module(вход)=1.5006` против `conv2d без bias=3.8041` | bias в эпилог: 1.569% → 0.789% |
| 6 | пинг-понг из двух буферов писал во вход | CUDA-граф не сходился с eager на 35 млн кодов | свежий приёмник на хоп |
| 7 | **эталон считался после `calibrate`** | «2.9x хуже» не сходилось с 0.03% | эталон до калибровки: сборка **в 1.8 раза лучше** симуляции на 4K; **отменяет вывод о недостижимости** |
| 8 | кросс-инструментальное отношение 35.85/56.82 из двух процессов | разбор | 0.820x вместо 0.631x |
| 9 | fp32 slope промотировал тайл: 12.8 мс против 1.5 | свип блоков/типов | fp16 slope + BLOCK 256 → 1.52 мс |
| 10 | `static_range` разворачивал 9 копий тела цикла | диагностика «память или dot» | `tl.range(num_stages=1)`: 2.92 → 2.198 мс |
| 11 | голова в приложении — `PadInputConv` (3→8), не `Conv2d` | лог: `expected input[1,3,...] to have 64 channels` | голова берётся как `model.body[0]` любого типа |

Общий вывод: **каждый раз, когда число расходилось с ожиданием, виноват был прибор, а не модель** — включая
два случая (№4, №7), где на основании сломанного прибора был сделан отрицательный вывод и потом отменён.
Плюс №12, найденный при этом разборе: **сегментные замеры через `synchronize()` внутри кадра** (§4).

---

##### 8. Порядок работ (заменяет §6 редакции 1)

1. **Продуктовый вопрос закрыт отрицательно** (§1): тело инертно под дефолтом, а его включение требует
   отключить гейт, который стоит 3.3x. Пока это так, оптимизировать ядро не для чего. Если полоса нужна —
   сначала ветка тела **под гейтом**: переиспользованный кадр стоит 0 инференсов, и тело должно обслуживать
   только промахи гейта, не отменяя сам гейт.
2. **Атрибутировать разрыв 96.88 против 48.55** (§4) без `synchronize()` внутри кадра — CUDA-событиями на
   границах и внешним таймером. 2x, чей владелец неизвестен, блокирует планирование.
3. **Кодген ядра**: 47% пика против 75% у TRT — вот где 1.5x (§5). Это перекрытие памяти и dot'а, не
   «переписать layout» и не dtype (dtype стоит 17%, проверено).
4. Уточнить точность рендера на **разнесённой выборке** и назвать худший кадр (§2).
5. Починить агрегацию в первой строке §3 (0.953 против 0.962) — одного числа с подписью достаточно.

---

##### 9. Что изменилось против редакции 1

| # | было | стало |
|---|---|---|
| 1 | «доставлено в настоящий рендер», 0.484x как итог | доставлено **в конфигурации, которую продукт не поставляет**; под дефолтом тело инертно; включение требует отказаться от гейта в 3.3x |
| 2 | «потолок 1.06x» | 1.06x — предел текущего кодгена; из пика получается **≈1.5x** |
| 3 | «квантование 27.4 мс, главный дефект, −26 мс доступно» | в изоляции **1.6–1.9 мс**, это трафик; 27.4 — артефакт `synchronize`; механизм не подтверждён |
| 4 | «30 мс накладных, не понятно чьи» | то же, плюс признание, что и сегментная разбивка этим методом недействительна |
| 5 | 0.962x без пояснения | 0.953x (отношение медиан) против 0.962x (медиана отношений) |
| 6 | точность 0.07592% как итог рендера | **один кадр** одного клипа, не худший из выборки; пик 24 назван |
| 7 | 4x-модель вне области | **в области**: общая причина чинится общим исправлением |
| 8 | «2x против torch достигнуто» | eager никто не поставляет; это не цель и не предъявляется |

---

##### 10. Воспроизведение

```bash
PY=/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/python/python/bin/python3
B=/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2

#### ВНИМАНИЕ: без --no_temporal_reuse тело не запускается вообще (гейт включён по умолчанию)
cd backend && $PY rve-backend.py --input clip1080.mkv --output out_body.mkv \
  --ffmpeg_path $B/bin/ffmpeg --backend tensorrt \
  --upscale_model "$B/custom_models/2x-Futsuu-Anime(1).pth" --no_temporal_reuse --overwrite

#### тот же клип без тела
RVE_INT8_BODY=off $PY rve-backend.py ... --output out_fp16.mkv

#### утверждения
$PY harness/scripts/claim_check.py harness/docs/claims-int8-stage4.txt
```

<a id="evidence-docs-wsp-verified-int8-tactics-txt"></a>

### `harness/docs/wsp/verified/int8-tactics.txt`

#### The int8 body's per-convolution rate (E33, first pass)

Measured 2026-11-17 with `harness/scripts/probe_int8_tactics.py` on the plan the 2026-11-16 lane left behind
(`c2x_int8_fp16heads.plan`), same session, 20 executions x 3 rounds, `trt.IProfiler` for the per-layer
times.

The reproduction is exact: 67.67 ms against the recorded 64.3-70.4, 84 layers, 22 reformat and 19
PointWise against the recorded 19-22 and 19. What the per-layer split adds is the third number: the
convolutions run at a **median 79.2 TOPS** (min 76.2, max 119.7 across 18 conv layers), not the 91 the
queue's ceiling arithmetic assumes. At 79 TOPS the 16-conv body floor is 30.9 ms instead of 26.9.

Not answered by this file: whether any tiling level or tactic source raises it - that is a build per
configuration, and each configuration now needs >=3 builds (the C=32 build is bimodal, see
`prelu-forms.txt`).

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/int8-tactics.txt \
       --records harness/docs/wsp/verified/records/int8-tactics.jsonl --bars harness/docs/claims-bars.json

@claim R001 67.67 ms "plan repeats [826cec6405e36217] n=3" measured
@claim R002 79.2 TOPS "body convolutions [68e9d0cf981c5027] n=18" measured

<a id="evidence-docs-wsp-verified-int8-block-txt"></a>

### `harness/docs/wsp/verified/int8-block.txt`

#### The int8 block, re-measured 2026-11-16 with the repo's own kernel.
#
#### Ledger (line 1019): '1.91 ms and 80 TOPS, measured, not projected', with no instrument named
#### (the inventory flagged it as the load-bearing number of the int8 section and the weakest
#### provenance in the range). Re-run: harness/scripts/int8_block_probe.py, three separate invocations,
#### torch.cuda.Event over 20 iterations, one GPU job at a time.
####   runs: 1.840 / 1.779 / 1.841 ms  -> 83.1 / 86.0 / 83.0 TOPS
####   correctness: max |diff| vs the int32 reference = 0, share of differing outputs 0.00%
#### The recorded 1.91-1.93 ms is 3-7% slower than today's session, which is the drift this machine
#### shows between sessions; the shape of the claim survives.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/int8-block.txt \
       --records harness/docs/wsp/verified/records/int8-block.jsonl --bars harness/docs/claims-bars.json

@claim R001 1.84 ms "int8 block launch [ea432364037f4b16] n=3" measured
@claim R002 83.1 TOPS "int8 block launch [ea432364037f4b16] n=3" measured

@below R001 bar_block_time

<a id="evidence-docs-claims-int8-stage4-txt"></a>

### `harness/docs/claims-int8-stage4.txt`

#### Stage 4 claims, 2026-11-16. Run: harness/scripts/claim_check.py harness/docs/claims-int8-stage4.txt
#### The probe reads adjacent pairs from a clip, calibrates on a disjoint half, and reports two statistics
#### separately because they answer different questions.

@bar bar_share 0.1 %      # the acceptance bar: share of pixels whose worst channel is >8 levels off
@bar kill_prob 0.3   x    # P(pass) below this and int8 becomes a narrow opt-in

@claim chainsaw_engine      0.009 %  "a5_chainsaw, 1920x1080, 7 gated pairs, int8(t) vs fp16(t)" measured  @session=stage4
@claim chainsaw_composition 0.010 %  "a5_chainsaw, int8(t-1) vs fp16(t): what the bar actually sees" measured  @session=stage4
@claim adv_chainsaw_comp    0.022 %  "adv_chainsaw, same population" measured  @session=stage4
@claim c10_comp             0.011 %  "c10_p010, same population" measured  @session=stage4
@claim violet_engine        0.057 %  "a5_violet, int8(t) vs fp16(t) - UNDER the bar" measured  @session=stage4
@claim violet_comp          0.170 %  "a5_violet, int8(t-1) vs fp16(t) - OVER the bar" measured  @session=stage4
@claim adv_violet_engine    0.058 %  "adv_violet, same population" measured  @session=stage4
@claim adv_violet_comp      0.165 %  "adv_violet, composition" measured  @session=stage4
@claim pass_rate            0.6   x  "derived: 3 of 5 readable titles pass on composition" derived  @session=stage4
@derived pass_rate "a5_chainsaw, adv_chainsaw and c10_p010 pass; a5_violet and adv_violet fall back = 0.6"

#### The engine's own defect is not the criterion: a5_violet passes it and fails composition.
#### The three passing titles must sit under the bar.
@below chainsaw_composition bar_share
@below adv_chainsaw_comp bar_share
@below c10_comp bar_share
#### The two Violet titles do NOT sit under it, and that is the stage's finding, not a broken claim:
####   a5_violet     engine 0.057 (pass)  composition 0.170 (fail) -> fp16 fallback
####   adv_violet    engine 0.058 (pass)  composition 0.165 (fail) -> fp16 fallback
#### So the engine's own defect is not the criterion; composition is, exactly as the plan warned.
#### P(pass) = 0.6 is ABOVE the 0.3 kill threshold, so the lane survives; there is no 'above'
#### directive, so this is recorded rather than asserted.

#### ---- the body: accuracy first, then the stopwatch (2026-11-16)
#### History, kept because it is why the conversion was reverted: with the fused Triton head conversion the
#### body read 3.53381% at 1080p and 1.27525% at 512. Those codes differ from torch's on 98.34% of elements.
@claim body_before_1080    3.53381 % "Int8Body with the fused Triton head conversion, 1920x1080" measured  @session=stage4
@claim body_before_512     1.27525 % "the same at 512x512" measured  @session=stage4
@claim head_codes_differ   98.34 %   "fused Triton head conversion vs torch, NCHW-to-NCHW, mean offset +16.92" measured  @session=stage4

#### With the head conversion in torch, as the validated recipe does it:
@claim body_share_256      0.00000 % "Int8Body.certify, 256x256" measured  @session=stage4
@claim body_share_512      0.00000 % "Int8Body.certify, 512x512" measured  @session=stage4
@claim body_share_1080     0.00000 % "Int8Body.certify, 1920x1080" measured  @session=stage4
@claim body_vs_engine_share 0.00001 % "Int8Body vs the app's fp16 TRT engine picture, 1920x1080" measured  @session=stage4
@below body_share_1080 bar_share
@below body_share_512 bar_share
@below body_vs_engine_share bar_share

#### The stopwatch, against the app's own cached engine, three interleaved rounds per condition:
@bar gate_ratio 1.40 x
@bar kill_ratio 1.15 x
@bar zero_excess 0.0 ms

@claim engine_idle_ms   79.86 ms "fp16 TensorRT engine, 1920x1080, NVENC idle" measured  @session=stage4
@claim body_idle_ms     60.41 ms "Int8Body with the NHWC-aware head conversion, same rounds" measured  @session=stage4
@claim ratio_idle        1.322 x "derived: engine_idle_ms / body_idle_ms" derived  @session=stage4
@derived ratio_idle "79.86 / 60.41 = 1.322; the three rounds read 1.319 / 1.322 / 1.327"
@claim engine_nvenc_ms  91.65 ms "fp16 engine, h264_nvenc encoding concurrently" measured  @session=stage4
@claim body_nvenc_ms    69.03 ms "Int8Body, same condition" measured  @session=stage4
@claim ratio_nvenc       1.328 x "derived: engine_nvenc_ms / body_nvenc_ms" derived  @session=stage4
@derived ratio_nvenc "91.65 / 69.03 = 1.328; rounds read 1.328 / 1.326 / 1.328"

#### The verdict: the lane PAYS, and misses the 1.40 gate by a measured amount.
@claim kill_excess   -9.13 ms "body_idle_ms - engine_idle_ms/1.15 = 60.41 - 69.45" derived  @session=stage4
@derived kill_excess "60.41 - 79.86/1.15 = -9.13 ms: 9.13 ms FASTER than the kill line, so the lane pays"
@claim gate_excess    3.34 ms "body_idle_ms - engine_idle_ms/1.40 = 60.41 - 57.04" derived  @session=stage4
@derived gate_excess "60.41 - 79.86/1.40 = +3.34 ms SHORT of the 1.40 gate; recorded, not asserted"
@claim fixed_costs_ms 16.1 ms  "head 5.5 + quantise ~2 + dequantise/tail ~8.4: the non-body work the gate's own note blames" derived  @session=stage4
@derived fixed_costs_ms "60.41 - 16.1 = 44.3 ms of int8 body, which alone would be 1.80x against the engine"

#### The plan's kill line is the one that decides whether the lane pays at all. It passes.
@below kill_excess zero_excess
#### The gate is NOT asserted: the measured shortfall is recorded above, because a claim file that fails
#### on a number it has already measured cannot also state the verdict.

#### ---- the 16.1 ms of non-body work, measured and attacked three times
@claim seg_head_ms      5.272 ms "clamp + conv 3->64 + PReLU, 1920x1080" measured  @session=stage4
@claim seg_quant_ms     1.603 ms "quantise NHWC, no copy" measured  @session=stage4
@claim seg_body_ms     43.851 ms "16 int8 hops, 2.74 per launch" measured  @session=stage4
@claim seg_dequant_ms   2.320 ms "dequantise" measured  @session=stage4
@claim seg_tailconv_ms  2.258 ms "tail conv 64->12" measured  @session=stage4
@claim seg_upsampler_ms 0.742 ms "pixel shuffle" measured  @session=stage4
@claim seg_residual_ms  1.708 ms "nearest upsample + add" measured  @session=stage4
@claim seg_sum_ms      57.754 ms "sum of the segments; the full call is 60.115, so 2.4 ms is launch and allocation" measured  @session=stage4
@claim tail_nhwc_gain  -0.2 ms "tail dequantise staying NHWC: 60.2 -> 60.4 ms, inside the noise" measured  @session=stage4
@claim prelu_fuse_loss  9.0 ms "PReLU fused into the quantise kernel: 60.4 -> 69.4 ms, and the fused kernel alone is 12.816 ms" measured  @session=stage4
@claim prelu_passes_ms  3.632 ms "the two passes it replaced: 2.029 + 1.603" measured  @session=stage4
@claim contig_head_gain -0.3 ms "contiguous head input: 5.286 -> 5.557 ms" measured  @session=stage4

#### ---- the PReLU fusion fixed, and the gate CLEARED (2026-11-16)
@claim prelu_fuse_fp16_ms 1.522 ms "fused PReLU+quantise, fp16 slope, BLOCK 256" measured  @session=stage4
@claim prelu_fuse_fp32_ms 12.818 ms "the same kernel with an fp32 slope and BLOCK 1024" measured  @session=stage4
@claim dequant_block256   1.570 ms "dequantise at BLOCK 256; BLOCK 1024 measured 2.329" measured  @session=stage4
@claim cycle_idle_ms     56.82 ms  "full body call, 1920x1080, NVENC idle" measured  @session=stage4
@claim ratio_idle2        1.411 x  "derived: 80.14 / 56.82" derived  @session=stage4
@derived ratio_idle2 "80.14 / 56.82 = 1.411; rounds read 1.411 / 1.420 / 1.427"
@claim cycle_nvenc_ms    64.24 ms  "full body call with h264_nvenc alongside" measured  @session=stage4
@claim ratio_nvenc2       1.428 x  "derived: 91.72 / 64.24" derived  @session=stage4
@derived ratio_nvenc2 "91.72 / 64.24 = 1.428; rounds read 1.428 / 1.425 / 1.426"
@claim gate_margin       -0.78 ms  "cycle_idle_ms - engine_idle_ms/1.40 = 56.82 - 57.24" derived  @session=stage4
@derived gate_margin "56.82 - 80.14/1.40 = -0.42 ms; against the median 80.5/1.40 = 57.5 it is -0.68 ms: the gate is MET"
#### SUPERSEDED: the 1.40x comparisons above are against the MONOLITHIC engine in my harness (80 ms), and the
#### app does not build that engine. Its own log shows a tiled, head-fused, opt-5 engine at 35.85 ms. The two
#### margins are kept as history and are NOT asserted.
@claim gate_margin_nvenc -1.28 ms "cycled_nvenc - engine_nvenc/1.40, against the monolithic engine" derived  @session=stage4
@derived gate_margin_nvenc "64.24 - 91.72/1.40 = -1.28 ms against the monolithic engine; superseded below"

#### ---- THE GATE, MEASURED IN ONE PROCESS AGAINST THE ENGINE THE APP ACTUALLY BUILDS
#### The first version of this block divided the app's profile-log figure (35.85 ms) by a harness figure (56.82 ms):
#### two processes and two paths (tiled vs whole-frame), i.e. a ratio nobody measured. Both sides were then loaded
#### into ONE process, on the same frame, three interleaved rounds:
@claim app_engine_ms     46.40 ms "the app's own tiled engine (..._opt-5_tiling-full_headpad5_1920x1080), 1080p" measured  @session=stage4
#### SUPERSEDED VALUE, kept and labelled: the body before the loop was rolled.
@claim body_vs_app_unrolled 0.820 x "46.40 / 56.56, the kernel before tl.range: rounds 0.820 / 0.825 / 0.823" measured  @session=stage4
#### THE DELIVERED GATE: the same one-process method, after the rolled loop (the final kernel).
@claim body_ms_app_base  48.55 ms "Int8Body with the rolled loop, same process, same frame, same rounds" measured  @session=stage4
@claim body_vs_app        0.953 x "derived: app_engine_ms / body_ms_app_base - ratio of the two medians" derived  @session=stage4
@derived body_vs_app "46.29 / 48.55 = 0.953 - ratio of medians; the median of the per-round ratios is recorded separately as body_vs_app_median_ratio = 0.962"
@claim body_vs_app_median_ratio 0.962 x "median of the three per-round ratios 0.953 / 0.966 / 0.962" measured  @session=stage4
@claim app_kill_overspend 16.20 ms "body_ms - app_engine_ms/1.15 = 56.56 - 40.35" derived  @session=stage4
@derived app_kill_overspend "56.56 - 46.40/1.15 = +16.20 ms past the kill line"
@claim app_gate_overspend 23.42 ms "body_ms - app_engine_ms/1.40 = 56.56 - 33.14" derived  @session=stage4
@derived app_gate_overspend "56.56 - 46.40/1.40 = +23.42 ms past the 1.40x gate"
@claim body_app_share    0.00000 %  "Int8Body vs the app engine picture, same frames - peak 7" measured  @session=stage4
@claim superseded_ratio   0.631 x   "the cross-instrument ratio (35.85 / 56.82) that this block replaces" measured  @session=stage4

#### The verdict, asserted as the statement that is true: against the engine the app builds, and measured on one
#### instrument, the body is BELOW the kill line - the lane does not pay.
@below body_vs_app kill_ratio

#### ---- 4K: the body LOSES, and the app render never used it (verified from the full log)
@claim engine_4k_idle    186.14 ms "fp16 TensorRT engine, 3840x2160, NVENC idle" measured  @session=stage4
@claim body_4k_idle      240.51 ms "Int8Body, 3840x2160, same rounds" measured  @session=stage4
@claim ratio_4k_idle       0.774 x  "derived: 186.14 / 240.51" derived  @session=stage4
@derived ratio_4k_idle "186.14 / 240.51 = 0.774; rounds read 0.774 / 0.789 / 0.785 - the body is SLOWER at 4K"
@claim engine_4k_nvenc   198.68 ms "fp16 engine at 4K with NVENC alongside" measured  @session=stage4
@claim body_4k_nvenc     259.74 ms "Int8Body at 4K with NVENC" measured  @session=stage4
@claim ratio_4k_nvenc      0.765 x  "derived: 198.68 / 259.74" derived  @session=stage4
@derived ratio_4k_nvenc "198.68 / 259.74 = 0.765; rounds read 0.765 / 0.771 / 0.771"
@claim body_4k_share     0.11579 % "Int8Body vs the engine picture at 4K - also over the 0.1% bar" measured  @session=stage4
@claim render_frames        220 x   "app render, 1 s of a5_chainsaw at 4K, tensorrt backend" measured  @session=stage4
@claim render_ms         120.18 ms  "the same render, ms per frame" measured  @session=stage4
@claim render_inference   70.70 ms  "inference per frame in that render (engine 0.58)" measured  @session=stage4
@claim probe_oom            1 x     "'int8 probe unavailable (CUDA out of memory); int8 stays off' in the render log" measured  @session=stage4

#### ---- the app's REAL engine, from the app's own log: tiled, opt-5, head fused, and twice as fast as the
#### ---- monolithic engine the 1.42x was measured against
@claim app_engine_tiled    1 x     "log: 'tiling-full_headpad5_1920x1080', opt-5, head fused 3->8 channels" measured  @session=stage4
@claim app_inference_1080 35.85 ms "the app's own profile over 24 frames, 1920x1080 -> 3840x2160" measured  @session=stage4
@claim app_frame_ms       45.36 ms "the same profile, frame total" measured  @session=stage4
#### (the cross-instrument version of body_vs_app lived here; recorded once above as superseded_ratio)
@claim probe_admitted      1 x     "log: 'int8 probe: engine 0.002% composition 0.003% peak 14 (7 pairs) -> int8'" measured  @session=stage4
@claim probe_served        0 x     "log: no certification line; the render is tiled and the int8 hook only covers tilesize==0" measured  @session=stage4

#### ---- the kernel lever, tried: 15% back, and the ceiling is now bounded
@claim tap_static_ms    2.919 ms "nine taps fully unrolled (tl.static_range), BM 64 / 2 warps" measured  @session=stage4
@claim tap_rolled_ms    2.198 ms "the same loop rolled (tl.range num_stages=1), BM 128 / 4 warps" measured  @session=stage4
@claim tap_stages2_ms   2.497 ms "pipelined two stages - worse, the gathers are already L2-resident" measured  @session=stage4
@claim dot_floor_ms     1.900 ms "one gather plus nine dots: the tensor-core floor for one launch" measured  @session=stage4
@claim gather_ms        1.060 ms "nine gathers with no dot: the memory path alone" measured  @session=stage4
@claim body_rolled_ms  48.55 ms  "the full body after the rolled loop, 1920x1080" measured  @session=stage4
@claim ratio_rolled     0.962 x  "derived: 46.29 / 48.55 against the app's engine, one process" derived  @session=stage4
@derived ratio_rolled "46.29 / 48.55 = 0.953; rounds read 0.953 / 0.966 / 0.962 - up from 0.820"
@claim kill_overspend2  8.30 ms  "body_rolled_ms - app_engine_ms/1.15 = 48.55 - 40.25" derived  @session=stage4
@derived kill_overspend2 "48.55 - 46.29/1.15 = +8.30 ms still past the kill line"
@claim ceiling_body_ms 30.40 ms  "16 x the measured dot floor: no kernel can go below this without better MMA" derived  @session=stage4
@derived ceiling_body_ms "16 x 1.900 = 30.40 ms"
@claim ceiling_cycle_ms 43.70 ms "ceiling_body_ms + the 13.3 ms of non-body work" derived  @session=stage4
@derived ceiling_cycle_ms "30.40 + 13.3 = 43.70 ms"
@claim ceiling_ratio     1.062 x  "app_engine_ms / ceiling_cycle_ms - the limit of the CURRENT codegen (1.900 ms/hop)" derived  @session=stage4
@derived ceiling_ratio "46.29 / 43.70 = 1.062 using the current kernel's 1.900 ms/hop as if it were a floor - it is not"
#### THE REAL BOUND, from the measured int8 peak rather than from the current kernel:
@claim peak_tops       147.4 TOPS "measured int8 peak used for the bound" measured  @session=stage4
@claim hop_floor_ms    1.037 ms   "152.9 GFLOP / 147.4 TOPS per hop" derived  @session=stage4
@derived hop_floor_ms "2*1920*1080*64*64*9 / 147.4e12 = 1.037 ms"
@claim design_floor_ms  29.9 ms   "16 x hop_floor_ms + 13.3 ms of non-body work" derived  @session=stage4
@derived design_floor_ms "16.6 + 13.3 = 29.9 ms"
@claim design_ratio     1.548 x   "derived: 46.29 / design_floor_ms - what the design could reach with peak-rate dots" derived  @session=stage4
@derived design_ratio "46.29 / 29.9 = 1.548; with full memory/dot overlap max(1.060,1.037) it is 1.53x. ABOVE the kill line"
#### NOT asserted any more: the 1.062x figure is the current codegen's limit, and the peak-based bound is 1.53-1.55x,
#### which is ABOVE the kill line. Claiming "no tuning can reach it" was wrong, so the claim stays and the assertion goes.
@below body_vs_app kill_ratio

#### ---- why the body cannot serve THIS renderer, measured in the app's own run
@claim wrapper_model     0 x     "the render holds an SRVGGNetCompact whose body[0] weight is [64,64,3,3]" measured  @session=stage4
@claim app_supports      0 x     "Int8Body.supports(wrapper.get_model()) returns False" measured  @session=stage4
@claim build_error       1 x     "Int8Body build raises: expected input[1,3,1080,1920] to have 64 channels" measured  @session=stage4
@claim app_head_fused    1 x     "the app pads and fuses the head for its engine (log: 'body.0: 3 -> 8 zeros, padded inside the head')" measured  @session=stage4
@claim render_engine_ms 46.47 ms "the app's inference once temporal reuse is off - matches the 46.4 ms one-process engine" measured  @session=stage4

#### ---- DELIVERED: the body served a real app render (2026-11-16)
#### After building Int8Body against the app's TRANSFORMED module (body[0] is PadInputConv, 3->8, not a Conv2d):
@claim body_built        1 x     "app log: 'int8: built (supports=True)'" measured  @session=stage4
@claim certify_line      0.00823 % "app log: 'self-certification on the first frame: 0.00823% of pixels >8 levels -> serving'" measured  @session=stage4
@claim delivered_frames  24 x    "1920x1080 render, 1 s clip, --no_temporal_reuse (the gate is ON by default, so the body is INERT under it)" measured  @session=stage4
@claim delivered_int8_ms 96.88 ms "app profile with the int8 body serving: 'inference 96.88 ms'" measured  @session=stage4
@claim delivered_fp16_ms 46.92 ms "the same render with RVE_INT8_BODY=off: 'inference 46.92 ms'" measured  @session=stage4
@claim delivered_ratio    0.484 x "derived: delivered_fp16_ms / delivered_int8_ms" derived  @session=stage4
@derived delivered_ratio "46.92 / 96.88 = 0.484 - in the real render the body is 2.07x SLOWER"
@claim delivered_share   0.07592 % "frame 10 of the int8 render vs the fp16 render, 3840x2160 output" measured  @session=stage4
@claim delivered_peak    24 levels "the same comparison" measured  @session=stage4
@claim delivered_median   1.00 levels "the same comparison" measured  @session=stage4
#### The picture passes; the speed does not. Both asserted where they are true.
@below delivered_share bar_share
@below delivered_ratio kill_ratio

#### ---- FINAL STATE, 2026-11-16, after the requant move and the PReLU A/B
#### The profile field "inference" is a mean over frames that includes per-video work (~850 ms), so it is NOT the
#### per-frame cost; the host timeline is. Both windows below are host-timeline totals from one process.
@claim fp16_window_ms  51.06 ms "one process, host timeline, steady state, 1920x1080" measured  @session=stage4
@claim int8_window_ms  46.52 ms "one process, host timeline, steady state, 1920x1080" measured  @session=stage4
@claim quant_phase_ms   4.01 ms "head quantise phase with F.prelu + the conversion kernel" measured  @session=stage4
@claim quant_phase_where_ms 9.91 ms "the same phase with torch.where + torch.mul (my regression)" measured  @session=stage4
@claim body_cert_share  0.00832 % "certification on the first frame, epilogue scheme" measured  @session=stage4
@claim hop_ms           1.753 ms "one int8 hop, requant in the epilogue" measured  @session=stage4
@claim hop_tops         87.2 TOPS "derived from the hop time and 152.9 GFLOP" derived  @session=stage4
@derived hop_tops "152.9 GFLOP / 1.753 ms = 87.2 TOPS against TRT's 91 on this shape"

#### THE RATIO, asserted as a ratio so the arithmetic is checked rather than described
@ratio fp16_window_ms int8_window_ms 1.098 "one process, host timeline, steady state, 1920x1080"
@below body_vs_app kill_ratio

#### ---- accuracy re-shot on the EPILOGUE scheme, in the DEFAULT configuration (gate on)
@claim render_share_epilogue 0.07024 % "int8 render vs fp16 render, frame 10, gate ON, 3840x2160 output" measured  @session=stage4
@claim render_peak_epilogue  26 levels "the same frame" measured  @session=stage4
@claim render_share_prologue 0.07592 % "the earlier number, measured while the prologue still requanted" measured  @session=stage4
@below render_share_epilogue bar_share
#### The probe decides from the SIMULATION, not from these kernels, so its verdicts are unaffected by the move.
@claim probe_independent 1 x "Int8Probe uses _quantised_forward; the requant move cannot change its verdicts" measured  @session=stage4

#### ---- the FUSED head conversion adopted (PReLU + quantise in one pass)
@claim fused_head_ms     1.71 ms "fused PReLU+quantise, 1080p, 64 channels" measured  @session=stage4
@claim separate_head_ms  4.01 ms "F.prelu + conversion, same shape" measured  @session=stage4
@claim int8_window_ms2  42.75 ms "one process, host timeline, steady state, 1920x1080" measured  @session=stage4
@claim body_cert_fused 0.00820 % "certification, fused head, best of the three schemes" measured  @session=stage4
@ratio fp16_window_ms int8_window_ms2 1.194 "one process, host timeline, steady state, 1920x1080, fused head"
@below body_vs_app kill_ratio

#### ---- the probe's proxy is 10x optimistic: its verdicts disagree with the serving kernels
@claim probe_sim_chainsaw  0.010 % "probe simulation, composition, a5_chainsaw" measured  @session=stage4
@claim probe_real_chainsaw 0.116 % "the SAME pairs through Int8Body, composition" measured  @session=stage4
@claim probe_sim_violet    0.084 % "probe simulation, composition, a5_violet" measured  @session=stage4
@claim probe_real_violet   0.377 % "the same pairs through Int8Body" measured  @session=stage4
#### SUPERSEDED: that 11.6x was not the kernels, it was the BODY calibrated on ONE frame while the simulation was
#### calibrated on sixteen. With matching calibration:
@claim probe_real_same_cal_chainsaw 0.006 % "Int8Body calibrated on the same 16 frames as the simulation" measured  @session=stage4
@claim probe_real_same_cal_violet   0.050 % "the same on a5_violet" measured  @session=stage4
@claim kernel_vs_sim_chainsaw 1.71 x "derived: probe_sim_chainsaw / probe_real_same_cal_chainsaw" derived  @session=stage4
@derived kernel_vs_sim_chainsaw "0.010 / 0.006 = 1.71x BETTER than the simulation - the proxy is conservative"
@claim kernel_vs_sim_violet   1.69 x "derived: probe_sim_violet / probe_real_same_cal_violet" derived  @session=stage4
@derived kernel_vs_sim_violet "0.084 / 0.050 = 1.69x better, and both verdicts are int8"
@claim calibration_gap      1 x     "the 0.116% and 0.377% came from single-frame calibration, not from the kernels" measured  @session=stage4

#### ---- spread-frame calibration is now the production path (the probe hands over the frames it decoded)
@claim calib_frames_prod 1 x "Int8Body calibrates on the probe's frames; certification 0.00821%, window 42.61 ms" measured  @session=stage4
@claim comp_single_frame 0.116 % "composition with ONE-frame calibration, a5_chainsaw" measured  @session=stage4
@claim comp_spread_frame 0.006 % "composition with the probe's spread frames, same title" measured  @session=stage4
@claim comp_single_violet 0.377 % "composition with ONE-frame calibration, a5_violet" measured  @session=stage4
@claim comp_spread_violet 0.050 % "composition with spread frames, same title - under the 0.1 bar" measured  @session=stage4
@below comp_spread_frame bar_share
@below comp_spread_violet bar_share

#### ---- F_total, measured: the probe dominates it, and it moves the break-even by 20x
@claim probe_cost_s     64.6 s  "Int8Probe on a 1080p clip, 8 positions x 3 frames, end to end" measured  @session=stage4
@claim build_cost_s     0.85 s  "body build + calibrate + certify, frame-1 model-call" measured  @session=stage4
@claim F_total_s        65.4 s  "derived: probe + build" derived  @session=stage4
@derived F_total_s "64.6 + 0.85 = 65.45 s - the earlier 850 ms figure counted only the build"
@claim save_per_frame_ms 8.45 ms "fp16 window 51.06 - int8 window 42.61" measured  @session=stage4
@claim breakeven_min_now 16.2 min "derived: F_total / save, at a 33.3% render rate and 24 fps" derived  @session=stage4
@derived breakeven_min_now "65500 ms / 8.45 ms = 7746 rendered = 23260 frames = 16.2 min"
@claim breakeven_min_ht   5.6 min "the same after the head+tail fix (window 26.8)" derived  @session=stage4
@derived breakeven_min_ht "65500 / 24.26 = 2698 rendered = 8102 frames = 5.6 min"
@claim net_24min_now    +31.8 s "a 24-minute episode now: saves 97.2 s, costs 65.4 s" derived  @session=stage4
@derived net_24min_now "11520 rendered frames x 8.45 ms = 97.2 s against F 65.4 s"
@claim net_24min_ht    +213.7 s "the same episode after head+tail" derived  @session=stage4
@derived net_24min_ht "11520 x 24.26 ms = 279.2 s against 65.4 s"

#### ---- composition at the depth the GATE serves, not at depth 1
@claim comp_d1_chainsaw 0.002 % "composition depth 1, a5_chainsaw, 7 gated pairs" measured  @session=stage4
@claim comp_d2_chainsaw 0.390 % "composition depth 2, 6 pairs where BOTH steps clear the gate" measured  @session=stage4
@claim comp_d1_violet   0.104 % "composition depth 1, a5_violet" measured  @session=stage4
@claim comp_d2_violet   0.166 % "composition depth 2, same filter" measured  @session=stage4
@claim depth_ratio_chainsaw 195 x "derived: comp_d2_chainsaw / comp_d1_chainsaw" derived  @session=stage4
@derived depth_ratio_chainsaw "0.390 / 0.002 = 195x - depth, not quantisation, is what breaks the bar"
@claim depth_ratio_violet 1.6 x "derived: comp_d2_violet / comp_d1_violet" derived  @session=stage4
@derived depth_ratio_violet "0.166 / 0.104 = 1.6x"
#### The probe now decides at the worst depth the gate may serve, which is the whole point.
@claim verdict_depth_worst 2 x "the probe's verdict uses max(d1, d2); at depth 1 both titles would pass" measured  @session=stage4
@below comp_d2_chainsaw bar_share
@below comp_d2_violet bar_share

#### ---- the probe's verdicts with the depth-2 gate filter, verified on both titles
@claim verdict_chainsaw 0 x "probe verdict on a5_chainsaw at the served depth: worst 0.390% > 0.1% -> fp16" measured  @session=stage4
@claim verdict_violet   0 x "probe verdict on a5_violet: worst 0.166% > 0.1% -> fp16" measured  @session=stage4
@claim probe_bug_c2   1 x "the first version of this change crashed with NameError on c2; both probes reported 'probe failed'" measured  @session=stage4

<a id="evidence-docs-wsp-verified-streams-txt"></a>

### `harness/docs/wsp/verified/streams.txt`

#### Two streams do not overlap on this card (task C, strips)

Measured 2026-11-17 with `harness/scripts/probe_strips.py`: one plan deserialized into independent engines with
independent contexts, one CUDA stream each. The reported value is `concurrent / serial` per-frame time,
so **1.00 means no overlap** (the pair takes twice as long as one frame) and 0.50 would be perfect
overlap.

Two plans, three interleaved rounds each: a split fp16 plan (111 layers, 87 ms solo) and a fused one
(24 layers, 54 ms solo). Both read **1.001x**, with SM at 100.0% and 155-162 W of the card's 165 W cap
during the concurrent rounds. A third context on the fastest plan changes nothing (1.001x).

So the strips proposal cannot pay its halo tax (1.067x at two strips): the second stream contributes
1.00x of throughput. The card is power-limited at full SM occupancy, so there is no idle lane.

Instrument history, kept because both failures produced plausible numbers: contexts' buffers were
freed on return (dangling addresses, a 50 ms plan "ran" in 0.07 ms), and CUDA events recorded outside
the stream context timed the host. The current instrument is wall clock around one synchronize; it
reads ~6% above `bench_plan.py`'s events on the same plan, which is enqueue overhead.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/streams.txt \
       --records harness/docs/wsp/verified/records/streams.jsonl --bars harness/docs/claims-bars.json

@claim R001 1.001 x "rounds [a4c54229b5a868cd] n=3" measured
@claim R002 1.001 x "rounds [a4c54229b5a868cd] n=3" measured

<a id="evidence-docs-wsp-verified-pyramid-budget-txt"></a>

### `harness/docs/wsp/verified/pyramid-budget.txt`

#### The L2 pyramid's arithmetic (E29 phase 1)

Measured 2026-11-17 by `harness/scripts/pyramid_budget.py`, CPU only, no GPU job. Reproduces the queue's own
decomposition (8.49 GB logical traffic, 19.7 ms DRAM at 1/3 absorption, 33.1 ms fp16 compute floor,
56.8 ms measured) and adds the size the queue also asked about, 4K.

The finding is not the buffer size - 7.5 MB at 1080p and 15.0 MB at 4K both fit 32 MiB of L2 - but that
at BOTH sizes the fp16 compute floor (33.1 / 132.2 ms) sits above the unabsorbed DRAM time (29.5 /
118.0 ms), so L2 absorption cannot raise the roofline. 1080p: 56.8 ms measured against a 33.1 ms floor
= 1.72x whatever L2 does.

The 1/3 absorption is an assumption carried from one kernel's ncu sector-hit rate; the records below
declare it as an assumption instead of folding it into the ceiling.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/pyramid-budget.txt \
       --records harness/docs/wsp/verified/records/pyramid-budget.jsonl --bars harness/docs/claims-bars.json

@claim R001 7.5 MB "boundaries [6a025f8cf13027c9] n=16" assumed
@claim R002 29.5 ms "boundaries [6a025f8cf13027c9] n=16" assumed
@claim R003 33.1 ms "boundaries [6a025f8cf13027c9] n=16" assumed
@claim R004 56.8 ms "boundaries [6a025f8cf13027c9] n=16" assumed
@claim R005 1.716012 x "boundaries [6a025f8cf13027c9] n=16" derived
@claim R006 15.0 MB "boundaries [6a025f8cf13027c9] n=16" assumed
@claim R007 118.0 ms "boundaries [6a025f8cf13027c9] n=16" assumed
@claim R008 132.2 ms "boundaries [6a025f8cf13027c9] n=16" assumed

<a id="evidence-docs-wsp-verified-cone-budget-txt"></a>

### `harness/docs/wsp/verified/cone-budget.txt`

#### The cone against the cylinder (E36 phase 1)

Measured 2026-11-17 by `harness/scripts/cone_budget.py`, CPU only. Convention: layer L (1..16) reads the changed
region dilated by `17-L` px; the cylinder gives every layer 18. The 1-D control the ledger already
recorded (2 px band, cylinder 16*38*L against cone 304*L) must print 2.00x and does, so the convention
is the one the earlier number used.

Ratios at the radii the review named reproduce (2.54x at r=5, 2.12x at r=10 against 2.6x and 2.16x).
The time bound is `max(work, 16 launches at 5 us)` and the launch term binds below r=20, which is why
the useful statement is "0.08-0.27 ms instead of a 56.8 ms frame", not the ratio.

Not checked: whether a region-level certificate can exist at all - the gate's veto is frame-level and
the seams are the artifact class that killed tile reuse at 0.287%.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/cone-budget.txt \
       --records harness/docs/wsp/verified/records/cone-budget.jsonl --bars harness/docs/claims-bars.json

@claim R001 18.0 count "body layers [643761b71960b77b] n=16" assumed
@claim R002 35344.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R003 13904.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R004 2.542002 x "layers [b64185a2e7bb5a1c] n=16" derived
@claim R005 0.08 ms "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R006 51984.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R007 24464.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R008 2.124918 x "layers [b64185a2e7bb5a1c] n=16" derived
@claim R009 0.08 ms "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R010 94864.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R011 55184.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R012 1.719049 x "layers [b64185a2e7bb5a1c] n=16" derived
@claim R013 0.094 ms "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R014 219024.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R015 155024.0 count "layers [b64185a2e7bb5a1c] n=16" assumed
@claim R016 1.412839 x "layers [b64185a2e7bb5a1c] n=16" derived
@claim R017 0.265 ms "layers [b64185a2e7bb5a1c] n=16" assumed

<a id="evidence-docs-wsp-verified-int8-window-txt"></a>

### `harness/docs/wsp/verified/int8-window.txt`

#### The int8 window at 1920x1080, ungated: 37.5 ms = 26.6 fps, and the picture that does not hold

Measured 2026-11-17 with `harness/scripts/int8_pipeline.py`, one GPU job at a time, CUDA events (the same
instrument `bench_plan.py` uses), 8 executions per configuration, one process.

##### The number the objective asked for

| configuration | window | fps |
|---|---|---|
| fp16 TensorRT plan, same process, same events | **53.25 ms** | 18.78 |
| int8 chain, epilogue kernel, BM 256, channels_last head/tail | **37.46-37.73 ms** | **26.50-26.70** |

**1.41-1.42x** the same-session fp16 window, against the project's accepted bar `bar_int8_window` of
1.4x. The user-facing target (>=23 fps ungated at 1920x1080) is met with ~15% of margin.

Where the 37.5 ms goes (stages measured on the same run):

| stage | ms | what it is |
|---|---|---|
| head + quantise | 1.6-2.6 | head conv runs channels_last, so the quantise is a coalesced copy |
| 16 body hops | 29.4-29.6 | **1.84 ms/hop, 84 TOPS** |
| tail: dequant + conv + pixelshuffle + residual | 7.9-8.5 | dequant 2.1, tail conv + shuffle 3.6, the rest is the 4K residual |

Two measurements changed this from 55 to 37.5 ms, and both were layout, not arithmetic:

- **head-quantise 16.19 ms -> 1.96 ms.** The NCHW->NHWC version gathered 64 channels that sit 2 M
  elements apart; running the head convolution in `channels_last` makes the same 400 MB pass a copy.
- **tail 11.4 ms -> 7.9 ms.** The same transpose on the way out, fixed the same way (`dequant` to NHWC,
  the tail convolution reads channels_last).

and one swept parameter: `body_kernel` at BLOCK_M 256 / 4 warps is **1.82 ms/hop**, against 2.38 at 128
and 4.06 at 64. BLOCK_M 512 with 4 warps is pathological (79 ms/hop) - register pressure, not a slow
kernel - which is worth knowing before anyone "tunes" this again.

##### The picture: 0.09169% of pixels, inside the 0.1% bar, at the same 27.6 fps

Calibration on nine Violet Evergarden frames, evaluated on four (v120, v400, v800, v1200 - all held
out of the calibration set), worst-of:

| frame | share above 8 levels | peak |
|---|---|---|
| v120 | 0.00412% | 13 |
| v400 | 0.00814% | 17 |
| v800 | 0.04212% | 19 |
| **v1200** | **0.09169%** | 55 |
| **worst** | **0.09169%** (bar 0.1%) | 55 (proposed bar 20) |

###### What got it there, and what did not

The chain started at 0.728% (per-channel grids, pct 99.9, three calibration frames). Measured, in the
order they were tried:

| change | worst share | note |
|---|---|---|
| output margin 1.0 -> 1.5 | 0.728% -> 0.730% | no effect: the output grids were never the constraint |
| calibration frames 3 -> 9 | 0.234% | helps, but plateaus |
| per-channel -> **per-tensor** activation grids | 0.114% | counter-intuitive and consistent with the lane's own recipe: a per-channel grid is sized by that channel's outliers, so the step is coarse for everything else |
| percentile 99.9 -> 100 (max) | part of the above | with 9 frames the max is representative, so clipping is no longer the trade |
| margin below 1.0 (0.95 / 0.90 / 0.85) | 0.137 / 0.170 / 0.225% | finer steps, worse share: clipping returns |
| reference: torch fp16 vs **the TRT fp16 plan's own output** | 0.114% vs 0.114% | the measurement is not an artifact of the reference |
| **six of sixteen hops on the three-accumulator kernel** | **0.0917%** | see below |

Saturation measured on the held-out frame: 0.002% at layer 1 and 0.000% at every other layer, so the
residual was never clipping - it was the resolution of one weight scale per output channel.

###### The three-accumulator kernel: the ledger's "untried variant", measured

The ledger recorded per-tap weight scales as accurate (0.008%) and unaffordable (a float accumulator
takes the dot out of int8 mode, 2x), and named one untried variant: several int32 accumulators,
converted only after the loop. That is `body_kernel_g3` here - three int32 accumulators, one per kernel
row, three weight scales per output channel applied after the loop.

| kernel | share | ms/hop | 16-hop chain |
|---|---|---|---|
| epilogue (one weight scale per output channel) | 0.114% | 1.82 (BM 256) | 29.1 |
| **g3 on 6 of 16 hops, BM 128** | **0.0917%** | 2.63 on those hops | **30.5** |
| g3 on all 16 hops, BM 128 | 0.066% | 2.63 | 42.1 |

Three accumulators spill at BLOCK_M 256 - one block size for both kernels costs 79.76 ms instead of
36-37, which is how the first hybrid attempt read 12.5 fps. With the g3 hops at BLOCK_M 128 and the
rest at 256, the hybrid costs 1.3 ms over the pure epilogue chain and lands inside the bar.

###### The window, final

| configuration | window | fps |
|---|---|---|
| fp16 TensorRT plan, same process, same events | 48.94 ms | 20.43 |
| **int8 hybrid (6 g3 + 10 epilogue), channels_last head/tail** | **36.19 ms** | **27.63** |

1.35x on this draw (1.41x on an earlier one - the TRT plan itself read 48.94-55.61 ms across today's
runs, which is the drift the protocol section documents). Stages: head-quantise 1.60, chain 30.48,
tail+residual 5.78.

###### What is still not met

**The peak: 55 levels against the proposed bar of 20** (`bar_int8_peak`, still pending the owner's
decision). The share passes; the peak does not, and every configuration measured here lands between 29
and 55. A full release-grade sample is also larger than four frames - the bar's minimum is two and this
is four, but the 13-19 frame spread sample the protocol describes is not measured here.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/int8-window.txt \
       --records harness/docs/wsp/verified/records/int8-window.jsonl --bars harness/docs/claims-bars.json

@claim R001 47.81 ms "frames [72217853e9867e3f] n=8" measured
@claim R002 36.19 ms "frames [72217853e9867e3f] n=8" measured
@claim R003 48.94 ms "frames [72217853e9867e3f] n=8" measured
@claim R004 0.09169 % "frames [bf76f141c714826b] n=4" measured
@claim R005 55.0 levels "frames [bf76f141c714826b] n=4" measured

<a id="evidence-docs-wsp-verified-int8-live-txt"></a>

### `harness/docs/wsp/verified/int8-live.txt`

#### The live lane: the app runs the int8 body where it accepts it, and refuses it elsewhere

Measured 2026-11-17 by running this checkout's own backend (`backend/rve-backend.py`) with the
`REAL-2.4.2` bundled interpreter (torch 2.9.0+cu130, TensorRT 10.12.0.36 - the app's instrument), a
leased GPU slot, and clips cut from the remastering library at 1920x1080 -> 2160p. The raw logs are
committed next to this file under `logs/`, and every record cites one.

| clip | int8 probe: share / peak | verdict | gate | reuse | render loop |
|---|---|---|---|---|---|
| smoke, 1 s (24 frames) | 0.003% / 13 | **int8** | chain 2 | 16/24 = 66.7% | 37.50 ms/frame |
| Violet Evergarden 20 s | 0.058% / 37 | fp16 | chain 4 | 335/480 = 69.8% | 35.12 ms/frame |
| Chainsaw Man 20 s | 0.000% / 11 | fp16 | chain 3 | 100/481 = 20.8% | 47.58 ms/frame |

##### The chain in the product and the chain in the probe are the same chain

On the smoke clip the body runs and the app's own phase split reads

```
int8 phases: head 4.5  quantise 1.5  hops 24.3  tail 7.6  SUM 37.9  BODY-TOTAL 37.9 ms
int8 host: 16 launches cost 0.30 ms of HOST (0.019 ms each); whole hop loop 0.31 ms
```

37.9 ms against the offline chain's 36.19 ms - 5% apart, with the hop loop at 1.52 ms/hop. That closes
the "does the built chain reproduce" question: it does, in the product, under the gate.

##### Why the app refuses int8 on both longer clips: the DEPTH-2 composition, not the peak

`Int8Probe.py:227` takes the verdict at the **worst depth the gate may serve**
(`worst_depth = max(comp_d1, comp_d2) <= bar`), and the log line prints depth 1 only. Asking the probe
directly for its `reason` (the same clip, the same model, the bundled interpreter) resolves it:

```
violet20s.mkv:  reason: composition d1 0.058% / d2 0.122% -> worst 0.122% > bar 0.1%
                        (own defect 0.006%, 7 of 8 pairs at d1, 6 at d2)
csm20s.mkv:     reason: composition d1 0.000% / d2 0.218% -> worst 0.218% > bar 0.1%
                        (own defect 0.000%, 4 of 8 pairs at d1, 2 at d2)
```

So **both titles are refused on the depth-2 composition**, at 0.122% and 0.218% against the 0.1% bar,
while the peak (37 and 11) is reported but is not the deciding number. This is the lane's recorded
blocker (0.390% / 0.166% at depth 2) reproduced on two other clips by the product's own instrument -
and it is the number my offline 4-frame sample cannot see, because that sample is depth 0.

The correction matters: the first version of this file said Violet was refused on the peak bar, from the
log line alone. It was not - the log prints `composition` at depth 1 and the verdict is taken at
`max(d1, d2)`.

##### What the gate does on these clips (for E37)

- Violet: **chain 4** certified, p90 1.85 / 4.66 / 6.98 / 9.61 against a limit of 12 - monotonically
  worse with depth, i.e. the latest reference is the best one nearly everywhere.
- Chainsaw: **chain 3** certified, p90 0.73 / 0.69 / 0.66 - deeper is *better* here, by 0.07 units out
  of a 12-unit budget (0.6%).

The item `rve-hr-reference-cache` is written around "the gate stores ONE reference". It does not: the
code keeps a chain and certifies depth 3-4 on these clips. And where deeper is better the gain is
0.6% of the budget, while where the gain could matter (Chainsaw) the reuse is limited by the threshold
(20.8%), not by reachability.

Not measured: the wall with int8 under the gate on a real title (both titles decline it), and the 13-19
frame accuracy sample.

Run: PYTHONPATH=. python3 -m harness.cli check harness/docs/wsp/verified/int8-live.txt \
       --records harness/docs/wsp/verified/records/int8-live.jsonl --bars harness/docs/claims-bars.json

@claim R001 0.003 % "frames [d85c9dd235341520] n=24" measured
@claim R002 13.0 levels "frames [d85c9dd235341520] n=24" measured
@claim R003 37.9 ms "frames [d85c9dd235341520] n=24" measured
@claim R004 37.5 ms "frames [d85c9dd235341520] n=24" measured
@claim R005 0.058 % "frames [947a1391acebd169] n=480" measured
@claim R006 37.0 levels "frames [947a1391acebd169] n=480" measured
@claim R007 69.8 % "frames [947a1391acebd169] n=480" measured
@claim R008 35.12 ms "frames [947a1391acebd169] n=480" measured
@claim R009 9.61 levels "frames [947a1391acebd169] n=480" measured
@claim R010 20.8 % "frames [7219ce9813d28df0] n=481" measured
@claim R011 47.58 ms "frames [7219ce9813d28df0] n=481" measured
@claim R012 0.66 levels "frames [7219ce9813d28df0] n=481" measured

### Cited but missing on disk

- `eb6b3e85c391be484e7104411d4415d407706f4d897c9d17192f8608820cec7d`
- `761379ef7925d8dda2628d6a6b1e46934af984a225f4de70e314017a04b269fd`
- `c43ee834171a97be2a7d875fddd4691655183afaf2d1bc36a9bb836794c21062`
- `d80ab583a6237cc6e2ef4aa25548988be221fa91b182bdebdffbd1231b5a9cf3`
- `b2ebed58e64551564a5924c3ed287a15c752f2eb0dbbd2982dc10c0751cf7727`
- `16d4e2990c03e61dff0801173e2d5f19bdcc4655169395a938d932b4887214f8`
- `04351722a996cd39d3776b5cd586049fa8505a91f0b2f2b73ac77d95039331dd`


## What this document does not tell you

- whether a population label is true (the item-set hash is checked, the human label is not)
- whether the item set is the one the decision needs
- whether a cost model is complete
- free-text arithmetic in the legacy claims files

A green check is a floor, not a ceiling.
