# The int8 envelope: what must be true, what is already measured, what is only a hypothesis

Written 2026-11-19. **This is a plan, not evidence.** It carries no `@claim` and lives outside
`docs/wsp/verified/` on purpose: the numbers it quotes are records, and the record is authoritative - the
ids are given so a reader can check, not so this page can be cited as if it measured anything.

Why it exists: two forecasts in this lane were published and both were wrong - 1.55x (retracted: ffv1
encodes a quantised picture faster, so the number was the encoder) and 0.95x (one draw, not reproduced).
Both failed the same way: a ratio multiplied across instruments or populations. A forecast of "we will reach
X" is therefore not the right artifact. Three things are, and they are below: the **constraint**, **what is
already inside it**, and a **portfolio of probes** with what would falsify each and what it costs.

## 1. The constraint (from the registry and the product, not from a wish)

| what | value | source |
|---|---|---|
| wall speed-up over the fp16 engine | **>= 1.2x**, median of rounds 2+, direction "above" | `bar_int8_wall` in `docs/claims-bars.json` |
| picture | **<= 0.1%** of pixels >8 levels off, **worst frame** of a spread sample, n>=2 | `bar_share` |
| peak | 20 levels | `bar_int8_peak`, marked PENDING OWNER DECISION |
| what the body may serve | LR <= 2.5 Mpx, exactly 64 body channels, whole-frame path only (a created gate means tiled and the body is not wired in) | `Int8Body`, `UpscaleTorch` |
| frame reuse | shelved by the owner 2026-11-18, further investigation forbidden | `reuse-gate-shelved` |

The speed bar is the one that decides engineering effort right now; the picture bar decides whether any
speed matters.

## 2. What is already inside, and by how much it misses

Frame, int8 body, reuse off, one session, null sink (records `W039`-`W041`, `W007`-`W009`):

| clip | fp16 | int8 | ratio |
|---|---|---|---|
| Violet BD | 57.73 ms | 50.61 ms | **1.141x** |
| Chainsaw same-shot | 55.53 | 49.64 | **1.119x** |

Median of the four sink/clip quotients 1.126x, span 1.088-1.141x (`int8-wall-ab.txt`).

**UPDATED 2026-11-19, same day, after the tail-layout adoption** (`int8-tail-layout.txt`, records `W009`,
`W025`): int8 is now **1.221x on Violet and 1.230x on Chainsaw** - both medians clear `bar_int8_wall` = 1.2x,
by **0.82 and 1.15 ms of margin**. The 2.50 ms gap this section used to describe is closed; what is left of
that sentence is the caveat that one Violet round (r3, 1.178x) is still below the bar, so the claim is the
bar's own statistic (median) and not "every round".

That changes the shape of the search: with the constraint met, further speed work is justified only by what
it buys the product (see the reserve and `rve-fast-preset-per-hardware-2026-11-18`), and the next portfolio
revision starts from `docs/wsp/RESEARCH-candidates-2026-11-19.md`.

Where the frame goes (Violet, same session; `int8-tail-split.txt`, `int8-frame-accounting.txt`):

| part | ms | status |
|---|---|---|
| 16 hops | 26.95 | 152.9 GFLOP per hop at **91 TOPS**; the same hop also moves 265 MB in its 1.68 ms, i.e. **158 GB/s** - so it sits at ~55% of BOTH roofs (176 TOPS dense, 288 GB/s) and is schedule-bound rather than at a practical rate (checked against the researcher's report, 2026-11-19) |
| head + quantise | 6.25 | 0.72 GFLOP of arithmetic plus a ~400 MB pass |
| tail total | 7.77 | dequant 1.58, conv 4.86, shuffle 0.40, residual 0.93 |
| pack + queue | 3.87 | pinned-slot fill + `writeQueue.put` |
| read wait | 0.24 | decoder is not a factor |
| body total | 42.0-44.9 | the phase counter's own sum |

Two floors are already reached and must not be spent again: **dequant** (1.58 against a 1.4 ms traffic floor
for 400 MB at 288 GB/s) and **read wait**. The hard floor below everything: the body moves **4.25 GB per
frame**, which is **14.7 ms** at this card's bandwidth, so no kernel work takes the frame under ~15-20 ms.

## 3. The portfolio: every option, its cost, and what would falsify it

No `p` column. A probability I cannot measure is an opinion wearing a number's clothes; the falsifier is
the honest substitute, and it is what decides whether the option is worth its probe.

| option | measured today | ceiling if it works | probe cost | falsifier |
|---|---|---|---|---|
| **tail conv, NHWC view** | 4.86 | **2.10 - measured** (isolated: 5.296 -> 2.098; C_out 12 -> 16 changes nothing, so it is the view and not tile fill) | patch exists, parked at `/home/jorj/2Tb/tail-layout-nhwc-2026-11-18.patch` | already measured; blocked only by the calibration ordering question, now closed |
| tail conv, better tiling | 2.10 after the above | ~1.1 (25-30 TFLOPS) | 4 h | a Triton 64->12 conv reaching 25 TFLOPS on this shape |
| hops, MMA utilisation | 26.95 at 91 TOPS | ~20 (75% of dense int8 peak = 18.5) | 1 h to falsify: one hop, BLOCK/warps sweep | does any configuration move the achieved TOPS off 91 |
| head + quantise | 6.25 | ~3.0 | 30 min: isolate the head conv and the quantise call | whether 6.25 is layout/copy-bound or arithmetic |
| pack + queue | 3.87 | ~1.0 | 1 h: pooled slot against null sink | whether the 3.87 is a copy, a wait, or the writer's back-pressure |
| shuffle + residual fusion | 1.33 | ~0.9 | 2 h | byte-identity of a fused kernel, else it is different arithmetic |
| calibration on crops | 7.09 s -> **0.45 s, measured** | probe cost only, not the frame | 0 (measured; `int8-calibration-crops.txt`) | n=1 per arm today: one repeat before it is adopted |

The largest expected item is the hops and the cheapest probe is the tail layout - which is already measured
and now unblocked. That ordering is the whole point of writing this down.

## 3b. The reserve: what we know costs time but have never attacked

The portfolio above is the measured part. This is the part where the honest answer is "we do not know what is
possible here yet", and it is written as a queue of probes, not as a list of hopes.

| reserve | known today | why it is still open | cheapest probe |
|---|---|---|---|
| the Python wrapper around the body | ~2.3 ms of the frame (frame total minus named parts, `int8-frame-accounting.txt`) | never isolated; the CUDA-graph result in `docs/MODEL.md` (~0.3%) is about the TRT engine, not about the tensor conversions and handover in this path | time the wrapper with the body replaced by a memcpy: 1 h |
| quantise pass after the head conv | 1.6 ms as its own phase; the PReLU is already fused into it, and it sits at its own traffic floor (1.38 of 1.58) | the head CONV is not fused and is the expensive half: **7.17 GFLOP in 4.65 ms = 1.5 TFLOPS**, 4.8x its 0.96 ms traffic floor | isolate the head conv and sweep layouts/padding, then fuse it into the quantise (removes a 265 MB write); 2 h, filed as `rve-head-conv-2026-11-19` |
| the output colour path | not in the frame budget at all | the writer's rgb24 payload is twice yuv420p's bytes, and nothing in this corpus isolates the conversion from the transfer | render with a yuv420p payload against rgb24, same sink, 1 h |
| int8 buffer layout | NHWC, 64 contiguous channels per pixel, 4.25 GB/frame of traffic | never tried sideways: swizzled/Morton order or a channel-blocked layout could change the hop's read pattern at 91 TOPS | one hop with a swizzled index, compare TOPS, 1 h |
| **batching two frames** | the kernel already carries a batch dimension (`x.shape[0]`); the app feeds 1 | the hop is 152.9 GFLOP at 91 TOPS - whether a doubled M improves MMA occupancy is unmeasured either way | same hop with M x2, compare TOPS, 1 h |
| int8 head | the head runs fp16 (3 -> 64) | the input is 8-bit RGB without per-channel scales; whether an int8 head is even expressible here is unknown | arithmetic first: what grid would the 3 input channels need? 1 h, no GPU |
| pipeline depth | one frame in flight | the app's profile total already exceeds the wall clock in the pipelined path (`int8-frame-accounting.txt`), so the depth interacts with measurement itself | vary the queue depth, watch the loop line, 1 h |

The distinction that matters: the first four are **known costs with unknown ceilings**, the last three are
**unknown whether there is anything there at all**. The known-cost rows have a falsifier that can be run in
an hour; the unknown rows have a question that has to be phrased before it can be priced. Neither belongs in
a forecast, and both belong in the queue before they get built.

## 4. The stop rule

0. **The reserve (3b) is visited only when the constraint is not met and the measured pool (3) is exhausted** -
   otherwise it is how a lane spends a week on a curve it has not plotted.
1. **Speed work stops when `bar_int8_wall` is met and no remaining option's ceiling exceeds its probe cost.**
   The bar is 2.50 ms away and one measured lever is worth 2.19-3.20 ms, so stop looking for speed first -
   the remaining question is whether the lever's picture is clean.
2. **A probe is not run when its falsifier costs more than the decision it can flip.** The megakernel row
   was closed this way (`rve-megakernel-body-2026-11-19`): the boundary it would remove is already hidden,
   so the answer was arithmetic, not a week of kernels.
3. **Picture work stops when the render pair is under the bar with zero frames above it and the residual is
   either explained or 30x below.** Today's post-fix pair sits at 0.00318% with zero frames above - done,
   with the residual named as unexplained rather than asserted harmless.

## 5. Budget for surprises

The lane needs slack, because the frontier is set by what nobody measured yet. Today's example, and it is a
fair one: removing the probe's certificate - a change made for *cost* - turned out to have been masking a
calibration-ordering bug, which took four renders and two disproved arms to trace
(`int8-calibration-ordering.txt`). Nothing in a forecast would have contained it. A week of work should
carry roughly a day of budget that no plan claims.

## 6. Closed by measurement - do not re-enter

int8 implicit and FP8 on this graph; explicit Q/DQ per-channel (lossless, slower); scalar-slope int8 (1.40x
ceiling but destroys the picture); body width `C=32/48` (14-37% residual per layer); Winograd through cuDNN
(`E26`, negative); activation sparsity (PReLU output is non-zero everywhere); merging 3x3 into 5x5 (+39%
MAC); Strassen at N=64; fusing the 16 hops into a megakernel (the boundary is already overlapped); swapping
the model for a smaller one (owner decision - the model menu is the product); frame reuse (shelved).

## 7. How this file stays true

- Numbers live in `docs/wsp/verified/` with records; this page cites ids and dates, and a value that moves is
  changed here only after its record moves.
- Work items live in `docs/backlog/queue.yaml` with `gpu:`, `refutes:` and `acceptance:`; a probe that this
  page lists as worth its cost must exist there before it is run.
- **The lane ratio (1.141x) is a lane-level number, not a per-change one.** It is re-taken when a MATERIAL
  change lands (one that could move the frame by more than the digest of the changes since), or when a
  decision needs it - not after every optimisation. A per-change acceptance is the change's own property:
  picture identity for a layout swap, a phase line for a kernel, a byte-identical render for a fusion.
  Conflict resolution: if a change's own acceptance does not mention the ratio, the ratio is not re-measured
  for it.
- A forecast with probabilities in it should not be written again. If one is, the reason must be that its
  falsifiers were listed first.
