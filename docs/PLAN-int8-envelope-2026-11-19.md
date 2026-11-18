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

**The gap to the constraint is 2.50 ms on Violet** (57.73 / 1.2 = 48.11 against today's 50.61). That is a
number to close, not an uncertainty to model.

Where the frame goes (Violet, same session; `int8-tail-split.txt`, `int8-frame-accounting.txt`):

| part | ms | status |
|---|---|---|
| 16 hops | 26.95 | 152.9 GFLOP per hop at **91 TOPS** - near the practical int8 rate, not near the peak |
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

## 4. The stop rule

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
- A forecast with probabilities in it should not be written again. If one is, the reason must be that its
  falsifiers were listed first.
