# Plan: the int8 body path on long digital titles

Proposed 2026-11-17. **The thresholds below are proposals until the owner confirms them**; the bars are
already records (`harness/docs/claims-bars.json`), so a confirmation is a one-line change there rather than an
edit to a document.

Why this plan exists: every lever measured so far tops out at 1.23x (channel width) and the
quantisation path is the only one that halves the arithmetic. The boundary between operators - not the
kernel - is what it has to beat (E9', E30a, E30b, reproduced 2026-11-17).

## Starting point (2026-11-17, from `REPORT-int8-lane-2026-11-16.md`)

This plan is a continuation, not a start. The lane was built and run on 2026-11-16 and this is where
it stands, all figures from that report:

| | |
|---|---|
| window, int8 body vs fp16 | **0.991x** (51.54 against 51.06 ms, same session) |
| certification on the int8 body | **0.00832%** above 8 levels - inside the bar |
| per-hop cost | 1.753 ms (87.2 TOPS) against TRT's int8 conv at 91 TOPS: the hop closed to 4.3% |
| projection after the two named fixes | **1.379x** at a realistic 1.3x PReLU floor; the 1.40 gate sits **inside** the 1.36-1.42 range |
| projection **under the gate** | **1.216-1.236x** |

**And the blocker is not speed.** The body does not run in the shipped configuration at all:
`--no_temporal_reuse` is a precondition, the gate creates a `tile_upscaler`, and the body branch in
`_call_single` is never taken. Turning the body on means turning the gate off - 45.36 ms/frame as
shipped against 104.44 with the gate off and the int8 body, i.e. **2.30x slower per frame**, and on
the wall it is the frame penalty times the gate's own value: 2.30 x 3.3 = **7.6x** if the gate's
3.3x is taken as the other factor (11.7 s against 38.85 s), or **6.4x** computed through the render
times (38.85 x 104.44/54.56 = 74.4 s against 11.7 s).

> **Correction to the source report.** `REPORT-int8-lane-2026-11-16.md` states "~6.9x slower on the
> wall". That number is derivable from neither of its own factors: the report's own 2.30x and 3.3x
> give 7.6x, and its own render times give 6.4x. Every other figure in that report's section 1 was
> re-derived here and holds (0.991x, 87.2 TOPS, 40.3/28.0 ms, -12.2 ms, 1.418x, 1.378x,
> 1.216-1.236x under the gate, 2.302x per frame). Quote 7.6x or 6.4x and say which.

So the first question this lane has to answer is not "how fast is int8" but "can the body serve the
frames the gate leaves". The architecture for that exists in the report's own next steps: calibrate and
certify full-frame before the first tile, then serve tiles on already-certified grids, and merge the
three consumers of the calibration forwards (otherwise F = 2.65 s and the break-even becomes 43 s
instead of 14).

The success table below is therefore the *target*, read against a measured 0.991x baseline and a
1.379x projection - not against zero.

## Goal

Cut full-render wall clock on **long digital titles** in the shipped configuration (gate on) without
taking the picture past the bar. Speed up the **int8 body path**: head fused, tail fused with the
dequantise. Everything else is closed (see the last section).

Do not aim for 2x. With the gate on, the engine is **64.6% of the render cycle**, so the wall ceiling
is bounded by the engine gain and not by the engine's own number.

## What counts as success

| level | model window vs fp16 (same session) | wall under the gate, long clip | picture |
|---|---|---|---|
| **accepted** | >= 1.4x (`bar_int8_window`) | >= 1.2x (`bar_int8_wall`) | bar passed |
| good | >= 1.7x | >= 1.35x | bar passed |
| ceiling (do not require) | ~1.9x | ~1.45x | |

These pairs are not independent: they are the ledger's own conversion of an engine gain through a
64.6%-engine cycle (1.40x -> 1.23x wall, 1.59x -> 1.32x, ledger:1528). Ungated, the reference point is
**~16.3 fps at 1080p -> 2160p**; the 23-24 fps target is the same 1.4-1.45x applied to it.

**One figure in the brief is withdrawn.** "Even perfect int8 gives at most ~1.75x of wall" does not
survive the arithmetic: at a 64.6% engine share a 1.9x engine gives 1.44x wall, and an *infinitely
fast* engine gives 1/0.354 = 2.82x, so 1.75x would need about 3.0x on the engine - beyond the int8
ceiling of 1.4-1.9x measured or projected anywhere in this project. The ceiling row (1.45x) is the
consistent one.

## Quality gates - checked before any speed number is read

- Share of pixels above **8 levels < 0.1%**, RGB, channel-max, **worst frame** of a spread sample
  (`bar_int8_share`; the same statistic the project already ships against).
- Peak: `bar_int8_peak` = **20 levels**, pending the decision below.
- The comparison is against the **fp16 render of the same clip and the same configuration**.
- Checked at the **chain depth the gate actually serves**, not on neighbouring pairs. The probe today
  measures depth 1 while the gate serves depth 2.
- At least two titles of different material; calibration and test frames disjoint.
- Mandatory control: `fp16(t-k)` against `fp16(t)` on the same pairs, or the defect cannot be
  attributed (suspicion is on Chainsaw depth 2: **0.390%** against 0.005% for the fp16 gate).

## Measurement protocol

- **A/A first**: two runs of one configuration must be byte-identical (the pipeline is deterministic:
  502/502 frames, ledger:308).
- fp16 baseline **in the same session**, >= 3 interleaved rounds, clocks in the log, nothing else on
  the GPU except NVENC.
- Steady state only; the first frame separately.
- Ratios inside one session only: same-configuration absolutes drift 20-30% between days.
- Effects smaller than ~10% are not measured with one pair.

## Amortisation over the clip

The full price on video, not per frame: probe **64.6 s** plus build and certification **0.85 s** =
**65.45 s**. Break-even is **16.2 min** of video today and **5.6 min** after the head and tail fix
(65.45 s divided by 8.45 ms and 24.26 ms saved per rendered frame, at a 33.3% render rate).

Acceptance requires one of:
- the probe is shortened **with its verdicts validated against the full probe**, or
- int8 switches itself off below the threshold.

Shortening the probe without validation is forbidden: that changes the instrument, not the cost.

## What the hypothesis loop must return

Every hypothesis fixes in advance: the population the number is computed on and the population the
decision applies to; the comparison base; the predicted value and the falsifier; and a **discriminating
measurement**, not a mechanism named on top of a number.

Loop rules: fit on one set of titles, accept on another; three conclusions in this project were
withdrawn for one pair or one frame. `probe failed` and a `NameError` are not measurements - three
states, and `error` never enters a results table. Every number goes through `claim_check.py`, and a
green check is not proof of a mechanism. Stop if several iterations in a row gain under 5% or the
budget is spent.

## Closed - do not reopen

Channel cutting, low-rank, Winograd, spatial routing by edges, fp8, per-tap and grouped scales,
TensorRT flags and tiling, register/warp occupancy redistribution, PTX port with a register cap.

## The final report (six points)

1. The built chain reproduces the simulation's numbers.
2. Model window against fp16, same session, NVENC alongside, >= 3 rounds.
3. The int8<->fp16 boundaries, measured inside the built pipeline.
4. Accuracy on a spread sample at the gate's depth, per title, with the probe's verdict.
5. Wall on a clip above the break-even, including the full price on video.
6. For each closed hypothesis: the mechanism it names and the measurement that isolates it.

## Provenance of every figure in this plan

| figure | record | line |
|---|---|---|
| engine is **64.6%** of the render cycle | `RESEARCH-LEDGER.md` | 1527 |
| 1.40x engine = 1.23x wall; 1.59x = 1.32x | `RESEARCH-LEDGER.md` | 1528 |
| probe **64.6 s** + build/cert **0.85 s** = 65.45 s | `claims-int8-stage4.txt` | 264-265 |
| break-even **16.2 min** now, **5.6 min** after head+tail | `claims-int8-stage4.txt` | 267-270 |
| Chainsaw depth 2 **0.390%** / 0.166% against **0.005%** for the fp16 gate | `claims-int8-stage4.txt` | 278-288 |
| ungated anime **16.3 fps** at 1080p -> 2160p | `MODEL.md` | 414 |
| the shipped tile-reuse opt-in reads **0.071%, peak 23** | `RESEARCH-LEDGER.md` | 113 |
| peak bar **20 levels** | `claims-bars.json` (`bar_peak`) | - |
| A/A control 502/502 frames byte-identical | `RESEARCH-LEDGER.md` | 308 |
| int8 block kernel 1.78-1.85 ms / 83-86 TOPS, bit-exact | `harness/docs/wsp/verified/int8-block.txt` | measured 2026-11-17 |

## The two decisions this plan needs before it runs

### 1. Which peak bar: 20 or 23-26

**Recommendation: 20 (`bar_int8_peak`), and 23 is not available to this lane.** The 23 comes from the
tile-reuse opt-in, which reads 0.071% at peak 23 - an off-by-default path that is already eight times
the shipped defect area and is rejected as a default for that reason (ledger:113). Adopting its peak
as the int8 bar would raise the release bar to the level of an artifact the project has already
declined. The E9' int8 build's own peak was 4; a lane that cannot reach 20 needs its own measurement to
say so, and the requant-form accuracy is exactly the unmeasured number today
(`harness/docs/wsp/verified/engine-int8.txt`).

### 2. Below which length does int8 refuse itself

**Recommendation: 2x the break-even - 32 min today, 11 min once the head and tail are fused.** The
break-even is derived from two population-dependent inputs (8.45 ms saved per rendered frame, a 33.3%
render rate on that clip); at exactly 1.0x the break-even the gain is zero and the noise floor is
larger than the effect. The project's own rule - effects under ~10% are not measured with one pair -
sets the same direction. The threshold belongs in `bar_breakeven_len` once the owner fixes it; until
then the safe default is the larger number, because a short clip that loses time is a visible
regression while a long clip that skips int8 is not.
