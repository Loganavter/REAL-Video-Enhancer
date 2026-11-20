# The frame budget, as it stands

**Generated** by `harness/scripts/frame_budget.py` - do not edit. Every number is parsed from the render log
named below or resolved by id from `harness/docs/wsp/verified/records/`. The prose says what each line IS and
what its floor is, so a reader who was never in this lane can challenge any of it.

## The machine, the model, the build

| | |
|---|---|
| card | NVIDIA GeForce RTX 4060 Ti, 34 SMs, **176 dense int8 TOPS** and **288 GB/s** (the two roofs every number below is measured against) |
| ridge | 176 TOPS / 288 GB/s = **611 FLOP/byte** - and the hop sits at 610, i.e. exactly on it |
| clip | `bd_violet20s`, 494 frames, 1920x1080 in, 4K out |
| model | `2x-Futsuu-Anime(1).pth`, SRVGGNetCompact 2x, 64 body channels, 18 convolutions |
| precision | fp16 engine vs the int8 body: 8-bit codes between the 16 body hops, fp16 head and tail weights where noted |
| build | fused head kernel, int8 tail kernel, tail shuffle+residual inside that tail kernel, zero-halo hop buffers, and the tail writing the yuv420p payload itself |
| tail output, as the run resolved it | **yuv420p payload** - the run printed `int8 tail writes the payload`; it predates the explicit gate line, so this is read from the payload line rather than from the flag |
| the bar | `bar_int8_wall` = **1.2x** wall clock against the fp16 engine; the picture bar is the share of pixels more than 8 levels off on the worst frame of a spread sample, 0.1% |

| commit | `948304a6` - the tree these numbers were taken on |

The pipeline, in the order a frame is paid for: **decode** (ffmpeg, outside the loop) -> **reader** (yuv420p10 -> a tensor) -> **head** (3 -> 64 channels) -> **sixteen hops** (the body) -> **tail** (64 -> 12 channels, pixel shuffle to 3 at 2x) -> **residual** (add the upsampled input) -> **pack** (4K payload to the host) -> **writer** (ffmpeg stdin). The render loop below is the wall clock of that chain; the body is what the model costs, measured by the app's own events.

## The body, from the app's own phase counter

| line | ms | share | what it is |
|---|---|---|---|
| head | 0.60 | 2% | the first convolution, 3 -> 64 channels, 3x3, fused with its PReLU and with the quantisation into int8 codes (one kernel) |
| quantise | 0.00 | 0% | 0.00 by construction: the head writes codes directly, so this pass no longer exists |
| hops | 24.50 | 94% | the body proper: sixteen 64 -> 64 int8 convolutions, 3x3, one Triton kernel launched sixteen times |
| tail | 1.00 | 4% | the last convolution 64 -> 12 channels, its pixel shuffle to 3 channels at 2x, and the residual add - all three in one kernel now |
| **BODY TOTAL** | **26.20** | 100% | measured with the app's own CUDA events, window by window |

Tail split on the same run: dequant 0.00, conv 1.00, shuffle 0.00, residual 0.00, sum 1.00 ms. The zeroes are the second adoption: the dequant pass was folded into the tail's weights and the shuffle+residual into its epilogue.

### The tail did not get faster, and its floor says why

Its floor is the traffic it cannot avoid: 145.15 MB of codes and residual to read, plus the
output. Writing yuv420p moved that floor from 676.800 ms to
547.200 ms - a 0.13 ms shift - and the measured phase read 1.00 ms
both before and after. So the tail was never bounded by its write; it is bounded upstream (the code read,
the convolution and the epilogue), and the whole saving of the yuv payload appeared in the pack instead.
A floor is not a promise: a member that is not the binding one can be removed for free in time and only
for certain on paper.

## The hop against its two roofs - why every single lever came out flat

| quantity | value |
|---|---|
| one hop, measured | 1.531 ms |
| its compute floor (152.9 GFLOP at 176 TOPS) | 0.869 ms = **57% of the roof** |
| its DRAM floor (250.6 MB at 288 GB/s, counter-measured) | 0.870 ms = **57% of the roof** |
| the two floors agree | that is what 'on the ridge' means: 610 FLOP/byte against a ridge of 611 |
| ncu's own reading | Compute (SM) 74%, DRAM 56%, LSU the busiest pipe (73.9% by instruction count), L2 6.1x the DRAM traffic |

The 74% and the 57% are not a contradiction: ncu measures **pipe saturation** (including the load/store unit), the 57% measures **arithmetic against the int8 roof**. Both say the same thing - the kernel presses on compute and on memory at the same time, in equal measure - and that is why a change that improves only one of the two reads as flat noise. Thirty configurations in seven families were tried on this line; every
one lost (the table three sections down).

## The frame

| what | ms | floor, and where the floor comes from |
|---|---|---|
| body (above) | 26.20 | **13.9 ms** for the hops alone at the 176 TOPS roof (152.9 GFLOP per hop x 16); DRAM floor **13.9 ms** (250.6 MB per hop x 16 at 288 GB/s) - the two agree, which is the ridge again |
| pack + queue | 2.06 | **1.9-2.4 ms**: the 4K yuv payload is 12.44 MB and the link is **PCIe gen 3 x8** - measured under load at 6.55 GB/s round-trip, 25 of 25 samples, against 7.88 GB/s theoretical. The card is a PCIe 4.0 part but the slot negotiates gen 3, so this floor is a property of the motherboard |
| **render loop** | **31.48** | what the user waits for; `null` sink, so no encoder |
| everything else | +3.22 | reader, the wrapper around the model call, and the writer's back-pressure. The input path inside it is measured at 1.38 ms, of which 0.07 was removable |

The phase lines account for -2.31 ms of the tail-CSC row and the loop for -2.16, and the 0.15 ms between
them lands here, in `everything else`: the payload branch in `Frame`, the Y-plane reads for the preview
and the gate's luma, and the guard's event. Small, and named rather than absorbed - an unattributed
0.15 ms is exactly the size of finding that turns out to be a second copy somewhere.

## What is in this frame because it was measured and adopted

| change | saved on the loop | records |
|---|---|---|
| head as one fused kernel | -5.27 ms (11.9%) | int8-head-fusion:H006 / H007 |
| tail as a 17th int8 hop | -3.16 ms (6.9%) | int8-tail-hop:C001 / C002 |
| tail shuffle + residual folded in | -0.80 ms (2.0%) | int8-tail-shuffle:Q005 / Q006 |
| zero-halo hop buffers (mask-free) | -0.98 ms (2.8%) | int8-hop-halo:H004 / H005 |
| tail writes the yuv420p payload itself | -2.16 ms (6.4%) | int8-tail-csc:JC002 / JC003 |

## What was measured and deliberately NOT adopted

An outsider reading this list is the intended use: every entry is a lever someone expected to work.

| lever | what it was supposed to buy | what it actually read | against | record |
|---|---|---|---|---|
| epilogue divide folded into a and b | 1.3-2.7 ms: one fp32 divide per element, in the kernel all 16 hops run | **-0.655 ms** | 16 hops chained, 29.3 ms before it | `harness/docs/wsp/verified/int8-epilogue-fold.txt` : R002 |
| 6-bit codes between hops | 25% of the hop's 265 MB of traffic | **2.71431 %** | picture bar 0.1% of pixels | `harness/docs/wsp/verified/int8-sub8-codes.txt` : S005 |
| vectorised loads + weight hoist | the hop is at 52% of both roofs, so its schedule | **1.772 ms** | shipped hop 1.756 ms | `harness/docs/wsp/verified/int8-hop-loads.txt` : L002 |
| persistent-CTA grid | the same schedule, one CTA per SM looping over tiles | **3.709 ms** | shipped hop 1.703 ms | `harness/docs/wsp/verified/int8-hop-persistent.txt` : N002 |
| cache eviction / modifier policy | L1 hit rate is only 22%, L2 traffic 6.1x DRAM | **1.594 ms** | shipped hop 1.580 ms | `harness/docs/wsp/verified/int8-hop-cache.txt` : C002 |
| 2-D CTA tiles | halve the window amplification (3.05x -> 1.59x) | **1.55 ms** | shipped tile 1.504 ms | `harness/docs/wsp/verified/int8-hop-tiles.txt` : T004 |
| output channels split across CTAs | halve the accumulator to double resident warps | **2.632 ms** | shipped hop 1.591 ms | `harness/docs/wsp/verified/int8-hop-csplit.txt` : X002 |
| M x 2 batching | amortise the per-launch constant - WHICH DOES NOT EXIST: a whole 256x256 hop is 0.0572 ms, so any per-launch fee is <= 0.011 ms. Flat is the expected result, not a clue | **1.764 ms** | batch 1 of the same kernel 1.769 ms/frame | `harness/docs/wsp/verified/int8-hop-batch.txt` : Y002 |
| pack ring (pinned slots) | the 12.44 MB payload copy is synchronous | **34.36 ms** | ring off 34.42 ms loop | `harness/docs/wsp/verified/int8-pack-ring.txt` : K002 |
| reverse tile order on alternate hops | the previous hop's tail is still in L2 (32 MB of 132) | **26.577 ms** | forward 26.551 ms over 16 hops | `harness/docs/wsp/verified/int8-hop-traverse.txt` : V002 |

## What is still open, and what it costs

| idea | expected | what it needs | state |
|---|---|---|---|
| the hop: taps, staging, cascade, tiles, schedules | -5.3 ms at the very best, and only by removing the FIRST tap's exposed latency | **closed by mechanism**: staging ADDS LSU instructions (9 -> 11.26 at a 2-D tile, 9 -> 15.09 for a band) on the busiest pipe, `tl.gather` prices the register route at 13.5x, a register-local dp4a roofs at 44.2 TOPS against the 103.4 already reached, and the per-launch constant does not exist (a whole 256x256 hop is 0.057 ms) | `int8-hop-tapreuse.txt`, `int8-hop-constant.txt` |
| overlap the IO with the body (streams) | the pack was expected to be a copy that could hide | **-0.01 ms**: it is YCbCr arithmetic, and the tail-CSC adoption removed that arithmetic instead | `frame-pipeline.txt` |
| a PCIe 4.0 slot | ~1.2 ms off `pack`, since the link is gen 3 x8 today | hardware | not software |
| skipping duplicate frames / static regions | large on anime, and the owner's call forbids it | - | closed by policy, `rve-reuse-gate-forbidden` |
| cut the model (fewer channels or hops) | 2.9-4.5 ms per hop removed | 14-37% per-layer residual, 1.58% of pixels over the picture bar | closed by measurement |

## Which side of the ratio a millisecond lands on

The bar is a RATIO, `fp16 / int8`, so a millisecond saved is not worth the same everywhere:

| a millisecond saved in | int8 becomes | fp16 becomes | ratio |
|---|---|---|---|
| the int8 body only (a kernel) | 30.47 | 52.39 | **1.719** |
| a stage BOTH arms pay (pack, reader, writer) | 30.47 | 51.39 | 1.687 |

So a body-side millisecond is worth ~2.8x a shared one against this bar, and the largest wall-clock win in
the budget can be the smallest ratio win. That is why `rve-hop-fuse2-cascade` (body) and the pack (shared)
must not be reported as if their milliseconds were interchangeable.

## The ratio this produces

Three arms in one session, 3 interleaved rounds, shipped default (`int8-tail-csc.jsonl`): the fp16 engine at 52.39 ms, the int8 body at 31.47 ms, so **1.66476 x** against `bar_int8_wall` = 1.2x. The same session WITHOUT the tail-CSC row read 1.557835 x, i.e. the row is worth 1.068637 x on the int8 arm.

With `hevc_nvenc` in the loop - the product's own condition - the same two arms read 52.59 ms and 31.54 ms, so the product's ratio is **1.667406 x**. On this clip the encoder is almost entirely overlapped: the loop sits within 0.2 ms of the null sink in both arms.

## What this file is NOT

- **The encoder is in the ratio, not only in the stand-in.** The product's number above is quoted from
  `hevc_nvenc` (JC013). On this clip the encoder is almost entirely overlapped - both arms sit within
  0.2 ms of their null-sink numbers - so the ~10 ms penalty that earlier reports warned about is
  CONTENT-dependent, not a constant: a claim made from one clip's null-sink loop is a claim about that
  clip, and the more mobile the frame, the more the encoder can cost.
- **One clip, one model, one card.** The gains were replayed on three clips (1920x1080 8-bit, native
  1440x1080 10-bit, and a same-shot cut) and hold within 20% of their size; other cards are untested and
  the lane has external evidence that int8's advantage is a property of the card.
- **Not a picture claim.** The picture is judged elsewhere, on 494-frame lossless pairs, and the two
  boundary changes are byte-identical (`int8-tail-hop.txt`, `int8-hop-halo-chain.txt`).
- **Not the whole story of the hop.** The hop has its own record: 30 configurations in seven families,
  ncu counters, DRAM/L2 traffic, and the page of levers that lost (`int8-hop-occupancy.txt`,
  `int8-hop-dram.txt`, `int8-hop-ncu.txt`).

## Regenerate

```bash
python harness/scripts/frame_budget.py --logs harness/docs/wsp/verified/logs/wall-final --arm int8
```

