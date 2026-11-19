# The frame budget, as it stands

**Generated** by `scripts/frame_budget.py` - do not edit. Every number is parsed from the render
log named below or resolved by id from `docs/wsp/verified/records/`.

Source run: `docs/wsp/verified/logs/wall-defaults/null/bd_csm_same_int8_r3_null.log` (539 frames, null sink, reuse off).

## The body, from the app's own phase counter

| line | ms | share of body |
|---|---|---|
| head | 0.70 | 2% |
| quantise | 0.00 | 0% |
| hops | 25.40 | 89% |
| tail | 2.60 | 9% |
| **BODY TOTAL** | **28.60** | 100% |

Tail split on the same run: dequant 0.00, conv 0.90, shuffle 0.75, residual 0.90, sum 2.56 ms.

## The frame

| what | ms |
|---|---|
| body (above) | 28.60 |
| pack + queue | 3.42 |
| **render loop** | **35.17** |
| everything else (reader, wrapper, writer wait) | +3.15 |

## What is in this frame because it was measured and adopted

| change | saved on the loop | records |
|---|---|---|
| head as one fused kernel | -5.27 ms (11.9%) | int8-head-fusion:H006 / H007 |
| tail as a 17th int8 hop | -3.16 ms (6.9%) | int8-tail-hop:C001 / C002 |
| tail shuffle + residual folded in | -0.80 ms (2.0%) | int8-tail-shuffle:Q005 / Q006 |

## What was measured and deliberately NOT adopted

| lever | measured result | record |
|---|---|---|
| epilogue divide folded | -0.655 ms (the arm that lost) | int8-epilogue-fold:R002 |
| 6-bit codes between hops | 2.71431 % (the arm that lost) | int8-sub8-codes:S005 |
| vectorised loads / weight hoist | 1.772 ms (the arm that lost) | int8-hop-loads:L002 |
| persistent-CTA grid | 3.709 ms (the arm that lost) | int8-hop-persistent:N002 |
| cache policies | 1.594 ms (the arm that lost) | int8-hop-cache:C002 |
| 2-D CTA tiles | 1.55 ms (the arm that lost) | int8-hop-tiles:T004 |
| output-channel split | 2.632 ms (the arm that lost) | int8-hop-csplit:X002 |
| M x 2 batching | 1.764 ms (the arm that lost) | int8-hop-batch:Y002 |
| pack ring (pinned slots) | 34.36 ms (the arm that lost) | int8-pack-ring:K002 |

## The ratio this produces

See `docs/wsp/int8-wall-defaults.txt`: 1.399x and 1.481x on a null sink, 1.511x and 1.506x
under `hevc_nvenc`, against `bar_int8_wall` = 1.2x.

