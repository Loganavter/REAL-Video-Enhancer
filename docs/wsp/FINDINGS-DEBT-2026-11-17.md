# Findings debt, 2026-11-17: what the harness protects now, and what is accepted as debt

Companion to `FINDINGS-DEBT-2026-11-16.md`, which is the machine-generated inventory. This one is the
human decision record for `rve-findings-rederivation`, whose acceptance has two admissible endings:
`harness.migrate` reaches PASS, or every remaining unverifiable claim is listed as accepted debt **with
the reason**. The second ending is what this file takes, plus three items actually re-derived today.

## State of the gate, measured today

```
PYTHONPATH=. python3 -m harness.migrate "docs/claims-*.txt"
 213 claim lines (205 names), 16 bars, 80 names cited
 withdrawn by the source file: 19      protected: 0
 CONFLICTED within one run:      0      unverifiable (no tuple): 194
 decision-bearing, not protected: 26
 same name in two runs (re-measurement, no session to say which is current): 5
 three values on one population LABEL (fp16 / int8 / int8-fused): 1 label
GATE: FAIL
```

Nothing about that changed today, and nothing about it *can* change by translation: a legacy claim's
name (`ratio_idle`, `body_vs_app`, `engine_idle_ms`) corresponds to no surviving record, so protecting
it means re-measuring it, not renaming it.

## What the harness does protect: thirteen files, all green

`docs/wsp/verified/*.txt` + `records/*.jsonl`, checked in one loop (see `README.md`):

| file | claims | state |
|---|---|---|
| int8-block | 2 | PASS |
| int8-body-ve | 3 | PASS |
| int8-pipeline-ve | 3 | PASS |
| engine-fp16 | 6 | PASS |
| engine-int8 | 43 | PASS |
| model-prelu | 3 | PASS |
| residual-scale | 6 | PASS |
| residual-scale-ve | 6 | PASS |
| pyramid-budget (new) | 8 | PASS |
| cone-budget (new) | 17 | PASS |
| prelu-forms (new) | 4 | PASS |
| int8-tactics (new) | 2 | PASS |
| streams (new) | 2 | PASS |

Two of these were failing their **own documented command** until today, and both failures were
documentation, not measurement: `int8-block.txt` had no run line at all, and `model-prelu.txt` plus
`residual-scale.txt` pointed their `--records` at `/tmp/opencode/...`, i.e. at tmpfs - the same class of
loss that took the Lean toolchain this week. Every run line now names the committed copy and passes
`--bars docs/claims-bars.json`; without that flag two files fail on bars that do exist.

## Re-derived today (records in `docs/wsp/verified/`)

- **E29** - the L2 pyramid's buffer arithmetic and its roofline (`pyramid-budget.txt`).
- **E36** - the cone/cylinder work ratio with its 1-D control (`cone-budget.txt`).
- **B3** - the three activation forms at C=32, plus the discovery that the C=32 build is bimodal
  (`prelu-forms.txt`, and the ledger's B3 section for the rebuild evidence).
- **E33 (first pass)** - the int8 body's per-convolution rate is 79.2 TOPS, not 91 (`int8-tactics.txt`).
- **C (strips)** - two streams buy 1.00x of throughput, so the halo tax is unpayable (`streams.txt`).

## Accepted debt, by reason

1. **Engine timings whose plan files are gone.** The recorded engine numbers were built by a TRT 10.12
   runtime that now exists only inside `~/.cache/opencode/venvs/trt310`; the plans themselves lived in
   `/tmp` and did not survive. Re-measuring is exactly what `E33` and `B2` do - so those claims are
   superseded by measurement, not by text. Reason: **artifact destroyed, instrument still available.**
2. **Picture claims from the int8 lane.** `engine-int8.txt` records that the same recipe differs by
   10-16 dB depending on three knobs nobody wrote down (calibration percentile, Q/DQ placement, the
   frame compared on). Protecting them requires a recipe decision first. Reason: **the record is
   under-specified at the recipe, not at the number.**
3. **The five two-valued names** (`body_vs_app` 0.82/0.953, `engine_idle_ms` 79.86/80.47,
   `engine_nvenc_ms` 91.65/91.91, `ratio_idle` 1.322/1.489, `ratio_nvenc` 1.328/1.505) are
   re-measurements in different sessions; the files now carry `@session=<stage>`, so a decision citing
   one must name its session. Reason: **session drift, documented, not resolved.**
4. **The 16 legacy bars** have no `of` field, so `gates.below` refuses all of them by design. Reason:
   **the bar does not say what it is a bar of** - that is the refusal working, not a defect to patch.

## What is deliberately not done

No claim is marked withdrawn wholesale. The item's own refutes clause allowed that, but it also names
the better resolution: *re-measure what the decision actually needs*. Three items were re-measured
today, and the two live lanes (E33, B2) are re-measurements of exactly the claims a decision would
cite.

## Appendix: the 26 decision-bearing claims, named

Generated from `harness.migrate` itself (not typed by hand). "n=?" is the record's own complaint: the
number was never attached to a sample size, which is why it cannot be a verdict's evidence.

| name | value | n | source |
|---|---|---|---|
| `body_vs_app` | 0.82 x | ? | claims-int8-stage2.txt:49 |
| `body_vs_app` | 0.953 x | ? | claims-int8-stage4.txt:115 |
| `pipeline_share` | 0 % | ? | claims-int8-stage1.txt:9 |
| `ratio_violet_4k` | 0.552 x | ? | claims-int8-stage1.txt:18 |
| `shortfall_idle` | -3.45 ms | ? | claims-int8-stage2.txt:32 |
| `shortfall_nvenc` | -4.59 ms | ? | claims-int8-stage2.txt:34 |
| `kill_excess` | -9.13 ms | ? | claims-int8-stage4.txt:62 |
| `delivered_ratio` | 0.484 x | ? | claims-int8-stage4.txt:196 |
| `fp16_window_ms` | 51.06 ms | ? | claims-int8-stage4.txt:208 |
| `futsuu_short` | -3.45 ms | ? | claims-int8-stage5.txt:28 |
| `simulation_share` | 0 % | ? | claims-int8-stage1.txt:10 |
| `chainsaw_composition` | 0.01 % | ? | claims-int8-stage4.txt:9 |
| `adv_chainsaw_comp` | 0.022 % | ? | claims-int8-stage4.txt:10 |
| `c10_comp` | 0.011 % | ? | claims-int8-stage4.txt:11 |
| `body_share_512` | 0 % | ? | claims-int8-stage4.txt:40 |
| `body_share_1080` | 0 % | ? | claims-int8-stage4.txt:41 |
| `body_vs_engine_share` | 1e-05 % | ? | claims-int8-stage4.txt:42 |
| `delivered_share` | 0.07592 % | ? | claims-int8-stage4.txt:198 |
| `int8_window_ms` | 46.52 ms | ? | claims-int8-stage4.txt:209 |
| `render_share_epilogue` | 0.07024 % | ? | claims-int8-stage4.txt:222 |
| `int8_window_ms2` | 42.75 ms | ? | claims-int8-stage4.txt:232 |
| `comp_spread_frame` | 0.006 % | ? | claims-int8-stage4.txt:255 |
| `comp_spread_violet` | 0.05 % | ? | claims-int8-stage4.txt:257 |
| `comp_d2_chainsaw` | 0.39 % | 6 | claims-int8-stage4.txt:278 |
| `comp_d2_violet` | 0.166 % | ? | claims-int8-stage4.txt:280 |
| `realesr_share` | 0.00058 % | ? | claims-int8-stage5.txt:18 |
