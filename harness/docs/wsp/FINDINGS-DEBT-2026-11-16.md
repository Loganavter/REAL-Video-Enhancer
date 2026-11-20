# Existing findings, audited by the new harness

Generated 2026-11-16, **verbatim**, by:

```
PYTHONPATH=. python3 -m harness.migrate "harness/docs/claims-*.txt"
```

Everything below the line is the tool's output, unedited. It is not a summary: the gate fails while
decision-bearing findings are unprotected, so this file can be regenerated as a check rather than read
as a document (`harness/docs/backlog/queue.yaml` -> `rve-findings-rederivation-2026-11-16`).

What the audit can decide with no artifact at all: a claim the project's own auditor fails, a claim name
that carries two values, and a directive that no parser reads. Everything else needs the artifact it was
computed from, and the corpus cites none.

---

213 claim lines (205 names), 16 bars, 80 names cited
  withdrawn by the source file: 19   (claims-example.txt)
  protected:                    0
  CONFLICTED (two values, one name/population): 11
  unverifiable (no tuple):      183
  decision-bearing (cited by a relation): 26, not protected: 26
  derivation-only (nothing cites it; it has a free-text @derived): 38

One name, two values - a re-measurement or a contradiction, and nothing in the file says which:
  body_vs_app                  0.82 x (claims-int8-stage2.txt:49); 0.953 x (claims-int8-stage4.txt:115)
  engine_idle_ms               79.86 ms (claims-int8-stage4.txt:52); 80.47 ms (claims-int8-stage2.txt:8)
  engine_nvenc_ms              91.65 ms (claims-int8-stage4.txt:56); 91.91 ms (claims-int8-stage2.txt:12)
  pipeline_share               0 % (claims-int8-stage1.txt:9); 1.87 % (claims-example.txt:30)
  ratio_idle                   1.322 x (claims-int8-stage4.txt:54); 1.489 x (claims-int8-stage2.txt:10)
  ratio_nvenc                  1.328 x (claims-int8-stage4.txt:58); 1.505 x (claims-int8-stage2.txt:14)

Three or more values on one population LABEL, and the label does not say which arm each belongs to (fp16 / int8 / int8-fused). Not a contradiction - a population that is not a population:
  [ms] one process, host timeline, steady state, 1920x1080        42.75 (int8_window_ms2), 46.52 (int8_window_ms), 51.06 (fp16_window_ms)

What the legacy auditor says about these files today:
  claims-example.txt            11 failure(s)  9: unit families differ - margin_chainsaw is pp, reuse_chainsaw is %
  claims-int8-stage1.txt         0 failure(s)  PASS
  claims-int8-stage2.txt         0 failure(s)  PASS
  claims-int8-stage4.txt         2 failure(s)  287: comp_d2_chainsaw = 0.39 % is not below 0.1 (bar_share) on 'composition depth 2, 6 pai
  claims-int8-stage5.txt         0 failure(s)  PASS

  missing field -> findings:
    items_sha    183
    instrument   183
    config       183
    session      183
    statistic    183
    artifact     183
    n            176

DECISION-BEARING AND UNPROTECTED (re-derive before using again):
  body_vs_app                       0.82 x      n=? [conflicted; below@claims-int8-stage2.txt:53, below@claims-int8-stage4.txt:127, below@claims-int8-stage4.txt:180, below@claims-int8-stage4.txt:219, below@claims-int8-stage4.txt:235, derived-text@claims-int8-stage2.txt:50, derived-text@claims-int8-stage4.txt:116]  claims-int8-stage2.txt:49
  body_vs_app                      0.953 x      n=? [conflicted; below@claims-int8-stage2.txt:53, below@claims-int8-stage4.txt:127, below@claims-int8-stage4.txt:180, below@claims-int8-stage4.txt:219, below@claims-int8-stage4.txt:235, derived-text@claims-int8-stage2.txt:50, derived-text@claims-int8-stage4.txt:116]  claims-int8-stage4.txt:115
  pipeline_share                       0 %      n=? [conflicted; below@claims-example.txt:35, below@claims-int8-stage1.txt:26, below@claims-int8-stage1.txt:27]  claims-int8-stage1.txt:9
  ratio_violet_4k                  0.552 x      n=? [unverifiable; below@claims-int8-stage1.txt:30, derived-text@claims-int8-stage1.txt:19]  claims-int8-stage1.txt:18
  shortfall_idle                   -3.45 ms     n=? [unverifiable; below@claims-int8-stage2.txt:36, derived-text@claims-int8-stage2.txt:33]  claims-int8-stage2.txt:32
  shortfall_nvenc                  -4.59 ms     n=? [unverifiable; below@claims-int8-stage2.txt:37, derived-text@claims-int8-stage2.txt:35]  claims-int8-stage2.txt:34
  kill_excess                      -9.13 ms     n=? [unverifiable; below@claims-int8-stage4.txt:70, derived-text@claims-int8-stage4.txt:63]  claims-int8-stage4.txt:62
  delivered_ratio                  0.484 x      n=? [unverifiable; below@claims-int8-stage4.txt:203, derived-text@claims-int8-stage4.txt:197]  claims-int8-stage4.txt:196
  fp16_window_ms                   51.06 ms     n=? [unverifiable; ratio@claims-int8-stage4.txt:218, ratio@claims-int8-stage4.txt:234]  claims-int8-stage4.txt:208
  futsuu_short                     -3.45 ms     n=? [unverifiable; below@claims-int8-stage5.txt:32, derived-text@claims-int8-stage5.txt:29]  claims-int8-stage5.txt:28
  simulation_share                     0 %      n=? [unverifiable; below@claims-example.txt:36]  claims-int8-stage1.txt:10
  chainsaw_composition              0.01 %      n=? [unverifiable; below@claims-int8-stage4.txt:21]  claims-int8-stage4.txt:9
  adv_chainsaw_comp                0.022 %      n=? [unverifiable; below@claims-int8-stage4.txt:22]  claims-int8-stage4.txt:10
  c10_comp                         0.011 %      n=? [unverifiable; below@claims-int8-stage4.txt:23]  claims-int8-stage4.txt:11
  body_share_512                       0 %      n=? [unverifiable; below@claims-int8-stage4.txt:44]  claims-int8-stage4.txt:40
  body_share_1080                      0 %      n=? [unverifiable; below@claims-int8-stage4.txt:43]  claims-int8-stage4.txt:41
  body_vs_engine_share             1e-05 %      n=? [unverifiable; below@claims-int8-stage4.txt:45]  claims-int8-stage4.txt:42
  delivered_share                0.07592 %      n=? [unverifiable; below@claims-int8-stage4.txt:202]  claims-int8-stage4.txt:198
  int8_window_ms                   46.52 ms     n=? [unverifiable; ratio@claims-int8-stage4.txt:218]  claims-int8-stage4.txt:209
  render_share_epilogue          0.07024 %      n=? [unverifiable; below@claims-int8-stage4.txt:225]  claims-int8-stage4.txt:222
  int8_window_ms2                  42.75 ms     n=? [unverifiable; ratio@claims-int8-stage4.txt:234]  claims-int8-stage4.txt:232
  comp_spread_frame                0.006 %      n=? [unverifiable; below@claims-int8-stage4.txt:258]  claims-int8-stage4.txt:255
  comp_spread_violet                0.05 %      n=? [unverifiable; below@claims-int8-stage4.txt:259]  claims-int8-stage4.txt:257
  comp_d2_chainsaw                  0.39 %      n=6 [unverifiable; below@claims-int8-stage4.txt:287]  claims-int8-stage4.txt:278
  comp_d2_violet                   0.166 %      n=? [unverifiable; below@claims-int8-stage4.txt:288]  claims-int8-stage4.txt:280
  realesr_share                  0.00058 %      n=? [unverifiable; below@claims-int8-stage5.txt:19]  claims-int8-stage5.txt:18

16 bars: none names what it is a bar of (bar.of is absent from the legacy format), so gates.below refuses every one of them by design
NOT checked: whether a label is true; whether the item set is the one the decision needs; whether the artifact still exists; free-text @derived arithmetic

GATE: FAIL
