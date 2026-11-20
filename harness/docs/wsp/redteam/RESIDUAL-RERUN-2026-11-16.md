# Re-run of the 22 red-team repros against the patched harness

Cell: `/home/jorj/.cache/opencode/clones/rve-rt-residual/REAL-Video-Enhancer`
Harness: `harness/` (core, gates, run, cli, migrate, astcheck)

## How this was run

Each script was executed as

```
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3 .agent/inputs/repro_<n>_<name>.py
```

from the clone root. The repros hard-code `sys.path.insert(0, "/home/jorj/Загрузки/projects/REAL-Video-Enhancer/scripts")`,
so I first confirmed the two trees are byte-identical (`sha256sum` of all seven modules matches);
nothing below imported an older harness.

Two properties of the scripts had to be worked around, and both are visible in the output:

- Several scripts print a *conclusion* line unconditionally (`-> 'the mechanism was varied' certified by
  two frames of one unchanged configuration`, `-> a cross-session number enters the store as 'derived'`).
  Those lines are stale narration. Every verdict
  below is taken from the runtime result — the `problems:` list, the `scope.checked:` line, or the raised
  exception — not from the narration.
- Several scripts crash on the first refusal (`below()` raising where the script assumed it would pass),
  so their later sub-claims never execute. Those sub-claims were re-run as the same code path in a
  short probe; every such line is marked `probe:` in the table, and no verdict rests on a script
  merely *not printing* something.

`exit` code is not used as an oracle anywhere: repro_5, repro_16, repro_13 and repro_17 print a bypass
and exit 0, while repro_3 and repro_9 exit 0 with a refusal already in the text.

## Verdicts

| repr | F | verdict | evidence line from the run |
|---|---|---|---|
| 1 | F1 | **CLOSED** | `harness.core.RecordError: ratio: units differ - R001 is ms, R002 is s (same family, different scale: 2000 ms / 2 s is not 2x) - convert into a declared derived record, never inside a relation` |
| 1 | F1 | **CLOSED** | probe: `below(ms bar) -> REFUSED: R002: 2 s cannot be judged against bar twenty_minutes = 500 ms (same family, different scale)` |
| 1 | F1 | **CLOSED** | probe: `below(MiB bar) -> REFUSED: R003: 1 GiB cannot be judged against bar vram = 500 MiB (same family, different scale)` |
| 2 | F9 | **CLOSED** | `harness.core.RecordError: ratio: populations differ - R001 on 'tiles [0d2544d03b8f5f78] n=3', R002 on 'rendered frames [0e6232eea7b6f32f] n=3'` (kind is in the hash: `empty 'rendered frames' sha: 80bd94fd4e2f28aa` vs `empty 'tiles' sha: 86522746bb6a3ae7`) |
| 2 | F9 | **OPEN** | probe: `ratio of two zero-item records -> 0.5` (`empty n: 0`) |
| 3 | F3 | **CLOSED** | `verify_derived() -> ["R003: formula 'speedup_model_v3' is not one this verifier can recompute - an unrecomputable derived number is not a result"]` |
| 3 | F3 | **CLOSED** | probe: `verify_derived() -> ["R002: formula 'twice_of' needs two inputs, has 1"]` (no IndexError) |
| 4 | F2 | **CLOSED** | `harness.core.RecordError: R001 rests on n=1 and bar pixel_share needs n>=2 - one sample is a smoke test, not a verdict` |
| 4 | F2 | **CLOSED** | probe: `below(7 levels, invented bar.of) -> REFUSED: bar whatever: of='the thing i measured' is not one of ('share', 'peak', 'frame_time', 'engine_time', 'aa_delta', 'pixel_defect')` |
| 4 | F2 | **OPEN** | probe: `below(peak 7% labelled of='share', 8% share threshold) -> PASSED` |
| 5 | F8 | **CLOSED** | `problems: ["C1: C1: 'INT8 tensor-core fusion with fused PReLU' cites two different populations ('rendered frames [709fd72303a72d60] n=3' vs 'rendered frames [e30e84588a2d57a6] n=3') - a population difference is the error, not the mechanism"]` |
| 6 | F10 | **CLOSED** | `verdict over worst_of=0.009% and median=99% -> problems: ['V1: V1: verdict mixes 2 (instrument, config, session, population) keys - each claim may be true and the verdict is not']` |
| 6 | F10 | **CLOSED** | `verdict with a fabricated derived record of value 0.0 (formula 'model') -> problems: ["V2: V2: a verdict is taken over measurements; cites [('R003', 'derived')] ..."]` |
| 6 | F10 | **CLOSED** | `verdict over % and ms with the same digits -> problems: ['V3: V3: verdict mixes 2 (instrument, config, session, population) keys ...']` |
| 6 | F10 | **OPEN** | `equal over two DIFFERENT item sets -> problems: []` |
| 7 | F23 | **CLOSED** | `ratio (cross_session=True) tier = assumed` / `delta (cross_session=True) tier = assumed  value = -5.67 ms` |
| 8 | F11 | **OPEN** | `(a) tier: measured run: '' artifact: '' -> a 'measured' record that names no run and no artifact` + probe: `below(mutated, n=2, no provenance) -> PASSED` |
| 8 | F11 | **CLOSED** | `(b/c) refused: R001 rests on n=1 and bar pixel_share needs n>=2 - one sample is a smoke test, not a verdict` |
| 8 | F11 | **CLOSED** | `harness.core.RecordError: /tmp/.../records.jsonl:1: R001 claims items_sha 0000000000000000 but its items hash to 80bd94fd4e2f28aa - the file was edited` |
| 9 | F4 | **CLOSED** | `sci_notation.txt -> exit 1` / `FAIL (1): 4: F1 recorded 0.009, claimed 0.001` (the `1e-3` claim is read and compared) |
| 9 | F4 | **CLOSED** | `typo_directive.txt -> exit 1` / `4: directive not read by this checker: @bounds F1 pixel_share` |
| 9 | F4 | **CLOSED** | `unparsed_relation.txt -> exit 1` / `4: directive not read by this checker: @ratio F1 / F2 = 0.5` |
| 9 | F4 | **CLOSED** | `unit_glued.txt -> exit 1` / `4: directive not read by this checker: @claim F1 0.009% "rendered frames"` |
| 10 | F16 | **CLOSED** | `FAIL (1): 4: summands disagree on (instrument, config, session, statistic): [('acceptance_stat', 'cfg', 's-2025-06-01T00:00:00Z', 'mean'), ...] - they are not parts of one partition` |
| 10 | F16 | **OPEN** | probe: `SUM (3 item sets, one session/statistic) exit: 0` / `PASS` |
| 11 | F11 | **OPEN** | `loaded ids: ['R001', 'R002']  seq after load: 0` / `(a) ratio on a freshly loaded store -> duplicate record id R001` |
| 11 | F11 | **CLOSED** | `harness.core.RecordError: /tmp/.../dupes.jsonl:2: duplicate record id R001 - a hand-edited file cannot silently overwrite a record` |
| 11 | F11 | **OPEN** | probe: `a record with id '' loads under key ''` |
| 11 | F11 | **CLOSED** | probe: `Store.load` of a measured record with `run=''`/`artifact=''` -> `RecordError: a measured record must name its run or its artifact` |
| 12 | F12 | **CLOSED** | `harness.run.RunError: probe fp16 kernel correctness: no run and not declared synthetic - a probe that never ran is not a result` |
| 13 | F17 | **OPEN** | `status: ok claimable: True` / `parsed.fields: {} missing: ['registers', 'static_smem', 'dynamic_smem']` |
| 13 | F17 | **OPEN** | `record accepted as MEASURED with artifact: /nonexistent/capture-that-was-never-taken.txt` + probe: `verdict over two digest-less artifact records -> problems: []` |
| 13 | F17 | **CLOSED** | probe: `below(record citing /nonexistent artifact) -> REFUSED: R001 cites artifact '/nonexistent/capture-that-was-never-taken.txt' with no digest - a record that names a file it never read is the class-5 error` |
| 13 | F17 | **OPEN** | `empty inspector json: claimable: True total: 0 missing: ['total', 'fused_mylin', 'reformat', 'pointwise', 'convolutions']` |
| 14 | F5 | **CLOSED** | `run 2 (same claims + records): exit 1 GATE: FAIL` / `['protected: 0']` |
| 14 | F5 | **CLOSED** | `conflict, with its own emit:   exit 1 GATE: FAIL   (CONFLICTED line: ['CONFLICTED (two values, one name/population): 2'] )` |
| 15 | F13 | **CLOSED** | `(a) claims-example-2026.txt    exit 1  GATE: FAIL` (and `(a) claims-example-archive/  exit 1  GATE: FAIL`) |
| 15 | F14 | **CLOSED** | `(b) cause_unquoted.txt   exit 1  GATE: FAIL  ['decision-bearing (cited by a relation): 0, not protected: 0']` (fails on the unparsed-directive count) |
| 15 | F19 | **OPEN** | `records emitted: [('ratio_idle', 1.489)] -> the second value was silently dropped` |
| 16 | F20 | *no bypass* | `dead store suppressed by an unrelated '# noqa: E501': [Finding(kind='dead store', name='buf', ...)]` (the marker sits on line 2, the finding on line 3 — reported in both versions) |
| 16 | F20 | **OPEN** | `dead store suppressed by a bare '# noqa':            []` |
| 16 | F20 | **OPEN** | `use-before-assignment via '+=':                     []` |
| 16 | F20 | **OPEN** | `dead store through an attribute:                   []` |
| 17 | F22 | **OPEN** | `refuse() with a hand-written AA(identical=True) -> ACCEPTED; refusal record emitted: ['R004']` |
| 17 | F22 | **OPEN** | `aa_gate on artifact-only records -> a measured record must name its run or its artifact` |
| 18 | F6 | **CLOSED** | `harness.core.RecordError: legacy_share is assumed - an assumed number is not evidence for a bar` |
| 18 | F6 | **CLOSED** | probe: `ratio of two assumed records -> tier: assumed value: 0.2` |
| 18 | F6 | **CLOSED** | probe: `equal over two assumed records -> problems: ['E1: E1: equality of an assumed number is not evidence']` |
| 18 | F6 | **CLOSED** | probe: `cause over two assumed records -> problems: ['C1: C1: a cause needs two measurements']` |
| 18 | F6 | **CLOSED** | probe: `verdict over two laundered ratios -> problems: ["V9: V9: a verdict is taken over measurements; cites [('R001', 'assumed'), ('R002', 'assumed')] ..."]` |
| 19 | F7 | **CLOSED** | `harness.core.RecordError: non-finite value nan: it would pass every comparison` |
| 19 | F7 | **CLOSED** | probe: `Store.load of a NaN line -> REFUSED: RecordError non-finite value nan: it would pass every comparison` |
| 20 | F4 | **CLOSED** | `(a) harness-emitted claims line: @claim SHARE 1.23e-05 % "rendered frames [709fd72303a72d60] n=3" measured` / `(a) cli exit: 0` / `PASS` (no "directive not read") |
| 20 | F21 | **CLOSED** | `harness.core.RecordError: R001: a claim after a graph edit must cite layer counts (total, fused_mylin, reformat, pointwise), not only time (got {})` |
| 20 | F15 | **CLOSED** | `(c) scope.checked: ['ratio'] -> finalize(): REFUSED: green over an uncovered scope: ['below', 'verdict'] expected but not checked` |
| 20 | F15 | **CLOSED** | `(d) sc2.expected is now {'approximate', 'below'}` / `empty claims ... finalize(): REFUSED: green over an uncovered scope` |
| 21 | — | *no bypass* | `timed_out: True status: error error_kind: timeout wall_s: 0.50` / `grandchild pid 53768 alive after 2.5s: False` / `-> the process GROUP was killed; no bypass found for this claim` |
| 22 | F18 | **CLOSED** | `name_reuse computed INSIDE migrate(): []` / `gate exit: 0 ['GATE: PASS']` (live claim is `unverifiable`, not `conflicted`) |

## Count

- **Per repro, 22 rows:** 12 fully CLOSED (1, 3, 5, 7, 9, 12, 14, 18, 19, 20, 21, 22), 1 fully OPEN (17),
  9 mixed (2, 4, 6, 8, 10, 11, 13, 15, 16).
- **Per sub-claim, 58 rows:** 40 CLOSED, 2 rows where no bypass exists (16's `# noqa: E501` line and
  repro_21's process-group control), 16 OPEN.

Counted per repro instead of per row, the picture is 12 closed, 1 open, 9 partially closed — an OPEN
sub-claim in a repro whose headline bypass was closed (2, 4, 6, 8, 10, 11, 13, 15, 16) is still an OPEN
row, and the three that matter are F2's self-declared `of`, F11's post-emit mutation, and F17's
`claimable()`.

Of the eight high findings, F1, F3, F4, F5, F6, F7 and F8 are closed in the form the red team
demonstrated; F2 is closed for n=1 and for an invented `bar.of`, and still open for a peak that writes
`of='share'`. The provenance-deletion row is F11 (medium).

## What each OPEN row still permits

1. **repro_2 / F9 — zero-item populations.** `ratio` of two records whose item set is empty returns
   `0.5`; no rule requires a non-empty population, so a number measured on nothing can be a numerator
   or a denominator.
2. **repro_4 / F2 — self-declared `of`.** `peak 7%` passes an `8%` share threshold when the record
   writes `of='share'`; the vocabulary check constrains the string, not the measurement that produced it,
   so the class-7 retraction is one field away.
3. **repro_6 / F10 — `equal` skips the population.** Two records with the same value and unit on
   `frames [1,2]` and `frames [7,8,9]` are "equal" with `problems: []`.
4. **repro_8 / F11 — mutation after emit.** `Record` is still a mutable dataclass; `validate()` does not
   re-check the measured-provenance rule, so a record emitted with an artifact and then stripped to
   `run='' artifact=''` passes `below()` at n≥2.
5. **repro_10 / F16 — `@sum` ignores the item set.** The sum key is
   `(instrument, config, session, statistic)`; three parts measured on three different tile sets in one
   session with one statistic sum to 100% and the file exits 0.
6. **repro_11 / F11 — `seq` is not restored on load.** The first relation on a freshly loaded store
   allocates `R001` again and is refused as a duplicate, so `@ratio` cannot be checked against a store
   the CLI dumped itself unless ids were hand-named.
7. **repro_11 / F11 — empty id.** A JSONL line with `"id": ""` loads under the key `''`.
8. **repro_13 / F17 — `claimable()` accepts an all-missing parse.** A `Schema` whose `missing` list
   holds every field is claimable, so "the parser ran" is still evidence that "the artifact contained
   the thing".
9. **repro_13 / F17 — artifact is a free string.** `Store.emit` accepts a measured record citing
   `/nonexistent/...` with no digest, and `check_claims`' `verdict` branch never calls `validate()`, so
   two such records carry a verdict (`problems: []`); only `below()` refuses them.
10. **repro_13 / F17 — empty engine inspector.** `total: 0` with `missing=[...]` is claimable, so
    "reformat 0 of 0 layers" can be cited from a dump with no layers.
11. **repro_15 / F19 — one record per name.** `emit_records` skips a name already in the store, so of two
    contradictory values under one name only the first is written and the contradiction never reaches
    the store.
12. **repro_16 / F20 — bare `# noqa`.** `NOQA` still contains the bare token, so `# noqa` on the second
    store's line silences the dead-store check.
13. **repro_16 / F20 — `+=`.** `used_before_assignment` still collects only `ctx=Load` nodes, so
    `c2 += 1` before `c2` is assigned is not reported (the `_loads` fix applies to `dead_stores` only).
14. **repro_16 / F20 — attribute dead stores.** `self.buf = ...` twice is still not a store the check
    can see.
15. **repro_17 / F22 — forged AA.** `refuse()` accepts a hand-built `AA(identical=True, ...)`, so a
    refusal can be recorded against a base that never passed its own A/A — the class-3 shape, certified
    by the gate built to prevent it.
16. **repro_17 / F22 — artifact-only A/A.** `aa_gate` emits `run=ra.run` only, so a base whose
    provenance is a capture cannot pass its own A/A: the gate refuses to run on exactly the records
    `run.py` tells you to create when there is an artifact instead of a command.

## Note on F15 (repro_20 c)

`check_claims` still calls `cover()` before the rule runs, so `scope.checked` reports a refused check as
covered. The exploitable outcome is gone for the default and CLI callers because `expected` is now merged
into the supplied scope and the refusal lands in `problems`; a caller that passes `expected=()` explicitly
can still `finalize()` green over a refused ratio
(`probe: scope.checked: ['ratio'] -> finalize(): GREEN although the only ratio was refused`). I am
recording that as the residual of a CLOSED row rather than an OPEN one because no caller in the harness
does it.
