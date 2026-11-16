# The acceptance statistic, stated formally

Written 2026-11-16 because the Lean spike (`rve-lean-statistic-spike-2026-11-16`) cannot run here:
there is no Lean toolchain on this host (`lean`, `elan`, `lake` all absent), and installing one is a
system change with a network dependency. The spike's declared deliverable was never the proof - it was
**the list of assumptions that had to be made explicit before a statement could be written**. That list
is below, and it is the part the harness can act on today.

## The statement

For a render `r`, a reference `g`, a channel set `C = {R, G, B}` and a threshold `t = 8` levels:

```
off(p)        = max_{c in C} |r_c(p) - g_c(p)|          -- the worst channel, in levels
bad(r, g)     = { p : off(p) > t }                       -- the pixels the bar counts
share(r, g, S)= |bad ∩ S| / |S|                          -- S is the frame set (the population)
decision      = worst_of_{f in S} share(f) < 0.1%
```

Three properties worth having, and what each one costs to state:

1. **Monotonicity.** `worst_of` over `S ∪ {f}` is `max(worst_of(S), share(f))`, so a larger sample can
   only raise the number. This is why "it passed on 13 frames" is not a bound on 19 - the harness
   enforces it as `statistic = worst_of` on the record, not as a proof.
2. **Relabel invariance.** The comparison is invariant under renaming a population **iff the item set
   is equal**; a population with the same name and different items is a different population. This is
   exactly class 1 of the retraction list, and the harness enforces it with `items_sha`, which now
   includes the population `kind` (`Population.items_sha`).
3. **Determinism.** `share` is a function of `(r, g, S)` alone: same inputs, same number, no seeds.
   This is what makes the A/A control meaningful - and it is measured, not assumed: 502 of 502 frames
   byte identical.

## The assumptions that had to be named

Each one is a place where the informal statistic was under-specified, and each is now a field or a
documented limitation:

| assumption | why it is not innocent | where it lives now |
|---|---|---|
| levels are measured in the **output** colour space (RGB, after `bt709`), not the source's | the pipeline applies a colour matrix, and an 8-level bar in YUV is a different bar in RGB | `MODEL.md` C1; the render's `config` field |
| `max` over channels, not a per-channel bar | "any channel" is stricter than each channel; the two give different pictures | `off(p) = max_c` above |
| the threshold is on **absolute** difference, not signed | a signed mean hides the film's worst frame (mean 1.74, peak 196) | inventory §4 |
| the sample `S` is **spread** over the clip, not adjacent frames | adjacent frames hide a defect that a viewer meets at a scene change | `population.items` |
| `n >= 2` | an n=1 smoke test has no worst-of | `Bar.min_n` |
| the reference is the **ungated** render of the same configuration | comparing against another configuration measures the configuration | `instrument` on the record |

The last row is the one that produced three retractions: the reference is part of the statistic, not
context around it, and it belongs in the record's provenance rather than in a sentence.

## What to run when a toolchain exists

```
elan toolchain install leanprover/lean4:stable
lake new statistic && cp this file's three statements into Statistic.lean
```

The kill criterion from the queue item still applies: if the three lemmas prove in under four hours,
Lean added nothing a docstring would not have - report that and close the lane. The value was always in
the table above, and it is already extracted.
