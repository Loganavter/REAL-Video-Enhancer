# Ledger inventory: 1 525 numbers, and how many of them can be checked

Produced 2026-11-16 by a read-only research cell over two harvests of `docs/RESEARCH-LEDGER.md`
(`part1.jsonl`: lines 10–1015, 751 entries; `part2.jsonl`: lines 1019–2362, 774 entries), then
**verified by the supervisor**: the headline rates reproduce exactly with the rule stated in §0 —
part1 653/751 (87.0%), part2 770/774 (99.5%), decision-bearing 1 209/1 300. The cell's own
correction of an earlier worker's summary (§7: 47 both-null rows is really 51) also reproduces.

What this file is for: it is the input to `rve-findings-rederivation-2026-11-16`, not a verdict.
A number here is a *candidate finding*, and its provenance fields say what is missing before it can
become a record.

The four numbers that matter:

| | |
|---|---|
| entries | **1 525** on 702 ledger lines |
| verbatim (quote occurs in the ledger) | **1 423 (93.3%)**; the other 102 are paraphrases, and **0** carry a value absent from the file |
| wrong `line` | **72** (59 of them off by +1 — a paragraph's first line instead of its continuation) |
| verified decision-bearing entries with a null `instrument` or `population` | **634** |

My own earlier statement that part1 stopped at line 466 was **wrong** - the JSONL is not sorted by
line, and I read its first and last rows as if it were. The range is complete.

---

# Inventory audit — the two harvests of `docs/RESEARCH-LEDGER.md`

Sources: `.agent/inputs/part1.jsonl` (751 entries, claimed ledger range 1–1018),
`.agent/inputs/part2.jsonl` (774 entries, claimed range 1019–2362). Ledger:
`docs/RESEARCH-LEDGER.md`, 2362 lines.

**Rule (everything below is recomputable from these three files by it).** Normalise a
string by removing backticks and asterisks, collapsing every whitespace run to one space
and lowercasing; an entry is VERIFIED when a 40-character run of its normalised `text`
(the whole text, when the text is shorter than 40 characters) occurs in the normalised
ledger, and its `line` is correct only when such a run sits inside the normalised text of
that one ledger line — otherwise the entry is WRONG-LINE and `true_line` is the line where
the matching run starts.

Two consequences of the rule, stated once because they are not cosmetic:

- 787 of the 1525 texts are shorter than 40 characters (486 in part1, 301 in part2), so for
  them the "40-character run" would be empty and they would be unverifiable by construction.
  The whole text is used as the run instead. §6 shows the rates barely move between
  thresholds 20 and 40, so the choice does not drive the headline.
- A quote wrapped by the ledger's own ~100-column paragraph formatting matches across a
  newline. Those are counted separately as WRAP, not as a wrong line.

---

## 1. Verbatim provenance

| file | entries | verified | unverified | rate |
|---|---|---|---|---|
| part1 | 751 | 653 | 98 | 86.9% |
| part2 | 774 | 770 | 4 | 99.5% |
| **combined** | **1525** | **1423** | **102** | **93.3%** |

Restricted to `decision: true`:

| file | decision entries | verified | unverified | rate |
|---|---|---|---|---|
| part1 | 710 | 621 | 89 | 87.5% |
| part2 | 590 | 588 | 2 | 99.7% |
| **combined** | **1300** | **1209** | **91** | **93.0%** |

The premise in the brief reproduces: part2 is effectively verbatim (99.5%, not 100% — the
four exceptions are named in §6), part1 is not (653/751 = 86.9%; the brief's 86.8% is the
same figure one entry lower on a different long-text tie-break). No row is duplicated: 1525
distinct `(line, value, about)` triples in 1525 rows; 1518 distinct `(line, value, unit)`.

**The unverified entries are paraphrases, not fabrications.** Of the 102 unverified
entries, 88 carry a numeric `value` and every one of those values is present on the claimed
line or within three lines of it (part1: 79 literal, 5 adjacent, 14 with no numeric value;
part2: 2 literal, 2 adjacent). The prose was rewritten; the numbers were not invented. See §5 for the independent value check.

## 2. Line provenance

| class | entries | meaning |
|---|---|---|
| exact | 1210 | quote sits inside the claimed line |
| wrap | 141 | quote crosses the claimed line's own line break (normal paragraph wrapping) |
| **wrong line** | **72** | quote not in the claimed line at all |

Per file: part1 543 exact / 68 wrap / 42 wrong; part2 667 exact / 73 wrap / 30 wrong.
67 of the 72 wrong-line entries are decision-bearing.

The error is systematic, not random:

| `true_line` − `line` | entries |
|---|---|
| −2 | 1 |
| −1 | 10 |
| +1 | 59 |
| +2 | 1 |
| +3 | 1 |

59 of 72 report the line *before* the one that carries the words, i.e. a paragraph's first
line instead of the continuation line the quote actually starts on. Nine of the 72 still
have their `value` on the claimed line (the sentence began there, the figure landed on the
next line), so for those the defect is in `line`, not in `value`.

## 3. Provenance fields, VERIFIED entries only (n = 1423)

| field | null | share |
|---|---|---|
| `instrument` | 733 | 51.5% |
| `population` | 83 | 5.8% |
| `session_or_date` | 1132 | 79.6% |

Per file, verified only:

| file | verified | null instrument | null population | null session |
|---|---|---|---|---|
| part1 | 653 | 270 | 23 | 622 |
| part2 | 770 | 463 | 60 | 510 |

Restricted further to verified `decision: true` (n = 1209): null instrument 623, null
population 76, null session 942; **634 decision-bearing verified entries have a null
instrument or a null population**, 65 of them have both null.

## 4. The 20 weakest VERIFIED decision-bearing entries

Selection: verified **and** `decision: true` **and** (null `instrument` **or** null
`population`) — 634 entries. Ranked both-null first, then instrument-null, then
population-null; inside a bucket by tier (`measured_here`, `derived`, `read_elsewhere`,
`assumed`) and then by line; de-duplicated by `(line, unit)` so one sentence quoted for
four figures appears once per unit. Each is quoted from the ledger line that carries it.

| # | line | value | tier | what it decides | ledger says |
|---|---|---|---|---|---|
| 1 | 39 | 8.11 ms | measured_here | non-GEMM share of a frame | "what is left of the non-GEMM 8.11 ms of the frame is" |
| 2 | 666 | 255 levels | measured_here | a cut frame's miss margin | "a cut frame misses the threshold by 255 levels regardless" |
| 3 | 31 | 58–73 % (value null) | derived | share of borrowed peak achieved | "That is **58-73%** of a *borrowed*" |
| 4 | 39 | 4.7 % | derived | residual non-GEMM share | "**2.68 ms, 4.7%**: the layout kernels…" (next line) |
| 5 | 91 | 128 MAC/byte | derived | ridge point at 74 TF/s and 288 GB/s | "The ridge is **128 MAC/byte** at the borrowed 74 TF/s and 288 GB/s" |
| 6 | 91 | 257 FLOP/byte | derived | arithmetic intensity at the ridge | "(257 FLOP/byte), not 42" |
| 7 | 91 | 24 TF/s | derived | fp32-shader peak implied by the withdrawn 42 | "42 corresponds to a ~24 TF/s fp32-shader peak" |
| 8 | 91 | 3 x | derived | size of the withdrawn error | "the earlier figure was off by 3x in the wrong direction" |
| 9 | 777 | 68 % | derived | share of borrowed peak the conv runs at | "the convolution is already at 68% of the" |
| 10 | 908 | 0.4 ratio | derived | tolerance the proportional error model implies | "0.1% of pixels sit above 19.4 levels, so `e ~ 0.4`" |
| 11 | 913 | 0.93 levels | derived | peak error the model predicts | "predicts a peak error of `0.021 * 44.5 = 0.93` levels" |
| 12 | 915 | 20 x | derived | how wrong the proportional model is | "The mean is off by 20x" |
| 13 | 1041 | 1.53 x | derived | int8 block vs fp16 fused block | "a block is 1.53x cheaper" |
| 14 | 1045 | 1.45 x | derived | projected engine-time speedup | "i.e. 1.45x of engine time and 1.25x of wall" |
| 15 | 1045 | 1.25 x | derived | projected wall-clock speedup | same sentence |
| 16 | 1083 | 14 x | derived | fp16 step fineness vs int8 per-channel | "the fp16 step is ~0.031 and the per-channel int8 step is ~0.435" |
| 17 | 1088 | 0.9 bits | derived | precision gain from asymmetric boundary | "i.e. ~0.9 of a bit, folded into the bias" |
| 18 | 1294 | 1.5 x | derived | per-tap block time penalty, low end | "A 1.5-2x penalty buys 7x on the" |
| 19 | 1307 | 1.77 x | derived | per-tap slowdown, BM 256, nine taps | "1.58-1.77x at BM 256 and 1.30x at BM 128" |
| 20 | 1315 | 20 % | derived | BM 128 slowdown relative to BM 256 | "BM 128 which is already 20% slower" |

The shape of the weakness is the point: 18 of the 20 are `derived` — ratios, shares and
model outputs — with no instrument and no population, so nothing lets a reader recompute
them from the rig the ledger names. Only two are `measured_here` (8.11 ms, 255 levels).
Note also that 4 of the 20 (rows 5–8) are one sentence, and two of those four — the
24 TF/s peak and the 3x error — are numbers *about* a figure the ledger has withdrawn.

## 5. Value check — separate from paraphrase

Rule: the `value` is looked for as a numeric token on the claimed line (digit-boundary
match, trailing-zero forms such as `26.40`/`26.4` accepted), then as the same number in a
formatted rendering (`16,384` / `49 152` / `1.03M` / `260k` / `5e3`), then on lines ±3 of
the claimed line, then anywhere in the file.

| status | all entries | verified only |
|---|---|---|
| value literal on claimed line | 1324 | 1243 |
| value on claimed line, formatted | 21 | 21 |
| value on a line within ±3 (wrap / off-by-one) | 121 | 114 |
| value only elsewhere in the file | 6 | 6 |
| entry has no numeric value at all | 53 | 39 |
| **value nowhere in the file** | **0** | **0** |

**No fabrication signal.** Not one of the 1472 numeric entries stores a value that is
absent from the ledger. The six "elsewhere" cases are figures the ledger writes as words or
in a different section, not stray numbers: "three pixels" (L1132), "six frames" (L1163),
"a fifth" (L1068, i.e. 20%), "five addresses" (L1990), and 55.2 vs "~55" (L2294). They are
recorded as explained, not as defects.

The ±3 bucket is a coarser check than the quote check: it says a number equal to `value`
sits within three lines, which is what a wrapped sentence or the adjacent table row looks
like, but it cannot tell that equal number from a coincidence. It is reported separately
from the quote verdict for that reason. The 53 no-value entries (all part1) are ranges and
series recorded with `value: null` ("58-73%", "1.3-1.6x", "37-59") — a limitation of the
harvest schema, not a provenance defect.

## 6. Sensitivity, and what the four part2 misses are

Verification rate against the run-length threshold (part1 / part2):

| threshold | 20 | 30 | 39 | 40 |
|---|---|---|---|---|
| part1 | 93.6% | 90.0% | 87.5% | 87.0% |
| part2 | 99.7% | 99.6% | 99.5% | 99.5% |

The split is stable, so "part1 paraphrases, part2 quotes" is not an artefact of 40. The
four part2 exceptions, each a normalisation edge rather than a rewrite:

- **L1210** — `worst channel 59`: the claimed line is blank; the table row carrying
  `59.3` / `37-59` is L1205–L1209. Off-by-one, and a genuine wrong-line case.
- **L1919** — `the risk ... measures at **8 bytes of stack**`: the text elides a
  parenthetical clause with `...`, so no 40-character stretch of it is contiguous in the
  ledger; the remaining fragment is 31 characters.
- **L2060** — the text re-quotes the ledger with `'126 on 0.66%'` where the line has
  `"126 on 0.66%"`. The normalisation in the brief does not touch quote characters, so this
  fails by one character class; folding `'`/`"` would verify it.
- **L2362** — `23-43% 'gap'`: the words are on **L2361**, not 2362 (off by one), and the
  ledger writes `"gap"` where the entry writes `'gap'`. Two independent reasons.

## 7. Audit of the two earlier summaries (recomputed, not reused)

Every figure in `.agent/inputs/SUMMARY1.md` reproduces exactly: 751 entries, 710 decision,
282 distinct lines, tiers 550/157/22/22, nulls 323/27/708, 18 decision entries with both
instrument and population null.

`.agent/inputs/SUMMARY2.md` reproduces almost everywhere — 774 entries, 590 decision, 420
distinct lines, tiers 468/239/36/31, nulls 464/60/512, `inputs` filled on 164 of 239
derived rows, 262 rows dated 2026-11-15, `ncu` first at L1818 — **except one number**:

- SUMMARY2 says "the 47 rows with both instrument and population null are all derived (46)
  or read_elsewhere (1)". The count is **51** (50 derived, 1 read_elsewhere). The tier
  composition claim is right; the total is off by four.

That is the wrong claim the brief anticipated. It does not change SUMMARY2's conclusion
(the both-null rows really are derived ratios with auditable `inputs`), but 47 is not
quotable.

## 8. Named part1 paraphrases

Three, with the ledger's own wording beside them:

1. **L40** — harvest: "the tail convolution **is** 0.83 against its 1.09 traffic floor".
   Ledger: "the tail convolution**'s** 0.83 against its 1.09" — the possessive was
   rewritten as a clause.
2. **L63** — harvest: "expected 1.3-1.6x; isolated engine 94.3 -> 57.5 ms". Ledger:
   "| 1.3-1.6x | isolated engine 94.3 -> 57.5 ms |" — two table cells joined with "; " and
   the word "expected" added; the fragments are verbatim, the sentence is not.
3. **L67** — harvest: "measures 49.84 ms against the baseline 50.09". Ledger: "measures
   49.84 ms against the baseline**'s** 50.09 (0.995x, inside noise)" — possessive dropped
   and the qualifier cut.

All 98 part1 unverified entries are of this family: their longest verbatim run in the
ledger has a median of 20 characters — 23 reach 30–39 characters (one token changed), 27
reach 20–29, 48 stay below 20. They record what a sentence *says*, not what it says *word
for word*, and the value on the line is right in every case.

---

## Arithmetic: checked, and it holds

Added 2026-11-16 by the supervisor, after the inventory above.

Every `a × b = c` / `a / b = c` / `a + b = c` equality written out in the ledger's prose was extracted
and recomputed, accepting a unit conversion (×1000, ×1024, ×60) when that is what the sentence does
(`94 × 29.7 = 2.79` is 94 frames times 29.7 ms in 2.79 s), and parsing `x`/`×`/`*`/`/` left to right so
`2.355 × 12/16 = 1.766` is read as written.

**Result: 21 equalities recognised, 21 consistent, 0 mismatches.**

Two honest limits, both of which matter more than the result:

- 21 of 1,525 numbers is all the corpus writes as a *closed arithmetic statement*. The other derived
  numbers (`1.53x cheaper`, `58-73% of a borrowed peak`) are ratios of quantities that are not both
  printed in the sentence, so no reader and no script can recompute them from the text.
- A first version of this check reported five "mismatches". All five were the checker's own: a unit
  conversion it did not know, and an operator precedence it assumed wrong. An arithmetic checker
  without units and precedence is a noise generator - the same class of error as a number without its
  tuple, applied to the checker instead of the claim.

So the debt is **provenance, not mathematics**: the ledger's arithmetic is sound, and the 634
decision-bearing entries without an instrument or a population cannot be fixed by recomputing them.
They need the artifact they were read from, and no probe artifact is committed anywhere in the repo
(`git ls-files` finds no JSON, CSV, log or npz).
