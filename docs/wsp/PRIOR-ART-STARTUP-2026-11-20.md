# Prior art on the start-up work: scale selection, clipping bounds, and calibration without a sweep

Searched 2026-11-20 after the start-up split showed where the seconds go (`startup-split.txt`: 23.89 s probe,
6.56 s Triton's first call, 2.39 s interpreter). The question put to the literature is narrow: **is there a way to
pick the quantisation scale/clipping range without a sweep, and to calibrate without decoding sixteen frames?**

Our own numbers, which every paper below is read against:

| what | ours | record / file |
|---|---|---|
| the grid clamp K is chosen by building 2 candidates and scoring 6 frames | ~3 s, on top of the body build | `REVIEW-int8-grid-k-predictor.md` §7 |
| calibration decodes 16 frames and measures per-layer scales | 23.89 s in-app, 29.30 s standalone | ST003, `int8-probe-cost-2026-11-19.txt` |
| a linear sum of weight and activation error ranks the ladder | **4/22 at every beta** - refuted | `int8-grid-k-predict.txt`, REVIEW §6 |
| K's optimum is interior (2/3/4) and model-dependent | 14 of 22 models improve on K=2 by >=2x somewhere on the ladder | REVIEW §3 |
| the acceptance statistic is a thresholded picture share, not L2 | 0.1% of pixels above 8 levels | `int8-tail-csc.txt` |

## The papers, and what each would actually replace

| paper | idea in one line | would replace | fits us? |
|---|---|---|---|
| **Beacon** (2508.20293, Zhang & Saab, 2025) | quantise on an *unscaled* integer grid, then solve the optimal scale in closed form: `c = <Xw,Xq>/‖Xq‖²` | the K sweep's search | **partly** - removes the search, but the closed form is least-squares, and L2 proxies are what we already refuted |
| **AdaBM** (2404.03296, Hong & Lee, 2024) | SR-specific: layer sensitivity = std of activations; bit width per layer from sensitivity percentiles; clipping re-searched per bit width by a 1-D L2 minimisation; image complexity sets a global factor | per-layer K, and the clip range | **partly** - cheap (71 s total vs 40-70 h QAT), but needs QAT-style finetuning of the bit factors |
| **2DQuant** (2406.06649, Liu et al., 2024) | clipping bound search shaped by the distribution: symmetric narrowing for bell-shaped tensors, fixed lower bound + right-bound traversal for exponential ones (DOBI), then distillation calibration | the clip range per quantiser | **partly** - the shaped search is cheap and needs no builds, but the acceptance is PSNR |
| **CondiQuant** (2502.15478, Liu et al., 2025) | instead of choosing a scale, make the weights quantisation-friendly: minimise the condition number of W by a proximal/SVD step, keeping the output | nothing we have - a new lever | **no** - needs SVD per layer plus an optimisation loop, and the objective is L2 |
| **Predicting Quantization Price** (2609.28270, Qiu et al., 2026) | the configuration must be ranked *before* building it; the correct price is the candidate's layer-output error covariance weighted by the reference model's downstream curvature | the sweep's *decision* | **as theory** - it explains our 4/22: reconstruction and diagonal scores drop output curvature and input covariance, so they preserve rankings only when those are constant across candidates |
| **LSGQuant** (2602.03182, Wu et al., 2026) | real-world **video** SR (one-step diffusion): DRAQ computes the per-channel max-abs range **online**, per input, so there is no calibration set; VOLTS spends training budget by activation variance | the 24 s calibration entirely | **yes, testably** - the risk is exactly our kind: a per-frame range can move frame to frame |
| **CafeQ** (2511.19705, Sun et al., 2025) | calibration-free weight quantisation: learn an invertible per-matrix transform that removes outliers, using a Frobenius-norm proxy validated by rank correlation with downstream scores | the calibration set | **no** - weight-only, LLM, and its proxy is the class we measured as non-predictive |
| **QuantVSR** (2508.04485) | video SR PTQ with temporal consistency as a constraint | - | read in full earlier (`PRIOR-ART-2026-11-20.md`) |

Reading scope, stated so the next reader knows what was and was not read: Beacon in full (all five sections);
AdaBM §3.2-§3.5 and its algorithm/experiments; 2DQuant §3.1-§3.3; CondiQuant §3.2-§3.3; the price paper's
introduction and §2-§3 framing; LSGQuant §3.3-§3.4; CafeQ §4 and the proxy study. None of the appendices were read.

## The three that change something for us

**1. Beacon says the search can be removed, but not the objective.** Their argument is geometric: once the integer
codes are chosen, the scale that minimises the layer-output error has a closed form, so no grid search over scales
is needed. That is exactly the shape of our K sweep - except our K is chosen by the *picture* statistic, and the
lane measured that a least-squares criterion over the same candidates agrees with the ladder's winner on 4 of 22
models. So Beacon would make the sweep cheaper and, on our evidence, choose the wrong K more often than not. The
honest use of it is as a **baseline**: build once at a fixed grid, solve the per-layer scale in closed form, and see
whether that beats the swept K on the ladder. If it does not (expected), it is a two-line refutation with a strong
citation instead of a hunch.

**2. DRAQ (LSGQuant) removes the calibration set, and its risk is our risk.** A per-channel max-abs range computed
from the current input needs no 16-frame pass, which is the 23.89 s. The cost is temporal: the range can change
between frames, and a changing range changes the picture. Our harness already measures exactly that - the 494-frame
pair, worst-frame share - so the experiment is a render pair, not a new instrument. This is the only paper here that
attacks the block we actually measured as expensive.

**3. The price paper explains our negative result and says what a correct predictor needs.** Our linear sum of the
two measured errors dropped the output-side curvature and the input covariance; the paper shows that such reduced
scores preserve configuration rankings only when the dropped factors are effectively constant across candidates,
which on sixteen hops with different slopes they are not. A predictor built the way the paper describes - error
covariance from the candidate, curvature from the fp16 reference - is the one class we have not tried, and the
ladder (22 models, 6 K values, 3 held-out frames) is a ready-made test set for it.

## What does not help, and why

- **CondiQuant** needs an SVD and a proximal loop per layer and optimises L2; the gain is at 2-4 bits where the
  condition number dominates, and our body is already at 8 bits with a picture bar.
- **AdaBM and 2DQuant** both end in a training or distillation stage; their cheap part (sensitivity percentiles,
  shaped bound search) is a proxy search like ours, and their acceptance is PSNR/SSIM on benchmarks, not a share of
  frames above a level threshold.
- **CafeQ** is weight-only and its proxy is the Frobenius norm of the reconstruction error - the same reduced score
  the price paper explains as insufficient.
- **None of them addresses the 6.56 s Triton first call.** That block is compiler time, not quantisation theory;
  the levers there are an ahead-of-time compile or a kernel cache, which is engineering, not literature.

## Three experiments this suggests, cheapest first

1. **Beacon as a baseline for K** (no new kernels): one build at a fixed grid, per-layer scale solved in closed
   form on 6 frames, compared against the swept K on the ladder's 22 models. Falsifier: agreement with the ladder's
   winner on <= 4/22, i.e. no better than the linear proxy we already refuted.
2. **DRAQ as a replacement for the 24 s** (a knob, not a kernel): compute the per-channel range per frame instead of
   calibrating, then run the 494-frame pair and the start-up split. Falsifier: any frame above the bar, or a
   frame-to-frame range swing that shows up in the pair.
3. **A curvature-priced predictor** (the one the price paper says is correct): error covariance per candidate,
   weighted by the fp16 model's output curvature; scored on the same ladder. Falsifier: ranking no better than the
   4/22 the linear sum achieved.

Ordered by cost, and each has a falsifier that costs less than the experiment it replaces - which is the only reason
to run them at all.

## Sources

- arXiv:2508.20293v2 Beacon · arXiv:2404.03296v1 AdaBM · arXiv:2406.06649v1 2DQuant · arXiv:2502.15478v1 CondiQuant
- arXiv:2609.28270v1 Predicting Quantization Price · arXiv:2602.03182v2 LSGQuant · arXiv:2511.19705v1 CafeQ
- arXiv:2508.04485v2 QuantVSR (read earlier) · PAMS (ECCV 2020, arXiv:2011.04212, read earlier)
