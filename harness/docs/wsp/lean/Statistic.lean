/-
  The acceptance statistic, formalized as far as it is worth formalizing.

  This file cannot certify a measurement: it cannot see a population label, an instrument or a config.
  It is here to force the definitions to be stated and the properties the harness relies on to be
  named. Everything is about the aggregation, nothing about the GPU.

  Built with Lean 4.36.0, no Mathlib: `lean Statistic.lean`.
-/

namespace RVE

/-- One pixel's disagreement with the reference, in levels: the worst channel. -/
def off (channels : List Nat) : Nat := channels.foldl max 0

/-- The share of bad pixels, in per-mille. Integer arithmetic on purpose: no floating point in the
    statement. An EMPTY sample reads 0. -/
def share (pixels : List Nat) (t : Nat) : Nat :=
  (pixels.filter (fun p => p > t)).length * 1000 / pixels.length

/-- The decision over a sample of frames. -/
def worstOf (shares : List Nat) : Nat := shares.foldl max 0

/-- 1. An empty sample passes every positive bar. This is the `min_n` rule, proved rather than
    argued: absence of evidence aggregates to zero, and zero is under any bar. -/
theorem empty_sample_passes (bar : Nat) (h : 0 < bar) : worstOf [] < bar := by
  simpa [worstOf] using h

/-- The same for a share over no pixels: `x / 0` is `0` in Nat, so an empty population reads 0%. -/
theorem empty_share_is_zero (t : Nat) : share [] t = 0 := by
  simp [share]

/-- 2. To certify a sample you must certify every frame in it: the enlarged sample passes only if
    both the old sample and the new frame pass. This is why "it passed on 13 frames" is not a bound
    on 19. -/
theorem both_must_pass (s : Nat) (x : Nat) (bar : Nat) (h : max s x < bar) : s < bar ∧ x < bar :=
  Nat.max_lt.mp h

/-- ... and the contrapositive, which is how the statistic is actually used: one failing frame fails
    the sample, whatever the others scored. -/
theorem one_failure_fails_the_sample (s : Nat) (x : Nat) (bar : Nat)
    (h : bar ≤ x) : ¬ (max s x < bar) := by
  intro hc
  exact Nat.not_lt.mpr h (both_must_pass s x bar hc).2

/-- 3. Determinism: the statistic is a function of the pixel list and the threshold alone. Same
    inputs, same number - which is what makes an A/A control meaningful. -/
theorem share_det (p q : List Nat) (t : Nat) (h : p = q) : share p t = share q t := by
  rw [h]

/-- 4. A population is (kind, items). Two populations are interchangeable for the statistic exactly
    when their ITEM LISTS are equal; the kind is part of the identity, so a relabelled item set is a
    different population. This is the class-1 error, stated. -/
structure Pop where
  kind : String
  items : List Nat
  deriving DecidableEq

def Pop.sameItems (a b : Pop) : Bool := a.items == b.items

theorem relabel_needs_equal_items (a b : Pop) (h : a.sameItems b = true) :
    a.items = b.items := by
  simpa [Pop.sameItems] using h

theorem different_items_are_different_populations (a b : Pop) (h : a.items ≠ b.items) : a ≠ b := by
  intro hab
  exact h (by rw [hab])

/-- 5. A stricter threshold cannot lower the share: the bar's threshold is part of the statistic, so
    two numbers computed at different thresholds are not comparable. Stated over one list. -/
theorem stricter_threshold_not_less (pixels : List Nat) {t u : Nat} (h : t ≤ u) :
    (pixels.filter (fun p => u < p)).length ≤ (pixels.filter (fun p => t < p)).length := by
  induction pixels with
  | nil => simp
  | cons p ps ih =>
    simp only [List.filter_cons]
    by_cases hup : u < p
    · have htp : t < p := Nat.lt_of_le_of_lt h hup
      simp [hup, htp]
      exact ih
    · by_cases htp : t < p
      · simp [hup, htp]
        exact Nat.le_succ_of_le ih
      · simp [hup, htp]
        exact ih

end RVE
