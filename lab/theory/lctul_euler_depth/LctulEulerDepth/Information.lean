/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import TulInformation.SlotLoop
import Mathlib.Tactic.Ring

/-!
# No Euler depth adds information about the span

The sampler is `F (ctx, z₀)`: a deterministic function of the context (tape + seed) and
the source noise `z₀`, for every step count `k` (`Euler.lean`, `eulerEndpoint`). The
noise is drawn by `torch.randn`, independent of everything. So the sampled code at ANY `k`
carries at most what the context carries about the span.

This reuses `tul_information` (`TulInformation.Joint`, `mi_map_le`) and adds the one lemma
that development left unproved: an independent extra input carries nothing.

## Main results

* `Joint.mi_prodIndep` : `I((ctx, N); span) = I(ctx; span)` when `N` is independent of
  `(ctx, span)`.
* `Joint.sample_mi_le` : `I(F(ctx, N); span) ≤ I(ctx; span)` for every sampler `F`, hence
  for every Euler depth `k`.
* `Joint.sample_condEntropy_ge` : the log-loss form: no depth lowers the Bayes conditional
  entropy of the span below `H(span | ctx)`.
-/

namespace TulInformation

namespace Joint

open Finset

variable {α β γ ν : Type*} [Fintype α] [Fintype β] [Fintype γ] [Fintype ν]

/-- The joint law of `((ctx, N), span)` when `N ~ r` is independent of `(ctx, span) ~ Q`. -/
def prodIndep (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n) (hr1 : ∑ n, r n = 1) :
    Joint (α × ν) β where
  p := fun an b => Q.p an.1 b * r an.2
  nonneg := fun an b => mul_nonneg (Q.nonneg _ _) (hr0 _)
  total := by
    rw [Fintype.sum_prod_type]
    simp only
    have : ∀ a, ∑ n, ∑ b, Q.p a b * r n = ∑ b, Q.p a b := by
      intro a
      rw [Finset.sum_comm]
      simp_rw [← Finset.mul_sum, hr1, mul_one]
    simp_rw [this]
    exact Q.total

lemma prodIndep_p (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n) (hr1 : ∑ n, r n = 1)
    (a : α) (n : ν) (b : β) : (Q.prodIndep r hr0 hr1).p (a, n) b = Q.p a b * r n := rfl

lemma prodIndep_left (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n) (hr1 : ∑ n, r n = 1)
    (a : α) (n : ν) : (Q.prodIndep r hr0 hr1).left (a, n) = Q.left a * r n := by
  simp only [left, prodIndep_p, Finset.sum_mul]

lemma prodIndep_right (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n) (hr1 : ∑ n, r n = 1)
    (b : β) : (Q.prodIndep r hr0 hr1).right b = Q.right b := by
  simp only [right, Fintype.sum_prod_type, prodIndep_p]
  simp_rw [← Finset.mul_sum, hr1, mul_one]

/-- **Independent noise carries nothing.**  `I((ctx, N); span) = I(ctx; span)`. -/
theorem mi_prodIndep (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n)
    (hr1 : ∑ n, r n = 1) : (Q.prodIndep r hr0 hr1).mi = Q.mi := by
  have hterm : ∀ (a : α) (n : ν) (b : β),
      Q.p a b * r n * Real.log ((Q.p a b * r n) / (Q.left a * r n * Q.right b))
        = r n * (Q.p a b * Real.log (Q.p a b / (Q.left a * Q.right b))) := by
    intro a n b
    by_cases hn : r n = 0
    · simp [hn]
    · rw [show Q.left a * r n * Q.right b = (Q.left a * Q.right b) * r n by ring,
        mul_div_mul_right _ _ hn]
      ring
  simp only [mi, Fintype.sum_prod_type, prodIndep_p, prodIndep_left, prodIndep_right]
  simp_rw [hterm]
  have hswap : ∀ a, ∑ n, ∑ b, r n * (Q.p a b * Real.log (Q.p a b / (Q.left a * Q.right b)))
      = ∑ b, Q.p a b * Real.log (Q.p a b / (Q.left a * Q.right b)) := by
    intro a
    simp_rw [← Finset.mul_sum, ← Finset.sum_mul, hr1, one_mul]
  simp_rw [hswap]

/-- **No sampler at any depth adds information.**  For every deterministic sampler
`F : (ctx, N) → sample` (for LCTUL, `F = eulerEndpoint (v ctx) k` composed with the float
grid, for any `k`), `I(F(ctx, N); span) ≤ I(ctx; span)`. -/
theorem sample_mi_le [DecidableEq γ] (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n)
    (hr1 : ∑ n, r n = 1) (F : α × ν → γ) :
    ((Q.prodIndep r hr0 hr1).map F).mi ≤ Q.mi := by
  rw [← mi_prodIndep Q r hr0 hr1]
  exact mi_map_le _ F

/-- The same for a whole family of samplers indexed by the depth `k`. -/
theorem sample_mi_le_depth [DecidableEq γ] (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n)
    (hr1 : ∑ n, r n = 1) (F : ℕ → α × ν → γ) (k : ℕ) :
    ((Q.prodIndep r hr0 hr1).map (F k)).mi ≤ Q.mi :=
  sample_mi_le Q r hr0 hr1 (F k)

lemma prodIndep_entropyRight (Q : Joint α β) (r : ν → ℝ) (hr0 : ∀ n, 0 ≤ r n)
    (hr1 : ∑ n, r n = 1) : (Q.prodIndep r hr0 hr1).entropyRight = Q.entropyRight := by
  simp only [entropyRight, prodIndep_right]

/-- **The log-loss form.**  The Bayes conditional entropy of the span given the sampled code
is at least `H(span | ctx)`, at every depth. The K-curve's floor is set by the context, not
by `k`. -/
theorem sample_condEntropy_ge [DecidableEq γ] (Q : Joint α β) (r : ν → ℝ)
    (hr0 : ∀ n, 0 ≤ r n) (hr1 : ∑ n, r n = 1) (F : α × ν → γ) :
    Q.condEntropy ≤ ((Q.prodIndep r hr0 hr1).map F).condEntropy := by
  have h := condEntropy_map_ge (Q.prodIndep r hr0 hr1) F
  have hQ : (Q.prodIndep r hr0 hr1).condEntropy = Q.condEntropy := by
    simp only [condEntropy, prodIndep_entropyRight, mi_prodIndep]
  rw [hQ] at h
  exact h

end Joint

end TulInformation
