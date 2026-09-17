/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import LctulEulerDepth.Euler

/-!
# Second moments: what the flow loss at `t = 0` selects, and what a conditional draw costs

Finite model, in the spirit of `tul_information`: a distribution is a weight function on a
finite index type summing to one (float vectors are a finite set). `F` is a real inner
product space (`EuclideanSpace ℝ (Fin 1024)` for one cell).

## Main results

* `sum_norm_sub_sq` : the variance decomposition `∑ wᵢ ‖c - yᵢ‖² = ‖c - m‖² + ∑ wᵢ ‖yᵢ - m‖²`.
* `fm_loss_t0` / `fm_loss_t0_ge` / `fm_loss_t0_optimal` / `fm_loss_t0_optimal_unique` : the
  conditional-flow-matching loss at `t = 0` with an independent source is minimised, and
  only minimised, by the field `g x = m - x`; its minimum is the code's variance.
* `k1_endpoint_of_fm_optimal` : with that field, the one-step Euler endpoint is `m` from
  every source point (the bridge to `Euler.lean`).
* `residual_two_draws` : two independent draws from one law sit at `2 × variance` from
  each other in expectation.
* `total_variance` : the law of total variance across contexts.
* `residual_ratio_conditional_draw` : an independent draw from the conditional law reads
  residual `2 (1 - R²)` of the total variance, the conditional mean reads `1 - R²`, where
  `R²` is the fraction of the code's variance explained by the context. The measured
  `1.7–1.9` is this identity at `R² ∈ [0.05, 0.15]`.
* `norm_wmean_le` : the mean of points on a shell lies inside it (why the `k = 1` code is
  off the RMS shell the coda trained on).
-/

noncomputable section

namespace LctulEulerDepth

open Finset

variable {ι κ : Type*} [Fintype ι] [Fintype κ]
variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- The weighted mean `∑ wᵢ • yᵢ`. -/
def wmean (w : ι → ℝ) (y : ι → F) : F := ∑ i, w i • y i

/-! ## The variance decomposition -/

lemma sum_smul_sub_wmean (w : ι → ℝ) (hw1 : ∑ i, w i = 1) (y : ι → F) :
    ∑ i, w i • (wmean w y - y i) = 0 := by
  simp only [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul, hw1, one_smul, wmean,
    sub_self]

/-- **The variance decomposition.**  For any point `c`,
`∑ wᵢ ‖c - yᵢ‖² = ‖c - m‖² + ∑ wᵢ ‖yᵢ - m‖²` with `m` the weighted mean. Only `∑ wᵢ = 1`
is used. -/
theorem sum_norm_sub_sq (w : ι → ℝ) (hw1 : ∑ i, w i = 1) (y : ι → F) (c : F) :
    ∑ i, w i * ‖c - y i‖ ^ 2
      = ‖c - wmean w y‖ ^ 2 + ∑ i, w i * ‖y i - wmean w y‖ ^ 2 := by
  set m := wmean w y with hm
  have hpt : ∀ i, ‖c - y i‖ ^ 2
      = ‖c - m‖ ^ 2 + 2 * inner ℝ (c - m) (m - y i) + ‖y i - m‖ ^ 2 := by
    intro i
    have : c - y i = (c - m) + (m - y i) := by abel
    rw [this, norm_add_sq_real, norm_sub_rev (m) (y i)]
  simp_rw [hpt, mul_add, Finset.sum_add_distrib]
  have h1 : ∑ i, w i * ‖c - m‖ ^ 2 = ‖c - m‖ ^ 2 := by
    rw [← Finset.sum_mul, hw1, one_mul]
  have h2 : ∑ i, w i * (2 * inner ℝ (c - m) (m - y i)) = 0 := by
    have : ∀ i, w i * (2 * inner ℝ (c - m) (m - y i))
        = 2 * inner ℝ (c - m) (w i • (m - y i)) := by
      intro i; rw [inner_smul_right]; ring
    simp_rw [this]
    rw [← Finset.mul_sum, ← inner_sum, sum_smul_sub_wmean w hw1 y, inner_zero_right, mul_zero]
  rw [h1, h2, add_zero]

/-- The weighted mean is the unique minimiser of the weighted squared distance. -/
theorem sum_norm_sub_sq_ge (w : ι → ℝ) (hw1 : ∑ i, w i = 1) (y : ι → F) (c : F) :
    ∑ i, w i * ‖y i - wmean w y‖ ^ 2 ≤ ∑ i, w i * ‖c - y i‖ ^ 2 := by
  rw [sum_norm_sub_sq w hw1 y c]
  have : 0 ≤ ‖c - wmean w y‖ ^ 2 := by positivity
  linarith

/-! ## The conditional-flow-matching loss at `t = 0` -/

/-- The CFM loss at `t = 0` for a candidate field `g` (a function of the source point):
`∑ₐ ∑_b rₐ q_b ‖g xₐ - (y_b - xₐ)‖²`. `r` weights the source alphabet (`z₀`), `q` the
code alphabet (`z₁`); the product weight is independence, which is how `cfm_pair` draws
`z₀` (`torch.randn`, never a function of the code). -/
def fmLoss0 (r : κ → ℝ) (q : ι → ℝ) (x : κ → F) (y : ι → F) (g : κ → F) : ℝ :=
  ∑ a, ∑ b, r a * q b * ‖g a - (y b - x a)‖ ^ 2

/-- **The flow loss at `t = 0` decomposes.**  `fmLoss0 g = ∑ₐ rₐ ‖(xₐ + g xₐ) - m‖² + Var(y)`:
the loss charges the field exactly for the distance between the one-step endpoint
`x + g x` and the code's mean `m`, plus the code's own variance, which no field can remove. -/
theorem fm_loss_t0 (r : κ → ℝ) (hr1 : ∑ a, r a = 1) (q : ι → ℝ) (hq1 : ∑ b, q b = 1)
    (x : κ → F) (y : ι → F) (g : κ → F) :
    fmLoss0 r q x y g
      = ∑ a, r a * ‖(x a + g a) - wmean q y‖ ^ 2 + ∑ b, q b * ‖y b - wmean q y‖ ^ 2 := by
  simp only [fmLoss0]
  have hinner : ∀ a, ∑ b, r a * q b * ‖g a - (y b - x a)‖ ^ 2
      = r a * (‖(x a + g a) - wmean q y‖ ^ 2 + ∑ b, q b * ‖y b - wmean q y‖ ^ 2) := by
    intro a
    rw [← sum_norm_sub_sq q hq1 y (x a + g a), Finset.mul_sum]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [show g a - (y b - x a) = (x a + g a) - y b by abel]
    ring
  simp_rw [hinner, mul_add, Finset.sum_add_distrib, ← Finset.sum_mul, hr1, one_mul]

/-- No field beats the code's variance at `t = 0`. -/
theorem fm_loss_t0_ge (r : κ → ℝ) (hr0 : ∀ a, 0 ≤ r a) (hr1 : ∑ a, r a = 1)
    (q : ι → ℝ) (hq1 : ∑ b, q b = 1) (x : κ → F) (y : ι → F) (g : κ → F) :
    ∑ b, q b * ‖y b - wmean q y‖ ^ 2 ≤ fmLoss0 r q x y g := by
  rw [fm_loss_t0 r hr1 q hq1 x y g]
  have : 0 ≤ ∑ a, r a * ‖(x a + g a) - wmean q y‖ ^ 2 :=
    Finset.sum_nonneg fun a _ => mul_nonneg (hr0 a) (by positivity)
  linarith

/-- **The optimal field at `t = 0` is `m - x`**, and it reaches the variance floor. -/
theorem fm_loss_t0_optimal (r : κ → ℝ) (hr1 : ∑ a, r a = 1)
    (q : ι → ℝ) (hq1 : ∑ b, q b = 1) (x : κ → F) (y : ι → F) :
    fmLoss0 r q x y (fun a => wmean q y - x a) = ∑ b, q b * ‖y b - wmean q y‖ ^ 2 := by
  rw [fm_loss_t0 r hr1 q hq1 x y]
  simp

/-- **Only the mean field reaches the floor.**  Any field at the floor satisfies
`x + g x = m` on the support of the source. -/
theorem fm_loss_t0_optimal_unique (r : κ → ℝ) (hr0 : ∀ a, 0 ≤ r a) (hr1 : ∑ a, r a = 1)
    (q : ι → ℝ) (hq1 : ∑ b, q b = 1) (x : κ → F) (y : ι → F) (g : κ → F)
    (hopt : fmLoss0 r q x y g = ∑ b, q b * ‖y b - wmean q y‖ ^ 2) :
    ∀ a, 0 < r a → x a + g a = wmean q y := by
  rw [fm_loss_t0 r hr1 q hq1 x y g] at hopt
  have hzero : ∑ a, r a * ‖(x a + g a) - wmean q y‖ ^ 2 = 0 := by linarith
  have hnn : ∀ a ∈ (univ : Finset κ), 0 ≤ r a * ‖(x a + g a) - wmean q y‖ ^ 2 :=
    fun a _ => mul_nonneg (hr0 a) (by positivity)
  have hterm := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hzero
  intro a ha
  have h := hterm a (mem_univ a)
  rcases mul_eq_zero.mp h with h | h
  · exact absurd h ha.ne'
  · exact sub_eq_zero.mp (norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp h))

/-- **Bridge to the sampler.**  With the `t = 0`-optimal field, the one-step Euler endpoint
from ANY source point is the mean `m`. The `k = 1` sample of LCTUL is the conditional mean
of the code; it carries none of the source randomness. -/
theorem k1_endpoint_of_fm_optimal (q : ι → ℝ) (y : ι → F) (v : F → ℝ → F)
    (hv : ∀ z, v z 0 = wmean q y - z) (z0 : F) :
    eulerEndpoint v 1 z0 = wmean q y :=
  eulerEndpoint_one_of_meanField v (wmean q y) hv z0

/-! ## What a conditional draw costs: the residual identities -/

/-- **Two independent draws sit at twice the variance.**
`∑ᵢ ∑ⱼ wᵢ wⱼ ‖yᵢ - yⱼ‖² = 2 ∑ᵢ wᵢ ‖yᵢ - m‖²`. -/
theorem residual_two_draws (w : ι → ℝ) (hw1 : ∑ i, w i = 1) (y : ι → F) :
    ∑ i, ∑ j, w i * w j * ‖y i - y j‖ ^ 2 = 2 * ∑ i, w i * ‖y i - wmean w y‖ ^ 2 := by
  have hrow : ∀ i, ∑ j, w i * w j * ‖y i - y j‖ ^ 2
      = w i * (‖y i - wmean w y‖ ^ 2 + ∑ j, w j * ‖y j - wmean w y‖ ^ 2) := by
    intro i
    rw [← sum_norm_sub_sq w hw1 y (y i), Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  simp_rw [hrow, mul_add, Finset.sum_add_distrib, ← Finset.sum_mul, hw1, one_mul]
  ring

/-- **The law of total variance.**  Contexts `c` with weights `qc`, and for each context a
conditional law `w c` over the codes. For ANY reference point `m`,
`∑_c q_c ∑ᵢ w_{c,i} ‖yᵢ - m‖² = ∑_c q_c ∑ᵢ w_{c,i} ‖yᵢ - m_c‖² + ∑_c q_c ‖m_c - m‖²`:
total = within + between. -/
theorem total_variance (qc : κ → ℝ) (w : κ → ι → ℝ) (hw1 : ∀ c, ∑ i, w c i = 1)
    (y : ι → F) (m : F) :
    ∑ c, qc c * ∑ i, w c i * ‖y i - m‖ ^ 2
      = ∑ c, qc c * ∑ i, w c i * ‖y i - wmean (w c) y‖ ^ 2
        + ∑ c, qc c * ‖wmean (w c) y - m‖ ^ 2 := by
  have hc : ∀ c, ∑ i, w c i * ‖y i - m‖ ^ 2
      = ‖m - wmean (w c) y‖ ^ 2 + ∑ i, w c i * ‖y i - wmean (w c) y‖ ^ 2 := by
    intro c
    rw [← sum_norm_sub_sq (w c) (hw1 c) y m]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [norm_sub_rev]
  simp_rw [hc, mul_add, Finset.sum_add_distrib, norm_sub_rev]
  ring

/-- **The residual-variance reading is the second-moment prediction.**  Write `W` for the
within-context variance (what the context does not determine), `B` for the between-context
variance (what it does), `T = W + B`. An independent draw from the conditional law has
expected squared distance `2 W` from the true code; the conditional mean has `W`. As a
fraction of `T` with `R² = B / T`: the draw reads `2 (1 - R²)`, the mean reads `1 - R²`.
The measured `1.7–1.9` on every LCTUL arm is `R² ∈ [0.05, 0.15]`: the context explains a
twentieth to a seventh of the code, and a sampler that lands ON the conditional law can do
no better than `2 (1 - R²)` on this instrument. -/
theorem residual_ratio_conditional_draw (qc : κ → ℝ) (w : κ → ι → ℝ)
    (hw1 : ∀ c, ∑ i, w c i = 1) (y : ι → F) (m : F) :
    let W := ∑ c, qc c * ∑ i, w c i * ‖y i - wmean (w c) y‖ ^ 2
    let B := ∑ c, qc c * ‖wmean (w c) y - m‖ ^ 2
    let T := ∑ c, qc c * ∑ i, w c i * ‖y i - m‖ ^ 2
    (∑ c, qc c * ∑ i, ∑ j, w c i * w c j * ‖y i - y j‖ ^ 2 = 2 * W) ∧ T = W + B := by
  intro W B T
  refine ⟨?_, total_variance qc w hw1 y m⟩
  simp only [W]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [residual_two_draws (w c) (hw1 c) y]
  ring

/-- Solving the identity for `R²`: a residual ratio `ρ` of the draw against the total
means the context explains `1 - ρ / 2` of the code's variance. At `ρ = 1.8`, one tenth. -/
theorem explained_of_residual_ratio (W B T ρ : ℝ) (hT : 0 < T) (hWB : T = W + B)
    (hρ : 2 * W = ρ * T) : B / T = 1 - ρ / 2 := by
  have : B = T - W := by linarith
  rw [this]
  field_simp
  linarith

/-! ## The mean sits inside the shell -/

/-- `‖∑ wᵢ • yᵢ‖ ≤ ∑ wᵢ ‖yᵢ‖` for non-negative weights. -/
theorem norm_wmean_le (w : ι → ℝ) (hw0 : ∀ i, 0 ≤ w i) (y : ι → F) :
    ‖wmean w y‖ ≤ ∑ i, w i * ‖y i‖ := by
  simp only [wmean]
  calc ‖∑ i, w i • y i‖ ≤ ∑ i, ‖w i • y i‖ := norm_sum_le _ _
    _ = ∑ i, w i * ‖y i‖ := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [norm_smul, Real.norm_of_nonneg (hw0 i)]

/-- **The `k = 1` code is off the shell.**  If every code has norm `R` (the RMS shell the
coda trained on), the mean has norm at most `R`; the `k = 1` sample, being the mean
(`k1_endpoint_of_fm_optimal`), is inside the shell. -/
theorem norm_wmean_le_of_shell (w : ι → ℝ) (hw0 : ∀ i, 0 ≤ w i) (hw1 : ∑ i, w i = 1)
    (y : ι → F) (R : ℝ) (hy : ∀ i, ‖y i‖ = R) : ‖wmean w y‖ ≤ R := by
  calc ‖wmean w y‖ ≤ ∑ i, w i * ‖y i‖ := norm_wmean_le w hw0 y
    _ = R := by simp_rw [hy, ← Finset.sum_mul, hw1, one_mul]

end LctulEulerDepth

end
