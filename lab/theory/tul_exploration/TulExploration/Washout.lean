/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Algebra.Module.LinearMap.End
import Mathlib.Algebra.Ring.GeomSum
import Mathlib.Algebra.Module.BigOperators
import Mathlib.Logic.Function.Iterate

/-!
# T5: a contractive loop erases a choice made once, and keeps a choice re-supplied

## Model

The pass is a map `F` on the state space with Lipschitz constant `L` (the loop is held
subcritical: measured typical gain 0.87 to 0.90 under the fixed-point term).  A code that
enters ONCE, at the loop entry, is a difference between entry states.  A code that is
RE-INJECTED at every pass is a constant added after every application of `F`
(`h ↦ F h + c`, the way `inj[k]` enters `_tul_core` every pass).

## Main results

* `washout` : `dist (F^[T] x) (F^[T] y) ≤ L^T dist x y`.  An entry-only code decays
  geometrically with depth under a contraction.
* `reinjection_separates` : the fixed points of `h ↦ F h + c` and `h ↦ F h + c'` sit at
  least `‖c - c'‖ / (1 + L)` apart.
* **`reinjected_linear_diff`** : for a linear pass `h ↦ A h + b + c` the two rollouts from one
  entry differ by `∑_{t<T} Aᵗ (c - c')` after `T` passes: every pass ADDS a term.
* `entry_linear_diff` : a code that enters only at the entry leaves `Aᵀ (e - e')`.
* **`reinjected_eigen_diff`** / `entry_eigen_diff` : along an eigen-direction of gain `λ` the
  re-injected separation is `(∑_{t<T} λᵗ) (c - c')` (it grows with depth, toward
  `1/(1-λ)` of the one-pass value) while the entry-only one is `λᵀ (e - e')` (it decays).
  `geom_times` gives `(1 - λ) ∑_{t<T} λᵗ = 1 - λᵀ`.
* **`reinjected_rollouts_stay_apart`** : two rollouts with re-injected codes `c`, `c'` are,
  after `T` passes, at least `‖c - c'‖/(1+L) - L^T (‖h₀ - x‖ + ‖h₀' - x'‖)` apart: the
  separation converges to a positive floor instead of to zero.
-/

namespace TulExploration

/-- **Entry-only codes wash out.** -/
theorem washout {E : Type*} [PseudoMetricSpace E] (F : E → E) (L : ℝ) (hL : 0 ≤ L)
    (hF : ∀ x y, dist (F x) (F y) ≤ L * dist x y) (x y : E) :
    ∀ T : ℕ, dist (F^[T] x) (F^[T] y) ≤ L ^ T * dist x y
  | 0 => by simp
  | T + 1 => by
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', pow_succ]
    calc dist (F (F^[T] x)) (F (F^[T] y)) ≤ L * dist (F^[T] x) (F^[T] y) := hF _ _
      _ ≤ L * (L ^ T * dist x y) :=
          mul_le_mul_of_nonneg_left (washout F L hL hF x y T) hL
      _ = L ^ T * L * dist x y := by ring

/-- **Re-injected codes keep the fixed points apart.** -/
theorem reinjection_separates {E : Type*} [SeminormedAddCommGroup E] (F : E → E) (L : ℝ)
    (hF : ∀ x y, ‖F x - F y‖ ≤ L * ‖x - y‖) (c c' x x' : E) (hx : F x + c = x)
    (hx' : F x' + c' = x') : ‖c - c'‖ ≤ (1 + L) * ‖x - x'‖ := by
  have hc : c = x - F x := by
    rw [eq_sub_iff_add_eq, add_comm]
    exact hx
  have hc' : c' = x' - F x' := by
    rw [eq_sub_iff_add_eq, add_comm]
    exact hx'
  have e : c - c' = (x - x') - (F x - F x') := by
    rw [hc, hc']
    abel
  rw [e]
  calc ‖(x - x') - (F x - F x')‖ ≤ ‖x - x'‖ + ‖F x - F x'‖ := norm_sub_le _ _
    _ ≤ ‖x - x'‖ + L * ‖x - x'‖ := by linarith [hF x x']
    _ = (1 + L) * ‖x - x'‖ := by ring

/-- **Re-injected rollouts stay apart.**  Rollout `k` runs `h ↦ F h + c` from `h₀`, rollout
`k'` runs `h ↦ F h + c'` from `h₀'`; `x`, `x'` are their fixed points.  After `T` passes they
are at least `‖c - c'‖/(1+L)` apart, up to a transient that decays like `L^T`. -/
theorem reinjected_rollouts_stay_apart {E : Type*} [SeminormedAddCommGroup E] (F : E → E)
    (L : ℝ) (hL : 0 ≤ L) (hF : ∀ x y, ‖F x - F y‖ ≤ L * ‖x - y‖) (c c' x x' h₀ h₀' : E)
    (hx : F x + c = x) (hx' : F x' + c' = x') (T : ℕ) :
    ‖c - c'‖ / (1 + L) - L ^ T * (‖h₀ - x‖ + ‖h₀' - x'‖)
      ≤ ‖(fun h => F h + c)^[T] h₀ - (fun h => F h + c')^[T] h₀'‖ := by
  set G : E → E := fun h => F h + c with hG
  set G' : E → E := fun h => F h + c' with hG'
  have hGL : ∀ a b, dist (G a) (G b) ≤ L * dist a b := by
    intro a b
    rw [dist_eq_norm, dist_eq_norm]
    simpa [hG, add_sub_add_right_eq_sub] using hF a b
  have hG'L : ∀ a b, dist (G' a) (G' b) ≤ L * dist a b := by
    intro a b
    rw [dist_eq_norm, dist_eq_norm]
    simpa [hG', add_sub_add_right_eq_sub] using hF a b
  have hfix : ∀ T, G^[T] x = x := by
    intro T
    induction T with
    | zero => rfl
    | succ n ih => rw [Function.iterate_succ_apply', ih]; exact hx
  have hfix' : ∀ T, G'^[T] x' = x' := by
    intro T
    induction T with
    | zero => rfl
    | succ n ih => rw [Function.iterate_succ_apply', ih]; exact hx'
  have d1 : ‖G^[T] h₀ - x‖ ≤ L ^ T * ‖h₀ - x‖ := by
    have := washout G L hL hGL h₀ x T
    rw [hfix T, dist_eq_norm, dist_eq_norm] at this
    exact this
  have d2 : ‖G'^[T] h₀' - x'‖ ≤ L ^ T * ‖h₀' - x'‖ := by
    have := washout G' L hL hG'L h₀' x' T
    rw [hfix' T, dist_eq_norm, dist_eq_norm] at this
    exact this
  have sep : ‖c - c'‖ / (1 + L) ≤ ‖x - x'‖ := by
    rw [div_le_iff₀ (by linarith)]
    have := reinjection_separates F L hF c c' x x' hx hx'
    linarith
  have tri : ‖x - x'‖ ≤ ‖G^[T] h₀ - x‖ + ‖G^[T] h₀ - G'^[T] h₀'‖ + ‖G'^[T] h₀' - x'‖ := by
    have e : x - x' = -(G^[T] h₀ - x) + (G^[T] h₀ - G'^[T] h₀') + (G'^[T] h₀' - x') := by abel
    rw [e]
    calc ‖-(G^[T] h₀ - x) + (G^[T] h₀ - G'^[T] h₀') + (G'^[T] h₀' - x')‖
        ≤ ‖-(G^[T] h₀ - x) + (G^[T] h₀ - G'^[T] h₀')‖ + ‖G'^[T] h₀' - x'‖ := norm_add_le _ _
      _ ≤ ‖-(G^[T] h₀ - x)‖ + ‖G^[T] h₀ - G'^[T] h₀'‖ + ‖G'^[T] h₀' - x'‖ := by
          linarith [norm_add_le (-(G^[T] h₀ - x)) (G^[T] h₀ - G'^[T] h₀')]
      _ = ‖G^[T] h₀ - x‖ + ‖G^[T] h₀ - G'^[T] h₀'‖ + ‖G'^[T] h₀' - x'‖ := by rw [norm_neg]
  nlinarith [d1, d2, sep, tri]

/-! ## The linear pass: depth integrates a re-injected choice -/

section Linear

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- **Re-injected codes accumulate over passes.**  Two rollouts of the linear pass
`h ↦ A h + b + c` and `h ↦ A h + b + c'` from the same entry `h₀` differ, after `T` passes, by
`∑_{t<T} Aᵗ (c - c')`.  Pass `t` adds the term `Aᵗ (c - c')`. -/
theorem reinjected_linear_diff (A : E →ₗ[ℝ] E) (b c c' h₀ : E) :
    ∀ T : ℕ, (fun h => A h + b + c)^[T] h₀ - (fun h => A h + b + c')^[T] h₀
      = ∑ t ∈ Finset.range T, (A ^ t) (c - c')
  | 0 => by simp
  | T + 1 => by
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', Finset.sum_range_succ']
    have ih := reinjected_linear_diff A b c c' h₀ T
    have e : (A ((fun h => A h + b + c)^[T] h₀) + b + c)
        - (A ((fun h => A h + b + c')^[T] h₀) + b + c')
        = A ((fun h => A h + b + c)^[T] h₀ - (fun h => A h + b + c')^[T] h₀) + (c - c') := by
      rw [map_sub]; abel
    rw [e, ih, map_sum]
    simp only [pow_zero, Module.End.one_apply, pow_succ', Module.End.mul_apply]

/-- **Entry-only codes are multiplied by `Aᵀ`.**  Two rollouts of the same pass
`h ↦ A h + b` from entries `x`, `y` differ by `Aᵀ (x - y)`. -/
theorem entry_linear_diff (A : E →ₗ[ℝ] E) (b x y : E) :
    ∀ T : ℕ, (fun h => A h + b)^[T] x - (fun h => A h + b)^[T] y = (A ^ T) (x - y)
  | 0 => by simp
  | T + 1 => by
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', pow_succ',
      Module.End.mul_apply, ← entry_linear_diff A b x y T, map_sub]
    abel

lemma pow_apply_eigen (A : E →ₗ[ℝ] E) (v : E) (lam : ℝ) (hv : A v = lam • v) :
    ∀ t : ℕ, (A ^ t) v = lam ^ t • v
  | 0 => by simp
  | t + 1 => by
    rw [pow_succ', Module.End.mul_apply, pow_apply_eigen A v lam hv t, map_smul, hv, smul_smul,
      pow_succ]

/-- **Along an eigen-direction of gain `λ` a re-injected choice grows with depth.**  If the
code difference `c - c'` is an eigenvector of the pass with eigenvalue `λ`, the rollouts are
`(∑_{t<T} λᵗ) (c - c')` apart after `T` passes: `1, 1 + λ, 1 + λ + λ², …` times the one-pass
separation. -/
theorem reinjected_eigen_diff (A : E →ₗ[ℝ] E) (b c c' h₀ : E) (lam : ℝ)
    (hv : A (c - c') = lam • (c - c')) (T : ℕ) :
    (fun h => A h + b + c)^[T] h₀ - (fun h => A h + b + c')^[T] h₀
      = (∑ t ∈ Finset.range T, lam ^ t) • (c - c') := by
  rw [reinjected_linear_diff, Finset.sum_smul]
  exact Finset.sum_congr rfl (fun t _ => pow_apply_eigen A _ lam hv t)

/-- **Along the same direction an entry-only choice decays with depth**: `λᵀ (e - e')`. -/
theorem entry_eigen_diff (A : E →ₗ[ℝ] E) (b x y : E) (lam : ℝ)
    (hv : A (x - y) = lam • (x - y)) (T : ℕ) :
    (fun h => A h + b)^[T] x - (fun h => A h + b)^[T] y = lam ^ T • (x - y) := by
  rw [entry_linear_diff]
  exact pow_apply_eigen A _ lam hv T

/-- The geometric factor: `(1 - λ) ∑_{t<T} λᵗ = 1 - λᵀ`.  At `λ = 0.89` (the loop's measured
typical gain) six passes give `(1 - 0.89⁶)/0.11 ≈ 4.6` times the one-pass separation. -/
theorem geom_times (lam : ℝ) (T : ℕ) :
    (1 - lam) * ∑ t ∈ Finset.range T, lam ^ t = 1 - lam ^ T := by
  have h := geom_sum_mul lam T
  linarith [mul_comm (∑ t ∈ Finset.range T, lam ^ t) (lam - 1)]

/-- Each extra pass adds a non-negative amount when `λ ≥ 0`: the separation factor is
monotone in depth. -/
theorem geom_mono (lam : ℝ) (hl : 0 ≤ lam) (T : ℕ) :
    ∑ t ∈ Finset.range T, lam ^ t ≤ ∑ t ∈ Finset.range (T + 1), lam ^ t := by
  rw [Finset.sum_range_succ]
  linarith [pow_nonneg hl T]

end Linear

end TulExploration
