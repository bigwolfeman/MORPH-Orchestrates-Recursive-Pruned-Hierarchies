/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent (ParticleProver)
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Order.Monotone.Basic
import TulExploration.Washout

/-!
# P6: what depth buys a particle in a multi-basin pass

## Model

`G` is the pass for one context.  `x*` is an attracting basin point, `x_u` a repelling
point on the basin boundary (the saddle of a double well such as `x ↦ tanh(βx) + c`).  A
particle starts at `x` and is read after `T` passes by a reader `R` that is `Lr`-Lipschitz.

No lower bound on depth holds for a general pass: a pass that sends every start straight to
its basin point needs one pass.  Two pieces of local structure give one, and they are the
hypotheses below.

* Near the attractor, a CO-Lipschitz bound `λ dist x x* ≤ dist (G x) x*` (the pass contracts
  toward `x*` no faster than `λ`).
* Near the saddle, an upper Lipschitz bound `μ` (the pass expands away from `x_u` no faster
  than `μ`).

## Main results

* `iterate_mem` : an invariant set stays invariant under iteration.
* **`colip_lower`** : `λᵀ dist x x* ≤ dist (Gᵀ x) x*`.
* **`passes_to_settle`** : to get within `ε` of `x*` from distance `d₀ > ε` needs
  `T ≥ log(d₀/ε) / log(1/λ)` passes.
* **`saddle_read_margin`** : `|R(Gᵀ x) - R(x_u)| ≤ Lr μᵀ dist x x_u`.  A particle that starts
  `δ` from the boundary reads like the boundary point for about `log(m/(Lr δ))/log μ` passes.
* `passes_to_decide` : the log form, `log (m / (Lr δ)) / log μ ≤ T` before a reader with
  margin `m` can tell the basins apart.
* **`monotone_side_preserved`** : for a strictly increasing 1-D pass, the side of `x_u` a
  particle starts on is the side it stays on.  In 1-D the basin IDENTITY is fixed by the
  start; depth buys the MARGIN, not the choice.
-/

namespace TulExploration

/-- An invariant set stays invariant under iteration. -/
lemma iterate_mem {E : Type*} (G : E → E) (S : Set E) (hS : ∀ x ∈ S, G x ∈ S) :
    ∀ T, ∀ x ∈ S, G^[T] x ∈ S
  | 0, _, hx => hx
  | T + 1, x, hx => by
    rw [Function.iterate_succ_apply']
    exact hS _ (iterate_mem G S hS T x hx)

/-- **The co-Lipschitz lower bound: a particle approaches its basin point no faster than
`λᵀ`.** -/
theorem colip_lower {E : Type*} [PseudoMetricSpace E] (G : E → E) (S : Set E) (xs : E)
    (lam : ℝ) (hlam : 0 ≤ lam) (hS : ∀ x ∈ S, G x ∈ S)
    (hlow : ∀ x ∈ S, lam * dist x xs ≤ dist (G x) xs) :
    ∀ T, ∀ x ∈ S, lam ^ T * dist x xs ≤ dist (G^[T] x) xs
  | 0, x, _ => by simp
  | T + 1, x, hx => by
    rw [Function.iterate_succ_apply', pow_succ]
    have ih := colip_lower G S xs lam hlam hS hlow T x hx
    have h1 := hlow _ (iterate_mem G S hS T x hx)
    calc lam ^ T * lam * dist x xs = lam * (lam ^ T * dist x xs) := by ring
      _ ≤ lam * dist (G^[T] x) xs := mul_le_mul_of_nonneg_left ih hlam
      _ ≤ dist (G (G^[T] x)) xs := h1

/-- **Passes needed to settle.**  With `0 < λ < 1`, a particle at distance `d₀` from `x*`
that is within `ε < d₀` of it after `T` passes has `T ≥ log(d₀/ε) / log(1/λ)`. -/
theorem passes_to_settle {lam d₀ ε : ℝ} {T : ℕ} (hl0 : 0 < lam) (hl1 : lam < 1) (hε : 0 < ε)
    (hd : ε < d₀) (hT : lam ^ T * d₀ ≤ ε) :
    Real.log (d₀ / ε) / Real.log (1 / lam) ≤ T := by
  have hd0 : 0 < d₀ := lt_trans hε hd
  have hlog : 0 < Real.log (1 / lam) := Real.log_pos (by rw [lt_div_iff₀ hl0]; linarith)
  rw [div_le_iff₀ hlog]
  have h1 := Real.log_le_log (mul_pos (pow_pos hl0 T) hd0) hT
  rw [Real.log_mul (pow_pos hl0 T).ne' hd0.ne', Real.log_pow] at h1
  rw [Real.log_div hd0.ne' hε.ne', one_div, Real.log_inv]
  linarith

/-- **Near the saddle every Lipschitz reader reads the boundary point.**  `G` is
`μ`-Lipschitz, `x_u` a fixed point (the basin boundary), `R` is `Lr`-Lipschitz. -/
theorem saddle_read_margin {E : Type*} [PseudoMetricSpace E] (G : E → E) (μ : ℝ) (hμ : 0 ≤ μ)
    (hG : ∀ x y, dist (G x) (G y) ≤ μ * dist x y) (xu : E) (hxu : G xu = xu)
    (R : E → ℝ) (Lr : ℝ) (hLr : 0 ≤ Lr) (hR : ∀ x y, |R x - R y| ≤ Lr * dist x y)
    (x : E) (T : ℕ) : |R (G^[T] x) - R xu| ≤ Lr * (μ ^ T * dist x xu) := by
  have hfix : G^[T] xu = xu := Function.iterate_fixed hxu T
  have hw := washout G μ hμ hG x xu T
  rw [hfix] at hw
  exact (hR _ _).trans (mul_le_mul_of_nonneg_left hw hLr)

/-- **Passes needed to decide.**  If a reader must move at least `m` away from its boundary
value to tell the basins apart, a particle that starts `δ` from the boundary needs
`T ≥ log(m/(Lr δ)) / log μ` passes (expansion rate `μ > 1`). -/
theorem passes_to_decide {μ Lr δ m : ℝ} {T : ℕ} (hμ : 1 < μ) (hLr : 0 < Lr) (hδ : 0 < δ)
    (hm : 0 < m) (hT : m ≤ Lr * (μ ^ T * δ)) :
    Real.log (m / (Lr * δ)) / Real.log μ ≤ T := by
  have hμ0 : 0 < μ := by linarith
  have hlog : 0 < Real.log μ := Real.log_pos hμ
  rw [div_le_iff₀ hlog]
  have hLd : 0 < Lr * δ := mul_pos hLr hδ
  have e : Lr * (μ ^ T * δ) = μ ^ T * (Lr * δ) := by ring
  rw [e] at hT
  have h1 := Real.log_le_log hm hT
  rw [Real.log_mul (pow_pos hμ0 T).ne' hLd.ne', Real.log_pow] at h1
  rw [Real.log_div hm.ne' hLd.ne']
  linarith

/-- **In 1-D a strictly increasing pass never changes a particle's side of the boundary.**
So basin identity is a function of the start, read at depth `0` by `sign (x - x_u)`; what the
passes add is distance from the boundary (margin) for a reader that cannot resolve it. -/
theorem monotone_side_preserved {G : ℝ → ℝ} (hG : StrictMono G) {xu : ℝ} (hxu : G xu = xu) :
    ∀ T, ∀ x, xu < x → xu < G^[T] x
  | 0, _, hx => hx
  | T + 1, x, hx => by
    rw [Function.iterate_succ_apply']
    have := hG (monotone_side_preserved hG hxu T x hx)
    rwa [hxu] at this

end TulExploration
