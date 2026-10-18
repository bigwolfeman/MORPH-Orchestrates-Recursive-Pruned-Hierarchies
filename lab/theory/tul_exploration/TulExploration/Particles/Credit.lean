/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent (ParticleProver)
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Order.Filter.AtTopBot.Field
import Mathlib.Topology.Algebra.Order.Field
import TulExploration.Smooth
import TulExploration.HardMin

/-!
# P5: gradient credit under the soft mixture and under winner-take-all

## Model

`ℓ k = log p(y | z_k)` is particle `k`'s log-likelihood.  The soft (mixture) objective is
`log ((1/K) ∑ exp ℓ_k)`; the hard (winner-take-all) objective is `maxₖ ℓ_k`.  For the
stability statements the particles are two points `μ ± η` on a line (a symmetric split of
size `η`) and the reader is the Gaussian reader of `Smooth.lean`, `ℓ = -a (x - y)²`, with
targets `y_i` of law `P`.  `V = ∑ P (μ - y)²` is the target spread around the split's centre.

## Main results

* `mixture_grad_resp` : the derivative of the soft objective along particle `k`'s
  log-likelihood is its responsibility `r_k = exp ℓ_k / ∑ exp ℓ_j`.
* **`resp_tendsto_one`**, **`resp_tendsto_zero`** : with temperature `1/β`, the strict winner's
  responsibility tends to `1` and every strict loser's to `0` as `β → ∞`.  Hard WTA is the
  zero-temperature limit of the soft credit.
* **`soft_grad_at_collapse`** : at a collapsed configuration (all particles at one state) the
  soft objective gives EVERY particle the same gradient `ℓ'/K`, for any differentiable
  reader.  Soft credit cannot break a collapse at first order.
* **`soft_split_collapses`** / `soft_split_grows` : for the symmetric split the soft
  objective has `η = 0` as a local max if `2aV < 1` and a local min if `2aV > 1` (the
  second-derivative test; `2aV = 1` is left open).  This is `enumerated_pair_deriv2` read as
  particle dynamics.
* **`hardPair_eq`**, **`hardPair_escape`** : the WTA objective of the split is
  `J(0) + a (2η E|μ - y| - η²)`: it gains at FIRST order, so `η = 0` is never a maximiser
  when `E|μ - y| > 0`, for every reader sharpness.

So "soft credit collapses particles, WTA keeps them apart" is TRUE for a broad reader
(`2aV < 1`) and FALSE as a general claim: a sharp soft reader (`2aV > 1`) also splits.  For
particles drawn from iid random starts (the proposal) the soft threshold is the stricter
`2aV(1 - 1/K) > 1` (`iid_deriv2`, `iid_matched_isLocalMax` in `Smooth.lean`).
-/

namespace TulExploration

open Finset Filter Topology

section Resp

variable {K : ℕ}

/-- **The soft objective's derivative along particle `k` is its responsibility.** -/
theorem mixture_grad_resp (hK : 0 < K) (ℓ : Fin K → ℝ) (k : Fin K) :
    HasDerivAt (fun t => Real.log ((∑ j, Real.exp (ℓ j + if j = k then t else 0)) / K))
      (Real.exp (ℓ k) / ∑ j, Real.exp (ℓ j)) 0 := by
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK
  have hterm : ∀ j : Fin K, HasDerivAt (fun t => Real.exp (ℓ j + if j = k then t else 0))
      (if j = k then Real.exp (ℓ k) else 0) 0 := by
    intro j
    by_cases hj : j = k
    · subst hj
      simp only [if_true]
      have h := ((hasDerivAt_id (0 : ℝ)).const_add (ℓ j)).exp
      simpa using h
    · simp only [hj, if_false, add_zero]
      exact hasDerivAt_const _ _
  have hsum := HasDerivAt.fun_sum (u := univ) (fun j _ => hterm j)
  simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true] at hsum
  have hS : 0 < ∑ j, Real.exp (ℓ j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, mem_univ k⟩
  have hval : (∑ i, Real.exp (ℓ i + if i = k then (0 : ℝ) else 0))
      = ∑ j, Real.exp (ℓ j) := by simp
  have h := (hsum.div_const (K : ℝ)).log (by rw [hval]; exact (div_pos hS hKR).ne')
  rw [hval] at h
  refine h.congr_deriv ?_
  field_simp

/-- Tempered responsibilities, inverse temperature `β`. -/
noncomputable def resp (β : ℝ) (ℓ : Fin K → ℝ) (k : Fin K) : ℝ :=
  Real.exp (β * ℓ k) / ∑ j, Real.exp (β * ℓ j)

lemma resp_nonneg (β : ℝ) (ℓ : Fin K → ℝ) (k : Fin K) : 0 ≤ resp β ℓ k :=
  div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg (fun _ _ => (Real.exp_pos _).le))

lemma resp_sum (β : ℝ) (ℓ : Fin K → ℝ) (k : Fin K) : ∑ j, resp β ℓ j = 1 := by
  unfold resp
  rw [← Finset.sum_div, div_self]
  exact (Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, mem_univ k⟩).ne'

/-- **A strict loser's responsibility tends to `0` at zero temperature.** -/
theorem resp_tendsto_zero (ℓ : Fin K → ℝ) (k j : Fin K) (h : ℓ k < ℓ j) :
    Tendsto (fun β => resp β ℓ k) atTop (𝓝 0) := by
  have hup : ∀ β, resp β ℓ k ≤ Real.exp (β * (ℓ k - ℓ j)) := by
    intro β
    unfold resp
    rw [div_le_iff₀ (Finset.sum_pos (fun i _ => Real.exp_pos _) ⟨j, mem_univ j⟩)]
    have hj : Real.exp (β * ℓ j) ≤ ∑ i, Real.exp (β * ℓ i) :=
      Finset.single_le_sum (f := fun i => Real.exp (β * ℓ i))
        (fun i _ => (Real.exp_pos _).le) (mem_univ j)
    have e : Real.exp (β * (ℓ k - ℓ j)) * Real.exp (β * ℓ j) = Real.exp (β * ℓ k) := by
      rw [← Real.exp_add]; ring_nf
    nlinarith [Real.exp_pos (β * (ℓ k - ℓ j))]
  have hlim : Tendsto (fun β : ℝ => Real.exp (β * (ℓ k - ℓ j))) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp (tendsto_id.atTop_mul_const_of_neg (sub_neg.2 h))
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hlim
    (fun β => resp_nonneg β ℓ k) hup

/-- **The strict winner's responsibility tends to `1`: hard WTA is the zero-temperature
limit of the soft credit.** -/
theorem resp_tendsto_one (ℓ : Fin K → ℝ) (k : Fin K) (h : ∀ j, j ≠ k → ℓ j < ℓ k) :
    Tendsto (fun β => resp β ℓ k) atTop (𝓝 1) := by
  have e : ∀ β, resp β ℓ k = 1 - ∑ j ∈ univ.erase k, resp β ℓ j := by
    intro β
    have := Finset.add_sum_erase univ (fun j => resp β ℓ j) (mem_univ k)
    rw [resp_sum β ℓ k] at this
    linarith
  have hs : Tendsto (fun β => ∑ j ∈ univ.erase k, resp β ℓ j) atTop (𝓝 0) := by
    have := tendsto_finsetSum (univ.erase k) (fun j hj =>
      resp_tendsto_zero ℓ j k (h j (Finset.ne_of_mem_erase hj)))
    simpa using this
  have := (tendsto_const_nhds (x := (1 : ℝ))).sub hs
  simp only [sub_zero] at this
  exact this.congr (fun β => (e β).symm)

end Resp

/-- **At a collapsed configuration the soft credit gives every particle the same gradient.**
All `K` particles sit at `x₀`; move particle `k` alone.  The soft objective's derivative is
`ℓ'(x₀)/K`, the same for every `k`, for any differentiable per-particle score `ℓ`. -/
theorem soft_grad_at_collapse {K : ℕ} (hK : 0 < K) (ℓ : ℝ → ℝ) (ℓ' x₀ : ℝ)
    (hℓ : HasDerivAt ℓ ℓ' x₀) (k : Fin K) :
    HasDerivAt (fun t => Real.log ((∑ j : Fin K, Real.exp (ℓ (if j = k then t else x₀))) / K))
      (ℓ' / K) x₀ := by
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK
  have hterm : ∀ j : Fin K, HasDerivAt (fun t => Real.exp (ℓ (if j = k then t else x₀)))
      (if j = k then Real.exp (ℓ x₀) * ℓ' else 0) x₀ := by
    intro j
    by_cases hj : j = k
    · simp only [hj, if_true]
      exact hℓ.exp
    · simp only [hj, if_false]
      exact hasDerivAt_const _ _
  have hsum := HasDerivAt.fun_sum (u := univ) (fun j _ => hterm j)
  simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true] at hsum
  have hval : (∑ i : Fin K, Real.exp (ℓ (if i = k then x₀ else x₀)))
      = K * Real.exp (ℓ x₀) := by simp [Finset.sum_const, Finset.card_univ]
  have hpos : 0 < (K : ℝ) * Real.exp (ℓ x₀) := mul_pos hKR (Real.exp_pos _)
  have h := (hsum.div_const (K : ℝ)).log (by rw [hval]; exact (div_pos hpos hKR).ne')
  rw [hval] at h
  refine h.congr_deriv ?_
  have : Real.exp (ℓ x₀) ≠ 0 := (Real.exp_pos _).ne'
  field_simp

section Split

variable {ι : Type*} [Fintype ι]

/-- **A broad soft reader collapses the split.**  If `2aV < 1`, `η = 0` is a local maximum of
the soft objective of two particles at `μ ± η`. -/
theorem soft_split_collapses (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ : ℝ)
    (ha : 0 < a) (h : 2 * a * (∑ i, P i * (μ - y i) ^ 2) < 1) :
    IsLocalMax (quadObj P (fun _ : Unit => (1 : ℝ)) y pairW a μ) 0 := by
  apply quadObj_isLocalMax (by norm_num) P hP1 _ y pairW a μ ha
  · simp [m1, pairW, Fin.sum_univ_two]
  · simp [m1, m2, pairW, Fin.sum_univ_two]
    linarith

/-- **A sharp soft reader splits.**  If `2aV > 1`, `η = 0` is a local minimum. -/
theorem soft_split_grows (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ : ℝ)
    (ha : 0 < a) (h : 1 < 2 * a * (∑ i, P i * (μ - y i) ^ 2)) :
    IsLocalMin (quadObj P (fun _ : Unit => (1 : ℝ)) y pairW a μ) 0 := by
  apply quadObj_isLocalMin (by norm_num) P hP1 _ y pairW a μ ha
  · simp [m1, pairW, Fin.sum_univ_two]
  · simp [m1, m2, pairW, Fin.sum_univ_two]
    linarith

/-- The WTA objective of the split `μ ± η`: the closer particle scores each target. -/
noncomputable def hardPair (P y : ι → ℝ) (a μ η : ℝ) : ℝ :=
  ∑ i, P i * max (-a * (μ + η - y i) ^ 2) (-a * (μ - η - y i) ^ 2)

lemma hardPair_term (a μ η yi : ℝ) (ha : 0 ≤ a) (hη : 0 ≤ η) :
    max (-a * (μ + η - yi) ^ 2) (-a * (μ - η - yi) ^ 2)
      = -a * (μ - yi) ^ 2 + a * (2 * η * |μ - yi| - η ^ 2) := by
  have h := max_quad a (μ - yi) η ha hη
  have e1 : μ + η - yi = (μ - yi) + η := by ring
  have e2 : μ - η - yi = (μ - yi) - η := by ring
  rw [e1, e2, h]
  have habs : |μ - yi| ^ 2 = (μ - yi) ^ 2 := sq_abs _
  nlinarith [habs]

/-- **The WTA objective gains at first order.**
`hardPair η = hardPair 0 + a (2 η E|μ - y| - η²)` for every `η ≥ 0`. -/
theorem hardPair_eq (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ η : ℝ) (ha : 0 ≤ a)
    (hη : 0 ≤ η) :
    hardPair P y a μ η
      = hardPair P y a μ 0 + a * (2 * η * ∑ i, P i * |μ - y i| - η ^ 2) := by
  unfold hardPair
  simp_rw [hardPair_term a μ η _ ha hη, hardPair_term a μ 0 _ ha le_rfl]
  have e : ∀ i, P i * (-a * (μ - y i) ^ 2 + a * (2 * η * |μ - y i| - η ^ 2))
      = P i * (-a * (μ - y i) ^ 2 + a * (2 * 0 * |μ - y i| - 0 ^ 2))
        + (2 * a * η) * (P i * |μ - y i|) - (a * η ^ 2) * P i := by
    intro i; ring
  simp_rw [e]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hP1]
  ring

/-- **WTA never collapses the split** when the targets are not all at the centre: every
`0 < η < 2 E|μ - y|` beats `η = 0`, whatever the reader's sharpness `a > 0`. -/
theorem hardPair_escape (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ η : ℝ)
    (ha : 0 < a) (hη0 : 0 < η) (hη : η < 2 * ∑ i, P i * |μ - y i|) :
    hardPair P y a μ 0 < hardPair P y a μ η := by
  rw [hardPair_eq P hP1 y a μ η ha.le hη0.le]
  have : 0 < a * (2 * η * ∑ i, P i * |μ - y i| - η ^ 2) := by
    apply mul_pos ha
    nlinarith
  linarith

end Split

end TulExploration
