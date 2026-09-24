/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.Calculus.Deriv.Polynomial
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Order.Filter.AtTopBot.Basic

/-!
# T2: the best-of-K (hard-min) objective escapes `σ = 0` at first order, and pays for it

## Model

The latent is `x = μ + σ n` with symmetric two-point noise, `n = ±1`, drawn independently
`K = n + 1` times: the `2^K` sign patterns `o : Fin K → Bool` are equally likely.  The reader
scores a latent by `S(x) = -a (x - y)²`, a Gaussian reader with curvature `a` (`a = 1/(2s²)`;
the normaliser does not depend on `x` and is dropped).  The target `y` has a finite law
`P i` on values `y i`.  The best-of-`K` objective (XM's hard min, "only the closest candidate
gets gradient") is `E_y E_o maxₖ S(μ + σ n_k)`.

## Main results

* `bestOf_eq` / `sum_bestOf` : with two-point noise the best of `K` draws depends only on
  whether both signs occur (probability `1 - 2^{1-K}`).  Exact for every score.
* `max_quad` : `max (-a(e+σ)², -a(e-σ)²) = -a(|e|-σ)²` for `σ ≥ 0`.
* **`hardObj_eq`** : the exact closed form
  `E max = -a E e² - a σ² + 2 a (1 - 2/2^K) σ E|e|` for every `σ ≥ 0`, `e = μ - y`.
* `hardObj_hasDerivWithinAt` : the right derivative at `σ = 0` is `2 a (1 - 2/2^K) E|e|`,
  positive for `K ≥ 2` whenever the target is not always at the latent.
* `hardObj_escape` : `σ = 0` is not a maximiser: `σ* = (1 - 2/2^K) E|e|` is strictly better.
* `hardObj_le_opt` : `σ*` is the maximiser over `σ ≥ 0`, with gain `a (σ*)²`.
* `hardObj_one` : at `K = 1` (a single draw read without selection, the deployed sample)
  the objective is `-a E e² - a σ²`: every unit of spread is a pure price.
* **`hard_gain_eq_deploy_price`** : the hard-min gain at `σ*` EQUALS the price a
  single-sample deploy pays at `σ*`.
* `hardObj_useless_spread` : if the target always sits at the latent, spread only costs.
-/

namespace TulExploration

open Finset

section BestOf

variable {n : ℕ}

/-- The best score among the `K = n+1` draws of sign pattern `o`, draw `k` scoring `s (o k)`. -/
def bestOf (s : Bool → ℝ) (o : Fin (n + 1) → Bool) : ℝ :=
  univ.sup' univ_nonempty (fun k => s (o k))

lemma bestOf_const (s : Bool → ℝ) (b : Bool) : bestOf (n := n) s (fun _ => b) = s b := by
  unfold bestOf
  exact Finset.sup'_const univ_nonempty (s b)

lemma bestOf_mixed (s : Bool → ℝ) (o : Fin (n + 1) → Bool) (ht : ∃ k, o k = true)
    (hf : ∃ k, o k = false) : bestOf s o = max (s true) (s false) := by
  unfold bestOf
  apply le_antisymm
  · apply Finset.sup'_le
    intro k _
    cases o k
    · exact le_max_right _ _
    · exact le_max_left _ _
  · obtain ⟨k1, hk1⟩ := ht
    obtain ⟨k2, hk2⟩ := hf
    apply max_le
    · have := Finset.le_sup' (fun k => s (o k)) (mem_univ k1)
      simpa [hk1] using this
    · have := Finset.le_sup' (fun k => s (o k)) (mem_univ k2)
      simpa [hk2] using this

/-- The best of `K` two-point draws, written as `max` plus two corrections that fire only on
the two constant sign patterns. -/
lemma bestOf_eq (s : Bool → ℝ) (o : Fin (n + 1) → Bool) :
    bestOf s o = max (s true) (s false)
      + (if o = (fun _ => true) then s true - max (s true) (s false) else 0)
      + (if o = (fun _ => false) then s false - max (s true) (s false) else 0) := by
  by_cases h1 : o = (fun _ => true)
  · have h2 : ¬ o = (fun _ => false) := by
      intro h
      have := congrFun (h1.symm.trans h) 0
      simp at this
    subst h1
    rw [bestOf_const]
    simp only [if_true, h2, if_false]
    ring
  · by_cases h2 : o = (fun _ => false)
    · subst h2
      rw [bestOf_const]
      simp only [h1, if_false, if_true]
      ring
    · simp only [h1, h2, if_false, add_zero]
      apply bestOf_mixed
      · by_contra hc
        apply h2
        funext k
        simpa using (not_exists.1 hc) k
      · by_contra hc
        apply h1
        funext k
        simpa using (not_exists.1 hc) k

/-- **The sum of the best of `K` over all `2^K` sign patterns.** -/
lemma sum_bestOf (s : Bool → ℝ) :
    ∑ o : Fin (n + 1) → Bool, bestOf s o
      = 2 ^ (n + 1) * max (s true) (s false)
        + (s true - max (s true) (s false)) + (s false - max (s true) (s false)) := by
  simp only [bestOf_eq s, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, nsmul_eq_mul,
    Finset.sum_ite_eq', Finset.mem_univ, if_true]
  push_cast
  ring

end BestOf

/-- For `σ ≥ 0` the closer of the two draws `μ ± σ` scores `-a (|e| - σ)²`, `e = μ - y`. -/
lemma max_quad (a e σ : ℝ) (ha : 0 ≤ a) (hσ : 0 ≤ σ) :
    max (-a * (e + σ) ^ 2) (-a * (e - σ) ^ 2) = -a * (|e| - σ) ^ 2 := by
  rcases le_total 0 e with he | he
  · rw [abs_of_nonneg he]
    apply max_eq_right
    nlinarith [mul_nonneg ha (mul_nonneg he hσ)]
  · rw [abs_of_nonpos he, show (-e - σ) ^ 2 = (e + σ) ^ 2 by ring]
    apply max_eq_left
    nlinarith [mul_nonneg ha (mul_nonneg (neg_nonneg.2 he) hσ)]

section Objective

variable {ι : Type*} [Fintype ι]

/-- The sign of a two-point draw. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

/-- **The best-of-`K` objective**, `K = n + 1`: `E_y E_o maxₖ S_y(μ + σ nₖ)`. -/
noncomputable def hardObj (n : ℕ) (P : ι → ℝ) (y : ι → ℝ) (a μ σ : ℝ) : ℝ :=
  ∑ i, P i * ((1 / 2 ^ (n + 1)) *
    ∑ o : Fin (n + 1) → Bool, bestOf (fun b => -a * (μ + σ * sgn b - y i) ^ 2) o)

/-- The selection gain factor `c_K = 1 - 2/2^K = 1 - 2^{1-K}`: the probability that the `K`
draws contain both signs. -/
noncomputable def cK (n : ℕ) : ℝ := 1 - 2 / 2 ^ (n + 1)

/-- **The exact closed form of the hard-min objective.**  For every `σ ≥ 0`:
`hardObj = -a E e² - a σ² + 2 a c_K σ E|e|`, with `e = μ - y` and `c_K = 1 - 2^{1-K}`.
It is a concave quadratic in `σ` whose LINEAR term is the selection gain. -/
theorem hardObj_eq (n : ℕ) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ σ : ℝ)
    (ha : 0 ≤ a) (hσ : 0 ≤ σ) :
    hardObj n P y a μ σ
      = -a * ∑ i, P i * (μ - y i) ^ 2 - a * σ ^ 2
        + 2 * a * cK n * σ * ∑ i, P i * |μ - y i| := by
  have h2 : (2 : ℝ) ^ (n + 1) ≠ 0 := by positivity
  have term : ∀ i, (1 / 2 ^ (n + 1)) *
      ∑ o : Fin (n + 1) → Bool, bestOf (fun b => -a * (μ + σ * sgn b - y i) ^ 2) o
      = -a * (μ - y i) ^ 2 - a * σ ^ 2 + 2 * a * cK n * σ * |μ - y i| := by
    intro i
    rw [sum_bestOf]
    have hmax : max (-a * (μ + σ * sgn true - y i) ^ 2) (-a * (μ + σ * sgn false - y i) ^ 2)
        = -a * (|μ - y i| - σ) ^ 2 := by
      have := max_quad a (μ - y i) σ ha hσ
      simp only [sgn, if_true, Bool.false_eq_true, if_false]
      convert this using 2 <;> ring
    rw [hmax]
    simp only [sgn, if_true, Bool.false_eq_true, if_false, cK]
    have habs : |μ - y i| ^ 2 = (μ - y i) ^ 2 := sq_abs _
    field_simp
    ring_nf
    rw [habs]
    ring
  unfold hardObj
  simp_rw [term]
  simp only [mul_add, mul_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  rw [show ∑ i, P i * (a * σ ^ 2) = a * σ ^ 2 by rw [← Finset.sum_mul, hP1, one_mul]]
  simp only [Finset.mul_sum]
  apply congrArg₂ _ (congrArg₂ _ ?_ rfl) ?_
  · apply Finset.sum_congr rfl; intro i _; ring
  · apply Finset.sum_congr rfl; intro i _; ring

/-- `c_K > 0` once there are at least two draws (`K = n + 1 ≥ 2`). -/
lemma cK_pos (n : ℕ) (hn : 1 ≤ n) : 0 < cK n := by
  unfold cK
  have : (4 : ℝ) ≤ 2 ^ (n + 1) := by
    calc (4 : ℝ) = 2 ^ 2 := by norm_num
      _ ≤ 2 ^ (n + 1) := pow_le_pow_right₀ (by norm_num) (by omega)
  have : 2 / (2 : ℝ) ^ (n + 1) ≤ 1 / 2 := by
    rw [div_le_iff₀ (by positivity)]
    linarith
  linarith

/-- At one draw (`K = 1`) there is no selection: `c_1 = 0`. -/
lemma cK_zero : cK 0 = 0 := by norm_num [cK]

/-- **The right derivative at `σ = 0` is `2 a c_K E|e|`.**  First order in `σ`: the
selection pays for spread linearly, while the curvature cost is quadratic. -/
theorem hardObj_hasDerivWithinAt (n : ℕ) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ)
    (a μ : ℝ) (ha : 0 ≤ a) :
    HasDerivWithinAt (hardObj n P y a μ) (2 * a * cK n * ∑ i, P i * |μ - y i|)
      (Set.Ici 0) 0 := by
  set E2 := ∑ i, P i * (μ - y i) ^ 2
  set E1 := ∑ i, P i * |μ - y i|
  have hpoly : HasDerivAt (fun σ : ℝ => -a * E2 - a * σ ^ 2 + 2 * a * cK n * σ * E1)
      (2 * a * cK n * E1) 0 := by
    have h1 := (hasDerivAt_pow 2 (0 : ℝ)).const_mul a
    have h2 := ((hasDerivAt_id (0 : ℝ)).const_mul (2 * a * cK n)).mul_const E1
    have h := ((hasDerivAt_const (0 : ℝ) (-a * E2)).sub h1).add h2
    have h' : HasDerivAt (fun σ : ℝ => -a * E2 - a * σ ^ 2 + 2 * a * cK n * σ * E1)
        (0 - a * ((2 : ℕ) * (0 : ℝ) ^ (2 - 1)) + 2 * a * cK n * 1 * E1) 0 := h
    convert h' using 1
    norm_num
  refine hpoly.hasDerivWithinAt.congr_of_mem ?_ (Set.mem_Ici.2 le_rfl)
  intro σ hσ
  exact hardObj_eq n P hP1 y a μ σ ha hσ

/-- The optimal spread of the hard-min objective. -/
noncomputable def sigmaStar (n : ℕ) (P : ι → ℝ) (y : ι → ℝ) (μ : ℝ) : ℝ :=
  cK n * ∑ i, P i * |μ - y i|

/-- **`σ*` maximises the hard-min objective over `σ ≥ 0`, and gains exactly `a (σ*)²`.** -/
theorem hardObj_le_opt (n : ℕ) (P : ι → ℝ) (hP : ∀ i, 0 ≤ P i) (hP1 : ∑ i, P i = 1)
    (y : ι → ℝ) (a μ σ : ℝ) (ha : 0 ≤ a) (hσ : 0 ≤ σ) (hn : 1 ≤ n) :
    hardObj n P y a μ σ ≤ hardObj n P y a μ (sigmaStar n P y μ)
      ∧ hardObj n P y a μ (sigmaStar n P y μ) - hardObj n P y a μ 0
          = a * sigmaStar n P y μ ^ 2 := by
  have hE1 : 0 ≤ ∑ i, P i * |μ - y i| :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (hP i) (abs_nonneg _))
  have hs : 0 ≤ sigmaStar n P y μ := mul_nonneg (cK_pos n hn).le hE1
  rw [hardObj_eq n P hP1 y a μ σ ha hσ, hardObj_eq n P hP1 y a μ _ ha hs,
    hardObj_eq n P hP1 y a μ 0 ha le_rfl]
  unfold sigmaStar
  constructor
  · nlinarith [mul_nonneg ha (sq_nonneg (σ - cK n * ∑ i, P i * |μ - y i|))]
  · ring

/-- **`σ = 0` is not a maximiser of the hard-min objective** once `K ≥ 2` and the target is
not always at the latent (`E|e| > 0`, `a > 0`). -/
theorem hardObj_escape (n : ℕ) (P : ι → ℝ) (hP : ∀ i, 0 ≤ P i) (hP1 : ∑ i, P i = 1)
    (y : ι → ℝ) (a μ : ℝ) (ha : 0 < a) (hn : 1 ≤ n) (hE : 0 < ∑ i, P i * |μ - y i|) :
    hardObj n P y a μ 0 < hardObj n P y a μ (sigmaStar n P y μ) := by
  have h := (hardObj_le_opt n P hP hP1 y a μ 0 ha.le le_rfl hn).2
  have hs : 0 < sigmaStar n P y μ := mul_pos (cK_pos n hn) hE
  have : 0 < a * sigmaStar n P y μ ^ 2 := by positivity
  linarith

/-- **One draw, no selection: spread is a pure price.**  At `K = 1` the objective is
`-a E e² - a σ²`.  This is also what a deployed model that reads ONE sample pays. -/
theorem hardObj_one (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ σ : ℝ) (ha : 0 ≤ a)
    (hσ : 0 ≤ σ) :
    hardObj 0 P y a μ σ = -a * ∑ i, P i * (μ - y i) ^ 2 - a * σ ^ 2 := by
  rw [hardObj_eq 0 P hP1 y a μ σ ha hσ, cK_zero]
  ring

/-- **The hard-min gain equals the single-sample deploy price.**  Train the spread with the
best-of-`K` objective and it settles at `σ*`, gaining `a (σ*)²` in that objective.  A
deployment that reads one sample (no selection) loses exactly `a (σ*)²` against `σ = 0`. -/
theorem hard_gain_eq_deploy_price (n : ℕ) (P : ι → ℝ) (hP : ∀ i, 0 ≤ P i)
    (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ : ℝ) (ha : 0 ≤ a) (hn : 1 ≤ n) :
    hardObj n P y a μ (sigmaStar n P y μ) - hardObj n P y a μ 0
      = hardObj 0 P y a μ 0 - hardObj 0 P y a μ (sigmaStar n P y μ) := by
  have hE1 : 0 ≤ ∑ i, P i * |μ - y i| :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (hP i) (abs_nonneg _))
  have hs : 0 ≤ sigmaStar n P y μ := mul_nonneg (cK_pos n hn).le hE1
  rw [(hardObj_le_opt n P hP hP1 y a μ 0 ha le_rfl hn).2, hardObj_one P hP1 y a μ 0 ha le_rfl,
    hardObj_one P hP1 y a μ _ ha hs]
  ring

/-- **Useless spread.**  If the target always sits at the latent (`y i = μ` wherever
`P i > 0`), the hard min buys nothing and the spread costs `a σ²` at every `K`. -/
theorem hardObj_useless_spread (n : ℕ) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ)
    (a μ σ : ℝ) (ha : 0 ≤ a) (hσ : 0 ≤ σ) (hy : ∀ i, P i ≠ 0 → y i = μ) :
    hardObj n P y a μ σ = hardObj n P y a μ 0 - a * σ ^ 2 := by
  have z1 : ∑ i, P i * |μ - y i| = 0 := by
    apply Finset.sum_eq_zero; intro i _
    by_cases h : P i = 0
    · simp [h]
    · simp [hy i h]
  rw [hardObj_eq n P hP1 y a μ σ ha hσ, hardObj_eq n P hP1 y a μ 0 ha le_rfl, z1]
  ring

end Objective

end TulExploration
