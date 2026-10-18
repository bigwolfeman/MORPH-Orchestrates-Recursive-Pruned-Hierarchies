/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent (ParticleProver)
-/
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fin.Tuple.Basic
import TulExploration.Bayes

/-!
# P3: exchangeable particles make the mixture read monotone in `K` (and role cells do not)

## Model

One context and one target `y`, held fixed.  `Z` is the finite set of particle exit states.
A particle is a draw `z ~ q` (the law of the coarse loop's exit state from a random start;
every particle is run by the same map from an independent start, so the `K` draws are iid).
`p z = p(y | z) > 0` is the likelihood the reader gives the target from particle `z`.  The
uniform mixture read of `K` particles is `log ((1/K) ∑ₖ p(zₖ))`; its expectation over the iid
draws is `iwae q p K` (the IWAE bound of Burda, Grosse and Salakhutdinov 2015).

## Main results

* `iid_total` : the iid product law sums to one.
* `iid_drop` : dropping coordinate `j` of an iid `(n+1)`-tuple leaves an iid `n`-tuple.
* `iid_marginal` : one coordinate of an iid tuple has law `q`.
* `iwae_one` : `iwae q p 1 = ∑ q log p` (one particle, the plain loop's read).
* **`iwae_mono`** : `iwae q p K ≤ iwae q p (K + 1)` for every `K ≥ 1`.  Adding an
  exchangeable particle never lowers the expected mixture read.
* **`iwae_le_log_mean`** : `iwae q p K ≤ log ∑ q p` for every `K ≥ 1`.
* `iwae_mono_avg`, `iwae_avg_le_bayes` : the same statements averaged over targets, and the
  Bayes ceiling (`mixture_le_bayes`).
* **`add_candidate_iff`** : for FIXED candidates (role-indexed cells, not exchangeable),
  adding a candidate raises the uniform mixture read iff its likelihood is at least the
  current mean likelihood.
* `role_cell_can_hurt` : so a worse role cell strictly lowers the read.  The monotonicity in
  `K` is a property of exchangeability, not of mixtures.
-/

namespace TulExploration

open Finset

section IID

variable {Z : Type*} [Fintype Z]

/-- **The expected uniform-mixture read of `K` iid particles** (the IWAE bound). -/
noncomputable def iwae (q p : Z → ℝ) (K : ℕ) : ℝ :=
  ∑ z : Fin K → Z, (∏ k, q (z k)) * Real.log ((∑ k, p (z k)) / K)

/-- The iid product law sums to one. -/
lemma iid_total (q : Z → ℝ) (hq1 : ∑ a, q a = 1) (n : ℕ) :
    ∑ w : Fin n → Z, ∏ i, q (w i) = 1 := by
  classical
  have h := Fintype.prod_sum (fun (_ : Fin n) (a : Z) => q a)
  simp only [hq1, Finset.prod_const_one] at h
  exact h.symm

/-- **Dropping one coordinate of an iid tuple.**  Coordinate `j` of an iid `(n+1)`-tuple is
summed out; what is left is an iid `n`-tuple. -/
lemma iid_drop (q : Z → ℝ) (hq1 : ∑ a, q a = 1) {n : ℕ} (j : Fin (n + 1))
    (g : (Fin n → Z) → ℝ) :
    ∑ z : Fin (n + 1) → Z, (∏ k, q (z k)) * g (fun i => z (j.succAbove i))
      = ∑ w : Fin n → Z, (∏ i, q (w i)) * g w := by
  rw [← (Fin.insertNthEquiv (fun _ => Z) j).sum_comp, Fintype.sum_prod_type]
  simp only [Fin.insertNthEquiv_apply, Fin.prod_univ_succAbove _ j, Fin.insertNth_apply_same,
    Fin.insertNth_apply_succAbove]
  have e : ∀ a : Z, ∑ w : Fin n → Z, q a * (∏ i, q (w i)) * g w
      = q a * ∑ w : Fin n → Z, (∏ i, q (w i)) * g w := by
    intro a
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro w _
    ring
  simp_rw [e]
  rw [← Finset.sum_mul, hq1, one_mul]

/-- **One coordinate of an iid tuple has law `q`.** -/
lemma iid_marginal (q : Z → ℝ) (hq1 : ∑ a, q a = 1) {n : ℕ} (j : Fin (n + 1))
    (f : Z → ℝ) :
    ∑ z : Fin (n + 1) → Z, (∏ k, q (z k)) * f (z j) = ∑ a, q a * f a := by
  rw [← (Fin.insertNthEquiv (fun _ => Z) j).sum_comp, Fintype.sum_prod_type]
  simp only [Fin.insertNthEquiv_apply, Fin.prod_univ_succAbove _ j, Fin.insertNth_apply_same,
    Fin.insertNth_apply_succAbove]
  apply Finset.sum_congr rfl
  intro a _
  have e : ∑ w : Fin n → Z, q a * (∏ i, q (w i)) * f a
      = q a * f a * ∑ w : Fin n → Z, ∏ i, q (w i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro w _
    ring
  rw [e, iid_total q hq1, mul_one]

/-- **One particle is the plain read**: `iwae q p 1 = ∑ q log p`. -/
theorem iwae_one (q p : Z → ℝ) (hq1 : ∑ a, q a = 1) :
    iwae q p 1 = ∑ a, q a * Real.log (p a) := by
  unfold iwae
  have := iid_marginal q hq1 (n := 0) 0 (fun a => Real.log (p a))
  simpa [Fin.sum_univ_one] using this

/-- The leave-one-out identity: summing, over `j`, the sum of the other `n` coordinates
counts every coordinate `n` times. -/
lemma sum_leave_one_out {n : ℕ} (f : Fin (n + 1) → ℝ) :
    ∑ j : Fin (n + 1), ∑ i : Fin n, f (j.succAbove i) = n * ∑ k, f k := by
  have h : ∀ j : Fin (n + 1), ∑ i : Fin n, f (j.succAbove i) = (∑ k, f k) - f j := by
    intro j
    rw [Fin.sum_univ_succAbove f j]
    ring
  simp_rw [h]
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  push_cast
  ring

/-- **The per-draw Jensen step.**  The mean of `n+1` positive numbers is the average of
the `n+1` leave-one-out means, so its log is at least the average of their logs. -/
lemma log_mean_ge_avg_loo {n : ℕ} (hn : 0 < n) (f : Fin (n + 1) → ℝ) (hf : ∀ k, 0 < f k) :
    (∑ j : Fin (n + 1), Real.log ((∑ i : Fin n, f (j.succAbove i)) / n)) / (n + 1)
      ≤ Real.log ((∑ k, f k) / (n + 1 : ℕ)) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hpos : ∀ j : Fin (n + 1), 0 < (∑ i : Fin n, f (j.succAbove i)) / n := by
    intro j
    have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
    exact div_pos (Finset.sum_pos (fun i _ => hf _) univ_nonempty) hnR
  have hJ := sum_mul_log_le_log_sum (fun _ : Fin (n + 1) => (1 : ℝ) / (n + 1))
    (fun j => (∑ i : Fin n, f (j.succAbove i)) / n) (fun _ => by positivity)
    (by simp [Finset.sum_const]; field_simp) hpos
  have eL : ∑ j : Fin (n + 1), (1 : ℝ) / (n + 1) * Real.log ((∑ i : Fin n, f (j.succAbove i)) / n)
      = (∑ j : Fin (n + 1), Real.log ((∑ i : Fin n, f (j.succAbove i)) / n)) / (n + 1) := by
    rw [← Finset.mul_sum]; ring
  have eR : ∑ j : Fin (n + 1), (1 : ℝ) / (n + 1) * ((∑ i : Fin n, f (j.succAbove i)) / n)
      = (∑ k, f k) / (n + 1 : ℕ) := by
    rw [← Finset.mul_sum, ← Finset.sum_div, sum_leave_one_out f]
    push_cast
    field_simp
  rw [eL, eR] at hJ
  exact hJ

/-- **IWAE monotonicity: an extra exchangeable particle never lowers the expected mixture
read.**  For every `n ≥ 1` (`K = n` particles), `iwae q p n ≤ iwae q p (n + 1)`.

Proof (Burda et al. 2015, Thm 1): per draw, the `(n+1)`-mean is the average of its `n+1`
leave-one-out `n`-means (`log_mean_ge_avg_loo`, Jensen); each leave-one-out tuple of an iid
`(n+1)`-tuple is an iid `n`-tuple (`iid_drop`), so each term averages to `iwae q p n`. -/
theorem iwae_mono (q p : Z → ℝ) (hq : ∀ a, 0 ≤ q a) (hq1 : ∑ a, q a = 1)
    (hp : ∀ a, 0 < p a) {n : ℕ} (hn : 0 < n) :
    iwae q p n ≤ iwae q p (n + 1) := by
  have step : ∀ z : Fin (n + 1) → Z,
      (∏ k, q (z k)) *
        ((∑ j : Fin (n + 1), Real.log ((∑ i : Fin n, p (z (j.succAbove i))) / n)) / (n + 1))
      ≤ (∏ k, q (z k)) * Real.log ((∑ k, p (z k)) / (n + 1 : ℕ)) := by
    intro z
    apply mul_le_mul_of_nonneg_left _ (Finset.prod_nonneg (fun k _ => hq _))
    exact log_mean_ge_avg_loo hn (fun k => p (z k)) (fun k => hp _)
  have hsum := Finset.sum_le_sum (fun z (_ : z ∈ (univ : Finset (Fin (n + 1) → Z))) => step z)
  have lhs : ∑ z : Fin (n + 1) → Z, (∏ k, q (z k)) *
        ((∑ j : Fin (n + 1), Real.log ((∑ i : Fin n, p (z (j.succAbove i))) / n)) / (n + 1))
      = iwae q p n := by
    have e : ∀ z : Fin (n + 1) → Z, (∏ k, q (z k)) *
        ((∑ j : Fin (n + 1), Real.log ((∑ i : Fin n, p (z (j.succAbove i))) / n)) / (n + 1))
        = (∑ j : Fin (n + 1), (∏ k, q (z k)) *
            Real.log ((∑ i : Fin n, p (z (j.succAbove i))) / n)) / (n + 1) := by
      intro z
      rw [mul_div_assoc', Finset.mul_sum]
    simp_rw [e]
    rw [← Finset.sum_div, Finset.sum_comm]
    have e2 : ∀ j : Fin (n + 1), ∑ z : Fin (n + 1) → Z, (∏ k, q (z k)) *
        Real.log ((∑ i : Fin n, p (z (j.succAbove i))) / n) = iwae q p n := by
      intro j
      exact iid_drop q hq1 j (fun w => Real.log ((∑ i, p (w i)) / n))
    simp_rw [e2]
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    push_cast
    field_simp
  rw [lhs] at hsum
  exact hsum

/-- **The ceiling: `iwae q p K ≤ log E_q p`** for every `K ≥ 1`.  Jensen over the draw, and
the mean of `K` iid likelihoods averages to `E_q p`. -/
theorem iwae_le_log_mean (q p : Z → ℝ) (hq : ∀ a, 0 ≤ q a) (hq1 : ∑ a, q a = 1)
    (hp : ∀ a, 0 < p a) (n : ℕ) :
    iwae q p (n + 1) ≤ Real.log (∑ a, q a * p a) := by
  have hρ : ∀ z : Fin (n + 1) → Z, 0 ≤ ∏ k, q (z k) :=
    fun z => Finset.prod_nonneg (fun k _ => hq _)
  have hpos : ∀ z : Fin (n + 1) → Z, 0 < (∑ k, p (z k)) / (n + 1 : ℕ) :=
    fun z => div_pos (Finset.sum_pos (fun k _ => hp _) univ_nonempty) (by positivity)
  have hJ := sum_mul_log_le_log_sum (fun z : Fin (n + 1) → Z => ∏ k, q (z k))
    (fun z => (∑ k, p (z k)) / (n + 1 : ℕ)) hρ (iid_total q hq1 (n + 1)) hpos
  have e : ∑ z : Fin (n + 1) → Z, (∏ k, q (z k)) * ((∑ k, p (z k)) / (n + 1 : ℕ))
      = ∑ a, q a * p a := by
    have e1 : ∀ z : Fin (n + 1) → Z, (∏ k, q (z k)) * ((∑ k, p (z k)) / (n + 1 : ℕ))
        = (∑ k, (∏ k, q (z k)) * p (z k)) / (n + 1 : ℕ) := by
      intro z
      rw [mul_div_assoc', Finset.mul_sum]
    simp_rw [e1]
    rw [← Finset.sum_div, Finset.sum_comm]
    simp_rw [iid_marginal q hq1 _ p]
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have : ((n + 1 : ℕ) : ℝ) ≠ 0 := by positivity
    field_simp
  unfold iwae
  rw [e] at hJ
  exact hJ

/-- **Train at `m`, infer at `n ≥ m`.**  The expected mixture read of iid particles at `n`
is at least that at `m`, for every `1 ≤ m ≤ n`. -/
theorem iwae_mono_le (q p : Z → ℝ) (hq : ∀ a, 0 ≤ q a) (hq1 : ∑ a, q a = 1)
    (hp : ∀ a, 0 < p a) {m n : ℕ} (hm : 0 < m) (hmn : m ≤ n) :
    iwae q p m ≤ iwae q p n := by
  induction n, hmn using Nat.le_induction with
  | base => exact le_rfl
  | succ k hmk ih => exact ih.trans (iwae_mono q p hq hq1 hp (lt_of_lt_of_le hm hmk))

/-- **Averaged over targets.**  With a reader `r z y > 0` and any target weights `P ≥ 0`,
`∑ P y · iwae q (r · y) K` is nondecreasing in `K ≥ 1`. -/
theorem iwae_mono_avg {Y : Type*} [Fintype Y] (P : Y → ℝ) (hP : ∀ y, 0 ≤ P y)
    (q : Z → ℝ) (hq : ∀ a, 0 ≤ q a) (hq1 : ∑ a, q a = 1) (r : Z → Y → ℝ)
    (hr : ∀ z y, 0 < r z y) {n : ℕ} (hn : 0 < n) :
    ∑ y, P y * iwae q (fun z => r z y) n ≤ ∑ y, P y * iwae q (fun z => r z y) (n + 1) :=
  Finset.sum_le_sum (fun y _ => mul_le_mul_of_nonneg_left
    (iwae_mono q (fun z => r z y) hq hq1 (fun z => hr z y) hn) (hP y))

/-- **The Bayes ceiling still holds.**  Averaged over the true target law `P`, the expected
mixture read of `K` iid particles is at most `∑ P log P` (this is `multisample_le_bayes`
with the iid sampler). -/
theorem iwae_avg_le_bayes {Y : Type*} [Fintype Y] (P : Y → ℝ) (hP : ∀ y, 0 ≤ P y)
    (hP1 : ∑ y, P y = 1) (q : Z → ℝ) (hq : ∀ a, 0 ≤ q a) (hq1 : ∑ a, q a = 1)
    (r : Z → Y → ℝ) (hr : ∀ z y, 0 < r z y) (hr1 : ∀ z, ∑ y, r z y = 1) (n : ℕ) :
    ∑ y, P y * iwae q (fun z => r z y) (n + 1) ≤ ∑ y, P y * Real.log (P y) := by
  unfold iwae
  exact multisample_le_bayes (Nat.succ_pos n) P hP hP1 (fun z : Fin (n + 1) → Z => ∏ k, q (z k))
    (fun z => Finset.prod_nonneg (fun k _ => hq _)) (iid_total q hq1 (n + 1)) (fun z k => z k)
    r hr hr1

/-! ### Any concave read of the particles' MEAN value is monotone too -/

/-- The mean of the values `g` of `n` particles, as a convex combination. -/
noncomputable def vmean {V : Type*} [AddCommGroup V] [Module ℝ V] (g : Z → V) (n : ℕ)
    (z : Fin n → Z) : V :=
  ∑ k, ((1 : ℝ) / n) • g (z k)

/-- The vector leave-one-out identity. -/
lemma vsum_leave_one_out {V : Type*} [AddCommGroup V] [Module ℝ V] {n : ℕ}
    (f : Fin (n + 1) → V) :
    ∑ j : Fin (n + 1), ∑ i : Fin n, f (j.succAbove i) = (n : ℝ) • ∑ k, f k := by
  have h : ∀ j : Fin (n + 1), ∑ i : Fin n, f (j.succAbove i) = (∑ k, f k) - f j := by
    intro j
    rw [Fin.sum_univ_succAbove f j]
    abel
  simp_rw [h]
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    succ_nsmul, add_sub_cancel_right, Nat.cast_smul_eq_nsmul]

omit [Fintype Z] in
/-- The `(n+1)`-mean is the average of the `n+1` leave-one-out `n`-means. -/
lemma vmean_loo {V : Type*} [AddCommGroup V] [Module ℝ V] (g : Z → V) {n : ℕ} (hn : 0 < n)
    (z : Fin (n + 1) → Z) :
    ∑ j : Fin (n + 1), ((1 : ℝ) / (n + 1)) • vmean g n (fun i => z (j.succAbove i))
      = vmean g (n + 1) z := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold vmean
  simp_rw [Finset.smul_sum, smul_smul]
  have e : ∑ j : Fin (n + 1), ∑ i : Fin n, ((1 : ℝ) / (n + 1) * (1 / n)) • g (z (j.succAbove i))
      = ((1 : ℝ) / (n + 1) * (1 / n)) • ∑ j : Fin (n + 1), ∑ i : Fin n, g (z (j.succAbove i)) := by
    rw [Finset.smul_sum]
    simp_rw [Finset.smul_sum]
  rw [e, vsum_leave_one_out (fun k => g (z k)), smul_smul, Finset.smul_sum]
  apply Finset.sum_congr rfl
  intro k _
  congr 1
  push_cast
  field_simp

/-- **A concave read of the mean value is monotone in `K` for exchangeable particles.**
`f` is concave on a convex set `s` holding every particle value `g a`.  Then
`E f(mean of K iid values)` is nondecreasing in `K ≥ 1`.  The attention read with UNIFORM
weights and a linear head is this with `f = log softmax(·)(y)`, which is concave (the
convexity of log-sum-exp, a standard fact not proved in this file): its limit is the read of
the mean particle (a committee), not the mixture `log E p`. -/
theorem concave_mean_mono {V : Type*} [AddCommGroup V] [Module ℝ V] (s : Set V)
    (hs : Convex ℝ s) (f : V → ℝ) (hf : ConcaveOn ℝ s f) (q : Z → ℝ) (hq : ∀ a, 0 ≤ q a)
    (hq1 : ∑ a, q a = 1) (g : Z → V) (hg : ∀ a, g a ∈ s) {n : ℕ} (hn : 0 < n) :
    ∑ z : Fin n → Z, (∏ k, q (z k)) * f (vmean g n z)
      ≤ ∑ z : Fin (n + 1) → Z, (∏ k, q (z k)) * f (vmean g (n + 1) z) := by
  have hmem : ∀ (m : ℕ), 0 < m → ∀ z : Fin m → Z, vmean g m z ∈ s := by
    intro m hm z
    have hmR : (0 : ℝ) < m := by exact_mod_cast hm
    apply hs.sum_mem (fun k _ => by positivity) _ (fun k _ => hg _)
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  have step : ∀ z : Fin (n + 1) → Z,
      (∏ k, q (z k)) * ((∑ j : Fin (n + 1), f (vmean g n (fun i => z (j.succAbove i))))
        / (n + 1))
      ≤ (∏ k, q (z k)) * f (vmean g (n + 1) z) := by
    intro z
    apply mul_le_mul_of_nonneg_left _ (Finset.prod_nonneg (fun k _ => hq _))
    have hJ := hf.le_map_sum (t := univ) (w := fun _ : Fin (n + 1) => (1 : ℝ) / (n + 1))
      (p := fun j => vmean g n (fun i => z (j.succAbove i))) (fun _ _ => by positivity)
      (by simp [Finset.sum_const]; field_simp) (fun j _ => hmem n hn _)
    rw [vmean_loo g hn z] at hJ
    have e : ∑ j : Fin (n + 1), ((1 : ℝ) / (n + 1)) • f (vmean g n (fun i => z (j.succAbove i)))
        = (∑ j : Fin (n + 1), f (vmean g n (fun i => z (j.succAbove i)))) / (n + 1) := by
      simp only [smul_eq_mul]
      rw [← Finset.mul_sum]
      ring
    rw [e] at hJ
    exact hJ
  have hsum := Finset.sum_le_sum (fun z (_ : z ∈ (univ : Finset (Fin (n + 1) → Z))) => step z)
  have lhs : ∑ z : Fin (n + 1) → Z, (∏ k, q (z k)) *
      ((∑ j : Fin (n + 1), f (vmean g n (fun i => z (j.succAbove i)))) / (n + 1))
      = ∑ z : Fin n → Z, (∏ k, q (z k)) * f (vmean g n z) := by
    have e : ∀ z : Fin (n + 1) → Z, (∏ k, q (z k)) *
        ((∑ j : Fin (n + 1), f (vmean g n (fun i => z (j.succAbove i)))) / (n + 1))
        = (∑ j : Fin (n + 1), (∏ k, q (z k)) * f (vmean g n (fun i => z (j.succAbove i))))
          / (n + 1) := by
      intro z
      rw [mul_div_assoc', Finset.mul_sum]
    simp_rw [e]
    rw [← Finset.sum_div, Finset.sum_comm]
    simp_rw [fun j : Fin (n + 1) => iid_drop q hq1 j (fun w => f (vmean g n w))]
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    push_cast
    field_simp
  rw [lhs] at hsum
  exact hsum

end IID

/-! ## Role-indexed cells: no monotonicity without exchangeability -/

/-- **Adding a fixed candidate helps iff it beats the current mean.**  For `K ≥ 1` fixed
candidate likelihoods with sum `S > 0` and a new one `p > 0`:
`log (S/K) ≤ log ((S + p)/(K+1))` iff `S/K ≤ p`.  A role cell is not a fresh draw of the
same law, so nothing forces this; exchangeability is what makes it hold in expectation. -/
theorem add_candidate_iff {S p : ℝ} {K : ℕ} (hK : 0 < K) (hS : 0 < S) (hp : 0 < p) :
    Real.log (S / K) ≤ Real.log ((S + p) / (K + 1)) ↔ S / K ≤ p := by
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK
  rw [Real.log_le_log_iff (div_pos hS hKR) (div_pos (by linarith) (by linarith)),
    div_le_div_iff₀ hKR (by linarith), div_le_iff₀ hKR]
  constructor <;> intro h <;> nlinarith

/-- **A worse role cell strictly lowers the read.**  One good candidate (`p₁`) and a second,
worse one (`p₂ < p₁`): the two-candidate uniform mixture is strictly below the one-candidate
read.  In the measured variable-cell-count v1 run, extra fresh or cyclic cells cost
+0.05 to +0.14 nats; this is the shape of that failure. -/
theorem role_cell_can_hurt {p₁ p₂ : ℝ} (h₂ : 0 < p₂) (h : p₂ < p₁) :
    Real.log ((p₁ + p₂) / 2) < Real.log p₁ := by
  apply Real.log_lt_log (div_pos (by linarith) two_pos)
  linarith

end TulExploration
