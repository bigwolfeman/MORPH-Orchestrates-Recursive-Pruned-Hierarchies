/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Algebra.BigOperators.Field
import TulInformation.Basic

/-!
# T3: a sampled latent cannot beat the Bayes read, and when a bounded reader lets it win

## Model

One context, held fixed (the slot's prefix `c`; every statement is per context, and the
context average is a sum of these). `Y` is the finite alphabet of the target (a token, or a
whole span). `P` is the true law of the target given the context.

A *reader* is a family of pmfs `r z` on `Y`, one per latent value `z`, strictly positive
(a softmax output). A *sampler* is a finite law `ρ` over draws `o : Ω`; draw `o` hands the
reader `K` latents `z o 0, …, z o (K-1)`. They may be independent, antithetic, enumerated or
anything else: nothing below needs a product structure. The *multi-sample read* of draw `o`
is the mixture `(1/K) ∑ₖ r (z o k)`; the multi-sample bound (the GK training loss, and the
deployed per-span Bayesian read) is its expected log-likelihood.

## Main results

* `sum_mul_log_le_log_sum` : Jensen for `log` on a finite law.
* `mixture_le_bayes` : ANY random mixture read scores at most `∑ P log P`, the Bayes read.
* `multisample_le_bayes` : the multi-sample bound, at every `K`, under every sampler, with
  every reader, is at most the Bayes read.
* `multisample_bayes_attained` : a hedging reader that ignores the latent attains it, so a
  deterministic encoding is always optimal when the reader class can hedge.
* `product_reader_le` : on a perfectly correlated pair, every product reader (a parallel
  span decode) scores at most `-2 log 2`, whatever deterministic latent it reads.
* `enumerated_pair_gt` : two enumerated codes with committed product readers score
  `log (((1-ε)²+ε²)/2) > -2 log 2` for every `ε ≠ 1/2`: width strictly helps a bounded reader.
* `iid_pair_le` : the same readers fed two INDEPENDENT uniform codes never beat `-2 log 2`:
  at small `K` the collisions of iid sampling eat the gain, enumeration does not.
-/

namespace TulExploration

open Finset

/-- **Jensen for `log` on a finite law.**  `∑ ρ log q ≤ log ∑ ρ q` for positive `q`. -/
lemma sum_mul_log_le_log_sum {Ω : Type*} [Fintype Ω] (ρ q : Ω → ℝ)
    (hρ : ∀ o, 0 ≤ ρ o) (hρ1 : ∑ o, ρ o = 1) (hq : ∀ o, 0 < q o) :
    ∑ o, ρ o * Real.log (q o) ≤ Real.log (∑ o, ρ o * q o) := by
  have h := (strictConcaveOn_log_Ioi.concaveOn).le_map_sum (t := univ) (w := ρ) (p := q)
    (fun o _ => hρ o) hρ1 (fun o _ => Set.mem_Ioi.2 (hq o))
  simpa [smul_eq_mul] using h

/-- A positive combination of positive numbers under a law is positive. -/
lemma sum_mul_pos_of_law {Ω : Type*} [Fintype Ω] (ρ q : Ω → ℝ)
    (hρ : ∀ o, 0 ≤ ρ o) (hρ1 : ∑ o, ρ o = 1) (hq : ∀ o, 0 < q o) :
    0 < ∑ o, ρ o * q o := by
  have hne : ∃ o, 0 < ρ o := by
    by_contra hc
    simp only [not_exists, not_lt] at hc
    have : ∑ o, ρ o ≤ 0 := Finset.sum_nonpos (fun o _ => hc o)
    linarith
  obtain ⟨o0, ho0⟩ := hne
  have hle : ρ o0 * q o0 ≤ ∑ o, ρ o * q o :=
    Finset.single_le_sum (f := fun o => ρ o * q o)
      (fun o _ => mul_nonneg (hρ o) (hq o).le) (mem_univ o0)
  exact lt_of_lt_of_le (mul_pos ho0 (hq o0)) hle

/-- **Any random mixture read is at most the Bayes read.**  `q o` is the pmf the reader
outputs on draw `o`; the draw is random with law `ρ`; the target has law `P`.  Then the
expected log-likelihood `∑ P y ∑ ρ o log (q o y)` is at most `∑ P y log (P y)`, the log-loss
of the Bayes predictor (minus the conditional entropy of the target given the context).

Proof: Jensen moves the draw average inside the `log` (the mixture `m = ∑ ρ q` is a pmf),
then Gibbs' inequality (`TulInformation.sum_mul_log_div_le_zero`) says no pmf beats `P`. -/
theorem mixture_le_bayes {Y Ω : Type*} [Fintype Y] [Fintype Ω]
    (P : Y → ℝ) (hP : ∀ y, 0 ≤ P y) (hP1 : ∑ y, P y = 1)
    (ρ : Ω → ℝ) (hρ : ∀ o, 0 ≤ ρ o) (hρ1 : ∑ o, ρ o = 1)
    (q : Ω → Y → ℝ) (hq : ∀ o y, 0 < q o y) (hq1 : ∀ o, ∑ y, q o y = 1) :
    ∑ y, P y * ∑ o, ρ o * Real.log (q o y) ≤ ∑ y, P y * Real.log (P y) := by
  set m : Y → ℝ := fun y => ∑ o, ρ o * q o y with hm
  have hmpos : ∀ y, 0 < m y := fun y => sum_mul_pos_of_law ρ (fun o => q o y) hρ hρ1
    (fun o => hq o y)
  have hm1 : ∑ y, m y = 1 := by
    simp only [hm]
    rw [Finset.sum_comm]
    simp [← Finset.mul_sum, hq1, hρ1]
  -- step 1: Jensen, target by target
  have step1 : ∑ y, P y * ∑ o, ρ o * Real.log (q o y) ≤ ∑ y, P y * Real.log (m y) := by
    apply Finset.sum_le_sum
    intro y _
    exact mul_le_mul_of_nonneg_left
      (sum_mul_log_le_log_sum ρ (fun o => q o y) hρ hρ1 (fun o => hq o y)) (hP y)
  -- step 2: Gibbs
  have gibbs := TulInformation.sum_mul_log_div_le_zero P m hP (fun y => (hmpos y).le)
    (fun y _ => hmpos y) hP1 hm1.le
  have split : ∑ y, P y * Real.log (m y / P y)
      = ∑ y, P y * Real.log (m y) - ∑ y, P y * Real.log (P y) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro y _
    rcases (hP y).lt_or_eq with h | h
    · rw [Real.log_div (hmpos y).ne' h.ne']
      ring
    · simp [← h]
  linarith

/-- **The multi-sample bound never beats the Bayes read.**  At every `K ≥ 1`, under every
sampler `ρ` (independent, antithetic, enumerated), with every positive reader `r`, the
expected log of the `K`-sample mixture is at most `∑ P log P`. -/
theorem multisample_le_bayes {Y Z Ω : Type*} [Fintype Y] [Fintype Ω] {K : ℕ} (hK : 0 < K)
    (P : Y → ℝ) (hP : ∀ y, 0 ≤ P y) (hP1 : ∑ y, P y = 1)
    (ρ : Ω → ℝ) (hρ : ∀ o, 0 ≤ ρ o) (hρ1 : ∑ o, ρ o = 1)
    (z : Ω → Fin K → Z) (r : Z → Y → ℝ) (hr : ∀ w y, 0 < r w y) (hr1 : ∀ w, ∑ y, r w y = 1) :
    ∑ y, P y * ∑ o, ρ o * Real.log ((∑ k, r (z o k) y) / K)
      ≤ ∑ y, P y * Real.log (P y) := by
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  apply mixture_le_bayes P hP hP1 ρ hρ hρ1 (fun o y => (∑ k, r (z o k) y) / K)
  · intro o y
    apply div_pos _ hKpos
    have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
    exact Finset.sum_pos (fun k _ => hr _ y) univ_nonempty
  · intro o
    rw [← Finset.sum_div, Finset.sum_comm]
    simp [hr1, hKpos.ne']

/-- **A hedging reader attains the Bayes read with any latent.**  If the reader outputs the
true conditional law whatever latent it gets, the multi-sample bound equals the Bayes read,
for every `K` and every sampler.  So a deterministic encoding loses nothing, and width can
only matter for a reader class that cannot output the conditional law. -/
theorem multisample_bayes_attained {Y Ω : Type*} [Fintype Y] [Fintype Ω] {K : ℕ}
    (hK : 0 < K) (P : Y → ℝ) (ρ : Ω → ℝ) (hρ1 : ∑ o, ρ o = 1) :
    ∑ y, P y * ∑ o, ρ o * Real.log ((∑ _k : Fin K, P y) / K)
      = ∑ y, P y * Real.log (P y) := by
  have hKne : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  apply Finset.sum_congr rfl
  intro y _
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [mul_div_cancel_left₀ _ hKne, ← Finset.sum_mul, hρ1, one_mul]

/-! ## A bounded reader: the parallel (product) span decode

The target is a pair of bits that are always equal, each value with probability `1/2`: the
smallest span with a joint structure.  A *product reader* predicts the two positions
independently, `q₁(y₁) q₂(y₂)`, which is what a parallel span decoder from one latent does.
It cannot represent the correlation, so it cannot hedge. -/

/-- The correlated pair: `P (b, b) = 1/2`, `P (b, ¬b) = 0`. -/
noncomputable def pairLaw (y : Bool × Bool) : ℝ := if y.1 = y.2 then 1 / 2 else 0

/-- For a positive pmf on `Bool`, `log q(true) + log q(false) ≤ log (1/4)`. -/
lemma log_add_log_le_bool (q : Bool → ℝ) (hq : ∀ b, 0 < q b) (hq1 : q true + q false = 1) :
    Real.log (q true) + Real.log (q false) ≤ Real.log (1 / 4) := by
  rw [← Real.log_mul (hq true).ne' (hq false).ne']
  apply Real.log_le_log (mul_pos (hq true) (hq false))
  nlinarith [sq_nonneg (q true - q false)]

/-- **Every product reader scores at most `-2 log 2` on the correlated pair.**  This holds for
every deterministic latent, because each latent value only picks one product reader. -/
theorem product_reader_le (q₁ q₂ : Bool → ℝ) (h₁ : ∀ b, 0 < q₁ b) (h₂ : ∀ b, 0 < q₂ b)
    (h₁1 : q₁ true + q₁ false = 1) (h₂1 : q₂ true + q₂ false = 1) :
    ∑ y : Bool × Bool, pairLaw y * Real.log (q₁ y.1 * q₂ y.2) ≤ -2 * Real.log 2 := by
  have e : ∑ y : Bool × Bool, pairLaw y * Real.log (q₁ y.1 * q₂ y.2)
      = (1 / 2) * ((Real.log (q₁ true) + Real.log (q₁ false))
          + (Real.log (q₂ true) + Real.log (q₂ false))) := by
    simp only [Fintype.sum_prod_type, Fintype.sum_bool, pairLaw]
    simp only [if_true, Bool.true_eq_false, Bool.false_eq_true, if_false, zero_mul, add_zero,
      zero_add]
    rw [Real.log_mul (h₁ true).ne' (h₂ true).ne', Real.log_mul (h₁ false).ne' (h₂ false).ne']
    ring
  have l4 : Real.log (1 / 4) = -2 * Real.log 2 := by
    rw [one_div, Real.log_inv, show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
    push_cast
    ring
  rw [e]
  have a := log_add_log_le_bool q₁ h₁ h₁1
  have b := log_add_log_le_bool q₂ h₂ h₂1
  linarith

/-- A committed bit reader: it puts `1 - ε` on its code's bit. -/
noncomputable def commit (ε : ℝ) (z b : Bool) : ℝ := if b = z then 1 - ε else ε

/-- The committed product reader of code `z` on the pair. -/
noncomputable def commitPair (ε : ℝ) (z : Bool) (y : Bool × Bool) : ℝ :=
  commit ε z y.1 * commit ε z y.2

/-- **Two enumerated codes let committed product readers beat every product reader.**  The
draw always hands the reader BOTH codes (`K = 2`, enumerated: no sampling).  The mixture
scores `log (((1-ε)² + ε²)/2)`, strictly above the product bound `-2 log 2` for every
`ε ∈ (0,1)` except `1/2`, and it tends to the Bayes read `-log 2` as `ε → 0`. -/
theorem enumerated_pair_gt (ε : ℝ) (h0 : 0 < ε) (h1 : ε < 1) (hne : ε ≠ 1 / 2) :
    -2 * Real.log 2 <
      ∑ y : Bool × Bool, pairLaw y *
        Real.log ((commitPair ε true y + commitPair ε false y) / 2) := by
  have hA : 0 < ((1 - ε) * (1 - ε) + ε * ε) / 2 := by positivity
  have e : ∑ y : Bool × Bool, pairLaw y *
      Real.log ((commitPair ε true y + commitPair ε false y) / 2)
      = Real.log (((1 - ε) * (1 - ε) + ε * ε) / 2) := by
    simp only [Fintype.sum_prod_type, Fintype.sum_bool, pairLaw, commitPair, commit]
    simp only [if_true, Bool.true_eq_false, Bool.false_eq_true, if_false, zero_mul, add_zero,
      zero_add]
    have : ε * ε + (1 - ε) * (1 - ε) = (1 - ε) * (1 - ε) + ε * ε := by ring
    rw [this]
    ring
  rw [e]
  have l4 : -2 * Real.log 2 = Real.log (1 / 4) := by
    rw [one_div, Real.log_inv, show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
    push_cast
    ring
  rw [l4]
  apply Real.log_lt_log (by norm_num)
  have : (1 - 2 * ε) ^ 2 > 0 := by
    have : 1 - 2 * ε ≠ 0 := by
      intro h
      apply hne
      linarith
    positivity
  nlinarith

/-- **Two INDEPENDENT uniform codes never beat the product bound with these readers.**  The
same committed product readers, fed two iid uniform codes (`K = 2`, four equally likely
draws), score `(1/2) log (ε(1-ε)) + (1/2) log (((1-ε)²+ε²)/2) ≤ -2 log 2` for every
`ε ∈ (0,1)`.  Half the draws are collisions, and a collision on the wrong code costs
`log ε²`. -/
theorem iid_pair_le (ε : ℝ) (h0 : 0 < ε) (h1 : ε < 1) :
    ∑ y : Bool × Bool, pairLaw y * ∑ o : Bool × Bool, (1 / 4 : ℝ) *
        Real.log ((commitPair ε o.1 y + commitPair ε o.2 y) / 2)
      ≤ -2 * Real.log 2 := by
  have hε1 : 0 < 1 - ε := by linarith
  have hA : 0 < ((1 - ε) * (1 - ε) + ε * ε) / 2 := by positivity
  have e : ∑ y : Bool × Bool, pairLaw y * ∑ o : Bool × Bool, (1 / 4 : ℝ) *
        Real.log ((commitPair ε o.1 y + commitPair ε o.2 y) / 2)
      = (1 / 4) * (Real.log ((1 - ε) * (1 - ε)) + Real.log (ε * ε)
          + 2 * Real.log (((1 - ε) * (1 - ε) + ε * ε) / 2)) := by
    simp only [Fintype.sum_prod_type, Fintype.sum_bool, pairLaw, commitPair, commit]
    simp only [if_true, Bool.true_eq_false, Bool.false_eq_true, if_false, zero_mul, add_zero,
      zero_add]
    have c1 : ((1 - ε) * (1 - ε) + (1 - ε) * (1 - ε)) / 2 = (1 - ε) * (1 - ε) := by ring
    have c2 : (ε * ε + ε * ε) / 2 = ε * ε := by ring
    have c3 : (ε * ε + (1 - ε) * (1 - ε)) / 2 = ((1 - ε) * (1 - ε) + ε * ε) / 2 := by ring
    rw [c1, c2, c3]
    ring
  rw [e]
  have l4 : -2 * Real.log 2 = Real.log (1 / 16) / 2 := by
    rw [one_div, Real.log_inv, show (16 : ℝ) = 2 ^ 4 by norm_num, Real.log_pow]
    push_cast
    ring
  rw [l4]
  have hu : 0 < ε * (1 - ε) := mul_pos h0 hε1
  -- combine the logs into one
  have comb : Real.log ((1 - ε) * (1 - ε)) + Real.log (ε * ε)
      + 2 * Real.log (((1 - ε) * (1 - ε) + ε * ε) / 2)
      = 2 * Real.log (ε * (1 - ε) * (((1 - ε) * (1 - ε) + ε * ε) / 2)) := by
    rw [Real.log_mul hu.ne' hA.ne', Real.log_mul h0.ne' hε1.ne',
      Real.log_mul hε1.ne' hε1.ne', Real.log_mul h0.ne' h0.ne']
    ring
  rw [comb]
  have key : ε * (1 - ε) * (((1 - ε) * (1 - ε) + ε * ε) / 2) ≤ 1 / 16 := by
    nlinarith [sq_nonneg (4 * (ε * (1 - ε)) - 1), sq_nonneg (1 - 2 * ε)]
  have := Real.log_le_log (mul_pos hu hA) key
  linarith

end TulExploration
