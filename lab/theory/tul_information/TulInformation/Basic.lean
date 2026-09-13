/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Finite-alphabet mutual information and the deterministic data-processing inequality

This file is the machinery. `TulInformation/SlotLoop.lean` states the corollaries that the
MORPH slot loop needs.

## Model

A joint distribution of two random variables with finite alphabets, written as a
non-negative function `p : α → β → ℝ` that sums to one. Mutual information is
`I(P) = ∑ a, ∑ b, p a b * log (p a b / (pL a * pR b))`. Lean's `Real.log 0 = 0` and
`x / 0 = 0` give the usual `0 * log 0 = 0` convention for free.

## Main results

* `sum_mul_log_div_le_zero` : Gibbs' inequality.
* `Joint.mi_nonneg` : mutual information is non-negative.
* `Joint.mi_map_le` : the deterministic data-processing inequality.
* `Joint.condEntropy_map_ge` : its log-loss form.
-/

namespace TulInformation

open Finset

/-! ## Gibbs' inequality -/

/-- Gibbs' inequality on a finite alphabet.  `p` is a probability mass function, `q` is a
sub-probability mass function that is absolutely continuous with respect to `p`.  The proof
is `log x ≤ x - 1` applied term by term. -/
theorem sum_mul_log_div_le_zero {ι : Type*} [Fintype ι] (p q : ι → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hac : ∀ i, 0 < p i → 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i ≤ 1) :
    ∑ i, p i * Real.log (q i / p i) ≤ 0 := by
  have key : ∀ i ∈ (univ : Finset ι), p i * Real.log (q i / p i) ≤ q i - p i := by
    intro i _
    rcases (hp i).lt_or_eq with h | h
    · have hqi : 0 < q i := hac i h
      have hpos : 0 < q i / p i := div_pos hqi h
      have hlog := Real.log_le_sub_one_of_pos hpos
      have : p i * Real.log (q i / p i) ≤ p i * (q i / p i - 1) :=
        mul_le_mul_of_nonneg_left hlog h.le
      calc p i * Real.log (q i / p i) ≤ p i * (q i / p i - 1) := this
        _ = q i - p i := by field_simp
    · simp [← h, hq i]
  calc ∑ i, p i * Real.log (q i / p i) ≤ ∑ i, (q i - p i) := Finset.sum_le_sum key
    _ = (∑ i, q i) - ∑ i, p i := by rw [Finset.sum_sub_distrib]
    _ ≤ 0 := by rw [hps]; linarith

/-! ## Joint distributions on a pair of finite alphabets -/

variable {α β γ δ : Type*}

/-- A joint probability mass function on `α × β`. -/
structure Joint (α β : Type*) [Fintype α] [Fintype β] where
  /-- The mass function. -/
  p : α → β → ℝ
  /-- Masses are non-negative. -/
  nonneg : ∀ a b, 0 ≤ p a b
  /-- Masses sum to one. -/
  total : ∑ a, ∑ b, p a b = 1

namespace Joint

variable [Fintype α] [Fintype β] [Fintype γ] [Fintype δ]

/-- The marginal on the first coordinate. -/
def left (P : Joint α β) (a : α) : ℝ := ∑ b, P.p a b

/-- The marginal on the second coordinate. -/
def right (P : Joint α β) (b : β) : ℝ := ∑ a, P.p a b

lemma left_nonneg (P : Joint α β) (a : α) : 0 ≤ P.left a :=
  Finset.sum_nonneg fun b _ => P.nonneg a b

lemma right_nonneg (P : Joint α β) (b : β) : 0 ≤ P.right b :=
  Finset.sum_nonneg fun a _ => P.nonneg a b

lemma le_left (P : Joint α β) (a : α) (b : β) : P.p a b ≤ P.left a :=
  Finset.single_le_sum (f := fun b => P.p a b) (fun b _ => P.nonneg a b) (mem_univ b)

lemma sum_left (P : Joint α β) : ∑ a, P.left a = 1 := P.total

/-- Mutual information of a joint distribution, in nats. -/
noncomputable def mi (P : Joint α β) : ℝ :=
  ∑ a, ∑ b, P.p a b * Real.log (P.p a b / (P.left a * P.right b))

/-! ## Pushing a joint distribution through a deterministic map -/

/-- Summing a fibrewise-constant weight:
`∑ c, (mass of the fibre of c) * g c = ∑ a, h a * g (f a)`. -/
lemma sum_fiber_mul [DecidableEq γ] (f : α → γ) (h : α → ℝ) (g : γ → ℝ) :
    ∑ c, (∑ a ∈ univ.filter fun a => f a = c, h a) * g c = ∑ a, h a * g (f a) := by
  have step : ∀ c : γ, (∑ a ∈ univ.filter fun a => f a = c, h a) * g c
      = ∑ a ∈ univ.filter fun a => f a = c, h a * g (f a) := by
    intro c
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl ?_
    intro a ha
    rw [(Finset.mem_filter.mp ha).2]
  rw [Finset.sum_congr rfl fun c _ => step c]
  exact Finset.sum_fiberwise _ _ _

/-- The image of a joint distribution under a deterministic map of the first coordinate.
This is the only kind of channel this development needs: `Z ↦ f Z` with `Y` untouched. -/
noncomputable def map [DecidableEq γ] (P : Joint α β) (f : α → γ) : Joint γ β where
  p := fun c b => ∑ a ∈ univ.filter fun a => f a = c, P.p a b
  nonneg := by
    intro c b
    exact Finset.sum_nonneg fun a _ => P.nonneg a b
  total := by
    have : ∀ c : γ, ∑ b, (∑ a ∈ univ.filter fun a => f a = c, P.p a b)
        = ∑ a ∈ univ.filter fun a => f a = c, ∑ b, P.p a b := by
      intro c; exact Finset.sum_comm
    rw [Finset.sum_congr rfl fun c _ => this c]
    have := sum_fiber_mul (α := α) (γ := γ) f (fun a => ∑ b, P.p a b) (fun _ => 1)
    simpa using (by simpa using this : ∑ c, (∑ a ∈ univ.filter fun a => f a = c, ∑ b, P.p a b)
        = ∑ a, ∑ b, P.p a b).trans P.total

variable [DecidableEq γ] [DecidableEq δ]

lemma map_p (P : Joint α β) (f : α → γ) (c : γ) (b : β) :
    (P.map f).p c b = ∑ a ∈ univ.filter fun a => f a = c, P.p a b := rfl

/-- A mass of the source is at most the mass of its image point. -/
lemma le_map_p (P : Joint α β) (f : α → γ) (a : α) (b : β) :
    P.p a b ≤ (P.map f).p (f a) b := by
  rw [map_p]
  exact Finset.single_le_sum (f := fun a => P.p a b) (fun a _ => P.nonneg a b)
    (Finset.mem_filter.mpr ⟨mem_univ a, rfl⟩)

/-- The first marginal of the image is the fibre mass of the first marginal. -/
lemma map_left (P : Joint α β) (f : α → γ) (c : γ) :
    (P.map f).left c = ∑ a ∈ univ.filter fun a => f a = c, P.left a := by
  simp only [left, map_p]
  exact Finset.sum_comm

lemma le_map_left (P : Joint α β) (f : α → γ) (a : α) :
    P.left a ≤ (P.map f).left (f a) := by
  rw [map_left]
  exact Finset.single_le_sum (f := fun a => P.left a) (fun a _ => P.left_nonneg a)
    (Finset.mem_filter.mpr ⟨mem_univ a, rfl⟩)

/-- A deterministic map of the first coordinate leaves the second marginal alone. -/
lemma map_right (P : Joint α β) (f : α → γ) (b : β) : (P.map f).right b = P.right b := by
  simp only [right, map_p]
  have := sum_fiber_mul (α := α) (γ := γ) f (fun a => P.p a b) (fun _ => 1)
  simpa using this


/-! ## The deterministic data-processing inequality -/


/-- The surrogate mass function used in the proof of the data-processing inequality:
`r a b = pL a * Q (f a) b / QL (f a)`, where `Q = P.map f`.  It is the joint one would get
by re-splitting the image mass `Q (f a) b` across the fibre of `f a` in proportion to the
source marginal. -/
noncomputable def surrogate (P : Joint α β) (f : α → γ) (a : α) (b : β) : ℝ :=
  P.left a * (P.map f).p (f a) b / (P.map f).left (f a)

lemma surrogate_nonneg (P : Joint α β) (f : α → γ) (a : α) (b : β) :
    0 ≤ P.surrogate f a b := by
  refine div_nonneg (mul_nonneg (P.left_nonneg a) ?_) ((P.map f).left_nonneg (f a))
  exact (P.map f).nonneg (f a) b

lemma surrogate_pos (P : Joint α β) (f : α → γ) {a : α} {b : β} (h : 0 < P.p a b) :
    0 < P.surrogate f a b := by
  have hL : 0 < P.left a := lt_of_lt_of_le h (P.le_left a b)
  have hQ : 0 < (P.map f).p (f a) b := lt_of_lt_of_le h (P.le_map_p f a b)
  have hQL : 0 < (P.map f).left (f a) := lt_of_lt_of_le hL (P.le_map_left f a)
  exact div_pos (mul_pos hL hQ) hQL

lemma sum_surrogate_le (P : Joint α β) (f : α → γ) (a : α) :
    ∑ b, P.surrogate f a b ≤ P.left a := by
  have hfac : ∀ b : β, P.surrogate f a b
      = (P.left a / (P.map f).left (f a)) * (P.map f).p (f a) b := by
    intro b; rw [surrogate, mul_div_right_comm]
  have hsum : ∑ b, P.surrogate f a b
      = P.left a * (P.map f).left (f a) / (P.map f).left (f a) := by
    rw [Finset.sum_congr rfl fun b _ => hfac b, ← Finset.mul_sum]
    rw [show ∑ b, (P.map f).p (f a) b = (P.map f).left (f a) from rfl]
    ring
  rcases eq_or_lt_of_le ((P.map f).left_nonneg (f a)) with h | h
  · rw [hsum, ← h]; simpa using P.left_nonneg a
  · rw [hsum, mul_div_assoc, div_self (ne_of_gt h), mul_one]

lemma sum_surrogate_le_one (P : Joint α β) (f : α → γ) :
    ∑ a, ∑ b, P.surrogate f a b ≤ 1 := by
  calc ∑ a, ∑ b, P.surrogate f a b ≤ ∑ a, P.left a :=
        Finset.sum_le_sum fun a _ => P.sum_surrogate_le f a
    _ = 1 := P.sum_left

/-- The pointwise identity behind the data-processing inequality: the difference of the two
mutual-information summands is a log-likelihood-ratio term against the surrogate. -/
lemma mi_term_diff (P : Joint α β) (f : α → γ) (a : α) (b : β) :
    P.p a b * Real.log ((P.map f).p (f a) b / ((P.map f).left (f a) * P.right b))
      - P.p a b * Real.log (P.p a b / (P.left a * P.right b))
      = P.p a b * Real.log (P.surrogate f a b / P.p a b) := by
  rcases (P.nonneg a b).lt_or_eq with h | h
  · have hL : 0 < P.left a := lt_of_lt_of_le h (P.le_left a b)
    have hR : 0 < P.right b :=
      lt_of_lt_of_le h (Finset.single_le_sum (f := fun a => P.p a b)
        (fun a _ => P.nonneg a b) (Finset.mem_univ a))
    have hQ : 0 < (P.map f).p (f a) b := lt_of_lt_of_le h (P.le_map_p f a b)
    have hQL : 0 < (P.map f).left (f a) := lt_of_lt_of_le hL (P.le_map_left f a)
    have hA : (0:ℝ) < (P.map f).p (f a) b / ((P.map f).left (f a) * P.right b) :=
      div_pos hQ (mul_pos hQL hR)
    have hB : (0:ℝ) < P.p a b / (P.left a * P.right b) := div_pos h (mul_pos hL hR)
    have hratio : ((P.map f).p (f a) b / ((P.map f).left (f a) * P.right b))
        / (P.p a b / (P.left a * P.right b)) = P.surrogate f a b / P.p a b := by
      rw [surrogate]
      have h1 : P.p a b ≠ 0 := ne_of_gt h
      have h2 : P.left a ≠ 0 := ne_of_gt hL
      have h3 : P.right b ≠ 0 := ne_of_gt hR
      have h4 : (P.map f).left (f a) ≠ 0 := ne_of_gt hQL
      field_simp
    rw [← mul_sub, ← Real.log_div (ne_of_gt hA) (ne_of_gt hB), hratio]
  · simp [← h]

/-- **The deterministic data-processing inequality.**  Sending the first coordinate through a
deterministic map cannot increase the mutual information with the second coordinate.
In MORPH's terms: no pass of the slot loop can add information about any target. -/
theorem mi_map_le (P : Joint α β) (f : α → γ) : (P.map f).mi ≤ P.mi := by
  -- Rewrite the image mutual information as a sum over the source alphabet.
  have himg : (P.map f).mi
      = ∑ a, ∑ b, P.p a b
          * Real.log ((P.map f).p (f a) b / ((P.map f).left (f a) * P.right b)) := by
    rw [Joint.mi, Finset.sum_comm]
    rw [show (∑ b, ∑ c, (P.map f).p c b
          * Real.log ((P.map f).p c b / ((P.map f).left c * (P.map f).right b)))
        = ∑ b, ∑ a, P.p a b
          * Real.log ((P.map f).p (f a) b / ((P.map f).left (f a) * P.right b)) from ?_]
    · exact Finset.sum_comm
    · refine Finset.sum_congr rfl fun b _ => ?_
      have := sum_fiber_mul (α := α) (γ := γ) f (fun a => P.p a b)
        (fun c => Real.log ((P.map f).p c b / ((P.map f).left c * P.right b)))
      simpa [Joint.map_p, Joint.map_right] using this
  -- The difference is a Gibbs term against the surrogate.
  have hdiff : (P.map f).mi - P.mi
      = ∑ a, ∑ b, P.p a b * Real.log (P.surrogate f a b / P.p a b) := by
    rw [himg, Joint.mi, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun b _ => P.mi_term_diff f a b
  -- Gibbs, on the product alphabet.
  have hgibbs : ∑ a, ∑ b, P.p a b * Real.log (P.surrogate f a b / P.p a b) ≤ 0 := by
    have := sum_mul_log_div_le_zero (ι := α × β)
      (fun ab => P.p ab.1 ab.2) (fun ab => P.surrogate f ab.1 ab.2)
      (fun ab => P.nonneg ab.1 ab.2) (fun ab => P.surrogate_nonneg f ab.1 ab.2)
      (fun ab h => P.surrogate_pos f h)
      (by rw [Fintype.sum_prod_type]; exact P.total)
      (by rw [Fintype.sum_prod_type]; exact P.sum_surrogate_le_one f)
    rwa [Fintype.sum_prod_type] at this
  linarith [hdiff ▸ hgibbs]


/-! ## Functoriality, non-negativity, and conditional entropy -/

@[ext]
lemma ext {P Q : Joint α β} (h : ∀ a b, P.p a b = Q.p a b) : P = Q := by
  cases P; cases Q
  congr
  funext a b
  exact h a b

lemma map_id (P : Joint α β) [DecidableEq α] : P.map (id : α → α) = P := by
  ext c b
  rw [map_p]
  simp [Finset.filter_eq']

lemma map_comp (P : Joint α β) (f : α → γ) (g : γ → δ) :
    (P.map f).map g = P.map (g ∘ f) := by
  ext d b
  rw [map_p, map_p]
  have hmaps : ∀ a ∈ (univ.filter fun a => (g ∘ f) a = d),
      f a ∈ (univ.filter fun c => g c = d) := by
    intro a ha
    simpa using (Finset.mem_filter.mp ha).2
  have hfib : ∀ c ∈ (univ.filter fun c => g c = d),
      (univ.filter fun a => f a = c)
        = ((univ.filter fun a => (g ∘ f) a = d).filter fun a => f a = c) := by
    intro c hc
    have hgc : g c = d := (Finset.mem_filter.mp hc).2
    ext a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Function.comp_apply,
      Finset.filter_filter]
    constructor
    · intro hfa; exact ⟨by rw [hfa]; exact hgc, hfa⟩
    · intro h; exact h.2
  calc ∑ c ∈ univ.filter fun c => g c = d, (P.map f).p c b
      = ∑ c ∈ univ.filter fun c => g c = d,
          ∑ a ∈ (univ.filter fun a => (g ∘ f) a = d).filter fun a => f a = c, P.p a b := by
        refine Finset.sum_congr rfl fun c hc => ?_
        rw [map_p, hfib c hc]
    _ = ∑ a ∈ univ.filter fun a => (g ∘ f) a = d, P.p a b :=
        Finset.sum_fiberwise_of_maps_to hmaps _

/-- Mutual information is non-negative. -/
theorem mi_nonneg (P : Joint α β) : 0 ≤ P.mi := by
  have hterm : ∀ a : α, ∀ b : β,
      P.p a b * Real.log (P.p a b / (P.left a * P.right b))
        = -(P.p a b * Real.log ((P.left a * P.right b) / P.p a b)) := by
    intro a b
    rcases (P.nonneg a b).lt_or_eq with h | h
    · have hL : 0 < P.left a := lt_of_lt_of_le h (P.le_left a b)
      have hR : 0 < P.right b :=
        lt_of_lt_of_le h (Finset.single_le_sum (f := fun a => P.p a b)
          (fun a _ => P.nonneg a b) (Finset.mem_univ a))
      have hlog : Real.log (P.p a b / (P.left a * P.right b))
          = -Real.log ((P.left a * P.right b) / P.p a b) := by
        rw [← Real.log_inv, inv_div]
      rw [hlog, mul_neg]
    · simp [← h]
  have hgibbs : ∑ a, ∑ b, P.p a b * Real.log ((P.left a * P.right b) / P.p a b) ≤ 0 := by
    have := sum_mul_log_div_le_zero (ι := α × β)
      (fun ab => P.p ab.1 ab.2) (fun ab => P.left ab.1 * P.right ab.2)
      (fun ab => P.nonneg ab.1 ab.2)
      (fun ab => mul_nonneg (P.left_nonneg ab.1) (P.right_nonneg ab.2))
      (fun ab h => mul_pos (lt_of_lt_of_le h (P.le_left ab.1 ab.2))
        (lt_of_lt_of_le h (Finset.single_le_sum (f := fun a => P.p a ab.2)
          (fun a _ => P.nonneg a ab.2) (Finset.mem_univ ab.1))))
      (by rw [Fintype.sum_prod_type]; exact P.total)
      (by
        rw [Fintype.sum_prod_type]
        have : ∀ a : α, ∑ b, P.left a * P.right b = P.left a := by
          intro a
          rw [← Finset.mul_sum]
          have hr : ∑ b, P.right b = 1 := by
            simp only [right]
            rw [Finset.sum_comm]
            exact P.total
          rw [hr, mul_one]
        rw [Finset.sum_congr rfl fun a _ => this a, P.sum_left])
    rwa [Fintype.sum_prod_type] at this
  have : P.mi = -(∑ a, ∑ b, P.p a b * Real.log ((P.left a * P.right b) / P.p a b)) := by
    rw [mi]
    rw [Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => hterm a b]
    simp [Finset.sum_neg_distrib]
  rw [this]
  linarith

/-- Entropy of the second marginal, in nats. -/
noncomputable def entropyRight (P : Joint α β) : ℝ :=
  -∑ b, P.right b * Real.log (P.right b)

/-- Conditional entropy of the second coordinate given the first, DEFINED as `H(Y) - I(X;Y)`.
Its identification with the Bayes log-loss of predicting `Y` from `X` is standard and is not
proved here. -/
noncomputable def condEntropy (P : Joint α β) : ℝ := P.entropyRight - P.mi

/-- Processing the first coordinate cannot lower the conditional entropy of the second.
This is the log-loss reading of `mi_map_le`. -/
theorem condEntropy_map_ge (P : Joint α β) (f : α → γ) :
    P.condEntropy ≤ (P.map f).condEntropy := by
  have hH : (P.map f).entropyRight = P.entropyRight := by
    simp only [entropyRight]
    exact congrArg Neg.neg (Finset.sum_congr rfl fun b _ => by rw [P.map_right f b])
  simp only [condEntropy, hH]
  linarith [P.mi_map_le f]

end Joint

end TulInformation
