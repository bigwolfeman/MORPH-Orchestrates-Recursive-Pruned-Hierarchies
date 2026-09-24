/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Analysis.Normed.Operator.Basic
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Abel
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-!
# T6: what sets the loop's per-pass map (the cause behind the measured gain)

The measured effect: the plain loop's per-pass typical gain sits near 1, varies by position
and has a tail above 1; the slot loop's sits at a uniform 0.87 to 0.88 with nothing above 1.
This file proves four statements that together say what the gain is a reading OF, and what
in the loss moves it.

## Model

A pass is `h ↦ D h + u h + c`: `D` is the injection's diagonal decay (identity on the
channels it does not touch, `A` on the context slice; `DiagonalInjection`), `u` is the
blocks' state-dependent part, `c` the per-pass write (the re-injected source and every
state-independent term).  The scalar and linear models below keep one direction of that map.

## Main results

* **`gain_sandwich`** : the finite-difference gain of the pass along `d` is within `L‖d‖`
  of `‖D d‖`, `L` the blocks' Lipschitz constant.  A silent core reads the injection.
* `silent_gain_le` : if `D` is non-expansive (it is: `A ≤ 0.9999`), no direction reads above
  `1 + L`.  `diag_sq_le`, `injection_frob` : the concrete diagonal and its Frobenius sum
  `(n - |S|) + ∑_{i∈S} Aᵢ²`, which is the "floor" typical gain squared times `n`.
* **`depth_invariant_iff`** : a re-injected linear direction `h ↦ λ h + b` from entry `x₀`
  gives the same exit at two depths `1 ≤ T₁ < T₂` iff `λ = 0` (a one-step map) or the entry
  is already the fixed point.  `random_depth_zero_loss` : so a depth-independent target read
  at a random depth is fit exactly only by those two maps.  **`fixed_entry_flat`** : along
  the second family the loss does not depend on `λ` at all (and `fp_zero_fixed_entry` : the
  fixed-point term is zero there): the loss cannot see the gain.
* **`evLoss_eq`**, **`newEv_zero_iff`**, **`newEv_strict`**, `newEv_monotone_above` : when
  every pass brings NEW evidence (orthonormal per-pass inputs, the reader wants their sum),
  the loss is `∑_{s<T} (b λˢ - 1)²`; its only zero is `λ = 1, b = 1`, and at `b = 1` it
  strictly decreases toward `λ = 1` from below and increases above it.  The loss asks for
  gain exactly 1 on the directions that carry per-pass information.
* `relay_zero_iff` : when the reader wants only the newest evidence (a relay), the only exact
  fit is `λ = 0`.  Multi-pass work asks for gain 1 only where earlier passes' work is KEPT.
* `roll_budget` : a non-expansive direction with per-pass write `|b| ≤ β` cannot move further
  than `|x₀| + βT`: a small entry and a far target force expansion.
* **`hasDerivAt_blockRoll`**, **`firstVar_split`**, **`settled_block_is_write`** : the
  derivative of the exit state with respect to a perturbation `ε G` of the pass's linear part
  is `∑ J^{T-1-t} G a_t`.  Split at the fixed point `h*`, it is the derivative of a WRITE of
  `G h*` plus a recurrence-only term driven by `J^t (h₀ - h*)`, whose norm is bounded by
  `‖h₀ - h*‖` times a constant.  On a settled trajectory (`h₀ = h*`) the recurrence-only term
  is exactly zero: to first order the loss cannot tell a change of the recurrence from a
  change of the per-pass write.
-/

open Finset

namespace TulExploration

/-! ## 1. The silent-core floor -/

section Floor

variable {E : Type*} [SeminormedAddCommGroup E]

/-- **The gain of a pass is the injection's gain up to the blocks' Lipschitz constant.**
For `F h = D h + u h + c` with `D` additive and `u` `L`-Lipschitz, the finite difference along
`d` is within `L ‖d‖` of `‖D d‖`. -/
theorem gain_sandwich (D : E →+ E) (u : E → E) (c : E) (L : ℝ)
    (hu : ∀ x y, ‖u x - u y‖ ≤ L * ‖x - y‖) (h d : E) :
    ‖D d‖ - L * ‖d‖ ≤ ‖(D (h + d) + u (h + d) + c) - (D h + u h + c)‖ ∧
      ‖(D (h + d) + u (h + d) + c) - (D h + u h + c)‖ ≤ ‖D d‖ + L * ‖d‖ := by
  have e : (D (h + d) + u (h + d) + c) - (D h + u h + c) = D d + (u (h + d) - u h) := by
    rw [map_add]; abel
  have hud : ‖u (h + d) - u h‖ ≤ L * ‖d‖ := by simpa using hu (h + d) h
  rw [e]
  constructor
  · have := norm_sub_norm_le (D d) (-(u (h + d) - u h))
    rw [norm_neg, sub_neg_eq_add] at this
    linarith
  · calc ‖D d + (u (h + d) - u h)‖ ≤ ‖D d‖ + ‖u (h + d) - u h‖ := norm_add_le _ _
      _ ≤ ‖D d‖ + L * ‖d‖ := by linarith

/-- **A silent core never reads above `1 + L`.**  If the injection is non-expansive, every
direction's finite-difference gain is at most `1 + L`. -/
theorem silent_gain_le (D : E →+ E) (u : E → E) (c : E) (L : ℝ)
    (hu : ∀ x y, ‖u x - u y‖ ≤ L * ‖x - y‖) (hD : ∀ d, ‖D d‖ ≤ ‖d‖) (h d : E) :
    ‖(D (h + d) + u (h + d) + c) - (D h + u h + c)‖ ≤ (1 + L) * ‖d‖ := by
  have := (gain_sandwich D u c L hu h d).2
  have := hD d
  nlinarith

end Floor

/-- The injection's diagonal is non-expansive coordinatewise: `∑ (aᵢ dᵢ)² ≤ ∑ dᵢ²` when every
`|aᵢ| ≤ 1`. -/
theorem diag_sq_le {n : ℕ} (a d : Fin n → ℝ) (ha : ∀ i, |a i| ≤ 1) :
    ∑ i, (a i * d i) ^ 2 ≤ ∑ i, d i ^ 2 := by
  apply Finset.sum_le_sum
  intro i _
  have h1 : (a i) ^ 2 ≤ 1 := by
    have := ha i
    nlinarith [abs_nonneg (a i), sq_abs (a i)]
  rw [mul_pow]
  nlinarith [sq_nonneg (d i)]

/-- **The floor.**  The injection's diagonal is `Aᵢ` on the context slice `S` and `1`
elsewhere; its Frobenius sum is `(n - |S|) + ∑_{i∈S} Aᵢ²`.  Divided by `n` and square-rooted it
is the typical gain of a pass whose blocks are silent.  MORPH: `n = 1024`, `|S| = 320`,
`Aᵢ ≈ 0.44` gives `0.865`. -/
theorem injection_frob {n : ℕ} (S : Finset (Fin n)) (A : Fin n → ℝ) :
    ∑ i, (if i ∈ S then A i else 1) ^ 2 = ((n : ℝ) - S.card) + ∑ i ∈ S, A i ^ 2 := by
  have h1 : ∀ i, (if i ∈ S then A i else 1) ^ 2 = 1 + (if i ∈ S then A i ^ 2 - 1 else 0) := by
    intro i; split_ifs <;> ring
  simp_rw [h1, Finset.sum_add_distrib, Finset.sum_ite_mem, Finset.univ_inter,
    Finset.sum_sub_distrib]
  simp
  ring

/-! ## 2. A re-injected source read at a random depth: one step, or a flat gain -/

/-- One direction of a re-injected linear loop: entry `x₀`, pass `h ↦ λ h + b`. -/
def roll (lam b x0 : ℝ) : ℕ → ℝ
  | 0 => x0
  | T + 1 => lam * roll lam b x0 T + b

theorem roll_succ_sub (lam b x0 : ℝ) :
    ∀ T, roll lam b x0 (T + 1) - roll lam b x0 T = lam ^ T * ((lam - 1) * x0 + b)
  | 0 => by simp [roll]; ring
  | T + 1 => by
    have ih := roll_succ_sub lam b x0 T
    have e : roll lam b x0 (T + 1 + 1) - roll lam b x0 (T + 1)
        = lam * (roll lam b x0 (T + 1) - roll lam b x0 T) := by
      simp only [roll]; ring
    rw [e, ih, pow_succ]; ring

theorem roll_add_sub (lam b x0 : ℝ) (T : ℕ) :
    ∀ k, roll lam b x0 (T + k) - roll lam b x0 T
      = ((lam - 1) * x0 + b) * (lam ^ T * ∑ t ∈ range k, lam ^ t)
  | 0 => by simp
  | k + 1 => by
    have ih := roll_add_sub lam b x0 T k
    have hs := roll_succ_sub lam b x0 (T + k)
    have e : roll lam b x0 (T + (k + 1)) - roll lam b x0 T
        = (roll lam b x0 (T + k + 1) - roll lam b x0 (T + k))
          + (roll lam b x0 (T + k) - roll lam b x0 T) := by
      rw [← Nat.add_assoc]; ring
    rw [e, hs, ih, Finset.sum_range_succ, pow_add]; ring

/-- The entry at the fixed point stays there at every depth, whatever the gain. -/
theorem roll_fixed_entry {lam b x0 : ℝ} (h : (lam - 1) * x0 + b = 0) :
    ∀ T, roll lam b x0 T = x0
  | 0 => rfl
  | T + 1 => by
    have hs := roll_succ_sub lam b x0 T
    rw [h, mul_zero, roll_fixed_entry h T] at hs
    linarith

/-- A one-step map (`λ = 0`) gives the same state after every pass. -/
theorem roll_zero_gain (b x0 : ℝ) : ∀ T, 1 ≤ T → roll 0 b x0 T = b
  | 0, h => absurd h (by norm_num)
  | T + 1, _ => by simp [roll]

theorem sum_geom_pos {lam : ℝ} (hlam : 0 ≤ lam) {k : ℕ} (hk : 1 ≤ k) :
    0 < ∑ t ∈ range k, lam ^ t := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Finset.sum_range_succ']
  have : 0 ≤ ∑ t ∈ range j, lam ^ (t + 1) :=
    Finset.sum_nonneg (fun t _ => pow_nonneg hlam _)
  simp only [pow_zero]
  linarith

/-- **Depth invariance forces one step or a settled entry.**  For a non-negative gain and two
depths `1 ≤ T₁ < T₂`, the exits agree iff the map is one-step (`λ = 0`) or the entry is the
fixed point. -/
theorem depth_invariant_iff {lam b x0 : ℝ} (hlam : 0 ≤ lam) {T1 T2 : ℕ} (hT1 : 1 ≤ T1)
    (hT : T1 < T2) :
    roll lam b x0 T1 = roll lam b x0 T2 ↔ lam = 0 ∨ (lam - 1) * x0 + b = 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, T2 = T1 + k := ⟨T2 - T1, by omega⟩
  have hk : 1 ≤ k := by omega
  have hd := roll_add_sub lam b x0 T1 k
  have hS := sum_geom_pos hlam hk
  constructor
  · intro h
    have h0 : ((lam - 1) * x0 + b) * (lam ^ T1 * ∑ t ∈ range k, lam ^ t) = 0 := by
      rw [← hd, h, sub_self]
    rcases mul_eq_zero.1 h0 with h1 | h2
    · exact Or.inr h1
    · rcases mul_eq_zero.1 h2 with h3 | h4
      · exact Or.inl ((pow_eq_zero_iff (by omega)).1 h3)
      · exact absurd h4 hS.ne'
  · rintro (h | h)
    · subst h
      rw [roll_zero_gain b x0 T1 hT1, roll_zero_gain b x0 (T1 + k) (by omega)]
    · rw [roll_fixed_entry h, roll_fixed_entry h]

/-- **A random depth with two depths in its support is fit exactly only by those two maps.**
The loss is `∑_T P(T) (r · h_T - m)²` over a finite support; if it is zero, `r ≠ 0`, and two
depths `1 ≤ T₁ < T₂` carry positive weight, then `λ = 0` or the entry is the fixed point. -/
theorem random_depth_zero_loss {lam b x0 : ℝ} (hlam : 0 ≤ lam) (P : ℕ → ℝ) (Ts : Finset ℕ)
    (hP : ∀ T ∈ Ts, 0 ≤ P T) (r m : ℝ) (hr : r ≠ 0) {T1 T2 : ℕ} (h1 : T1 ∈ Ts) (h2 : T2 ∈ Ts)
    (hp1 : 0 < P T1) (hp2 : 0 < P T2) (hT1 : 1 ≤ T1) (hT : T1 < T2)
    (hzero : ∑ T ∈ Ts, P T * (r * roll lam b x0 T - m) ^ 2 = 0) :
    lam = 0 ∨ (lam - 1) * x0 + b = 0 := by
  have hnn : ∀ T ∈ Ts, 0 ≤ P T * (r * roll lam b x0 T - m) ^ 2 :=
    fun T hT => mul_nonneg (hP T hT) (sq_nonneg _)
  have hall := (Finset.sum_eq_zero_iff_of_nonneg hnn).1 hzero
  have e1 : r * roll lam b x0 T1 - m = 0 := by
    have := hall T1 h1
    rcases mul_eq_zero.1 this with h | h
    · exact absurd h hp1.ne'
    · exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h
  have e2 : r * roll lam b x0 T2 - m = 0 := by
    have := hall T2 h2
    rcases mul_eq_zero.1 this with h | h
    · exact absurd h hp2.ne'
    · exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h
  have : roll lam b x0 T1 = roll lam b x0 T2 := by
    apply mul_left_cancel₀ hr
    linarith
  exact (depth_invariant_iff hlam hT1 hT).1 this

/-- **Along the settled family the loss cannot see the gain.**  With the write `b = (1 - λ) x₀`
the entry is the fixed point, and the random-depth loss is the same for every `λ`. -/
theorem fixed_entry_flat (P : ℕ → ℝ) (Ts : Finset ℕ) (r m x0 lam lam' : ℝ) :
    ∑ T ∈ Ts, P T * (r * roll lam ((1 - lam) * x0) x0 T - m) ^ 2
      = ∑ T ∈ Ts, P T * (r * roll lam' ((1 - lam') * x0) x0 T - m) ^ 2 := by
  have h : ∀ l : ℝ, ∀ T, roll l ((1 - l) * x0) x0 T = x0 :=
    fun l => roll_fixed_entry (by ring)
  simp only [h]

/-- The fixed-point term `h_{T+1} - h_T` is zero along the settled family, at every gain. -/
theorem fp_zero_fixed_entry (x0 lam : ℝ) (T : ℕ) :
    roll lam ((1 - lam) * x0) x0 (T + 1) - roll lam ((1 - lam) * x0) x0 T = 0 := by
  rw [roll_fixed_entry (by ring), roll_fixed_entry (by ring), sub_self]

/-- **A bounded write cannot travel far without expansion.**  With `0 ≤ λ ≤ 1` and
`|b| ≤ β`, `|h_T| ≤ |x₀| + β T`. -/
theorem roll_budget {lam b x0 β : ℝ} (h0 : 0 ≤ lam) (h1 : lam ≤ 1) (hb : |b| ≤ β) :
    ∀ T : ℕ, |roll lam b x0 T| ≤ |x0| + β * T
  | 0 => by simp [roll]
  | T + 1 => by
    have ih := roll_budget (x0 := x0) h0 h1 hb T
    simp only [roll]
    push_cast
    have ha : |lam * roll lam b x0 T| ≤ |roll lam b x0 T| := by
      rw [abs_mul, abs_of_nonneg h0]
      exact mul_le_of_le_one_left (abs_nonneg _) h1
    calc |lam * roll lam b x0 T + b| ≤ |lam * roll lam b x0 T| + |b| := abs_add_le _ _
      _ ≤ |x0| + β * T + β := by linarith
      _ = |x0| + β * (T + 1) := by ring

/-- A target further than `|x₀| + βT` is out of reach of every non-expansive direction. -/
theorem budget_needs_expansion {lam b x0 β m : ℝ} (h0 : 0 ≤ lam) (h1 : lam ≤ 1)
    (hb : |b| ≤ β) (T : ℕ) (hm : |x0| + β * T < |m|) : roll lam b x0 T ≠ m := by
  intro h
  have := roll_budget (x0 := x0) h0 h1 hb T
  rw [h] at this
  linarith

/-! ## 3. New evidence every pass: the loss asks for gain exactly 1 -/

/-- Evidence `x_t` (coordinate `t`) arrives at pass `t`; the state is `h_{t+1} = λ h_t + b x_t`
from `h_0 = 0`.  Coordinate `k` of the state after `T` passes. -/
def evState (lam b : ℝ) : ℕ → ℕ → ℝ
  | 0 => fun _ => 0
  | T + 1 => fun k => lam * evState lam b T k + b * (if k = T then 1 else 0)

theorem evState_eq (lam b : ℝ) :
    ∀ T k, evState lam b T k = if k < T then b * lam ^ (T - 1 - k) else 0
  | 0, k => by simp [evState]
  | T + 1, k => by
    simp only [evState, evState_eq lam b T k]
    rcases lt_trichotomy k T with hk | hk | hk
    · have e : T + 1 - 1 - k = (T - 1 - k) + 1 := by omega
      rw [if_pos hk, if_neg (by omega), if_pos (by omega), e, pow_succ]; ring
    · subst hk; simp
    · rw [if_neg (by omega), if_neg (by omega), if_neg (by omega)]; ring

/-- The squared error, on the arrived coordinates, of reading the state as the sum of the
evidence.  (The coordinates not yet arrived add a constant that no map can change.) -/
def evLoss (lam b : ℝ) (T : ℕ) : ℝ := ∑ k ∈ range T, (evState lam b T k - 1) ^ 2

/-- The loss of the per-pass-evidence model in closed form. -/
def newEvLoss (lam b : ℝ) (T : ℕ) : ℝ := ∑ s ∈ range T, (b * lam ^ s - 1) ^ 2

theorem evLoss_eq (lam b : ℝ) (T : ℕ) : evLoss lam b T = newEvLoss lam b T := by
  unfold evLoss newEvLoss
  have h : ∀ k ∈ range T, (evState lam b T k - 1) ^ 2 = (b * lam ^ (T - 1 - k) - 1) ^ 2 := by
    intro k hk
    rw [evState_eq, if_pos (Finset.mem_range.1 hk)]
  rw [Finset.sum_congr rfl h]
  exact Finset.sum_range_reflect (fun s => (b * lam ^ s - 1) ^ 2) T

/-- **The only exact fit keeps every pass's evidence: gain 1, write 1.** -/
theorem newEv_zero_iff {lam b : ℝ} {T : ℕ} (hT : 2 ≤ T) :
    newEvLoss lam b T = 0 ↔ b = 1 ∧ lam = 1 := by
  constructor
  · intro h
    have hnn : ∀ s ∈ range T, 0 ≤ (b * lam ^ s - 1) ^ 2 := fun _ _ => sq_nonneg _
    have hall := (Finset.sum_eq_zero_iff_of_nonneg hnn).1 h
    have h0 := hall 0 (Finset.mem_range.2 (by omega))
    have h1 := hall 1 (Finset.mem_range.2 (by omega))
    have hb : b = 1 := by
      have := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h0
      simp at this; linarith
    refine ⟨hb, ?_⟩
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h1
    rw [hb] at this; simp at this; linarith
  · rintro ⟨rfl, rfl⟩
    simp [newEvLoss]

/-- **Below gain 1, more gain is better, pass by pass.** -/
theorem newEv_antitone {lam mu : ℝ} (h0 : 0 ≤ lam) (h : lam ≤ mu) (h1 : mu ≤ 1) (T : ℕ) :
    newEvLoss mu 1 T ≤ newEvLoss lam 1 T := by
  apply Finset.sum_le_sum
  intro s _
  have a1 : lam ^ s ≤ mu ^ s := pow_le_pow_left₀ h0 h s
  have a2 : mu ^ s ≤ 1 := pow_le_one₀ (h0.trans h) h1
  have a3 : 0 ≤ lam ^ s := pow_nonneg h0 s
  simp only [one_mul]
  nlinarith

/-- **Strictly so once there are two passes.** -/
theorem newEv_strict {lam mu : ℝ} {T : ℕ} (hT : 2 ≤ T) (h0 : 0 ≤ lam) (h : lam < mu)
    (h1 : mu ≤ 1) : newEvLoss mu 1 T < newEvLoss lam 1 T := by
  apply Finset.sum_lt_sum
  · intro s _
    have a1 : lam ^ s ≤ mu ^ s := pow_le_pow_left₀ h0 h.le s
    have a2 : mu ^ s ≤ 1 := pow_le_one₀ (h0.trans h.le) h1
    have a3 : 0 ≤ lam ^ s := pow_nonneg h0 s
    simp only [one_mul]
    nlinarith
  · refine ⟨1, Finset.mem_range.2 (by omega), ?_⟩
    simp only [one_mul, pow_one]
    nlinarith

/-- **Above gain 1, more gain is worse.** -/
theorem newEv_monotone_above {lam mu : ℝ} (h1 : 1 ≤ lam) (h : lam ≤ mu) (T : ℕ) :
    newEvLoss lam 1 T ≤ newEvLoss mu 1 T := by
  apply Finset.sum_le_sum
  intro s _
  have a1 : lam ^ s ≤ mu ^ s := pow_le_pow_left₀ (by linarith) h s
  have a2 : 1 ≤ lam ^ s := one_le_pow₀ h1
  simp only [one_mul]
  nlinarith

/-- The same, averaged over any law of depths: the random-depth loss also falls toward 1. -/
theorem newEv_random_antitone {lam mu : ℝ} (h0 : 0 ≤ lam) (h : lam ≤ mu) (h1 : mu ≤ 1)
    (P : ℕ → ℝ) (Ts : Finset ℕ) (hP : ∀ T ∈ Ts, 0 ≤ P T) :
    ∑ T ∈ Ts, P T * newEvLoss mu 1 T ≤ ∑ T ∈ Ts, P T * newEvLoss lam 1 T :=
  Finset.sum_le_sum fun T hT => mul_le_mul_of_nonneg_left (newEv_antitone h0 h h1 T) (hP T hT)

/-- A RELAY: the reader wants only the newest evidence (coordinate `T - 1`), the older
coordinates must be forgotten.  The loss in closed form. -/
def relayLoss (lam b : ℝ) (T : ℕ) : ℝ :=
  ∑ s ∈ range T, (b * lam ^ s - if s = 0 then 1 else 0) ^ 2

/-- **A relay asks for gain 0, not 1.**  Its only exact fit is `b = 1, λ = 0`: moving
information along a chain needs no memory of the previous pass, so it does not ask the map to
keep anything. -/
theorem relay_zero_iff {lam b : ℝ} {T : ℕ} (hT : 2 ≤ T) :
    relayLoss lam b T = 0 ↔ b = 1 ∧ lam = 0 := by
  constructor
  · intro h
    have hnn : ∀ s ∈ range T, 0 ≤ (b * lam ^ s - if s = 0 then 1 else 0) ^ 2 :=
      fun _ _ => sq_nonneg _
    have hall := (Finset.sum_eq_zero_iff_of_nonneg hnn).1 h
    have h0 := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 (hall 0 (Finset.mem_range.2 (by omega)))
    have h1 := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 (hall 1 (Finset.mem_range.2 (by omega)))
    have h0' : b - 1 = 0 := by simpa using h0
    have h1' : b * lam = 0 := by simpa using h1
    have hb : b = 1 := by linarith
    refine ⟨hb, ?_⟩
    rw [hb, one_mul] at h1'
    exact h1'
  · rintro ⟨rfl, rfl⟩
    unfold relayLoss
    apply Finset.sum_eq_zero
    intro s _
    rcases Nat.eq_zero_or_pos s with rfl | hs
    · simp
    · simp [zero_pow hs.ne', hs.ne']

/-! ## 4. On a settled trajectory the recurrence's gradient is a write's gradient -/

section Settled

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The rollout with the pass's linear part perturbed: `h ↦ J h + ε • G h + c`. -/
def blockRoll (J G : E →L[ℝ] E) (c h0 : E) (ε : ℝ) : ℕ → E
  | 0 => h0
  | t + 1 => J (blockRoll J G c h0 ε t) + ε • G (blockRoll J G c h0 ε t) + c

/-- The first variation: `v₀ = 0`, `v_{t+1} = J v_t + G a_t`, `a_t` the unperturbed rollout. -/
def firstVar (J G : E →L[ℝ] E) (c h0 : E) : ℕ → E
  | 0 => 0
  | t + 1 => J (firstVar J G c h0 t) + G (blockRoll J G c h0 0 t)

/-- **The first variation is the derivative.**  `d/dε` of the exit state at `ε = 0` is
`firstVar T`. -/
theorem hasDerivAt_blockRoll (J G : E →L[ℝ] E) (c h0 : E) :
    ∀ T, HasDerivAt (fun ε => blockRoll J G c h0 ε T) (firstVar J G c h0 T) 0
  | 0 => by simpa [blockRoll, firstVar] using hasDerivAt_const (0 : ℝ) h0
  | T + 1 => by
    have ih := hasDerivAt_blockRoll J G c h0 T
    have hJ : HasDerivAt (fun ε => J (blockRoll J G c h0 ε T)) (J (firstVar J G c h0 T)) 0 :=
      HasFDerivAt.comp_hasDerivAt (0 : ℝ) J.hasFDerivAt ih
    have hG : HasDerivAt (fun ε => G (blockRoll J G c h0 ε T)) (G (firstVar J G c h0 T)) 0 :=
      HasFDerivAt.comp_hasDerivAt (0 : ℝ) G.hasFDerivAt ih
    have hS : HasDerivAt (fun ε => ε • G (blockRoll J G c h0 ε T))
        ((0 : ℝ) • G (firstVar J G c h0 T) + (1 : ℝ) • G (blockRoll J G c h0 0 T)) 0 :=
      (hasDerivAt_id (0 : ℝ)).smul hG
    have h1 := HasDerivAt.add hJ hS
    have h2 := (hasDerivAt_add_const_iff c).2 h1
    exact HasDerivAt.congr_deriv h2 (by simp [firstVar])

/-- The write-only part of the variation: `w₀ = 0`, `w_{t+1} = J w_t + g`, with `g = G h*`. -/
def writeVar (J : E →L[ℝ] E) (g : E) : ℕ → E
  | 0 => 0
  | t + 1 => J (writeVar J g t) + g

/-- The recurrence-only part: `r₀ = 0`, `r_{t+1} = J r_t + G (Jᵗ δ)`, `δ = h₀ - h*`. -/
def recVar (J G : E →L[ℝ] E) (δ : E) : ℕ → E
  | 0 => 0
  | t + 1 => J (recVar J G δ t) + G ((J ^ t) δ)

/-- The unperturbed rollout from `h₀` relaxes to the fixed point `h*` as `h* + Jᵗ (h₀ - h*)`. -/
theorem blockRoll_zero_eq (J G : E →L[ℝ] E) (c h0 hs : E) (hfix : J hs + c = hs) :
    ∀ t, blockRoll J G c h0 0 t = hs + (J ^ t) (h0 - hs)
  | 0 => by simp [blockRoll]
  | t + 1 => by
    simp only [blockRoll, zero_smul, add_zero]
    rw [blockRoll_zero_eq J G c h0 hs hfix t, map_add, pow_succ', mul_apply_eq_comp]
    have e : J hs + J ((J ^ t) (h0 - hs)) + c = (J hs + c) + J ((J ^ t) (h0 - hs)) := by abel
    rw [e, hfix]

/-- **The split.**  The first variation is the variation of a per-pass WRITE of `G h*` plus a
recurrence-only term driven by the transient `h₀ - h*`. -/
theorem firstVar_split (J G : E →L[ℝ] E) (c h0 hs : E) (hfix : J hs + c = hs) :
    ∀ t, firstVar J G c h0 t = writeVar J (G hs) t + recVar J G (h0 - hs) t
  | 0 => by simp [firstVar, writeVar, recVar]
  | t + 1 => by
    simp only [firstVar, writeVar, recVar]
    rw [firstVar_split J G c h0 hs hfix t, blockRoll_zero_eq J G c h0 hs hfix t, map_add,
      map_add]
    abel

/-- **Settled trajectory: the recurrence is invisible to first order.**  If the entry is the
fixed point, the derivative of the exit state along a change `G` of the recurrence is exactly
the derivative along a change `G h*` of the per-pass write. -/
theorem settled_block_is_write (J G : E →L[ℝ] E) (c hs : E) (hfix : J hs + c = hs) (T : ℕ) :
    HasDerivAt (fun ε => blockRoll J G c hs ε T) (writeVar J (G hs) T) 0 := by
  have h := hasDerivAt_blockRoll J G c hs T
  have hz : ∀ t, recVar J G 0 t = 0 := by
    intro t
    induction t with
    | zero => rfl
    | succ t ih => simp [recVar, ih]
  rw [firstVar_split J G c hs hs hfix T, sub_self, hz, add_zero] at h
  exact h

/-- The write variation is what adding `ε • g` to the per-pass source does, exactly:
the rollout of `h ↦ J h + (c + ε • g)` from `h*` is `h* + ε • writeVar T`. -/
theorem write_rollout (J : E →L[ℝ] E) (c hs g : E) (hfix : J hs + c = hs) (ε : ℝ) :
    ∀ T, (fun h => J h + (c + ε • g))^[T] hs = hs + ε • writeVar J g T
  | 0 => by simp [writeVar]
  | T + 1 => by
    rw [Function.iterate_succ_apply', write_rollout J c hs g hfix ε T]
    simp only [writeVar, map_add, map_smul, smul_add]
    have e : J hs + ε • J (writeVar J g T) + (c + ε • g)
        = (J hs + c) + (ε • J (writeVar J g T) + ε • g) := by abel
    rw [e, hfix]

theorem pow_apply_norm_le (J : E →L[ℝ] E) (x : E) : ∀ t, ‖(J ^ t) x‖ ≤ ‖J‖ ^ t * ‖x‖
  | 0 => by simp
  | t + 1 => by
    rw [pow_succ', mul_apply_eq_comp, pow_succ', mul_assoc]
    exact (J.le_opNorm _).trans
      (mul_le_mul_of_nonneg_left (pow_apply_norm_le J x t) (norm_nonneg _))

/-- The recurrence-only term is bounded by the transient: `‖r_T‖ ≤ ‖δ‖ · V_T`, with
`V₀ = 0`, `V_{t+1} = ‖J‖ V_t + ‖G‖ ‖J‖ᵗ`. -/
def recBound (nJ nG : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => nJ * recBound nJ nG t + nG * nJ ^ t

theorem recVar_le (J G : E →L[ℝ] E) (δ : E) :
    ∀ t, ‖recVar J G δ t‖ ≤ ‖δ‖ * recBound ‖J‖ ‖G‖ t
  | 0 => by simp [recVar, recBound]
  | t + 1 => by
    have ih := recVar_le J G δ t
    simp only [recVar, recBound]
    have hp : ‖(J ^ t) δ‖ ≤ ‖J‖ ^ t * ‖δ‖ := pow_apply_norm_le J δ t
    calc ‖J (recVar J G δ t) + G ((J ^ t) δ)‖
        ≤ ‖J (recVar J G δ t)‖ + ‖G ((J ^ t) δ)‖ := norm_add_le _ _
      _ ≤ ‖J‖ * ‖recVar J G δ t‖ + ‖G‖ * ‖(J ^ t) δ‖ :=
          add_le_add (J.le_opNorm _) (G.le_opNorm _)
      _ ≤ ‖J‖ * (‖δ‖ * recBound ‖J‖ ‖G‖ t) + ‖G‖ * (‖J‖ ^ t * ‖δ‖) :=
          add_le_add (mul_le_mul_of_nonneg_left ih (norm_nonneg _))
            (mul_le_mul_of_nonneg_left hp (norm_nonneg _))
      _ = ‖δ‖ * (‖J‖ * recBound ‖J‖ ‖G‖ t + ‖G‖ * ‖J‖ ^ t) := by ring

end Settled

end TulExploration
