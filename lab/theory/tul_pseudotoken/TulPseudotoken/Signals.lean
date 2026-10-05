/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fin.VecNotation

/-!
# P2: which trace-free signal pays a carrier for copy content

## Model

One span's carrier holds a set `S` of the span's own tokens (`ι` indexes the span positions).
`Carrier.lean` justifies the set form: an exact copy needs a vertex, one per position, so a
carrier of `N` positions holds at most `N` exact tokens.

Later in the row, reads `r : ρ` copy a token of this span. Read `r` copies span position
`src r`, sits `hz r` spans ahead, and saves `w r ≥ 0` nats of CE when the carrier holds its
source (`w` folds in the read's probability). This is the shape the exact-recall probe
measured: 19 % of LXTUL's tokens are bigram repeats from more than 32 tokens back, and they
carry 67 % of the gap.

The signals, as functions of `S`:

* `ceValue` : the coda's own CE through the carrier. The coda of EVERY later span reads the
  carrier, so every read counts, at every horizon.
* `nextValue` : the next-span decoder (`tul.spandec`) and the latent rank head both score the
  carrier against the NEXT span only: reads with `hz r = 1`.
* `recValue` : own-span reconstruction. Carrying token `i` earns its reconstruction value
  `a i` (its surprisal given the rest), whatever later spans do.
* `kdLoss` : CODI's hidden-state distillation. The teacher (the coda with the raw span in
  view) is shifted by `f i` per span token (CODI eq. 4: context enters as an additive shift);
  the student with carrier `S` gets `∑_{i ∈ S} f i`; the loss is the squared distance.

## Main results

* `ceValue_eq_next_add_far`, **`next_blind`** : the next-span signals see only the horizon-1
  part of the CE value. When every copy is two or more spans ahead they are flat: the empty
  carrier is optimal for them and worth nothing to the coda.
* **`rec_misaligned`** : reconstruction can select a token no later span copies over one that
  a later span does: its optimum can be worth zero to the coda.
* **`aux_price`** : adding any auxiliary bounded in `[0, R]` at weight `λ` to the CE value
  costs at most `λ R` of CE value at the optimum. An auxiliary is safe as a small helper.
* `norm_sq_sum_orth`, **`kdLoss_orth`**, **`kd_misaligned`** : with orthogonal shifts the
  hidden-state loss is `∑_{i ∉ S} ‖f i‖²`, so it selects by the size of a token's
  hidden-state shift, not by the CE it saves. It can pick the CE-worthless token.
-/

namespace TulPseudotoken

open Finset

/-- One span's copy task: later reads that copy this span's tokens. -/
structure CopyTask (ι ρ : Type*) where
  /-- CE (nats) a read saves when the carrier holds its source, times its probability. -/
  w : ρ → ℝ
  w_nonneg : ∀ r, 0 ≤ w r
  /-- The span position the read copies. -/
  src : ρ → ι
  /-- How many spans ahead the read is (`1` = the next span). -/
  hz : ρ → ℕ

namespace CopyTask

variable {ι ρ : Type*} [Fintype ρ] [DecidableEq ι] (T : CopyTask ι ρ)

/-- **The coda's own CE through the carrier**: every later read, every horizon. -/
def ceValue (S : Finset ι) : ℝ := ∑ r, if T.src r ∈ S then T.w r else 0

/-- **The next-span signals** (span decoder, latent rank head): horizon-1 reads only. -/
def nextValue (S : Finset ι) : ℝ := ∑ r, if T.hz r = 1 ∧ T.src r ∈ S then T.w r else 0

/-- The part of the CE value the next-span signals cannot see. -/
def farValue (S : Finset ι) : ℝ := ∑ r, if T.hz r ≠ 1 ∧ T.src r ∈ S then T.w r else 0

theorem ceValue_eq_next_add_far (S : Finset ι) :
    T.ceValue S = T.nextValue S + T.farValue S := by
  unfold ceValue nextValue farValue
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun r _ => ?_
  by_cases h1 : T.hz r = 1 <;> by_cases h2 : T.src r ∈ S <;> simp [h1, h2]

lemma ceValue_empty : T.ceValue ∅ = 0 := by simp [ceValue]

/-- **The next-span signals are blind past one span.**  If no read is in the next span, the
next-span value of every carrier is zero: every carrier, the empty one included, is optimal
for those signals, and the empty carrier is worth nothing to the coda. -/
theorem next_blind (hfar : ∀ r, T.hz r ≠ 1) :
    (∀ S : Finset ι, T.nextValue S = 0) ∧ T.ceValue ∅ = 0 := by
  refine ⟨fun S => ?_, T.ceValue_empty⟩
  unfold nextValue
  exact Finset.sum_eq_zero fun r _ => by simp [hfar r]

end CopyTask

/-- **Own-span reconstruction**: carrying token `i` earns `a i`, whatever later spans read. -/
def recValue {ι : Type*} (a : ι → ℝ) (S : Finset ι) : ℝ := ∑ i ∈ S, a i

/-- **Reconstruction can pick the token nobody copies.**  Two span tokens, a one-token
carrier. Token `1` is harder to reconstruct (`a₀ < a₁`, `0 < a₁`); only token `0` is copied
later (saving `w₀ > 0`). The reconstruction optimum over carriers of at most one token is
`{1}`, which saves the coda nothing; `{0}` saves `w₀`. -/
theorem rec_misaligned (a₀ a₁ w₀ : ℝ) (h01 : a₀ < a₁) (h1 : 0 < a₁) (hw : 0 < w₀) :
    ∃ T : CopyTask (Fin 2) Unit,
      (∀ S : Finset (Fin 2), S.card ≤ 1 → recValue ![a₀, a₁] S ≤ recValue ![a₀, a₁] {1}) ∧
      T.ceValue {1} = 0 ∧ T.ceValue {0} = w₀ := by
  refine ⟨⟨fun _ => w₀, fun _ => hw.le, fun _ => 0, fun _ => 2⟩, ?_, ?_, ?_⟩
  · intro S hS
    have hle : ∀ i ∈ S, (![a₀, a₁] : Fin 2 → ℝ) i ≤ a₁ := by
      intro i _
      fin_cases i <;> simp [h01.le]
    have h := Finset.sum_le_card_nsmul S _ a₁ hle
    rw [nsmul_eq_mul] at h
    have hc : (S.card : ℝ) ≤ 1 := by exact_mod_cast hS
    have : (S.card : ℝ) * a₁ ≤ a₁ := by nlinarith
    simp only [recValue, Finset.sum_singleton]
    simpa using h.trans this
  · simp [CopyTask.ceValue]
  · simp [CopyTask.ceValue]

/-- **The price of an auxiliary.**  Let `J` be the CE value and `A` an auxiliary with values
in `[0, R]` on the admissible carriers.  A carrier that maximises `J + λ A` (`λ ≥ 0`) over the
admissible carriers is within `λ R` of every admissible carrier's CE value. -/
theorem aux_price {κ : Type*} (adm : κ → Prop) (J A : κ → ℝ) {R lam : ℝ}
    (hA : ∀ S, adm S → 0 ≤ A S ∧ A S ≤ R) (hlam : 0 ≤ lam) {Sopt : κ} (hSopt : adm Sopt)
    (hopt : ∀ S, adm S → J S + lam * A S ≤ J Sopt + lam * A Sopt) {S : κ} (hS : adm S) :
    J S - lam * R ≤ J Sopt := by
  have h1 := hopt S hS
  have h2 : 0 ≤ lam * A S := mul_nonneg hlam (hA S hS).1
  have h3 : lam * A Sopt ≤ lam * R := mul_le_mul_of_nonneg_left (hA Sopt hSopt).2 hlam
  linarith

section KD

variable {ι E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Pythagoras for a finite orthogonal family. -/
lemma norm_sq_sum_orth (f : ι → E) (hf : Pairwise fun i j => inner ℝ (f i) (f j) = 0)
    (s : Finset ι) : ‖∑ i ∈ s, f i‖ ^ 2 = ∑ i ∈ s, ‖f i‖ ^ 2 := by
  classical
  rw [← real_inner_self_eq_norm_sq, sum_inner]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [inner_sum, Finset.sum_eq_single i (fun j _ hji => hf (Ne.symm hji))
    (fun h => absurd hi h), real_inner_self_eq_norm_sq]

/-- CODI's hidden-state loss for a carrier `S`: the teacher is shifted by every span token,
the student only by the carried ones. -/
noncomputable def kdLoss [Fintype ι] (f : ι → E) (S : Finset ι) : ℝ :=
  ‖∑ i, f i - ∑ i ∈ S, f i‖ ^ 2

/-- **The hidden-state loss ranks tokens by the size of their shift.**  With orthogonal
shifts it is `∑_{i ∉ S} ‖f i‖²`. -/
theorem kdLoss_orth [Fintype ι] [DecidableEq ι] (f : ι → E)
    (hf : Pairwise fun i j => inner ℝ (f i) (f j) = 0) (S : Finset ι) :
    kdLoss f S = ∑ i ∈ Finset.univ \ S, ‖f i‖ ^ 2 := by
  unfold kdLoss
  rw [← Finset.sum_sdiff (Finset.subset_univ S), add_sub_cancel_right]
  exact norm_sq_sum_orth f hf _

/-- **Hidden-state distillation can carry the CE-worthless token.**  Two span tokens with
orthogonal shifts, `‖f 0‖ < ‖f 1‖`; only token `0` is copied later.  The hidden-state loss
strictly prefers carrying token `1`. -/
theorem kd_misaligned (f : Fin 2 → E) (hf : Pairwise fun i j => inner ℝ (f i) (f j) = 0)
    (hsize : ‖f 0‖ < ‖f 1‖) : kdLoss f {1} < kdLoss f {0} := by
  rw [kdLoss_orth f hf, kdLoss_orth f hf]
  have h0 : (Finset.univ \ {1} : Finset (Fin 2)) = {0} := by decide
  have h1 : (Finset.univ \ {0} : Finset (Fin 2)) = {1} := by decide
  rw [h0, h1, Finset.sum_singleton, Finset.sum_singleton]
  exact pow_lt_pow_left₀ hsize (norm_nonneg _) (by norm_num)

end KD

end TulPseudotoken
