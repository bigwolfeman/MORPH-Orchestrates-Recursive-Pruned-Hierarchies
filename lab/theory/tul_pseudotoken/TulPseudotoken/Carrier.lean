/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.InnerProductSpace.Orthonormal
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.LinearAlgebra.Dimension.Finite
import Mathlib.LinearAlgebra.FiniteDimensional.Defs

/-!
# P1: what one carrier position can hold for an exact copy

## Model

A carrier position is a vector `v` in the coda's input space `E`. The coda reads it after an
RMS norm, so what a token `i` sees is the cosine `strength e v i = ⟪v, e i⟫ / ‖v‖`, where
`e i` is the token's (tied) input embedding. A raw token at that position has strength `1`
along itself: that is the exact-copy read. The candidate tokens of one span (at most 32) are
modelled as an orthonormal family; nothing here needs the whole vocabulary to be orthonormal.

Three kinds of carrier are compared.

* A FREE vector (a CODI / Coconut hidden state, or LXTUL's `W_prefix` write): any `v`.
* A VOCAB MIXTURE (PonderLM's pondering embedding): `∑ p i • e i` with `p ≥ 0`.
* A SNAPPED mixture: the mixture with `p = softmax (logits)`, logits read from a cell.

## Main results

* `abs_strength_le_one`, `strength_eq_one_iff` : one position reaches strength 1 along a token
  only if it is a positive multiple of that token's embedding.
* **`sum_sq_strength_le_one`** (Bessel) and **`card_strong_mul_sq_le_one`** : the squared
  strengths of one position along a span's tokens sum to at most 1, so at most `1/τ²` tokens
  are read at strength `τ` or more. One position carries ONE exact token, free or mixture.
* **`mixture_strength_eq_one_iff`**, `vertex_mixture_eq` : a vocab mixture reaches strength 1
  along `y` exactly when it is the vertex `p = δ_y`, and then it IS `p y • e y`, the raw
  token's input up to scale. The exact-copy optimum of the free class is a vertex, so the
  mixture class loses nothing for an exact copy (A3, the extractive cell, is A2's vertex).
* **`affine_write_card_le`**, **`exact_tuple_write_needs_dim`** : an AFFINE write `c ↦ L c + b`
  from a cell space `V` that must produce `N` exact tokens, each chosen from `r + 1` tokens
  whose differences are linearly independent, needs `N * r ≤ dim V`. A real embedding table
  spans its space, so `r = d` and one `d`-dimensional cell cannot be written linearly into two
  exact arbitrary tokens.
* **`softmax_ge`**, **`snap_close`** : the snapped mixture is within `2 (n - 1) exp (-Δ)` of the
  exact embedding `e y` whenever the logits put `y` first by a margin `Δ`. Snapping is the
  nonlinear clean-up that the affine write lacks: a cell that only gets the ARGMAX right
  yields an exact token.
* **`exists_scale_commit`** : with a free logit scale, ONE snap commits to within any `ε`.
* **`passes_to_commit`** : if the logits are accumulated over `T` passes and each pass moves the
  margin of `y` over any other token by at most `m`, a commitment `p y ≥ 1 - ε` needs
  `T * m ≥ log ((n - 1) (1 - ε) / ε)`. Depth is necessary for commitment only when the scale
  per pass is bounded.
-/

namespace TulPseudotoken

open Finset

/-! ## Strength of one carrier position -/

section Strength

variable {ι E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The strength of carrier position `v` along token `i`: the cosine between `v` and the
token's embedding `e i`.  After the coda's RMS norm this is, up to the constant `√d`, the
coordinate of the normalised position along a unit-norm token embedding. -/
noncomputable def strength (e : ι → E) (v : E) (i : ι) : ℝ := inner ℝ v (e i) / ‖v‖

/-- A position's strength along any token is at most one in absolute value. -/
theorem abs_strength_le_one {e : ι → E} (he : Orthonormal ℝ e) (v : E) (i : ι) :
    |strength e v i| ≤ 1 := by
  unfold strength
  rcases eq_or_ne v 0 with rfl | hv
  · simp
  · rw [abs_div, abs_norm, div_le_one (norm_pos_iff.2 hv)]
    calc |inner ℝ v (e i)| ≤ ‖v‖ * ‖e i‖ := abs_real_inner_le_norm _ _
      _ = ‖v‖ := by rw [he.1 i, mul_one]

/-- **Strength one means the raw token.**  A position is read at strength exactly one along
token `i` if and only if it is a positive multiple of `e i`. -/
theorem strength_eq_one_iff {e : ι → E} (he : Orthonormal ℝ e) (v : E) (i : ι) :
    strength e v i = 1 ↔ ∃ r : ℝ, 0 < r ∧ v = r • e i := by
  have hei : ‖e i‖ = 1 := he.1 i
  have hform : strength e v i = inner ℝ v (e i) / (‖v‖ * ‖e i‖) := by
    rw [hei, mul_one]; rfl
  rw [hform, real_inner_div_norm_mul_norm_eq_one_iff]
  constructor
  · rintro ⟨_, r, hr, h⟩
    refine ⟨r⁻¹, inv_pos.2 hr, ?_⟩
    rw [h, smul_smul, inv_mul_cancel₀ hr.ne', one_smul]
  · rintro ⟨r, hr, rfl⟩
    have he0 : e i ≠ 0 := by
      intro h0; rw [h0, norm_zero] at hei; exact zero_ne_one hei
    refine ⟨smul_ne_zero hr.ne' he0, r⁻¹, inv_pos.2 hr, ?_⟩
    rw [smul_smul, inv_mul_cancel₀ hr.ne', one_smul]

/-- **Bessel for one carrier position.**  The squared strengths of a nonzero position along
any finite set of orthonormal tokens sum to at most one. -/
theorem sum_sq_strength_le_one {e : ι → E} (he : Orthonormal ℝ e) {v : E} (hv : v ≠ 0)
    (s : Finset ι) : ∑ i ∈ s, strength e v i ^ 2 ≤ 1 := by
  have hB := Orthonormal.sum_inner_products_le (𝕜 := ℝ) v (s := s) he
  simp only [Real.norm_eq_abs, sq_abs] at hB
  have hn : 0 < ‖v‖ ^ 2 := by positivity
  have hsum : ∑ i ∈ s, strength e v i ^ 2 = (∑ i ∈ s, inner ℝ (e i) v ^ 2) / ‖v‖ ^ 2 := by
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun i _ => ?_
    unfold strength
    rw [div_pow, real_inner_comm]
  rw [hsum, div_le_one hn]
  exact hB

/-- **One position, few strong tokens.**  The number of tokens read at strength `τ` or more,
times `τ²`, is at most one.  At most one token is read above `1/√2`; `k` tokens superposed in
one position cannot all be read above `1/√k`. -/
theorem card_strong_mul_sq_le_one {e : ι → E} (he : Orthonormal ℝ e) {v : E} (hv : v ≠ 0)
    (s : Finset ι) {τ : ℝ} (hτ : 0 ≤ τ) :
    ((s.filter fun i => τ ≤ |strength e v i|).card : ℝ) * τ ^ 2 ≤ 1 := by
  have hle : ∀ i ∈ s.filter (fun i => τ ≤ |strength e v i|), τ ^ 2 ≤ strength e v i ^ 2 := by
    intro i hi
    have h := (Finset.mem_filter.1 hi).2
    rw [← sq_abs (strength e v i)]
    exact pow_le_pow_left₀ hτ h 2
  have h1 := Finset.card_nsmul_le_sum _ _ _ hle
  have h2 : ∑ i ∈ s.filter (fun i => τ ≤ |strength e v i|), strength e v i ^ 2
      ≤ ∑ i ∈ s, strength e v i ^ 2 :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
      (fun i _ _ => sq_nonneg _)
  rw [nsmul_eq_mul] at h1
  linarith [sum_sq_strength_le_one he hv s]

/-! ## The vocab mixture -/

/-- PonderLM's pondering embedding restricted to a finite candidate set `s`:
`∑ i ∈ s, p i • e i`. -/
noncomputable def mixture (e : ι → E) (s : Finset ι) (p : ι → ℝ) : E := ∑ i ∈ s, p i • e i

lemma inner_mixture {e : ι → E} (he : Orthonormal ℝ e) (s : Finset ι) (p : ι → ℝ) {j : ι}
    (hj : j ∈ s) : inner ℝ (mixture e s p) (e j) = p j := by
  unfold mixture
  rw [he.inner_left_sum p hj]
  rfl

lemma norm_sq_mixture {e : ι → E} (he : Orthonormal ℝ e) (s : Finset ι) (p : ι → ℝ) :
    ‖mixture e s p‖ ^ 2 = ∑ i ∈ s, p i ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  unfold mixture
  rw [he.inner_sum p p s]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp [sq]

/-- **A vertex mixture is the raw token.**  If the weights vanish off `y`, the pondering
embedding is `p y • e y`: after the coda's norm, exactly the raw token's input. -/
theorem vertex_mixture_eq (e : ι → E) (s : Finset ι) (p : ι → ℝ) {y : ι} (hy : y ∈ s)
    (hzero : ∀ i ∈ s, i ≠ y → p i = 0) : mixture e s p = p y • e y := by
  unfold mixture
  rw [Finset.sum_eq_single y (fun i hi hiy => by rw [hzero i hi hiy, zero_smul])
    (fun h => absurd hy h)]

/-- **A mixture is exact only at a vertex.**  With non-negative weights, a nonzero vocab
mixture is read at strength one along `y` if and only if every other weight is zero.
Spreading mass over a second token always costs strength: the strength is
`p y / √(∑ p i²)`. -/
theorem mixture_strength_eq_one_iff {e : ι → E} (he : Orthonormal ℝ e) (s : Finset ι)
    {p : ι → ℝ} (hp : ∀ i ∈ s, 0 ≤ p i) {y : ι} (hy : y ∈ s)
    (hne : mixture e s p ≠ 0) :
    strength e (mixture e s p) y = 1 ↔ ∀ i ∈ s, i ≠ y → p i = 0 := by
  classical
  have hpos : 0 < ‖mixture e s p‖ := norm_pos_iff.2 hne
  have hstr : strength e (mixture e s p) y = p y / ‖mixture e s p‖ := by
    unfold strength; rw [inner_mixture he s p hy]
  rw [hstr, div_eq_one_iff_eq hpos.ne']
  have hsplit : ∑ i ∈ s, p i ^ 2 = p y ^ 2 + ∑ i ∈ s.erase y, p i ^ 2 :=
    (Finset.add_sum_erase s (fun i => p i ^ 2) hy).symm
  constructor
  · intro h i hi hiy
    have hsq : p y ^ 2 = ‖mixture e s p‖ ^ 2 := by rw [h]
    rw [norm_sq_mixture he, hsplit] at hsq
    have hz : ∑ i ∈ s.erase y, p i ^ 2 = 0 := by linarith
    have := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => sq_nonneg (p i))).1 hz i
      (Finset.mem_erase.2 ⟨hiy, hi⟩)
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 this
  · intro h
    have hz : ∑ i ∈ s.erase y, p i ^ 2 = 0 :=
      Finset.sum_eq_zero fun i hi => by
        rw [h i (Finset.mem_of_mem_erase hi) (Finset.ne_of_mem_erase hi)]; ring
    have hsq : p y ^ 2 = ‖mixture e s p‖ ^ 2 := by
      rw [norm_sq_mixture he, hsplit, hz, add_zero]
    exact (sq_eq_sq₀ (hp y hy) hpos.le).1 hsq

end Strength

/-! ## The affine write cannot make several exact tokens from one cell -/

section AffineWrite

variable {V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]

/-- **An affine write cannot exceed its source dimension.**  If an affine map
`c ↦ L c + b` sends a base point `c₀` and points `c i` to outputs whose differences from the
base output are linearly independent, there are at most `dim V` such points. -/
theorem affine_write_card_le [FiniteDimensional ℝ V] {κ : Type*} [Fintype κ]
    (L : V →ₗ[ℝ] W) (b : W) (c₀ : V) (c : κ → V)
    (hind : LinearIndependent ℝ fun i => (L (c i) + b) - (L c₀ + b)) :
    Fintype.card κ ≤ Module.finrank ℝ V := by
  have hfun : (fun i => (L (c i) + b) - (L c₀ + b)) = L ∘ fun i => c i - c₀ := by
    funext i; simp [map_sub]
  rw [hfun] at hind
  exact (LinearIndependent.of_comp L hind).fintype_card_le_finrank

/-- The one-slot change of a tuple of tokens: slot `n` moves from token `0` to token
`j.succ`, every other slot stays at token `0`.  As a vector of `N` embeddings, the change is
`e j.succ - e 0` at slot `n` and zero elsewhere. -/
lemma slot_change_linearIndependent {E : Type*} [AddCommGroup E] [Module ℝ E] {N r : ℕ}
    (e : Fin (r + 1) → E) (he : LinearIndependent ℝ fun j : Fin r => e j.succ - e 0) :
    LinearIndependent ℝ fun q : Fin N × Fin r =>
      (fun m : Fin N => if m = q.1 then e q.2.succ - e 0 else 0) := by
  rw [Fintype.linearIndependent_iff] at he ⊢
  intro g hg q
  have hslot := congrFun hg q.1
  simp only [Finset.sum_apply, Pi.smul_apply, Pi.zero_apply, smul_ite, smul_zero] at hslot
  rw [Fintype.sum_prod_type] at hslot
  have := he (fun j => g (q.1, j)) (by simpa [eq_comm] using hslot) q.2
  simpa using this

/-- **No affine write of exact token tuples.**  Let each of `N` coda positions need an EXACT
token chosen from `r + 1` tokens whose differences from token `0` are linearly independent
(a real embedding table spans its space, so `r` can be taken equal to the model width `d`).
If one cell space `V` feeds all `N` positions through an affine map, then `N * r ≤ dim V`.
With `r = d` and a `d`-dimensional cell this forces `N ≤ 1`: a linear read-out of one cell
(LXTUL's `W_prefix`, a wider read-out, a CODI / Coconut hidden state) cannot place two exact
arbitrary tokens. -/
theorem exact_tuple_write_needs_dim [FiniteDimensional ℝ V] {E : Type*} [AddCommGroup E]
    [Module ℝ E] {N r : ℕ} (e : Fin (r + 1) → E)
    (he : LinearIndependent ℝ fun j : Fin r => e j.succ - e 0)
    (L : V →ₗ[ℝ] (Fin N → E)) (b : Fin N → E) (c : (Fin N → Fin (r + 1)) → V)
    (hexact : ∀ x, L (c x) + b = fun m => e (x m)) :
    N * r ≤ Module.finrank ℝ V := by
  let base : Fin N → Fin (r + 1) := fun _ => 0
  let tup : Fin N × Fin r → Fin N → Fin (r + 1) := fun q => Function.update base q.1 q.2.succ
  have hdiff : (fun q => (L (c (tup q)) + b) - (L (c base) + b))
      = fun q : Fin N × Fin r => (fun m : Fin N => if m = q.1 then e q.2.succ - e 0 else 0) := by
    funext q m
    rw [hexact, hexact]
    simp only [Pi.sub_apply, tup, base, Function.update_apply]
    split_ifs <;> simp
  have hind := slot_change_linearIndependent (N := N) e he
  rw [← hdiff] at hind
  have := affine_write_card_le L b (c base) (fun q => c (tup q)) hind
  simpa [Fintype.card_prod, Fintype.card_fin] using this

end AffineWrite

/-! ## Snapping: the softmax clean-up -/

section Snap

variable {ι E : Type*}

/-- The softmax over a finite candidate set. -/
noncomputable def softmax (s : Finset ι) (ℓ : ι → ℝ) (i : ι) : ℝ :=
  Real.exp (ℓ i) / ∑ j ∈ s, Real.exp (ℓ j)

lemma softmax_nonneg (s : Finset ι) (ℓ : ι → ℝ) (i : ι) : 0 ≤ softmax s ℓ i :=
  div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)

lemma sum_softmax {s : Finset ι} (hs : s.Nonempty) (ℓ : ι → ℝ) :
    ∑ i ∈ s, softmax s ℓ i = 1 := by
  unfold softmax
  rw [← Finset.sum_div, div_self]
  exact (Finset.sum_pos (fun _ _ => Real.exp_pos _) hs).ne'

lemma sum_exp_split [DecidableEq ι] {s : Finset ι} {y : ι} (hy : y ∈ s) (ℓ : ι → ℝ) :
    ∑ j ∈ s, Real.exp (ℓ j) = Real.exp (ℓ y) + ∑ j ∈ s.erase y, Real.exp (ℓ j) :=
  (Finset.add_sum_erase s _ hy).symm

/-- **The snap commits when the margin is large.**  If every other candidate's logit is at
least `Δ` below `y`'s, the softmax puts at least `1 / (1 + (n - 1) e^{-Δ})` on `y`. -/
theorem softmax_ge {s : Finset ι} {y : ι} (hy : y ∈ s) (ℓ : ι → ℝ) (Δ : ℝ)
    (hΔ : ∀ j ∈ s, j ≠ y → ℓ j ≤ ℓ y - Δ) :
    1 / (1 + ((s.card : ℝ) - 1) * Real.exp (-Δ)) ≤ softmax s ℓ y := by
  classical
  have hrest : ∑ j ∈ s.erase y, Real.exp (ℓ j)
      ≤ ((s.card : ℝ) - 1) * (Real.exp (ℓ y) * Real.exp (-Δ)) := by
    have hterm : ∀ j ∈ s.erase y, Real.exp (ℓ j) ≤ Real.exp (ℓ y) * Real.exp (-Δ) := by
      intro j hj
      rw [← Real.exp_add]
      exact Real.exp_le_exp.2 (by
        linarith [hΔ j (Finset.mem_of_mem_erase hj) (Finset.ne_of_mem_erase hj)])
    have := Finset.sum_le_card_nsmul _ _ _ hterm
    rw [nsmul_eq_mul, Finset.card_erase_of_mem hy, Nat.cast_sub (Finset.card_pos.2 ⟨y, hy⟩),
      Nat.cast_one] at this
    exact this
  have hey : 0 < Real.exp (ℓ y) := Real.exp_pos _
  have hden : 0 < ∑ j ∈ s, Real.exp (ℓ j) := Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨y, hy⟩
  have hc : (0 : ℝ) ≤ (s.card : ℝ) - 1 := by
    have : 1 ≤ s.card := Finset.card_pos.2 ⟨y, hy⟩
    have : (1 : ℝ) ≤ s.card := by exact_mod_cast this
    linarith
  have hA : 0 < 1 + ((s.card : ℝ) - 1) * Real.exp (-Δ) := by
    have := Real.exp_pos (-Δ); positivity
  unfold softmax
  rw [div_le_div_iff₀ hA hden, sum_exp_split hy]
  nlinarith [hrest]

/-- The mass off `y` is at most `(n - 1) e^{-Δ}`. -/
theorem one_sub_softmax_le {s : Finset ι} {y : ι} (hy : y ∈ s) (ℓ : ι → ℝ)
    (Δ : ℝ) (hΔ : ∀ j ∈ s, j ≠ y → ℓ j ≤ ℓ y - Δ) :
    1 - softmax s ℓ y ≤ ((s.card : ℝ) - 1) * Real.exp (-Δ) := by
  classical
  have h := softmax_ge hy ℓ Δ hΔ
  set a := ((s.card : ℝ) - 1) * Real.exp (-Δ) with ha
  have hc : (0 : ℝ) ≤ (s.card : ℝ) - 1 := by
    have : (1 : ℝ) ≤ s.card := by exact_mod_cast Finset.card_pos.2 ⟨y, hy⟩
    linarith
  have ha0 : 0 ≤ a := mul_nonneg hc (Real.exp_pos _).le
  have h1 : 1 - 1 / (1 + a) ≤ a := by
    rw [one_sub_div (by linarith), add_sub_cancel_left, div_le_iff₀ (by linarith)]
    nlinarith
  linarith

/-- **The snapped pseudo token is the exact token, up to `2 (n - 1) e^{-Δ}`.**  The vocab
mixture under the softmax weights is within twice the mass off `y` of the raw embedding
`e y`, for any embeddings of norm at most one.  A cell whose logits put the right token first
by a margin `Δ` therefore writes an exact token, whatever else the cell holds. -/
theorem snap_close [NormedAddCommGroup E] [NormedSpace ℝ E] {s : Finset ι}
    {y : ι} (hy : y ∈ s) (e : ι → E) (he : ∀ i, ‖e i‖ ≤ 1) (ℓ : ι → ℝ) (Δ : ℝ)
    (hΔ : ∀ j ∈ s, j ≠ y → ℓ j ≤ ℓ y - Δ) :
    ‖∑ i ∈ s, softmax s ℓ i • e i - e y‖ ≤ 2 * (((s.card : ℝ) - 1) * Real.exp (-Δ)) := by
  classical
  have hsum1 := sum_softmax ⟨y, hy⟩ ℓ
  have hrewrite : ∑ i ∈ s, softmax s ℓ i • e i - e y
      = ∑ i ∈ s.erase y, softmax s ℓ i • (e i - e y) := by
    have hey : e y = ∑ i ∈ s, softmax s ℓ i • e y := by
      rw [← Finset.sum_smul, hsum1, one_smul]
    conv_lhs => rw [hey, ← Finset.sum_sub_distrib]
    simp_rw [← smul_sub]
    rw [← Finset.add_sum_erase s _ hy, sub_self, smul_zero, zero_add]
  rw [hrewrite]
  have hnorm : ‖∑ i ∈ s.erase y, softmax s ℓ i • (e i - e y)‖
      ≤ ∑ i ∈ s.erase y, softmax s ℓ i * 2 := by
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun i _ => ?_)
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (softmax_nonneg s ℓ i)]
    refine mul_le_mul_of_nonneg_left ?_ (softmax_nonneg s ℓ i)
    calc ‖e i - e y‖ ≤ ‖e i‖ + ‖e y‖ := norm_sub_le _ _
      _ ≤ 2 := by linarith [he i, he y]
  have hoff : ∑ i ∈ s.erase y, softmax s ℓ i = 1 - softmax s ℓ y := by
    rw [← hsum1, ← Finset.add_sum_erase s _ hy]; ring
  rw [← Finset.sum_mul, hoff] at hnorm
  have := one_sub_softmax_le hy ℓ Δ hΔ
  linarith

/-- **A free scale commits in one snap.**  If the logits put `y` first by any positive
margin, some scale `β ≥ 0` makes the softmax of `β ℓ` put at least `1 - ε` on `y`.  With a
learnable temperature, one pass is enough for an exact token. -/
theorem exists_scale_commit {s : Finset ι} {y : ι} (hy : y ∈ s) (ℓ : ι → ℝ)
    {Δ : ℝ} (hΔpos : 0 < Δ) (hΔ : ∀ j ∈ s, j ≠ y → ℓ j ≤ ℓ y - Δ) {ε : ℝ} (hε : 0 < ε) :
    ∃ β : ℝ, 0 ≤ β ∧ 1 - ε ≤ softmax s (fun j => β * ℓ j) y := by
  classical
  have hn : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.2 ⟨y, hy⟩
  refine ⟨|Real.log (s.card / ε)| / Δ, div_nonneg (abs_nonneg _) hΔpos.le, ?_⟩
  set β := |Real.log (s.card / ε)| / Δ with hβ
  have hβ0 : 0 ≤ β := div_nonneg (abs_nonneg _) hΔpos.le
  have hmargin : ∀ j ∈ s, j ≠ y → β * ℓ j ≤ β * ℓ y - β * Δ := by
    intro j hj hjy
    have := hΔ j hj hjy
    nlinarith
  have h := one_sub_softmax_le hy (fun j => β * ℓ j) (β * Δ) hmargin
  have hβΔ : β * Δ = |Real.log (s.card / ε)| := by
    rw [hβ, div_mul_cancel₀ _ hΔpos.ne']
  have hexp : Real.exp (-(β * Δ)) ≤ ε / s.card := by
    rw [hβΔ]
    have hle : Real.log (s.card / ε) ≤ |Real.log (s.card / ε)| := le_abs_self _
    have h1 : Real.exp (-|Real.log (s.card / ε)|) ≤ Real.exp (-Real.log (s.card / ε)) :=
      Real.exp_le_exp.2 (by linarith)
    rw [Real.exp_neg (Real.log _), Real.exp_log (div_pos hn hε), inv_div] at h1
    exact h1
  have hc : (0 : ℝ) ≤ (s.card : ℝ) - 1 := by
    have : (1 : ℝ) ≤ s.card := by exact_mod_cast Finset.card_pos.2 ⟨y, hy⟩
    linarith
  have hfin : ((s.card : ℝ) - 1) * Real.exp (-(β * Δ)) ≤ ε := by
    calc ((s.card : ℝ) - 1) * Real.exp (-(β * Δ)) ≤ ((s.card : ℝ) - 1) * (ε / s.card) :=
          mul_le_mul_of_nonneg_left hexp hc
      _ ≤ ε := by
          rw [mul_div_assoc', div_le_iff₀ hn]
          nlinarith
  linarith

/-- The softmax mass on `y` is at most `1 / (1 + (n - 1) e^{-M})` when `y`'s logit exceeds
every other candidate's by at most `M`. -/
theorem softmax_le {s : Finset ι} {y : ι} (hy : y ∈ s) (ℓ : ι → ℝ) (M : ℝ)
    (hM : ∀ j ∈ s, ℓ y - ℓ j ≤ M) :
    softmax s ℓ y ≤ 1 / (1 + ((s.card : ℝ) - 1) * Real.exp (-M)) := by
  classical
  have hrest : ((s.card : ℝ) - 1) * (Real.exp (ℓ y) * Real.exp (-M))
      ≤ ∑ j ∈ s.erase y, Real.exp (ℓ j) := by
    have hterm : ∀ j ∈ s.erase y, Real.exp (ℓ y) * Real.exp (-M) ≤ Real.exp (ℓ j) := by
      intro j hj
      rw [← Real.exp_add]
      exact Real.exp_le_exp.2 (by linarith [hM j (Finset.mem_of_mem_erase hj)])
    have := Finset.card_nsmul_le_sum _ _ _ hterm
    rw [nsmul_eq_mul, Finset.card_erase_of_mem hy, Nat.cast_sub (Finset.card_pos.2 ⟨y, hy⟩),
      Nat.cast_one] at this
    exact this
  have hden : 0 < ∑ j ∈ s, Real.exp (ℓ j) := Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨y, hy⟩
  have hc : (0 : ℝ) ≤ (s.card : ℝ) - 1 := by
    have : (1 : ℝ) ≤ s.card := by exact_mod_cast Finset.card_pos.2 ⟨y, hy⟩
    linarith
  have hA : 0 < 1 + ((s.card : ℝ) - 1) * Real.exp (-M) := by
    have := Real.exp_pos (-M); positivity
  unfold softmax
  rw [div_le_div_iff₀ hden hA, sum_exp_split hy]
  nlinarith [hrest]

/-- **Bounded scale per pass: commitment needs passes.**  Let the logits be accumulated over
`T` passes, `ℓ j = ∑_{t < T} δ t j`, and let each pass raise `y`'s logit over any other
candidate's by at most `m`.  If the softmax then puts at least `1 - ε` on `y`
(`0 < ε < 1`, at least two candidates), `T * m ≥ log ((n - 1) (1 - ε) / ε)`. -/
theorem passes_to_commit {s : Finset ι} {y : ι} (hy : y ∈ s)
    (hcard : 2 ≤ s.card) (δ : ℕ → ι → ℝ) (T : ℕ) (m : ℝ)
    (hδ : ∀ t < T, ∀ j ∈ s, δ t y - δ t j ≤ m) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε < 1)
    (hcommit : 1 - ε ≤ softmax s (fun j => ∑ t ∈ Finset.range T, δ t j) y) :
    Real.log (((s.card : ℝ) - 1) * (1 - ε) / ε) ≤ T * m := by
  classical
  set ℓ := fun j => ∑ t ∈ Finset.range T, δ t j with hℓ
  have hM : ∀ j ∈ s, ℓ y - ℓ j ≤ T * m := by
    intro j hj
    simp only [hℓ, ← Finset.sum_sub_distrib]
    have := Finset.sum_le_card_nsmul (Finset.range T) (fun t => δ t y - δ t j) m
      (fun t ht => hδ t (Finset.mem_range.1 ht) j hj)
    simpa [nsmul_eq_mul] using this
  have hup := softmax_le hy ℓ (T * m) hM
  have hc : (1 : ℝ) ≤ (s.card : ℝ) - 1 := by
    have : (2 : ℝ) ≤ s.card := by exact_mod_cast hcard
    linarith
  set a := ((s.card : ℝ) - 1) * Real.exp (-(T * m)) with ha
  have ha0 : 0 < a := mul_pos (by linarith) (Real.exp_pos _)
  have h1 : 1 - ε ≤ 1 / (1 + a) := hcommit.trans hup
  have h2 : a ≤ ε / (1 - ε) := by
    rw [le_div_iff₀ (by linarith)]
    rw [le_div_iff₀ (by linarith)] at h1
    nlinarith
  have h3 : ((s.card : ℝ) - 1) * (1 - ε) / ε ≤ Real.exp (T * m) := by
    rw [div_le_iff₀ hε0]
    have hexp : Real.exp (-(T * m)) * Real.exp (T * m) = 1 := by
      rw [← Real.exp_add]; simp
    have hpos := Real.exp_pos (T * m)
    have : ((s.card : ℝ) - 1) * Real.exp (-(T * m)) * (1 - ε) ≤ ε := by
      rw [le_div_iff₀ (by linarith)] at h2; linarith
    nlinarith
  have hpos : 0 < ((s.card : ℝ) - 1) * (1 - ε) / ε :=
    div_pos (mul_pos (by linarith) (by linarith)) hε0
  calc Real.log (((s.card : ℝ) - 1) * (1 - ε) / ε) ≤ Real.log (Real.exp (T * m)) :=
        Real.log_le_log hpos h3
    _ = T * m := Real.log_exp _

end Snap

end TulPseudotoken
