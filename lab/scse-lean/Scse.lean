/-
# SCSE loop-level residual: stability of the corrected deviation recurrence

Setting (arXiv:2607.27656 adaptation). The paper's looped recurrence is
  `Δ_{t+1} = Δ_t + s • G Δ_t`
where `G` carries NO top-level identity (the residual is hoisted to loop level).

Our core stack DOES carry its own residual. We model it as
  `stack x = C x + U x`
with `C` norm-preserving (`‖C x‖ = ‖x‖`, input-dependent, NOT assumed linear)
and `U 0 = 0`. Our shipped adaptation uses `G(x) := stack x - x`, i.e.

  `corrected s x = x + s • (stack x - x)`   -- ours
  `naive     s x = x + s • stack x`         -- the first (wrong) form

Results:
  T1  one-step bound for `corrected` (carry contributes nothing to growth)
  T2  iterated bound: growth ≤ linear in loop depth T
  T3  the naive form is geometric in T and admits no such bound (counterexample)
  T4  when `C = id`, `corrected` IS the paper's recurrence with `G = U`
  T5  zero is a fixed point of `corrected` and of the masked recurrence

Toolchain: Lean 4.31.0 / Mathlib v4.31.0.  Build: `lake build Scse` in this
directory (Mathlib oleans are prebuilt under `.lake/packages`).
-/
import Mathlib

noncomputable section

namespace SCSE

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The core block stack, decomposed as carry plus update. -/
def stack (C U : E → E) (x : E) : E := C x + U x

/-- Our corrected recurrence: `x + s • (stack x - x)`. -/
def corrected (C U : E → E) (s : ℝ) (x : E) : E := x + s • (stack C U x - x)

/-- The naive recurrence we shipped first: `x + s • stack x`. -/
def naive (C U : E → E) (s : ℝ) (x : E) : E := x + s • stack C U x

/-- The paper's recurrence (SCSE, residual hoisted to loop level): `x + s • G x`. -/
def paper (G : E → E) (s : ℝ) (x : E) : E := x + s • G x

/-- Convex-combination form of the corrected step. -/
lemma corrected_eq (C U : E → E) (s : ℝ) (x : E) :
    corrected C U s x = (1 - s) • x + s • C x + s • U x := by
  simp only [corrected, stack]
  module

/-- **T1 (one-step bound, corrected form).**
For `0 ≤ s ≤ 1`, the norm-preserving carry contributes nothing to growth:
`‖corrected s x‖ ≤ ‖x‖ + s * ‖U x‖`. -/
theorem T1_one_step_bound (C U : E → E) (hC : ∀ x, ‖C x‖ = ‖x‖)
    {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (x : E) :
    ‖corrected C U s x‖ ≤ ‖x‖ + s * ‖U x‖ := by
  rw [corrected_eq]
  have h1 : ‖(1 - s) • x‖ = (1 - s) * ‖x‖ := by
    rw [norm_smul, Real.norm_of_nonneg (by linarith)]
  have h2 : ‖s • C x‖ = s * ‖x‖ := by
    rw [norm_smul, Real.norm_of_nonneg hs0, hC]
  have h3 : ‖s • U x‖ = s * ‖U x‖ := by
    rw [norm_smul, Real.norm_of_nonneg hs0]
  calc ‖(1 - s) • x + s • C x + s • U x‖
      ≤ ‖(1 - s) • x + s • C x‖ + ‖s • U x‖ := norm_add_le _ _
    _ ≤ ‖(1 - s) • x‖ + ‖s • C x‖ + ‖s • U x‖ := by
        have := norm_add_le ((1 - s) • x) (s • C x)
        linarith
    _ = ‖x‖ + s * ‖U x‖ := by rw [h1, h2, h3]; ring

/-- **T2 (iterated bound, corrected form).**
If the update branch is bounded, `‖U x‖ ≤ B`, then `T` loop iterations grow
the deviation at most LINEARLY in the depth: `‖corrected^[T] Δ₀‖ ≤ ‖Δ₀‖ + s*T*B`. -/
theorem T2_iterated_bound (C U : E → E) (hC : ∀ x, ‖C x‖ = ‖x‖)
    {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) {B : ℝ} (hU : ∀ x, ‖U x‖ ≤ B)
    (Δ₀ : E) (T : ℕ) :
    ‖(corrected C U s)^[T] Δ₀‖ ≤ ‖Δ₀‖ + s * T * B := by
  induction T with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      set y := (corrected C U s)^[n] Δ₀ with hy
      have step : ‖corrected C U s y‖ ≤ ‖y‖ + s * ‖U y‖ :=
        T1_one_step_bound C U hC hs0 hs1 y
      have hUb : s * ‖U y‖ ≤ s * B := mul_le_mul_of_nonneg_left (hU y) hs0
      have : ‖corrected C U s y‖ ≤ ‖Δ₀‖ + s * n * B + s * B := by linarith
      calc ‖corrected C U s y‖ ≤ ‖Δ₀‖ + s * n * B + s * B := this
        _ = ‖Δ₀‖ + s * (n + 1) * B := by ring
        _ = ‖Δ₀‖ + s * ((n : ℕ) + 1 : ℕ) * B := by push_cast; ring

/-! ## T3: the naive form does NOT admit T1/T2.

Witness: `E = ℝ`, `C = id` (a legitimate norm-preserving map), `U = 0`
(so `U 0 = 0` and `‖U x‖ ≤ 0 = B` hold). Then the naive step is exactly
`x ↦ (1 + s) * x`, whose iterates are geometric in `T`. -/

/-- On the witness, one naive step is multiplication by `1 + s`. -/
lemma naive_witness_step (s x : ℝ) :
    naive (id : ℝ → ℝ) (fun _ => (0 : ℝ)) s x = (1 + s) * x := by
  simp only [naive, stack, id, add_zero, smul_eq_mul]
  ring

/-- On the witness, `T` naive steps multiply by `(1 + s)^T`. -/
lemma naive_witness_iter (s : ℝ) (T : ℕ) (x : ℝ) :
    (naive (id : ℝ → ℝ) (fun _ => (0 : ℝ)) s)^[T] x = (1 + s) ^ T * x := by
  induction T with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ_apply', ih, naive_witness_step]
      ring

/-- **T3a (exact geometric growth of the naive form).**
For `s ≥ 0`: `‖naive^[T] Δ₀‖ = (1+s)^T * ‖Δ₀‖` — geometric in the loop depth. -/
theorem T3_naive_geometric {s : ℝ} (hs : 0 ≤ s) (Δ₀ : ℝ) (T : ℕ) :
    ‖(naive (id : ℝ → ℝ) (fun _ => (0 : ℝ)) s)^[T] Δ₀‖ = (1 + s) ^ T * ‖Δ₀‖ := by
  rw [naive_witness_iter, Real.norm_eq_abs, Real.norm_eq_abs, abs_mul,
    abs_of_nonneg (pow_nonneg (by linarith) T)]

/-- **T3b (unboundedness).** For any `s > 0` the naive iterates from `Δ₀ = 1`
exceed every bound `M` at some finite depth `T`: no bound independent of the
form `f(T)` with sub-geometric `f` can hold; in particular no linear bound. -/
theorem T3_naive_unbounded {s : ℝ} (hs : 0 < s) (M : ℝ) :
    ∃ T : ℕ, M < ‖(naive (id : ℝ → ℝ) (fun _ => (0 : ℝ)) s)^[T] (1 : ℝ)‖ := by
  obtain ⟨n, hn⟩ := Archimedean.arch M hs
  rw [nsmul_eq_mul] at hn
  refine ⟨n + 1, ?_⟩
  rw [T3_naive_geometric hs.le, norm_one, mul_one]
  have hber : 1 + ((n : ℝ) + 1) * s ≤ (1 + s) ^ (n + 1) := by
    have h := one_add_mul_le_pow (a := s) (by linarith) (n + 1)
    push_cast at h
    linarith [h]
  nlinarith [hn, hs]

/-- **T3c (the separation).** The conclusion of T2 is FALSE for the naive form:
it is not the case that for every norm-preserving carry `C`, bounded branch `U`,
and `0 ≤ s ≤ 1`, the naive iterates satisfy `‖naive^[T] Δ₀‖ ≤ ‖Δ₀‖ + s*T*B`.
Refuted by the witness `C = id`, `U = 0`, `s = 1/2`, `Δ₀ = 1`, `T = 1`. -/
theorem T3_no_T2_for_naive :
    ¬ (∀ (C U : ℝ → ℝ) (s B : ℝ), (∀ x, ‖C x‖ = ‖x‖) → 0 ≤ s → s ≤ 1 →
        (∀ x, ‖U x‖ ≤ B) → ∀ (Δ₀ : ℝ) (T : ℕ),
        ‖(naive C U s)^[T] Δ₀‖ ≤ ‖Δ₀‖ + s * T * B) := by
  intro h
  have hbad := h id (fun _ => (0 : ℝ)) (1/2) 0 (fun x => rfl)
    (by norm_num) (by norm_num) (fun x => by simp) 1 1
  rw [Function.iterate_one, naive_witness_step] at hbad
  norm_num [Real.norm_eq_abs] at hbad

/-- **T4 (reduction to the paper).** When the carry is the plain identity,
our corrected recurrence IS the paper's recurrence with `G = U`:
`corrected s x = x + s • U x = paper U s x`. Our form strictly generalises the
published algorithm and collapses to it exactly when `C = id`. -/
theorem T4_reduces_to_paper (C U : E → E) (hC : ∀ x, C x = x) (s : ℝ) (x : E) :
    corrected C U s x = paper U s x := by
  simp only [corrected, stack, paper, hC, add_sub_cancel_left]

omit [NormedSpace ℝ E] in
/-- `C 0 = 0` is FORCED by norm preservation (not an extra assumption). -/
lemma carry_zero (C : E → E) (hC : ∀ x, ‖C x‖ = ‖x‖) : C 0 = 0 := by
  have h := hC 0
  rwa [norm_zero, norm_eq_zero] at h

/-- **T5a (zero fixed point, corrected form).**
With `U 0 = 0` (and `C 0 = 0` derived from norm preservation),
zero is a one-step fixed point of the corrected recurrence. -/
theorem T5_zero_fixed_point (C U : E → E) (hC : ∀ x, ‖C x‖ = ‖x‖)
    (hU : U 0 = 0) (s : ℝ) :
    corrected C U s 0 = 0 := by
  simp [corrected, stack, carry_zero C hC, hU]

/-- The masked recurrence: only update positions whose squared deviation
exceeds the threshold `eps`. -/
def masked (C U : E → E) (s eps : ℝ) (x : E) : E :=
  if ‖x‖ ^ 2 > eps then corrected C U s x else x

omit [NormedSpace ℝ E] in
/-- For `eps ≥ 0`, zero never passes the mask: the anchor is frozen by
construction. -/
lemma mask_excludes_zero {eps : ℝ} (heps : 0 ≤ eps) :
    ¬ (‖(0 : E)‖ ^ 2 > eps) := by
  simp only [norm_zero]
  nlinarith

/-- **T5b (zero fixed point, masked recurrence — the paper's reading).**
For `eps ≥ 0` the mask itself freezes zero: `masked s eps 0 = 0` with NO
hypotheses on `C` or `U` at all. "The designated anchor is a one-step fixed
point by construction." -/
theorem T5_masked_zero_by_mask (C U : E → E) (s : ℝ) {eps : ℝ}
    (heps : 0 ≤ eps) :
    masked C U s eps 0 = 0 := by
  unfold masked
  rw [if_neg (mask_excludes_zero heps)]

/-- **T5c (zero fixed point, masked recurrence — either way).** Under the
model hypotheses (`‖C x‖ = ‖x‖`, `U 0 = 0`), zero is a fixed point of the
masked recurrence through EITHER branch, for EVERY `eps` (no sign condition):
if the mask freezes it we are done, and if it updates, T5a fixes it. -/
theorem T5_masked_zero_fixed_point (C U : E → E) (hC : ∀ x, ‖C x‖ = ‖x‖)
    (hU : U 0 = 0) (s eps : ℝ) :
    masked C U s eps 0 = 0 := by
  unfold masked
  split_ifs with h
  · exact T5_zero_fixed_point C U hC hU s
  · rfl

end SCSE
