/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import TulExploration.Hops

/-!
# P3: what a raw-context teacher can and cannot add

## Model

The orchestrator's seed: the "explicit trace" of CODI already exists in LXTUL for free, as
the raw past tokens. A coda that reads raw cross-span context is the TEACHER; the coda that
reads only the carrier is the STUDENT. Two questions decide whether that teacher is worth its
cost.

1. Does it change WHAT the student should carry, or only how fast it learns? Model: a raw
   context `x` with law `P x`, a target `y` with conditional law `Q x y`, and a student loss
   `ℓ x y` (its CE `-log q(y | carrier(x))`, or any one coordinate of its gradient). CE trains
   on the sampled `y`; LOGIT self-distillation trains on the teacher's law `Q x ·`.
2. Can it make the loop's depth necessary? Model: `TulExploration.Prog`, an adaptive lookup
   program over a context table (T4 of `tul_exploration`). The teacher is a program `p` that
   reads the raw context (the table `f`) with the coda's fixed path of lookups.

## Main results

* **`kd_mean_eq`** : logit self-distillation from a calibrated teacher has exactly the CE
  objective in expectation, so it has the same optimum.
* **`kd_variance_le`** : its variance around the common mean is never larger than CE's.
  Logit distillation is a variance reduction (Rao-Blackwell), nothing more.
* **`transcript_replay`** : a student that holds the teacher's lookup TRANSCRIPT (the table
  entries the teacher read, `cost f p` of them) reproduces the teacher's output exactly on
  every table, with zero loop passes. A same-depth teacher only asks the carrier to HOLD
  entries, never to COMPUTE: it cannot make the loop's depth necessary.
-/

namespace TulPseudotoken

open Finset

section LogitKD

variable {χ υ : Type*} [Fintype χ] [Fintype υ]

/-- **Logit distillation is CE in expectation.**  `∑ₓ P x ∑_y Q x y ℓ x y` is the expected
loss under the teacher's soft labels; it equals the expected one-hot loss when the teacher's
law is the data's conditional law.  Same objective, same optimum. -/
theorem kd_mean_eq (P : χ → ℝ) (Q : χ → υ → ℝ) (ℓ : χ → υ → ℝ) :
    ∑ x, P x * ∑ y, Q x y * ℓ x y = ∑ x, ∑ y, P x * Q x y * ℓ x y := by
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun y _ => ?_
  ring

/-- Weighted Jensen for the square: `(∑ q a)² ≤ ∑ q a²` for a probability vector `q`. -/
lemma sq_wsum_le {q a : υ → ℝ} (hq : ∀ y, 0 ≤ q y) (hq1 : ∑ y, q y = 1) :
    (∑ y, q y * a y) ^ 2 ≤ ∑ y, q y * a y ^ 2 := by
  set m := ∑ y, q y * a y with hm
  have h0 : 0 ≤ ∑ y, q y * (a y - m) ^ 2 :=
    Finset.sum_nonneg fun y _ => mul_nonneg (hq y) (sq_nonneg _)
  have hexp : ∑ y, q y * (a y - m) ^ 2
      = ∑ y, q y * a y ^ 2 - 2 * m * ∑ y, q y * a y + m ^ 2 * ∑ y, q y := by
    have hy : ∀ y, q y * (a y - m) ^ 2 = q y * a y ^ 2 - 2 * m * (q y * a y) + m ^ 2 * q y :=
      fun y => by ring
    simp_rw [hy, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [hexp, hq1, ← hm] at h0
  nlinarith

/-- **Logit distillation never raises the variance.**  Around any centre `μ` (take the
common mean of `kd_mean_eq`), the spread of the teacher-target loss is at most the spread of
the one-hot loss.  `ℓ` may be the loss or any gradient coordinate, so the statement covers
the stochastic gradient too. -/
theorem kd_variance_le (P : χ → ℝ) (Q : χ → υ → ℝ) (ℓ : χ → υ → ℝ) (hP : ∀ x, 0 ≤ P x)
    (hQ : ∀ x y, 0 ≤ Q x y) (hQ1 : ∀ x, ∑ y, Q x y = 1) (μ : ℝ) :
    ∑ x, P x * (∑ y, Q x y * ℓ x y - μ) ^ 2 ≤ ∑ x, ∑ y, P x * Q x y * (ℓ x y - μ) ^ 2 := by
  refine Finset.sum_le_sum fun x _ => ?_
  have hc : ∑ y, Q x y * ℓ x y - μ = ∑ y, Q x y * (ℓ x y - μ) := by
    have : ∑ y, Q x y * (ℓ x y - μ) = ∑ y, Q x y * ℓ x y - μ * ∑ y, Q x y := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun y _ => ?_
      ring
    rw [this, hQ1 x, mul_one]
  rw [hc]
  have hj := sq_wsum_le (a := fun y => ℓ x y - μ) (hQ x) (hQ1 x)
  have : ∑ y, P x * Q x y * (ℓ x y - μ) ^ 2 = P x * ∑ y, Q x y * (ℓ x y - μ) ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun y _ => ?_
    ring
  rw [this]
  exact mul_le_mul_of_nonneg_left hj (hP x)

end LogitKD

section Transcript

open TulExploration TulExploration.Prog

/-- The table a student sees: the teacher's entries on the nodes in `A`, anything else
(`g`) off them. -/
def patch {V : Type} [DecidableEq V] (f g : V → V) (A : List V) : V → V :=
  fun x => if x ∈ A then f x else g x

/-- **A same-depth teacher asks only for a transcript.**  Run the teacher program `p` on the
raw-context table `f`.  A student whose carrier holds the `cost f p` entries the teacher read
(`asked f p`), and knows nothing else (`g` is arbitrary), runs the same program and gets the
teacher's output exactly.  No loop pass is involved: whatever the teacher computes over the
raw context with the coda's fixed path, the carrier only has to HOLD, not compute. -/
theorem transcript_replay {V : Type} [DecidableEq V] (f g : V → V) (p : Prog V) :
    eval (patch f g (asked f p)) p = eval f p ∧ (asked f p).length = cost f p :=
  ⟨(run_congr f (patch f g (asked f p)) p fun x hx => by simp [patch, hx]).1,
    length_asked f p⟩

/-- **The carrier a distilled student needs is no larger than the teacher's depth.**  If
the teacher makes at most `Lc` lookups (the coda's fixed path) then a carrier of at most `Lc`
table entries reproduces its output, on every table, with zero passes. -/
theorem distilled_carrier_le {V : Type} [DecidableEq V] (p : Prog V) (Lc : ℕ)
    (hcost : ∀ f : V → V, cost f p ≤ Lc) (f g : V → V) :
    ∃ A : List V, A.length ≤ Lc ∧ eval (patch f g A) p = eval f p :=
  ⟨asked f p, (transcript_replay f g p).2 ▸ hcost f, (transcript_replay f g p).1⟩

end Transcript

end TulPseudotoken
