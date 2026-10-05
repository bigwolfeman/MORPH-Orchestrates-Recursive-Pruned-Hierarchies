/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import TulInformation.SlotLoop

/-!
# P4: when a carrier takes the loop's job

## Model

`tul_information` (`Joint`, `mi`, `map`): finite alphabets, a deterministic map of the first
coordinate. Here the first coordinate is a PAIR. `ω` is the coda's view that does not depend
on the number of passes (its own span, and any carrier written from the loop's ENTRY rather
than its exit). `ε` is the loop's entry data. A pass is `F : ε → ε`; the loop state after `n`
passes is `F^[n]`.

## Main results

* **`bypass_zeroes_passes`** : if the pass-independent view is SUFFICIENT for the target
  (`I(view; Y) = I((view, entry); Y)`), then the loop's state after any passes adds exactly
  zero information beside it. A Bayes reader's K-curve is then flat.
* **`deeper_never_more_informative`** : beside any view, the state after `k + n` passes
  carries at most what the state after `n` passes carries. A positive K1−K6 is never
  information; it is always a bounded reader extracting better from the deeper state.
* **`raw_span_sufficient`** : `Joint.mi_relay_redundant` read for the carrier: a coda that sees
  the raw span and its other inputs gains nothing from ANY computed carrier. Raw copies of
  the span dominate every computed summary of it, once the carrier is wide enough.

Together with `tul_information`'s `mi_iterate_le`, these say where loop contribution can live
once a copy carrier exists: only on content that the coda's view does not make as easy to
extract as the deep loop state makes it.
-/

namespace TulPseudotoken

open TulInformation

variable {ω ε σ β : Type*} [Fintype ω] [Fintype ε] [Fintype σ] [Fintype β]
  [DecidableEq ω] [DecidableEq σ]

/-- **A sufficient bypass makes every pass worth zero.**  `P` is the joint law of
`((view, entry), target)`.  If the view alone carries what view and entry carry together,
then the view together with ANY function `F` of the entry (the loop state after any number
of passes) carries exactly what the view carries. -/
theorem bypass_zeroes_passes (P : Joint (ω × ε) β) (F : ε → σ)
    (hsuff : (P.map Prod.fst).mi = P.mi) :
    (P.map fun x => (x.1, F x.2)).mi = (P.map Prod.fst).mi := by
  apply le_antisymm
  · rw [hsuff]
    exact P.mi_map_le _
  · have hcomp : (Prod.fst : ω × σ → ω) ∘ (fun x : ω × ε => (x.1, F x.2)) = Prod.fst := rfl
    calc (P.map Prod.fst).mi = ((P.map fun x => (x.1, F x.2)).map Prod.fst).mi := by
          rw [Joint.map_comp, hcomp]
      _ ≤ (P.map fun x => (x.1, F x.2)).mi := Joint.mi_map_le _ _

/-- **Deeper is never more informative, beside any view.**  The state after `k + n` passes is
a function of the state after `n` passes, so with the same view beside it, it carries no more
about the target. -/
theorem deeper_never_more_informative [DecidableEq ε] (P : Joint (ω × ε) β) (F : ε → ε)
    (k n : ℕ) :
    (P.map fun x => (x.1, F^[k + n] x.2)).mi ≤ (P.map fun x => (x.1, F^[n] x.2)).mi := by
  have hcomp : (fun z : ω × ε => (z.1, F^[k] z.2)) ∘ (fun x : ω × ε => (x.1, F^[n] x.2))
      = fun x => (x.1, F^[k + n] x.2) := by
    funext x
    simp [Function.iterate_add_apply]
  rw [← hcomp, ← Joint.map_comp]
  exact Joint.mi_map_le _ _

/-- **Raw copies dominate computed summaries.**  A coda that reads the raw span `α` and its
other inputs `ν` (the earlier cells) gains exactly nothing from any carrier `g` computed from
them.  This is `Joint.mi_relay_redundant` of `tul_information`, restated for the carrier. -/
theorem raw_span_sufficient {α ν γ : Type*} [Fintype α] [Fintype ν] [Fintype γ]
    [DecidableEq α] [DecidableEq ν] [DecidableEq γ]
    (P : Joint (α × ν) β) (g : α × ν → γ) :
    (P.map fun x => (g x, x.1, x.2)).mi = P.mi :=
  Joint.mi_relay_redundant P g

end TulPseudotoken
