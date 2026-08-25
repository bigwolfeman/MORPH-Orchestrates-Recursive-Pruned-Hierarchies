# Proving the SCSE residual correction

Status: PROVED and machine-checked. Lean 4 + Mathlib, `Scse.lean`, 2026-08-25.

## The problem this exists to solve

SCSE (arXiv:2607.27656) defines its looped recurrence in deviation coordinates as

    Delta_{t+1} = Delta_t + s * G(Delta_t)          s = 0.5, the "residual step scale"

The paper's `G` carries **no top-level identity**. Its block residual has been hoisted out to
loop level — that hoisting *is* the `+ s * G(...)`.

MORPH's core block stack carries its **own** residual (the HyperConnection Cayley carry), so
feeding `stack(D)` straight into that formula applies a residual twice. The first version of
this port did exactly that and gained ~1.41x per loop iteration, about 16x over eight — the
opposite of what the method is for. It was caught by audit, not by us.

The repair was `G(D) := stack(D) - D`, chosen by inspection and measurement. That is the
problem: the repair put us on a boundary the paper's theory does not cover, justified only by
a numerical argument. This directory closes that gap with a proof.

## What is proved

Abstractly, model the core as `stack x = C x + U x`, where

* `C` is the carry, assumed **norm-preserving** (`‖C x‖ = ‖x‖`) and **NOT assumed linear** —
  our Cayley mixer is exactly orthogonal but input-dependent, so linearity is unavailable;
* `U` is the branch (attention + MLP writes), with `U 0 = 0`.

| | statement | meaning |
| --- | --- | --- |
| **T1** | `‖corrected s x‖ <= ‖x‖ + s*‖U x‖` for `0 <= s <= 1` | the carry contributes **nothing** to growth |
| **T2** | `‖corrected^[T] D0‖ <= ‖D0‖ + s*T*B` given `‖U‖ <= B` | growth is at most **linear** in loop depth |
| **T3** | for `C = id, U = 0`: `‖naive^[T] D0‖ = (1+s)^T * ‖D0‖`, unbounded for every `s > 0`, and the T2 schema is explicitly refuted | the form we first shipped has **no** such guarantee |
| **T4** | `C = id` implies `corrected s x = paper s x` with `G = U` | our form is a strict **generalisation** that collapses onto the published algorithm exactly when the carry is a plain identity |
| **T5** | zero is a fixed point, masked and unmasked; `C 0 = 0` is **derived** from norm preservation | the paper's "the anchor is a one-step fixed point by construction" |

**T4 is the one that resolves the boundary.** It turns "our adaptation has an equivalent
effect" from an assertion into a theorem. **T3 turns the bug into a proved separation** rather
than a measurement.

## Verify it yourself

    cd lab/scse-lean
    lake build Scse                 # -> exit 0, "Build completed successfully (8559 jobs)"
    grep -c '\bsorry\b' Scse.lean   # -> 0

Axiom audit — every theorem must depend only on the three standard Mathlib axioms:

    echo 'import Scse
    open SCSE
    #print axioms SCSE.T1_one_step_bound' > AxCheck.lean && lake env lean AxCheck.lean

All nine theorems return `[propext, Classical.choice, Quot.sound]`. No `sorryAx`, no custom
axioms. Run and confirmed 2026-08-25.

`.lake/` (3.7 GB of Mathlib oleans) is gitignored; restore with `lake exe cache get`.
Build parallelism was capped (`LEAN_NUM_THREADS=4`) because a GPU campaign was live and the
UPS is marginal — do the same.

## What this does NOT license

The theorem is about the abstraction, not the network. Stated plainly so nobody over-reads it:

1. **bf16 breaks exact norm preservation.** A per-step norm error `d` turns T2 into
   `‖D_T‖ <= ‖D0‖ + sT(B + d)` — still linear, so the qualitative claim survives, but the
   perturbed version is not formalised.
2. **The real stack is only additively `C + U` if no normalisation sits outside the HC
   residual.** Where it does, `U` absorbs the difference and T2's `B` bounds that residue,
   which nobody has measured.
3. **`‖U‖ <= B` is a global bound and real attention/MLP branches do not satisfy it.** The
   honest reading is local: on any trajectory staying in a region where `‖U‖ <= B`, growth is
   linear in `T`.
4. **T3 is a separation, not a prediction.** It shows the naive form has no theory-side
   guarantee. A trained network might keep naive iterates bounded anyway; nothing protects it.
5. **It does NOT say SCSE works on MORPH.** The 2026-08-25 campaign says it does not — seeds 1
   and 2 give +1.74 and +1.64 nats against the control, with training loss flat from step 200.
   See `docs/experiments/`. A proof cannot dismiss a measurement.

## What it DOES buy, and it is pointed

The carry cannot be the source of expansion. So **if the deviation blows up, it must come
through `U`.** The measured stall agrees: core MLP `sigma_max` runs 1.44 -> 6.04 on the SCSE
arm against 1.44 -> 3.03 on the control, and `sigma_max` is a statement about `U`. Theory and
measurement point at the same place, which is what makes this worth having: it narrows the
open question from "why does SCSE stall" to "why does the branch term grow".
