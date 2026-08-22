# Agent Note: Db B1 L2 Pathfinder

Status: implemented
Archived: 2026-08-21

Origin: Ai-notes/08-19-2026/DB-B1-L2-Pathfinder/RESULT.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# db_b1_l2 pathfinder — result (2026-08-19)

**Arm:** `db_b1_l2` (Option A: faithful L2 embedding denoising, x0_inject, b1, batch14/seq1024, 2000 steps).
Branch `feat/db-objective-l2`. wandb run `db-b1-l2` (id 3qfmy6gt). ckpt `checkpoints/morph/db-b1-l2/step_2000.pt`.

## Verdict: FAILS the context gate (autoencoding), but the objective fix is validated.

Falsifier (`ignore/db-concat-run/l2_freeride_probe.py`, matched vs scrambled context, 8 batches):

| σ | matched raw L2 | scrambled raw L2 | gap | matched wL2 |
|---|---|---|---|---|
| 0.05 | 0.00112 | 0.00112 | 0.00000 | 0.4537 |
| 0.30 | 0.01827 | 0.01828 | 0.00000 | 0.2761 |
| 1.00 | 0.05472 | 0.05475 | 0.00003 | 0.2736 |
| 3.00 | 0.11821 | 0.12065 | 0.00244 | 0.4860 |
| 9.79 | 0.19949 | 0.20401 | 0.00452 | 0.8001 |

Chance raw L2 = 0.25. Pre-registered pass = gap > 0.3 AND matched clearly below scrambled at high σ.

## Two separable findings

1. **Free ride FIXED (win).** `matched wL2` is O(1) at every σ — no σ decodes for free. The L2
   objective (no readout in training) removes the CE+tied-head free ride by construction, as designed.
2. **x0_inject conditioning too weak (blocker).** matched ≈ scrambled at every σ (gap < 0.005). The
   model autoencodes + predicts position-marginals; it does NOT use context. A faint gap grows with σ
   (0 → 0.0045), so the additive channel is near-dead, not fully dead.

## Correction to the checklist §3 claim

Earlier claim: "x0_inject's autoencoding was the free ride, not a conditioning problem." FALSIFIED.
With the free ride removed (L2), x0_inject STILL autoencodes → weak additive conditioning is a
SEPARATE, real problem. The concat rebuild targets it correctly. Update checklist §3 last bullet.

## Loss curve (do not trust as a language signal)

Train weighted-L2: 1.69 (0) → 0.75 (400) → 0.45 (1000) → falling. This drop is NOT evidence of
context use — it is the low-σ copy + marginal-mean regression improving. `ppl` column in the train
log is `exp(L2)`, a meaningless display bug for loss_kind=l2 (fix pending).

## VRAM note (measured)

x0_inject DB forward does NOT gradient-checkpoint (`transformer.py:1124` else-branch runs `layer(x)`
bare, ignoring the `ckpt` flag the concat path + core loop respect). Activations dominate:
params 1.2 GB, optimizer 2.4 GB, activations 5.7 GB @ batch2 → ~40 GB @ batch14 (eager; compiled fit
26 GB). HC Cayley n=4 makes each residual tensor 4× wide. Run needs
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at batch 14. Fix: wrap the else-branch in
`checkpoint()` like its siblings (pending Wolfe's OK — gradient checkpointing).

## Next

Phase 2: concat / two-source attention (context via attention at every layer, not one additive inject).
This is the pre-registered next step and is now motivated by data.

## AR generation test (2026-08-19) — GIBBERISH, confirms loss is misleading

`ignore/db-b1-l2-run/gen_test.py` (greedy, 4 Euler steps, feeds generated tokens back as x0):

    'The history of the Roman Empire the\nousal, under dileg International al0 he 2, of it a by on and...'

Word salad of high-frequency tokens (the/of/and/in/a/numbers), no grammar, incoherent from
token 1. Confirms: the 0.45 L2 loss / val "10" have ZERO correlation with generation quality
(this is why the paper uses MAUVE). The model is a marginal/position predictor.

WORSE THAN EXPOSURE BIAS: incoherent immediately after a clean prompt → the base conditional
next-token model was never learned; exposure bias is secondary. Two root causes, both HIGH-σ:
  1. x0_inject conditioning too weak (falsifier: matched≈scrambled).
  2. σ distribution oversamples the trivial low-σ regime (median 0.30, 84% <1.0), starving the
     high-σ "predict from noise" skill. AR generation starts at z_0=σ_max·ε (pure noise) → lives
     entirely in the high-σ regime the model barely trained → gibberish.

IMPLICATION: Phase 2 (concat) fixes #1 (conditioning) but NOT #2 (σ mismatch). Need both:
reshape σ sampling toward high σ AND deliver context via attention.

CORRECTION (2026-08-19, later): the "26GB→10.6GB, no-BPTT win restored" claim above is WRONG —
it was an EAGER probe. The real COMPILED training path still peaks ~26 GB (measured 26071 MiB /
21566 process). The win is NOT restored. See the VRAM section of the FIX RUN below.

---

# FIX RUN — db-b1-l2-fix (detach + CE anchor λ=1.0 + σ-reshape) — 2026-08-19 23:00

Config (`db_b1_l2.yaml`, wandb.name=db-b1-l2-fix): x0_inject, loss_kind=l2, edm weighting.
THREE changes vs the buggy run: (1) **target detached ALWAYS** in `db_setup` (the L2 target is
DATA — the un-detached target had let the loss train the tied embedding by moving TARGETS toward
predictions → directional collapse); (2) `ce_anchor_lambda=1.0` (Diffusion-LM rounding term
λ·CE(D̂@E.T, labels), fused, no [N,V]); (3) σ-reshape `p_mean=0.0 p_std=1.6` (median σ 0.30→1.0,
P(σ>2) 5.7%→~33%). Checkpoint: `checkpoints/morph/db-b1-l2-fix/step_2000.pt`.

Training loss fell cleanly: anchored total 7.41(0) → 3.00(200) → 1.53(1800); `db/anchor_ce`
7→1.27. So the never-before-run λ>0 anchor path executes and learns. That is the ONLY positive.

## OBJECTIVE VERDICT — STILL AUTOENCODING (measured, negative)

Falsifier (`l2_freeride_probe.py`, 8 batches, on the CORRECT `-fix` checkpoint — the first attempt
scored the OLD `db-b1-l2/` checkpoint by default-path; caught and re-run):

    sigma | matched raw | scrambled raw |      gap | verdict
    1.000 |     0.14937 |       0.15128 |  0.00190 | autoencoding/at-chance
    3.000 |     0.29642 |       0.32060 |  0.02419 | autoencoding/at-chance
    9.790 |     0.37122 |       0.38292 |  0.01170 | autoencoding/at-chance

The matched-vs-scrambled gap stays ~0 at EVERY high σ. The model does not use the x0 context to
predict the next-token embedding — even at σ=9.79 where z is nearly pure noise and context is the
ONLY signal. detach + anchor + σ-reshape did NOT fix it.

Generation (`gen_test.py`, greedy, 4 Euler steps): still token salad —
`"The history of the Roman Empire view because LRA (~),ative through discount x…"`. Incoherent
from token 1. Behaviorally confirms the falsifier.

## ROOT CAUSE (revised) — the shortcut is STRUCTURAL to x0_inject

The noisy input z_t IS a noised copy of the target y. Denoising z_t→y is easier than using x0
context, so the model shortcuts through it. The CE anchor free-rides the SAME shortcut: at low σ,
D̂≈y is recovered from z_t alone, so CE(D̂@E.T, labels)→0 without any context. σ-reshape reduced
but did not remove the low-σ mass (median σ=1.0 ⇒ half the mass below 1, where z_t leaks the
target), so the shortcut still dominates the gradient. **Objective tweaks cannot close a
STRUCTURAL leak** — x0_inject puts the target (as z_t) in the same stream the model reads, so the
model can always ignore x0. This is exactly what the checklist Phase-2 concat design prevents: the
clean context is a SEPARATE causally-masked stream that withholds y from the query. Conclusion:
Phase 1 (prove-on-x0_inject) has reached its limit — x0_inject cannot do conditional prediction by
construction. Move to Phase 2 (concat / two-source attention).

## VRAM VERDICT — win IS realized in LIVE memory; the "blowup" is RESERVED (measured, 4-cell probe)

`vram_probe.sh` (b9ord8dcx), real train.py, db_b1_l2, batch 14, expandable_segments=on:

    cell            compile ckpt | allocated(live) reserved
    eager_nockpt    off     off  | OOM (>31 GB)    —
    eager_ckpt      off     on   | 10.22 GB        10.61 GB
    compiled_nockpt on      off  | OOM (>31 GB)    —
    compiled_ckpt   on      on   | 10.17 GB        20.44 GB   <-- real training path

**My "compile defeats the checkpoint" hypothesis is REFUTED.** The checkpoint WORKS under compile:
live activation memory is 10.17 GB, identical to eager (10.22). The recurrent-depth no-BPTT win IS
realized at the tensor level — 10 GB with ckpt vs OOM (>31 GB) without, in BOTH eager and compiled.
Checkpointing is load-bearing and functioning.

What we saw as a "26 GB blowup" is RESERVED memory, not live tensors: compile's caching allocator
reserves ~20 GB while only ~10 GB is live (≈2× slack — Inductor workspace + the dynamic-batch core-
MLP compiled variants). `nvidia-smi`/process memory reports RESERVED, which is why it looked like no
win. The earlier 26071 MiB (no expandable_segments) vs this 20.44 GB (with it) is the fragmentation
difference; live allocated is ~10 GB either way.

Practical: (1) the memory mechanism is CORRECT — do not "fix" the checkpoint, it isn't broken;
(2) keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for DB runs (drops reserved 26→20 GB);
(3) if reserved slack matters for batch headroom, the lever is the compile config (the core MLPs
compile with dynamic=True → multiple variants each holding workspace), NOT the checkpoint. The
3-4× no-BPTT win vs an AdEMAMix+BPTT loop is real in live memory; the reserved footprint is a
softer, allocator-level concern.

---

# CONCAT VERDICT + THE REAL DIAGNOSIS (2026-08-20) — it was never the conditioning

db_b1_concat_l2 = db_b1_l2 with ONLY conditioning x0_inject→concat. Three lenses, all agree
concat ≈ x0_inject (NO improvement):

  lens                      x0_inject (db-b1-l2-fix)   concat (db-b1-concat-l2)
  falsifier gap @ σ=9.79    0.01170                    0.01163
  gen-PPL (gpt2 ref)        ~50k (18k–78k)             45,832 (English baseline = 13)
  shape model rawL2 @9.79   0.371                      0.371

## Wolfe's leak hypothesis: RULED OUT (3 independent proofs)

Wolfe read the falsifier SHAPE (near-0 L2 at low σ) as answer-leakage / causal-mask off.
Investigated step by step (leak_watcher.py, shape_decomp.py):
1. Isolated-mask probe (leak_probe.py): perturb clean target → 0.0 movement, both branches;
   n_skip_rope=2 positive control leaks +16 (probe can detect leaks).
2. FULL-FORWARD watcher (leak_watcher.py): flip input_ids[p], hold z fixed, measure Δ D̂[p-1].
   x0_inject = EXACTLY 0.0; concat = ~1e-5 (bf16 noise) vs reachability Δ D̂[p]=~0.4. No leak
   through CCA conv / value-shift / CoPE / GLA / two-source attention.
He was RIGHT that "the answer is in the prediction" — but via z, not a mask (see below).

## The real diagnosis: EDM z-identity + regression-to-mean on embedding-space L2

shape_decomp.py, per σ: model rawL2 vs F=0 identity (c_skip·z) vs chance (σ_d²=0.25). IDENTICAL
for x0_inject and concat:

  σ      model   identity(F=0)  chance   reading
  0.05   0.0030  0.0025         0.25     model≈identity → F adds nothing (z passes through)
  0.30   0.059   0.066          0.25     model≈identity
  1.00   0.150   0.200          0.25     model beats chance (narrow real-prediction band)
  3.00   0.289   0.243          0.25     model WORSE than chance → F HURTS
  9.79   0.371   0.249          0.25     model WORSE than chance → F HURTS badly

Low σ: D̂ = c_skip·z ≈ y (z = y+tiny·ε) — the target reaches the prediction through z itself
(EDM identity), NOT learned. This is the "shape looks like leak" — same symptom, structural cause.
High σ (the regime AR generation STARTS in, z_0=σ_max·ε): model is WORSE than a zero predictor.
That is the gibberish. Mechanism: embedding-space L2 is a regression problem; the L2-optimal
prediction of a many-valid-next-token target is E[y|context] = a MEAN embedding = not a decodable
token → high raw L2 AND garbage decode. Intrinsic to L2-in-embedding-space; NOT fixable by
conditioning (proven: x0≡concat) or by removing a leak (proven: none).

## Reframing — REVERSES the Option-A-over-Option-B call

We chose Option A (L2) to kill the low-σ tied-head CE free ride. But L2 has a WORSE, intrinsic
failure: high-σ regression-to-mean → worse than chance → gibberish from the generation start
point. CE (Option B) predicts a DISTRIBUTION: at high σ it can output high-entropy/near-uniform
(the CORRECT "many tokens possible"), degrading GRACEFULLY to ln V instead of to a garbage mean.
Next arm: CE (distributional) objective, conditioning irrelevant so use the cheap x0_inject +
σ-reshape + unweighted CE, and read the shape (CE at high σ should approach ln V, not exceed it).
UNVERIFIED: that CE actually escapes the high-σ collapse — that is the next experiment.
