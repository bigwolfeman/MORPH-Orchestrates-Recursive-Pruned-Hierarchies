# Planned: horizon-indexed passes with a gated all-pass readout (LoopMTP port)

Status: failure

Date: 2026-09-14 (frozen before launch). Panel: `slot-job-panel-2026-09-13`
(`vlt thread slot-job-panel-2026-09-13`), builder `Builder-horizon`. Literature source:
`docs/references/looping-depth/2026-09-13-lit-mining/D_objective.md` §1 (LoopMTP,
arXiv 2608.03624) and the panel brief `00-brief.md`.

## Question

Every per-pass target this tree has tried supervises pass t against the SAME label
pass t-1 already had (a growing span, a frozen oracle trajectory, the exit's own
next-span CE) — eleven arms read a per-pass K-curve of ~0. LoopMTP gives pass t a
DIFFERENT target — the embedding of what is t steps ahead, not the same next-thing
restaged — and reports gains at MORPH's own parameter scale (260-280M). Ported to the
slot loop: does giving pass t (t=2..6) a horizon-indexed cosine target (span i+t's
tied-embedding mean) plus a content-conditional gated readout over all six passes (not
only the last) move the slot loop's per-pass K-curve off zero?

## Hypothesis

H-horizon: the horizon-indexed target differentiates the passes enough that later
passes read a nonzero, DIFFERENT-FROM-PASS-1 contribution — K3-K6 moves outside
[-0.0005, +0.0005] (twice the ruler's own CI half-width) and the T x T cosine matrix
(`horizon_pass_probe.py`) shows pass t predicting horizon t BETTER than pass 1 predicts
horizon t, off the diagonal-vs-row-1 comparison. H-horizon': the passes still meet the
target in one step (pass 1's row of the matrix is already close to the diagonal), the
same outcome every prior per-pass objective produced, because the map itself — not the
label — is what collapses depth here (the power-iteration / rank-collapse finding,
`morph-loop-is-a-power-iteration`).

## Method

Two arms, one factor apart, both one factor from the ruler:

| Arm | Config | wandb name | Factor vs ruler | Factor vs partner |
|---|---|---|---|---|
| ruler (partner) | `tul_slot_spandec_strict.yaml` | `slot-spandec-strict` | — | — |
| fixed6 (control) | `tul_slot_spandec_strict_fixed6.yaml` | `slot-spandec-strict-fixed6` | `slot_depth_fixed: 6` (Poisson → fixed) | — |
| horizon (arm) | `tul_slot_spandec_strict_horizon.yaml` | `slot-spandec-strict-horizon` | `slot_depth_fixed: 6`, `horizon_weight: 1.0`, `pass_readout: gated` | `horizon_weight` + `pass_readout` vs fixed6 |

`fixed6` exists because `horizon_weight`/`pass_readout="gated"` both REQUIRE
`slot_depth_fixed > 0` (`TULConfig.__post_init__` raises on the Poisson draw) — the
horizon arm is therefore TWO factors from the ruler at once (fixed depth AND the new
mechanism). `fixed6` isolates "does merely fixing T at the Poisson mean move anything",
so the horizon arm is read against `fixed6`, never against the Poisson-depth ruler
directly, for any per-pass or K-curve claim. Everything else (strict geometry, the span
decoder as the local objective, norm_match ternary, the gain constraint, the warmup
ramp, 5,000 steps, batch 6, seq 1024) is the shared panel recipe.

5,000-step run, both arms and the ruler scored on the SAME 480-row sweep the panel has
used throughout, at 2,500 and 5,000 steps. Commit pinned at launch.

## Readout

* **Loop contribution.** Token K1-K6 and K3-K6, `lab/divergence/core_depth_sweep.py`,
  paired bootstrap CI over the 480 rows — read against `fixed6`, not the ruler.
* **Does z_t predict horizon t better than z_1?** `lab/divergence/horizon_pass_probe.py`
  (new, this arm): loads a checkpoint, packs 48 validation rows, and reports the T x T
  matrix `M[t, h] = mean_slot cos(proj(readout(db_traj[t])), target_h)` — target_h the
  same detached mean tied-embedding `_tul_horizon_loss` trains against. Row 1 vs row t
  at a FIXED column h is the direct answer: does a later pass carry more of horizon h
  than pass 1 does.
* **Consecutive-pass cosine of the updates.** Same script: `cos(h_t - h_{t-1},
  h_{t+1} - h_t)`, mean over valid slots, one number per consecutive pair — the
  "do the passes cancel" reading every prior arm has shown at -0.2 to -0.6.
* **The gate's mean weight per pass.** Same script, `pass_readout="gated"` checkpoints
  only: `mean_slot g~_t` for t=1..6, read off the SAME computation the coda's write
  used, not an offline re-derivation.
* **CE.** Depth-6 CE on the 480-row sweep, reported as a horizon reading (the standing
  rule: a 5k CE cannot rank looped against unlooped, or one arm against another arm of
  a different objective — `short-horizon-ce-is-not-a-verdict`).

Partner numbers (the ruler, `slot-spandec-strict`, this panel's existing reference):
token K1-K6 +0.0016 [+0.0013, +0.0019], K3-K6 +0.0002, depth-6 CE 4.3474 on the 480-row
sweep, `all_slots` worth 0.1865 (`worth_profile`).

## Predictions (frozen)

* **P-1 (survival).** Both arms HEALTHY to 5,000 steps (no `preclip/total > 1e4` at
  step >= 200): **85%** each. The gated readout is a new real-forward path (unlike
  every prior per-pass arm, which was training-only), so it gets a lower prior than the
  panel's usual 90%+.
* **P-2 (fixed6 is a null factor).** `fixed6`'s K1-K6 stays within 2x the ruler's own CI
  half-width of the ruler's 0.0016 (i.e. within [+0.0010, +0.0022]), and its K3-K6 stays
  within [-0.0003, +0.0007] of the ruler's 0.0002: **70%**. `slot_depth_fixed` predates
  this arm and nothing else in the strict panel has read a fixed-vs-Poisson difference.
* **P-3 (the term trains and the CE reads worse before better, spandec_per_pass's
  precedent).** `horizon_weighted` is positive and finite at every logged step, and the
  horizon arm's depth-6 CE at 2,500 is WORSE than `fixed6`'s (a new objective's tax,
  matching `spandec_per_pass`/`oracle_z`'s own 2,500-step readings): **65%**.
* **P-4 (K-curve moves, the headline claim).** K3-K6 (horizon vs fixed6) moves outside
  [-0.0005, +0.0005]: **30%**. This is deliberately the LOW-prior claim — eleven arms
  have read ~0 here, and H-horizon' (met in one pass) is the modal prior from this
  panel's own history, not H-horizon.
* **P-5 (the matrix shows differentiation even if K3-K6 does not).** In the T x T
  matrix, at least one horizon h has `M[h, h] > M[1, h]` by more than 0.05 cosine,
  row 1 vs row h: **45%** — higher than P-4 because the matrix can show pass-specific
  representational content even when it changes the CODA's final CE by less than the
  K-curve's noise floor (the `spandec_per_pass`/DiscoLoop precedent: alignment and
  downstream-usefulness are separable facts).
* **P-6 (consecutive-pass cancellation weakens).** Mean consecutive-pass cosine (passes
  2-5, the four consecutive pairs among the six) is less negative than the ruler's
  historical -0.2 to -0.6 range (i.e. above -0.2 for at least one pair): **40%** — a
  differentiated target is the one mechanism in this literature pass with a stated
  reason to break the cancellation (LoopMTP Fig 2 bottom).
* **P-7 (the gate does not simply collapse onto pass 1).** Mean gate weight on pass 1
  at step 5,000 is below 0.90 (it starts near 0.96 at init by construction): **55%** —
  Wg escapes zero on the first backward, but nothing forces it away from the init bias.
* **P-8 (cost).** Wall-clock within 1.15x of `fixed6`'s (the arm adds one [C,C] linear
  training-only and one [C,C] linear in the real forward, no extra core pass, no extra
  vocab-sized readout): **75%**.

## Binding

* H-horizon (P-4 or P-5 clearly true, arm HEALTHY) ⇒ promote a decoder-free
  horizon-indexed target as a real lever; the next step is scaling `T` and re-testing
  under the paid-token variant `tul.loop_reads_tokens` (a different arm, not this one).
* H-horizon' (both P-4 and P-5 false, arm HEALTHY) ⇒ the twelfth arm with a per-pass
  K-curve of ~0. Files under `slot-loop-target-family-exhausted` if the panel's other
  concurrent arms (RecurTrace, DiscoLoop-style realignment, the relay arm) also close —
  Wolfe decides whether the family is done.
* A DETONATION on either arm (P-1 false) ⇒ file under the gain-constraint break-glass
  doc before touching the objective again; do NOT read K-curves off an unstable run.
* NO 20k run from this experiment. NO claim about the paid loop
  (`tokens_through_core`) — this arm never touches it.

## Not verified before launch

* No GPU forward of either config, on any size model, at this tree's queued commit —
  the compose-through-Hydra check (`reject_unknown_tul_keys`, the resolved `tul.*` and
  `model.*` block) ran; full model CONSTRUCTION at the real `d_model`/`n_core` size did
  not, to avoid adding CPU/GPU load on the shared machine during the mid-run CPU cap
  (`OMP_NUM_THREADS=2` etc., 2026-09-14).
* `lab/divergence/horizon_pass_probe.py` has NOT been run against a real checkpoint —
  none exists yet for this un-launched arm. Its cosine-matrix, consecutive-pass-cosine
  and gate-weight arithmetic were dry-run against the tiny CPU test fixture only
  (`tests/test_tul_horizon.py`'s fixture, not the script's own CLI path: no
  `_build.build_model`/`load_ckpt`/`create_dataloader` call has executed).
* Whether `training.ademamix_alpha_cap`/`t_beta3` (inherited from `tul_to_panel.yaml`,
  scaled for a 5,000-step run) interact with the new terms at all — untested, inherited
  as-is from the ruler.
* The interaction with `tul.core_token_aux`, `tul.slot_chain`, `tul.oracle_z`, or any
  other per-pass mechanism this panel has tried is UNDEFINED and untested; none is
  composed with this arm.

## Results

Both arms ran 2026-09-14, 5,000 steps, seed 1, `slot_depth_fixed` 6, `prefix_k` 2:
`horizon-fixed6` (control: no horizon term, `pass_readout: last`; 12,860 tok/s, one
gradient spike of 228 at step 1539, HEALTHY) and `horizon-arm` (`horizon_weight` 1.0,
`horizon_free_first`, `pass_readout: gated`; 12,753 tok/s, HEALTHY). Artifacts:
[`results/2026-09-14-horizon-passes/`](../results/2026-09-14-horizon-passes/).

Two instruments refused the arms as built and were fixed before any reading here was
taken: the gated readout refused every forced eval depth (f0c9fd7: eval reads
`beta[min(t, T-1)]`, the token-loop gate's convention; the sweeps re-ran on the Spark)
and `slot_state_probe` forced depth through the Poisson knob a k-fixed model ignores
(d1e93e3; both state probes re-ran on the Spark). `horizon_pass_probe.py`'s CLI ran
for the first time here (ef50a91, a device bug) and only on the horizon arm: the
control returns no trajectory under `pass_readout: last`, so it has no matrix.

**Own sweeps at 5,000 (480 rows), token CE:**

| arm | depth-6 CE | K1−K6 | K2−K6 | K3−K6 | worth `all_slots` |
|---|---|---|---|---|---|
| ruler `slot-spandec-strict` (Poisson 6) | 4.3474 | +0.0016 | | +0.0002 | 0.1865 |
| `horizon-fixed6` | 4.3430 | +0.278 (OOD: never trained below 6; span-first 5.99 at depth 1) | +0.030 | +0.0086 | 0.1898 |
| `horizon-arm` | 4.3429 | +0.0040 [+0.0036, +0.0044] | +0.0005 | +0.0002 [+0.0001, +0.0003] | 0.1836 |

At 2,500: `horizon-arm` 4.6468 (K1−K6 +0.0012, K3−K6 −0.0002), `horizon-fixed6` 4.6507.

**Paired (480 identical rows):** horizon − fixed6 at depth 6: **−0.0001 [−0.0023,
+0.0025]** at 5,000, −0.0039 [−0.0061, −0.0016] at 2,500; horizon − ruler at depth 6:
−0.0045 [−0.0071, −0.0018] (the control reads the same −0.0044, so that is the fixed
draw, not the targets).

**State probes (12 rows, `prefix_project` input, Spark, d1e93e3):** `fixed6` |h| 747 →
1388 → 1951 → 2039 at depths 1/2/3/6, cosine to the depth-1 state 0.49 / 0.29 / 0.24;
`horizon-arm` |h| 340 → 390 → 392 → 391, cosine 0.90 / 0.89 / 0.88. (The Poisson family:
`trajrep` |h| 88 → 98, cosine 0.98.)

**Pass probe on `horizon-arm` (48 rows, T=6, the learned projection):** the cosine
matrix M[pass, horizon] is 0.603 / 0.595 / … / 0.592 on pass 1 and 0.645 / 0.642 /
0.641 / 0.640 / 0.639 / 0.638 on EVERY pass 2–6; M[h,h] − M[1,h] = 0.046–0.047 for every
h. Consecutive-pass cosine (1→2 … 5→6): 0.43, 0.13, 0.81, 0.97, 0.95. Gate mean weight
per pass: 0.832, 0.036, 0.028, 0.027, 0.028, 0.028. The training term fell 1.00 → 0.35
over 5,000 steps, positive and finite at every logged step.

**Scores.**

- **P-1 holds.** Both HEALTHY.
- **P-2 fails.** `fixed6`'s K1−K6 (+0.278) and K3−K6 (+0.0086) are far outside the
  bands, because a model trained only at depth 6 is out of distribution at every other
  eval depth. Its depth-6 CE is within 0.005 of the ruler.
- **P-3 splits.** The term is positive and finite throughout (holds); the horizon arm's
  depth-6 CE at 2,500 is BETTER than the control's by 0.004, not worse (fails).
- **P-4 passes the letter and fails the meaning.** K3−K6 (horizon − fixed6) = −0.0084 is
  outside [−0.0005, +0.0005], but the movement is the control's OOD inflation; the horizon
  arm's own K3−K6 is +0.0002, the ruler's number.
- **P-5 fails** (0.047 < 0.05), and the matrix says why: the six horizon targets are
  within 0.01 cosine of each other for every pass, so "a different job per pass" was
  never posed. A mean-pooled tied embedding of span i+t barely depends on t on web text.
- **P-6 holds.** No consecutive pair below −0.2; passes 3–6 are near-identical states
  (0.81, 0.97, 0.95) and the 2→3 pair is near orthogonal (0.13).
- **P-7 holds.** Pass-1 weight 0.832 < 0.90; the other five passes share 0.17.
- **P-8 holds.** 1.008x the control's rate.

## Verdict

**Failure.** The LoopMTP port to the slot loop buys nothing: paired CE against its
fixed-depth control is −0.0001 ± 0.0024 at 5,000, the K-curve past pass 3 is the
ruler's +0.0002, and the gate reads 83 % from pass 1. The instrument the arm was built
to move (P-5) shows the horizon targets are degenerate on this corpus, so the arm never
gave the passes different jobs. The one thing the fixed draw does is turn the loop into
a growing chain (norm x2.7, exit at cosine 0.24 to pass 1) that the gate then tames to
0.88; the coda uses neither.

## Updated hypothesis

Horizon-indexed targets made of span-mean tied embeddings do not distinguish horizons
on web text and cannot give passes different jobs; any future per-pass target must be
shown non-degenerate (M's columns separated) before an arm is queued. The fixed depth
draw is a real change to the loop's dynamics (a growing chain) with no CE or depth
consequence at 5k, which is one more instrument moving while the coda does not pay.
The token-loop LoopMTP rungs (`2026-09-14-arc-loopmtp-token-loop.md`) are the ship
question and are unaffected by this reading.
