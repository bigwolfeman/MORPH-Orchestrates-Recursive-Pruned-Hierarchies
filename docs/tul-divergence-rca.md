# TUL v1 divergence — root cause analysis (2026-08-17)

Both TUL arms of the first short-schedule pass diverged. `A0` (dense baseline) and
`A3` (`n_core 0`, prelude+coda only) completed 20 000 steps. This note records what
is **measured**, what is **inferred**, and the experiments that separate the two.

Everything below is CPU-only forensics on the saved checkpoints and the wandb
history. No GPU was available while writing it, so no hypothesis here has been
tested by a training run.

## 1. What happened

| arm | config | result | final val loss |
|---|---|---|---|
| A0 | `tul_a0` | completed 20 000 | 3.2736 (ppl 26.41) |
| A3 | `tul_a3` | completed 20 000 | 3.2407 (ppl 25.55) |
| A1 | `tul_a1` | **ABORT step 4540** | — (val 6.409 @4500) |
| A1r | `tul_a1r` (seed 1) | **ABORT step 3240** | — (val 6.674 @3000) |

The divergence guard fired on both TUL arms: two consecutive eval points with train
ppl over 1000. Forensic checkpoints `DIVERGED_step_4540.pt` / `DIVERGED_step_3240.pt`
hold the failed state.

A1's val curve is *not* a slow drift. It improves through step 1500 (4.847, better
than A0 at 500 and 1000) and then climbs monotonically to 6.409.

Two seeds fail the same way. This is structural, not seed luck.

## 2. The decisive number

`val/ppl_tokens` — token positions only, directly comparable to the baseline:

* A3, prelude+coda with **no core at all**: **25.55**
* A1, prelude+coda **plus** the slot core: **754.67** at step 4500 and rising

A1's token path is the same 8 layers as A3 plus slot positions in the sequence. It
ends 30× worse than deleting the core entirely. The slot core is not failing to
help — it is actively destroying the token path it feeds.

## 3. Where the damage is — measured

`_cms.block_score_ema` is an EMA of per-tile ‖W ⊙ ∇W‖_F
(`morph/model/layers/block_sparse.py:480-483`), recorded per module inside every
checkpoint. It is a gradient magnitude, stored by name. At (near) iso-step:

| module | A1 @4540 | A1r @3240 | A0 @5000 | A3 @5000 |
|---|---|---|---|---|
| prelude.0 mlp.down | 2.2e7 | 1.3e8 | 2.89 | 6.71 |
| prelude.3 mlp.down | 2.8e6 | 8.2e7 | 1.68 | 3.14 |
| **core.0 mlp.down** | **1.1e8** | **3.5e8** | **0.976** | — |
| core.1 mlp.down | 3.1e7 | 1.7e7 | 0.998 | — |
| core.2 mlp.down | 3.1e6 | 1.2e7 | 0.913 | — |
| core.3 mlp.down | 1.8e6 | 4.1e6 | 1.10 | — |
| core.4 mlp.down | 2.5e5 | 1.9e6 | 1.17 | — |
| core.5 mlp.down | 3.8e4 | 2.4e5 | 1.06 | — |
| coda.0 mlp.down | 15.9 | 9.71 | 2.58 | 3.35 |
| coda.3 mlp.down | 12.4 | 7.05 | 3.85 | 4.63 |

Read it backward, the direction gradient flows: **coda ≈ 5× the baseline, then the
core amplifies ≈ 5× per layer** (core.5 → core.0 is 3.8e4 → 1.1e8, a factor of 3000
across six layers), and the flood continues into the prelude at 1e7.

A0's core profile over the same six layers is **flat**: 1.06, 1.17, 1.10, 0.91,
1.00, 0.98. Same weights, same depth schedule, same optimiser. The only thing that
changed is what the loop runs on.

**Ruled out — quantiser artefact.** The Taylor score is `|W ⊙ ∇W|` with `W` the
*effective ternary* weight, so an inflated quantiser scale would fake this. It did
not: the core.0 shadow weights are `0.0163` (A1) against `0.0219` (A0) — smaller,
not larger. The 1e8 is gradient.

## 4. What the explosion did to the rest of the model — measured

`training.grad_clip = 1.0` was active for every arm. With a global norm of order
1e8, the clip factor is of order 1e-8 on every step of both TUL arms.

Comparing A1 @4540 against A0 @5000, every parameter region matches within 1.12×
**except** the injection-scale parameters:

| parameter | A1 | A1r | A0 | A3 |
|---|---|---|---|---|
| `embed.hybrid.euc_embed` (token table) | 0.0379 | 0.0378 | 0.0403 | 0.0399 |
| `embed.bigram.lambdas` | **0.107** | **0.149** | 0.771 | 0.672 |
| `x0_injects.1.log_scale` | **0.100** | **0.124** | 0.575 | 0.701 |
| `x0_injects.2.log_scale` | **0.149** | **0.146** | 0.579 | 0.695 |
| `tul.W_prefix` ‖W−I‖_F | 7.36 | 6.66 | — | — |
| `tul.E_mask` rms | 0.0024 | — | — | — |

The token embedding table is **fine** (within 6 %). What stalled is the family of
small, slow-growing scale parameters that control how strongly `x0` and the bigram
signal are injected into each block. A0 and A3 both grow them to ≈0.6–0.7; both TUL
arms sit at ≈0.10–0.15.

`W_prefix` barely moved from its identity init (diagonal 0.995, off-diagonal rms
0.007), so the prefix projection is not the amplifier. `E_mask` never left zero.

**Inferred, not measured:** that the sustained ~1e-8 clip is what stalled those
scales. Adam-family updates are invariant to a *uniform* rescale, so the mechanism
has to be a floor effect — the optimiser epsilon, or `ademamix_g_snr_gate_kappa`
— and neither was instrumented. `optim/snr_gate_*` never appeared in either run's
history, which argues against the SNR gate but does not settle it.

## 5. Why the core backward explodes — NOT established

This is the open question. The core is a weight-shared recurrence, 6 layers × 8
iterations with BPTT over the last 4. In A1 it runs, for the first time, on a
**64-position compact slot sequence** instead of 1024 token positions, and about
22 % of those positions (`tul/spans` ≈ 50 of `max_slots` 64) are **exact zeros**
from `gather_valid`.

Ranked candidates and the experiment that kills each:

1. **The loop is unstable on a short sequence.** Attention over 64 positions is far
   less contractive than over 1024, so the per-iteration Jacobian is larger and 4
   BPTT steps compound it. *Test:* run **A0 unchanged at `data.seq_len 64`** for 500
   steps and read `train/grad_norm`. This uses no TUL code at all. If A0 at 64 also
   explodes, TUL is exonerated and the finding is about the looped core itself.
2. **Zero pad slots.** RMSNorm's Jacobian at an exactly-zero input is 1/√eps — an
   amplifier of ~1e3 per traversal. Pads sit at the tail of the compact sequence so
   causal attention should keep them off the loss path, but `retention_carry=true`
   carries GLA state from the END of iteration *t* into position 0 of iteration
   *t+1*, which is a path from pads back to real slots. *Test:* `retention_carry
   false`, and separately seed pads with a nonzero constant.
3. **Depth schedule interaction.** `_sample_slot_depths` draws 14×64 = 896 samples,
   so `total_iters` is 8 essentially every step, against A0's max over 14 samples.
   *Test:* `bptt_depth 1`.

`core_gain_clip` — the L1 governor written for exactly this failure mode ("the β1=0
gain runaway mode", `transformer.py:227-232`) — was **0.0, i.e. off**, on every arm.
Turning it on (τ ≈ 1.5) is the cheapest mitigation to try, but it treats the
symptom; run experiment 1 first, because it tells us whether this is a TUL result
or a MORPH result.

## 6. Defects fixed in this commit

1. **An aborted run exited 0.** `train.py` returned normally after the divergence
   guard, so the queue read "exit=0" and started the next arm on a dead result. Now
   `wandb.finish(exit_code=1)` and `raise SystemExit(4)`.
2. **The queue trusted the exit code alone.** `run_tul_arms.sh` now also requires the
   arm to have printed `Final val_loss`, and stops with exit 5 if it did not.
3. **The pre-clip gradient norm was computed and thrown away.**
   `clip_grad_norm_` returns it; it is now logged every 20 steps as
   `train/grad_norm`, with `train/clip_factor` beside it.
4. **No per-region gradient visibility.** Added `gradnorm/<region>` every 100 steps
   (post-clip, so ratios between regions are exact and absolutes are recovered by
   multiplying by `train/grad_norm / grad_clip`).

Without (3) this diagnosis needed a checkpoint autopsy of a saliency buffer. With
it, the failure is one glance at a chart.

## 7. What this does NOT say

* Nothing here tests the TUL *hypothesis*. Whether a plan slot helps is untouched —
  the arms never reached a state where `val/plan_nats` means anything. It sat inside
  ±0.04 for the whole run, which is what "the plan channel carried nothing" looks
  like, but under a 1e-8 clip that is not evidence about the design.
* The chain in §4 (clip → floor → stalled scales) is inferred. The gradient
  explosion in §3 is measured.
* Section 5 is ranked speculation. No experiment has been run.
* The A1/A1r comparison is not a noise floor. Both arms failed, so the spread
  between them measures two failures, not run-to-run variance.

## 8. Verification

* `pytest tests/ -q` on CPU: **114 passed, 2 skipped** (the 2 skips are CUDA-only;
  the same suite is 116 passed on GPU).
* `bash -n ignore/tul_logs/run_tul_arms.sh`: clean.
* **Not verified:** none of the four fixes in §6 has run inside a live training
  step. `train/grad_norm`, `train/clip_factor`, `gradnorm/*` and the non-zero exit
  path all need one GPU run to confirm. The `gradnorm/*` block in particular walks
  `named_parameters()` under `torch.compile`, and the name-stripping is reasoned,
  not observed.

---

# Part 2 — the trigger, measured (2026-08-17, GPU)

Part 1 was a checkpoint autopsy with the GPU unavailable. §5 ranked three candidate
causes. **All three are wrong.** The trigger is `tul.token_state_dropout`.

## 9. The prior art was ours

`00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies/Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/`
(Task #276, June 2026) already characterised this failure for the looped core:

> optimizer coherence (β1=0 + heavy slow-EMA α) → the six core blocks' top singular
> subspaces rotate into **alignment** → composition σ_max(J_core) runs away → detonation.

Order parameter = alignment ratio (composition σ_max ÷ worst single block): **0.90
healthy, 0.91 AdamW-stable, 9.5 at the cliff.** Non-aligned blocks cancel; aligned
blocks chain multiplicatively. That campaign ruled out, with runs: `core_gain_clip`
(masks), per-linear spectral penalty (detonated, idle 0/12), loop-STP (harmful),
per-coordinate `update_clip` ("the disease is directional; a per-coord magnitude cap
is the wrong instrument"), `stale_push_cap` (delays ~300 steps).

The shipped mitigation is `ademamix_alpha_cap = 3.5`, sized from an α sweep where α≤4
survived and α=6 detonated. **Every TUL arm ran with the full cure stack** —
`alpha_cap 3.5`, `update_clip 5.0`, `g_snr_gate_kappa 0.3`,
`stale_push_cap_coord 0.5` — and detonated anyway.

## 10. The measured trigger

Batch 6, 800 steps, one flag apart. `core share` = `gradnorm/core` ÷ the norm over all
regions, at steps 100 / 200 / 300 / 500 / 700:

| arm | flag | core share | grad_norm | val @400 |
|---|---|---|---|---|
| D8 | `token_state_dropout=0.15` | 0.009 → 0.056 → **0.866** → 0.994 → **0.999** | → 6.4e5 | 6.62, ABORT @2040 |
| E9 | `token_state_dropout=0.0` | 0.008 → 0.037 → 0.016 → 0.014 → **0.008** | **2.1–2.8 flat** | **6.26** |
| E0 | A0 baseline, no TUL | 0.19 → 0.25 → 0.23 → 0.26 → 0.19 | 1.8–2.8 flat | 6.14 |

With dropout off the TUL arm is stable and tracks the baseline. With dropout at 0.15
the core owns 87 % of the gradient by step 300 and 99.9 % by step 600, `grad_clip=1.0`
then divides everything by 1e5–1e9, and every other region stops learning (Part 1 §4).

**Mechanism.** §3.4 dropout replaces 15 % of token coda inputs with `E_mask` AND zeroes
their x0/bigram injection, so those positions must be reconstructed from context — and
the slots are the most informative thing left. That aims a large COHERENT gradient at
the slots, which are the only path into the looped core. A coherent directional push on
the core is exactly what Task #276 measured as the thing that rotates the blocks into
alignment, and exactly what the per-coordinate cure stack cannot catch.

**The uncomfortable part:** the mechanism designed to force the model to USE the plan is
the mechanism that detonates the loop. At 400 steps E9's `plan_nats` is +0.0018 — with
the pressure removed the plan carries nothing. 400 steps is far too early to conclude
that, but it is the tension the design has to resolve.

## 11. Refuted

* **Short sequence in the loop (§5 candidate 1).** A0 at seq 64 with tokens/step held at
  14336 has a HIGHER core share (0.232) than at 1024 (0.145) and a LOWER peak norm.
  The looped core is fine on a short sequence, and MORPH's machinery is exonerated.
* **Quantisation.** `ternary=false embed_quant=off` (D3/E3) and `adam8bit=false`
  (D7/E7) change nothing about the takeover.
* **§5 candidates 2 and 3** — `retention_carry=false`, `bptt_depth=1` — no effect.
* **A0 is not on the same trajectory.** Its core gradient grows 1.0 → 15 → 42 → 165
  over 5k → 20k steps, but FLAT across core.0–core.5 (117–221 at 20k). Uniform growth
  from growing weights, not the graded backward amplification A1 shows (1.1e8 at core.0
  vs 3.8e4 at core.5).

## 12. Not verified

* One run per condition, no seed replicate. The separation is three orders of magnitude,
  which is not the same as replicated.
* The alignment ratio itself has NOT been measured on any TUL checkpoint. The
  fingerprints match Task #276 but the order parameter has not been read.
  `E9_diag_sigmas.py` / `E10_localize_sigma.py` in the b1zero repo are the instrument.
* Whether TUL trains to a useful `plan_nats` with dropout off, or with a smaller p, is
  untested. 800 steps says nothing about quality.
