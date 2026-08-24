# Root cause analysis: the TUL core takeover

Status: measured 2026-08-23/24. The mechanism is established on five trajectories. The
intervention panel is a controlled experiment at n=1 per arm, made valid by
bit-reproducibility. One arm is still pending and is marked as such.

## Summary

The looped core's **backward operator amplifies**, and because the core is weight-shared
across the loop that amplification compounds `n_core × bptt_depth` blocks deep. The
per-block backward gain crossing 1 is the disease. Everything previously treated as the
disease — the forward per-iteration gain, the core's share of the gradient, the loss — is
downstream of it.

**No forward-side intervention reverses it once the weights are in that state.** Tested
from a fixed pre-onset checkpoint, `core_gain_clip` delays the takeover and then loses,
at every iteration range.

## Method, and why it is finally trustworthy

Three things had to land first, in order:

1. **Bit-reproducibility.** `deterministic=true` + `use_kernels=false` +
   `CUBLAS_WORKSPACE_CONFIG` — two 300-step runs agree on all 300 steps across 85 series
   ([evidence](2026-08-23-morph-bit-reproducible.md)). Before this the divergence programme
   ran ~40 single-run arms whose run-to-run spread was 6.5 % on the gradient norm, with two
   byte-identical runs disagreeing on whether the takeover happened at all
   ([the failed gate](../failures/2026-08-23-tul-run-replication.md)).
2. **A pre-onset checkpoint.** `onset-capture` aborted at step 1866 with a rolling buffer
   intact; `ROLL_step_1750` sits 72 steps clear of the last healthy step.
3. **Verified replay.** Resuming `ROLL_step_1750` reproduces steps 1751–1861 with **0 of
   111 differing** across 87 series, recreating the takeover to every logged digit.

So each arm below resumes the SAME state, sees the SAME data, and differs from its control
by ONE setting. That is a controlled experiment, and it costs ~3 minutes per arm instead of
~20 because the failure is 116 steps away instead of 1866.

## The mechanism

The core is `n_core` blocks in sequence and the backward runs core.5 → core.0, so a uniform
per-block amplification `g` leaves block 0 a factor `g^(N-1)` above the last block. Fit
`log‖grad_i‖ = a + b·i` over the per-block PRE-CLIP gradient norms and report `g = exp(-b)`
with the fit's `r²`.

| state | `g` | `r²` |
|---|---:|---:|
| healthy | 0.93 – 1.04 | 0.01 – 0.55 |
| taken over | 1.43 – 1.88 | 0.89 – 0.99 |

`r²` is half the signal. A healthy profile is flat and noisy so `g` means nothing there; a
sick one is cleanly geometric — median `r²` **0.971** (n = 1493) on one control, **0.942**
on another. The profile really is geometric, so this is a uniform operator gain and not a
spike confined to one block.

**Why a gain of 1.9 is fatal here and would be unremarkable elsewhere.** The core is
weight-shared, so the unrolled backward is `n_core × bptt_depth` = 6 × 4 = 24 blocks deep:

    1.88^24 ≈ 3e6

against an observed pre-clip core gradient rising from ~0.015 to ~1e6, a ratio of ~7e7.
Same order of magnitude from one measured number and two config constants. It is also why
one global clip starves every other region — the core's gradient is larger by `g^24`.

### It separates every trajectory, and it leads

| run | outcome | block gain fires | core share fires |
|---|---|---:|---:|
| `phase1-onset-s0` | took over | **1434** | 2033 |
| `repl-det-b` | took over | **3368** | 3874 |
| `repl-det-a` | **recovered** | **never** | never |
| `repro-ctrl` | took over | 857 ¹ | 1093 |
| `onset-capture` | took over | — ² | 1866 |

¹ consecutive-100 variant used during analysis. ² captured at 25-step resolution; at
`ROLL_1775` the core share is still 0.0541, which reads as healthy, while the block-gain
fit's `r²` has already jumped to **0.869**.

`repl-det-a` and `repl-det-b` are byte-identical runs at one seed; one died and one lived,
and the block gain says which. The **forward** gain does not: `core_gain_t0` reaches 2.06 in
the run that lived against 2.79 in the run that died, and a forward-gain criterion fires on
the survivor too.

## The intervention panel

All arms resume `ROLL_step_1750`, 229 steps. Verdict: core share > 0.5 on more than 30 % of
the last 50 probed steps.

| arm | what it changes | end share | verdict |
|---|---|---:|---|
| ctrl | nothing | 0.977 | took over |
| `gclip_all` | forward gain cap 1.5, all iterations | 0.917 | took over (**delayed**) |
| `gclip_t0` | forward gain cap 1.5, iteration 0 only | 0.986 | took over |
| `gclip_rest` | forward gain cap 1.05, iterations ≥ 1 | 0.884 | took over |
| `nocarry` | `retention_carry=false` | 0.994 | took over |
| `lr_half` | lr 1e-4 → 5e-5 | 0.963 | took over |
| `placebo` | `token_state_dropout` 0.15 → 0.145 | 0.989 | took over |
| `acap1` ³ | `ademamix_alpha_cap` 3.5 → 1.0 | 0.501 | took over |
| `spec_cap2` / `spec_cap15` | spectral penalty mid-run | — | **confounded, see below** |

³ against its own matched control `ctrlfix` (0.330), because the optimizer-hyperparameter
fix also moved `t_beta3`.

**Nothing prevents it.** The share trajectory shows `gclip_all` delaying it hard — 0.072 at
step 1850 where the control is 0.890 — and then losing by 1960. `gclip_all` and `gclip_t0`
are *identical for the first 100 steps* and then split, exactly as predicted: a 1.5 cap
initially binds only at iteration 0 (40.9 % of pre-onset steps at t0, **0.0 %** at t2–t7),
and only later does clipping the rest add anything.

The placebo takes over, so survival is not free and the panel is calibrated.

## What is NOT the cause

- **The forward gain.** It rises in runs that recover, and clamping it does not cure.
- **The GLA cross-iteration carry.** `nocarry` takes over. It still breaks causality and
  must be fixed, but it is not this.
- **The learning rate.** Halving it does not cure.
- **`ademamix_alpha_cap`.** Against a matched control, 1.0 does not cure. Notably the
  β3 warmup horizon `t_beta3` — which silently tracks `training.steps` — moved the
  trajectory MORE than `alpha_cap` did (control end share 0.977 at `t_beta3`=5000 versus
  0.330 at 1980).
- **The hyperbolic embeddings.** The Lorentz log map is the embedding channel's
  normalisation and it had a real latent defect (a dead guard, 38 % coefficient error below
  ‖s‖ 2e-3), but the zone is unreachable — 0 of 49169 vocabulary rows, minimum ‖s‖ 0.072 —
  and `preclip/embed` follows the core by ~600 steps rather than leading.
  [Note](../../../.agents/notes/implemented/bug-fix/2026-08-23-lorentz-log-map-asinh.md).

## The weights barely move

Between `ROLL_1750` (healthy) and `ROLL_1850` (taken over), 100 steps apart, over 111 core
matrices: median spectral norm **1.2376 → 1.2714**, a **3.2 %** rise, with 71 % of matrices
growing. The realized per-block backward gain over the same interval moves **1.066 → 1.434**,
**+34.5 %**. Largest movers are the CCA attention projections (`W_v_curr` +17.5 %,
`W_v_prev` +13.5 %, `W_down_q` +11.9 %), ahead of the MLP down-projections (+7.6–7.9 %).

`σ_max` is an upper bound, so these numbers do not by themselves separate "magnitudes grew"
from "the per-block amplifying directions aligned". The clean discriminator is the spectral
arm below.

Two cautions on this table, both found later. It reads the RAW parameter, so for the core
MLP — which is ternarised by a weight parametrization — it is not the matrix the forward
applies, and it is not comparable to `spec/sigma_max`, which measures the EFFECTIVE map and
reports 3.30 at step 1800 on this same run. And a spectral norm of the weights is not the
spectral norm of the block's Jacobian; the operator itself is measured in
[the cure experiment](../failures/2026-08-24-tul-takeover-cure.md).

## Confound found and declared: the mid-run spectral arms

`spec_cap2` and `spec_cap15` introduced the penalty AT the resume. Ten steps later the
control is at share 0.032 while cap2.0 is at 0.808 and cap1.5 at 0.994 — the hinge is
active immediately (σ 3.3 ≫ cap) and injects a one-off gradient. `spec_cap15` ending low is
a *recovery from a perturbation*, not a prevented takeover. **Both arms are withdrawn.**

An earlier version of this analysis read `spec_cap2`'s failure as evidence that the core MLP
spectral norm is not the lever. That reading is withdrawn: the manipulation check shows the
penalty did work on its target (σ_max 3.41 → 2.00), but the arm cannot be separated from its
own introduction transient.

### The unconfounded test: the penalty ON FROM STEP 0

`spec-scratch` runs the identical configuration as `onset-capture` — deterministic, batch
6, seed 0, 2100 steps — plus `spectral_penalty_cap=1.5`, `spectral_penalty_lambda=10.0`
from step 0. No resume, so no introduction transient.

| | control (`onset-capture`) | `spec-scratch` |
|---|---:|---:|
| core share at step 1866 | 0.602 | **0.011** |
| block gain (median, last 50) | 1.303 | **0.968** |
| block-gain fit r2 | 0.898 | 0.161 |
| block-gain criterion fires | step 1760 | **never** |
| core-share criterion fires | step 1788 | **never** |
| train loss at 1800 | 5.1289 | 5.1348 |
| outcome | aborted at 1866 | ran to 2100 |

The intervention holds the block gain below 1 through the step at which the control took
over, and it costs 0.006 nats at step 1800. **This is the affirmative test the mechanism
needed.** It is followed up at a second seed and a longer horizon in
[the cure experiment](../failures/2026-08-24-tul-takeover-cure.md), which also measures the OPERATOR
and finds that what changes across the onset is not the map's size but the alignment of
its blocks' amplifying directions.

Caveat kept in view: 2100 steps is 234 steps past the control's abort. A perturbation that
merely delayed the takeover would look the same over that window, which is why the seed-1
arms run to 6000 and 12000.

## Harness defects this exposed

1. **Resumes silently ignored every optimizer hyperparameter except `lr`.**
   `load_state_dict` replaces `param_groups` wholesale. An arm resuming with
   `alpha_cap=1.0` against a 3.5 checkpoint was BIT-IDENTICAL to its control. Fixed and
   gated by `tests/test_optimizer_resume_hparams.py`.
2. **`ademamix_t_beta3` is null and therefore tracks `training.steps`**, so two arms at
   different step budgets do not share an optimizer schedule.
3. Passing the same Hydra key twice silently keeps the first.

## Not verified

- ~~That the block gain is CAUSAL.~~ ANSWERED above by `spec-scratch`, and followed up in
  [the cure experiment](../failures/2026-08-24-tul-takeover-cure.md). What is still not verified is
  that the block gain is the ONLY route in: nothing here rules out a second failure mode
  that this cure does not touch.
- Anything about interventions applied EARLY. Every arm here starts at step 1750. A cure
  that works by shaping weights across thousands of steps could still work while failing
  here, and the old "cures" were all applied from step 0.
- One trajectory per arm. Bit-reproducibility makes n=1 a controlled comparison, not a
  sample of the phenomenon's variability.
- All of it is the reproducible configuration: eager attention, batch 6. Mechanism should
  transfer; step numbers will not.
