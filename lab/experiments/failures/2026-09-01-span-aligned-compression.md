# Planned: E-SAC — span-aligned compression, the E1-derived recovery arm

Status: failure
Date: 2026-09-01 (frozen before launch; Wolfe: "Prereg the span-aligned
compression arm and run it this is our most promising direction right now.")

## Question

Does restoring the pooled-compression mechanism inside the TG restriction —
snapped to span boundaries — recover the TUL arm's token-axis deficit?

## Hypothesis

E1 (successes/2026-09-01-mask-surgery-decomposition.md) showed the deleted
pooled global branch was worth 0.231 nats on the trained noTUL model, and E2
showed the input-time slot seed cannot carry span content (E_slot dominance;
write-once pre-context pooling). The compressed branch has the three
properties the seed lacks: live per-layer hidden-state pooling, no shared
additive constant, and direct read-gradient. Giving TUL that mechanism at
span granularity recovers a large part of the 0.357-nat gap.

## Method

One construction-time flag, `tul.tg_span_comp` (commit carries it), on top of
the EXACT tul-20k recipe: config `tul_sac` = `tul_g0c0` + the flag. ONE
change vs the measured tul-20k arm. The prelude/coda compressed branch
attends per-SPAN mean-pooled K/V of the span's token positions, pooled from
the layer's own post-projection k/v; visibility is span-granular causal
(summary j visible to position i iff span j's last token position < i, so no
token ever sees its own span's summary). Slots stay in the sequence, keep
their loop, and stay readable through the window branch's tg_allow. Zero new
parameters; byte-identical construction (tests).

v1 simplifications, recorded: mean pooling (not the GatedPoolCompressor's
learned gates — that is the follow-up arm if this one moves); pooled keys are
post-RoPE (the summary carries the span's mean phase); the comp branch
REPLACES slot-position attention rather than adding to it.

Run: 20000 steps, panel flags identical to the h2h pair (batch 6, seed 1,
alpha_cap 3.5, t_beta3 3500, eval_every 250, gen_every 0, grad_probe_every 1),
eager kernels, ckpt_every 2000 + prune to step_20000, core depth sweep 1..8,
gen samples. DELIBERATE DEVIATION from the recovery-program note's
scaled-horizon guidance: horizons stay at the panel values so the comparison
against tul-20k (last-5 3.8461) and notul-20k (last-5 3.4894) is
single-variable. The horizon rebaseline is a separate experiment.

Verification before launch: tests/test_tg_span_comp.py (7 tests: validation,
byte-identity, hand-value pool+visibility, forward/backward, flag-changes-
function, strict causality under fixed layout) — 7 passed; full suite 739
passed 1 xfailed; smoke gate requires the "TUL TG SPAN-COMP ON" print, no
acausal warning, retention keys 0, loss < 14.

## Predictions (frozen)

- **P-S1.** tul-sac last-5 val CE ≤ 3.746 (≥ 0.10 nats better than
  tul-20k's 3.8461): 65%.
- **P-S2.** last-5 ≤ 3.668 (recovers ≥ half the 0.357 gap): 45%.
- **P-S3.** Clean 20k — no div-guard abort, no takeover abort: 75%.
- **P-S4.** The branch is load-bearing at eval: loading the trained
  checkpoint into a tg_span_comp=false build (slot-position comp branch)
  degrades val CE by ≥ 0.05: 70%.
- **Binding.** P-S1 TRUE ⇒ SAC becomes the TUL mainline; next arms stack
  learned gated pooling and the E3 seed rebalance on top, and the wall-clock
  cost of the pooling gets measured against the 1.47 sps reference. P-S1
  FALSE ⇒ mean-pooled live K/V is insufficient — run ONE learned-gated-
  pooling variant before abandoning the lane. Detonation/takeover ⇒ file
  forensics; SAC changed span-ward gradient flow.

## Not verified before run

Training dynamics of the new branch (only 30-step smoke); the pooling's
wall-clock cost at seq 1152 x 20k steps; post-RoPE mean-phase keys are a
guess v1 accepts; P-S4's ablation harness (checkpoint loads into the
tg_span_comp=false build trivially — zero new keys — but the eval run itself
is post-hoc).

## Results (2026-09-01, run tul-sac, wandb fpdokxt2, filed same day)

Run: 20000/20000 steps, exit 0, 79 evals, no div-guard, no NaN. Wall clock
10:05:49 -> 14:41:46 = 4.60 h vs tul-20k's 4.50 h (+2.1 % for the pooling).

| Metric | tul-sac | tul-20k | notul-20k |
|---|---|---|---|
| last-5 val CE (18750-19750) | **3.8623** | 3.8461 | 3.4894 |
| sweep K6 ce (48 rows, matched harness) | **3.7853** | 3.7568 | — |
| sweep K6 span_first | 3.3896 | 3.3580 | — |
| sweep K1-K8 loop contribution | 0.0022 | 0.0152 | 0.207 (token axis) |

P-S4 ablation (`core_depth_sweep.py --ckpt sac-off=tul_g0c0=...step_20000.pt`,
same 48 rows): sac-off K6 ce = 3.9721 vs sac-on 3.7853 -> **0.187 nats**
degradation; span_first 3.8273 vs 3.3896 (**0.438**). Artifacts:
`$Q/core_depth_sweep_tul-sac.json`, `$Q/core_depth_sweep_sac-off.json`,
`$Q/tul-sac/run.log`, prereg commit 5f1bcb6.

- **P-S1 FALSE** (65% claimed). 3.8623 > 3.746; did not even match tul-20k
  (+0.016 last-5, +0.028 on the matched sweep).
- **P-S2 FALSE** (45%). No gap recovery at all.
- **P-S3 TRUE** (75%). Clean run.
- **P-S4 TRUE** (70%). The branch is heavily load-bearing at eval.

## Verdict

FAILURE. Span-aligned mean-pooled compression is load-bearing (0.187 nats at
eval) yet recovers zero of the 0.357-nat gap — it SUBSTITUTES for capacity
the slot-comp model finds elsewhere instead of adding to it. The E1
mask-surgery arithmetic (0.231 nats for the compressor swap on trained
notul-20k) does not transfer to training-from-scratch: a model trained under
either comp-branch variant converges to the same token-axis CE. The loop
contribution also fell (0.015 -> 0.002); the pooled summaries make the slot
loop even less necessary.

## Updated hypothesis

Post-hoc surgery on a trained model measures what THAT model built on the
deleted pathway, not what a fresh model can rebuild around the restriction.
The tul-20k gap is not explained by which global-compression carrier the coda
reads (slot states vs span-pooled K/V both land ~3.78-3.85); the binding's
one learned-gated-pooling variant tests the last pooling degree of freedom,
after which the lane closes and the gap hunt moves to the window-branch
visibility restriction (e1b's 0.487 > e1c's 0.231) or the training recipe.
