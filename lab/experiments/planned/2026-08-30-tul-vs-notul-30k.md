# Planned: the 30k head-to-head — winning TUL (l2cap) vs no-TUL MORPH, both capped

Status: planned
Date: 2026-08-30 (frozen while the speed bench runs; bench numbers enter Method
as a dated amendment before launch — predictions are frozen now and do not
depend on them). Wolfe's parameters, verbatim: no-TUL gets the cap too (1);
scored token-matched AND wall-clock-matched (2); 30k steps; eval_every=700
("it helps throughput a lot"); ckpt_every=10000; no pruning, no TST; QAT on.

## Question

Is the winning TUL recipe worth it against MORPH-without-TUL at a long horizon —
on wall-clock (the practical axis: TUL loops ≤64 slots, no-TUL loops all 1152
token positions) and on tokens (the quality-per-data axis: no-TUL gives every
token the full looped core)? Secondary science: does the TOKEN-path loop earn
depth under the σ-cap? (The old, uncapped full model's core loop was worth 0.017
nats — `morph-core-loop-worth-002-nats` — but that predates the cap.)

## Arms

Both: seed 1, batch 6, 30k steps, flat LR 1e-4, panel optimizer
(ademamix_alpha_cap=3.5, **ademamix_t_beta3=3500 pinned** — the t_beta3-null
trap would silently set 30000), σ≤1.5 post-step projection, full BPTT
(bptt_depth ≥ max_depth), ternary QAT backbone + embed QAT (chain defaults),
prune/carve/route disabled (chain default 999999999), TST off (chain default),
eval_every=700, ckpt_every=10000 (NO checkpoint pruning — keep 10k/20k/30k),
gen_every=0, grad_probe_every=50 (thinned for throughput).

- **A — tul-30k**: `tul_l2` exactly (the recipe note's five ingredients; eager
  attention, `use_kernels=false` forced by TG-restrict). If the running speed
  bench shows compiled attention is a clean win (≥10% sps, no recompile churn,
  loss curve sane over 450 steps), A runs with
  `training.compile_attention=true` — recorded here as a conditional, decided
  in the Method amendment, because it changes wall-clock only, not the recipe.
- **B — notul-30k**: `tul_l2` + `tul.activate_at=never` +
  `model.use_kernels=true` (its best implementation; kernels are legal without
  TG). No mux (no slots). Same cap, same loop depths on the token path.

## Predictions (frozen)

- **H1 (binding, wall-clock axis).** At matched wall-clock (read at arm A's
  total duration; B's curve interpolated at that time), A's val CE is lower:
  75%. B pays the full-sequence loop tax in steps/hour.
- **H2 (binding, token axis).** At matched tokens (CE@30k vs CE@30k — same
  batch and seq, so steps are tokens), A's val CE is lower: 35%. My prior
  leans B: every token gets ~6 iterations of capped core, while A taxes tokens
  with slot overhead and p_drop; the old A3/coreless results run this way too.
- **H3 (token-loop depth).** B's token-path forced-depth sweep at 30k earns
  ≥ 0.10 nats: 45%. The cap has never been given the token loop at scale; the
  0.017-nat history was uncapped.
- **H4 (stability).** Both arms S1-clean over 30k (no eval >0.20 over running
  min ×2 consecutive after step 1000): 70% — 6.7× the validated horizon is new
  territory for the capped recipe.
- **H5 (A's depth curve holds).** A's depth-earned CE at 30k ≥ 0.20 nats
  (does earning persist/grow past 4500?): 70%.
- **Decision rules (binding).** H1 AND H2 pass ⇒ TUL wins outright; it becomes
  the default MORPH configuration in the tree (base.yaml surgery proposal goes
  to Wolfe). H1 passes, H2 fails ⇒ TUL is the wall-clock/efficiency
  configuration; record the crossover token count; conditional-compute framing
  stands (matches the TUL merge rationale). H1 fails ⇒ the slot geometry does
  not even pay on wall-clock at horizon — the TUL loop program pauses pending
  postmortem; no further TUL arms before it. H3 passes ⇒ the cap generalizes
  beyond slot geometry — a major result on its own; write the Agent Note.

## Method

1. Speed bench first (4×450 steps: l2cap as-is / +compiled attention / no-TUL
  kernels / no-TUL eager) — its sps table enters here as a dated amendment and
  sets the conditional in arm A plus the expected durations.
2. Launch A then B sequentially (ONE trainer on the 5090), each followed by:
  depth sweep at the 10k/20k/30k checkpoints (A: the standard 48-row TUL sweep;
  B: needs the token-path variant of the sweep — mutate `model.cfg.mean_depth`
  at eval over a plain forward; SMALL INSTRUMENT TO BUILD before B finishes,
  flagged), tul_samples for A, plain gen samples for B, no checkpoint pruning.
3. Score H1 at t = A's wall-clock end against B's eval-history interpolation
  (wandb stamps); H2 at step 30000 exactly; S1 from eval histories.
4. Artifacts → `lab/experiments/results/2026-08-30-tul-vs-notul-30k/`; wandb
  `adew-me/morph-tul`, names tul-30k / notul-30k.

### Method amendment — 2026-08-30 (speed bench results; predictions untouched)

Bench complete (4×450 steps, seed 1, batch 6, queue log 22:57–23:26):

| arm | sps | tok/s |
|---|---|---|
| l2cap as-is (eager) | 1.31 | 8,042 |
| l2cap + compiled attention | **1.55** | **9,498** |
| no-TUL + kernels | 1.28 | 7,877 |
| no-TUL eager | 0.72 | 4,422 |

- **compile_attention conditional: RESOLVED TRUE.** All three criteria met:
  +18% sps (≥10%), zero recompile churn in the log (the single grep hit is the
  standard `eager_on_recompile` stance line), loss sane at step 400 (12.32 vs
  12.94 as-is — within run-decorrelation noise). Arm A runs with
  `training.compile_attention=true`.
- **Arm B implementation confirmed**: kernels are worth 1.78× on the no-TUL
  full-sequence path (1.28 vs 0.72) — `use_kernels=true` is its best form, as
  planned.
- **Expected durations**: A ≈ 30000/1.55 = 5.4 h; B ≈ 30000/1.28 = 6.5 h.
  With compiled attention, A is ~21% faster than B in steps/hour — the ≤64-slot
  loop plus compile beats the 1152-position loop plus kernels.

### Method amendment — 2026-08-31 (ABORT after arm A)

Arm A completed (5.18 h, exit 0) but its val CE (1.24) triggered the carry-leak
audit (`../successes/2026-08-31-carry-leak-audit.md`): the depth-earning and CE
collapse are retention-carry exploitation, not modeling. Arm B was paused at
~step 500 for the audit and then CANCELLED — under the current recipe H1/H2
would compare leak-exploitation capacity across geometries (B's carry
summarizes 1152 positions vs A's 64), H3/H5 are leak-confounded by
construction, and only H4 (stability; A was S1-clean through 30k) survives.
This file stays in planned/ as the record; the head-to-head re-runs under a
carry-fixed recipe with a fresh prereg. Predictions above were never scored.

## Not verified before launch

The capped recipe has never run past 4500 steps (H4 is a real question, and
t_beta3=3500 at 30k steps is an untested optimizer regime). compile_attention
is benchmarked at 450 steps only — 30k-step compile stability is unverified.
The token-path depth-sweep instrument for B does not exist yet. `tul.activate_at=never`
composed from the tul_l2 chain is smoke-tested only via the bench arms.
