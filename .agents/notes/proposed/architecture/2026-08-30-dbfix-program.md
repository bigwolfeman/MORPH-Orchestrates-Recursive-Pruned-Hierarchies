# Agent Note: DB-fix program — faithful DiffusionBlocks, then interleave with l2cap

Status: proposed

## Problem

The loop ladder proved the σ≤1.5-capped full-BPTT loop (l2cap) is the first
load-bearing loop (0.233 nats depth-earned CE) but costs 66 min/4500 steps eager.
Our "DB-shaped" arm (tul-l3, `tul.db_loop`) was DiffusionBlocks in name only: the
paper (arXiv 2506.14202, mirrored in
`docs/references/training-objectives/diffusionblocks/`) conditions every pass on σ
(AdaLN + EDM preconditioning) and, for recurrent-depth models (App. E.5), trains
ONE conditioned pass per step with the recurrence appearing only at inference as an
Euler ladder — our version had no conditioning and unrolled a detached 6-loop the
paper never trains. Result: depth-inert loop (depth 1 ≡ depth 6). Wolfe's goal:
recover DB's wall-clock advantage without losing l2cap's loop effectiveness.

## Proposal

Build first, run later (nothing launches before the FM gate finishes and Wolfe gets
a report). Sequencing rule, binding: **the fixed DB must validate (a depth curve
exists at eval) BEFORE any interleave arm runs.**

Task list:

1. [ ] Core stage-conditioning module (AdaLN-lite, zero-init ⇒ bit-identical at
   init; `core_stage_cond: none|iter|sigma`, baked at construction).
2. [ ] Faithful one-pass DB step: per-slot σ (log-normal equal-mass, σ_data 0.5),
   z_σ = z + σε, EDM c_skip/c_in/c_out/c_noise, one conditioned core application,
   existing mux/CE as the loss, w(σ) hook default 1. Eval: deterministic Euler
   ladder, K = mean_depth.
3. [ ] Per-forward step-mode data argument (`slot_layout` pattern) +
   `training.step_mix` interleave schedule (integer ratios; 50/50 = {bptt:1,db1:1});
   derived from global step only (resume-safe); spectral projection after every
   optimizer step in both modes.
4. [ ] Configs: `tul_dbfix` (pure one-pass DB), `tul_db_cond` (old db_loop +
   iter-conditioning — the deviation-#1 falsifier), `tul_l2cap_cond`, `tul_ilv50`
   (50/50). Every new key compose-verified against its consumer (the tul_l2
   misplaced-key lesson).
5. [ ] Verify the build subagent's work: diff review, re-run its CPU tests,
   bit-identity regression.
6. [ ] torch.compile coverage investigation (today only the MLPs compile; the eager
   TG attention is the big uncompiled slab) — measure before promising wall-clock.
7. [ ] FM gate (P1 revival on l2cap states) runs first when its adapted pipeline
   lands → then REPORT to Wolfe. No arm launches before that report.
8. [ ] Preregs per arm; run fixed DB (`tul_dbfix` + `tul_db_cond`). Gate: depth
   curve ≥ 0.02 nats or the faithful recipe also fails on TUL geometry.
9. [ ] Only if 8 validates: run `tul_ilv50` (+ `tul_l2cap_cond`).

Wall-clock guesstimates (eager, from measured arms): coreless 30 min, l2cap 66,
one-pass DB ~36 (est.), 50/50 ~51±5 (est.) per 4500 steps @ batch 6.

## Alternatives considered

- **Interleave immediately without fixing DB** (Wolfe's first phrasing): rejected by
  Wolfe himself — an interleave with a broken DB step tests nothing; fix-then-mix.
- **Only the minimal conditioning falsifier (db_cond), skip the faithful rebuild**:
  cheaper, but even if conditioning wakes the old db_loop, the paper's one-pass
  training is where the wall-clock win lives; we need both pieces built anyway for
  the interleave.
- **Optimize kernels for the DB path first**: rejected — all panel numbers are
  already eager; guesstimate wall-clock now, invest in compile coverage only after
  measurement (task 6).

## Acceptance criteria

- Conditioning-at-init bit-identity test passes; baseline forward regression passes.
- db1 step: gradients reach core + conditioning, and cross at most ONE core
  application (autograd check).
- Euler ladder deterministic; step_mix 10-step sequence is a pure function of step
  index.
- All four configs compose with keys proven to land at their consumers.
- The sequencing rule (8 before 9) is honored in the run queue.

## Risks

- The paper's "clean target" has no exact analog for slot states; our choice
  (supervise the one-pass output through the existing coda losses) is a judgment
  call the build agent must flag — wrong choice could reproduce inertness for a new
  reason.
- Conditioning module may break torch.compile's dynamic-batch core warmup
  (smoke-gated before any real run).
- 50/50 could get neither benefit (half the depth training, half the wall-clock
  saving) — that is what the arm measures.

## Addendum (2026-08-30): gate-vs-cap arm queued behind the DB-fix pair

Wolfe flagged arXiv 2608.15062 (Gated Recurrent Transformers — references.md row
69): a learned convex gate per iteration is contractivity control by architecture,
bias-init near-identity. Next-batch arm after task 8/9: **cap vs gate vs both** on
the TUL loop, plus a uniform-{1..R} depth-sampling variant (our clamped Poisson(6)
barely trains depths 1-2). Not built yet; prereg before any run, same rules.
