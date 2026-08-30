# Planned: the DB-fix pair — faithful one-pass DB (dbfix) + the conditioning falsifier (db_cond)

Status: planned
Date: 2026-08-30 (frozen before launch). Configs: `tul_dbfix.yaml` (one-pass
σ-conditioned DB, EDM precond, Euler-ladder eval, step_mix {db1:1}),
`tul_db_cond.yaml` (the OLD db_loop + iter-conditioning — deviation #1 of the paper
audit, everything else = tul_l3). Machinery commit 81a9674; program note
`.agents/notes/proposed/architecture/2026-08-30-dbfix-program.md`; paper audit in the
2026-08-29 loop-ladder filing lineage (arXiv 2506.14202 App. E.5).

## Question

The paper's recurrent-depth recipe (one σ-conditioned pass in training, Euler ladder
at inference) demonstrably composes depth on Huginn. Our old db_loop lacked its one
mechanism (stage conditioning). Does the faithful recipe produce a depth-earning loop
on TUL slot geometry — and is conditioning ALONE enough to wake the old db_loop?

## Reference numbers (fixed)

l2cap: depth curve 0.233 nats (4.6220@1 → 4.3892@6), CE @4250 4.3489, 66 min.
l3 (old db_loop): depth-flat (|Δ|≤0.001), CE @4250 4.3519, 70 min. Coreless nomask
4.3102 @4250, ~30 min. Replicate CE spread 0.030–0.036. All eager, batch 6, 4500
steps, seed 1.

## Predictions (frozen)

- **D1 (dbfix stability).** Completes S1-clean: 80% (first GPU run of the db1 path;
  smoke gates VRAM/NaN/step_mix wiring).
- **D2 (dbfix, binding).** Ladder-step sweep (patched core_depth_sweep varying
  db1_ladder_steps, 48 rows) shows CE(K=6) ≤ CE(K=1) − 0.02: 45%. The paper's
  mechanism transfers or it does not; our clean-target adaptation (supervise the
  one-pass output through the existing CE/mux, no state regression) is the novel
  piece.
- **D3 (dbfix CE).** val/ce_tokens @4250 ≤ 4.46: 50%.
- **D4 (dbfix wall-clock).** Full 4500-step run ≤ 45 min (vs l2cap 66): 70%.
- **C1 (db_cond stability).** S1-clean: 90%.
- **C2 (db_cond, the deviation-#1 falsifier).** Depth sweep ≥ 0.02 nats: 30% — my
  prior says conditioning alone does NOT rescue db_loop, because its 4 mux states
  share one target (the same-job problem), and the paper differentiates jobs via the
  σ-dependent Euler update, which db_loop lacks.
- **Decision rules (binding).** D2 passes ⇒ task 9 unlocks (tul_ilv50 +
  tul_l2cap_cond run next). D2 fails but C2 passes ⇒ conditioning is the active
  ingredient and the one-pass objective is the problem — interleave stays LOCKED,
  next arm is db_loop+cond+σ-Euler-update hybrid, new prereg. Both fail ⇒ faithful
  DB does not transfer to TUL slot geometry at this budget; interleave CANCELLED
  (sequencing rule), the loop program proceeds on the gate-vs-cap ladder instead.

## Method

1. Smoke dbfix FIRST (steps=12, eval_every=5 — must exercise the db1 step, the
   ladder eval, and the step_mix logging live; gate on exit 0 + a `train/steps_db1`
   line reaching wandb-disabled stdout or log). Smoke db_cond (same shape; its new
   surface is iter-conditioning, zero-init proven bit-identical on CPU).
2. Run tul-dbfix then tul-db-cond, panel flags, seed 1, each followed by the
   (patched) core_depth_sweep at 48 rows and tul_samples, checkpoints pruned.
3. Artifacts → lab/experiments/results/2026-08-30-tul-dbfix-pair/. Wall-clock read
   from the queue-log timestamps (START→DONE), stated per arm.

## Not verified before launch

The db1 GPU path (bf16 autocast, memory) has never run — CPU tests only (691
passed). The step_mix trainer wiring has never executed in a live main() loop; the
smoke is its first execution. The ladder-K sweep extension was added today
(compile-checked, not yet run).
