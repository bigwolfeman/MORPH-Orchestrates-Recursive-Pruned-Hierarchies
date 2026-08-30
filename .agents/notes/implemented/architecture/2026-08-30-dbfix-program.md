# Agent Note: DB-fix program — faithful DiffusionBlocks, then interleave with l2cap

Status: implemented

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

## Decision

Fix-then-mix, with a binding sequencing rule: the faithful DB had to validate
(depth curve ≥ 0.02 nats at eval) BEFORE any interleave arm ran. The machinery
shipped in commit 81a9674: `CoreStageConditioning` (AdaLN-Zero, iter/σ keyed,
bit-identical at init), `DB1Sampler` (equal-mass log-normal σ + EDM precond),
one-pass `_tul_core_db1` training step, `_tul_core_db1_ladder` deterministic Euler
eval, per-forward `tul_step_mode` + `training.step_mix` interleave schedule, and
four configs (`tul_dbfix`, `tul_db_cond`, `tul_l2cap_cond`, `tul_ilv50`).

The validation pair ran 2026-08-30 under the frozen prereg
(`lab/experiments/successes/2026-08-30-tul-dbfix-pair.md`). Both depth gates
failed: dbfix's Euler ladder is monotonically WORSE with depth (K=1 4.4652 →
K=8 4.5101), and iter-conditioning leaves the old db_loop dead flat (spread
0.005 nats over K=1..8). The binding rule fired on the both-fail branch.

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

## Consequences

- **Interleave CANCELLED.** `tul_ilv50` and `tul_l2cap_cond` never run; their
  configs remain in-tree as built-but-unrun artifacts of the decision.
- The DB machinery stays: the one-pass path is fast (37.7 min vs l2cap's 66) and
  stable, and `step_mix` is a general interleaving primitive independent of DB.
- Standing conclusion strengthened: three DB-style mechanisms (target scheduling,
  σ+EDM one-pass, iter-AdaLN) all yield good CE with an inert loop; only gradient
  through the iterated map under contractivity control (l2cap) earns depth. Any
  future depth recipe must keep the composition inside the training graph.
- Next program (from the addendum below): the gate-vs-cap ladder.
- torch.compile coverage of the eager TG attention (task 6) remains open and
  unmeasured — the biggest known wall-clock lever.

## Addendum (2026-08-30): gate-vs-cap arm queued behind the DB-fix pair

Wolfe flagged arXiv 2608.15062 (Gated Recurrent Transformers — references.md row
69): a learned convex gate per iteration is contractivity control by architecture,
bias-init near-identity. Next-batch arm after task 8/9: **cap vs gate vs both** on
the TUL loop, plus a uniform-{1..R} depth-sampling variant (our clamped Poisson(6)
barely trains depths 1-2). Not built yet; prereg before any run, same rules.
