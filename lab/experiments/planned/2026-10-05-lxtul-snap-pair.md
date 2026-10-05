# Planned: do span-restricted pseudo tokens close the copy gap without killing the loop? (pair)

Status: planned

Date: 2026-10-05 02:21 CDT, before any smoke or run. Item A2 of the
[CE queue](../../../.agents/notes/proposed/architecture/2026-10-04-lxtul-ce-test-queue.md);
design and theorems:
[`2026-10-05-trace-free-pseudo-token-carrier`](../../../.agents/notes/proposed/architecture/2026-10-05-trace-free-pseudo-token-carrier.md),
`lab/theory/tul_pseudotoken/`. Wolfe (2026-10-05): "Testing the A line there needs to be
very careful to measure loop contribution, we may kill it."

| arm | config | what the 3 loser positions of each slot carry |
| --- | --- | --- |
| snap | `lxtul_snap` | pseudo tokens read from the loop's FINAL WINNER cell: a softmax over the span's own token ids through the tied (detached) embedding table |
| snap, seed source | `lxtul_snap_entry` | the same, read from the slot's ENTRY state (no loop pass) |

Both compose `lxtul.yaml` and differ only in the `tul.pseudo_*` keys, `training.steps` (5000) and
`wandb.name`. Gate init 0: step 0 is byte-identical to LXTUL (`tests/test_pseudo_snap.py`).

## Question

Can an exact-copy carrier in token-embedding space close the part of the gap that is exact
bigram copying (75 % of the gap at 5k), and does the loop keep its K1-K6? The Lean results
say the coda's CE alone pays for copy content, and that copy content does not need loop
depth (`deeper_never_more_informative`). Stage 0 put about a quarter of LXTUL's K1-K6 on the
far-bigram tokens. The seed-source twin separates "a copy channel" from "a copy channel the
loop chooses".

## Predictions (mine, orchestrator)

References: LXTUL 5k seed 1 / 2: gap to plain 5k +0.272 / +0.282, K1-K6 +0.0215 / +0.0172,
far-bigram bucket gap +1.150 / +1.154.

- **S-1** snap: no detonation (HEALTHY or one recovered spike). 80 %.
- **S-2** snap K1-K6 lower CI at least +0.0130 (the note's survive line). 45 %.
- **S-3** snap K1-K6 point at least +0.0086 (the note's kill line not crossed). 70 %.
- **S-4** snap gap to plain 5k below +0.262. 45 %.
- **S-5** snap far-bigram bucket gap at least 0.10 below +1.150. 45 %.
- **S-6** snap `fan/pseudo_vertex_mass` at the last val above 0.5 (the pseudo tokens commit to single tokens). 40 %.
- **S-7** snap tok/s at least 11000. 70 %.
- **E-1** seed-source K1-K6 below snap's (CIs separate). 50 %.
- **E-2** seed-source gap within 0.01 of snap's. 50 %.

Pass rule (snap): S-3 and S-4 hold. A snap result that closes the gap while K1-K6 falls
below +0.0086 is a FAILURE under Wolfe's ranking, whatever the CE.

## Method

5000 steps each, seed 1, `model.ckpt_grad_iters: 4`, `runner_steps.sh`, at the front of the
queue. Mechanism check and a 60-step smoke with a val pass first (`smoke_queue.sh`). Readouts
per arm (`arm_after.sh`): depth sweep, gap vs plain 5k, worth, exploration ledger,
LayerNorm probe and `--vablate`, the exact-recall probe with per-bucket loop K1-K6, the
repetition eval; plus the val logs' `fan/pseudo_vertex_mass`, `fan/pseudo_entropy`,
`fan/pseudo_copy_hit`. Known gap: the fan oracle readings (`fan/single_ce`, `fan/oracle_ce`)
replay without the pseudo positions.
