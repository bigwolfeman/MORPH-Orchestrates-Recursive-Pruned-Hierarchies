# Planned: does packing more of the span into the LXTUL cell narrow the gap? (three arms)

Status: planned

Date: 2026-10-05 01:55 CDT, before any smoke or run. Item B of the
[CE queue](../../../.agents/notes/proposed/architecture/2026-10-04-lxtul-ce-test-queue.md)
(items 9 and 10). Each config composes `lxtul.yaml` and differs only in its own keys,
`training.steps` (5000) and `wandb.name` (asserted in the tests).

| arm | config | change |
| --- | --- | --- |
| multi-head span pool | `lxtul_pool16` | `tul.slot_pool_heads: 16`: each cell's seed pool over the span's prelude states becomes 16 heads (same parameters, reshaped) instead of one softmax average |
| own-span reconstruction | `lxtul_recon` | `tul.recon_weight: 1.0`, 2 layers: a train-only teacher-forced decoder rebuilds the slot's OWN span from the final winner cell (the cell the coda reads), gradient into the loop |
| both | `lxtul_pool16_recon` | both keys |

## Question

65 % of LXTUL's 10k gap (and 75 % at 5k) is exact bigram repeats from earlier spans. Do the
span's tokens fail to get INTO the cell (one single-head pool, a weighted average), or fail
to be KEPT (no objective asks the cell to hold this span's tokens)? The Lean package
`lab/theory/tul_pseudotoken` (`exact_tuple_write_needs_dim`) predicts that a linear write
of one cell cannot deliver two exact tokens, so these arms should move the far-bigram
buckets only a little; a small effect would not show that span content does not matter.

## Predictions (mine, orchestrator)

References: LXTUL 5k seed 1 / 2: gap to plain 5k +0.272 / +0.282, K1-K6 +0.0215 / +0.0172,
far-bigram bucket gap +1.150 / +1.154. Seed spread on the gap about 0.01.

- **all three**: no detonation (HEALTHY or one recovered spike). 80 % each.
- **P-1** pool16 K1-K6 > +0.0108, CI above 0. 65 %.
- **P-2** pool16 gap below +0.262. 30 %.
- **P-3** pool16 far-bigram bucket gap at least 0.05 below +1.150. 30 %.
- **P-4** pool16 tok/s at least 11000 (the 10k run's 12402 at the same checkpointing, minus 10 %). 75 %.
- **R-1** recon K1-K6 > +0.0108, CI above 0. 55 %.
- **R-2** recon gap below +0.262. 30 %.
- **R-3** recon far-bigram bucket gap at least 0.05 below +1.150. 35 %.
- **R-4** recon tok/s at least 10000. 60 %.
- **C-1** both: gap at least 0.005 below the better single arm. 30 %.
- **C-2** both K1-K6 > +0.0108, CI above 0. 50 %.

Pass rule, per arm: gap below +0.262 AND K1-K6 above +0.0108 with CI above 0.

## Method

5000 steps, seed 1, `model.ckpt_grad_iters: 4` (lxtul's), `runner_steps.sh`. A 60-step smoke
with a val pass first for each arm (exit 0, memory, the mechanism's log line); an arm whose
smoke fails is not queued and the failure is filed. Readouts per arm (`arm_after.sh`): depth
sweep (480 rows), gap vs plain 5k, worth, exploration ledger, LayerNorm probe and
`--vablate`, the exact-recall probe with per-bucket loop K1-K6, the repetition eval.
