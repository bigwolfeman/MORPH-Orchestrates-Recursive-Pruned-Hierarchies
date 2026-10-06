# Planned: does packing more of the span into the LXTUL cell narrow the gap? (three arms)

Status: failure

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

## Results

References as preregistered: LXTUL 5k seed 1 / seed 2 K1-K6 +0.0215 / +0.0172, gap to plain 5k +0.272 / +0.282, zero-cell worth +0.176 / +0.154, far-bigram bucket gap +1.150 / +1.154. Readouts: 480-row depth sweep, `paired_vs_ruler.py` gap vs plain 5k, `worth_profile.py` on 192 rows, the exploration ledger, `exact_recall_gap.py` vs plain 5k, the LayerNorm probe's `--vablate` on 96 rows (K6-K1 change when the cells' shared-direction amplitude is set to its mean). Artifacts: [`../results/2026-10-05-lxtul-cell-packing-panel/`](../results/2026-10-05-lxtul-cell-packing-panel/).

| reading | multi-head pool | own-span reconstruction | pool + reconstruction |
| --- | --- | --- | --- |
| tripwire | DETONATED at 2844 (4.0e5), recovered | HEALTHY, max 301 | AMBIGUOUS, max 9.2e3 |
| K1-K6 | +0.0221 [+0.0210, +0.0231] | +0.0010 [+0.0008, +0.0013] | +0.0039 [+0.0034, +0.0044] |
| gap to plain 5k | +0.2789 [+0.2627, +0.2965] | +0.3774 [+0.3633, +0.3932] | +0.3942 [+0.3797, +0.4102] |
| zero-cell worth / shuffle | +0.166 / +0.160 | +0.161 / +0.122 | +0.145 / +0.113 |
| ledger CE | 4.3593 | 4.4580 | 4.4749 |
| far-bigram bucket gap | +1.149 | +1.178 | +1.208 |
| mean-ablation K6-K1 change | -1.8 % | | |
| tok/s, peak | 12318, 20.2 GB | 10852, 21.8 GB | 10723, 22.0 GB |

| prediction | held |
| --- | --- |
| no detonation | pool: one recovered spike; recon: yes; both: yes |
| P-1 pool K1-K6 > +0.0108 | yes |
| P-2 pool gap < +0.262 | no |
| P-3 pool far-bigram >= 0.05 below +1.150 | no |
| P-4 pool tok/s >= 11000 | yes |
| R-1 recon K1-K6 > +0.0108 | no (+0.0010) |
| R-2 recon gap < +0.262 | no |
| R-3 recon far-bigram >= 0.05 below +1.150 | no (+1.178, worse) |
| R-4 recon tok/s >= 10000 | yes |
| C-1 both gap 0.005 below the better single arm | no |
| C-2 both K1-K6 > +0.0108 | no |

## Verdict

Failure on every arm's pass rule. The multi-head pool is neutral: loop earning at the seed-1 level,
gap unchanged. Own-span reconstruction is harmful: it kills loop contribution (K1-K6 +0.0010) and
widens the gap by 0.10; asking the cell to keep its own span crowds out what the loop carries
forward. Combined is worse than either.

## Updated hypothesis

Getting span tokens INTO the cell (pool heads) or making the cell KEEP them (reconstruction) does
not close the gap. Together with the wider write and the carrier-constant twin, the cell-content
axis of the CE queue is closed at 5k.
