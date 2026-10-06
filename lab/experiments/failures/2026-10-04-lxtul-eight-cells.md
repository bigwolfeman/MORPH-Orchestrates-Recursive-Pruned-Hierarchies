# Planned: does LXTUL with 8 cells per slot carry a larger share of the cross-span budget?

Status: failure

Date: 2026-10-04, before the run (the `date` stamp is in the commit). Config
`lxtul_fan8`: `lxtul.yaml` with `tul.fan_k` 4 -> 8 and `tul.prefix_k` 4 -> 8
(`fan_mix: all` requires them equal), 5000 steps; the composed configs differ in those two
keys, `training.steps` and `wandb.name` only. Wolfe (2026-10-04): "How about we try more
cells?"

## Question

LXTUL's gap to plain at the same step is +0.272 (5k) and +0.356 (10k). The gap plus the
cells' zero-ablation worth is about 0.45 at 5k and 0.59 at 10k, and the cells hold a nearly
constant 39-40 % of that sum (0.176 / 0.448; 0.233 / 0.589). The measured cross-span budget
at 5k is 0.40 nats. So the gap is about the part of the budget the cells do not carry, and
it grows because the budget grows. Is that share limited by the channel's WIDTH (4 cells of
1024 per span) or by what the loop learns to write? If width, 8 cells raise the share and
cut the gap. If not, the share stays near 40 %.

Prior evidence for width: fan4-all vs the one-cell strict ruler, +0.208 vs +0.254 against
the reachall ruler (2026-09-21); the register's 0.022 CE win was its 4x wider readout
(2026-09-13).

## Predictions (mine, orchestrator)

References: LXTUL 5k seed 1 (gap +0.272, worth +0.176, K1-K6 +0.0215, 9800 tok/s) and seed
2 (+0.282, +0.154, +0.0172, 8876 tok/s). Seed spread on the gap is ~0.01.

- **E-1**: no detonation (HEALTHY or one recovered spike). 80 %.
- **E-2**: gap to plain 5k below +0.262 (0.01 better than seed 1, outside the seed
  spread). 55 %.
- **E-3**: gap below +0.245. 25 %.
- **E-4**: zero-cell worth above +0.190. 55 %.
- **E-5**: share worth / (worth + gap) above 0.42 (both 5k seeds: 0.39, 0.35). 50 %.
- **E-6**: K1-K6 > +0.0108, CI above 0 (the loop still earns depth with 8 cells). 65 %.
- **E-7**: mean-ablating the cells' shared direction changes K6-K1 by under 30 %. 65 %.
- **E-8**: tok/s 15-35 % below LXTUL's (the loop runs 2x the cells and the coda reads 2x
  the prefix positions). 60 %.

Pass rule: E-2 and E-6 hold.

Correction to the Question, 2026-10-05 01:02 CDT, after training and BEFORE any readout was read (the
predictions above are unchanged): the Question is framed wrongly. Under LXTUL's
latent-selected loop the coda reads only the final winner cell per slot; the other
positions are exactly zero. So 8 cells do NOT widen the coda's channel (still one nonzero
position per slot). They give the loop 8 proposals per pass instead of 4. This arm tests
proposal count. The positions question moves to `lxtul_fan4x2` (each cell written into 2
positions). E-8's stated reason ("the coda reads 2x the prefix positions") is wrong for the
same cause.

## Method

5000 steps, seed 1, `runner_steps.sh`. Readouts as for every LXTUL arm: depth sweep (480
rows), gap vs plain 5k, worth, the exploration ledger, the LayerNorm probe (geometry and
`--vablate`), the repetition eval. A 60-step smoke with a val pass precedes the run, to
read memory and step time.

Method amendment, 2026-10-04 14:12 CDT, before the run: `model.ckpt_grad_iters` -1 (checkpoint every
pass) instead of lxtul's 4. The first smoke ran out of memory at step 1 (25.2 GiB in use)
with 4 eager passes. Checkpointing is exact (gradients unchanged), so the arm is the same;
only its speed is. The 5k LXTUL seeds this pairs with also ran at -1 (9800 / 8876 tok/s),
so E-8 is read against those, not the 12402 of the 10k run. Smoke at -1: 60 steps and a val
pass, peak 18.46 GB, about 8.1k tok/s by step 40.

## Results

References as preregistered: LXTUL 5k seed 1 / seed 2 K1-K6 +0.0215 / +0.0172, gap to plain 5k +0.272 / +0.282, zero-cell worth +0.176 / +0.154, far-bigram bucket gap +1.150 / +1.154. Readouts: 480-row depth sweep, `paired_vs_ruler.py` gap vs plain 5k, `worth_profile.py` on 192 rows, the exploration ledger, `exact_recall_gap.py` vs plain 5k, the LayerNorm probe's `--vablate` on 96 rows (K6-K1 change when the cells' shared-direction amplitude is set to its mean). Artifacts: [`../results/2026-10-04-lxtul-eight-cells/`](../results/2026-10-04-lxtul-eight-cells/).

| reading | more cells (fan8) |
| --- | --- |
| tripwire | EXCURSION: one-step outliers at 383, 1374, 2782 (max 4.5e4), each recovered |
| K1-K6 | +0.0103 [+0.0097, +0.0110] |
| gap to plain 5k | +0.2697 [+0.2537, +0.2878] |
| zero-cell worth / shuffle | +0.158 / +0.137 |
| share worth / (worth + gap) | 0.37 |
| ledger CE | 4.3470 |
| far-bigram bucket gap (share of gap) | +1.154 (75.7 %) |
| mean-ablation of the shared direction, K6-K1 change | +13 % |
| tok/s, peak (checkpoint every pass) | 7456, 19.9 GB |
| repetition eval | not measured: OOM twice (2.06 GiB alloc), alone on the GPU the second time |

| prediction | held |
| --- | --- |
| E-1 no detonation | yes (recovered one-step outliers) |
| E-2 gap < +0.262 | no |
| E-3 gap < +0.245 | no |
| E-4 worth > +0.190 | no |
| E-5 share > 0.42 | no (0.37) |
| E-6 K1-K6 > +0.0108 | no (+0.0103) |
| E-7 mean-ablation change < 30 % | yes (+13 %) |
| E-8 tok/s 15-35 % below LXTUL at -1 | yes (24 % / 16 % below the two seeds) |

## Verdict

Failure: the pass rule (E-2 and E-6) fails on both. Eight proposals per pass leave the gap at the
LXTUL seed level and HALVE the loop's depth earning. Under the latent-selected loop the coda
still reads one winner per slot, so more cells only widen the per-pass search; the wider search
costs 24 % in speed and buys no CE. Cell count as search breadth is not a CE lever at 5k.

## Updated hypothesis

More search breadth per pass does not help a loop whose carrier is 98 % one slot-shared vector
(see the 2026-10-05 data-flow probe); the variable-cell-count idea stays open as a test-time
knob but has no CE prior from this run.
