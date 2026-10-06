# Planned: cut the slot loop's carrier constant (centered loop attention, uniform loop attention HC)

Status: failure

Date: 2026-10-05 15:36 CDT, before either arm is built. Wolfe (2026-10-05): "Let's run that as a twin run.
Top of the queue." Design and evidence:
[note](../../../.agents/notes/rejected/architecture/2026-10-05-slot-loop-carrier-constant.md).

## Question

The slot loop spends 98 % of its carrier on one slot-independent vector that the core
attention builds from layer 2 on. If the loop's attention cannot read or write that
vector, does the loop carry more per-slot content forward and earn more depth?

Configs (both compose `lxtul.yaml`, 5000 steps, seed 1): `lxtul_center`
(`tul.loop_attn_center: ema`) and `lxtul_hcuni` (`tul.loop_attn_hc: uniform`).

## Predictions (mine, orchestrator)

References: LXTUL 5k seed 1 / seed 2: K1-K6 +0.0215 / +0.0172, gap to plain 5k +0.272 /
+0.282, zero-cell worth +0.176 / +0.154; per-slot share of the carrier after pass 1 0.19 %
(seed 1), at the exit 1.65 %; exit next-span R^2 0.055.

| id | prediction | centered | uniform HC |
| --- | --- | --- | --- |
| C-1 | per-slot share of the carrier after pass 1 above 10 % | 70 % | 60 % |
| C-2 | K1-K6 above +0.0215, CI lower bound above +0.0172 | 40 % | 35 % |
| C-3 | gap to plain 5k below +0.262 | 35 % | 30 % |
| C-4 | exit next-span R^2 above 0.08 | 40 % | 35 % |
| C-5 | no detonation (HEALTHY or one recovered spike) | 80 % | 70 % |
| C-6 | tok/s within 5 % of LXTUL at the same checkpointing | 85 % | 85 % |

Pass rule, per arm: C-1 holds, and C-2 or C-3 holds, with K1-K6 not below +0.0172. Kill
reading: if C-1 fails (per-slot share stays under 2 %), training rebuilt the constant by
another path; the arm says nothing about the hypothesis and the next step is to find that
path.

## Method

Each arm: a mechanism check on CPU and a 60-step smoke with a val pass before it queues;
5000 steps through `runner_steps.sh`; the standard readouts (480-row depth sweep, gap to
plain 5k, worth, exploration ledger, recall probe); plus `dataflow_probe.py` and
`carrier_constant_probe.py` on the 3070 for C-1 and C-4.

## Results

Training (5090, 5000 steps each, `runner_steps.sh`; 480-row sweep, gap vs plain 5k, worth on
192 rows, exploration ledger) and the mechanism readout (3070 at efbd4989, 96 rows,
`dataflow_probe.py` + `carrier_constant_probe.py`; CE bit-equal under the hooks, the source
split sums to the carrier, the LXTUL reference reproduces exactly). Artifacts:
[`../results/2026-10-05-carrier-constant-twin/`](../results/2026-10-05-carrier-constant-twin/).

| reading | LXTUL 5k s1 / s2 | uniform loop attention HC | centered loop attention |
| --- | --- | --- | --- |
| per-slot carrier share: entry / after pass 1 / exit | 53.5 / 0.19 / 1.65 % | 50.9 / 1.64 / 1.33 % | 68.3 / 27.3 / 6.83 % |
| next-span R^2 at exit (own span) | 0.055 (0.349) | 0.054 (0.348) | 0.033 (0.312) |
| K1-K6 (480 rows) | +0.0215 / +0.0172 | +0.0171 [+0.0163, +0.0180] | +0.0534 (broken model) |
| gap to plain 5k | +0.272 / +0.282 | +0.270 [+0.255, +0.288] | +0.839 [+0.822, +0.859] |
| zero-cell worth | +0.176 / +0.154 | +0.160 | +0.041 |
| ledger CE | 4.353 / 4.362 | 4.350 | 4.919 |
| tripwire | | HEALTHY, max 91 | DETONATED at 470 (2.3e6), again from 1617 (max 2.7e12) |
| tok/s, peak | 12402 (10k run) | 12777, 19.4 GB | 12328, 20.1 GB |

| prediction | uniform HC | centered |
| --- | --- | --- |
| C-1 per-slot share after pass 1 > 10 % | no (1.64 %, under the 2 % kill line) | yes (27.3 %), on a broken model |
| C-2 K1-K6 > +0.0215, CI low > +0.0172 | no | not readable (broken model) |
| C-3 gap < +0.262 | no | no |
| C-4 exit next-span R^2 > 0.08 | no | no |
| C-5 no detonation | yes | no |
| C-6 tok/s within 5 % | yes | yes |

## Verdict

Failure on both arms. The uniform HC arm hit the prereg's kill reading: training rebuilt the
constant by another path. In LXTUL the per-slot share collapses at core attention layer 2;
here at layer 3, where the (uniform) attention write is 97.7 % shared, and the MLP's Cayley
stream mixers at layers 3-5 turn that stream-identical write into stream differences (the
mixer step is inferred from the decomposition, not causally tested). The new constant has
cosine ~0 to LXTUL's, a different sign pattern (++-- vs +-+-), still cancels in the stream
mean, and is still load-bearing (removing it costs +0.20 nats). Core attention layers 1-2 now
return slot-specific content (shared share 0.11 / 0.62 vs 0.91 / 0.88), but nothing reached
the scores. The centered arm detonated in the LR ramp (step 470, nearly all of it in core
block 0, single-slot gain 44) and never trained: the EMA mean grew with depth to RMS 130-160,
and the centered input at the last layer is 3.9 % of the raw input, a cancellation of two
large vectors. The trigger is inferred; mu is not logged in training and no pre-onset
checkpoint exists.

## Updated hypothesis

The shared vector is not an accident of one attention path: the trained loop builds it by
whatever path is left, with a new direction each time, and depends on it. It is more likely
a function the loop needs (a bias or sink in the stream differences) than the reason the loop
earns little; cutting it does not free capacity. The next test should not be another MORPH
arm on this axis. Wolfe's 2026-10-05 option applies: a minimal (~1k-line) slot-loop model
with a plain residual, to see whether the loop earns depth at all without the HC streams.
