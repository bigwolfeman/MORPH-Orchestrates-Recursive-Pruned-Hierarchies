# Planned: cut the slot loop's carrier constant (centered loop attention, uniform loop attention HC)

Status: planned

Date: 2026-10-05 15:36 CDT, before either arm is built. Wolfe (2026-10-05): "Let's run that as a twin run.
Top of the queue." Design and evidence:
[note](../../../.agents/notes/proposed/architecture/2026-10-05-slot-loop-carrier-constant.md).

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
