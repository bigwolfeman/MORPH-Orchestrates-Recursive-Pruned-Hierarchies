# Agent Note: The loop earns when paid — A2 closes the write-side campaign

Status: implemented

## Problem

After the 20k head-to-head (TUL 0.357 nats behind noTUL), the campaign
chased the TUL loop's dead depth-earning (K1−K6 ≈ 0.01 nats vs noTUL's
0.207 token-axis) through the write side: span-aligned compression (E-SAC),
learned gated pooling (E-SAC-G), and full-rank slot seeds (content / HRR
bound). Every lever moved absolute CE a little and moved loop earning not
at all. The structural suspicion: under TUL, every token in the window
skips the core always — the loop's output can only reach token CE through
a ~rank-collapsed slot bottleneck the loss barely rewards.

## Decision

Run the paid axis: arm A2 (`tul.tokens_through_core=true`) — tokens AND
slots run the ordinary per-sample core. Config `tul_a2` = `tul_g0c0` plus
exactly four deltas (tokens_through_core on; tg_restrict off and mux off,
because both interactions are unspecified and raise by design;
use_kernels on, legal without the restriction). Zero model-code changes —
the A2 branch shipped with the TUL v1 spec (§7.1) and is CPU-tested.

Result (5k, panel flags, 48 paired sweep rows,
[lab/experiments/successes/2026-09-01-a2-paid-loop.md](../../../../lab/experiments/successes/2026-09-01-a2-paid-loop.md)):
**K1−K6 = 0.1685 nats with slots present** (87% of the matched notul run's
0.1937), best absolute CE of all five arms (K6 4.1711; final val 4.2315 vs
notul-20k's matched-step 4.3216). Every free-ride arm sits at ≤ 0.0113
regardless of seed mode
([lab/experiments/failures/2026-09-01-write-side-ladder.md](../../../../lab/experiments/failures/2026-09-01-write-side-ladder.md)).
The mechanism matches the l2cap identity-escape framing: the loop stays
near-identity when nothing in the loss rewards it; when token CE depends on
core output directly, it earns.

## Alternatives considered

- **More seed engineering (E3 rebalance, per-slot embeddings).** Rejected:
  the write-side ladder showed rank and depth-earning do not move together
  (W1 rank 73 vs R0's 40; both loops at the floor). The lane is closed.
- **Compressed-branch content (E-SAC/E-SAC-G lineage).** Closed by its own
  prereg binding: real but sub-noise improvements (≤0.034 nats); the
  optimizer's own trace showed all six core-layer gates bit-zero — the core
  never sees token K/V under the restriction.
- **A2 with tg_restrict threaded into the core (A2s).** Deferred, not
  rejected: it requires carrying the per-sample TG mask through the core's
  active-set sorting and checkpointing — hot-path surgery gated on A2-plain
  showing signal first. A2-plain did; A2s is now the next build.
- **Doing nothing / accepting TUL as conditional-compute only.** The 20k
  h2h verdict already ships TUL default-off; this campaign existed to find
  whether the loop deficit is structural or fixable. It is fixable — by
  payment.

## Consequences

- The next build is the **efficient hybrid**: A2 pays ~44 passes/token,
  forfeiting TUL's 1.6x conditional-compute win. Candidates: A2s (restrict
  threaded into `_core_region`) and asymmetric depth (few paid token
  iterations, more slot iterations).
- **Stability debt on the paid axis:** 2 of 4 paid-axis 5k draws detonated
  (β1=0 gradient explosion, no spectral guard in the winner recipe; the
  same command retried clean both times). The guard question from
  [2026-08-30-l2cap-winning-recipe.md](2026-08-30-l2cap-winning-recipe.md)
  reopens for paid arms specifically.
- The write-side results stay useful at second order: the content seed (W1)
  is worth 0.13 nats of absolute CE over the boundary seed and costs
  nothing; a future arm may keep it.
- Instrumentation debt: the train-line `proxy=`/tflops accounting uses
  slot-depth math and underreports A2 ~4x (wandb perf metrics too);
  `val/layer_passes_per_token` is the direct instrument until fixed.
- 5k-scale caveat: A2's absolute-CE edge over a healthy notul draw is
  inside the n=1 noise band; the earning number is paired/within-run and
  robust. A 20k confirmation is gated on the hybrid design, not run first.
