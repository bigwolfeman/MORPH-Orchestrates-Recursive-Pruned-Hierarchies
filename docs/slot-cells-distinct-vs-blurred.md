# Slot cells: distinct vs blurred

Written 2026-09-27 13:34. One table across the slot-cell experiments of 2026-09-13 to 2026-09-27. The
question it answers: what happens to the cells when the design lets them blur (copies, a mean,
a mix the reader is trained on), and what happens when every cell must stand alone (WTA).
Each number links to the filing that owns it; do not restate a number here without its source.

All runs: strict geometry, seed 1, 5000 steps, OpenWebText, d=1024. "epivol" is the within-slot
volume + epiplexity diversity term (`tul.fan_repel_mode: epivol`, lambda 0.1, passes 1-2).
"WTA" is `tul.fan_all_wta_lambda: 1.0`: on each span the best single cell alone is trained to
carry the span (Multiple Choice Learning).

| Design | epivol | WTA | What the cells became | Result | Source |
|---|---|---|---|---|---|
| Thought Register (4 looped cells, shared write) | no | no | copies: cell rank 1.2445 of 4, within-slot cosine 0.9432 | no gain over a wider prefix | [failures/2026-09-13-arc-thought-register.md](../lab/experiments/failures/2026-09-13-arc-thought-register.md) |
| Fan, no diversity term (norepel / mean mix) | no | no | copies, stream cosine 0.94-0.997 | oracle over streams only 0.013 / 0.016 below the mixture | [failures/2026-09-19-lxtul-fan4.md](../lab/experiments/failures/2026-09-19-lxtul-fan4.md), [failures/2026-09-20-lxtul-fan4-epivol.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md) |
| Fan + epivol, softmax mix into one cell | yes | no | a clean simplex (rank 2.823 of 3 at pass 1), but stream 0 is the best stream on 71 % of spans (`fan/oracle_pick0` 0.711) | the other three streams 0.14-0.42 nats worse than the mixture: the coda reads one stream | [failures/2026-09-20-lxtul-fan4-epivol.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md) |
| a1: fp01 with 4 prefix cells, same content | no | no | copies | +0.0074 vs fp01 on the same tokens (worse) | [successes/2026-09-26-slot-channel-arms-ab.md](../lab/experiments/successes/2026-09-26-slot-channel-arms-ab.md) |
| fan4-all: epivol + WTA + all 4 cells written 1:1, the coda selects per token | yes | yes | distinct cells, rank 2.820 of 3 at pass 1 (2.060 at pass 6) | -0.0342 [-0.0367, -0.0316] vs pk4 at depth 6 | [successes/2026-09-20-lxtul-fan4-all.md](../lab/experiments/successes/2026-09-20-lxtul-fan4-all.md) |
| a2: fan4-all on the fp01 recipe | yes | yes | distinct cells | -0.0472 vs fp01; gap to plain +0.2181; K1-K6 +0.0057 | [successes/2026-09-26-slot-channel-arms-ab.md](../lab/experiments/successes/2026-09-26-slot-channel-arms-ab.md) |
| (b) lxfan4-wta: a2 + the 4 LX code rollouts | yes | yes | distinct cells under each code (cell rank 2.72, stream rank 2.92 / 2.54 at passes 1 / 6) | a2's CE (+0.0014 [-0.0010, +0.0037]) with 2.8x its depth use: K1-K6 +0.0162, K3-K6 +0.0020 (a2 +0.0005) | [failures/2026-09-26-lx-credit-arms.md](../lab/experiments/failures/2026-09-26-lx-credit-arms.md) |
| (a) lxfan4: (b) without WTA | yes | no | a committee: a single cell alone 0.164 nats behind the joint read (b: 0.049); cell rank 2.22 (b 2.72); stream rank 1.97 of 3 at pass 6 (b 2.54) | draw 1 detonated at 3311; draw 2: +0.0303 [+0.0280, +0.0325] vs (b) on the same tokens, with the same depth use (K1-K6 +0.0161, K3-K6 +0.0024). lxfan6 (6 cells, no WTA) detonated 2 of 2 | [failures/2026-09-27-lx-soft-fan-retry.md](../lab/experiments/failures/2026-09-27-lx-soft-fan-retry.md) |

## Reading

- The cells blur whenever nothing asks a single cell to be useful by itself: a shared write, a
  mean, or a reader trained only on a mixture. The epivol filing named this first: "Diversity
  is not the limit; the READER is: the coda trained on the mixture only."
- WTA is that per-cell signal. With WTA and a write-all read, the cells stay distinct and the
  channel pays. Removing only WTA (under the codes) turns the cells into a committee and costs
  0.030 nats. 3 of 4 soft draws detonated, and all 4 grew the pass-0 write past 10x. Amended
  2026-09-28: WTA does not hold that write. (b) at seed 2 grew it to 23x and stayed healthy
  ([successes/2026-09-27-lxfan-wta-seed2.md](../lab/experiments/successes/2026-09-27-lxfan-wta-seed2.md)),
  so the write's size does not by itself separate the fans that detonated.
- The blur that hurt is a blur of STATES (or of what the reader is trained on). The LX Bayes
  read mixes PREDICTIONS, which is the correct use of K hypotheses for next-token CE.

## Not yet shown

- WTA's effect is isolated only WITH the codes: (a) draw 2 vs (b), one surviving draw. WTA buys
  0.030 nats of CE and no depth; the codes buy the depth (K1-K6 ~0.016 with or without WTA).
- Missing control: a2 with WTA 0 and no codes (epivol kept), the one-factor WTA test with
  nothing else changed.
- (b)'s depth use is one seed.
- Every row is a 5k-step reading. A 5k CE or K-curve difference is not a verdict on a looped
  model (deep models converge slower); read the direction, not the size.
