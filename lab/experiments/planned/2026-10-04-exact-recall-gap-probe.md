# Planned: is LXTUL's gap to plain concentrated on tokens plain can copy from far back?

Status: planned

Date: 2026-10-04 14:25 CDT, before the probe is built or run. Item 6 of the
[CE test queue](../../../.agents/notes/proposed/architecture/2026-10-04-lxtul-ce-test-queue.md).

## Question

LXTUL 10k trails plain 10k by +0.356 nats on the 480 sweep rows. The cells hold about 40 % of
the cross-span value. Hypothesis: much of the other 60 % is exact recall, where plain's
attention copies a token it saw far back, and a few compressed cells per span cannot hold
many exact tokens. If so, the gap concentrates on tokens that repeat an earlier bigram.

## Method

Offline, CPU only, no model forward: the two existing per-token sweep files
(`sweep_..._rank-cnorm-10k_10000...tokens.npz` and `plain-10k/sweep_plain-panel-norm-match_10000...tokens.npz`),
paired by `tok_index` as `paired_vs_ruler.py` pairs them, LXTUL at depth 6 against plain at
depth 6. The token ids come from the same validation rows the sweep reads. For a target
token at row position i (i >= 64), with the nearest earlier occurrence j:

- **far bigram repeat**: the bigram (x[i-1], x[i]) occurred with its second token at j <= i-33;
- **far token repeat**: x[i] occurred at j <= i-33, the bigram did not;
- **novel**: x[i] did not occur at j <= i-33.

Distance 33 or more guarantees the earlier copy is outside the target's own span (span cap
32), so only the slot channel can carry it to LXTUL. Each repeat bucket is also split by the
distance i - j: 33-64 (about the previous span), 65-256, 257+. Report per bucket the token
count, the mean paired gap with a 95 % bootstrap CI over rows, and the bucket's share of the
total gap against its share of tokens.

## Predictions (mine, orchestrator)

- **R-1**: per-token gap on far bigram repeats is at least 2x the gap on novel tokens. 60 %.
- **R-2**: far bigram repeats carry a share of the total gap at least 1.5x their share of
  tokens. 60 %.
- **R-3**: the gap on novel tokens is above 0, CI above 0 (plain also wins on topic and style,
  not only copies). 85 %.
- **R-4**: among far bigram repeats, the gap at distance 33-64 is no larger than at 257+
  (the slot channel loses exact detail at every distance, not only far away). 50 %.

## What each outcome changes

- R-1 and R-2 hold: exact recall is the bottleneck. Wider writes (item 2) and, if R-4 fails,
  the previous-span window (item 4) move up the queue.
- The gap is flat across buckets: it is general context, not copying. Cell content and the
  reader (items 1 and 3) are the levers.
