# Planned: is LXTUL's gap to plain concentrated on tokens plain can copy from far back?

Status: success

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

## Results

Run 2026-10-04 on one CPU core (4 s). Script `lab/divergence/exact_recall_gap.py`, test
`tests/test_exact_recall_gap.py` (5 passed). Artifact:
[`../results/2026-10-04-exact-recall-gap-probe/exact_recall_gap.json`](../results/2026-10-04-exact-recall-gap-probe/exact_recall_gap.json).
Gate: the overall paired gap reproduces +0.3564 [+0.3356, +0.3791] with `sweep_score.paired`.
461,280 targets at i >= 64; CIs are 95 % over rows (1000 resamples). The orchestrator reran
the script and got the same table.

| bucket | token share | mean gap | share of gap | plain CE | LXTUL CE |
| --- | --- | --- | --- | --- | --- |
| far bigram repeat | 0.190 | +1.335 [+1.258, +1.419] | 0.667 | 1.44 | 2.78 |
| ... at 33-64 | 0.051 | +1.778 [+1.660, +1.904] | 0.241 | 0.80 | 2.57 |
| ... at 65-256 | 0.097 | +1.596 [+1.511, +1.695] | 0.408 | 1.18 | 2.78 |
| ... at 257+ | 0.041 | +0.171 [+0.144, +0.195] | 0.019 | 2.86 | 3.03 |
| far token repeat | 0.352 | +0.181 [+0.174, +0.188] | 0.168 | 3.19 | 3.37 |
| ... at 33-64 | 0.127 | +0.204 | 0.069 | 2.60 | 2.80 |
| ... at 65-256 | 0.165 | +0.190 | 0.083 | 3.29 | 3.49 |
| ... at 257+ | 0.060 | +0.104 | 0.017 | 4.13 | 4.24 |
| novel | 0.458 | +0.136 [+0.130, +0.143] | 0.165 | 5.05 | 5.19 |

| prediction | reading | held |
| --- | --- | --- |
| R-1 far bigram gap >= 2x novel | 9.8x | yes |
| R-2 far bigram gap share >= 1.5x token share | 3.5x (66.7 % of the gap on 19 % of tokens) | yes |
| R-3 novel gap above 0 | +0.136, CI above 0 | yes |
| R-4 gap at 33-64 no larger than at 257+ | 10.4x larger | no |

## Verdict

Success on the question asked (no pass rule was written; three of four predictions held).
LXTUL's gap is exact copying: bigram repeats 33-256 tokens back are 15 % of tokens and
65 % of the gap. R-4 failed in an informative way: the gap falls off a cliff at 256 tokens,
which is plain's exact sliding-window size (`model.window_size: 256`). Past it, plain sees
the past only through compressed attention, its copy CE rises 1.18 -> 2.86, and LXTUL's
cells nearly match it (+0.17). Where both models must compress, the cells are about as good
as plain's compression; the gap is plain's raw window.

Not verified: frequent bigrams and document separators are not filtered; `far_gap` and
`min_i` were not varied; the 257+ bucket is a different population (bigrams with no repeat
inside 256), so the window explanation is consistent with the data, not proven. A direct
test is a plain model with a shorter window, or LXTUL with a raw window.

## Updated hypothesis

The cross-span gap is mostly the coda's lack of a raw local window, not the cells'
capacity. Item 4 of the CE queue (a raw token window for the coda across span boundaries)
is the main lever, sized up to plain's 256. More cells (item 1) and wider writes (item 2)
compete on the 257+ and novel parts, which hold about 20 % of the gap. Open risk for item 4:
every arm whose coda read raw tokens across spans lost its loop contribution.
