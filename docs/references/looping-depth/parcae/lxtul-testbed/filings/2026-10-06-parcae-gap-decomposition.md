# Parcae gap decomposition: what makes strict LXTUL trail plain by 0.29 nats?

Status: failure (written 2026-10-06 13:12 CDT, before either control has run; smokes of
30 steps only, read for crashes)

## Question

The strict LXTUL gap to plain is +0.27-0.28 on MORPH and +0.29 on Parcae, so it belongs to the
design, not the backbone. The bare loop (no LX objectives) still trails by +0.252, so the LX
objectives are at most about 0.04 of it. Wolfe (2026-10-06): the purpose of this round is data to
postulate what reduces the gap. Is the rest (a) cross-span information (a strict token reads
earlier spans only through one winner cell per slot) or (b) per-token compute (a strict token gets
8 layer passes; a plain token 4 + 6 x 6 + 4)?

## Arms (5000 steps, seed 1, round-1 recipe and rows)

| run | config | what it is |
| --- | --- | --- |
| gpt8 | `arm=gpt8` | Parcae's GPT, 8 layers (= prelude + coda), full causal context, no loop |
| lxtul-open | `arm=lxtul_open` | Parcae-LXTUL, tokens also read every earlier token in prelude and coda; cells strict |

References: plain-5k CE 3.8303; lxtul-5k 4.1242 / seed 2 4.1252; bare-5k 4.0826.

## Predictions

1. gpt8 CE in [3.90, 4.10] (plain at eval depth 1, 14 passes per token, reads 3.877).
2. lxtul-open CE within [-0.05, +0.04] of gpt8 (with full token context, the cells add little).
3. lxtul-open K1-K6 below strict LXTUL's +0.0585 and at most +0.03 (tokens bypass the loop).
4. The hypothesis Wolfe leans to: strict LXTUL minus lxtul-open (paired) >= +0.147, half the gap,
   i.e. cross-span information is the larger share. I give it 55 %.
5. Neither run raises.

Decomposition read from these (all paired, same 480 rows): depth share = gpt8 - plain;
cross-span share = lxtul - lxtul-open; residual = lxtul-open - gpt8 (the loop and objectives
with full context). The gap probe (lxtul/gap_probe.py, plain-5k vs lxtul-5k, per-token buckets by
position in span, span index, cross-span repeat) reads the same split per token.

## Method

lxtul/train.py, as round 1. Queued behind round 2 under the GPU lock.

## Results

480 paired rows, seed 1, 5000 steps, commit 9566285 (gpt8 needed the plain linear head: GPT's
fused head asserts vocab % 4096 and MORPH's padded vocab is 49216). Readouts and the probe's
JSON in `results/2026-10-06-parcae-gap-decomposition/`. gpt8 is checked causal on CPU (future
tokens changed, earlier logits bit-equal).

| run | CE@6 | K1-K6 | gap to gpt8 | tok/s |
| --- | --- | --- | --- | --- |
| gpt8 (8 layers, no loop) | 3.8086 | 0 | - | 80.3k |
| plain Parcae (looped) | 3.8303 | +0.0462 | +0.0217 [+0.0193, +0.0241] | 21.8k |
| lxtul-open | 3.8508 | +0.0042 [+0.0039, +0.0046] | +0.0423 [+0.0393, +0.0454] | 24.3k |
| bare strict loop | 4.0826 | -0.0001 | +0.2740 [+0.2559, +0.2945] | 37.2k |
| strict Parcae-LXTUL | 4.1242 | +0.0585 | +0.3156 [+0.2972, +0.3373] | 24.4k |

Strict minus open, paired: +0.2734 [+0.2556, +0.2942] (seed 2: +0.2744).

Decomposition of strict LXTUL's gap to plain (+0.2939):
- per-token depth: gpt8 - plain = -0.022. Depth is no part of it; 8 layers with full context
  beat the looped model at 5k.
- cross-span information (the strict mask): +0.273, 93 %.
- the rest (open LXTUL vs plain): +0.021.

Gap probe (strict LXTUL vs plain, per token): tokens whose bigram appeared in an earlier span
are 17.9 % of tokens and 85.7 % of the gap (plain 1.38, LXTUL 2.79 nats); tokens repeated from an
earlier span 102.9 %; novel tokens -5.2 % (LXTUL 0.034 better than plain). The gap grows with
span index (-0.06 in span 0, +0.33 from span 16). The cell write is worth +0.32 overall and +1.92
on a span's first token.

| # | prediction | result |
| --- | --- | --- |
| 1 | gpt8 CE in [3.90, 4.10] | fails (3.8086, better than plain) |
| 2 | lxtul-open within [-0.05, +0.04] of gpt8 | fails narrowly (+0.0423 [+0.0393, +0.0454]) |
| 3 | lxtul-open K1-K6 below +0.0585 and at most +0.03 | holds (+0.0042) |
| 4 | strict - open >= +0.147 | holds (+0.2734) |
| 5 | no run raises | holds |

## Verdict

Failure on 1 and 2, but the question is answered: the strict LXTUL gap is the strict geometry,
and inside it, copying. Per-token depth contributes nothing at 5k. With tokens free to read
earlier tokens the loop goes almost flat (K1-K6 +0.0042): it is the cross-span channel or it is
bypassed, as the 2026-09-12 MORPH strict-geometry panel found.

## Updated hypothesis

The gap is a copy-channel problem. A channel that carries token IDENTITY across spans but no
computed content (a non-parametric copy distribution, or an induction head whose keys and values
are token embeddings only) should recover most of the 1.41 nats on the 18 % far-bigram tokens
while leaving the loop the only path for anything that is not a copy. Predicted ceiling: the
open arm (+0.021 to plain); predicted bypass risk: low, because identity alone cannot carry a plan.
