# Parcae learned copy channels: pointer head and identity reach

Status: failure (written 2026-10-06 19:33 CDT, before any run; unit tests, GPU geometry
tests and 60-step smokes only)

## Question

The exact-copy cache closed strict LXTUL's gap (+0.294 -> +0.022, two seeds) with the loop kept
(`failures/2026-10-06-parcae-copy-cache.md`, `successes/2026-10-06-parcae-copy-cache-seed2.md`).
Wolfe (2026-10-06): a winning recipe that most researchers would call a hack; build less hacky
options, "something I would be willing to include in the final version". Two learned channels:

1. **Pointer head** (pointer-generator / pointer sentinel): output-only, learned, fuzzy matching.
   Does a learned head match or beat exact counting?
2. **Identity reach**: tokens attend earlier tokens' RAW embedding windows before the coda, so
   copied identity enters the hidden state. Does the loop keep its job when identity, but no
   computation, crosses spans inside the model?

## Mechanism

`lxtul/copy_heads.py`. Pointer: 4 heads, dh 64, keys and queries from coda outputs (strict:
span-local), each head's attention summed onto the attended tokens' targets, null key, gate
over (model, heads) from the token's coda output. Reach: 8 heads, dh 128, query from the coda
input, keys and values from [e(x_{j-1}), e(x_j), e(x_{j+1})] for j < i, null key, output
projection zero-initialised, slot cells never written. Tests: `test_copy_heads.py` (brute force
for both, causality; a j <= i sabotage fails both brute-force tests), `test_geometry.py` 8 passed
(pointer: end-to-end causal NLL, a training step reaches the head; reach: causal, span s reads
span s-1 with every cell write zeroed, and the slot cells after the coda are bit-identical with
the branch live).

## Runs (5k, seed 1, round-1 recipe, 480 paired rows)

| run | config |
| --- | --- |
| pointer head | `arm=lxtul_pointer` |
| identity reach | `arm=lxtul_reach` |

Then: `lxtul.mixture_off_eval` on the pointer run (leak check), `lxtul.gap_probe` on both
(per-token buckets; for the pointer run the probe scores the model WITHOUT its head).

References: plain 3.8303; strict LXTUL 4.1242 (K1-K6 +0.0585, write worth +0.32); strict + cache
3.8518 / 3.8533 (gap +0.022 / +0.023, K1-K6 +0.049 / +0.069); open LXTUL 3.8508 (K1-K6 +0.0042).

## Predictions

1. Pointer: gap to plain at most +0.05 (60 %).
2. Pointer: CE below strict + cache's 3.8518 (40 %).
3. Pointer: K1-K6 at least +0.03 (70 %).
4. Pointer, head OFF: CE within 0.03 of strict LXTUL's 4.1242 (75 %).
5. Reach: gap to plain at most +0.10 (55 %).
6. Reach: K1-K6 at least +0.03 (45 %): the bypass risk.
7. Reach: zeroing the cell write still costs at least +0.16 (half of strict LXTUL's +0.32) (45 %).
8. Both: training tok/s within 10 % of strict LXTUL's 24.4k (70 %).
9. No run raises.

What each outcome changes: pointer at or under the cache's CE with the loop kept -> the pointer
replaces the cache in the recipe. Reach closing the gap with K1-K6 and the cell worth kept ->
identity reach is the geometry rule for the final version (no output mixture). Reach closing the
gap while K1-K6 or the cell worth collapses -> identity in the hidden state is a bypass, and the
copy channel must stay output-only.

## Method

`/home/wolfe/morph-scratch/parcae/queue_copyheads.sh` (one job at a time under the GPU lock).

## Results

Runs 19:35-20:22 CDT at bc67bd0, 480 paired rows (`results/2026-10-06-parcae-copy-heads/`).

| run | CE@1 | CE@6 | K1-K6 | gap to plain 5k | tok/s |
| --- | --- | --- | --- | --- | --- |
| plain | 3.876 | 3.8303 | +0.0462 | - | 21.8k |
| strict LXTUL | 4.183 | 4.1242 | +0.0585 | +0.2939 | 24.4k |
| strict + exact cache | 3.901 | 3.8518 | +0.0491 | +0.0215 | 22.4k |
| strict + pointer head | 3.863 | 3.8047 | +0.0579 [+0.0558, +0.0600] | -0.0256 [-0.0381, -0.0142] | 22.3k |
| strict + identity reach | 4.154 | 4.1089 | +0.0446 [+0.0428, +0.0463] | +0.2786 [+0.2607, +0.2987] | 23.1k |

Pointer leak check (`pointer_off.json`): head OFF reads CE@1 4.2220, CE@6 4.1661, K1-K6 +0.056,
0.042 WORSE than strict LXTUL: the model learned no copy path of its own and hands copying to the
head. The head is worth 0.361 nats (the exact cache 0.266).

Gap probe (the model's own distribution; for the pointer arm, without its head):

| run | far-bigram gap | other gap | novel-token gap | cell write worth |
| --- | --- | --- | --- | --- |
| strict LXTUL | +1.411 | +0.051 | -0.034 | +0.316 |
| identity reach | +1.316 | +0.053 | -0.028 | +0.257 |
| pointer (head off) | +1.579 | +0.065 | -0.071 | +0.253 |

The reach branch learned (output projection norm 0 -> 12.3) yet removed only 0.095 of the 1.41
far-bigram gap.

| # | prediction | result |
| --- | --- | --- |
| 1 | pointer gap <= +0.05 | holds (-0.0256) |
| 2 | pointer CE below the cache's 3.8518 | holds (3.8047) |
| 3 | pointer K1-K6 >= +0.03 | holds (+0.0579) |
| 4 | pointer head OFF within 0.03 of 4.1242 | fails, harmless side (4.1661, +0.042 worse) |
| 5 | reach gap <= +0.10 | fails (+0.2786) |
| 6 | reach K1-K6 >= +0.03 | holds (+0.0446) |
| 7 | reach cell write worth >= +0.16 | holds (+0.257) |
| 8 | tok/s within 10 % of 24.4k | holds (22.3k, 23.1k) |
| 9 | no raise | holds |

## Amendment (2026-10-06 21:40 CDT, after filing)

**Normalisation caveat (found 2026-10-06 21:20 CDT, porting the head to MORPH).** In
`lxtul/copy_heads.PointerHead` as run here, the mass each head puts on its null (abstain) key is
dropped, so the mixture sums to less than 1. The scored probability of the true token can only be
too SMALL, so every pointer CE reported from this code is pessimistic (an upper bound on the
normalised mixture's CE), by an amount not measured. The MORPH port (`morph/model/tul_pointer.py`,
c7424802) returns the null mass to the model; a re-run of the Parcae pointer arms with that rule
is owed before the pointer numbers are final.

## Verdict

Failure on 4 and 5. The pointer head is the result: output-only, learned, fuzzy, it takes strict
LXTUL from 0.294 behind plain to 0.026 ahead at 5k, keeps K1-K6 at strict LXTUL's level (+0.058;
+0.056 with the head off), keeps most of the cell worth (+0.253 of +0.316), costs 9 % throughput,
and beats exact counting by 0.047. The model's own novel-token CE is 0.071 better than plain's
with the head off: copying handed to the head left capacity for the rest.

Identity reach does not work as built: raw-embedding keys through one dot product barely learn
induction (far-bigram gap 1.41 -> 1.32). That is a statement about this module, not about
identity in the hidden state in general; the pointer's advantage is that its keys and queries
are hidden states, which already encode the bigram and its context.

"Beats plain" is against plain as built. Plain has no pointer; the exact cache gave untrained
plain +0.030, so a trained pointer on plain likely gains too.

## Updated hypothesis

The strict LXTUL recipe gains a pointer head: the copy channel stays output-only and the loop
keeps its job. Next: seed twin; plain + pointer for the like-for-like gap; 10k; MORPH port.
