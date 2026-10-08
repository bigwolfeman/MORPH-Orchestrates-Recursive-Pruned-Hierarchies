# Plain Parcae + pointer head: the like-for-like gap

Status: failure (written 2026-10-06 21:11 CDT, before the run; 9 GPU geometry tests and a
60-step smoke only)

## Question

Strict LXTUL + pointer head reads 0.026 BETTER than plain at 5k (`failures/2026-10-06-parcae-copy-heads.md`),
but plain had no pointer. With the same head on plain, what is the gap?

## Run

`arm=plain_pointer` (lxtul/plain_pointer.py: the same PointerHead, 4 heads, gated with plain
Parcae's logits; no slots). 5000 steps, seed 1. Then `lxtul.mixture_off_eval` (head off), and two
repeat sweeps of plain-5k with different RNG states: plain Parcae's recurrent state starts from
random noise even in eval (found while testing: the same input twice differs by 0.05 in a logit),
so every plain number so far is one draw; the repeats give that noise on the 480-row CE.

## Predictions

1. Plain + pointer CE at most 3.81 (a gain of at least 0.02 over plain's 3.8303) (60 %).
2. Strict + pointer (3.8047) minus plain + pointer, paired: at most +0.03 (55 %).
3. Plain + pointer K1-K6 at least +0.03 (plain +0.046) (70 %).
4. Head off: within 0.03 of plain's 3.8303 (70 %).
5. The two plain-5k repeat sweeps agree to within 0.002 on CE@6 (80 %).
6. No raise.

## Method

`/home/wolfe/morph-scratch/parcae/queue_plainptr.sh`.

## Results

Run 21:11-21:36 CDT at 0583651, 480 paired rows (`results/2026-10-06-parcae-plain-pointer/`).

| run | CE@1 | CE@6 | K1-K6 | gap to plain + pointer | head OFF CE@6 |
| --- | --- | --- | --- | --- | --- |
| plain + pointer | 3.755 | 3.7072 | +0.0478 [+0.0466, +0.0489] | - | 4.1310 |
| plain | 3.876 | 3.8303 | +0.0462 | +0.1231 [+0.1111, +0.1360] | - |
| strict LXTUL | 4.183 | 4.1242 | +0.0585 | +0.4171 [+0.3884, +0.4477] | - |
| strict LXTUL + pointer | 3.863 | 3.8047 | +0.0579 | +0.0975 [+0.0933, +0.1017] | 4.1661 |

Plain eval noise (`plain_noise.log`): plain-5k CE@6 3.83030 and 3.83033 under two RNG draws.

| # | prediction | result |
| --- | --- | --- |
| 1 | plain + pointer CE at most 3.81 | holds (3.7072) |
| 2 | strict + pointer minus plain + pointer at most +0.03 | fails (+0.0975) |
| 3 | plain + pointer K1-K6 at least +0.03 | holds (+0.0478) |
| 4 | head off within 0.03 of plain's 3.8303 | fails (4.1310, +0.30) |
| 5 | plain repeat sweeps agree within 0.002 | holds (0.00003) |
| 6 | no raise | holds |

**Normalisation caveat (found 2026-10-06 21:20 CDT, porting the head to MORPH).** In
`lxtul/copy_heads.PointerHead` as run here, the mass each head puts on its null (abstain) key is
dropped, so the mixture sums to less than 1. The scored probability of the true token can only be
too SMALL, so every pointer CE reported from this code is pessimistic (an upper bound on the
normalised mixture's CE), by an amount not measured. The MORPH port (`morph/model/tul_pointer.py`,
c7424802) returns the null mass to the model; a re-run of the Parcae pointer arms with that rule
is owed before the pointer numbers are final.

## Verdict

Failure on 2 and 4. Like for like, the pointer closes about two thirds of strict LXTUL's gap
(+0.294 without heads, +0.098 with a head on both). The head is worth 0.12 nats to plain too, and
plain hands it its copying: with the head off, plain + pointer reads 4.131, where strict LXTUL
sits. With the heads off the two models' own CEs differ by 0.035, so most of the remaining
+0.098 is the difference between the two HEADS: plain's keys see the whole context, the strict
keys their own span and the loop's cells.

## Updated hypothesis

The pointer belongs in the LXTUL recipe (Wolfe 2026-10-06: "serviceable for lxtul"). The open
gap is a key-quality question for the strict head. Next: re-run both pointer arms with the
normalised head; a gap probe of strict + pointer against plain + pointer; then a 10k pair.
