# Parcae flat references with the pointer head at 5k

Status: failure (prediction 3 fails: flat14 + pointer is 0.048 better than plain + pointer; filed 2026-10-14)

## Question

Wolfe's publishing claim is Parcae vs LXTUL-Parcae, with a flat transformer as the reference. Every
slot-loop candidate carries the normalised pointer head, and plain Parcae has it (plain + pointer); the
flat model does not yet. The only flat number at 5k is gpt8 with no pointer, 3.8086. Where do the two
flat sizes sit with the same head, on the same rows and tokens?

## Arms (5k, batch 6, `lxtul.train` defaults, the SAME `PointerHead` as plain + pointer)

| arm | config | seeds | what it matches |
| --- | --- | --- | --- |
| gpt8 + pointer | `arm=gpt8_pointer` | 1, 2 | 8 layers, the slot loop's prelude + coda depth (403M params) |
| flat14 + pointer | `arm=flat14_pointer` | 1, 2 | 14 layers = Parcae's prelude + one core pass + coda (630M) |

Bench speeds (`morph-scratch/abq/pz-bench/bench4.md`, batch 6 train): gpt8 + pointer 79.8 ms (77.0k tok/s),
flat14 + pointer 123.3 ms (49.8k), plain + pointer 281.8 ms, minimal 153.4 ms.

## Predictions

1. gpt8 + pointer CE, mean of 2 seeds, in [3.670, 3.750]: the pointer was worth 0.110 to plain at 5k
   (3.8303 to 3.7206), and gpt8 with no pointer is 3.8086 (60 %).
2. flat14 + pointer is better than gpt8 + pointer by at least 0.010, mean of 2 seeds (60 %).
3. Both flat arms are within 0.030 of plain + pointer's pooled 3.7206 (55 %): at 5k a deep loop has not
   yet passed a flat model of equal width (`deep-models-converge-slower`).
4. Head-off check: each flat + pointer run's head-off CE is above its head-on CE (95 %).

## Method

Code: testbed (merge of `lxtul/bench4` and `lxtul/minimal`, b083841 plus a dead-code cleanup). Queue
`morph-scratch/abq/pz-panel/flat_5k.sh`, started when the minimal fan panel finishes; one job at a time
under the GPU lock; head-off check after every run. A flat model has no depth axis, so its sweep is the
same CE at every K. Amendments get a date and a reason here.

Amendment 2026-10-14 15:08: flat14 + pointer seed 2 was not run. Wolfe stopped the remaining seeds of
this legacy panel when the redesign started; gpt8 + pointer seed 2 had already finished. Prediction 2 is
read on flat14 seed 1 against the gpt8 mean.

## Results

Same 510,003 tokens as plain + pointer and the 4-cell fan (sweep.json counts match). A flat model has the
same CE at every depth.

| arm | seed | CE | head-off CE |
| --- | --- | --- | --- |
| gpt8 + pointer | 1 | 3.7034 | 4.0874 |
| gpt8 + pointer | 2 | 3.6958 | 4.0735 |
| flat14 + pointer | 1 | 3.6731 | 4.0517 |

gpt8 + pointer mean 3.6996. References on the same tokens: plain + pointer 3.7155 (s1) / 3.7256 (s2),
pooled 3.7206; minimal fan 3.7442 (2 seeds).

Prediction check:
1. gpt8 + pointer mean 3.6996, inside [3.670, 3.750]. HOLDS.
2. flat14 + pointer (one seed) 0.0265 better than the gpt8 mean. HOLDS (one seed, see the amendment).
3. gpt8 is 0.021 better than plain + pointer, inside 0.030; flat14 is 0.0475 better, outside. FAILS.
4. Head-off above head-on in all three runs. HOLDS.

## Verdict

Failure by prediction 3, in the direction that matters: at 5k a flat model beats the looped plain Parcae
at a third (gpt8, 80 ms) or half (flat14, 123 ms) of its step time (240 ms). flat14 + pointer costs the same
FLOPs per step as the minimal slot loop (13.5 vs 13.2 TFLOP) and is 0.071 better than the minimal fan.

## Updated hypothesis

At 5k every looped arm, plain or slot, is behind flat models of equal or lower cost; this is the
deep-models-converge-slower effect (Wolfe 2026-10-14: loops need more tokens to beat flat; 5k is before the
knee). The slot channel under strict geometry also costs 0.041 against plain attention with the same token
layers (minimal fan vs gpt8 + pointer). The redesign (decode loop, LCM-Parcae reference) is judged against
flat14 + pointer and gpt8 + pointer as well as plain + pointer.
