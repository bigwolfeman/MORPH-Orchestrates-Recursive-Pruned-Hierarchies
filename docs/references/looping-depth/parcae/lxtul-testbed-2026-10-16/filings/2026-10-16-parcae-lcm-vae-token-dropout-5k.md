# Variational LCM with decoder token dropout

Status: success (filed 2026-10-16)

## Question

Does hiding some of the decoder's input tokens make the variational LCM use its concept?

## Hypothesis

I think the concept collapsed (0.018 nats per token in `lcm-vae-5k`) because the span-local decoder
learns the span from its own teacher-forced prefix faster than the concept channel forms. This is
posterior collapse (Bowman et al. 2015). Hiding decoder input tokens takes away part of that prefix, so
the decoder has a reason to read the concept.

## Arms (`lxtul/lcm.py`, key `lcm.dec_token_dropout`; seed 1, 5000 steps)

- control: `lcm-vae-5k` (dropout 0, already run).
- `lcm_vae_drop25`: each decoder input token replaced by the slot id with probability 0.25, training only.
- `lcm_vae_drop50`: the same at 0.5.

Eval has no dropout, so each score is the ordinary bound of the trained model, on the stage-2 scored set.

## Predictions

1. Main: the concept carries more information as dropout rises. KL per scored token orders
   control < drop25 < drop50.
2. The backbone's K-curve on the bound slopes down in the dropout arms, unlike the control's flat curve.
3. The bound itself: I cannot say which way it moves. More concept use helps; a decoder trained on
   hidden tokens is weaker at eval. I report it next to the control.

If 1 is supported, the LCM's collapse was the decoder shortcut, and the same lever (token dropout on the
coda input) is the first thing to try where the TUL coda bypasses the loop. If 1 is not supported, the
collapse is in the optimization of the concept channel itself (init scale, unit noise), and the next arm
tests that.

Gates (not predictions): finite losses, peak memory under 26 GiB, scored token count equal to the control's.

## Method

Code and CPU tests from a sonnet builder (`morph-scratch/abq/lcmvae-drop/REPORT.md`), GPU steps from its
`gpu_todo.sh`, then `lxtul.lcm_eval --rows-from minimal-5k-s2` over the three runs.

## Results

One seed (1) per arm, one run each, so no seed spread was measured. Sources:
`/home/wolfe/morph-scratch/abq/lcmvae-drop/eval/{lcm-vae-5k,lcm-vae-drop25-5k,lcm-vae-drop50-5k}/summary.json`
(lcm_eval on the same 480,118 scored tokens of minimal-5k-s2's rows), the runs' `summary.json` scalars under
`/home/wolfe/parcae-runs/`, and `/home/wolfe/morph-scratch/abq/lcmvae-drop/gpu_todo.log`.

Backbone depth 6:

| arm | KL per scored token | bound | recon | oracle | bound d1 minus d6 | bound minus gpt8 + pointer s1 (3.6920) |
| --- | --- | --- | --- | --- | --- | --- |
| control (dropout 0) | 0.0185 | 4.4085 | 4.3901 | 4.3843 | +0.00003 | +0.7165 |
| drop25 | 0.0420 | 4.5168 | 4.4748 | 4.4371 | +0.00008 | +0.8248 |
| drop50 | 0.1088 | 4.6833 | 4.5745 | 4.4805 | +0.00016 | +0.9913 [+0.9605, +1.0269] |

The run summaries give KL 0.01842, 0.04196, 0.10877 and the same order. All three arms: finite losses, peak
allocated 8.00 GiB (gpu_todo.log; the control's peak is not in this log), 480,118 scored tokens for each
(equal to the control). Speed not measured.

Prediction check:

1. Main: KL per scored token orders control < drop25 < drop50: 0.0185 < 0.0420 < 0.1088. Held. Each step is
   2.3x and 2.6x the one before.
2. The K-curve on the bound slopes down in the dropout arms: the bound at depth 1 minus depth 6 is +0.00008
   (drop25) and +0.00016 (drop50), against +0.00003 for the control. The order is right and the drop50
   interval excludes zero, but the size is about 0.0002 nats, where the plain Parcae loops read +0.003 to
   +0.008. Did not hold in any size that matters (flat curve). Noise-limited in direction: only one seed.
3. The bound itself (reported, not graded): it gets worse as dropout rises, +0.108 (drop25) and +0.275
   (drop50) over the control at depth 6.

## Verdict

Supported. KL per scored token rises with decoder token dropout, 0.0185 (control), 0.0420 (drop25), 0.1088
(drop50). Hiding decoder input tokens makes the concept carry more bits, so the collapse was at least partly
the decoder shortcut (the span-local decoder reading its own prefix).

What this does not show. The extra bits do not help the score: the bound is 0.108 and 0.275 nats worse, and
the recon term moves from 4.390 to 4.475 and 4.574. The oracle is only 0.044 (drop25) and 0.096 (drop50)
above the control's 4.384, so the decoder is weaker with hidden tokens and the concept pays back only part
of it. The backbone still shows no K-curve. One seed per arm; the ordering has a 2.3x to 2.6x step, which is
large against any reasonable seed spread, but the spread was not measured.

## Updated hypothesis

Token dropout buys concept information and costs likelihood, and the loop still does not use depth. The
concept carries bits but the backbone cannot turn them into a depth gain at 5k. Next question: is a K-curve
visible if the concept is forced to carry the span by a different lever (a lower-capacity decoder, or a
longer run) rather than by weakening the decoder? The sibling lever on the TUL coda (token-state dropout on
the coda input) is worth a try where the coda bypasses the loop, since the bypass is the same shortcut.

## What this changes

As the plan says: 1 is supported, so the LCM's collapse was the decoder shortcut, and token dropout on the
coda input is the first thing to try where the TUL coda bypasses the loop. Do not report the dropout bound as
an LCM improvement: it is worse than the control at every level.

## Artifacts

- `/home/wolfe/morph-scratch/abq/lcmvae-drop/eval/lcm-vae-drop25-5k/summary.json`, `.../lcm-vae-drop50-5k/summary.json`, `.../lcm-vae-5k/summary.json`
- `/home/wolfe/morph-scratch/abq/lcmvae-drop/gpu_todo.log`
- `/home/wolfe/parcae-runs/lcm-vae-5k`, `lcm-vae-drop25-5k`, `lcm-vae-drop50-5k` (`summary.json`)
