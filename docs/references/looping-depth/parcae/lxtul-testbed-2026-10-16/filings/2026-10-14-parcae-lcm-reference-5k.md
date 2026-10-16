# LCM-Parcae: a pure Large Concept Model reference on the TUL rows

Status: failure (prereg written 2026-10-14 before any GPU run; filed 2026-10-15)

## Question

Wolfe (2026-10-14): build a pure LCM (Meta FAIR, Dec 2024) on Parcae as a reference. It is the
hierarchy with nothing of ours: a concept autoencoder stands in for the frozen SONAR encoder/decoder, a
looped Parcae backbone predicts the next span's concept, and the decoder reads only that concept and the
span's own tokens (no token memory, no pointer). Two readings: (a) what the hierarchy alone costs against
the minimal slot loop on the same tokens; (b) whether a looped concept backbone earns depth when the
decoder CANNOT do the cross-span work, the cleanest test of span-level depth on web text.

## Arms (`lxtul/lcm.py`, branch lxtul/lcm; seed 1, 5000 steps each)

| arm | what |
| --- | --- |
| concept autoencoder | stage 1: span-local encoder (4 layers, attention pool to one RMS-normed vector), span-local decoder (4 layers, concept as prefix), fixed noise 0.1 on the concept |
| Base-LCM | stage 2: frozen stage 1; Parcae 4/6/4 backbone over the concept sequence, Poisson depth mean 6, MSE to the next concept |
| diffusion LCM | stage 2: the same backbone + a 2-layer position-wise denoiser, cosine schedule, 40 sampling steps; eval CE = -log mean over 8 sampled concepts |

Eval (`lxtul/lcm_eval.py --rows-from minimal-5k-s2 --compare minimal-5k-s2`): the target tokens of complete
spans with index >= 1 of minimal-5k-s2's eval rows; minimal-5k-s2 is scored on exactly those tokens.
ce_pred (predicted concept), ce_oracle (true concept: the autoencoder ceiling), ce_null (zero concept),
depth sweep 1..8 of the backbone.

## Predictions

1. Every run finishes with finite losses (85 %).
2. Base-LCM ce_pred@6 is at least 0.30 nats worse than minimal-5k-s2 on the same tokens (75 %).
3. ce_oracle is at least 1.0 nats below Base-LCM ce_pred (80 %).
4. Diffusion LCM's marginal ce_pred@6 is below Base-LCM's (55 %).
5. Base-LCM backbone K1-K6 at least +0.01 (40 %).
6. Base-LCM ce_pred@6 below ce_null (90 %).

## What each outcome changes

- 5 holds: span-level depth exists on web text at 5k when nothing else can do the cross-span work; our TUL
  loop is flat because the coda does that work, and the shallow-coda result should agree.
- 5 fails: even a backbone that alone carries all cross-span context does not use depth at 5k; span-level
  depth is not demanded by this data at this horizon.
- 2 by a wide margin: the token memory and the end-to-end likelihood are what make TUL competitive; that
  is the paper's contrast with LCM.

## Method

`morph-scratch/abq/lcm/gpu_todo.sh` from the lxtul/lcm worktree, steps under the GPU lock between the
queue's jobs. CPU tests: `lxtul/tests/test_lcm.py` 20 passed. Amendments get a date and a reason here.

## Results

The chain's own eval died on an argparse error (`--compare` swallowed the run list); the fixed
`lcm_eval` (branch lxtul/ncm) scored the runs on 2026-10-14 23:47, `morph-scratch/abq/lcm/eval/`.
Scored set: target tokens of complete spans with index >= 1 of minimal-5k-s2's 480 eval rows, 480,118
tokens. Uniform is ln 49152 = 10.80.

| run | CE@1 | CE@6 | K1-K6 [95 % CI] | ce_oracle | ce_null |
| --- | --- | --- | --- | --- | --- |
| Base-LCM (MSE) | 14.9201 | 14.9124 | +0.0077 [-0.0015, +0.0186] | 0.2328 | 14.6326 |
| diffusion LCM | 13.4497 | 13.4475 | +0.0021 | 0.2328 | 14.6326 |
| minimal-5k-s2, same tokens | 3.7500 | 3.7500 | 0 | | |
| gpt8 + pointer s1 / s2, same tokens | 3.6920 / 3.6843 | | | | |

The concept autoencoder: train-time val 0.157, 0.233 on this set.

Prediction check:
1. HOLDS. All three runs finished with finite losses.
2. HOLDS. Base-LCM is 11.16 nats worse than minimal-5k-s2 on the same tokens.
3. HOLDS. ce_oracle 0.233 is 14.7 nats below Base-LCM's ce_pred.
4. HOLDS. Diffusion 13.45 against Base-LCM 14.91.
5. FAILS. Base-LCM K1-K6 +0.0077, and its interval crosses 0.
6. FAILS. Base-LCM ce_pred 14.91 is ABOVE ce_null 14.63: the predicted concept decodes worse than a zero concept.

## Verdict

Failure on predictions 5 and 6. The continuous LCM does not work as a reference at 5k. A regressed
concept sits off the decoder's manifold: it decodes 0.28 nats worse than no concept, and both stage-2
arms read above uniform. The backbone's depth reading cannot be judged on a model this far from the
data. The autoencoder itself is good (0.233), so the failure is the prediction-to-decoder interface,
not the compression.

## Updated hypothesis

The interface needs the decoder to see predicted-like inputs. Tested next: heavy concept noise in the
autoencoder (`2026-10-14-parcae-lcm-noise-repair-5k.md`) and discrete codes (span NCM,
`2026-10-14-parcae-span-ncm-5k.md`), both on branch lxtul/ncm.
