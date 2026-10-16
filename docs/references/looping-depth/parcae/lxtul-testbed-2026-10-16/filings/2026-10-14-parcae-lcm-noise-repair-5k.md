# Parcae LCM reference: does heavy concept noise repair the predicted-concept decode?

Status: failure (prereg written 2026-10-14 before any run; filed 2026-10-15)

## Question

The continuous LCM reference (`2026-10-14-parcae-lcm-reference-5k.md`) decodes a predicted concept at
CE 15.10 (Base-LCM, MSE) and 13.49 (diffusion), above uniform (10.80), against 0.157 from the true
concept. The autoencoder saw noise of only 0.1 x the concept RMS, so a regressed concept sits off the
decoder's manifold. NCM's answer is to corrupt the decoder's input in training. If heavy noise repairs
the continuous decode, a codebook is not what fixes it (Wolfe's "almost certainly not a code book"),
which the discrete span NCM (`2026-10-14-parcae-span-ncm-5k.md`) cannot tell alone.

## Arms (branch lxtul/ncm; config overrides only; seed 1; 5000 steps)

| arm | what |
| --- | --- |
| noisy concept autoencoder | `arm=lcm_ae arm.lcm.ae_noise=0.5` |
| noisy Base-LCM | `arm=lcm_mse` on the noisy autoencoder |
| noisy diffusion LCM | `arm=lcm_diffusion` on the noisy autoencoder |

The existing noise-0.1 runs are scored first with the fixed `lcm_eval` (the old chain's eval failed on an
argparse error: `--compare` swallowed the run list).

## Predictions

1. Noisy Base-LCM ce_pred@6 at least 2.0 nats below the noise-0.1 Base-LCM's (60 %).
2. Noisy Base-LCM ce_pred@6 below 10.80 (70 %).
3. Noisy autoencoder ce_oracle above the noise-0.1 one's 0.157 (90 %).
4. Noisy diffusion LCM ce_pred@6 below noisy Base-LCM's (55 %).

## Method

GPU chain `morph-scratch/abq/ncm/gpu_todo.sh`, after the span NCM runs. Eval with `lxtul.lcm_eval` on
the rows of minimal-5k-s2. Amendments get a date and a reason here.

## Results

`lxtul.lcm_eval` on the rows of minimal-5k-s2 (480,118 scored tokens, the same set as the noise-0.1
runs), `morph-scratch/abq/ncm/eval-n05/`. Uniform is 10.80.

| run | ae noise | CE@1 | CE@6 | K1-K6 [95 % CI] | ce_oracle | ce_null |
| --- | --- | --- | --- | --- | --- | --- |
| Base-LCM | 0.1 | 14.9201 | 14.9124 | +0.0077 [-0.0015, +0.0186] | 0.2328 | 14.6326 |
| Base-LCM | 0.5 | 12.8861 | 12.8326 | +0.0535 [+0.0453, +0.0630] | 0.2729 | 11.7577 |
| diffusion LCM | 0.1 | 13.4497 | 13.4475 | +0.0021 | 0.2328 | 14.6326 |
| diffusion LCM | 0.5 | 11.6226 | 11.6196 | +0.0030 | 0.2729 | 11.7577 |

Same tokens: minimal-5k-s2 3.7500, gpt8 + pointer s1 3.6920.

Prediction check:
1. HOLDS. Noisy Base-LCM 12.833 is 2.08 nats below the noise-0.1 one's 14.912.
2. FAILS. 12.833 is still above uniform (10.80).
3. HOLDS. Noisy autoencoder ce_oracle 0.273, above 0.157 (and above 0.233 on this set).
4. HOLDS. Noisy diffusion 11.620 below noisy Base-LCM 12.833.

## Verdict

Failure on prediction 2. Heavy concept noise helps the continuous LCM by about 2 nats and costs the
autoencoder 0.04 nats, but every stage-2 arm still decodes above uniform; only the noisy diffusion arm
beats its own zero concept, by 0.14 nats. The continuous concept interface is not repaired by noise at
5k. One side reading: the noisy Base-LCM backbone shows K1-K6 +0.054, the largest depth reading of the
batch, on a model 9 nats from the data, so it is not a usable depth result.

## Updated hypothesis

Corruption narrows the train/predict gap but does not close it for a regressed continuous concept.
The discrete span NCM (pure arm, same scored set) reaches 6.26 at m = 4 and 5.52 at its best m, 8.6 and
7.2 nats below the continuous arms: codes, not noise, are what repair the interface.
