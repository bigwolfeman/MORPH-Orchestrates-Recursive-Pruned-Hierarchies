# Variational LCM: the LCM reference scored by its likelihood bound

Status: success (filed 2026-10-16)

## Question

Wolfe (2026-10-15): "your LCM reference failure must be a failure of implementation or data scale."
The LCM reference (`2026-10-14-parcae-lcm-reference-5k.md`) and its noise repair
(`2026-10-14-parcae-lcm-noise-repair-5k.md`) read 11.6 to 14.9 nats per token, above uniform (10.80).
Is that the design of the reference or a real property of a pure concept model on these rows?

## Hypothesis

The reference was broken in two ways, neither of them data scale:

1. Its readout is not a likelihood. It decodes the true span from one predicted concept (Base-LCM) or
   from 8 prior samples (diffusion). The concept code is near-lossless (0.233 nats per token given the
   true concept): z is RMS-normalised to a fixed norm with noise 0.1, so 768 dimensions at about SNR 100
   hold the whole span. A predicted concept that misses that point decodes garbage, and 8 prior samples
   never land on it. The model's real likelihood needs a posterior.
2. Nothing charges the code for its information. Stage 1 makes the code as rich as it can, stage 2
   then has to predict all of it.

A variational LCM fixes both and keeps the LCM shape (span-local encoder, span-local decoder that reads
only the concept and the span's own tokens, a Parcae backbone over concepts, no token memory). Posterior
N(mu_q, I) and prior N(mu_p, I), both at a FIXED unit variance (no learned sigma). The KL is then
0.5 ||mu_q - mu_p||^2, which is the Base-LCM MSE term with a fixed scale. Trained end to end on the
negative ELBO, the encoder sends only the bits the prior can pay for, and the score is an upper bound
on token CE on the same scale as every other arm.

## Arm (`lxtul/lcm.py`, kind `lcm_vae`, branch lxtul/lcm-vae; seed 1, 5000 steps)

One stage. Loss = (recon NLL + KL) summed over complete spans / tokens of complete spans. Eval on the
stage-2 scored set (tokens of complete spans with index >= 1 of minimal-5k-s2's rows, 480,118 tokens):
the bound (one posterior sample), IWAE with 16 samples, the prior-sample CE (generation mode, a
diagnostic only), the oracle (decode from mu_q) and KL and recon per token, per backbone depth.

## Predictions

1. Every run finishes with finite losses (90 %).
2. The bound at depth 6 is below uniform, 10.80 (97 %). This is the repair test.
3. The bound at depth 6 is below the pure span NCM's best bound on the same set, 5.52 (50 %).
4. The bound at depth 6 is at least 1.0 nats above gpt8 + pointer s1 on the same tokens, 3.6920 (85 %):
   a decoder with no token memory pays for it.
5. KL per scored token at depth 6 is at least 0.05 nats: the code is used, no posterior collapse (80 %).
6. Backbone K1-K6 on the bound at least +0.005 (35 %).

## What each outcome changes

- 2 holds: the old above-uniform numbers were the readout and the unpriced code, not the LCM idea and
  not data scale. The variational LCM becomes the LCM reference row in the paper's table.
- 2 fails: the bug is elsewhere (the span-local decoder or the scored set); audit before any other LCM work.
- 3 and 5 together: a continuous concept priced by the ELBO beats a residual-VQ code, which bears on
  NCM v2 (RePR) as well.
- 6 holds: a looped concept backbone uses depth when it alone carries every cross-span bit.

## Method

Code and CPU tests from a sonnet builder (`morph-scratch/abq/lcmvae/REPORT.md`), GPU smoke from its
`gpu_todo.sh`, then one 5k run and `lxtul.lcm_eval --rows-from minimal-5k-s2` against minimal-5k-s2
and gpt8-pointer-5k-s1. Amendments get a date and a reason here.

## Results

The predictions above use the old numeric format. They are graded here under the 2026-10-16 protocol. The
percent figures are not graded. One seed (1), one run. No seed spread was measured for this arm.

Sources: `/home/wolfe/parcae-runs/lcm-vae-5k/summary.json` (key `scalars`),
`/home/wolfe/morph-scratch/abq/lcmvae/eval/lcm-vae-5k/summary.json` (lcm_eval, same-token comparison),
`/home/wolfe/morph-scratch/abq/lcmvae/logs/lcm-eval.log`. Scored set: 480,118 tokens, 480 rows of
minimal-5k-s2. Uniform is 10.80 (ln 49152, from the plan).

Backbone depth 6 unless noted (eval summary):

| readout | nats per scored token |
| --- | --- |
| bound (one posterior sample) | 4.4085 |
| IWAE, 16 samples | 4.3946 |
| prior sample (generation mode) | 4.3945 |
| oracle (decode from mu_q) | 4.3843 |
| recon term | 4.3901 |
| KL per scored token | 0.0185 |
| bound at depth 1 minus depth 6 (K1-K6) | +0.00006 (run summary: 4.40855 - 4.40849) |

Same tokens, paired (480 rows), LCM bound minus run: against gpt8 + pointer s1 (3.6920) +0.7165
[+0.6837, +0.7547]; against minimal-5k-s2 (3.7500) +0.6585 [+0.6253, +0.6968]. The old LCM reference read
11.6 to 14.9 (plan); the pure span NCM's best bound was 5.52 (plan).

Prediction check:

1. Finite losses: held (the run finished, rc 0; chain log). A gate.
2. Bound at depth 6 below uniform 10.80: 4.4085, lower by 6.39. Held, far outside noise. This is the main prediction.
3. Bound below the pure span NCM's best, 5.52: 4.4085, lower by 1.11. Held.
4. Bound at least 1.0 above gpt8 + pointer s1 (3.6920): +0.7165 [+0.6837, +0.7547]. Did not hold. The gap is
   smaller than the guess.
5. KL per scored token at least 0.05: 0.0185. Did not hold. The concept is used very little.
6. Backbone K1-K6 on the bound at least +0.005: +0.00006. Did not hold. The K-curve is flat.

## Verdict

Supported. The bound-scored LCM reads 4.4085 nats per token, 6.39 below uniform (10.80), where the old
reference read 11.6 to 14.9. So the above-uniform score came from the readout (decoding from a predicted
concept, no posterior and no price on the code), not from data scale or an unusable concept model.

The caveat: the concept barely matters. KL is 0.0185 nats per token, and the oracle (decode from the true
posterior mean, 4.3843) is only 0.024 below the bound. The prior sample (4.3945) equals the bound. So the
4.41 mostly comes from the span-local decoder reading its own tokens, not from the backbone's concept. The
model is a collapsed VAE (posterior collapse), and a flat K-curve (+0.00006) matches that. The LCM idea is
scored fairly now, but its concept channel carries almost no information at 5k.

## Updated hypothesis

The bound fixes the readout and the unpriced code, but it also prices the code so hard that the encoder sends
almost nothing: the standard collapse. The next question, already planned in
`2026-10-16-parcae-lcm-vae-token-dropout-5k.md`, is whether hiding decoder input tokens forces the concept to
carry information.

## What this changes

Per the plan: 2 holds, so the old above-uniform numbers were the readout and the unpriced code. The
variational LCM becomes the LCM reference row in the paper's table, with the KL column shown beside it so
nobody reads 4.41 as a working concept channel. The comparison "3 and 5 together" (a continuous concept
priced by the ELBO beats a residual-VQ code) cannot be claimed: 3 holds, but 5 fails, and the win over the
NCM bound comes from the decoder's own token path. Prediction 6 failed: the looped concept backbone does not
use depth here.

## Artifacts

- `/home/wolfe/parcae-runs/lcm-vae-5k/summary.json`
- `/home/wolfe/morph-scratch/abq/lcmvae/eval/lcm-vae-5k/summary.json`, `sweep.json`
- `/home/wolfe/morph-scratch/abq/lcmvae/logs/lcm-eval.log`, `lcm-vae-5k.log`
