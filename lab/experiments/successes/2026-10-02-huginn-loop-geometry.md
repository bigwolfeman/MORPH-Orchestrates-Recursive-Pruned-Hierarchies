# Planned: does a loop that earns depth rotate its state or grow it?

Status: success

Date: 2026-10-02 15:18 CDT. Predictions frozen before the probe exists.

## Question

In the latent-pulled MORPH slot-loop arms, pass 2 grows the cells about 4x along one shared
direction, passes 3 to 6 keep growing it, and the slots spread apart 10x to 20x
([interim note](../../../.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md)).
Wolfe (2026-10-02): "I feel like magnitude isn't true diversity." Huginn-0125 (3.5B, trained
at recurrence mean 32) is a loop that earns depth on web text: K3-K6 +0.566, K6-K16 +0.206
on our 480 OpenWebText rows (2026-09-04 filing). What does its recurrent state do across
iterations: grow along a shared direction, or rotate at a stable scale? And does its coda
read the state's magnitude at all?

## Hypothesis

An earning loop rotates. Huginn's state norm settles within a few iterations, and its
per-iteration change is mostly perpendicular to the shared direction. Its coda ignores the
per-token scale (pre-norm), so rescaling the state to a fixed norm costs nothing, while the
state's direction carries the earning.

## Predictions (mine, orchestrator)

Readings on 64 rows x 1024 tokens of the 480-row set, at iterations k = 1..32. "v" is the
unit mean of the state over all tokens and iterations; "centred" subtracts the
per-coordinate mean over tokens; "normalised" divides each token's state by its own norm.

- **G-1**: the state's mean norm at k = 32 is within 1.5x of its norm at k = 4. 75 %.
- **G-2**: from k = 8 on, under 30 % of the mean squared step lies along v. 70 %.
- **G-3**: the cosine of the mean state to v stays below 0.7 for k >= 4. 60 %.
- **G-4 (magnitude is not read)**: rescaling each token's state at k = 32 to its norm at
  k = 4 changes CE by less than 0.01 nats. 70 %.
- **G-5 (direction is read)**: removing the v component of the state at k = 32 costs more
  than 0.05 nats. 60 %.
- **G-6**: the participation ratio of the centred, normalised states grows from k = 4 to
  k = 16 by more than 10 %. 50 %.

If G-1, G-2 and G-4 hold, an earning loop works by rotation at a fixed scale, and the
MORPH lever is to remove the magnitude channel (renormalise the cells after each pass) so
the slot loop can only rotate. If Huginn also grows a shared direction, magnitude growth is
not by itself a sign of a fake contribution, and the lever is elsewhere.

## Method

`lab/huginn/huginn_loop_geometry.py` (to be built): load Huginn-0125 from the local HF
cache in bf16 with the `~/.venvs/huginn` env, run the recurrence with the state captured
after each iteration, compute the readings above per k, and the two interventions (rescale,
v removal) at k = 4, 8, 16 and 32 with a paired bootstrap over rows. GPU under the gpu lock.

**Method amendment, 2026-10-02 15:49 CDT, before any GPU run.** Reading Huginn's
code (`raven_modeling_minimal.py`, `SandwichBlock.forward`): every block ends in
`x = norm_4(mlp(norm_3(x)) + x)`, so each token's recurrent state leaves every block
RMS-normalised (times norm_4's learned gain), and the exit applies `ln_f` per token
before the coda. Two consequences, recorded before the result: G-1 holds by construction
(the CPU smoke reads norm 76.37 at k = 1..4), and G-4 holds by construction (a per-token
rescale is undone by `ln_f`; the CPU smoke reads exactly 0.0). Neither is evidence about
what the loop learned. They are kept in the output and scored as "architectural". The
Huginn finding they carry is structural: an earning loop whose state CANNOT use per-token
magnitude. The readings that test learned behaviour are G-2, G-3, G-5 and G-6. Probe
built at `lab/huginn/huginn_loop_geometry.py`; CPU faithfulness diff 0.0. Run with
`/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python` (the `~/.venvs/huginn`
transformers 4.48 cannot load it).

## Results

Run 2026-10-03 01:39 CDT at cd97bd4, 64 rows x 1024 tokens, 167 s on the 5090.
Faithfulness: max abs logit difference 0 at k = 32 on 8 rows. Artifacts:
[`../results/2026-10-02-huginn-loop-geometry/`](../results/2026-10-02-huginn-loop-geometry/).

| k | state norm | cos to v | RMS along v | RMS perp | step / norm | step share along v | PR (centred, normalised) | CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 76.37 | 0.188 | 2.73 | 70.68 | | | 215 | 4.629 |
| 2 | 76.37 | 0.137 | 3.06 | 72.97 | 0.907 | 0.004 | 374 | 3.773 |
| 4 | 76.37 | 0.135 | 3.24 | 75.27 | 0.602 | 0.001 | 604 | 3.015 |
| 6 | 76.37 | 0.135 | 3.29 | 75.55 | 0.393 | 0.001 | 649 | 2.734 |
| 8 | 76.37 | 0.136 | 3.30 | 75.56 | 0.262 | 0.001 | 646 | 2.617 |
| 16 | 76.37 | 0.138 | 3.30 | 75.55 | 0.061 | 0.001 | 638 | 2.520 |
| 32 | 76.37 | 0.138 | 3.29 | 75.55 | 0.017 | 0.002 | 638 | 2.515 |

Interventions (CE delta vs the unmodified state at the same k, 95 % paired CI over rows):

| k | rescale to the k = 4 norm | remove v | remove a random direction |
| --- | --- | --- | --- |
| 4 | +0.00000 [+0.00000, +0.00000] | +1.188 [+1.040, +1.337] | +0.00014 [-0.00006, +0.00033] |
| 8 | +0.00000 [-0.00005, +0.00005] | +1.304 [+1.134, +1.479] | -0.00005 [-0.00023, +0.00012] |
| 16 | +0.00000 [-0.00005, +0.00005] | +1.305 [+1.134, +1.484] | -0.00003 [-0.00022, +0.00016] |
| 32 | +0.00003 [-0.00002, +0.00009] | +1.297 [+1.125, +1.477] | +0.00002 [-0.00015, +0.00020] |

| prediction | reading | held |
| --- | --- | --- |
| G-1 norm at 32 within 1.5x of 4 | 1.00001 | yes, architectural |
| G-2 step share along v < 30 % from k = 8 | max 0.0015 | yes |
| G-3 cos to v < 0.7 from k = 4 | max 0.138 | yes |
| G-4 rescale at 32 costs < 0.01 | +0.00003 | yes, architectural |
| G-5 removing v costs > 0.05 | +1.297 | yes (see reading 3) |
| G-6 PR grows > 10 % from k = 4 to 16 | +5.5 % | no |

## Verdict

Success on the hypothesis, with one prediction missed. Huginn's loop rotates at a fixed
per-token scale: every step from k = 2 on is 99.6 % to 99.9 % perpendicular to the shared
direction, and the step size falls about 0.8x per iteration (0.91 at k = 2, 0.017 at
k = 32), a contraction toward a fixed point. G-6 missed because the diversity growth comes
earlier than I placed it: the participation ratio triples from k = 1 to k = 4 (215 to 604),
peaks at k = 6 (649), then eases to 638. It grows exactly where CE falls fastest (4.63 to
2.73 over k = 1..6).

Readings:

1. **The earning loop moves only in direction.** Scale is fixed by `norm_4` at the end of
   every block. The shared direction holds 0.2 % of the state's energy (RMS 3.3 of 76).
2. **Its diversity is real and early.** PR 215 -> 649 of 5280 dimensions in six passes.
3. **The shared direction is a load-bearing bias, not where the earning happens.** Removing
   it costs 1.2 to 1.3 nats at every k, while removing a random direction costs nothing.
   This is a zero-ablation of a bias (the Sun et al. pattern), so it says the coda needs
   the bias; it does not say the loop's depth gain lives along it. Every step is
   perpendicular to it, so the gain is elsewhere. A mean-ablation was not run.

## Updated hypothesis

Contrast with MORPH's pulled slot-loop arms
([note](../../../.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md)):
there, pass index 2 takes a step 3.1x to 4.7x the cell norm with 79 % to 93 % of it along
the shared direction, the cells grow 26 -> 216 in norm, and the rank-only arm's whole
K1-K6 rides on one amplitude. Huginn's architecture forbids both. The next arm puts a
per-pass RMSNorm on the slot cells (Huginn's `norm_4` placement) on the detached
weight-1 arm, so the slot loop can only rotate.
