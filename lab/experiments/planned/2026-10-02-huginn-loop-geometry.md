# Planned: does a loop that earns depth rotate its state or grow it?

Status: planned

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
