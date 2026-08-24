# Measuring the looped core's operator, not its magnitudes

Everything the trainer logs about the core loop is a MAGNITUDE. `loop/core_gain` is
`||h_new|| / ||h||` along whatever direction the data happened to pick. `spec/sigma_max` is
the spectral norm of ONE weight matrix, an upper bound on one factor of one block.
`preclip/core_block_gain` is fitted from gradient norms, so it sees the realized direction
and nothing else.

None of those is the operator. When the question is "did the map become expansive, or did
the realized direction rotate into an amplifying direction the map always had", they cannot
answer it, and the two readings call for opposite cures. This is how to measure the
operator instead.

## The three numbers, and what each is for

`morph/training/core_jacobian.py` power-iterates `J^T J` at the live operating point, with
`J v` from the double-backward identity, in fp32 with autocast off, restricted to the
positions the iteration is actually updating.

| number | meaning | use it for |
|---|---|---|
| `sigma_max` | worst case over all directions | how much HEADROOM the map has |
| typical gain, `\|\|J\|\|_F / sqrt(n)` | what a generic direction sees | comparing against the realized gain |
| alignment | whole-step gain / product of the blocks' gains | whether the blocks' amplifying directions AGREE |

Read `sigma_max` with the convergence residual next to it. A residual Jacobian has its
singular values clustered near 1, so power iteration is slow here — 200 iterations is the
measured floor and the module reports how far it got.

Do NOT read `sigma_max` on its own. The first real measurement returned 1.5e6, because a
pad slot enters the core loop at `h = 0`, an RMSNorm at `h = 0` has a Jacobian of order
`1/eps`, and the top singular direction sat entirely in the pad subspace. The validity mask
fixes that, and the typical gain is the number that matches what the gradients do.

## In a training run

```
python -m morph.training.train --config-name tul_a1 \
    training.jac_probe_every=250 training.jac_probe_iters=[0,3] \
    training.jac_probe_power_iters=200
```

0 — the default — constructs nothing and traces the same graph the forward always did. The
probe runs one extra no-grad forward and saves and restores the RNG around it, so a probed
run stays bit-identical to an unprobed one
(`tests/test_core_jacobian.py::test_probe_is_rng_neutral`).

## Over a checkpoint ladder, offline

```
PYTHONPATH=$PWD python lab/divergence/jac_ladder.py \
    --ckpt-dir checkpoints/morph/<run> --power-iters 300 --iters 0 --out ladder.json
```

One fixed batch and one fixed depth draw per rung, so rungs are comparable. It applies the
QAT transforms and then uses the trainer's own `load_checkpoint`. **Do not hand-roll the
load.** The core MLP is ternarised by a weight parametrization applied after construction,
so a bare model's key is `..._cms.weight` while the checkpoint's is
`..._cms.parametrizations.weight.original`, and `torch.compile` puts `_orig_mod.` in one
path and not the other. `load_state_dict(strict=False)` then drops every MLP tensor in
silence — measured, on the first version of this script, which reported that no core linear
exceeded 2.0 while the run's own log had `sigma_max` at 3.30.

### Deriving a spectral cap instead of tuning one

```
PYTHONPATH=$PWD python lab/divergence/jac_ladder.py \
    --ckpt-dir checkpoints/morph/<run> --sweep-ckpt <sick>.pt \
    --sweep-caps 3.0,2.0,1.5,1.0 --project-scope mlp --dump-sigmas --out sweep.json
```

Projects each selected core linear onto `sigma_max <= cap` THROUGH its own forward, so it
acts on the effective ternary map, and re-measures afterwards and raises if it missed. Plot
the resulting alignment against the cap and read the knee off the curve.
`--project-scope {mlp,attn,all}` says which family carries the amplification.

### How concentrated is the backward cotangent

```
PYTHONPATH=$PWD python lab/divergence/jac_ladder.py \
    --ckpt-dir checkpoints/morph/<run> --rank-probe [--rank-token-path] --out rank.json
```

Participation ratio of the per-position cotangent norms, reported as an effective number of
positions, via `register_full_backward_hook` on each core block. `--rank-token-path` runs
the SAME weights with `slot_layout=None`, which is arm A0's code path, so the comparison
holds the operator fixed and varies only how many positions the cotangent is a sum over.

## What this was used to find

[The cure experiment](../experiments/results/2026-08-24-tul-takeover-cure.md): across the
onset the map's isotropic per-block gain moves +2.5 % while the alignment of its six blocks
moves x2.9 and the cotangent collapses from 13 effective slot positions to 2.5. The map
barely changes; its directions align, because a weight-shared loop applying the same `J^T`
24 times is power iteration.
