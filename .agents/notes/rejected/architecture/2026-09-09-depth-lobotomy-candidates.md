# Agent Note: why MORPH's loop earns 0.02 nats past iteration 1 where Parcae's fixed point is 8 iterations deep

Status: rejected — the panel ran (2026-09-09) and refuted every candidate but the ternary core; superseded by the ternary-core note

> Rejected 2026-09-09: the candidate panel this note proposed ran as the Parcae-entry panel
> (`lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md`) and the depth-candidates
> panel (`lab/experiments/failures/2026-09-09-arc-e20-loop-depth-candidates.md`). The entry, the
> fixed-point term, the state init, the carry rate and Parcae's depth schedule are cleared; the
> ternary core is the measured limiter. Superseded by
> [`../../implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md`](../../implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md).

## Problem

On the same OpenWebText, Parcae-140m's forced-depth curve earns 0.294 nats from iteration 1
to 8 (`~/parcae/outputs/parcae-140m-owt-10k/depth-010000.json`); MORPH's plain model earns
0.020 from 1 to 6 and nothing past 3 (E18 plain arm, `lab/experiments/results/2026-09-08-arc-e18/`).
TUL cannot have depth if the plain core has none. Probes on the E18 plain checkpoint
(2026-09-09, 96 rows, eval only): skipping the core costs 0.317 nats (T=0 4.256 vs T=1 3.939),
iterations 2-8 add 0.020; the state moves 9.9/4.9/3.4/2.7/2.3/2.1/2.0 % per iteration; branch
out/in norm ratios are 1-30 % (attention) and 4-13 % (MLP) per block; the ctx-channel
injection decay is 0.44 (init 0.447, untrained) on 320 of 1024 channels; from a zero or
noise init the loop lands at 4.67 / 6.30 nats and drifts past depth 8, so the fixed point
depends on the start. Parcae's learned decay is 0.424 (the same), its injection reaches every
channel through a learned B (diagonal mean 2.4), and its loop state starts from noise
(`state_init: like-init`), so its per-iteration CE gains (0.196, 0.058, 0.023, 0.007, 0.0015)
decay at its own contraction rate: a forced-depth K-curve measures convergence time to the
fixed point, not computation, and both models are flat past convergence (Parcae K8-K16 0.0002).

## Proposal

Candidates, ranked by the probe evidence:

```
MORPH's loop earns 0.02 past iteration 1; Parcae's fixed point is 8 iterations deep
├── 1. The per-iteration map is weak (out/in 2-13 % per block)
│   ├── ternary STE on the whole backbone (Parcae: dense bf16)
│   ├── HC Cayley residual at init gain 0.1, mixing weights still ~init (abs mean 0.003)
│   └── a state that starts near its fixed point has nothing to move
├── 2. Injection reaches 320 of 1024 channels (ctx only)
│   ├── 704 channels are never re-anchored to the input: from noise the loop lands at 6.3 nats
│   └── Parcae injects every channel through a learned B (diagonal mean 2.4)
├── 3. The fixed-point term at lambda 1.0 pays the loop to converge at T
│   └── flatness predates it (E13-era plain K3-K6 0.02-0.04), so a deepener, not the origin
├── 4. State init at the prelude output instead of noise (Parcae: like-init)
│   └── makes the K-curve read flat by construction; also removes the loop's job
└── 5. Lower: depth draw (E6 covered), full BPTT vs truncated 4, AdEMAMix vs Muon, attention (alive: block 2 at 30 %)
```

E19, plain model, E18 recipe, one hour per arm: **fp0** (fixed-point lambda 0; tests 3),
**inject-all** (`injection_channels: all`; tests 2), **noise-init** (Parcae's like-init for
the loop state, a small `core_init` mode; tests 4 and gives a like-for-like K-curve),
**dense-core** (ternary off on the core blocks only, a diagnostic, not a recipe; tests 1),
**hc-gain** (HC init gain 1.0; the other half of 1). Instruments on every arm: the K-curve,
CE at T=0 / 1 / 16, per-iteration state movement, the zero-init recovery curve. The value of
a loop is CE(trained looped, at its depth) minus CE(trained at depth 1, matched tokens), on
BOTH models (Parcae `mean_recurrence 1`, 20 min; MORPH `mean_depth 1 / max_depth 1`, 1 h):
that measures computation, which the K-curve cannot.

## Alternatives considered

- Read the Parcae gap as "MORPH broke depth" and search the diff blindly: rejected, the init
  probe shows part of the gap is where the loop starts, so the like-for-like control comes first.
- Widen the slot channel or change the slot target further: E18 closed it (0.35 behind at
  every width); the plain core is the bottleneck.
- Spectral caps / gain clamps: refuted in August; the map is over-contractive here, the
  opposite failure.

## Acceptance criteria

An arm whose fixed point is deeper: CE(T=16) − CE(T=1) beyond 0.1 nats with the tripwire
silent and end CE no worse than the E18 plain arm at matched tokens; and the depth-1-trained
control showing what the loop is worth in training.

## Risks

A stronger map is a less contractive one: the slot-loop detonations sat on the other side of
gain 1. Dense-core is a diagnostic only (Wolfe: no dense warmup as a recipe). Noise-init
changes what the coda reads at T=1 and may cost end CE at 5k steps.
