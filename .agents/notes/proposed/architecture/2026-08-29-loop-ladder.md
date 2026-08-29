# Agent Note: The loop ladder — is iteration the disease, or just uncontrolled iteration?

Status: proposed

## Problem

The GL line's winner is coreless: the "gradient through an iterated write is
fatal" reading of the TG2 failures removed the loop entirely, and Wolfe
challenged that reading directly — the TG2-era evidence never measured ρ or
state norms inside the failing loop, the arms had no write supervision, and the
old bag-mean atomics made those runs non-replicable. His counter-hypothesis:
looping is viable and the instability lives in runaway activations/gradients
(controllable), possibly amplified by AdEMAMix at low batch. Most prior art we
build on (Huginn, Parcae, UT) loops successfully — though their loops carry
token states with losses everywhere, never a sole-channel bottleneck like the
slot write.

## Proposal

Four ~30-min panel arms on the gl1b config (mask + MUX held fixed), each
instrumented with the per-iteration loop probe and σ_max logging that the TG2
era lacked:

- **L1** (`tul_l1.yaml`): core loop restored, FULL BPTT (bptt_depth 8 ≥
  max_depth), nothing protecting it. Affordable because the loop runs on the
  compact slot sequence.
- **L2** (`tul_l2.yaml`): L1 + the HARD spectral projection at cap 1.5 — the
  measured takeover cure (the soft penalty measurably lost the same fight).
- **L3** (`tul_l3.yaml`, `tul.db_loop`): the DiffusionBlocks-shaped loop —
  detached carry (and retention state), per-iteration LOCAL mux losses, seed
  injection live so every loss reaches the write through exactly ONE core
  application. Immune to both causal stories by construction.
- **L4** (`tul_l1` + `training.optimizer=adamw`): Wolfe's optimizer arm — if
  L1 fails and L4 doesn't, the instability was AdEMAMix-at-low-batch, not the
  loop.

## Alternatives considered

- **Truncated BPTT depth 1**: rejected as the primary arm — it puts some
  gradient through an iterate without deciding either hypothesis.
- **GL2 detached refiner with self-distillation**: deferred, not rejected — L3
  is its simplest local-objective form; a bespoke refiner head only pays if L3's
  shape works but its target is wrong.
- **Keeping the coreless champion and skipping the question**: rejected — the
  product goal is a looped model with amortized decoding, and the "fatal" claim
  is currently load-bearing while resting on unmeasured evidence.

## Acceptance criteria

Prereg `lab/experiments/planned/2026-08-29-tul-loop-ladder.md` (frozen before
smoke). The "iterated write is fatal" claim is falsified by any full-BPTT arm
sustaining worth_shuffle ≥ 0.04 at CE ≤ 4.50.

## Risks

L1/L4 detonation is a possible outcome, not a failure of the experiment — the
probes make a detonation informative. Memory: full BPTT on the slot loop adds
~6 core layers of params (+~22M) and checkpointed loop activations; the smoke
run gates VRAM before the panel.
