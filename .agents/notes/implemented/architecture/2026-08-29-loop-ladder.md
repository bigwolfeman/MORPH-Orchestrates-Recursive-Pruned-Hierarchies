# Agent Note: The loop ladder — is iteration the disease, or just uncontrolled iteration?

Status: implemented

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

## Decision

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

## Consequences

Filed as a SUCCESS: `../../../../lab/experiments/successes/2026-08-29-tul-loop-ladder.md`
(results + both new instruments). The binding falsifier F fired: `tul-l2-cap` (full
BPTT + σ≤1.5 hard projection) is the campaign's first load-bearing loop — CE @4250
4.3489, worth 0.146 still climbing, 0.233 nats of depth-earned CE, span-wide plan
carrier (worth_profile), uniquely degeneration-resistant generation. Every uncapped
loop trains stably but its iterations are CE-inert. Contractivity control is a
REQUIREMENT for a trainable iterated write, not a safety rail. The "iterated write is
fatal" claims are corrected in place (gist-loop note; 2026-08-23 block-backward-gain).
Follow-up: L3-WAKE (prereg 2026-08-30) tests waking l3's inert loop post-hoc.
