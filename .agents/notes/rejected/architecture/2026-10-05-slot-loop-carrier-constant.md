# Agent Note: the slot loop's carrier constant, and two ways to cut it

Status: rejected — both cuts failed: the loop rebuilt the constant by another path (uniform HC) or detonated (centering)

## Problem

The data-flow probe and its follow-up (2026-10-05,
[filing](../../../../lab/experiments/successes/2026-10-05-snap-dataflow-probe.md); probes
`lab/divergence/dataflow_probe.py`, `lab/divergence/carrier_constant_probe.py`; 96 rows of
LXTUL 5k and snap 5k) found why the slot loop's contribution stays small:

- After the loop's FIRST pass, 99.6-99.8 % of the cell carrier's energy is ONE vector shared
  by every slot of every row; at the exit 98.2 %. At entry the per-slot part is 54-57 %.
- The vector is one channel direction with a fixed sign per Hyper-Connection stream
  (+,-,-,+ / +,-,+,-). It cancels in the stream mean, so every earlier instrument (all of
  them read the stream mean, including the
  [LayerNorm common-mode note](2026-10-02-layernorm-common-mode-in-latent-targets.md)) saw a
  small shared direction, not 98 % of the state.
- Its direction is already in the loop's entry state (cosine 0.91-0.94 to the final
  constant). From core layer 2 on, the HC pre-map feeds the core ATTENTION an input that is
  99.9 % shared; the attention writes a large shared output (LXTUL layer 2: RMS 51) into one
  stream (Hpost row about [0, 0, 3.9, 0]); the per-pass RMSNorm divides the blow-up back
  out. The same layers rebuild it every pass. Norm gain, DiagonalInjection, MLP writes and
  the reset are not the source; the MLPs read the per-slot part.
- So the loop's cross-slot attention reads an average of the past, not specific spans, and
  the per-slot content is a 2 % ripple. Content is not destroyed (own-span R^2 0.40 at the
  seed, 0.33 at the exit), but the loop adds only +0.01 R^2 of next-span content.
- The trained models depend on the vector: removing it at eval costs +0.11 to +0.13 nats and
  turns K1-K6 negative. Any fix must be trained in.

## Proposal

Two arms on `lxtul.yaml`, run as a twin at 5k (prereg
[`2026-10-05-carrier-constant-twin`](../../../../lab/experiments/planned/2026-10-05-carrier-constant-twin.md)).

1. **Centered loop attention** (`tul.loop_attn_center: ema`): in the slot loop's core blocks,
   the attention sublayer reads `x_bar - mu_l` instead of `x_bar`. `mu_l` is a per-core-layer
   buffer, the EMA (decay 0.99) of `x_bar` averaged over valid slot positions, updated under
   no_grad AFTER use in training and frozen in eval. No learned parameter, so it cannot learn
   to switch itself off, and no batch statistic, so no slot sees a later slot's state.
2. **Uniform loop attention HC** (`tul.loop_attn_hc: uniform`): the core blocks' ATTENTION
   Hyper-Connection becomes a plain residual: read the stream mean (`Hpre = 1/n`), write the
   same output to every stream (`Hpost_row = 1`), `Hres = I`. The probe found the constant
   cancels in the stream mean and is written through one stream, so this removes both the
   read and the write path. The MLP HCs are unchanged.

## Alternatives considered

- **Remove the constant at eval.** Measured: +0.11 to +0.13 nats, K1-K6 negative.
- **A learned per-stream bias subtracted before the norm.** It can learn to stay 0, which
  keeps the constant; the EMA mean cannot.
- **Static (bias-only) attention HC.** Milder than uniform; a learned bias can still route
  the write into one stream. Kept as the fallback if uniform hurts CE.
- **Change the per-pass norm or its gain.** Gain is 1.00 +/- 0.05 and plays no part.
- **Batch-mean centering (BatchNorm statistics).** The mean over a row's slots includes later
  slots: a causality leak in the strict loop.

## Acceptance criteria

- Both keys build-time only (no forward branch on a flag), off by default, a no-op test
  (key absent gives a bit-equal forward), tests that the EMA updates only in training and
  only from valid slots, and that the uniform HC equals a plain residual.
- The twin is filed with the data-flow probe re-run on both checkpoints (the per-slot share
  after pass 1 is the mechanism reading).

## Risks

- The model may rebuild a shared vector through another path (the MLP HCs, the injection).
  The per-slot share reading catches this.
- The constant may be doing a job (a bias the coda reads); cutting it may cost CE.
- Uniform HC loses the attention's stream routing, which may cost the loop capacity.

## Outcome (2026-10-06)

Both arms tested and failed
([filing](../../../../lab/experiments/failures/2026-10-05-carrier-constant-twin.md)). Uniform
loop attention HC: training rebuilt a new load-bearing constant through the MLP stream
mixers; scores unchanged. Centered loop attention: detonated at step 470 and never trained.
Kept in `rejected/` because the finding (98 % of the carrier is a slot-shared vector hidden
in the HC stream differences) and the two failed cuts are the record that stops the next
attempt at the same path. Do not retry a single-path cut; a model without HC streams is the
cleaner test.
