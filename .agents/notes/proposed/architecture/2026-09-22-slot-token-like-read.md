# Agent Note: a token-like read of the slot state (the unpack at every coda layer)

Status: proposed

## Problem

The positives ledger ([docs/9-26-TUL-run-history-IMPORTANT.md](../../../../docs/9-26-TUL-run-history-IMPORTANT.md),
2026-09-22) splits the loops that earn from the loops that do not by ONE property. Every
loop whose contribution grows with steps and beats its depth-1-trained twin has the token
CE read the loop's output directly at the token's own position, with the token's own
identity re-supplied at every block (the plain loop 0.185, the paid loop 0.168, the aux
token path +0.0102 through the SAME six blocks that give the slot path +0.0005). The slot
loop's output reaches the token loss through one attention hop onto prefix cells, and
every slot-state positive is at most 0.026 and does not grow with steps. Wolfe's
2026-09-10 call names the same thing: the slot loop fails on the gradients in the loop.

The tree already has a one-shot direct read, `tul.bcast` (the "unpack" of Thought Unpack
Loop): z of the previous slot, through offset-indexed zero-init linears, added ONCE to the
coda input of every token of the next span. It ran on 2026-09-10 on the M-next MUX family
with a coda-only restriction and read token K1-K6 +0.0006
(`lab/experiments/failures/2026-09-10-arc-slot-map-levers.md`). It has never run on the
strict spandec ruler, and it is added once where a token's own `x0` is injected at every
block.

## Proposal

`tul.bcast_layers: entry | all` (default `entry`, the shipped one-shot add, bit-identical).
Under `all` the SAME unpack term is added again at every coda block's injection seam
(`_back_region`, beside the per-layer `x0` and bigram terms), scaled by a per-coda-layer
learned scalar gate that starts at zero. So at step 0 `all` equals `entry` equals the
ruler bit for bit, and the base weights are shared across the three at one seed (the gate
draws no RNG). The plan ablations (`plan_mode` zero / shuffle) act on `h_slots` before the
unpack reads it, so both adds vanish together under the ablation and the strict leak gate
keeps its meaning.

Arms, all on the strict spandec ruler (`tul_slot_spandec_strict.yaml`):

| arm | config | one factor against |
|---|---|---|
| `slot-spandec-strict-bcast` | `tul_slot_spandec_strict_bcast.yaml` | ruler: `tul.bcast` |
| `slot-spandec-strict-bcast-all` | `tul_slot_spandec_strict_bcast_all.yaml` | bcast: `tul.bcast_layers: all` |
| `slot-spandec-strict-bcast-all-d1` | `tul_slot_spandec_strict_bcast_all_d1.yaml` | bcast-all: `tul.slot_depth_fixed: 1` (the depth-1-trained twin) |
| `slot-spandec-strict-norecur` | `tul_slot_spandec_norecur.yaml` | ruler: `tul.slot_depth_fixed: 1` (the existing twin config; it already composes from the strict ruler) |

Two readings, both from the ledger's "never built on" list:

1. The K-curve and the paired CE of the direct per-layer read at 5k.
2. The HORIZON reading on the slot loop: the plain loop's value over its depth-1 twin went
   from 0.004 at 5k to 0.067 at 20k, and no slot arm was ever paired with its twin past
   5k. `slot-spandec-strict-20k` (checkpoint kept) pairs with a 20k `norecur`; the
   bcast-all arm pairs with its own d1 twin at 20k.

Prereg: [`lab/experiments/planned/2026-09-22-arc-slot-token-like-read.md`](../../../../lab/experiments/planned/2026-09-22-arc-slot-token-like-read.md).

## Alternatives considered

- **`tul.loop_reads_tokens` (tokloop).** Built and tested 2026-09-13, never trained: it
  runs tokens through the core (about 44 block passes per token) and abandons "think once,
  decode cheap", which is what TUL is for. Not queued.
- **Re-run `tul.bcast` alone on the ruler.** Kept as the first arm (config only). On its
  own it re-tests a lever that read flat, on a different base; the per-layer add is what
  is new.
- **A separate `W_bcast` per coda layer.** Four times 32 offset linears of d x d is 134 M
  parameters; a shared term with per-layer scalar gates costs n_coda parameters and tests
  the same question (is a direct per-layer read used).
- **`tul.reread` (the slot reads the frozen token states each pass).** Ran 2026-09-10,
  +0.0003. That re-supplies the loop's INPUT; this note is about the loop's OUTPUT reaching
  the loss.
- **Horizon only, no new mechanism.** The norecur 20k pair runs regardless of the bcast
  result and is listed first in the queue order, because it needs no build.

## Acceptance criteria

The prereg's clauses. In short: bcast-all's gates leave zero (the mechanism is used),
token K1-K6 at 5k above the +0.002 slot-loop floor with a CI clear of it, paired depth-6
CE not worse than the ruler, and at 20k a value gap over the depth-1 twin that is larger
than the +0.0020 read at 5k. A flat K-curve with used gates is a finding too: the read
was never the limit.

## Risks

- The 09-10 unpack arm read flat; the prior on this arm is low and stated in the prereg.
- The per-layer add gives the coda a second route to z; under strict geometry it is the
  same z the cells carry, so no new information crosses a span, but the leak gate must be
  re-run with the gates live (the builder's test does this with a positive control).
- The 20k runs cost about 3 hours each on the 5090; the queue order puts the no-build pair
  first.

Drafted 2026-09-22 20:36; builder branch `bcast-build`.
