# Agent Note: the cross-iteration GLA carry breaks per-position causality

Status: proposed

## Problem

`docs/runtime-invariants.md` §5 states the contract:

> **No module may pool statistics across the sequence axis.** Every position's output
> must depend only on positions ≤ t.

It is violated today, on `master`'s default config, by `model.retention_carry: true`
(`morph/configs/base.yaml:89`, and every `tul_*` arm through it).

The GLA retention branch returns a `final_state` — the recurrent state after the whole
sequence — and `_forward_single` feeds it back as the `initial_state` of the next core
iteration (`morph/model/transformer.py`, `track_ret`). That state summarises **all S
positions**. From core iteration 2 onward, therefore, position *i*'s output depends on
positions after *i*.

Measured, `tul-a0-acap1/step_20000`, `ignore/perf/future_corruption_probe.py`
(the gate §5 itself names: corrupt every token after position k, compare logits at ≤ k):

| k | max abs delta logit over positions 0..k |
|---:|---:|
| 8 | 3.96 |
| 16 | **4.08** |
| 32 | 1.97 |
| 48 | 1.18 |
| 62 | 0.71 |

against a mean `|logit|` of 2.41. With `retention_carry=false` the same probe returns
**exactly 0.000** at every k. Eager and fused-kernel paths behave identically, so this is
architectural and not a kernel bug. `ignore/perf/causality_bisect.py` hooks every
submodule and places the first divergence at `core.1.retention` **on its second call** —
clean on the first, which is the signature of a carry and not of a leaky attention mask.

### What it costs

`ignore/perf/carry_leak_cost.py`, same weights, same 20 validation batches
(81 920 tokens):

| | val CE | PPL |
|---|---:|---:|
| `retention_carry=true` (every published number) | 3.2952 | 26.98 |
| `retention_carry=false` (strictly causal) | 3.4385 | 31.14 |
| delta | **+0.1433 nats** | +15.4 % |

For scale: the TUL-gate result this project has been chasing is **−0.1054 nats**. The
lookahead is worth more than the effect being measured. It does not invalidate the arm
comparisons — every arm carries the same leak — but every **absolute** CE and PPL number
in `docs/` is inflated by it, and the generation numbers are not, which is the most
likely explanation for a model at PPL 27 emitting `rep4` 0.51 under top-k sampling.

The 0.1433 figure is an **upper bound on the leak premium**: these weights were trained
with the carry and have learned to use the state that is being removed. A model trained
from scratch without it would land somewhere between 3.2952 and 3.4385, and where is not
known.

### How it was missed

`docs/tul-divergence-rca.md` §5 candidate 2 already describes the path — *"`retention_carry
=true` carries GLA state from the END of iteration t into position 0 of iteration t+1"* —
but frames it purely as a stability risk (zero-pad slots contaminating real ones), tests it
for that, and records "no effect" (`E4`, table row). The causality consequence was never
drawn. The §5 gate that would have caught it is documented as living in the Olympiad tree
and was never run here.

## Proposal

1. **Gate it here, now.** `tests/test_causality_contract.py` (added with this note) runs
   the §5 probe on a tiny CPU model. Three tests pass; the `retention_carry=true` case is
   `xfail(strict=True)` so the defect is recorded rather than hidden, and the suite fails
   loudly if someone changes the behaviour without removing the marker.
2. **Decide what the carry is for.** It is not free memory — it is lookahead. Either:
   - **(a) turn it off** (`retention_carry: false` in `base.yaml`) and re-baseline. The
     RCA already measured no stability cost from doing so, and the note's own comment
     calls the alternative "global retention, no memory".
   - **(b) make it causal** by carrying a per-position state rather than the final state,
     so position *i* at iteration k+1 receives the state as of position *i* at iteration
     k. This keeps cross-iteration memory and costs an [S, H, D, D] carry instead of
     [H, D, D] — memory-expensive, and the size should be measured before it is chosen.
3. **Re-baseline before any absolute claim.** Any comparison against an external PPL, and
   any claim of the form "MORPH reaches PPL X", must be recomputed under the option chosen
   in 2. Arm-vs-arm results already published stay valid as differences.

## Alternatives considered

- **Leave it and document it as an accepted trade.** Rejected. It is not a trade the
  project ever made: §5 forbids it in writing, and the leak is larger than the effects
  the project is measuring, so it silently competes with every result.
- **Fix it only at inference.** Rejected: the weights learned to use the carry, so
  disabling it at eval measures a model that was never trained, which is what the +0.1433
  number above already is. It is a diagnostic, not a fix.
- **Blame the fused kernels.** Ruled out by measurement: `use_kernels=false` reproduces
  the violation to three digits (4.069 vs 4.079).
- **Blame the attention masks.** Ruled out: the module bisect shows every attention output
  bit-identical (0.000e+00) at positions ≤ k, including `core.1.attention` immediately
  before the first divergence. An earlier pass of that bisect *did* accuse the CCA
  compressor; that was an artefact of slicing a `[1, 8, 64]` block tensor along its channel
  axis, and is corrected in the script.

## Acceptance criteria

1. `pytest tests/test_causality_contract.py` runs in CI, and the
   `retention_carry=true` case is either passing (fixed) or `xfail(strict=True)` with a
   live link to this note. A silently-skipped test does not satisfy this.
2. `ignore/perf/future_corruption_probe.py` returns `CONTENT_CAUSAL_PASS` on a checkpoint
   from the chosen option's configuration, with the worst delta printed and equal to 0.000.
3. A decision is recorded here for option (a) or (b), with the measured number that
   settled it: for (a), the val CE of a model TRAINED with `retention_carry: false`; for
   (b), the measured resident-memory delta at batch 12 on the 5090.
4. Every `docs/` page that states an ABSOLUTE CE or PPL carries a one-line note saying
   which side of this change produced it. Arm-vs-arm differences need no change.
5. This note moves to `.agents/notes/implemented/bug-fix/` with `Status: implemented`
   only after 1-4 hold.

## Risks

- Turning the carry off changes the architecture, so every checkpoint on disk becomes a
  different model. Nothing is directly comparable across the change without a re-run.
- Option (b) has an unmeasured memory cost on the 5090, where the TUL arms already sit at
  ~23 GB resident at batch 12.
- The +0.1433 measurement is one checkpoint, one config, 81 920 tokens of one validation
  split. It has not been repeated on the TUL arms or at another scale.

## Not verified

- Whether a model TRAINED with `retention_carry=false` reaches a better or worse causal
  CE than 3.4385. That is the number that decides option (a), and it needs a run.
- Whether the leak's size grows with `mean_depth` (more iterations = more carries) or with
  sequence length. Both are plausible and neither was measured.
- Whether the JAX mirror (`morph/jax/`) has the same path.


## Addendum — 2026-08-31: the leak is LEARNED and full BPTT trains it

The +0.1433-nat figure above was measured on a truncated-BPTT arm and badly
understates the ceiling. The l2cap recipe (full BPTT, σ-cap) trained 30k steps
turns the channel into **3.85 nats** (carry-off CE@K6 4.965 vs carry-on 1.119,
same weights), while its carry-free CE@K1 improved only 0.19 nats over the last
20k steps. The recipe's entire "depth-earned CE" was this channel
(`lab/experiments/successes/2026-08-31-carry-leak-audit.md`). This note is now
the gating work item for all loop claims: no depth result is admissible without
a carry-off sweep, and the next loop run trains with `retention_carry=false`
or a causal (per-position prefix) carry.
