# Agent Note: the HCA compressed branch is dead on the TUL slot path

Status: proposed

## Problem

The looped core runs on SLOT positions when TUL is active. With `tul.max_slots: 64` that
is S = 64 positions. `base.yaml` sets `hca_compress_ratio: 256`.

`GatedPoolCompressor.forward` computes `n_blocks = S // m`. At S = 64 and m = 256 that is
**0**, and the method takes its `n_blocks == 0` branch, which returns `[B, 0, c]`. The
guard is correct — it was added 2026-08-18 to stop `F.pad` inventing a block during short
generation — but nothing downstream notices that the branch produced nothing.

`_CCAHCAAttention.forward` then calls `fused_hca_attention(q, C_comp, ...)` with an empty
`C_comp` and gets a zero output, and `_gate_combine_up` blends it in as
`g[..., 0:1] * out_comp + g[..., 1:2] * out_win`.

Measured 2026-08-25 (`lab/divergence/attn_sink_probe.py --geometry --token-path`,
`tul_a1`, batch 6, `use_kernels=false`):

| path | S | core blk 1/3/5 `n_blocks` | `\|out_comp\|` | `g_comp` | `\|out_win\|` |
|---|---:|---:|---:|---:|---:|
| slot | 64 | **0** | **0.0000** | ~0.50 | ~225 |
| token | 1152 | 4 | ~1030 | ~0.50 | ~1030 |

So **three of the six core blocks output exactly half of what they were built to output**,
and the learned gate still spends about half its mixture on a tensor that is identically
zero. Those blocks get all of their positional mixing from the window branch alone.

Two consequences:

1. The core is quietly running at reduced attention capacity whenever TUL is on with a
   slot budget below `hca_compress_ratio`.
2. It is an ARCHITECTURAL ASYMMETRY between the two TUL arms. A1 loops over slots and has
   the dead branch; A0 loops over tokens and does not. A1 diverges at
   `ademamix_alpha_cap` 3.5 in 4 of 4 runs; A0 is healthy at the same settings
   (`lab/divergence/takeover-campaign.md`). No hypothesis in that campaign has used this.

Found while auditing geometry for H18
([`docs/experiments/failures/2026-08-25-h18-positional-attention-sink.md`](../../../../docs/experiments/failures/2026-08-25-h18-positional-attention-sink.md)),
which is also where the ledger entry H24 comes from.

**Scope.** The deploy recipe is not affected: at `seq_len: 4096` the derived
`max_slots` is 512 and `n_blocks` is 2. The dead branch needs a slot budget under 256,
which is what every TUL short-schedule arm has run.

A second, smaller finding from the same audit, recorded here so it is not lost: at
`seq_len: 1024` the CSA layers have `n_blocks = 144` while `top_k: 256`, so
`tk = min(top_k, n_blocks) = n_blocks` and **CSA's sparse selection never fires** — the
probe measured `distinct_top_idx == n_blocks` at every call, on both paths. At the deploy
`seq_len: 4096` there are 512 blocks and selection does fire. It means every TUL arm
measured to date ran a CSA that was dense, which changes how those arms should be read but
is not a bug in the module.

## Proposal

1. **Fail loudly instead of silently.** `_CCAHCAAttention` (and `_CCACSAAttention`) should
   raise at FORWARD time when `S >= m` is expected and `n_blocks == 0` on a training path.
   The short-generation case the guard exists for is inference, where S is genuinely small,
   so the check belongs where the two can be told apart rather than in the compressor.
2. **Size the compressed branch from the slot budget.** When TUL is active, the core's
   `hca_compress_ratio` (and `csa_compress_ratio`, and `top_k`) should be derived from
   `max_slots`, not inherited from the token-stream values. `hca_compress_ratio: 16` gives
   the 64-slot core 4 real compressed blocks, which is what the token path gets at
   `seq_len: 1024`.
3. **Test it.** A runtime-invariant test that builds the model at the TUL slot budget and
   asserts `|out_comp| > 0` on every core block. The invariant table in
   `lab/runtime-invariants.md` §6b is where the row goes.

Point 2 changes the model, so it goes through an arm before it ships — H24 in the campaign
ledger, paired seeds, against a control at the same parameter count. `GatedPoolCompressor`
has no `m` in any weight shape (the projections are per-position and the reshape happens
after), so the change adds no parameters and existing checkpoints stay loadable.

## Alternatives considered

- **Leave it and only fix the ratio.** Rejected: the silence is the worse half. The same
  class of mismatch will recur the next time a subtree runs the core at a new sequence
  length, and nothing will say so.
- **Make the compressor pad up to one partial block instead of returning empty.** Rejected
  for now: it changes what a block MEANS at every sequence length, including the deploy
  recipe and the KV-cache decode path that was specifically made to reproduce the training
  selection. The 2026-08-18 comment on the `n_blocks == 0` branch documents a bug caused by
  exactly this kind of invented block.
- **Drop the HCA branch on the slot path and rebalance the gate.** Rejected: it bakes the
  reduced capacity in rather than fixing it, and it would make the A1/A0 asymmetry
  permanent while H24 is still untested.
- **Derive `top_k` from `n_blocks` in the same change.** Deferred to its own note: CSA
  selection being inert is a property of the short schedule, not a defect in the module,
  and folding two model changes into one arm makes the arm unreadable.

## Acceptance criteria

- A model built at `tul.max_slots: 64` has `|out_comp| > 0` on all six core blocks, and a
  test asserts it.
- A model built with a slot budget below the compressed ratio raises at build or forward
  time on a training path, and the test that covers short-S generation still passes.
- The H24 arm reports paired-seed validation CE against a control, and the note is moved to
  `implemented/` or `rejected/` on that result — not on the fix compiling.

## Risks

- Reviving the branch ADDS effective capacity to the slot core. An arm that improves could
  be improving for that reason and not because the asymmetry mattered. The control must
  hold parameter count fixed (it does — `m` is in no weight shape) and the writeup must
  say that capacity is still a live confound.
- `hca_compress_ratio: 16` on the slot path changes the trained pooling width the HCA
  compressor weights have seen on the TOKEN path, where they keep m = 256. Both paths share
  the same weights, so the arm is not a pure single-variable change unless the token path
  is held at its current ratio.
- One run is unreadable in this regime: MORPH runs decorrelate in 11 steps at a fixed seed
  with a 6.5 % median spread. Paired seeds are mandatory.

## Update 2026-08-25 — the no-training screen

[`docs/experiments/failures/2026-08-25-h24-hca-branch-screen.md`](../../../../docs/experiments/failures/2026-08-25-h24-hca-branch-screen.md)

Reviving the branch at fixed weights (core blocks only, `B_a` sliced to 16 rows) lifts the
loop's slot-state rank ratio by **+0.0939 on healthy rungs and +0.0947 on sick ones**. The
validity gate reproduced the published H4 ratios to 0.004 and iteration 0 was identical to
0.000e+00 between the arms, so the measurement is sound and the lift is real.

It is a UNIFORM capacity effect, not a targeted repair. That does not change the case for
fixing the defect — three of six core blocks at half attention output is not a design
choice — but it lowers what the fix should be expected to do for the TUL divergence, from
"cure" to "lever of the H5 class". The Acceptance criteria above are unchanged; the Risks
section's first bullet is now MEASURED rather than anticipated.

## Update 2026-08-25 — the training arm, and why this note stays `proposed`

The binary training arm is filed as a failure:
[`docs/experiments/failures/2026-08-25-h24-hca-branch-arm-binary.md`](../../../../docs/experiments/failures/2026-08-25-h24-hca-branch-arm-binary.md).

| run | ABORT | min CE |
|---|---|---:|
| ctrl-s0 | 2880 | 5.1713 |
| ctrl-s1 | none | 5.0040 |
| hca16-s0 | none | 4.7523 |
| hca16-s1 | 2940 | 5.5720 |

Reviving the branch does **not** stop the takeover: 1 of 2 seeds diverged in each group,
and the two aborts land 2.1 % apart on opposite seeds.

What that changes, and what it does not:

- **The defect is unaffected.** It is measured, not inferred: `n_blocks = 0`,
  `|out_comp| = 0.0000`, and the gate still spends `g_comp ≈ 0.50` on that zero tensor
  on every odd core block. Half of one gate's capacity buys nothing.
- **The fix mechanism is built and tested.** `model.core_hca_compress_ratio` reaches only
  the core, `tests/test_core_hca_ratio.py` covers it, and the default `null` is
  bit-identical to before.
- **The default is NOT changed, so this note is not `implemented`.** The branch is still
  dead on the slot path in every shipped recipe.

The note stays `proposed` because the remaining question is no longer about divergence.
It is whether a live core HCA branch is worth its compute on the slot path at all — a
capacity question that needs a matched-token CE comparison, not a divergence arm.
`hca16-s0` has the best min CE of the four runs (4.7523 against 5.0040), but one seed each
way sits well inside the 6.5 % run-to-run spread in
[`2026-08-23-tul-run-replication.md`](../../../../docs/experiments/failures/2026-08-23-tul-run-replication.md),
so it is not evidence. Do not cite it as any.
