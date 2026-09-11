# Agent Note: the HCA compressed branch was dead on the slot loop's sequence

Status: implemented

## Problem

`GatedPoolCompressor.forward` (`morph/model/attention.py`) computes `n_blocks = S // m`
and returns an empty stream when `n_blocks == 0`. The slot loop runs the looped core on
`tul.max_slots` = 64 cells while `hca_compress_ratio` (the ratio every HCA layer inherited
before this fix) is 256, so `n_blocks = 64 // 256 = 0` on every odd core block (1, 3, 5).
`fused_hca_attention` then attends an empty stream and returns zero, and
`_CCABase._gate_combine_up` still blends the learned gate weight (`g_comp` 0.42-0.52) into
that zero tensor. Three of the six core blocks delivered about half the attention output
they were built for, silently, for the whole of every slot-loop run.

Found during the 2026-09-10 slot-geometry audit
([`lab/experiments/results/2026-09-10-slot-geometry-audit/README.md`](../../../../lab/experiments/results/2026-09-10-slot-geometry-audit/README.md),
finding F1), which also confirmed the deficit does not explain the slot loop's flat
K-curve — a separate, unrelated finding. The same weights on the paid loop's 1,152-position
token sequence give `n_blocks = 4` and `|out_comp|` 680-830 there, so the prelude and coda
(which always run on the full token sequence) were never affected, and neither is
`base.yaml`'s deploy recipe (`seq_len: 4096` gives 512 slots and `n_blocks = 2`).

The mechanism and its build-time construction knob, `model.core_hca_compress_ratio`
(scoped to the core only), were added 2026-08-25 alongside an earlier attempt at this same
fix
([`.agents/notes/rejected/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md`](../../rejected/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md)),
which was rejected because the slot loop briefly left the tree on 2026-09-03 (the paid-loop
ship) — the bug's host path was gone. The slot loop came back 2026-09-04 and the defect
came back with it; the same knob is the fix shipped here.

## Decision

`morph/configs/tul_short.yaml` — the root every slot-loop config composes — sets
`model.core_hca_compress_ratio: 16` as the shipped default. `64 // 16 = 4` blocks, the same
count the token path gets at `seq_len` 1024 (`1152 // 256 = 4`). Every arm since
`tul_a1_hca16.yaml` (2026-08-25) had the mechanism available; none used it by default until
now.

Measured on the one-factor arm `slot-mux-hca-fix`
([`lab/experiments/successes/2026-09-10-arc-slot-mux-hca-fix.md`](../../../../lab/experiments/successes/2026-09-10-arc-slot-mux-hca-fix.md)),
against the ruler `slot-mux-norm-match`: the audit rerun on the trained checkpoint reads
`n_blocks` 4 and `|out_comp|` 17.7-68.9 on core blocks 1, 3, 5 (P-d TRUE); the attention
branch's slot/token output ratio on those blocks closes from 0.37 / 0.43 / 0.32 to
0.73 / 0.83 / 1.08 (P-e TRUE); the loop's depth contribution is unchanged (token K1−K6
+0.0004, forecast K1−K6 +0.0078 against the ruler's +0.0067, both inside the ruler's
interval); 480-row CE at depth 6 is 0.030 nats better; wall clock is 1.04x. Every
prediction in the arm's prereg held.

`morph/configs/tul_slot_mux_hca_fix.yaml`, the one-factor arm config that proved this, is
retired: every `tul_short`-derived config now carries the fix by default, so there is no
longer a control to run it against. `tests/test_tul_hca_fix.py` covers the shipped default
directly — every sampled slot-loop config composes with the ratio, a core HCA compressor
built at it produces 4 live blocks on a 64-cell sequence while the pre-fix ratio produces
zero, and `base.yaml`'s paid loop (which never composes `tul_short`) is unaffected.
`morph/configs/tul_slot_mnext_parcae_core.yaml` explicitly overrides the new default back
to `null`: the Parcae core has no pooled compressor to re-block and the build raises if the
key is non-null, and that arm predates the fix and never had the defect (Parcae has no HCA
branch at all).

## Alternatives considered

- **Ratio 16 vs other values.** 32 gives 2 blocks (the floor before the branch becomes
  under-provisioned relative to the token path); anything at or above 64 reproduces the
  defect. 16 was chosen because it matches the token path's block count at the arm's own
  `seq_len` (1024), not because it is the only value that unblocks the branch.
- **Disabling the compressed branch on the slot shape instead of re-blocking it.** Rejected
  in the 2026-08-25 note that first proposed this fix: it bakes the reduced capacity in
  rather than repairing it, and makes the architectural asymmetry between the slot and
  token paths permanent.
- **Padding the compressor to invent one partial block instead of returning empty at
  `n_blocks == 0`.** Rejected: the empty return is deliberate and load-bearing for short
  generation (`morph/model/CLAUDE.md` documents the 2026-08-18 crash a padded block would
  reintroduce). The defect here is a config mismatch between a ratio and a sequence length,
  not a bug in the compressor's edge case.
- **Leaving the default at the audited value (256, dead) and shipping the fix only as an
  opt-in arm.** Rejected by Wolfe 2026-09-11: three of six core blocks running at half
  attention output for the whole of every slot-loop run is a correctness defect, not a
  hyperparameter choice, once the measured cost (0.03 nats, 4% wall clock, no depth change)
  is known and small.

## Consequences

- Every slot-loop config gets the repaired core by default; no arm needs to opt in.
- `base.yaml` and the paid-loop lineage (`tul_a2`, `tul_fm1`, ...) are unaffected: they
  never compose `tul_short`, and the paid loop's `L_total` = 1152 already gave `n_blocks =
  4` under the old (still-256) ratio.
- `tul_slot_mux_hca_fix.yaml` is deleted; its arc row, filing, and the proposed note that
  motivated it
  ([`2026-09-10-slot-loop-readout-and-attention-defects.md`](../../proposed/architecture/2026-09-10-slot-loop-readout-and-attention-defects.md))
  keep their history and now note the fold-in.
- `tul_slot_mnext_parcae_core.yaml` needed one line (`core_hca_compress_ratio: null`) to
  keep composing, since its Parcae core refuses the key outright — the earlier arm predates
  the new default and never had the defect the default fixes.
- The 46,080-parameter reduction (`B_a` at `[16, 64]` instead of `[256, 64]` per HCA core
  block, 0.017% of the model) ships with the default. It is not iso-parameter against the
  pre-fix ratio, in the direction that makes the measured CE improvement harder to explain
  as added capacity.
