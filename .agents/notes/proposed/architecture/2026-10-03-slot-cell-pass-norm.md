# Agent Note: a per-pass RMSNorm on the slot loop's cells

Status: proposed

Date: 2026-10-03. Built and CPU-tested on branch `wt-cnorm` (from master `48f89c8`); no
training run yet. Key: `tul.slot_cell_pass_norm` (`off` | `rms`).

## Problem

The latent-pulled slot-loop arms change their cells mostly by SCALE. At pass index 2 the
cells take a step 3 to 5 times their own norm, 79 % to 93 % of it along one shared
direction, and the cell norm grows 26 -> 216 by pass index 5
([common-mode note](2026-10-02-layernorm-common-mode-in-latent-targets.md), "The puzzle").
In the rank-only arm the whole K1-K6 rides on ONE per-cell amplitude along that direction:
setting the amplitude to its mean takes K6 - K1 from -0.0052 to +0.0003 (the `--vablate`
table in the same note).

Huginn-0125, a looped LM that earns depth on our web text, does the opposite. Every block
ends in `x = norm_4(mlp(norm_3(x)) + x)`, so its recurrent state has a fixed per-token
scale; every step from k = 2 on is 99.6 % to 99.9 % perpendicular to the state's shared
direction, and the centred participation ratio triples from k = 1 to k = 4
([filing](../../../../lab/experiments/successes/2026-10-02-huginn-loop-geometry.md)).

The slot loop has no such pin. Its carrier leaves each pass at whatever scale the core
step produced. The question this key opens: if the slot loop can only rotate its cells,
does it find depth in direction instead of in amplitude?

## Proposal

`tul.slot_cell_pass_norm: rms`. After EVERY slot-loop pass each carried cell is
`RMSNorm(h) * g`:

- `RMSNorm` is the tree's `morph/model/attention.py::RMSNorm` (eps 1e-6). It runs per
  Hyper-Connection stream over the channel axis, so each of the four streams of a cell
  leaves every pass at RMS `rms(g)` (exactly 1 at init). The stream MEAN, which
  `_cell_readout` and the common-mode note read, is not pinned: it can still shrink when
  the streams disagree.
- `g` is ONE learned per-channel gain shared by every pass (Huginn's shared block norm):
  `MORPHTransformer.tul_cell_norm.weight`, init 1.0, fp32 master. The name contains
  "norm", so the optimizer puts it in the no-decay group; it is not an `nn.Linear`, so
  ternary QAT never selects it (`ternary_qat._categorize`). Building it draws no RNG.
- The entry state `core_init(input_norm(prelude))` is NOT normed by this key. Note that
  `input_norm` is itself an RMSNorm, so at init the entry already has per-stream RMS 1.

**Placement.** ONE call site, `_slot_cell_norm`, in `MORPHTransformer._tul_core`'s pass
loop. Every path that iterates the slot cells (the single-cell loop, the fan's M > 1
cells, the latent-selected loop, the stage-1 latent pretraining forward) runs that loop,
so none needs a copy. Relative to the other per-pass mechanisms:

| mechanism | where it sits | what it reads under `rms` |
| --- | --- | --- |
| `DiagonalInjection` (and the reread, the trigger, the carry re-injection) | inside the pass, before the norm | the previous carried (normed) cell |
| core step, `slot_state_renorm`, `core_gain_clip`, recurrence gate, LX / policy code | before the norm | the last two are refused with `rms` (below) |
| the norm | the end of the pass | |
| `_slot_pass_hook` (eval-only instrument) | after the norm | reads and edits the normed cell |
| fixed-point term (`core_fixed_point_lambda`), pass residual | after the norm | `_h_det` is the NORMED output; the previous state is the last normed cell, or the un-normed entry at a depth-1 slot |
| gate readout, loop probes, cotangent hooks (`slot_cot_clip`) | after the norm | the normed carried cell |
| freeze of finished slots, `db_traj` | after the norm | normed cells, so the fan's diversity term (`fan_repel`, epivol) and every per-pass instrument read normed cells |
| latent-selected loop: router, teacher, reset to the winner, exit head and loss | after the norm | the normed candidates; the reset copies a normed winner |
| `prefix_project` (the write) | after the loop | the normed exit |

The norm takes the pass's grad context: a pass outside the truncated-BPTT window runs it
under `no_grad`, and `tul.progressive_p`'s prefix cut detaches its output, so `g` gets no
gradient from passes the objective cut away (the recurrence gate's rule).

**What the slot-loop constraint terms mean under the norm (NOT changed):**

- The gain hinge (`model.slot_gain_lambda`, `_slot_gain_penalty`) re-runs `_core_step` at
  the normed operating point. `_core_step` does not contain the norm, so the hinge reads
  the gain of the RAW map f. The loop applies N(f). Along the cell, N removes the radial
  part; across it, N's Jacobian scales by `rms(g) / rms(f(h))` per stream. So the hinge
  still bounds f's local amplification in absolute units, but it does not bound the gain
  of the map the loop carries, and the forward cannot grow whatever it reads. The
  hinge's original job (keep the map's gain off 1 so the forward cannot inflate) is now
  done by construction in the forward; what the hinge does to the backward is untested.
  It stays on in both configs, unchanged. A test pins that it still reads f
  (`test_the_gain_hinge_still_reads_the_raw_map`).
- The cotangent clip (`model.slot_cot_clip`) hooks the normed pass outputs and the normed
  exit, so it compares the cotangents at the states the loop carries, as before. The
  backward through each pass now includes N's Jacobian.
- `loop/core_gain` (the probe's `||out|| / ||in||`) reads about 1 after pass 0 by
  construction. It says nothing about the map under this key.

**Refusals.** TULConfig: the paid loop and the token path (`_core_region`, no slot loop),
`tul.code` (no slot loop), `core_stage_cond` (the sigma arms bypass `_tul_core`; iter not
built), `gram` (noise added after the map), `loop_carry: persist` (an add to the exit
after the last pass), `loop_denoise` (each pass enters at a noised code), `xhc_streams`
(zero-initialised extra streams: RMSNorm's Jacobian at an exact zero is `g / sqrt(eps)`).
`MORPHTransformer.__init__`: the FM planner, `n_core: 0`, SCSE (the carrier is a
deviation), `slot_state_renorm` (a second scale pin), `core_gain_clip > 0` (a rescale the
norm divides out, so a silent no-op).

**Readings.** At eval only: `val/slot_cell_rms_t{t}` (mean per-cell RMS over streams and
channels after pass t; should sit at `rms(g)`), `val/slot_cell_mean_rms_t{t}` (the stream
mean's RMS), `val/slot_cell_rms_entry`, `val/slot_cell_norm_g_{mean,std,rms}`, and one
console line per val.

**Arms** (one key over their parents, plus the run name):

- `tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1_cnorm`: the detached,
  router-followed latent-selected loop at weight 1, the arm with the largest K1-K6
  (+0.0333) and the largest pass-2 step (4.72x).
- `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm`: the rank-only
  head, whose K1-K6 is all amplitude. If its depth signal survives the norm, it moved into
  direction; if it does not, amplitude was the only channel it had.

## Alternatives considered

- **`model.slot_state_renorm`** (built 2026-09-04): rescales each slot, over all streams
  and channels jointly, to the norm it ENTERED the loop with, with no learned gain. It held
  the 2026-09-04 spike train as a stability lever. It is per slot, not per stream, so the
  four streams can still trade scale, and it pins every slot to its own entry norm rather
  than one learned scale. Huginn's mechanism is a per-token RMSNorm with a learned gain,
  and that is the one this key copies. Refused together with `rms`.
- **A norm inside `_core_step`** (so the gain hinge would difference N(f), the
  `slot_gain_renorm` pattern). Rejected for this build: it would silently change what the
  hinge measures on both configs, and the brief says the hinge must not change silently.
  It is the follow-up if the hinge's reading under the norm turns out to matter.
- **One gain per pass.** Rejected: Huginn's recurrent blocks, and their norms, are the
  same weights at every iteration, and a gain per pass gives the loop a per-pass scale channel back, which is
  the thing this key removes.
- **Normalise the stream mean instead of each stream.** No natural operator on the
  carrier does that (the four streams would have to be rescaled jointly by the mean's
  RMS), and the core reads the streams, not their mean.
- **Remove the shared direction `v` at the write** (the common-mode note's third
  alternative). Rejected there: zero-ablating `v` costs 0.006 to 0.014 nats and part of
  the K-curve. This key does not remove a direction; it removes scale.

## Acceptance criteria

- Met 2026-10-03 (CPU): `off` is bit-identical to master `48f89c8` on three slot-loop
  fixtures at two dropouts, forward and backward; the placement, shared-gain, gradient,
  fixed-point, hinge, grad-context, refusal and config tests in
  `tests/test_slot_cell_pass_norm.py` pass; seven sabotages (norm at the exit only, norm
  on the entry, frozen `g`, a gain per pass, fixed-point term on pre-norm states, the norm
  inside `_core_step`, the norm ignoring the no-grad window) are each caught.
- GPU smokes of both configs (45 steps, val at 40): see the report of the 2026-10-03 build.
- Open: a prereg in `lab/experiments/planned/` before the two 5k runs, with predictions on
  K1-K6 against each parent (480 rows, the exploration ledger), the pass-2 step size and
  share along `v`, `val/slot_cell_mean_rms_t{t}`, and the token CE gap.

## Risks

- The hinge now bounds f, not the carried map (above). Under the norm the forward cannot
  inflate, so the spike train's forward half is closed by construction, but the backward
  through N scales by `rms(g) / rms(f(h))`: a map that shrinks f's output amplifies the
  cotangent. `slot_cot_clip` still bounds it. Not measured.
- `g` can grow: a larger `rms(g)` is a global scale knob, the same for every cell and
  pass, so it cannot carry per-input amplitude. Watch `val/slot_cell_norm_g_rms`.
- The stream mean's scale is free (it shrinks when the four streams disagree), so a
  per-cell amplitude can partly come back through stream alignment. Read
  `val/slot_cell_mean_rms_t{t}` before claiming the loop has no amplitude channel.
- The pulled arms' shared direction `v` was a bias the coda relies on (zero-ablation cost
  0.006 to 0.014 nats). Under the norm `v` can still exist as a direction, but its
  amplitude of 130 to 255 cannot: the coda must read a bias of fixed size. This may cost
  CE in the first thousand steps.
- Pads are normed too. They were never zero after pass 0, and every reader masks them by
  `slot_valid`; a new reader that forgets the mask would read unit-RMS pads.
