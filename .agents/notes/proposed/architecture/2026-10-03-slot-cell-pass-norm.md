# Agent Note: a per-pass RMSNorm on the slot loop's cells

Status: proposed

Date: 2026-10-03. Key: `tul.slot_cell_pass_norm` (`off` | `rms` | `rms_read`). `rms` was
built from master `48f89c8` and trained as a pair (5000 steps, filed in
[`2026-10-03-slot-cell-pass-norm-pair.md`](../../../../lab/experiments/failures/2026-10-03-slot-cell-pass-norm-pair.md));
`rms_read` was built the same day on `wt-cnorm` from master `bac34a9` after that filing,
with a 45-step smoke and no training run yet.

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

### Measured: the `rms` pair (2026-10-03, seed 1, 5000 steps, 480 ledger rows)

| arm | ledger CE | K1-K6 | parent K1-K6 |
| --- | --- | --- | --- |
| rank-only head + `rms` | 4.353 (parent 4.380) | +0.0215 [+0.0205, +0.0225] | +0.0052 |
| detached weight 1 + `rms` | 4.405 (parent 4.447) | +0.0119 [+0.0109, +0.0130] | +0.0333 |

The rank-only head's loop earns 4x its parent's depth, and all of it is directional: on
the shared direction's mean-vs-zero ablation, K6 - K1 reads -0.0219 shipped and -0.0217
mean-ablated (its parent's whole -0.0052 went to +0.0003 under the same ablation). The
detached weight-1 arm lost depth, and the LayerNorm probe still reads its cells jumping
onto one shared direction at pass index 2: cosine to it 0.02 -> 0.92, probe units 1.41 ->
3.24. The cells that probe reads are the stream MEAN, and `rms` pins each stream, not
their mean. The filing's reading: the read state can still grow when the streams align.

### Second build: `rms_read`, a fixed scale on the READ state

**What each consumer reads from the slot carrier `[B, S*M, n=4, C]`** (traced in code
2026-10-03):

| consumer | what it reads |
| --- | --- |
| `prefix_project` (the coda write) | every stream separately: `W_prefix[k]` is applied to each of the n streams of the cell, and the coda gets an n-stream carrier at the prefix position |
| the coda's blocks | `x_bar = sum_i Hpre_i x_i` (a softmax, so convex, per-token mix of the streams; the coefficients come from the RMS-normed flattened carrier), then the block's pre-norm RMSNorm. At init `Hpre` is near uniform, so `x_bar` is the stream mean |
| `_readout` (LM head, MUX, span decoder) | the stream MEAN, then `lm_mixer`, `final_norm` |
| latent head, teacher, router of the latent-selected loop (`_lsel_pass`, `_lsel_finish`) | `_cell_readout`: the stream MEAN of each candidate |
| fan diversity terms (`fan_repel_term`, `fan_epi_term`, `fan_vol_term`) | `_cell_readout` of `db_traj`: the stream MEAN |
| fixed-point and pass-residual terms | the whole carrier, flattened over streams and channels |
| `lab/divergence/ln_common_mode_probe.py`, `collect` -> `_cells4` | the stream MEAN of the latent-selected loop's per-pass candidates (`_lsel_capture` "pre") |

So the READ state is the stream mean: every latent consumer, the probe and `_readout`
read it, and the coda's own pre-map starts at it.

**`rms_read`** (`MORPHTransformer._cell_norm_apply`, the same one call site, the same
module and the same `g`). After every pass:

1. each stream goes to unit RMS over C, no gain: `u_i = h_i / rms(h_i)`;
2. the read `m = mean_i u_i` is swapped for `RMSNorm(m) * g`: `out_i = u_i + (RMSNorm(m) *
   g - m)`, ONE single-stream term broadcast into every stream (`_apply_injection`, the
   carrier's rule for every injection).

Then `mean_i out_i = RMSNorm(m) * g` exactly. Identical streams (`m` has RMS 1) and
orthogonal streams (RMS 0.5) give the same read; under `rms` they give read RMS 1 against
0.5, which is the growth-by-alignment the probe saw (a test pins both). The read's RMS is
1 at init and `rms(m_hat * g)` after, set by `g` and the read's direction, never by the
streams' size or alignment. Each stream stays bounded: `out_i - read = u_i - m`, RMS <= 2.

Everything else in the placement table above holds unchanged: the fixed-point term reads
the normed carrier, the latent consumers read the normed READ, the hinge still reads the
raw map f, and the grad context is the pass's.

**Arm:** `tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1_cnorm_read` (parent: the
detached weight-1 `rms` arm), the key plus `training.steps: 3000` (Wolfe, 2026-10-03). The
LR is flat after its 1000-step ramp and `ademamix_t_beta3` is pinned at 3500, so the run
shares its schedules with the parent's first 3000 steps: compare against the parent at
step 3000.

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
  same weights at every iteration, and a gain per pass gives the loop a per-pass scale
  channel back, which is the thing this key removes.
- **Normalise the stream mean instead of each stream.** Rejected for the FIRST build: no
  single per-stream operator does it, and the core reads the streams. The pair's result
  (above) made it the second build, `rms_read`, by the write-back below.
- **`rms_read` by one scalar per cell** (`out = h * rms(g) / rms(mean_i h_i)`, the brief's
  first example). It fixes the read too, but it divides every stream by the read's RMS,
  so streams that disagree are blown up without bound (a read near 0 sends the streams to
  infinity), and the coda reads the streams one by one through `W_prefix` and its
  pre-map. Rejected for the per-stream unit norm plus a broadcast shift of the common
  part, which keeps every stream within RMS 2 of the read.
- **`rms_read` without the per-stream step** (only shift the mean: `out_i = h_i - m +
  RMSNorm(m) * g`). The read is fixed, but the deviations `h_i - m` are free, so the
  streams can grow apart without bound while the read stays put, and the coda's write
  sees that growth. Rejected; a sabotage of exactly this is caught by the tests.
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
- Met 2026-10-03: the `rms` pair trained and is filed (above).
- Met 2026-10-03 (CPU), `rms_read`: `rms` is pinned bit-identical to its first build
  (master `bac34a9`); identical vs orthogonal streams give the same read RMS at input
  scales 0.1, 1 and 100 (and `rms` does not); the read matches an fp64 reference for a
  random `g`; every pass's read divided by `g` has RMS 1 on three fixtures; the build,
  gradient, fixed-point, hinge and grad-context tests run under both modes. Five
  sabotages are each caught: a per-stream norm in place of the read norm, the read taken
  from stream 0 instead of the mean, no per-stream unit step, the read normed without
  `g`, a gain per pass.
- Open: a prereg in `lab/experiments/planned/` before the 3000-step `rms_read` run, with
  predictions on K1-K6 against the parent at step 3000, the LayerNorm probe's cosine and
  size at pass index 2, and the token CE gap.

## Risks

- The hinge now bounds f, not the carried map (above). Under the norm the forward cannot
  inflate, so the spike train's forward half is closed by construction, but the backward
  through N scales by `rms(g) / rms(f(h))`: a map that shrinks f's output amplifies the
  cotangent. `slot_cot_clip` still bounds it. Not measured.
- `g` can grow: a larger `rms(g)` is a global scale knob, the same for every cell and
  pass, so it cannot carry per-input amplitude. Watch `val/slot_cell_norm_g_rms`.
- Under `rms` the stream mean's scale is free, and on the detached arm it was used (the
  probe's 1.41 -> 3.24). `rms_read` closes that; under `rms_read` the per-stream RMS is
  the free one (each stream is `u_i - m + read`, RMS between 0 and about 2 + rms(g)).
  Read `val/slot_cell_rms_t{t}` there.
- `rms_read` divides by the read's RMS inside `RMSNorm(m)`. If the four unit streams
  nearly cancel, `m` is near 0, the read's direction is noise and its Jacobian is about
  `rms(g) / rms(m)`. Not seen in the smoke (read RMS 1.000 at every pass); not measured
  in training.
- The pulled arms' shared direction `v` was a bias the coda relies on (zero-ablation cost
  0.006 to 0.014 nats). Under the norm `v` can still exist as a direction, but its
  amplitude of 130 to 255 cannot: the coda must read a bias of fixed size. This may cost
  CE in the first thousand steps.
- Pads are normed too. They were never zero after pass 0, and every reader masks them by
  `slot_valid`; a new reader that forgets the mask would read unit-RMS pads.
