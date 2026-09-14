# Agent Note: Horizon-indexed passes with a gated all-pass readout (LoopMTP port)

Status: rejected — measured 2026-09-14: paired −0.0001 ± 0.0024 vs the fixed-6 control, K3−K6 +0.0002, gate 0.83 on pass 1, and the six horizon targets sit within 0.01 cosine of each other (the passes were never given different jobs). lab/experiments/failures/2026-09-14-arc-horizon-passes.md

## Problem

Eleven prior slot-loop arms gave pass t a per-pass target and read a K3-K6 close to
zero every time: a growing decoded span (`spandec_per_pass`), a frozen gradient-descent
trajectory (`oracle_z`), staged own/forecast targets, gradient-conditioned passes, a
token map trained into the core. Every one of them scores pass t against a label pass
t-1 already had, restaged or grown — the literature-mining pass against this failure
(`docs/references/looping-depth/2026-09-13-lit-mining/D_objective.md` §1) reads LoopMTP
(arXiv 2608.03624) as the one paper in the batch whose mechanism is a genuine content
decomposition rather than a restaged copy of one label: pass t (t>=2) is graded on the
embedding of what is t steps AHEAD, a different target for every pass, and the paper
reports gains near MORPH's own parameter scale (260-280M) with a cheap cosine loss (no
hard-CE vocabulary projection). Nothing in this tree has ever given the slot loop a
target that actually differs from pass to pass.

## Proposal

Port LoopMTP's two mechanisms to the slot loop, span standing in for token:

1. **Horizon-indexed alignment** (`tul.horizon_weight`, `tul.horizon_free_first`,
   `tul.horizon_tokens`): pass t's state is compared by cosine similarity to the
   DETACHED, mean-pooled tied-embedding representation of span i+t, through a small
   learned projection (`tul_horizon_proj`) on the pass side only. Pass 1 is left
   unconstrained by default (`horizon_free_first: true`), matching the paper's own
   choice that it serve as a substrate the horizon-specific passes read from. No
   decoder: the target is a masked mean over token embeddings, so the term is cheap
   next to `spandec_per_pass`'s per-pass decode.
2. **Gated readout** (`tul.pass_readout: "gated"`): the state written to the prefix
   cells and graded by the span decoder becomes a content-conditional softmax mixture
   of every realised pass's state (`TULPassGate`, LoopMTP Eq 9-11: `softplus(Wg
   x_t + beta_t)`, normalised across t, weighted sum), not only the last pass. Every
   downstream reader of `h_slots` — the MUX, the span decoder, SIGReg, the gate budget,
   `prefix_project` — sees the same mixture, at the same seam `cond_layers` /
   `center_exit` / the register's mean / VQ already share.

Both knobs require `tul.slot_depth_fixed > 0`: pass t's target is a fixed offset from
the iteration count, and under the per-slot Poisson draw "pass t" is not the same thing
for every slot in a batch. `TULConfig.__post_init__` raises otherwise.

`pass_readout="last"` (default) with `horizon_weight=0.0` (default) is bit-identical to
the tree before this change — proven by running the pre-arm source in a second process
on the same CPU fixture and comparing loss / logit sum / n-finite / grad sum to the
last printed digit (`tests/test_tul_horizon.py::test_off_is_bit_identical_to_the_pre_change_source`),
not argued from the diff.

## Alternatives considered

* **All passes target one fixed offset** (e.g. every pass graded on span i+3), which is
  the queued `off2`/`off3` family already in this panel's tree
  (`tul_slot_spandec_strict_off2.yaml`, `tul_slot_spandec_strict_off3.yaml`). Rejected
  as THIS arm's design because it does not differentiate passes from each other at all
  — it moves the target, not the per-pass DECOMPOSITION LoopMTP's mechanism is about.
  Kept as a separate, already-queued family; not duplicated here.
* **A hard cross-entropy decoder per pass** (the `spandec_per_pass` shape, but with a
  horizon-indexed span instead of a growing one). Rejected on cost: `spandec_per_pass`
  measured 35.7 decoder block-passes per real token against the ruler's 14.7, and the
  panel's queue has a rate floor a decode-per-pass term would likely miss. LoopMTP's
  own paper explicitly frames its cosine/embedding-space MTP as the cheap alternative
  to hard-CE vocabulary-projection MTP (Gloeckle et al.), which it says "hurt
  performance at a small scale" — the same cost argument applies here twice over (this
  tree's target is a SPAN, so a decoder call is already per-span, and horizon-indexing
  it multiplies that by T).
* **Per-cell horizons on the Thought Register** (`tul.slot_cells > 1`): grade each of
  the M register cells against a DIFFERENT horizon offset, combining this arm's idea
  with the register's multi-cell capacity. Deferred: the register interacts with
  `cond_layers`, `prefix_source`, and the span decoder's cross-attention
  (`spandec_reads_cells`) in ways this arm does not touch, and stacking three
  unmeasured mechanisms in one launch would make any result unreadable. A clean
  follow-up once this arm's own K-curve reading is in.
* **DiscoLoop's decode-then-encode realignment** (the D_objective.md §1 paper 4 read,
  same literature pass) as an alternative to the gated readout. Not this arm — it is a
  different builder's arm on the same panel (`slot-job-panel-2026-09-13`, per the vlt
  thread's build list); duplicating it here would waste the panel's compute on the same
  question twice.
* **Reusing `db_traj[-1]` unchanged and adding ONLY the horizon loss, no gated
  readout.** Considered as a smaller, one-factor-at-a-time version. Rejected for THIS
  launch because LoopMTP's own ablation (Fig 1a, "overwritten and discarded") frames
  the gated aggregator as necessary for the horizon-indexed labels to matter at all —
  a pass whose state is thrown away at the readout has no reason to specialise for a
  target nobody reads. Kept as an easy follow-up split (`pass_readout: "last"` with
  `horizon_weight > 0` composes and passes every test in the file) if the combined
  arm's result needs decomposing.

## Acceptance criteria

* `tests/test_tul_horizon.py` green: off is bit-identical to the pre-arm source
  (pinned, not argued); the term is positive, enters the loss, and its gradient reaches
  the loop and `tul_horizon_proj` alone (never the tied embedding table, never the span
  decoder); pass t's target is span i+t and not i+1 (both a unit-level check on
  `span_slots(shift=t)` and an end-to-end check with two confounds named and removed);
  rows without a span i+t contribute nothing; pass 1 is unconstrained by default and the
  knob to change that works; the gate normalises to one and its argmax pass can move
  the written z more than its argmin pass; the coda and the span decoder read the SAME
  gated state; the Poisson-depth refusal fires for both new knobs; every new key is in
  `tul_setup.KNOWN_TUL_KEYS`.
* Both configs (`tul_slot_spandec_strict_horizon.yaml`,
  `tul_slot_spandec_strict_fixed6.yaml`) compose through Hydra with no missing/unknown
  key and pass `reject_unknown_tul_keys` — checked at this note's commit.
* `lab/experiments/failures/2026-09-14-arc-horizon-passes.md` frozen before launch.

## Risks

* **Two factors at once against the ruler** (fixed depth AND the new mechanism) is
  exactly why `tul_slot_spandec_strict_fixed6.yaml` exists as the control — any claim
  about the horizon arm that is read against the Poisson-depth ruler directly, rather
  than against `fixed6`, is not a clean reading and should be treated as unverified.
* **The gated readout is NOT training-only** — unlike every other per-pass mechanism
  this panel has shipped (which are training-only auxiliary losses, inert at eval), it
  changes the model's real forward at train and eval. A checkpoint trained with
  `pass_readout: "gated"` is not comparable to one trained with `"last"` on ANY
  downstream instrument that assumes the exit state is `db_traj[-1]`
  (`worth_profile`'s `zero`/`all_slots` reading, `slot_z_optimize`, the plan
  ablations) unless that instrument is re-read through the gate. Not audited here.
  `lab/divergence/horizon_pass_probe.py` reads the gate directly for exactly this
  reason (never re-derives it offline).
* **`horizon_pass_probe.py` is unexercised against a real checkpoint.** Its arithmetic
  was dry-run against the tiny CPU fixture (cosine matrix, consecutive-pass cosine, and
  gate weights all computed correctly and the gate weights summed to 1), but the CLI
  path — `build_cfg`, `load_ckpt`, `create_dataloader`, `pack_rows` — has not executed
  even once. The first real invocation, once a checkpoint exists, is the actual gate
  for this script, not this note.
* Eleven prior arms' K3-K6 ~ 0 is the strong prior. This arm's own prereg puts the
  headline claim (P-4, K3-K6 moves) at 30% — the honest expectation, going in, is
  another null result on the K-curve, with the T x T matrix as the more likely place to
  see anything move at all (P-5, 45%).
