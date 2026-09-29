# Agent Note: LX efficient-exploration knobs (score head, latent set loss, sampled decode, hypothesis merge)

Status: proposed

## Problem

The LX code-enum arm (`tul.code_enum_k`, K rollout-major expanded batch, exact per-span
mixture CE via `_enum_mix_losses`) buys a true posterior over K hypotheses per span at a
K-times cost: every rollout runs the full prelude, loop, and coda. Several efficiency and
read-out questions sit downstream of that posterior and are currently unanswered on this
tree:

- Can a cheap head learn to predict the mixture's own posterior, so eval can pick a
  rollout without recomputing the exact bound (rMCL 2311.01052, LatentRM 2510.07745,
  LTO 2509.26314)?
- Does an explicit diversity-preserving loss on the hypothesis latents (CALM
  2510.27688, CPC 1807.03748) change what the K rollouts converge to, independent of the
  token CE?
- Does the coda need to run all K rollouts every step, or does a uniform subsample
  (Kool et al. 2019/2020 SWOR, LTC 2608.01593) already train the mixture well enough
  (never for eval)?
- Is a merged hypothesis (mean, or a learned gate — Superposed Decoding 2405.18400, LPLT
  2409.00070, ParScale 2505.10475) ever better than the single best rollout, or does it
  repeat the blurred-mean losses this tree already measured (the soft LX fans 0.030 nats
  behind the WTA fan with 3 of 4 draws detonated,
  `lab/experiments/failures/2026-09-27-lx-soft-fan-retry.md`; the four-copies fan, arm a1
  of `docs/slot-cells-distinct-vs-blurred.md`)?

None of these have been tried. This note scaffolds five default-off knobs to try them,
without touching the shipped recipe or the arm under active development elsewhere
(latent-space WTA winner-pick, `_tul_fan_all`, built and merged separately).

## Proposal

Five `TULConfig` knobs, all default off, all gated behind `code_enum_k >= 2` and refusing
`fan_k > 1` (the WTA arm's own axis — the two are not composed in this note; that
combination is future work, not scaffolded here):

1. **Late fusion in the coda** (`tul.coda_fuse_layer`) — NOT implemented. See "What was
   not built" below.
2. **Score head** (`tul.hyp_score_head`) — a tiny
   linear head (`TULHypScoreHead`, `morph/model/tul_explore.py`) reads each rollout's
   DETACHED slot-exit latent and is trained by forward-KL
   (`score_head_kl_loss`) against the exact mixture's own DETACHED posterior
   (`_enum_mix_losses`'s `s_out["S"]`, added in c2d77a3 for exactly this read). No
   gradient reaches the loop, the codes, or the front — every input is detached before
   the head sees it (proved structurally, not just by convention: see Testing).
   The eval read stays the exact K-way Bayes read. What a head-picked read WOULD cost is
   reported every step, train and val: `hyp_read_top1_gap` / `hyp_read_top2_gap` (CE of
   the mixture restricted to the head's top 1 / top 2 rollouts, minus the exact
   mixture's CE, nats per token), beside the random-pick floor `hyp_read_rand1_gap` and
   the hindsight ceiling `hyp_read_best1_gap` (`score_head_read_gaps`), and
   `hyp_score_agree`. A head whose top-1 gap sits at the floor is guessing.
3. **Latent set loss** (`tul.latent_set_loss: none|energy|infonce`,
   `tul.latent_set_weight`) — an explicit diversity-preserving loss over the K rollouts'
   readout latents, graded against the SAME detached, prelude-pooled next-span target
   `tul.row_contrast_lambda` already uses (`next_span_pool` on `_readout(x.detach())`),
   never the loop's own next-slot exit state (that target is on record as collapsing,
   see `latent-prediction-on-slot-exit-collapses` in project memory). `energy`
   (Gneiting & Raftery 2007, `latent_energy_score`) and `infonce`
   (`latent_infonce`, in-batch negatives) are both strictly-proper / non-collapsing by
   construction; MSE was deliberately not offered (foot-gun: MSE-style latent targets
   collapse to the target's mean, per project memory on LCTUL and NextLat). This term is
   LIVE on the hypothesis side and does reach the loop's codes — this is the one knob of
   the five that is not gradient-isolated from the loop, by design, since its entire
   purpose is to shape what the loop's rollouts produce.
4. **Sampled decode** (`tul.enum_decode_k`) — training-only: the coda runs on only
   `enum_decode_k` of the `code_enum_k` rollouts, selected by uniform SWOR
   (`swor_uniform`, `torch.randperm`). A head-weighted draw is refused (see Review
   corrections). Row-block gather
   (`gather_rows`) cuts the coda's expanded-batch pass from `K*B0` to `k*B0` rows before
   `_back_region`. Eval always reads every rollout — this knob never touches
   `_enum_mix_losses`'s eval-time exact bound.
5. **Hypothesis merge** (`tul.hyp_merge: none|probe|learned`, `tul.hyp_merge_weight`) —
   `probe` computes a detached mean-merge and logs whether it beats the best single
   rollout (`hyp_merge_probe_stats`: `merge_dist`, `best_single_dist`,
   `merge_beats_best_single`) with zero training cost, so the mixture-non-closure
   question can be checked BEFORE spending a weight on it. `learned` additionally builds
   `TULHypMergeGate` (softmax-attention pool, convex combination) and trains it at
   `hyp_merge_weight` toward the detached next-span target.

Every knob's foot-gun is documented at its `TULConfig` field (arXiv citations + the
specific project-memory finding it is guarding against) and re-stated in its own config
YAML's header comment.

## Alternatives considered

- **MSE regression for the set loss.** Rejected: project memory
  (`latent-prediction-on-slot-exit-collapses`, `db-bridge-is-the-bug`) already shows
  MSE-style latent targets collapsing toward a common-mode mean on this codebase's own
  architecture family; energy score and InfoNCE are both strictly proper and resist that
  collapse by construction.
- **Averaging all K hypotheses unconditionally (no probe, no gate).** Rejected up front:
  the soft (mean-mixed) LX fans and the four-copies fan already lost on this tree (see
  Problem). `hyp_merge: probe` exists
  specifically so a future run checks `hyp_merge_beats_best_single` before any weight is
  spent, rather than re-discovering the same failure.
- **Training the score head with the loop's live gradient (end-to-end).** Rejected:
  would make the head a second, redundant path into the same codes the token CE already
  trains, at extra compute, with no way to attribute credit. Detaching keeps the head's
  cost near zero and its purpose singular (learn to predict a posterior that already
  exists, not to change it).
- **Score-weighted (Gumbel-top-k) SWOR for knob 5.** Built by the first draft and then
  removed in review: the forward ran the plain k-way mean on the head-biased sample with
  no importance weights, so it trained on the rollouts the head already preferred. A
  correct version needs the Gumbel-top-k inclusion probabilities (Kool et al. 2020).
- **Composing these knobs with the WTA arm's `fan_k`.** Rejected for this note: three
  agents were touching `_tul_fan_all`/`_tul_fan_all`'s winner-pick concurrently at
  authoring time; all five knobs here explicitly refuse `fan_k > 1` at construction so
  the two lines of work cannot silently interact before a human reviews both. Composing
  them is a legitimate follow-up, not scaffolded here.
- **A shared config key for all five knobs' costs/weights.** Rejected: each knob's cost
  or weight is semantically different (a boolean gate, an enum + float weight, an
  integer count, an enum + float weight) and collapsing them into one generic
  "explore_weight" would hide which knob a given run actually paid for.

## Acceptance criteria

- `TULConfig` carries all 5 (8 counting sub-fields) new fields, each refusing every
  invalid combination named above, validated at construction (`__post_init__`, at the
  TOP of the method — this file's own tail-placement validators are dead code, see
  Testing).
- `build_tul_runtime` (`morph/training/tul_setup.py`) passes every field through;
  `KNOWN_TUL_KEYS` and the wandb manifest both carry all 8 keys.
- Every knob defaults to off and is bit-identical to the tree without this note's changes
  (`test_every_knob_defaults_to_off_bit_identical`, state-dict AND forward-output
  equality, dropout 0.1).
- Each knob's defining property holds on a real forward+backward, at dropout 0 AND
  dropout > 0.
- Every refusal documented above fires as a config-construction error, not a runtime
  crash mid-forward.
- One config per knob, each composing `tul_slot_spandec_strict_e4probe_fp01` (not the WTA
  arm — see "Config parent" below), 3000 steps, `lr_decay_steps: 5000`, not queued.

## Risks

- **Knob 4 (sampled decode) forces `checkpoint_blocks=False` and bypasses
  `RolloutSharedDropout` for the affected coda call.** This was required to avoid a
  `batch % n_rep != 0` crash under a non-divisible gather (K=4, k=3 style draws) and to
  avoid a forward/recompute mismatch between the bypass context and a checkpointed
  block's later backward recompute. It means a knob-4 run pays full (non-checkpointed)
  activation memory for the coda's affected rows — untested at real K/seq sizes, only at
  the tiny CPU fixture.
- **Knob 3's `_tul_explore_post_coda` always adds `hyp_score_kl` at implicit weight 1.0**
  when the head is on; there is no `hyp_score_weight` knob to scale it down. If a real
  run finds the KL term dominates early training, that is a missing knob, not a bug —
  flagged here rather than added speculatively.
- **Knob 5 changes the OBJECTIVE, not only its cost.** The k-way mixture over a uniform
  subsample is the IWAE-k bound on the K-way mixture: its log is biased low, so the loss
  reads high, and its credit for rollouts that DIFFER falls with k. At k = 1 it is the
  mean single-rollout CE, which pays nothing for a rollout that explains a span the
  others miss, the opposite of what the LX mixture rewards. Read `enum_width_gain` on a
  knob-5 run against its k = K twin before anything else.
- **None of the four runnable knobs (3, 4, 5, 6) have been trained even to 3000 steps.**
  Every config in this note is explicitly not queued. The forward/backward tests prove
  the code paths are reachable, finite, and gradient-correct on a tiny CPU fixture —
  they do not prove any knob helps, or even trains stably, at real model scale.
- **Knob 2 (late fusion in the coda) was investigated and is NOT implemented.**
  `tul.coda_fuse_layer` exists with exactly one legal value (0, i.e. off) and raises
  `NotImplementedError` on any other value. Reasoning: under strict geometry the
  token-path rollout-invariance argument is architecturally sound (RoPE/CoPE cancel for
  same-position self-attention; strict-geometry prefix cells carry zero cross-rollout
  injection), which is what would make a shared token-path pass safe. But the cell-side
  compute saving that would justify building this needs either (a) a gather against
  GLA/retention cross-position state that has not been verified safe under strict
  geometry, or (b) a full-sequence-shape shared pass that saves nothing over running all
  K rollouts, defeating the point. Rather than ship a knob whose only implementable form
  saves nothing, this was left unbuilt. A future note should pick this back up if the
  GLA/retention interaction gets verified separately.

## Review corrections (orchestrator, 2026-09-29, before merge)

- train.py did not subtract `explore_set_weighted` / `explore_extra_weighted`, so an arm
  with knob 3 or 4 on would have logged `train/loss` and val loss with the auxiliary term
  inside. Both are now in both subtraction lists, and `explore_*` / `hyp_*` are logged
  (`tul/` at train, `val/` at eval). Before this, none of the new stats reached wandb.
- `hyp_score_read: top1|top2` was accepted and read by nothing. Removed; replaced by the
  read-gap instrument above. Its wiring test checks `hyp_read_rand1_gap` against the
  independently built `enum_width_gain` (equal, because the tokens before the first slot
  read no cell), and fails when the slot-to-span offset is shifted by one (sabotage run).
- The head-weighted draw of knob 5 claimed a self-normalized estimator it never called.
  Removed and refused; `swor_score_weighted` and `mixture_estimate_self_normalized` are
  gone with it.
- `score_head_topk_mask` claimed lowest-index tie breaking from `torch.topk`, which does
  not promise it (a test with constant scores failed). It now uses a stable sort.
- The knob-6 comments attributed "four copies of one state, 0.15-0.40 nats" to
  dmorph-v1. dmorph-v1 was the noisy-stream design; those numbers are its cost, not a
  merge's. Replaced with this tree's real blurred-mean measurements.

## Testing

- `tests/test_tul_explore.py`: pure-function math for every helper in
  `morph/model/tul_explore.py` against hand-computed references, including sabotage
  checks on `latent_energy_score` and `gather_rows`.
- `tests/test_tul_explore_wiring.py`: model-level. Off = bit-identical
  (state-dict and forward output, dropout 0.1). Every refusal. Every knob's defining
  property at dropout 0 and dropout > 0, including a deliberately non-divisible
  `enum_decode_k` draw (K=4, k=3) that exposed and fixed a real
  `RolloutSharedDropout: batch N is not a multiple of n_rep K` crash. The score head's
  no-leak guarantee (`test_hyp_score_head_trains_and_reports_kl`) is proved structurally
  by spying on the real `_tul_explore_pre_coda` call site and asserting the tensors
  reaching the head have `grad_fn is None` — an earlier version of this test compared
  `tul_code_enum.basis.grad` across two independently-run models and failed, because two
  independent forwards draw different Poisson core depths / dropout masks / code
  samples and so legitimately produce different `basis.grad` values for reasons that
  have nothing to do with the score head; that comparison was a broken test, not a real
  leak, and was replaced with the structural check.
- Full regression: `pytest tests/ -k tul` — 1905 passed, 1 skipped, 1 xfailed (pre-existing,
  unrelated to this note) at commit range 38ded25..c2d77a3 plus this note's changes.
- Full tree: `pytest tests/` run separately to confirm no non-TUL regression (see the
  handback report for the exact count and pass/fail).

## Config parent

The task's naming template (`tul_slot_spandec_strict_lxfan4_wta_fp01_<knob>.yaml`)
composes on the WTA/fan arm. All four runnable knobs here refuse `fan_k > 1`, so
composing on that parent would raise at Hydra construction. Every config in this note
instead composes `tul_slot_spandec_strict_e4probe_fp01.yaml` (the plain LX arm, no fan),
named `tul_slot_spandec_strict_e4probe_fp01_<knob>.yaml`. This deviation is documented in
`tests/test_tul_explore_wiring.py`'s module docstring and in each config's own header.
