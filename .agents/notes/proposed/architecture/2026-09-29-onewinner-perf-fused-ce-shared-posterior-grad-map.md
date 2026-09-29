# Agent Note: onewinner-perf — fused span CE, true-posterior pass sharing, grad-on-MAP-rollout

Status: proposed

## Problem

`_tul_fan_all`'s "map" mode (2026-09-29-onewinner-shared-map-rollout.md, "onewinner") cut the
picking table from `K*B0` to `B0` rows but kept three costs the 30-step GPU trace showed were
still large: (1) the WTA pick/rank/grad passes computed per-span CE with a per-row
`xh[b] @ w_head.T` → `.to` → `.contiguous().clone()` → `F.cross_entropy` loop — 480 `clone`
calls in 4 profiled steps, 0.158s of the trace's 1.402s of `aten::copy_` time; (2) "map"'s
ranking pass ranked rollouts by the MEAN-of-cells write, a proxy the coda never reads at train
(`fan_mix=all` reads every cell 1:1) — the model's real posterior (`_enum_mix_losses`'s exact
per-span mixture) was computed by a SEPARATE, later, full-cost coda pass in `_forward_tul` that
`_tul_fan_all` could not see; (3) the winner-alone responsibility (`wta`) term always ran its
grad pass over all `K*B0` rows (the mean over K rollouts), even though nothing in the design
requires averaging that term over K.

## Proposal

Three orthogonal changes, one fused kernel and two knob-gated behavior changes:

1. **`accumulate_span_ce_fused`** (`morph/model/transformer.py`): a twin of
   `accumulate_span_ce` built on `fused_linear_label_logprob`
   (`morph/model/fused_ce.py`'s chunked, non-materializing `_FusedLinearLabelLogProb`) instead
   of a per-row `F.cross_entropy` loop. Same unmasked partition as the eager path
   (`mask_token_id` left off — proven by `test_fused_call_leaves_mask_token_id_off_by_default`,
   which shows masking strictly LOWERS the CE and this call does not). Wired into every WTA call
   site: the "map" ranking-replacement pass, the picking passes (both modes), and the grad pass.
   Value tolerance measured on the real `span_ce_index` layout machinery: max abs diff 7.6e-6,
   max relative diff 1.1e-7 (shipped tolerance `rtol=1e-4, atol=1e-5`, looser on purpose).
   Gradient tolerance: max abs diff 1.2e-6 (`xh`), 2.9e-6 (`w_head`) (shipped tolerance
   `rtol=1e-3, atol=1e-4`). Not bit-identical (a chunked `gather − logsumexp` vs `F.cross_entropy`'s
   internal `log_softmax`+`nll_loss` disagree at the ULP level over a `[L]` sum per row) — this
   moved 4 pre-existing pin tests' grad-sum fields by ~2.6e-9 relative (loss/logit-sum/eval-loss
   stayed exactly bit-identical at every pinned index); the 4 numbers were re-pinned with a dated
   comment, not silently loosened.

2. **`_tul_fan_all`'s "map" branch now runs the model's OWN deployed coda pass, itself, with
   gradient, BEFORE the no-grad picks** — the shared mixture pass. It computes
   `_enum_mix_losses`'s exact per-span posterior `S` (not the old mean-of-cells CE proxy),
   picks each slot's MAP rollout from `S.argmax`, and caches `(xh, groups)` in a new
   `MORPHTransformer._fan_all_deployed_cache` attribute. `_forward_tul`'s own later call site
   (the ordinary `fan_mix=all` deployed pass — previously a SEPARATE, redundant, full `K*B0`-row
   `_back_region` call) now checks that cache first and reuses it instead of recomputing. Net:
   the separate ranking pass AND the separate downstream deployed pass both disappear; one
   shared pass does both jobs. Refused in combination with `tul.bcast` (bcast's unpack term
   reads `h_slots`, which does not exist yet at the point this pass runs — no defined
   interaction, so refused rather than silently wrong).

3. **`tul.fan_all_wta_grad_rollouts: "all" | "map"`** (new `TULConfig` field, refused unless
   `fan_all_wta_winner == "map"`). `"all"` (default) keeps the existing behavior: the
   winner-alone responsibility term averages over all `K` rollouts (`K*B0` rows, checkpointed).
   `"map"` runs that same term on ONLY each slot's MAP rollout (`B0` rows, no checkpoint — see
   below) — a real objective change (one rollout's term instead of the mean over `K`), which is
   why it is a knob and not a free optimization.

   `"map"`'s grad pass deliberately drops `checkpoint_blocks=True`. `bypass_rollout_sharing`
   (`morph/model/rollout_dropout.py`, from the prior "onewinner" task) is a plain Python context
   manager; a checkpointed block's backward RECOMPUTE runs later, inside `.backward()`, after
   that context has already exited — a checkpointed `_grad_map` pass would either silently mask
   the RNG-based `RolloutSharedDropout` mismatch or re-trigger its `rows % n_rep` raise during
   backward. Since `B0` rows is a quarter the memory of the `K*B0`-row path that needed
   checkpointing, dropping it here is judged unnecessary and makes the bypass scope trivially
   correct (one un-recomputed forward, no second dropout draw).

Wired through `TULConfig.__post_init__` (both fields' refusals), `morph/training/tul_setup.py`
(`KNOWN_TUL_KEYS`, `build_tul_runtime`, the wandb manifest, the build banner).

## Alternatives considered

- **Reuse the full per-rollout table's own `ce.min(-1)` for free** (rejected in the prior
  "onewinner" note; restated here because this note's design supersedes that note's mechanism
  for ranking): building any second table on top of the `K*B0`-row table is strictly MORE work,
  not less.
- **Keep the mean-of-cells ranking proxy, measure whether it agrees with the true posterior
  first** (`lab/divergence/wta_winner_agreement.py`, prior task). Superseded, not merely
  deferred: since the true posterior is now available for FREE (the pass that computes it is
  the SAME pass "map" already needs to build for its deployed write), there is no reason left to
  ship the CHEAPER-BUT-APPROXIMATE proxy — the exact posterior costs nothing extra once the
  pass is shared. The prior note's "Reorder the forward to reuse `_enum_mix_losses`'s exact
  posterior" alternative was rejected there as "invasive... for a design whose whole selling
  point is a small, contained diff" — that trade-off is void once eliminating the SEPARATE
  downstream pass is on the table as the reward, not merely dodging the reordering cost. See
  `2026-09-29-onewinner-shared-map-rollout.md`, which this note does not replace (it stays the
  correct historical record of the first shipped mechanism) but does functionally supersede on
  this one point.
- **Fold the grad pass into the same shared mixture pass** (avoid a 3rd/4th coda call
  entirely). Rejected: the shared mixture pass reads the MIXTURE write (every cell, softmax
  posterior); the responsibility term is specifically about ONE stream's OWN cell being blamed
  for ITS OWN span CE — a Frankenstein-of-rollouts write the mixture pass does not build. Folding
  them would change what the term measures, not just its cost.
- **Task 4 (per-call ternary/embed weight re-cast caching via `torch.nn.utils.parametrize.cached()`
  wrapping the whole train step): investigated, NOT shipped.** Measured 103
  `TernarySTE.forward` calls across 24 parametrized modules in one forward+backward of this arm
  (histogram `{1: 13, 6: 3, 8: 4, 10: 4}` — coda/core MLP modules get re-quantized 6-10 times per
  step). `torch.nn.utils.parametrize.cached()` is PyTorch's own built-in fix for exactly this
  ("using a parametrized parameter more than once in the forward pass"); on a tiny model
  (ternary QAT `scope=backbone, scale_mode=norm_match` — the shipped rule per this repo's
  `CLAUDE.md` — combined with `embed_quant=int8`, WTA "map" arm, dropout 0.1, real gradient
  checkpointing confirmed active via a `torch.utils.checkpoint.checkpoint` call counter, 9
  calls/step), wrapping BOTH the forward call and `loss.backward()` in one `with
  parametrize.cached():` block gave a bit-identical loss and 0/181 parameter gradients
  mismatched, while cutting `TernarySTE.forward` calls 103 → 24 (each module computed exactly
  once, correctly reused across every rollout AND every checkpoint recompute — autograd's
  ordinary multi-use gradient accumulation makes this safe by construction: quantization is a
  deterministic, RNG-free function of the constant-within-a-step shadow weight, and checkpoint
  recompute only needs the SAME cached tensor object to already be part of the graph, not a
  fresh forward). **Not shipped because `base.yaml` runs `training.compile: true` and
  `model.use_kernels: true` in production, and this proof covers only eager, uncompiled
  execution — `parametrize.cached()`'s interaction with `torch.compile`'s tracing and with the
  CUDA-graph capture path (`build_static_graphs`) is unverified.** This is exactly the task's
  own sanctioned fallback ("report the count and design and do not ship it") rather than a
  smaller, narrower ship: modifying `train.py`'s core step function is a materially larger blast
  radius than anything else in this note, and an unverified interaction there is a live-run
  risk, not a test-suite risk. `attn_proj_quant.py`'s `IntNLinearSTE` uses the identical
  `register_parametrization` mechanism (architecturally the same call pattern, not separately
  measured). TTQ/dual ternary scale modes (training-only, not the shipped `norm_match` rule)
  were not tested with the cache either.

## GPU evidence (steps 8-12, `torch.profiler`, `tul_slot_spandec_strict_lxfan4_wta_fp01_ev01`,
5090, under `flock`; `training.steps=13`, same window the orchestrator used)

| arm | total GPU/step | forward | backward+opt | tok/s (derived, see note) |
|---|---|---|---|---|
| baseline (pre-this-session trace, `lxfan_ev01.json`) | 2.278s | 0.948s | 1.330s | ~2704 |
| +1 fused CE only (`per_rollout`, current tree) | 2.299s | 0.981s | 1.318s | ~2678 |
| +1+2 (`fan_all_wta_winner=map`) | 1.923s | 0.607s | 1.315s | ~3202 |
| +1+2+3 (`+ fan_all_wta_grad_rollouts=map`) | 1.553s | 0.514s | 1.039s | ~3966 |

tok/s is derived, not separately logged per-step: the trainer's own step-0 line gives
`tok/s / sps = tokens/step` (≈6100-6200, consistent within ~2% across all 3 runs — this is the
fixed `batch_size*seq_len`), divided by the profiled window's per-step total GPU-busy time
(steps 8-12) as a wall-clock proxy. Step 0's OWN reported tok/s is noticeably lower
(1793-2732) than this windowed figure because step 0 carries one-time warm-up residue the
profiled steps 8-12 do not; the windowed figure is the steady-state estimate.

**Task 1 (fused CE) alone: not a measured win on this arm.** `aten::_log_softmax <
aten::contiguous < aten::clone` (480 calls, 0.158s of 1.402s total `copy_` time in the baseline
trace) is GONE from the fused trace's top `copy_` contributors — the fused kernel's named target
is confirmed eliminated. But total per-step GPU time did not improve (2.278s → 2.299s, flat to
very slightly worse, within run-to-run noise) because `copy_` was never the dominant cost
(16.9% of kernel time, behind `gemm`'s 29.8%) and the fused kernel's own internal chunked
`aten::to`/`_to_copy` casts (dtype casts inside `_FusedLinearLabelLogProb`) replace a similar
amount of overhead elsewhere. Task 1's real payoff is architectural — no `[L, V]` logits
materialized, a documented fp tolerance instead of an undocumented "should be equivalent" — not
wall clock in isolation.

**Tasks 2 and 3 are where the wall-clock win is.** Forward time drops 0.981s → 0.607s (Task 2,
38%) → 0.514s (Task 3, a further 15%; 48% total from +1). `_tul_fan_all`'s own internal coda call
count goes 5 → 6 → 6 (one MORE call under "map" than "per_rollout" — the shared mixture pass is
new), but the win is that `_forward_tul`'s SEPARATE downstream deployed pass (previously an
uncounted 7th, full `K*B0`-row call) is skipped entirely via the cache. The per-step "first
coda calls" timing sample makes the row-count claim directly visible in the trace, not just in
CPU call-count spy tests: `map` shows `[76.9, 18.0, 18.0, 18.2, 18.3, 77.5]` ms — one `K*B0`-row
pass (~77-79ms, matching the baseline's uniform ~78ms calls), four `B0`-row picks at ~18ms
(a clean ~4.3x size ratio, matching `K=4`), and one more `K*B0`-row call (the "all" grad pass).
`mapgrad` shows `[78.5, 18.3, 18.4, 18.2, 18.2, 18.3]` — that last `K*B0`-row call has ALSO
dropped to ~18ms, exactly the predicted signature of Task 3 shrinking the grad pass to `B0`
rows. Backward+optimizer time is flat across baseline/+1/+1+2 (1.330s/1.318s/1.315s — Tasks 1-2
are forward-only changes) and drops 1.315s → 1.039s (21%) only under Task 3, because the
grad-map pass is both `4x` fewer rows AND no longer checkpointed (no backward-time recompute for
that one pass) — both effects land in backward time specifically, which fanall.py's
`total − forward` bucket correctly attributes there since backward runs after the whole model
forward call (and its `R::_forward_tul` annotation) has already returned.

**Net: 2.278s → 1.553s per step, a 1.47x reduction in GPU time / 1.47x tok/s (steady-state,
same arm, same window, same hardware).** Task 4 not included (not shipped).

## Acceptance criteria

- `tests/test_tul_wta_fused_ce.py` (new, 4 tests), `tests/test_tul_wta_shared_winner.py`
  (rewritten for the shared-mixture-pass design, 18 tests including 3 new refusal tests and 2
  new Hydra-reach / real-dropout tests) — all pass.
- Full `-k tul` suite: 1843 passed, 1 skipped, 1 xfailed, 0 failed
  (`pytest tests/ -k tul`, this tree).
- Sabotage-checked (each caught by a named test, then restored): flipped `argmax`→`argmin` on
  the MAP pick; disabled the cache-reuse consumption in `_forward_tul`; disabled the posterior
  capture (`s_out["S"] =`); disabled `bypass_rollout_sharing` around the grad-map pass.
- 3 fresh GPU traces (`per_rollout`, `map`, `map`+`grad_rollouts=map`), steps 8-12, same config
  and window as the orchestrator's original baseline trace — all 3 runs completed cleanly
  (checked for `error`/`Traceback` in the run logs; none found), numbers reported above.

## Risks

- The fused-CE tolerance (`rtol=1e-4, atol=1e-5` value; `rtol=1e-3, atol=1e-4` gradient) is
  measured on ONE tiny fixture (`V=37, C=16`, ~2 rows). It has not been re-measured at the real
  arm's vocab size (49152+) or at bf16/autocast precision — the pin-test drift already observed
  (loss/logit-sum bit-identical, grad-sum moved ~2.6e-9 relative) is the only production-scale
  evidence available, and it is reassuringly small, but the tolerance itself was chosen on the
  toy fixture, not derived from the production-scale number.
- The `+1+2+3` combination (`fan_all_wta_grad_rollouts=map`) changes what the responsibility
  term measures (one rollout instead of the mean over `K`) — this is a real objective change,
  not merely a speed optimization, and has NOT been evaluated for training-quality effects (CE
  at matched steps, `fan/oracle_ce` vs `fan/single_ce` gap, or the WTA winner-agreement
  measurement from the prior task) — only correctness (finite loss/grads, bit-identical-where-
  claimed) and speed have been measured this session.
- Task 4's evidence (103→24 call reduction, bit-identical on the tiny model) is real and
  reproducible, but explicitly does not cover the production `torch.compile=true` /
  `use_kernels=true` / CUDA-graph-capture path — do not read this note as "ready to ship
  pending only a config flip."

## What was NOT verified this session

- No end-to-end training-quality comparison (CE at matched steps, or the WTA agreement
  measurement) for `fan_all_wta_grad_rollouts=map` vs `=all` — only forward-pass correctness and
  GPU speed.
- No `torch.compile` or CUDA-graph interaction test for Task 4's caching idea (the reason it was
  not shipped).
- `attn_proj_quant`'s `IntNLinearSTE` and TTQ/dual ternary scale modes were not separately
  tested with `parametrize.cached()` (architecturally identical to what WAS tested, not run).
- The GPU profiling window is 4 steps (8-12) on one seed, one arm — not a statistically
  controlled multi-seed measurement; the reported per-step numbers should be read as a single
  representative sample, consistent with (and, for the baseline row, identical to) the
  orchestrator's own original measurement.
