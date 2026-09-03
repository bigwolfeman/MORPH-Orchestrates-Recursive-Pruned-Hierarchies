# Agent Note: Ship the paid loop as the TUL recipe and cut the slot-only arms

Status: implemented

## Problem

By 2026-09-03 the TUL tree carried two forwards and a dozen arms around them. The
slot-only core (gather the slot states → loop on them with a per-slot masked Poisson
depth → scatter back; arms A0/A1/A1r/A3/A4/A5, the span-length gate and halt, the TG
restriction stack, the gist loop and mux targets, the compaction window, the DB
interleave, per-slot embeddings, the fixed-stride rule) had been measured to exhaustion:
its loop never earned depth (K1−K6 ≤ 0.011 nats at any horizon,
[docs/tul-paid-loop-recipe.md](../../../../docs/tul-paid-loop-recipe.md) §2), its
takeover was never cured (15 hypotheses, 11 refuted,
[lab/divergence/takeover-campaign.md](../../../../lab/divergence/takeover-campaign.md)),
and every one of its arms lost to the plain model or to A2. The paid loop — the ordinary
per-sample core run over the whole packed row, slots included — is the one arm whose loop
earns depth (0.168 nats at 5k, 0.100 at 20k). It sat behind a config flag while
`transformer.py` was 3715 lines, 59 arm YAMLs shared `morph/configs/`, and every new
change had to be proven bit-identical against paths nobody would run again.

The 20k matched pair on the 1000-step ramp
([lab/experiments/failures/2026-09-02-warmup-20k-pair.md](../../../../lab/experiments/failures/2026-09-02-warmup-20k-pair.md))
put the paid arm 0.022 nats behind the plain model on 480 identical rows at 1.33x the wall
clock, with its matched-step deficit closing 0.132 → 0.012 from 5k to 20k. The config
decision that followed was "ship the ramp, keep TUL default-off". Wolfe overrode it the
same day: TUL is only slower because of optimizations still to do; call it the winning
recipe, ship it to master, and clean up the arms that did not make the cut.

## Decision

1. **`base.yaml` is the paid-loop recipe.** `tul.activate_at: ${training.tst_ratio}` (TUL
   switches on when TST ends), seq 4096, TST on, prune/carve/route on, batch 4, the
   1000-step ramp, retention off, `bptt_depth: 8` (full BPTT), spectral cap 0, `slot_seed:
   boundary`, `emit_weight: 0.0`, `plast_weight: 1.0`, token-state dropout 0.15,
   `prefix_k: 2`. That conjunction has run only as a startup smoke; nothing about it is
   measured. The measured recipe is `tul_a2.yaml` (seq 1024, batch 6, 20k steps, TUL from
   step 0, TST/prune/route off) against `notul.yaml`, its matched control.
2. **The forward with a layout IS the paid loop.** `_forward_tul` = `_tul_front` →
   `_core_region` (the same per-sample core the plain path runs) → token-state dropout →
   `_back_region` → `_tul_group_losses`. Nothing is gathered, projected or scattered.
   `W_prefix` and `prefix_project` survive only on the FM planner (`cfg.fm`, `n_core 0`),
   which is the one remaining consumer; `TULSlots(d, tul, with_prefix=cfg.fm is not None)`.
3. **Cut, not flagged.** Removed: `_tul_core` and the slot-path helpers
   (`compact_index`, `window_drop_mask`, `cw2_retain_mask`, `mux_span_targets`,
   `gather_valid`'s consumers other than the planner), the gate (`TULGateConfig`,
   `TulGateSpec`, `insert_truncations`, `span_len` / `len_supervised` layout fields,
   `gate_audit.py`, the gate/halt generator paths, `tul_gate` no-decay group), the TG stack
   (`_tg_slot_attention`, `tg_allow_mask` / `tg_reset_mask`, `tg_scoped_kernels`,
   `recur_gate.py`, `iter_cond.py`, `attn_lift.py`, GLA `reset_mask`), `step_mix` /
   `tul_step_mode` / the DB interleave, `per_slot_embed`, `center_bag_mean`, the `e_slot`
   seed mode, `fixed_stride`, `_probe_loop` / `_loop_probe`, `tul_forward_with_plan_nats`,
   `tul_forward_cw_arms`, `tul_slot_state_probe`, and every arm key the spec listed as
   RAISE-if-set. 59 config YAMLs, 17 test files. `transformer.py` 3715 → ~2290 lines.
   `TULConfig` has six fields; `build_tul_runtime` rejects any other `tul:` key before it
   touches the tokenizer.
4. **Checkpoint compatibility is one loud exception.** Every A2 checkpoint from before this
   change carries `tul.W_prefix`. `load_checkpoint` raises on a homeless key by design, so
   `drop_retired_tul_keys` removes exactly that key, prints what it dropped, and only for a
   model without an FM planner (`tests/test_checkpoint_compat.py`). Nothing else is
   forgiven.
5. **Records stay, code goes.** `docs/tul-spec.md` §3.3, `docs/tul-gate-spec.md`,
   `docs/tul-tg-spec.md`, `docs/gist-mux-recipe.md` and the TUL table of
   `docs/ablation-ledger.md` carry a RETIRED banner naming `d9e04e6`, the last commit that
   runs the arms. The lab probes that import deleted code (`mask_surgery.py`,
   `mux_unigram_baseline.py`) are listed as retired in
   [lab/divergence/DIVERGENCE-README.md](../../../../lab/divergence/DIVERGENCE-README.md)
   and are not to be repaired against the paid loop. Proposed notes whose subject was the
   slot path moved to `rejected/` with the reason on their `Status:` line
   ([gated TUL](../../rejected/feature/2026-08-21-gated-tul.md),
   [gist loop](../../rejected/architecture/2026-08-29-gist-loop.md),
   [dead HCA branch](../../rejected/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md)).
6. **GLA stays in the tree, off by default** (Wolfe's call, same day): three draws under
   the ramp were inert
   ([failures/2026-09-02-a2-gla-under-warmup.md](../../../../lab/experiments/failures/2026-09-02-a2-gla-under-warmup.md)),
   so the retention branch is a measured non-lever, not dead code.

## Alternatives considered

- **Keep the slot-only core behind `tokens_through_core: false` and only delete the arm
  YAMLs.** Rejected: the flag is exactly the runtime branching the design principles
  forbid, every future change would have to stay bit-identical against a forward nobody
  runs, and the takeover campaign's instruments were built against that path — keeping it
  invites re-running refuted hypotheses. The record at `d9e04e6` serves the same purpose
  at zero maintenance.
- **Cut the FM planner and `lab/tulfm` too.** Offered; Wolfe kept them. The planner is a
  separate arc with its own prereg and it is the only consumer of `W_prefix`, so it is the
  reason that parameter still exists at all (`with_prefix`).
- **Delete the lab probes that no longer import.** Rejected: they are how the campaign's
  numbers were produced; a probe that fails at import is an honest record, a probe rewritten
  against a different forward is a fake one. Listed as retired instead.
- **Ship `tul_a2`'s measured settings as `base.yaml` (seq 1024, TST/prune/route off).**
  Wolfe chose the production conjunction (seq 4096, TST on, prune/carve/route on, batch 4)
  knowing it is unmeasured; this note says so where the next reader will look.
- **Forgive every unexpected checkpoint key with `strict=False`.** Rejected: that is the
  silent-partial-load the loader was hardened against (`load_checkpoint`'s docstring). One
  named key, one print.
- **Keep the gate's no-decay entry and the arm mentions in comments "for history".**
  Rejected: a name in the optimizer's no-decay list that matches no parameter is a trap for
  the next parameter that happens to contain it.

## Consequences

- `pytest tests/` on CPU: 531 passed, 8 skipped, 1 xfailed at the cut (was 526 + the new
  compat and key-guard tests). 17 test files deleted with the code they tested;
  `test_tul_forward.py` and `test_slot_seed.py` rewritten for the paid loop, with the
  per-invariant map in `lab/runtime-invariants.md` §6b.
- GPU smokes on the 5090 (`expandable_segments`, offline wandb), all at commit time:
  `tul_a2` 30 steps exit 0 (`val/layer_passes_per_token` 50.7 = 1152/1024 × 45, the
  paid-loop accounting); `notul` 30 steps exit 0; `tul_smoke` exit 0 with generation; the
  FULL `base.yaml` conjunction at `data.seq_len=1024 training.batch_size=6` with the
  schedule compressed to 80 steps (TST → TUL at step 24 → prune every 5 → carve at 50 →
  ReMoE at 60 → eval) exit 0, peak 12.5 GB. At the shipped `seq_len 4096, batch 4` the
  same smoke ran out of memory — with TUL at activation (step 24, 5120 packed positions;
  `tul.max_slots=256` too), and WITHOUT TUL at the routed phase after step 60 (peak
  22.35 GB plain before that) — on a card carrying ~7 GB of desktop and, per Wolfe, a
  VRAM leak at 17 days of uptime. Batch 2 at seq 4096 with TUL on ran the whole
  conjunction at 15.4 GB peak. So the 5090 numbers for the shipped geometry are not
  established here; Wolfe's guidance is to test at seq 1024 × batch 6.
- Throughput, measured in that seq-1024 smoke: 49.6k tok/s (42.9 TFLOPS) in the TST
  phase, 7.2k tok/s (10.4 TFLOPS) once TUL is on. That is the "TUL is only slower
  because of optimizations we have not done" item, now with a number attached.
- The old-checkpoint path is real, not a unit test: `lab/divergence/a2_depth_sweep.py`
  on `checkpoints/morph/tul-a2-20k-wu/step_7500.pt` loaded through `load_checkpoint`,
  printed the one dropped key, and scored K1 3.6028 / K6 3.5271 on 6 rows.
- The Jacobian probe (`morph/training/core_jacobian.py`) captures the FULL packed row under
  TUL; `_core_region` takes an optional `jac_active` mask so the packer's tail-pad slot
  positions stay out of the probe's active set.
- The eager generator reads a span's first token from the boundary TOKEN position by
  default (`emit_source="token"`), because `emit_weight: 0.0` never trains the slot's own
  head.
- What is NOT verified here: any training run of the `base.yaml` conjunction past a smoke;
  the 40k continuation (`lab/experiments/planned/2026-09-03-warmup-pair-continue-40k.md`,
  not launched); the JAX mirror, which never had TUL.
