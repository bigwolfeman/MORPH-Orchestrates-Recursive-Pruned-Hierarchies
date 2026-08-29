# Planned: FM1 — the flow-matching planner inside the live model

Status: planned
Date: 2026-08-28
Config: `morph/configs/tul_fm1.yaml` at commit `74d554d` (+ this file's overrides)
Prior work: P1/P1b/P1c (filed failures, same dir tree) proved a frozen-backbone CFM
planner reaches ~2.5x chance retrieval but never beats the 0-param copy baseline
(0.0678 within-row top-1). Wolfe's call: train the full model with it anyway.

## Question

When the backbone co-trains with the planner (targets live, SIGReg shaping them,
coda attending detached plans), does the plan become worth actual nats — something
frozen features could not deliver?

## Hypothesis

Co-training helps the *targets* (SIGReg will raise their effective rank) but the
coda will still mostly ignore the plans at this scale: the frozen-backbone
information cap was the binding constraint, and 4500 steps of joint training on a
207M backbone does not remove it. I expect a healthy, stable, fast run whose plan
worth stays under the pay threshold.

## Predictions (frozen — not edited after launch)

- **F1 stability.** The run completes 4500 steps. No takeover: `val/ce_tokens`
  never rises more than 0.10 nats above its running minimum after step 1000.
  Predict HOLDS (no core loop, no BPTT, plans detached — the disease has no organ).
- **F2 speed.** Mean throughput ≥ 3.0 steps/s (A1 ran 1.9; A3 ~4.5; FM1 adds 7
  passes of a 22M planner to A3's skeleton). Predict 3.0–4.0.
- **F3 CE band.** `val/ce_tokens` @3000 in [4.36, 4.58] (point: 4.48);
  @4250 in [3.99, 4.45] (point: 4.20). Reference: a3-s1/s2 = 4.4226/4.3564 @3000,
  4.0250/3.9873 @4250; a1-s2 = 4.4362/4.3472; a1noaux-s2 = 4.3448/4.1653.
  Predict FM1 lands between A3 and A1: slot-prefix + dropout tax, no core-loop tax.
- **F4 plan worth.** `val/plan_worth_shuffle` < 0.01 nats at every eval ≥ 3000.
  Predict HOLDS (i.e. the plan does NOT pay). GATE (what would flip the program):
  worth_shuffle ≥ 0.01 nats sustained over ≥ 3 consecutive evals.
- **F5 copy gap.** `val/copy_gap` ≤ 0.005 at step 4500. Predict HOLDS (planner
  stays at the continuity floor). GATE: copy_gap > 0.01 sustained.
- **F6 SIGReg target health.** `val/target_eff_rank` rises from ~25 (frozen value)
  to ≥ 100 by step 4500 and `val/target_pairwise_cos` falls below 0.35.
  Predict HOLDS — this is the one number I expect co-training to move decisively.
- **F7 FM objective learns.** mean `train/fm_rel` (per-band rel to empirical null)
  < 0.6 by step 4500 (P1b reached 0.27–0.38 on frozen features in 3 minutes).

## Verdict rule

- SUCCESS = F1 & F2 & F3 hold AND (F4-gate or F5-gate fires). The arm earns seed 2
  and a longer run.
- The expected outcome — healthy run, gates silent — files to `failures/` with the
  arm parked: mechanism sound, plan still not worth nats; A3 remains the ship.
- F1 failing (takeover with no core loop) would falsify the takeover model itself
  and outranks everything else in the writeup.

## Method

Work tree `/home/wolfe/morph-perf`, python `/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python`.
One trainer on the GPU (UPS rule); verified free before launch.

```
python -m morph.training.train --config-name tul_fm1 \
  hydra.run.dir=<scratch>/fm1-s1/hy \
  training.steps=4500 training.batch_size=6 training.seed=1 \
  training.ademamix_alpha_cap=3.5 training.ademamix_t_beta3=3500 \
  model.use_kernels=false \
  training.eval_every=250 training.gen_every=0 training.ckpt_every=500 \
  training.grad_probe_every=1 training.grad_probe_path=<scratch>/fm1-s1/probe.jsonl \
  fm.source_std=0.03125 \
  wandb.name=fm1-s1
```

Flags mirror the slot-pay panel launch (`slotpay3.sh`) exactly, so the curve drops
onto the existing A3/A1 graph. Two pre-launch decisions (2026-08-28, before any run):

1. **`fm.source_std=0.03125`** (config ships 1.0). Targets are unit-L2 in 1024
   dims → per-component std 1/√1024. At source_std 1.0 the velocity target has
   E‖v‖² ≈ 1025, dominated by reconstructing the source — the exact analogue of
   the σ_data=0.5 scar that killed P1's primary run. Matched scale is the
   pre-registered choice, not a fallback this time.
2. **`training.ademamix_t_beta3=3500`** overriding the config's 4500 — panel
   parity (every comparison run used 3500).

Seed 2 runs only if seed 1 is healthy (F1 holds). The panel's external
`slot_path_worth.py` probe is NOT reused — FM1 logs plan worth in-run
(`val/plan_worth_zero`, `val/plan_worth_shuffle`, `val/copy_gap`).

## Not verified before launch

The GPU path (bf16 ladder, memory at batch 6, step time), the fused AdEMAMix
with these params, and SIGReg at M=1024 slices — the CPU build could not touch
any of these; first 100 steps of seed 1 are the smoke test.
