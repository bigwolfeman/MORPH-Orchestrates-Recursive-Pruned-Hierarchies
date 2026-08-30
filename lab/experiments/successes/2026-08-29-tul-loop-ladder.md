# Planned: the loop ladder — L1 full-BPTT / L2 projected / L3 DB-shaped / L4 AdamW

Status: success
Date: 2026-08-29 (frozen before smoke; smoke and launch deferred until the GPU
is free). Configs: `tul_l1.yaml`, `tul_l2.yaml`, `tul_l3.yaml`, and L4 =
`tul_l1` + `training.optimizer=adamw`. Decision note:
`.agents/notes/proposed/architecture/2026-08-29-loop-ladder.md`.

## Questions

- **Q-loop.** Does restoring the core loop over the slot write (full BPTT,
  nothing protecting it) destroy the GL mechanism — and if it does, do the
  probes show instability signatures (σ_max runaway, compounding core_gain) or
  quiet homogenization (slot_pairwise_cos climbing at healthy norms)?
- **Q-control.** If L1 shows instability, does the hard σ ≤ 1.5 projection
  (L2) restore a working write at the same architecture?
- **Q-local.** Does the DB-shaped loop (L3: no gradient across iterations,
  per-iteration local mux) keep the write load-bearing with a looped forward?
- **Q-optim (Wolfe's).** Is the instability AdEMAMix-at-batch-6, not the loop?
  L4 differs from L1 only in `training.optimizer=adamw`.

## Reference numbers (fixed)

gl1b-s1 4.4047 (worth 0.05–0.096) | gl1b-s2 4.3714 (worth ≤ 0.028) |
gl1b-nomask 4.3102 | gl1-ctrl 4.6656. All `val/ce_tokens` @4250, batch 6,
4500 steps. Note the reliance metric's seed fragility (R1 of the line-2
filing): single-arm worth numbers are read as bands, not points.

## Predictions (frozen)

- **P1 (L1).** Completes without detonation (S1 holds, no NaN): 60%. Sustains
  worth_shuffle ≥ 0.04 (≥3 consecutive evals incl. one of the last three): 25%
  — my prior says the write homogenizes (slot_pairwise_cos > 0.4 by step 4500)
  even when training is numerically stable. CE @4250 ≤ 4.45: 40%.
- **P2 (L2).** Conditional on L1 showing instability signatures (σ_max > 3.0
  at any log, or per-iteration core_gain compounding with S1 violation): L2
  removes the signature AND sustains worth ≥ 0.04: 60%. If L1 is clean, L2
  lands within noise of L1 (projection inert at healthy σ): 80%.
- **P3 (L3).** Completes clean: 90%. worth ≥ 0.03 sustained: 55%. CE @4250 ≤
  4.46 (within 0.05 of gl1b-s1): 50%.
- **P4 (L4, Wolfe's hypothesis).** Decision rule, binding: if L1 violates S1
  (or aborts) with instability signatures AND L4 passes S1 with worth ≥ 0.04
  ⇒ optimizer-causal — AdEMAMix-at-batch-6 is implicated and the fatal-loop
  framing is dead. If L1 and L4 fail the same way ⇒ loop-side. If both pass
  S1 ⇒ no instability at this scale; the ladder is scored on worth/CE alone.
  My probability that L4 is strictly more stable than L1 (lower max σ_max,
  lower max core_gain, no S1 violation): 55%. Wolfe's stated position: most of
  the instability is the optimizer at this batch size.
- **S1 (stability, all arms).** No run sits > 0.20 nats above its running min
  for 2+ consecutive evals after step 1000.
- **F (the falsifier, binding).** ANY full-BPTT arm (L1, L2, L4) sustaining
  worth ≥ 0.04 at CE ≤ 4.50 falsifies "gradient through an iterated write is
  fatal", and the campaign docs that state it get corrected in the filing
  change.

## Method

Sequential, one trainer on the 5090, ~30-35 min each (the loop adds ~22M core
params and per-iteration compute on the compact slot sequence):

1. smoke: 30 steps of L1 and L3 (the two new code paths; L2/L4 are config-only
   deltas on L1), gating VRAM and NaNs. Smoke AFTER this file is committed.
2. `tul_l1` / `tul_l2` / `tul_l3` / `tul_l1 + training.optimizer=adamw
   wandb.name=tul-l1-adamw`, each with the panel flags: steps 4500, batch 6,
   seed 1, ademamix_alpha_cap 3.5, ademamix_t_beta3 3500 (inert under adamw),
   use_kernels false, eval_every 250, gen_every 0, ckpt_every 500,
   grad_probe_every 1 (per-iteration loop probe), tul.eval_ablations true.
   σ_max logs every 100 steps by default (spectral_penalty_log_every).
3. Prune each run's checkpoints to step_4500 after its verdict; artifacts to
   `lab/experiments/results/2026-08-29-tul-loop-ladder/`.

**Method amendment (2026-08-29 13:50, before any arm completed).** First launch
aborted: tul-l1 OOM'd at step-0 eval inside `tul_mux_grad_share` — the one
grad-enabled eval instrument, which runs with eval-mode checkpointing OFF and
so retained the full-BPTT loop's activations at batch 6. Fix (commit 28549d0):
the probe slices to 2 rows. The smoke gate is amended to steps=12/eval_every=5
so the eval path is exercised (the no-eval smoke waved the bug through).
Predictions are UNTOUCHED. tul-l1's 3-minute aborted run (yzmy9jli) is
discarded as a harness failure, not evidence about any hypothesis.

## Not verified before launch

The n_core>0 + tg_restrict + slot_seed=boundary + mux composition has never
run on GPU (CPU tests only: 683 passed, 2 xfailed at commit time). The db_loop
memory profile (db_mux_iters=4 × [B,S,V] fp32 logit graphs ≈ +0.3 GB) is
arithmetic, not measurement — the smoke gates it.

**Method amendment (2026-08-29 15:45, harness failure — L2 cap never armed):** the first
`tul-l2` run (wandb name `tul-l2`) trained with the spectral projection OFF: `tul_l2.yaml`
put `spectral_project_cap` under `model:` while `train.py` reads
`cfg.training.spectral_project_cap` (base.yaml defines it under `training:`). Hydra merged
the stray key silently and the run log printed `Core spectral-norm penalty OFF ... cap=0.0`;
σ_max free-climbed past 4.7. `tests/test_tul_loop_ladder.py` asserted the same wrong path,
so it validated the mistake instead of catching it. Both fixed in this change (yaml key moved
to `training:`, test asserts `("training","spectral_project_cap")`, 7/7 pass on CPU;
compose check prints `training.spectral_project_cap = 1.5`). The mis-configured run is
RECLASSIFIED as an L1 same-seed replicate (config-identical to tul-l1) and will be used only
to calibrate run-to-run spread; it does not score as L2. The corrected L2 arm runs after
tul-l1-adamw under wandb name `tul-l2-cap`, gated on its startup log printing
`Core spectral PROJECTION ON: cap=1.5`. Predictions untouched.

---

## Results (filed 2026-08-30; artifacts in `../results/2026-08-29-tul-loop-ladder/`)

All five arms (incl. the corrected `tul-l2-cap`) completed exit 0. Val CE @4250 /
final, worth_shuffle final, σ_max end:

| arm | CE @4250 | CE final | worth final | σ_max | S1 |
|---|---|---|---|---|---|
| tul-l2-cap | **4.3489** | 4.3844 | **0.146** (monotone from 0.050@2250) | 1.50 (pinned) | PASS |
| tul-l3 | 4.3519 | 4.4005 | 0.020 | 3.99 | PASS |
| tul-l1-rep | 4.4388 | 4.4843 | 0.051 (bouncy) | 4.89 | PASS |
| tul-l1 | 4.4749 | 4.5142 | 0.029 | 4.08 | PASS |
| tul-l1-adamw | 4.5278 | 4.5920 | 0.027 | 1.92 | PASS |

Coreless nomask ruler 4.3102. Same-config replicate spread (l1 vs l1-rep):
0.030–0.036 nats CE, 0.022 worth — the 0.04 worth bar sits inside run noise,
which is why the sustained definition and the stratified profile carry the verdicts.

- **S1**: all arms PASS (max consecutive evals >0.20 over running min = 0 everywhere).
  Nothing detonated. The "loop detonates" scenario did not occur at this scale.
- **P1 (L1)**: completed clean (60% held). Sustained worth FAILED — 0/14 post-1000
  evals ≥0.04 (25% prior: consistent). CE 4.4749 > 4.45 (40%: consistent). My stated
  mechanism was WRONG: no homogenization (slot_pairwise_cos 0.137, prior said >0.4).
- **P2 (L2)**: L1 DID show the frozen instability signature (σ_max 4.08 > 3.0), so the
  ACTIVE branch scored: "removes the signature AND sustains worth ≥0.04: 60%" —
  **HELD**. σ pinned at 1.50; worth_shuffle ≥0.04 for 9 consecutive evals incl. all of
  the last three, still climbing at 4500.
- **P3 (L3)**: clean ✓ (90%). CE 4.3519 ≤ 4.46 ✓ (50%). worth ≥0.03 sustained ✗
  (55% — miss; max 0.022 post-1000).
- **P4 (binding)**: both L1 and L4 pass S1 ⇒ "no instability at this scale; score on
  worth/CE." L4 strictly more stable on σ (1.92 vs 4.08; 55% held) but 0.05–0.09 nats
  WORSE CE and no worth. AdEMAMix-at-batch-6 was not the cliff — but see the l2cap
  nuance below: σ growth (an AdEMAMix side effect) was still the write's disease.
- **F (binding falsifier): FIRED.** tul-l2-cap is a full-BPTT arm sustaining worth
  ≥0.04 at CE ≤4.50. "Gradient through an iterated write is fatal" is falsified;
  corrections applied in this change (gist-loop note, block-backward-gain note,
  loop-ladder decision note moved to implemented/).

**Post-hoc instruments (built during the run, both committed):**

- `lab/divergence/worth_profile.py` (96 rows, bootstrap CI): the scalar worth metric
  conflated two mechanisms. Shuffle (plan CONTENT) profiles by offset-in-span:
  gl1bc/gl1b-s1 are first-token spikes (+1.08/+0.90 at offset 0, dead by offset 1);
  **l2cap is a span-wide carrier (+0.22 at offset 0 and still +0.11 at offset 16+)**
  — integrated ~3 nats/span vs the spike arms' ~1.2. l1/l3/l1adamw/nomask: no
  content reliance. Zero-ablation adds an anchor effect present in every arm.
- `lab/divergence/core_depth_sweep.py` (48 rows, paired): **l2cap is the only arm
  whose loop earns CE**: depth 1→6 = 4.6220→4.3892 (0.233 nats, saturating at the
  trained mean depth). l1 (4.519), l3 (4.392), l1adamw (4.584) are depth-FLAT to the
  third decimal. l3's good CE is one core pass + slot anchors; its per-iteration
  activation motion (delta_ratio ~0.47) lives in the readout's null space.
- Generation (LADDER_SAMPLES.md): l2cap uniquely resists greedy degeneration
  (rep4 0.61 / distinct3 0.34 vs ~0.87 / ~0.10 all others); sampled modes on the
  real-text anchor. No DB pathologies in l3.

## Verdict

Success. The ladder answered every question it was built to answer, and the binding
falsifier fired exactly as pre-registered. Wolfe's framing wins: instability
(uncontrolled σ growth under AdEMAMix) was the disease; looping was innocent — but
the surprise is bigger than the rescue: the σ≤1.5 projection did not merely stabilize
full BPTT, it produced the campaign's first LOAD-BEARING loop (0.233 nats of depth-
earned CE) and its first span-wide plan carrier, with the best generation health of
any TUL arm to date.

## Updated hypothesis

Contractivity control is not a safety rail; it is what makes the iterated write
trainable at all. Under full BPTT the uncapped core spends its capacity inflating
σ (the optimizer is blind to ρ(J_core) — the nested-dynamical-system note); capped,
the same gradient budget is forced into using the iterations. Next: (1) L3-WAKE
(prereg 2026-08-30) tests whether DB-init + standard training can grow a depth
curve post-hoc; (2) if wake fails, the recipe is "cap from step 0"; (3) longer-run
l2cap — worth was still climbing at 4500.
