# Experiment: does span-level soft-min credit make the plan load-bearing?

Status: **planned, NOT run. One design switch still open (the MUX local head — Wolfe
is reading the paper).** Predictions are frozen against measured baselines; do not
edit them after the run starts.

## Question

The campaign pivoted: the takeover is a credit-assignment failure, not a stability
failure ([ledger](../../divergence/takeover-campaign.md)). The measured chain:

- The core's Shapley value on `ce_main` is **0.0007 nats** at step 1750
  ([results](../results/2026-08-25-region-shapley/README.md)); removing the whole
  plan costs **0.0191 nats**.
- The plan's only direct supervision is `ce_emit` — predict ONE token — and it
  LOSES that race to the free token path (`cf` negative in every run on disk).
- The coda's readout of z is suppressed ~3–5x per relative unit but NOT dead
  ([readout Jacobian](../results/2026-08-25-readout-jacobian/README.md)):
  raw slot/token gradient ratio 0.063 → 0.042 from step 1650 to 1850.

First principles (XM, arXiv 2607.27372 §F.2): next-token with full-prefix
conditioning is near-unimodal, so the token path wins it structurally. A
deterministic plan trained on that target has a blur for a minimizer. The plan's
only winnable job is the multimodal part: WHICH span comes next.

Can we make the plan earn value by (a) retiring the unwinnable `ce_emit` race and
(b) giving it K candidate latents scored at SPAN level with annealed soft-min
credit, so distinct candidates commit to distinct futures?

## Hypothesis

With span-level soft-min credit over K candidate latents, the plan becomes
load-bearing: its ablation cost rises well above the 0.0191-nat baseline, the
coda's readout ratio rises instead of falling, and the takeover does not fire —
because the 90%-norm/zero-value gradient war was fueled by the `ce_emit` race,
which no longer exists.

## Arm design (v1)

Control: `tul_a1`, short schedule (3500 steps, batch 6, seed 1), untouched.

Arm `tul_xk` (one config, no runtime flags — construction-time switches per the
no-flags rule):

1. **Retire `ce_emit` as supervision.** `_tul_half_weights` emit weight → 0.0,
   plast weight → 1.0. `ce_emit` stays computed as a METRIC only.
2. **K = 4 candidate latents per sequence.** K learned embeddings `E_cand[k]`
   (d_model each) added to every slot input. v1 shares ONE candidate index per
   sequence (per-slot mixing is combinatorial across a shared coda — v2 territory).
   Core runs per candidate on slot positions (cheap); the coda runs per candidate
   with K folded into the batch dimension (the real cost, ~K× coda FLOPs; batch
   drops 6 → 4 if memory requires — record which).
3. **Soft-min span loss.** Per candidate k: `L_k` = mean token CE over all span
   tokens (`ce_main` shape). Combined:
   `L = −τ · log( (1/K) Σ_k exp(−L_k / τ) )` — the XM smooth objective, exact
   mixture NLL at τ=1, hard min as τ→0. Anneal τ 1.0 → 0.1, sigmoidal, midpoint
   at step 1750. All schedule constants in the Hydra config, logged to wandb.
4. **OPEN SWITCH — MUX local head** (decide before run, then amend Method with
   date): auxiliary KL( mux_geo(next span) ‖ softmax(W_unembed · z / τ_h) ),
   geometric ρ = 0.9 (MUX, arXiv 2607.18264, Prop 5i), weight β = 0.1. Gives z
   direct span-content gradient that does not route through the coda readout.
   Token path untouched — this is not a sequential single-vector decode, so the
   Huginn ban does not apply.

The TUL gate (Quiet-STaR mixing head, prior testing in another dir) is
deliberately OUT of v1 — one lever family per arm.

## Predictions (frozen 2026-08-25, before any run)

Baselines they are measured against: core Shapley 0.0007 / plan-off 0.0191 nats at
ROLL_step_1750; readout raw ratio 0.055 (1750), falling trend; A1 aborts at
1800–2940 depending on seed; control val CE at matched steps from the A1 short run.

- **P1 (survival):** `tul_xk` reaches step 3500 with no takeover abort and core
  pre-clip gradient share median < 0.5.
- **P2 (value):** plan-off ablation cost (`slot_path_worth.py --plan-off`) at the
  step-3000 checkpoint ≥ **0.04 nats** (>2× baseline).
- **P3 (readout):** raw slot/token Jacobian ratio (`readout_jacobian.py`) at step
  3000 is HIGHER than the arm's own step-1650 value — the falling trend reverses.
- **P4 (no tax):** val CE at step 3000 within **+0.05 nats** of the A1 control at
  the same step (control from the same seed, same schedule).
- **P5 (diversity, refuter):** after the τ-anneal midpoint, median per-step spread
  `max_k L_k − min_k L_k` > **0.02 nats**. If ~0, the candidates collapsed and the
  soft-min degenerated to mode-averaging — the arm is refuted regardless of P1–P4.

Failure reading: P1 holds but P2/P3 fail → the war was `ce_emit` all along and the
plan is decorative; file under failures and test the MUX head alone next.

## Method notes and known traps

- **PIN `training.ademamix_t_beta3=5000`** in the arm script. `null` falls back to
  `training.steps` and silently reshapes the optimizer
  ([trap](../failures/2026-08-25-scse-arm-c-long.md)). Check the
  `[opt] re-applied config hyperparameters` line in the log.
- n=1 seed comparisons are unreadable (memory: 6.5% median spread) — P4 uses the
  matched-step control, and P2/P3/P5 are within-run measurements by design.
- Sequential runs only on the 5090 (UPS). wandb with the full config dict.
