# Agent Note: LCTUL-J, a moving code target (EMA encoder with a variance floor) and the per-pass factor split

Status: proposed

Source paper: JEPA-Anything (Cui et al., arXiv 2609.20800, 2026-09-17; PhAI Labs, CUHK,
Fudan, CityU, Bristol, Stanford, Oxford, Princeton). Read in full from the PDF on
2026-09-22 (scratchpad copy). Its recipe: an EMA target encoder (eq. 2), per-factor L2
prediction (eq. 6), orthogonal projectors with within- and cross-factor orthogonality
(eq. 7), two variance-floor hinges (eq. 9: `L_fac` on the stopped target's factor
coordinates, `L_enc` on the online context representation), summed as eq. 10 with
`λ_orth 0.10, λ_fac 0.05, λ_enc 0.02` on every non-locomotion task (Table 10). The paper
does not state the EMA momentum `m` or the floor thresholds `γ`. It reports no text
result. Its mechanism audit (Table 5) is the one ablation: unconstrained multi-head
projectors overlap 0.455 and reach condition number 439; the orthogonal ones overlap
5e-16 at condition number 1.00005.

Parent notes: `2026-09-17-lctul-target-slot-loop.md` (the code target, arm A), the
frozen reference (`tul.code_target_ref`, spec §17.1), and the campaign synthesis
`../../implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md` (its filter:
condition A, no shallower route; condition B, consumed; the "not one-pass-easy" helper is
not a condition, Wolfe 2026-09-22; no more restriction geometries).

## Problem

Every LCTUL target so far was one of two kinds, and both are exhausted.

1. A FIXED reconstruction code. E is frozen (arm A, `tul-code-target`) or a frozen deep
   copy of the whole VAE stage computes it (`code_target_ref`, `tul-code-only-ref`). The
   code is E's verbatim code of the next span, of which the context explains 5 to 15 %
   (`lab/theory/lctul_euler_depth/`). Measured: the predictable part is a cosine of about
   0.15 whatever the front is, found in one pass
   (`lab/experiments/failures/2026-09-17-lctul-code-only-ref.md`, H0 stands). A frozen
   target cannot move toward what the context CAN predict.
2. A COMPUTED target (`tul-code-grade`): self-referential (the winner's code is 0.153
   from the truth's; the loop predicts the winner at 0.532).

The one attempt to let the target move, `tul-code-lejepa` (the flow loss into E at
weight 1.0 with SIGReg as the guard), moved it: the code stopped being a verbatim copy and
the rank held, but the coda paid +0.68 nats against the ruler and the sample stayed
unconditional (`failures/2026-09-16-tul-code-lejepa.md`). That arm had no EMA, no
predictor-side floor, and the online encoder's own input drift was the collapse route
(`frozen-encoder-on-live-front-is-not-a-fixed-target`, own cosine 0.61 → 0.99 by step
3000).

JEPA's answer to exactly this is the standard one in vision: the target encoder is an
EMA of the online network, receives no gradient, and follows the online representation
slowly, so the target drifts toward the predictable structure instead of staying a
reconstruction code, and an explicit variance floor on the online side stops the drift
from ending in a point. JEPA-Anything adds the factor split so that separate predictors
own separate parts of the target. Neither has run on the slot loop.

## Proposal

Two stages. Stage 1 is the build this note locks in. Stage 2 is written here so the
design is on record and is built only after Stage 1 reads healthy.

### Stage 1: the EMA target and the online variance floor

Mapping of the paper's objects onto LCTUL:

| JEPA-Anything | LCTUL-J |
|---|---|
| context encoder `f_θ` and predictor `q` | the live front, the strict slot loop and `TULCodeProj` (the loop IS the predictor; the descriptor `s_t` is the slot's own position, implicit) |
| target encoder `f_θ̄`, EMA, no gradient | the frozen twin of `code_target_ref` turned into an EMA twin: after every optimizer step `θ̄ ← m θ̄ + (1−m) θ` over every parameter and float buffer of the twin (the twin's E and its front pool the next span; the twin's coda serves `code_grade` only and is updated the same way, one rule) |
| `L_pred` (eq. 6) | the existing `code_target_regression` (`2 (1 − cos)` per unit-RMS cell), unchanged |
| `L_enc` (eq. 9) | NEW `tul.code_enc_var_lambda`: per cell index `m` and coordinate `j`, `σ_{m,j}` = std over the batch's valid slots of the ONLINE predicted cell (after the RMS norm), `mean_{m,j} max(0, γ − σ_{m,j})`, gradient into the loop and the projection |
| `L_fac`, `L_orth`, projectors | Stage 2 |

Knobs (every one in `KNOWN_TUL_KEYS` and the wandb manifest, printed by `tul_setup`):

- `tul.code_target_ema: float = 0.0`. `0.0` keeps the frozen twin, bit-identical to the
  tree before this change (loss, logits, parameter names, checkpoint keys). `0 < m < 1`
  makes the twin an EMA. Requires `code_target: true` and `code_target_ref: true`, else
  `TULConfig` raises. First value: `0.996` (time constant about 250 steps; I-JEPA starts
  its schedule at 0.996). Fixed for the run; no schedule in Stage 1.
- `tul.code_enc_var_lambda: float = 0.0` (`λ_enc`; paper 0.02) and
  `tul.code_enc_var_gamma: float = 1.0` (`γ`; the paper does not state it; VICReg's
  standard is 1.0 on `sqrt(Var + 1e-4)`, and a unit-RMS cell with zero mean has
  per-coordinate std 1). The term is an exact 0 with fewer than two valid slots. Requires
  `code_target: true`.

Trainer:

- The EMA update runs once per optimizer step, after the step and after the QAT and
  gain-constraint bookkeeping, on the un-compiled module, with `torch._foreach_lerp_`
  over the twin's parameters keyed by NAME against the live model's (the deep copy
  guarantees equal key sets; the update asserts it once at snapshot). Parametrised
  tensors (the ternary shadow weights) are updated through their `original` tensors, so
  the twin quantises an EMA of the shadow weights exactly as the live model quantises
  its own. Float buffers lerp; integer buffers copy.
- The checkpoint carries the twin under its existing `code_ref` key; a resume loads it
  strictly (the existing rule: never re-snapshot mid-run).
- The compile-warmup window before the snapshot keeps its existing fallback.
- New readouts at every log step: `tul/code_enc_var` (the term), `tul/code_enc_std`
  (mean `σ` over `m, j`), `tul/code_enc_active` (fraction of coordinates under `γ`), and
  the TARGET-side instrument `tul/code_tgt_std` (the same std on the twin's code cells,
  no loss: the collapse reading). At val: `val/code_tgt_std`, beside the existing own and
  shuffled cosines, `val/code_eff_rank` and `val/ce_tf`.

The arm `tul-code-ema` (config `tul_code_ema.yaml`, composed from `tul_code_target.yaml`):

- `training.init_from` the VAE stage `tul-code-vae/step_10000.pt` (E and the coda trained
  on truth codes), `data_skip_batches: 10000`, 5000 steps, the ramp and the panel
  settings, NO `train_only` (the whole model trains; the trainer's front check is
  satisfied by `code_target_ref`), `code_target_ref: true`, `code_target_ema: 0.996`,
  `code_enc_var_lambda: 0.02`, `code_enc_var_gamma: 1.0`, `code_target_detach: true` (the
  coda reads the predicted cells with stop-gradient, arm A's rule, so the reader adapts on
  tokens from step 0 while the loop learns from the regression alone),
  `token_state_dropout: 0.0`.
- Stated and not hidden: E's ONLINE copy receives no gradient on this arm (the online
  path never encodes a target), so the twin's E stays the VAE stage's E and the target
  moves only through the EMA of the front. That is the I-JEPA regime (one architecture,
  the target side an EMA of the context side). An arm that also trains the online E
  through the coda's read of the true code is a separate knob and is not in Stage 1.
- The one-factor control `tul-code-ema0`: the same config with `code_target_ema: 0.0` and
  `code_enc_var_lambda: 0.0` (the frozen twin, the whole model training). Both arms 5k on
  the runner, sweeps at 2500 and 5000, paired on the same 480 rows, plus the strict
  ruler and `fan4-all-fp0` at 5k for the token CE pairing.

### Stage 2: the per-pass factor split (written now, built after Stage 1 reads healthy)

`tul.code_factors: K` (`0` = off). One learned analysis matrix `P ∈ R^{C×C}` per cell
index (or shared across the `M` cells: an open choice for the Stage 2 prereg), split into
`K` blocks `P_k ∈ R^{C×r}`, `r = C / K`, `K | C`. The `K` predictors are the loop's
passes: the exit state of pass `t` through `TULCodeProj` is the pass-`t` cell, and only
factor `min(t, K)` of it is charged: `L_pred = mean_t ‖P_t^T pred_t − P_t^T z‖² / r` over
the realised passes. The coda reads the pseudo-inverse reassembly `(P^T)^† û` of the
factors the realised depth `T` produced, missing factors zero. `L_orth` (eq. 7) on `P`;
`L_fac` (eq. 9) on the target's factor coordinates trains `P` (the target itself is
stopped). Weights as Table 10 (`0.10, 0.05`).

Why this is a candidate for a pass job and not a restriction: the coda's direct token
read is untouched (no geometry is cut), and depth adds factors by construction, so a
deeper draw carries more of the code. The K-curve then rises mechanically, and the
reading is NOT the K-curve alone: it is whether the coda's CE at depth `T` tracks the
factor count (the factors carry usable content) or stays flat (they do not). The paper's
own caution applies: orthogonal factors are not causal factors, and the split forbids
the predictors from sharing information. `K` must be at most the max depth (8); the
first value is `K = 4` with passes 5 to 8 charging factor 4.

## Alternatives considered

- **LeJEPA's SIGReg on E with the full flow gradient** (ran, `tul-code-lejepa`): moved the
  code but cost the coda 0.68 nats and gave no conditional sample; no EMA, so the online
  encoder's input drift was the collapse route. Stage 1 keeps the target stopped and
  moves it by EMA instead.
- **A frozen semantic target** (SONAR, `code_target_source: sonar`, rung P3 of LXTUL-P):
  a fixed target of a different kind; not built for this tree (the Spark cannot run
  SONAR; the build was stopped 2026-09-22). Orthogonal to Stage 1 and stays queued behind
  it.
- **Variance floor on the TARGET as a loss** (the paper's `L_fac` without projectors):
  has no gradient path when the target is stopped and there is no projector; kept as an
  instrument (`code_tgt_std`) in Stage 1 and becomes a loss in Stage 2 through `P`.
- **A momentum schedule (0.996 → 1.0, I-JEPA)**: not in Stage 1; one fixed `m` keeps the
  arm one factor from its control and the schedule is a second knob for a second arm.
- **Training the online E through the coda's read of the true code** (the VAE objective
  continued online): a third gradient path into the target's definition; deferred so
  that Stage 1 measures the EMA effect alone.
- **Factors first (Stage 2 without Stage 1)**: per-pass targets were met in one step on
  every fixed target so far (`per-pass-targets-met-in-one-step`); a factor split on a
  target that cannot move inherits the 0.15 ceiling. Stage 1 first.

## Acceptance criteria

Build (Stage 1), each with a test that fails when the behaviour is broken:

- `code_target_ema: 0.0` and `code_enc_var_lambda: 0.0` are bit-identical to HEAD: loss,
  logits, parameter names, checkpoint keys, and the twin's state after N steps.
- One EMA update on a tiny `code_target_ref` model: every twin tensor equals
  `m · before + (1 − m) · live` (parametrised shadow weights included), the live model is
  untouched, and the twin's forward output moves after the update.
- Checkpoint round trip: save after k updates, resume, the twin's tensors equal the saved
  ones and the next update continues from them (no re-snapshot).
- The floor: on a hand-built `[B, S, M, C]` batch with known per-coordinate stds the term
  equals the hand value; exact 0 when every std is at or above `γ`; positive with finite
  gradients into the predicted cells on a collapsed batch (all valid cells equal); pads
  and invalid slots excluded; exact 0 with fewer than two valid slots.
- `TULConfig` refuses `code_target_ema` without `code_target_ref`, outside `(0, 1)`, and
  either knob without `code_target`; `tul_setup` accepts the keys and prints them.
- `tul_code_ema.yaml` and `tul_code_ema0.yaml` compose through Hydra and build a
  `TULConfig` (`compose-every-config-before-queueing`); a 12-step smoke on the 5090 runs
  the EMA update and the floor on the real model.

Run (the prereg in `lab/experiments/planned/` states the bars; the shape):

- Health: `val/code_tgt_std` at 5k stays above half its step-0 value and the shuffled
  cosine stays under 0.5 (the target neither collapses nor freezes).
- Movement: own minus shuffled cosine at 5k exceeds the `ema0` control's by a stated
  margin, i.e. the target moved toward the predictable.
- Decodability: `val/ce_tf` (the coda on the TRUE code) within a stated margin of the
  control's.
- Depth: the per-pass code cosine ladder and the token K-curve at forced depths, read
  against the slot-loop yardstick (+0.002) and the control.

## Risks

- The floor acts on the online cells; the target inherits the online front by EMA, so a
  slow drift of the target into a low-rank code is still possible. `code_tgt_std` and
  the effective rank read it; the prereg's health bar is the abort rule.
- The coda's token path stays a shallower route (condition A) for most of what the cell
  could carry; the moving target may make the cell predictable AND worthless to the
  reader. `val/ce_tf` against `ema0` and the worth profile read it.
- The EMA over 270M parameters per step is one `foreach` pass; the twin already costs
  about 1.1 GB. No new memory; the step cost is measured in the smoke.
- The paper reports no text result; its gains are 5 to 35 % relative on other domains,
  and control is "environment-dependent" in its own words.
