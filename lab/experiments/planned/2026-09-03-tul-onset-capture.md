# Planned: onset capture — what happens inside the slot loop when a MUX-next arm spikes?

Status: planned
Date: 2026-09-03 (frozen before any capture arm ran; only a 60-step replay smoke, whose
numbers are not metrics, precedes launch). Branch `tul/think-once`, worktree
`/home/wolfe/morph-to`. Supersedes nothing; the think-once panel prereg
(`2026-09-03-tul-think-once-panel.md`) stays as written, its remaining arms and readouts
deferred (queue log 22:34, "STOPPED BY WOLFE").

## Question

Five of five forecast arms of the think-once panel (M-next, cond4, coda8, and both
frozen-z twins) tripped the `preclip/total > 1e4` rule between steps 1009 and 3618, every
one after the 1000-step LR ramp ended. The probe shape is the same on all five: the coda's
gradient is flat, the conditioning stack's gradient (where one exists) is flat, and the
core, prelude, slot modules and embedding spike together, single steps that snap back and
escalate. The 2026-09-02 paid-axis record describes the same shape ("a spike train,
mostly recovering, divergence when a cluster lands") and measured its trigger as the
ternary cusp vault (a coherent drift of a tensor's mean weight re-thresholding many codes
in one step; ternary off ran 3/3 clean), and its amplifier as an aligned low-rank blow-up
across the six weight-shared core blocks. Two mechanisms fit today's data, and the
instruments that separate them exist. Wolfe's hypothesis (22:30): "the loop wants to be
asymptotic to 1 and sometimes we are stepping over the edge."

1. Is the trigger the ternary cusp vault (a forward, weight-side event) or the backward
   product through an expansive loop (a Jacobian-side event), or the first amplified by
   the second?
2. Is the spike regime reproducible bit-exactly, and does it appear at all on the eager
   kernels (the kernel-bug question)?
3. Does the loop's contraction rate drift toward 1 over training on a forecast arm, and
   does a spike step sit past 1?

## Arms

| # | run | config | one line |
|---|---|---|---|
| C0 | `cap-c0-nt` | `tul_cap_c0` | R4 (M-next) with `training.ternary=false`, fused, default mode, tripwire; the cheap trigger discriminator |
| C1 | `cap-c1-det` | `tul_cap_c1` | R4 in deterministic mode (eager kernels, compile off), rolling checkpoints every 50 (keep 12), grad probe every step, Jacobian probe every 25 steps at iterations 0/3/7, rank every 5, cotangent hook on; runs until the tripwire |
| C2 | `cap-c2-replay` | `tul_cap_c1` + `training.resume=<ROLL before onset>` `training.steps=<onset+30>` `jac_probe_every=1` `loop_rank_every=1` `batch_dump_every=1` `ckpt_rolling_every=0` | the replay through the first spike with every instrument on every step, no tripwire |

Onset = the first probed step ≥ 200 with `preclip/total > 1e3` in C1 (`lab/divergence/
onset_locate.py`); the replay starts from the newest rolling checkpoint at least 10 steps
before it, so both sides of the onset carry calm steps as the within-run control.

## Readouts (per step, from `probe.jsonl` of C1 and C2)

- `preclip/*` per module family; `loss/total`, `loss/ce_main`, `loss/mux_local` of the
  same step (a forward explosion moves the loss; a backward-only blow-up does not).
- `loop/cot_norm_t{k}`: the cotangent norm reaching iteration k's output in the backward.
  Its ratio between successive iterations is the realized backward gain per iteration.
- `loop/delta_mean_t{k}`, `loop/delta_ratio_t{k}`, `loop/in_norm_t{k}`: the forward step
  size per iteration; successive ratios estimate the contraction rate.
- `loop/eff_rank_t{k}`: entropy effective rank of the active slot states per iteration.
- `jac/sigma_t{k}`, `jac/rms_t{k}`, per block: the operator's worst and typical gain at
  the live operating point; `jac/sigma_conv_t{k}` is the convergence residual.
- `MORPH_DIAG_FWD=1` (env, C1 and C2): FWDNORM per block and TERNFLIP (ternary code
  flips since the previous step, per section) in the diag file.
- The C2 batch dumps: token ids, labels and the slot layout of every replayed step.

Spike step := a probed step with `preclip/total` above 10× the median of the 50 calm
steps before the onset. Calm steps := the 50 probed steps before the onset.

## Predictions (frozen)

Trigger discriminators:
- **P-C0.** C0 (ternary off) reaches 5000 steps with the tripwire silent: 60%.
- **P-C1a.** C1 (eager, deterministic) shows the spike regime (a spike step ≥ 200 with
  `preclip/total > 1e3`) before 5000 steps: 75%. (FALSE with C0 FALSE ⇒ the fused kernels
  are the prime suspect and the next arm is the eager-vs-fused gradient diff.)
- **P-C2a (reproducibility).** The replay's `preclip/total` at the onset step matches
  C1's at that step to within 1e-3 relative: 80%.
- **P-C2b (vault).** On spike steps, TERNFLIP's core-section flip count exceeds 5× its
  calm-step median: 45%.
- **P-C2c (forward explosion).** On spike steps, `loss/total` on that batch exceeds the
  calm median by more than 1.0 nat, OR FWDNORM's core max exceeds 10× its calm median:
  55%.

Amplifier discriminators:
- **P-C2d (backward product).** On spike steps the cotangent norm grows from the last
  grad iteration back to iteration 0 by a factor > 30 (mean successive ratio > 1.5 over
  8 iterations), against < 5 on calm steps: 55%.
- **P-C2e (operator).** On spike steps `jac/rms_t3` (the typical gain of one core step at
  iteration 3) exceeds 1.0, against < 1.0 on calm steps: 50%. `jac/sigma_t3` on spike
  steps exceeds 2× its calm median: 50%.
- **P-C2f (asymptotic to 1, Wolfe).** Over C1's calm steps the median `jac/rms_t3` rises
  monotonically with step (Spearman ρ > 0.8 over the 25-step probes from step 1000 to the
  onset) and its last calm reading sits within 0.15 of 1.0: 55%.
- **P-C2g (attractor collapse).** On spike steps `loop/eff_rank_t7` is below 0.5× its
  calm median: 35%.

Content:
- **P-C2h.** Spike-step batches contain a slot whose next span is a pad or the row end
  at more than twice the calm-step rate: 30%.

## Decision rule (binding)

- P-C0 TRUE and P-C2b TRUE ⇒ the trigger is the ternary cusp vault under the LR plateau.
  Fix lane: code-assignment hysteresis on the ternary threshold (DIVERGENCE-README open
  lever 2), validated on R4 with the tripwire. The contractivity question becomes a design
  note, not a bug.
- P-C0 FALSE and (P-C2d or P-C2e) TRUE ⇒ the amplifier is the loop's backward product
  and it needs no ternary trigger. Fix lane: clip-through-time (bound the cotangent
  between iterations, forward untouched) first, contractive-by-construction second.
- Both trigger and amplifier TRUE ⇒ the ternary flip is the impulse and the loop is the
  amplifier (the README's aligned low-rank blow-up). Both levers, hysteresis first.
- P-C1a FALSE and P-C0 FALSE ⇒ the fused kernels are the prime suspect; next arm is the
  eager-vs-fused per-batch gradient diff at the R4 checkpoint.
- P-C2a FALSE ⇒ the capture is not a capture; stop, find the nondeterminism, rerun.

## Method

Runner `/home/wolfe/morph-scratch/cap/run_onset_capture.sh`: C0 (tripwire watcher) →
C1 (tripwire watcher; `MORPH_DIAG_FWD=1`, `MORPH_DIAG_OPT=<dir>/diag.txt`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`) → `onset_locate.py` → C2 (no watcher; same env). One
trainer at a time (UPS). Before launch: a 60-step smoke of C1 with rolling checkpoints
every 20, then a replay from `ROLL_step_40` to step 50, and a row-by-row comparison of the
two probe files over steps 41–50 (the reproducibility gate on the instrumented path; its
numbers are not metrics). Artifacts to `lab/experiments/results/2026-09-03-tul-onset-
capture/` (probe files, diag files, the locate JSON, the batch dumps of spike steps).
Estimated: C0 ≤ 45 min; C1 ≈ 1.5 h to a 3600-step onset at ~1 step/s; C2 ≈ 30 min.

### Method note, 2026-09-03 23:45 (before launch; Predictions untouched)

The 60-step replay smoke FAILED its first pass: the replay diverged from the step after
every probed step, with weights and gradients bit-identical (`lab/divergence/
probe_state_diff.py`, `probe_grad_diff.py`, `probe_alloc_diff.py`, `probe_step_diff.py`,
all on `ROLL_step_40` of the smoke). The only state that moved across a probe was the CUDA
generator: `_jacobian_probe` restored the RNG around its capture forward but not around
`probe.measure`, whose power iterations run the core blocks in training mode with block
dropout 0.1. Fixed in `morph/training/train.py::_jacobian_probe` (restore now wraps the
measurement) and pinned by `tests/test_core_jacobian.py::
test_probe_measurement_is_rng_neutral_with_dropout`, which first proves a bare measurement
moves the generator on its fixture. After the fix the smoke's two resumes (per-step
Jacobian probe; the full C2 flag set) match the original run bit for bit over steps
41–49. The replay's Jacobian probe is restricted to iteration 3 at 60 power iterations
(convergence residual 0.0 on the smoke) because the three-iteration, per-block, 100-
iteration setting cost ~70 s per step; the C2 replay now costs ~15 s per step.

## Not verified before launch

The deterministic configuration on the MUX arm at this shape (the record's
bit-reproducibility was measured on A1 and A2); the Jacobian probe's cost at 100 power
iterations on the slot loop; rolling-checkpoint disk use (12 × ~3 GB on /home, 1.4 TB
free); whether the onset in eager mode lands inside 5000 steps at all (C1 has no
guarantee; a clean C1 is itself a reading, see P-C1a). The smoke gates the first and the
reproducibility of the instrumented path; nothing gates the last.
