# Success: onset capture — what happens inside the slot loop when a MUX-next arm spikes?

Status: success
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

### Method note, 2026-09-04 02:15 (after the runs; Predictions untouched)

The replay resumed the wandb run id of C1 (`checkpoints/morph/cap-c1-det/wandb_id.txt`,
`kg0xekg4`), so wandb dropped its rows as non-monotone steps; the replay's record is its
local `probe.jsonl`, `diag.txt` and batch dumps, copied to the results directory. The
tripwire watcher on C1 killed the trainer at step 1208 (first `preclip/total > 1e4`);
the locate step chose onset 1170 and `ROLL_step_1150.pt`; the replay ran steps 1151–1199
(49 rows, `training.steps=1200`). The offline gain sweep over the saved checkpoints
(`lab/divergence/jac_sweep.py`, written after the runs because C0 logged no Jacobian) is
reported as a separate results section and was not pre-registered.

## Results (2026-09-04 02:30; C0 00:32, C1 01:55, C2 02:12; scripts in the scratchpad, numbers from the files in `results/2026-09-03-tul-onset-capture/`)

**C0 (ternary off, fused, 5000 steps).** Tripwire silent. `preclip/total` median 2.2,
max 43.5 at step 3883; spikes above 10x per 500 steps from step 500: 13, 4, 3, 2, 0, 3,
1, 2, 1 (no escalation). Final val CE 4.2949 (one eval). The panel's ternary-on draw of
the same arm: max 33,625, tripped at 3618.

**C1 (ternary on, eager, deterministic).** Detonated at 1208 (max 1.25e4). Onset 1170.
The spike train: 35.7 at 1111, 16 at 1135, 15 at 1144, 66 at 1161, 1678 at 1170, 483 at
1185, 1119/1396/867 at 1198–1200, 12,521 at 1208, with `preclip/total` at 2.7–3.6 between
spikes. Median of the 50 calm steps before the onset: 3.642.

**C2 (replay from `ROLL_step_1150`, every instrument every step).** Bit-exact against C1:
`preclip/total` and `loss/total` differ by 0.0 on all 49 replayed steps.

Per prediction (spike step := `preclip/total` > 36.4, i.e. 10x the calm median; 16 spike
steps in the replay window 1151–1199; 33 others):

| prediction | credence | reading | verdict |
|---|---|---|---|
| P-C0 ternary off clean to 5000 | 60% | clean; max 43.5 | TRUE |
| P-C1a eager+deterministic shows the regime | 75% | onset 1170, detonation 1208 | TRUE |
| P-C2a replay matches to 1e-3 | 80% | relative error 0.0 on 49 steps | TRUE |
| P-C2b core TERNFLIP > 5x calm on spike steps | 45% | 0.1x–2.3x on all 22 spike steps of C1 after step 1160; Spearman(flips, log preclip) = −0.49 | FALSE |
| P-C2c forward explosion (loss +1 nat or FWDNORM core 10x) | 55% | FWDNORM core 0.80–1.15x; loss above +1.0 nat on 1 of 22 (step 1208, +1.64) | FALSE |
| P-C2d cotangent product > 30 on spike steps, < 5 calm | 55% | spike steps 39–2436 (all 16 above 30); calm steps 200–1140: 1.6–3.0; the non-spike steps INSIDE the window: median 18.5, max 125 | TRUE (the calm half holds only before the onset window) |
| P-C2e `jac/rms_t3` > 1 on spike steps, < 1 calm; `sigma_t3` > 2x calm | 50% | rms above 1 on 5 of 16 spike steps and 2 of 33 others (medians 0.979 vs 0.954; Spearman rms vs log preclip 0.65, p 5e-7); sigma above 2x on 4 of 16 | FALSE as frozen; graded, not binary |
| P-C2f gain rises monotonically to within 0.15 of 1 | 55% | Spearman 0.86 over 1000–1150 (p 0.014), 0.98 over all calm probes; last calm probe 0.951 | TRUE |
| P-C2g `eff_rank_t7` < 0.5x calm on spike steps | 35% | 56–61 on spike steps against 38–52 calm; higher, not lower | FALSE |
| P-C2h pad / row-end next span at 2x the calm rate | 30% | last-slot rate 0.0192 on spike steps, 0.0191 on others; valid slots 313 vs 314 | FALSE |

Operator readings along the run (C1, every 25 steps; whole slot-loop step at iteration 3):

| step | rms_t3 | sigma_t3 | per-block rms (range) | block sigma product |
|---|---|---|---|---|
| 0 | 0.872 | 3.21 | 1.001 | 74 |
| 500 | 0.887 | 7.8 | 1.003–1.005 | 474 |
| 1000 | 0.912 | 19.3 | 1.009–1.012 | 46,924 |
| 1150 | 0.951 | 29.6 | 1.014–1.016 | 241,642 |
| 1175 | 0.956 | 60.7 | 1.014–1.020 | 377,796 |
| 1200 | 0.996 | 99.5 | 1.016–1.021 | 949,286 |

`rms_t0` (iteration 0, the freshest state) crosses 1 first: 1.069 at 1150, 1.647 at
1200. The cotangent growth on spike steps sits in the last backward iterations (ratios
3–8 between iterations 2, 1 and 0 at step 1170: 796 → 2206 → 16,941 → 98,390).

Forward, per iteration (replay window medians, non-spike / spike steps): relative step
size `delta_mean` 0.62 / 0.64 at iteration 4 and 0.60 / 0.56 at 7; successive-step ratio
`delta_ratio` 0.72 / 0.78 at iteration 4, 0.69 / 0.71 at 7; realized gain `core_gain`
1.07 / 1.08 at 4; effective rank 172 → 48 across the eight iterations on both. The
trajectory is not near a fixed point; it contracts by ~0.7 per iteration along its own
path and by ~0.95 in a generic direction.

## Verdict

Decisive on the question, with a gap in the frozen decision rule. Ternary is necessary
(P-C0) but the named trigger, a ternary code-flip burst, is absent (P-C2b), and the
amplifier predictions hold (P-C2d, P-C2f; P-C2e graded). None of the three rule branches
fires as written; the record says so instead of patching the rule. Reading: training on
the forecast target drives the slot loop's typical gain toward 1 (with ternary QAT the
gain reaches it; the sweep below asks whether it moves at all without ternary), and the
spike is the backward product of eight iterations through a map that contracts a generic
direction by 0.95 and amplifies a few aligned directions 100x. The forward never
explodes; the loss on a spike step is the loss of a calm step. The kernel-bug lane and
the "cannot reproduce" lane are closed by P-C1a and P-C2a. Lever order, from the
decision rule's nearest branch: bound the backward product (clip-through-time, forward
untouched) first; a rate held below 1 by construction on the MAP second; hysteresis on
the ternary cusp has no evidence on this face.

## Updated hypothesis

"The loop wants to be asymptotic to 1" (Wolfe) is the measured behaviour on the forecast
target. The failure is not a bug and not a discontinuity; it is the regime the loop is
trained into with nothing holding its rate. Design note:
`.agents/notes/proposed/architecture/2026-09-04-loop-contractivity-as-design.md`. Next
prereg: R4 with clip-through-time (cotangent bound between iterations) and the gain panel
as a readout, tripwire on, 5000 steps; acceptance is `jac/rms_t3` < 0.95 for the run and
a slot depth-sweep CI above 0.

## Results, part 2 (2026-09-04 02:29; the offline gain sweep, NOT pre-registered)

`lab/divergence/jac_sweep.py` loads each saved checkpoint eager into the model its config
describes and measures the slot loop on ONE fixed batch of validation rows (batch 6, probe
seed 0, 60 power iterations, iterations 0/3/7, per block) plus one forward+backward for
the trajectory and cotangent rows. Rows in `results/.../jac_sweep.jsonl`. The coreless R0
arm has no loop and was skipped (the loader now says so instead of crashing).

| run (recipe) | step | `rms_t3` | `rms_t0` | `sigma_t3` | `sigma_t0` | step ratio iter 4 / 7 | cotangent t0/t_last | tripwire |
|---|---|---|---|---|---|---|---|---|
| cap-c1-det (M-next, ternary ON, eager) | 500 | 0.885 | 0.892 | 7.0 | 6.8 | 0.43 / 0.39 | 1.9 | tripped 1208 |
| | 800 | 0.902 | 0.906 | 15.0 | 15.2 | 0.55 / 0.64 | 1.9 | |
| | 1000 | 0.910 | 0.933 | 18.7 | 21.0 | 0.52 / 0.54 | 2.1 | |
| | 1150 | 0.914 | 0.960 | 33.1 | 43.4 | 0.55 / 0.56 | 2.9 | |
| | 1200 | 1.000 | 1.519 | 111.8 | 605.7 | 0.74 / 0.68 | 88.5 | |
| cap-c0-nt (M-next, ternary OFF, fused) | 2500 | 0.890 | 1.261 | 9.0 | 50.4 | 0.20 / 0.14 | 2.1 | silent to 5000 |
| | 5000 | 0.889 | 1.431 | 6.1 | 52.6 | 0.19 / 0.14 | 2.1 | |
| to-mnext (M-next, ternary ON, fused) | 2500 | 0.901 | 1.129 | 12.3 | 343.1 | 0.48 / 0.44 | 2.1 | tripped 3618 |
| to-mown (M-own, ternary ON, fused) | 2500 | 0.901 | 0.905 | 6.9 | 13.2 | 0.23 / 0.24 | 1.6 | silent to 5000 |
| | 5000 | 0.913 | 0.921 | 8.3 | 15.6 | 0.20 / 0.21 | 1.6 | |
| to-a1-s1 (A1, ternary ON, bptt 4) | 2500 | 0.975 | 1.642 | 175.6 | 572.6 | 0.84 / 0.85 | 1.7 (4 iterations) | silent to 5000 |
| | 5000 | 0.912 | 5.827 | 42.4 | 5615.1 | 0.37 / 0.53 | 1.4 (4 iterations) | |

Readings (one batch each; the C1 rows agree with the in-run probe on its own batches to
within 0.01 in `rms_t3`):

1. **The typical-gain drift needs ternary.** Ternary off holds `rms_t3` at 0.889–0.890 at
   2500 and 5000, the level the ternary run had at step 600; the ternary run climbs
   0.885 → 0.914 → 1.000. The fused ternary draw sits at 0.901 at 2500 and tripped at
   3618 (no checkpoint between; whether it reached 0.95 first is unmeasured).
2. **The drift needs the forecast target too.** M-own with ternary moves 0.901 → 0.913
   over 2500 steps with a worst gain of 7–8 and a trajectory that contracts by 0.2 per
   iteration; it ran clean.
3. **Two forward regimes.** Ternary-off M-next and M-own converge fast along their
   trajectory (successive-step ratio 0.14–0.24 after iteration 4: the loop is done by
   iteration 3 or 4). Ternary-on M-next stays at 0.5–0.7 (sequential regime) and A1 at
   2500 sits at 0.84 (barely contracting). The paid-loop record's K4 saturation and the
   slot arms' "free ride" (≤ 0.011 nats however long the loop ran) are what the fast
   regime looks like from the loss.
4. **A typical gain near 1 is not the spike train by itself.** A1 at 2500 has `rms_t3`
   0.975 and a worst gain of 176 (573 at iteration 0), and its cotangent does not grow
   through the loop (1.7 over its 4 gradient iterations); it ran clean at seed 1. Its
   differences from M-next are the loss (token CE through the coda, not a MUX loss at the
   loop's output) and `bptt_depth` 4 (the product runs over 4 iterations, not 8). The
   queued A1 pair at `bptt_depth: 8` (`tul_to_a1_b8`, Wolfe's ask on 2026-09-03) is the
   direct test of the second; the first needs an arm that moves the loss attachment alone.
   Seed 2 of A1 detonated at 2041 with the takeover shape, not the spike train.

Verdict unchanged. The discriminating quantity is the backward product through the loop
(`loop/cot_norm_t0 / cot_norm_t_last`): 1.4–2.1 on every healthy checkpoint at every
gain, 39–2436 on the spike steps. The typical gain's drift to 1 is the condition under
which the forecast arm's product grows; it is necessary here, not sufficient in general.
