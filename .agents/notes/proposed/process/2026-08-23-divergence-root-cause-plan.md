# Agent Note: plan to find the cause of the TUL core takeover

Status: proposed

**This note is the resume point.** It carries every number, path and decision rule needed
to continue cold after a context reset. Tick the boxes as work lands and record the
evidence inline — an unticked box with no evidence is unstarted work, whatever a summary
elsewhere says.

## Problem

The TUL core "takeover" — the core reaches ~99.9 % of the total gradient norm, the single
global clip then divides every other region by 1e5–1e9, and prelude, coda, embeddings and
the TUL parameters stop learning — is reproduced 5/5 on the control and is still
unexplained after roughly 40 diagnostic runs.

The reason it is unexplained is a method problem, not a shortage of runs:

- **Four interventions survive** — `token_state_dropout=0`, `core_gain_clip=1.5`,
  `ademamix_alpha_cap=1.0`, `spectral_penalty cap=2.0 lambda=10`. They change a forward
  pass, a per-iteration gain clip, an optimiser momentum weight and a weight-norm bound.
  They cannot all be the mechanism, and a fifth surviving arm cannot separate them.
- The placebo (`token_state_dropout` 0.15 → 0.145, 96.7 % strength, cannot be a cure)
  diverges **2/2**, so survival is not free and the four are not merely dice rolls.
  [`docs/tul-divergence-rca.md`](../../../../docs/tul-divergence-rca.md) Part 7.
- `spec/sigma_max` is a **correlate, not a trigger** — the placebo refutes the sigma 3.0
  threshold reading (seed 0 crossed at ~1950 and detonated at ~2900; seed 1 crossed at
  ~750 and detonated within ~50).
- The order-parameter framing the RCA closes on is **also not the discriminator**:
  measured 2026-08-23, `rho_eff` is 1.9–3.2 at depth 6–8 on every checkpoint that trains
  to 20k. [`docs/experiments/failures/2026-08-23-tul-forward-backward-asymmetry.md`](../../../../docs/experiments/failures/2026-08-23-tul-forward-backward-asymmetry.md).

**And the runs are not reproducible at fixed seed.** RCA §17 suspected `bag_mean`'s
`index_add_` float atomics and marked it NOT VERIFIED. Verified 2026-08-23:

| measurement | result |
|---|---|
| `bag_mean`, identical inputs, 20 repeats | **20/20 non-bit-identical, forward AND backward** |
| backward disagreement | 30.7 % of elements, max abs 3.906e-3 (bf16) |
| `torch.use_deterministic_algorithms(True)` | **does not flag it** — the standard guard gives false assurance |
| one full TUL training step, 4 repeats | **4/4 non-identical**, worst tensor 100 % of elements, **max relative gradient error 3.92e-2** |
| the same, with a deterministic `bag_mean` | 4/4 non-identical, **max relative error 3.17e-3** (12× better, not zero) |

A 4 % gradient perturbation every step is a candidate **driver** on a system the RCA
describes as a knife-edge, not only a nuisance. It is certainly why 30 runs settled
nothing: step 1900 of run A is not step 1900 of run B, so nothing can be bisected.

Remaining nondeterminism after the `bag_mean` fix is `tl.atomic_add` in the attention
backward — `morph/kernels/triton/fused_csa_attention.py:279` and
`morph/kernels/triton/fused_hca_attention.py:288`. Both are present in A0 as well, and
A0 does not take over, so they are not sufficient on their own.

## Proposal

Four phases. Each has a gate; do not start the next until the gate holds.

### Phase 0 — make the run reproducible  (~1 h for the first two tasks)

- [x] **0.1 Land the deterministic `bag_mean`.** Replaced the `index_add_` pair in
      `morph/model/tul.py` with the one-hot `bmm` formulation.
      EVIDENCE: `morph/model/tul.py:199-210`. `pytest tests/test_bag_mean_determinism.py -q`
      → **7 passed** on the new version. Prior scratchpad measurement: 0/20 non-identical
      fwd+bwd, agrees with `index_add_` within bf16 epsilon (fwd 1.56e-2, bwd 3.91e-3),
      **10 % faster** (0.165 ms vs 0.184 ms at B=12, L=1152, C=1024), +1.8 MB one-hot.
- [x] **0.2 Add a determinism gate test** — `tests/test_bag_mean_determinism.py`, 7 tests:
      3 CUDA determinism gates (forward, backward, and one under a bag permutation so the
      test cannot pass by the indices happening to be sorted), plus 4 contract tests
      (span mean, dump row exactly zero, empty bag → 0 not NaN) that also run on CPU.
      EVIDENCE: reverted `tul.py` to `index_add_` once and re-ran →
      **3 failed, 4 passed**, the three failures being exactly the determinism gates;
      restored the fix → **7 passed in 1.64s**. The gate does fail when the code is broken.
- [x] **0.3 Audit the rest of the TUL path** for other order-dependent accumulation.
      EVIDENCE: grep for `index_add_|scatter_add_|index_put_|put_|bincount|scatter_reduce`
      over `morph/model/**`, `morph/model/layers/**`, `morph/training/**`, plus `atomic_`
      over `morph/kernels/triton/**`. Findings:
      - The slot scatter-back (`tul.py:257` `scatter_positions`), the rank scatter
        (`tul.py:326`) and the label scatter (`transformer.py:1491`) are all `scatter_`
        **writes**, not accumulations. One source index per destination, and the backward
        of a write-scatter is a gather. Deterministic; nothing to do.
      - `fused_ce.py:127` — `scatter_add_` with a `[c, 1]` index, one index per row, so
        no two atomics can collide. Deterministic in practice.
      - `fused_ce.py:224` — `scatter_add_` with a `[c, K]` index on the **TST multi-label**
        path. Collisions are possible only when one row repeats a label. The TUL arms run
        `tst_bag_size: 0`, so this path is inactive for every run in this programme.
        NOT VERIFIED: whether it matters on a TST run. Left as a known edge.
      - The two Triton `tl.atomic_add` sites remain: `fused_csa_attention.py:279` and
        `fused_hca_attention.py:288`. Both are also in A0, which does not take over.
      **The residual is real and observable**: two 60-step runs at seed 0 differing ONLY in
      `grad_probe_every` (0 vs 1, a `no_grad`/`detach` readout that cannot change the math)
      reached `train/loss` 8.4143 and 8.4520 at step 40. Changed launch timing alone moves
      the trajectory by 4.5e-2 in loss within 40 steps.
- [x] **0.4 THE GATE — FAILED. The run is still not reproducible.** Run `tul_a1` at ONE seed, twice,
      byte-identical config, to step 2600, eval DISABLED so an eval pass cannot perturb
      the RNG stream. This is RCA §22's P2, prepared 2026-08-18 and never run.
      **Decision rule, pre-registered:** the two runs must agree on the onset step within
      ±25 AND on `train/loss` at step 1000 to 4 decimals. If they do, the phenomenon is
      bisectable and Phase 1 starts. If they do not, the kernel atomics are the next
      suspect and Phase 0 continues — do NOT proceed to Phase 1 on an irreproducible run.
      
      **RESULT (2026-08-23): FAIL, on the FIRST BACKWARD PASS.** Two runs, one seed, one
      driver script so no edit could land between them, code pinned at `a81a158`. They
      share a data order, so at every step they see the same batch and any difference is
      nondeterminism alone. `preclip/total` relative difference: **>1e-6 at step 0**,
      1e-4 at step 5, 1e-2 at step 9, 0.1 at step **11**, 1.0 at step 50; median 6.5 %
      after step 100. Step-0 forward loss is 11.0681 in BOTH, so the models are identical
      at init and the first backward is where they part.
      `bag_mean` was necessary and nowhere near sufficient. Per this plan's own decision
      rule, cross-run bisection stays blocked and the attention-backward atomics move to
      the top of the queue.
      **Two consequences:** (a) any n=1 two-run comparison on this model is unreadable
      unless its effect beats the measured spread — median 6.5 % on the pre-clip gradient
      norm and median 0.0788 on `core_gain_t0` after step 100; every past single-run arm in
      this programme should be re-read against those numbers. (b) Within-run measurements
      are untouched, which is why Phase 1's ordering stands.
      EVIDENCE: [`../../../../docs/experiments/failures/2026-08-23-tul-run-replication.md`](../../../../docs/experiments/failures/2026-08-23-tul-run-replication.md)

### Phase 1 — watch the onset  (~45 min, needs 0.4)

Nobody has looked inside the onset. Every checkpoint is post-mortem, `spec/sigma` logs
every 100 steps, and the onset lasts about 140.

- [x] **1.1 Add per-step instrumentation** behind a config flag.
      `training.grad_probe_every` (0 = off, bit-exact) and `training.grad_probe_path`
      (optional JSONL mirror) in `base.yaml`. Trainer half: `_preclip_probe()` in
      `morph/training/train.py`, called between `scaler.unscale_()` and
      `clip_grad_norm_` — the only window where the gradients are both unscaled and
      unclipped — emitting `preclip/<region>` and `preclip/<region>.<block>`. One fused
      `torch._foreach_norm` over every gradient and ONE host sync. Model half:
      `_probe_loop` in `TULTransformer._tul_core`, collecting the **GLA carried-state
      norm** and the realized per-iteration core gain on GPU, read once per step, emitted
      as `loop/ret_state_norm_t*` and `loop/core_gain_t*`.
      EVIDENCE: 12-step smoke wrote 12 JSONL rows × 46 keys with live values, e.g. at
      step 11 `preclip/lm_mixer` 912.5 of a `preclip/total` 912.5 (the LM head owns the
      whole gradient before the takeover), `preclip/core` 0.0670, and a carried state that
      **integrates across the loop**: `ret_state_norm` 4167 → 4465 → 4734 → 4882 → 4945 →
      4955 → 4943 → 4925 over t=0..7, against a core gain decaying 1.348 → 1.105.
      Overhead at `grad_probe_every=1`: 2.09 → 2.08 steps/s, **0.5 %**.
- [~] **1.2 Re-run the control** with the probe. Run `phase1-onset-s0`: `tul_a1` +
      `ademamix_alpha_cap=3.5` (the CONTROL — `tul_short.yaml` ships the cure 1.0, so the
      arm the pre-registration named could never diverge), seed 0, eval and gen off,
      `grad_probe_every=1`, 5000 steps. It took over: core share 0.0145 at step 1400 →
      0.9981 by 2400, divergence guard first strike at 2620.
      **DEFERRED, and this is a real gap:** no dense checkpoints inside the onset. 36
      optimizer-bearing checkpoints is ~120 GB and hours of I/O, which does not fit the
      session's 1.5 h ceiling, so the offline `depth_gain` / `lin_ratio` probes STILL have
      no checkpoint to run on inside the window. Per-step scalars were collected instead.
      EVIDENCE: `ignore/perf/phase1/onset_s0.jsonl`, 45 series × every step; console log
      `ignore/perf/phase1/onset_s0.log`; wandb `morph-tul/phase1-onset-s0`.
- [x] **1.3 Which quantity moves first** — at PER-STEP resolution, not 25.
      Full writeup and the threshold-sensitivity sweep:
      [`../../../../docs/experiments/results/2026-08-23-tul-onset-ordering.md`](../../../../docs/experiments/results/2026-08-23-tul-onset-ordering.md).
      Analysis rule fixed in `ignore/perf/phase1/onset_order.py`, written and dry-run at
      step 853 of the run, before any takeover was visible.
      EVIDENCE, the four findings:
      1. **The runaway is confined to the FIRST loop iteration**, at every threshold tried
         (K ∈ {5,10,20}): `core_gain_t0` 1.422 → 16.93, `t1` 1.080 → 1.596, and
         `t2..t7` never depart. `core_gain_max` equals `core_gain_t0` at EVERY probed step
         of the run. Evidence AGAINST a compounding-through-depth ρ^T story — the late
         iterations carry the least gain, and they contract what the first one expands.
      2. Within the core, **block 0 leads** (1975) and the **coda is last** (2229–2444);
         prelude/tul/embed sit between at 2031–2043.
      3. **`preclip/lm_mixer` NEVER departs** in any of 12 (K, R) settings — baseline
         1.202, value 1.931 at step 3000 while `preclip/core` is 1.3e6. The takeover is
         not a loss explosion propagating backwards.
      4. **The GLA carried state is a FOLLOWER** — departs 2330 (K=10), 2417 (K=20), never
         (K=40); after `preclip/core` at every K ≥ 10. The Phase 1 hypothesis that the
         carry drives this is REFUTED on this run. It still has to be fixed for causality.
      Also: the loss reaches its whole-run MINIMUM (4.5500 at 1600–1700) while
      `core_gain_t0` is already climbing, and is 4.7137 at step 2100 with the core holding
      81.7 % of the gradient. The loss curve shows a healthy run 500 steps after takeover.
      NOT ESTABLISHED: that `core_gain_t0` leads `preclip/core`. It leads by 35–183 steps
      at K ≤ 10 and the order REVERSES at K ≥ 20; MAD units cannot fairly compare a 1e7
      excursion with a 12× one.

### Phase 2 — mediation, NOT another cure hunt  (~5 h, needs Phase 1)

The question that can finish this. Run control + placebo + the four known cures, 2 seeds,
with Phase 1's instrumentation, and ask **not** "did it survive" but:

> Which single logged quantity is (a) driven monotonically past some value in BOTH the
> control and the placebo, and (b) held below that value by ALL FOUR cures?

- [ ] **2.1 Write the pre-registration** in `docs/experiments/planned/` BEFORE the runs,
      naming the candidate quantities and the threshold rule.
      **Phase 1 has now supplied the candidate list and it is short:** `core_gain_t0`,
      the pre-clip core share, and `preclip/core.0`. `ret_state_norm` and `lm_mixer` are
      DEMOTED by Phase 1 findings 3 and 4 — one never moves and the other follows.
      EVIDENCE: pending
- [ ] **2.2 Run the six arms, 2 seeds each.**
      EVIDENCE: pending
- [ ] **2.3 Score it.** Exactly one qualifying quantity = the mechanism, and the four
      cures are four ways of holding it down. **Zero qualifying quantities is also a
      result**: the cures act through different paths, "cause" is the wrong shape of
      question, and Phase 3 applies.
      EVIDENCE: pending

### Phase 3 — the fallback, if Phase 2 finds no common mediator

Then this is a **rate**, not an event — Task #276's phenotype of transient excursions
whose rate climbs until one catches. Two of four control seeds already showed `grad_norm`
spikes (7.7e4, 4.9e4) that did NOT run away.

- [x] **3.1 Model the excursion rate against training step** from Phase 1's per-step logs.
      MEASURED, and it is the #276 phenotype. Probed steps with pre-clip core share > 0.25,
      per 200-step bin: **0, 0, 0, 0, 0, 0, 0** for steps 0–1400, then **5, 3, 13** for
      1400–1600, 1600–1800, 1800–2000, then the permanent takeover. Nothing for 1400 steps,
      then excursions that appear, fall back, and grow more frequent until one does not
      fall back. **Every pre-takeover excursion above 0.5 is ONE probed step long**, which
      is why the abort rule's "sustained for N steps" clause is load-bearing — a bare
      threshold would have false-fired at step ~1450, 570 steps early.
      EVIDENCE: [`../../../../docs/experiments/results/2026-08-23-tul-onset-ordering.md`](../../../../docs/experiments/results/2026-08-23-tul-onset-ordering.md)
      finding 6, and panel 3 of `tul_onset.png`.
- [x] **3.2 Ship an abort criterion instead of a cause.** SHIPPED, then FIXED after it
      missed a real divergence. `morph/training/divergence_guard.py` + `training.abort_core_share`.
      The first rule was "N consecutive probed steps above the threshold", validated on ONE
      trajectory with 12 passing tests, and it missed `repl-det-b` — a run that really took
      over (final share 0.8131) whose onset is intermittent (0.967, 0.768, 0.264, 0.624,
      0.989) with a longest consecutive stretch of 21 against a patience of 25.
      The shipped rule is a FRACTION over a sliding window, validated on three labelled
      trajectories: fires at 2038 on `phase1-onset-s0` (took over) and 3369 on `repl-det-b`
      (took over), never on `repl-det-a`, which PEAKED at a core share of 0.9369 and
      recovered to 0.0152. 20 of 27 swept parameter combinations separate all three.
      Verified LIVE end to end, not only in replay: config → forced probe → fire at step 13
      → `TAKEOVER_step_13.pt` → stop, exit 4.
      EVIDENCE: `tests/test_divergence_guard.py` (15 tests, three real trajectories as
      fixtures) and `ignore/perf/phase1/guard_smoke.log`. Superseded detail below.
- [~] *(superseded)* **3.2 first attempt.** The rule is now MEASURED and it
      is 4× better than the post-clip ratchet it replaces. Against the divergence guard's
      first strike at step 2620, sustained for 25 consecutive probed steps:
      pre-clip core share > 0.25 fires at 2031 (**589 steps** of warning), > 0.50 at 2033
      (587), `preclip/core` > 1.0 at 2032 (588), `core_gain_t0` > 2.0 at 2063 (557),
      core share > 0.90 at 2192 (428). The old POST-clip `gradnorm/core` ratchet gave ~140.
      Recommended: **pre-clip core share > 0.5, sustained 25 steps** — 34× above the
      healthy baseline of 0.0145, against a highest-healthy-value of 0.031 before step
      1900, and a share needs no per-arm scale calibration.
      **NOT IMPLEMENTED in `train.py`.** The table is the evidence for the rule, not the
      rule. Implementing it is the next commit, and it is acceptance criterion 4.
      EVIDENCE: [`../../../../docs/experiments/results/2026-08-23-tul-onset-ordering.md`](../../../../docs/experiments/results/2026-08-23-tul-onset-ordering.md)
      "The abort criterion".

### Two arms to add to Phase 2

- [ ] **A. `retention_carry=false`.** Only ever tested in the invalid n=1 sweep (RCA §13,
      arm E4). We must ship it anyway for the causality defect
      ([`../bug-fix/2026-08-23-retention-carry-breaks-causality.md`](../bug-fix/2026-08-23-retention-carry-breaks-causality.md)),
      so it is two birds. The carry is also a second recurrent loop inside the core loop
      with a forget gate biased to alpha near 1 (`retention_gate_bias: 2.0`).
      EVIDENCE: pending
- [ ] **B. deterministic `bag_mean` alone.** If the 4 % per-step gradient kick is part of
      the driver, removing it should move the divergence rate. Phase 0 gives this arm for
      free.
      EVIDENCE: pending

## UNBLOCKED 2026-08-23: a bit-reproducible configuration exists

The gate failed, and then the blocker it exposed was solved the same evening. The fused
CSA/HCA attention backward is the dominant source and is outside PyTorch's determinism
machinery. With `model.use_kernels=false` + the new `training.deterministic=true` +
`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before the process, **two 300-step training runs
are bit-identical on all 300 steps across all 85 probe series**, every console loss matching
exactly. Cost: 2.28× fewer tokens/s and roughly half the batch.

- [`the reproducibility result`](../../../../docs/experiments/results/2026-08-23-morph-bit-reproducible.md)
- [`the Agent Note`](../../implemented/architecture/2026-08-23-deterministic-training-mode.md)

**What this changes for the rest of this plan.** Phase 2's mediation becomes possible for
the first time, because two arms can now differ by their intervention alone. Three caveats
that are not optional:

1. The halved batch changes the gradient noise scale, so **every arm in a comparison must
   use the reproducible configuration**; a reproducible arm cannot be compared against the
   stored fast-configuration runs.
2. The takeover statistics do not transfer. Different attention implementation and half the
   batch: the base rate, the onset distribution and the abort thresholds must all be
   re-measured there before any arm is read.
3. Only 300 steps are verified bit-identical. A 4000-step run is not.

## Alternatives considered

- **Run more intervention arms.** Rejected: five surviving cures do not separate better
  than four. RCA §14 already names the n=1 sweep as measuring trajectory sensitivity
  rather than causation, and §20 says the same of the long run.
- **Measure the Task #276 order parameter live**, which is where the RCA ends. Demoted,
  not dropped: `rho_eff` is 1.9–3.2 on runs that finish, so the composition sigma cannot
  be the line between healthy and sick on its own. It stays a Phase 2 candidate quantity.
- **Bisect the onset directly with checkpoints.** This is Phase 1 and it is the obvious
  move, but it is worthless before Phase 0 — on an irreproducible run, two checkpoints at
  the same step are from different trajectories.
- **Chase all the kernel atomics first.** Rejected as the opening move: `bag_mean` alone
  is a 12× cut and costs nothing, and the attention atomics are also in A0, which does not
  take over. Measure whether the residual 3.2e-3 matters before paying a day for it.

## Acceptance criteria

1. `tests/` contains a determinism gate for `bag_mean` that fails on the `index_add_`
   version, and it is in the default `pytest tests/` run.
2. Phase 0.4's replicate test has run, with both runs' onset steps recorded here.
3. Either a named mechanism with the mediation evidence from Phase 2.3, or an explicit
   recorded finding that no common mediator exists, with the per-arm quantity table that
   shows it.
4. An abort criterion is implemented in `morph/training/train.py` and fires on the stored
   control trajectories before step 2100.
5. Every run in Phases 0–2 is logged to wandb with the full config, and the arm table in
   `docs/ablation-ledger.md` is updated.

## Risks

- **Phase 0.4 may fail.** If the run still does not replicate after `bag_mean`, the
  attention-backward atomics are a day of work and may still not close it, in which case
  the whole programme has to be redesigned around rates rather than trajectories. That is
  Phase 3, and it is why Phase 3 exists.
- The deterministic `bag_mean` agrees to bf16 epsilon per call, but 20 000 steps of a
  different rounding path is not the same model. Any comparison across the change needs a
  re-run, and the existing checkpoints are on the old path.
- Phase 2 is ~5 h of GPU at ~24 GB peak. Away-from-keyboard only [W] — the arms sit near
  31 of 32 GB with a desktop running.
- Fixing `retention_carry` changes the architecture, so it confounds arm A with every
  stored checkpoint.

## Not verified

- That fixing `bag_mean` alone makes a full run replicate. The residual 3.17e-3 may be
  enough to keep it stochastic. Phase 0.4 decides it, and nothing downstream is readable
  until it does.
- That nondeterminism **drives** the divergence rather than merely obscuring it. This is
  the hypothesis behind arm B and it is not yet evidence.
- Whether the deterministic `bag_mean` changes trained quality over a full run.
- Whether the attention-backward atomics contribute materially at all.

## Where things are (for a cold start)

- Worktree `/home/wolfe/morph-perf`, branch `perf/throughput-lever-stack`.
- venv: `/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-TUL/.venv/bin/python`, run with
  `PYTHONPATH=$PWD`.
- vlt thread `tul-divergence-root-cause` (project `subagent-comms`) carries the three
  facts a cold session needs first: `vlt thread read tul-divergence-root-cause`.
- Probes, all in `ignore/perf/`: `determinism_probe.py` (the 20/20 result),
  `bagmean_deterministic.py` (the fix + its benchmark), `step_determinism.py` (full-step
  4/4 and the 12× improvement), `depth_gain.py` (forward gain, `lin_ratio`, validity
  guard), `saturation_check.py`, `lin_ratio_directions.py`,
  `future_corruption_probe.py`, `causality_bisect.py`, `sigma_history.py`,
  `order_param.py` (the validated sigma estimator with both gates).
- Checkpoints: survivors in `checkpoints/morph/tul-a1|tul-gate/step_{5000,15000,20000}.pt`;
  diverged in `checkpoints/morph/tul-a1r/DIVERGED_step_{2080,4160}.pt`; the batch-14
  08-18 campaign in `/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-TUL/checkpoints/morph/`.
- **No checkpoint exists inside the onset window** (~1900–2040). That gap is the reason
  every measurement so far is post-mortem, and Phase 1.2 exists to close it.
- Prior art to read before acting, in this order:
  [`docs/tul-divergence-rca.md`](../../../../docs/tul-divergence-rca.md) Parts 3, 5 and 7
  (the two withdrawn claims and the placebo), then
  [`docs/experiments/failures/2026-08-23-tul-forward-backward-asymmetry.md`](../../../../docs/experiments/failures/2026-08-23-tul-forward-backward-asymmetry.md)
  (0 of 5 predictions confirmed, and why `sigma_max` has no referent post-onset).
- Pre-registration for Phase 0.4 is already written:
  [`docs/experiments/planned/2026-08-23-tul-run-replication.md`](../../../../docs/experiments/planned/2026-08-23-tul-run-replication.md).
