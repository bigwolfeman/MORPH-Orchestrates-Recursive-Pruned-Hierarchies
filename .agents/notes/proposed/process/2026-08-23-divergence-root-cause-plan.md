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

- [ ] **0.1 Land the deterministic `bag_mean`.** Replace the `index_add_` pair in
      `morph/model/tul.py:203-205` with the one-hot `bmm` formulation. Already written and
      measured in the scratchpad: **0/20 non-identical fwd+bwd**, agrees with the current
      version within bf16 epsilon (fwd 1.56e-2, bwd 3.91e-3), **10 % FASTER**
      (0.165 ms vs 0.184 ms fwd+bwd at B=12, L=1152, C=1024), +1.8 MB for the one-hot.
      EVIDENCE: pending
- [ ] **0.2 Add a determinism gate test** — `tests/`, call `bag_mean` twice on identical
      input and require `torch.equal` on both the output and the input gradient. It must
      FAIL on the `index_add_` version; check that by reverting once.
      EVIDENCE: pending
- [ ] **0.3 Audit the rest of the TUL path** for other order-dependent accumulation
      (the slot scatter-back, the TST `ve_bagged` path, `fused_ce.py:127,224`).
      EVIDENCE: pending
- [ ] **0.4 THE GATE — does the control replicate?** Run `tul_a1` at ONE seed, twice,
      byte-identical config, to step 2600, eval DISABLED so an eval pass cannot perturb
      the RNG stream. This is RCA §22's P2, prepared 2026-08-18 and never run.
      **Decision rule, pre-registered:** the two runs must agree on the onset step within
      ±25 AND on `train/loss` at step 1000 to 4 decimals. If they do, the phenomenon is
      bisectable and Phase 1 starts. If they do not, the kernel atomics are the next
      suspect and Phase 0 continues — do NOT proceed to Phase 1 on an irreproducible run.
      EVIDENCE: pending

### Phase 1 — watch the onset  (~45 min, needs 0.4)

Nobody has looked inside the onset. Every checkpoint is post-mortem, `spec/sigma` logs
every 100 steps, and the onset lasts about 140.

- [ ] **1.1 Add per-step instrumentation** behind a config flag: pre-clip per-region
      gradient norms, per-core-block gradient norms, and the **GLA carried-state norm**
      (the cross-iteration carry is a second recurrent loop and nothing watches it).
      EVIDENCE: pending
- [ ] **1.2 Re-run the control to 2600** with checkpoints every 25 steps from 1700.
      EVIDENCE: pending
- [ ] **1.3 Which quantity moves first**, at 25-step resolution, using the probes already
      written: `ignore/perf/depth_gain.py` (forward carrier gain, `lin_ratio`; note
      `sigma_max` is meaningless once `lin_ratio` collapses), and the per-block gains.
      EVIDENCE: pending

### Phase 2 — mediation, NOT another cure hunt  (~5 h, needs Phase 1)

The question that can finish this. Run control + placebo + the four known cures, 2 seeds,
with Phase 1's instrumentation, and ask **not** "did it survive" but:

> Which single logged quantity is (a) driven monotonically past some value in BOTH the
> control and the placebo, and (b) held below that value by ALL FOUR cures?

- [ ] **2.1 Write the pre-registration** in `docs/experiments/planned/` BEFORE the runs,
      naming the candidate quantities and the threshold rule. Phase 1 supplies the
      candidate list, so this cannot be written earlier.
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

- [ ] **3.1 Model the excursion rate against training step** from Phase 1's per-step logs.
      EVIDENCE: pending
- [ ] **3.2 Ship an abort criterion instead of a cause.** This is available TODAY and does
      not depend on any phase: `gradnorm/core` ratchets 0.0092 → 0.0426 → 0.1083 → 0.8996
      over steps 1800–2100, about **140 steps before** `train/grad_norm` moves, and never
      returns. It is a RATCHET, not a level — the gate arm touches 0.3462 at step 700 and
      falls back to 0.0783 without dying, so the rule must be "N consecutive rises", not a
      threshold.
      EVIDENCE: pending

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
