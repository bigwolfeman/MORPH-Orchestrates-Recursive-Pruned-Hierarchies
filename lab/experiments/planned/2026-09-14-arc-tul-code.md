# Planned: TUL-Code — the slot holds a ground-truth code of the span it precedes, and the core body samples it

Status: planned

Date: 2026-09-14 (frozen before any code exists; this file is committed before the contracts
are written and before any GPU step). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Spec:
[`docs/tul-code-spec.md`](../../../docs/tul-code-spec.md) (the contract; this file owns the
NUMBERS). Design note:
[`2026-09-14-tul-span-code.md`](../../../.agents/notes/proposed/architecture/2026-09-14-tul-span-code.md).
Paper readings: `docs/references/tul-latent-emission/{ladir,ltm,diffusion-forcing}/`.

## Question

Every slot-loop arm since 2026-09-04 trained the slot state only through the next span's
token loss, and every one read a flat loop, a rank-6 state and a probe that scored the
state at entry as well as at exit. Wolfe, 2026-09-14: "if I just had ground truth latent
values I could do anything." TUL-Code gives every slot a ground-truth code: an encoder E
reads the span the slot precedes and writes its code into the cells; the strict coda speaks
the span reading the code plus the span's own tokens; the core body is trained, one pass
per slot, as a flow-matching velocity field from noise toward that code, and samples the
code in k passes at inference. **Three questions, in order: (1) is the code a code the
mouth reads, (2) does the sampler improve with k, (3) does a sampled code beat the
deterministic loop write on the same geometry, at a compute the d1 rung cannot match.**

## What is already known

- The strict geometry makes the loop's write the whole cross-span channel: 0.1865 nats at
  CE parity with the bypass arm (`failures/2026-09-12-arc-strict-geometry.md`, verdict 1).
  The reader problem that killed FM1 / FM1-CW / FM2 (worth 0.0000 on an additive detached
  prefix with tokens in reach) is not present on this geometry.
- There is 0.3994 nats [0.3838, 0.4162] of cross-span information at 5k, 0.3149 of it flat
  at offsets 8+ (`failures/2026-09-11-arc-span-budget.md`). That is the ceiling on what the
  code channel can be worth here.
- The slot family's state reads effective rank 5.7–7.3 of 1024, pairwise cos 0.72–0.77; the
  Thought Register 1.24 of 4 (`failures/2026-09-13-arc-thought-register.md`).
- The linear probe scores the slot state at ENTRY at AUC 0.60–0.64 (null 0.51), exit no
  higher (`failures/2026-09-12-arc-latent-z-gradient.md`, Step 0).
- The d1 rung (12 block passes per token) sits +0.0674 behind the Poisson-6 plain model at
  20k on paired rows; the plain model's own K1−K6 is 0.170 (`2026-09-13-arc-depth-ladder-ship.md`).
- The FM planner line's terminal reading (P1 on l2cap: trained ≈ untrained ≈ shuffled) and
  its binding rule; Wolfe waived the rule for this design on 2026-09-14 (note, "The binding
  rule").
- Rates: `slot-spandec-strict` on the queue runs at the panel shape (seq 1024, batch 12);
  its rate and resident memory are the comparison for the smoke's numbers.

## Method

Arms, all composing `tul_code.yaml` (spec §9) unless named otherwise, seq 1024, batch 12,
20,000 steps, `training.ademamix_t_beta3` pinned to 20000, wandb `morph-tul`, one seed
each in this panel (seed 1); a second seed of `tul-code` and of the ruler before any claim.

| arm | config | one factor |
| --- | --- | --- |
| `tul-code` | `tul_code.yaml` | the design, spec defaults (`code_noise 0.5`, phase2 0.10, phase3 0.50, rollout p 0.5 at 8 steps) |
| `tul-code-seeddetach` | `tul_code_seeddetach.yaml` | the FM gradient may not reach the seed path (Wolfe: an ablation) |
| `tul-code-nophase3` | `tul_code_nophase3.yaml` | `code_phase3_at: 1.0` (rollout never starts) — LaDiR's stage-2 ablation on our data |
| `slot-spandec-strict` | exists | the deterministic loop write on the same geometry: the MEAN control |
| `notul-d1` (`norm-match-20k-d1`) | exists, done | the compute bar |
| `norm-match-20k` | exists, done | the plain looped ceiling on CE |

Before any queue entry: contracts C1–C10 of the spec green on CPU
(`OMP_NUM_THREADS=2 nice -n 19 pytest tests/test_tul_code.py -q`, count reported); every
config composed through Hydra + `tul_setup` at the queued commit; a 12-step GPU smoke of
`tul_code_smoke.yaml` on the Spark with exit 0, `fm/rel` finite, `val/code_eff_rank`
printed, resident memory and tok/s recorded here under "Not verified before launch".

Instruments (spec §8), all on the 480 paired rows at 5k, 10k, 15k, 20k: `ce_tf`,
`ce_k{1,2,4,8,16}` (teacher-forced tape), `ce_k8_rolled`, `code_gap`, `code_eff_rank`,
`code_pairwise_cos`, `fm/rel` per band, the cells' shuffle cost, `ce_k8` by span-length
bucket; generation at k ∈ {1, 8} with rep4 / distinct-3 at 5k and 20k. Paired gaps by
`span_budget_profile.py` against the ruler and the two plain rungs; never the runner's
Final val_loss.

Phases: phase 2 starts at step 2,000, phase 3 at step 10,000 (spec defaults × 20k).

## Predictions (frozen)

Numbers are nats on the 480 paired rows unless stated. "The ruler" is `slot-spandec-strict`
at the same step.

- **P-1 (G1, sanity: the code is a code and it is read).** At 5k on `tul-code`: `ce_tf` at
  least 0.15 below the ruler; the cells' shuffle cost in the `ce_tf` regime at least 0.15;
  `code_eff_rank ≥ 16` of 1024. Prior 80 %. Failing P-1 means the mouth does not read the
  cells or E writes a constant; nothing below is readable and the panel stops after the
  first arm.
- **P-2 (G2: the sampler earns with k).** At 5k: `ce_k1 − ce_k8 ≥ 0.02`, `ce_k8 − ce_k16`
  within [−0.005, +0.02] (diminishing, not reversing), `fm/rel` below 0.9 in every t-band.
  Prior 60 %. My expectation is a curve that flattens between k = 4 and k = 8.
- **P-3 (the code is coarse, not the span).** `code_gap = ce_k16 − ce_tf` at 5k in
  [0.05, 0.35]. Below 0.05 says the code carries nothing the past cannot guess (the noise
  is too strong, or E collapsed onto the seed); above 0.35 says the code carries the span
  verbatim and the sampler cannot follow. Prior 55 %.
- **P-4 (G3: a sample beats a mean).** At 20k: `ce_k8` below the ruler, CI excluding 0.
  Prior 45 %. The honest alternative is that a mean suffices on web text at this horizon,
  which is LCM's result with a token path added, and it is a real possibility.
- **P-5 (the rolled tape holds).** At 20k: `ce_k8_rolled − ce_k8` within [0, 0.05] on
  `tul-code`, and larger by at least 0.02 on `tul-code-nophase3`. Prior 55 %. This is the
  LaDiR stage-2 claim on our data.
- **P-6 (G4: the ship bar).** At 20k: `ce_k8` within 0.02 of `notul-d1` (12 passes per
  token) at 6 + 8 = 14 block passes per token counted per position, i.e. the code model is
  at least as good as the d1 rung at near its compute. Prior 35 %. The plain looped model at
  42 stays 0.05–0.07 ahead; matching IT is not predicted.
- **P-7 (seed-path gradient).** `tul-code-seeddetach` vs `tul-code` at 20k: within
  ±0.01 on `ce_k8`. Prior 60 %. A larger gap in either direction decides the default.
- **P-8 (rate and memory).** `tul-code` runs at ≥ 1.3× the ruler's tok/s (one thinker pass
  per slot instead of a Poisson-6 loop on the slot positions) at resident memory ≤ the
  ruler's. Prior 70 %. The smoke measures it before launch; a miss here amends Method, not
  Predictions.
- **P-9 (no collapse over training).** `code_eff_rank` at 20k ≥ its 5k value − 4, and
  `code_pairwise_cos` ≤ 0.6 at every eval. Prior 60 %. The slot family's 0.72–0.77 is the
  number to beat.

## Binding

- P-1 fails ⇒ stop the panel, file under `failures/`, the next step is the reader
  (geometry or E), not the sampler.
- P-1 holds and P-2 fails on two seeds ⇒ the thinker is not integrating; file under
  `failures/`; the design note moves to `rejected/`.
- P-1, P-2 hold and P-4 fails on two seeds ⇒ multimodality does not pay on web text at
  this horizon; file under `failures/` with the K-curve as the finding; the next arm is the
  unrolled-trajectory phase 3 (the spec's follow-on), preregistered separately.
- P-1, P-2, P-4 hold ⇒ `successes/`; the note moves to `implemented/` once `tul_code.yaml`
  is on master with C1–C10 green; P-6 decides whether it is the ship candidate.
- No run beyond 20k without Wolfe's word. No PonderNet / ACT. k is chosen from the K-curve
  and stated; never tuned on the paired rows it is then reported on.

## Not verified before launch

(To be filled at launch: pytest count and exit code, the composed configs, the Spark smoke
line with rate and memory, the commit the runner checks out.)

## Results

(Empty until the runs finish.)
