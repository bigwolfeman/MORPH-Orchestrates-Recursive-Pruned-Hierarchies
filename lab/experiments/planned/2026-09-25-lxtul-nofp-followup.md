# Planned: does the fixed-point-off positive replicate, and can a weak term keep it stable

Status: planned

Date: 2026-09-25 09:21 (frozen before either arm's first GPU step).
Parent: [`../successes/2026-09-24-lxtul-stage3-map.md`](../successes/2026-09-24-lxtul-stage3-map.md).
Wolfe's go: 2026-09-25 ("Yes queue both.").

## Question

With `core_fixed_point_lambda 0` the strict e4probe slot loop earned coda K1−K6 +0.0123
[+0.0115, +0.0130] (seed 1), the code rollouts separated 3.47 at depth 6 (0.27 with the
term at 1.0), and the run detonated once at step 4048 and recovered. (1) Does it replicate
at a second seed? (2) Does the term at 0.1 keep the stability hold without stopping the
motion that earns?

## Hypothesis

The term at 1.0 charges the pass-to-pass motion the loop needs; the positive is the loop
moving, so it replicates. A tenth of the term is small next to the coda's pull on the
motion and still large enough to damp the single-slot excursion.

## Arms

5000 steps from scratch, one trainer at a time.

| arm | config | changes from `lxtul-e4probe-nofp` |
|---|---|---|
| `lxtul-e4probe-nofp-s2` | `tul_slot_spandec_strict_e4probe_nofp_s2.yaml` | `training.seed` 1 → 2 |
| `lxtul-e4probe-fp01` | `tul_slot_spandec_strict_e4probe_fp01.yaml` | `model.core_fixed_point_lambda` 0 → 0.1 |

References on disk: `lxtul-e4probe-nofp` (seed 1) and `lxtul-e4probe-map` (arm C, term 1.0).

## Instruments

The Stage 2 scorer with `--ref nofp` and arm C beside it (480 rows, forced depths 1..6):
coda mix CE, K1−K6, exit separation, width gain, paired CIs. `core_map_fd.py` fp32.
Sweeps, notul pairing. The sustained tripwire on each probe file.

## Predictions

nofp-s2:

- **S-1 (replicates).** Coda K1−K6 >= 0.005, CI clear of zero: **65 %.**
- **S-2 (same size).** Coda K1−K6 in [0.0083, 0.0163] (seed 1 ± 0.004): **45 %.**
- **S-3 (the codes move).** Exit separation at depth 6 >= 1.0: **65 %.**
- **S-4 (stable).** Tripwire HEALTHY: **45 %.**

fp01:

- **F-1 (still earns).** Coda K1−K6 >= 0.005, CI clear of zero: **45 %.**
- **F-2 (stable).** Tripwire HEALTHY: **70 %.**
- **F-3 (the coda gains).** fp01 coda mix @6 − nofp (seed 1) <= 0: **45 %.**
- **F-4 (the codes move).** Exit separation at depth 6 >= 1.0: **45 %.**

## Verdict rules

The positive replicates if S-1 holds. The follow-up is a success if S-1 holds; the recipe
question is answered YES if F-1 and F-2 both hold (a loop that earns and does not
detonate). If S-1 fails, the Stage 3 positive is filed as seed-dependent in the ledger.

No clause passes on a cosine.

## Method

Chain in `/home/wolfe/morph-wt-map` at the commit that adds this file: 30-step smokes,
nofp-s2, fp01, then `core_map_fd`, the scorer, sweeps and the notul pairing. Artifacts:
JSON and `.runlog.txt` files in `../results/2026-09-25-lxtul-nofp-followup/`.
