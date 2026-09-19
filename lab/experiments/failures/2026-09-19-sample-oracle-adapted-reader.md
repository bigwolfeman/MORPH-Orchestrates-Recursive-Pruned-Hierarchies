# Sample-oracle gate, second reading: the same draws under a reader trained on the cell

Status: failure

Date: 2026-09-19. Instrument: `lab/divergence/sample_oracle_probe.py` (99bd7c9 plus the
code-target exit site added in this change). Host: Spark. Arm: `tul-code-target-uf` @ 30000,
the only checkpoint in the tree whose coda was trained on the slot loop's own cells
(the-reader-was-the-limit, 2026-09-18). Written before the probe ran on this checkpoint.

## Question

The first reading (`../failures/2026-09-18-sample-oracle-gate.md`) closed the exploration
branch on the strict ruler, and its updated hypothesis put the cause on the READER: at
sigma 1.0 the sixteen exits differ by cosine 0.72 after six passes and the frozen coda's CE
moves by +0.006 on average and −0.030 at best-of-16. If the reader is the limit, the same
draws under a reader that was trained on the cell should move the CE far more, and the
question the gate was built to ask (does the loop turn entry variation into alternatives
the reader can use) becomes readable. If the adapted reader is as indifferent as the frozen
one, the cell's direction carries nothing usable and the branch is closed reader-independently.

## Hypothesis

The adapted coda is sensitive to the cell's direction (it values the cell at +0.177 nats
against −4.6 for the frozen one), so every perturbation now costs CE and every oracle now
recovers more. The loop still adds nothing: exit noise recovers at least as much as entry
noise, and the entry gain does not grow from depth 1 to depth 6. The recovered gain stays
under the branch-opening bar because the alternatives sampling finds are noise around one
answer, not competing continuations.

## Predictions (frozen)

Same notation as the first reading. Grid: entry noise sigma {0.3, 1.0} x depth {1, 6};
exit noise (at the code projection's input) sigma {0.3, 1.0} x depth 6; N = 16; 96 rows.

- **P1 (the reader now reacts).** Mean-over-samples minus `det` at entry sigma 1.0, depth 6
  is `>= +0.05` nats (frozen reader: +0.006). At sigma 0.3: `>= +0.01` (frozen: +0.0005).
- **P2 (the oracle recovers more, still small).** Entry `gain(16)` at sigma 0.3, depth 6 in
  `[0.02, 0.06]` (frozen: 0.011); at sigma 1.0 in `[0.04, 0.12]` (frozen: 0.025).
- **P3 (reader, not loop).** Exit `gain(16) >= entry gain(16)` at both sigmas.
- **P4 (no growth with depth).** Entry `gain(16)` at depth 6 minus depth 1 within
  `[-0.01, +0.01]` at both sigmas.
- **P5 (the loop's geometry is the same).** `cos_exit` within 0.05 of the strict ruler's at
  each sigma (0.96 at 0.3, 0.72 at 1.0): the reader changed, the loop did not.

Branch-opening rule, unchanged: entry `gain(16) >= 0.05` at sigma <= 0.3 AND `cos_exit <
0.90` at that sigma AND growth from depth 1 to 6 `> 0.02`. All three, or the LXTUL fan4
queue lines come out.

## Method

- Checkpoint `tul-code-target-uf/step_30000.pt` (md5-checked copy on the Spark), config
  `tul_code_target_uf`, `model.use_kernels=false`. The coda, its `x0_injects`, `final_norm`
  and `lm_mixer` were trained 20k→30k with everything else frozen (record:
  `../successes/2026-09-17-lctul-target-unfreeze.md`).
- A code-target arm replaces the `prefix_project` write with `TULCodeProj`'s cells, so the
  exit site is the projection's INPUT (the readout of `h_slots`); the probe selects it
  automatically and records `exit_site: code_proj`. The projection is also called for
  per-pass readings; the spy keeps the first call (the write) and the later calls carry no
  CE. The probe RAISES if the exit hook never fires.
- Everything else as in the first reading: `core_init` entry hook, relative-RMS noise at
  valid slots, depth forced through `slot_mean_depth`, per-span attribution, row bootstrap.
- Smoke at 8 rows / 2 samples first; the 96-row run only after the smoke exits 0 and prints
  `exit site = code_proj`.

## Risks

- The `_readout(h_slots)` input may not be `[B, S, ...]`; the hook asserts nothing about
  shape beyond what `relative_noise` needs, so a wrong layout would raise in the smoke.
- The adapted reader was trained on the DETERMINISTIC exit distribution; noise at sigma 1.0
  is off its training distribution too. P1 is a prediction that it reacts, not that it
  reacts helpfully; the oracle (P2) is what says whether any sample is better.

## Results

Run 2026-09-19 on the Spark, `tul-code-target-uf/step_30000.pt`, 96 rows, N = 16, seed 0,
`exit site = code_proj` (the probe printed it; the JSON records `exit_site: code_proj`).
Artifacts: `../results/2026-09-19-sample-oracle-adapted-reader/` (`sample_oracle_uf_30000.json`,
`.txt` log). Scored with the scratch scorer; the same scorer reproduces the first reading's
scorecard on the ruler's JSON. Intervals are the JSON's row bootstrap (96 units, 2000 draws, 95 %).

| cell | det | mean − det | gain(16) [lo, hi] | cos_exit | ruler (frozen reader): mean − det | gain(16) | cos_exit |
|---|---|---|---|---|---|---|---|
| entry σ0.3 d1 | 4.0788 | +0.0009 | +0.0158 [0.0151, 0.0166] | 0.979 | +0.0007 | +0.0117 | 0.954 |
| entry σ0.3 d6 | 4.0728 | +0.0005 | +0.0149 [0.0144, 0.0156] | 0.985 | +0.0004 | +0.0109 | 0.962 |
| entry σ1.0 d1 | 4.0788 | +0.0102 | +0.0403 [0.0387, 0.0422] | 0.799 | +0.0076 | +0.0251 | 0.670 |
| entry σ1.0 d6 | 4.0728 | +0.0073 | +0.0388 [0.0372, 0.0408] | 0.849 | +0.0057 | +0.0246 | 0.720 |
| exit σ0.3 d6 | 4.0728 | +0.0005 | +0.0130 [0.0124, 0.0137] | (isotropic) | +0.0008 | +0.0144 | |
| exit σ1.0 d6 | 4.0728 | +0.0056 | +0.0334 [0.0319, 0.0352] | (isotropic) | +0.0106 | +0.0296 | |

Scorecard:

- **P1 FAIL.** Mean − det at entry σ1.0 d6 is +0.0073 (bar +0.05); at σ0.3 d6 +0.0005 (bar
  +0.01). The adapted reader shifts by about as little as the frozen one (+0.0057 / +0.0004).
- **P2 FAIL, both cells just under the low edge.** gain(16) at σ0.3 d6 is 0.0149 (band
  [0.02, 0.06]); at σ1.0 d6 0.0388 (band [0.04, 0.12]). Both are 1.4 to 1.6x the frozen
  reader's, with disjoint intervals, so the reader does recover more; the prediction put the
  factor at 2 to 5.
- **P3 FAIL, and the sign is real.** Exit gain is BELOW entry gain at both sigmas
  (0.0130 vs 0.0149; 0.0334 vs 0.0388), intervals disjoint. On the frozen reader it was above.
- **P4 HOLDS.** Growth from depth 1 to 6: −0.0008 at σ0.3, −0.0015 at σ1.0.
- **P5 FAIL at σ1.0.** cos_exit 0.849 against the ruler's 0.720 (gap 0.13, bar 0.05); at σ0.3
  0.985 against 0.962 (holds). The code-target arm's loop pulls perturbed entries closer
  together than the ruler's does.

Branch-opening rule: entry gain(16) at σ0.3 is 0.015 (bar 0.05), cos_exit 0.985 (bar < 0.90),
growth −0.0008 (bar > 0.02). None of the three. The LXTUL fan4 lines leave the queue
(parked in `morph-scratch/arc/recon_parked_fan4.txt`, restorable by appending them back).

## Verdict

Failure: three of five predictions failed, and the branch-opening rule failed on all three
clauses. The hypothesis was half right. The reader recovers more than the frozen one
(P2's direction), but only by 1.5x, and the mean shift is unchanged, so "the reader was the
limit" does not carry over from the cell's VALUE (+0.177 nats when read by a trained coda)
to the cell's DIRECTION under noise: a 0.3-RMS isotropic perturbation of the exit state costs
this reader 0.0005 nats. The gate's own question has a clean answer on both readers: sampling
at the entry does not open alternatives the reader would use, and the loop does not turn
entry variation into more of them with depth (P4 holds twice, growth negative both times).

Two things the method could not distinguish, both recorded so the next design does not
repeat them:

1. Isotropic noise versus noise in the read subspace. `TULCodeProj` is a full-rank
   identity-initialised map (`tul_code.py::TULCodeProj`), so the site is not the issue, but
   the coda may read a low-dimensional part of the cell and isotropic noise in 1024
   dimensions puts little energy there. The probe cannot say which it is. A perturbation
   along the reader's own gradient direction (or the code's principal axes) would.
2. Sampling versus exploration. Every draw here is one noise vector at one site. A
   per-pass noise schedule (the LTF / GRAM shape) was not run and this reading does not
   speak to it.

## Updated hypothesis

The measured direction of P3 is the finding. Entry noise at σ1.0 leaves the exits at cosine
0.849 and recovers 0.039 nats; isotropic exit noise at σ1.0 leaves them at about cosine 0.71
(1/√2 for relative RMS 1) and recovers 0.033. Less spread, more recovery: the loop maps an
isotropic perturbation onto directions the reader values more than the same amount of
isotropic spread at the exit. That is what a power iteration does (E7: one pass sets the
scale, the rest rotate onto a rank-9 dominant subspace), and it is the opposite of
exploration. The map is contractive on perturbations (cos_exit rises with depth on both arms,
0.799 → 0.849 here, 0.670 → 0.720 on the ruler) and it is MORE contractive on the arm trained
to regress onto a conditional mean (P5). A contractive loop cannot hold K alternatives apart
from one perturbed entry; anything that keeps alternatives apart has to be re-injected per
pass or held in separate streams with a repulsion, which is the LXTUL fan4 design's premise
and the reason it is parked and not deleted. Before that arm earns a GPU slot, the cheaper
test is a per-pass noise schedule on the ruler (noise at every pass, same N and sigma, same
oracle), which the existing probe can do with one more hook site.
