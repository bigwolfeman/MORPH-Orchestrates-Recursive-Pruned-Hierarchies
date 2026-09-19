# Sample-oracle gate, second reading: the same draws under a reader trained on the cell

Status: planned

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
