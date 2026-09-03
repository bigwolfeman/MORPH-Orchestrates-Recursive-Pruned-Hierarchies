# Gates — optimizer-state decomposition and the coherent-drift severity measure

Task: the two zero-GPU-training steps ranked first by the second opinion.
Step 1 — decompose the AdEMAMix update at rungs 1625..1866 into `g/sqrt(v)` and
`alpha*m_slow`, projected onto dW.
Step 2 — compute the severity measure this work should have used: post-optimizer
coherent core drift, and re-score the arms with it.

Both read checkpoints that already exist. No training run was launched by this task.
Result: `lab/experiments/failures/2026-08-24-tul-optimizer-state-decomposition.md`.

- [x] G1  Index -> name map for the optimizer state is EXACT, not inferred.
      CHECK: `optstate_probe.py --self-test --ckpt .../ROLL_step_1850.pt`
      EVIDENCE: `MAP_OK 487/515 state entries, 28 parameters carry no state`, rc=0.
      The 28 are the CSA indexer projections, which `attention.py` documents as
      grad=None by design; verified separately as 7 blocks x 4 params.

- [x] G2  Dequantisation verified, not reimplemented by eye.
      EVIDENCE: `independent reconstruction of 5767168 codes: relative max error
      0.000e+00`; `dequant->quant->dequant residual: 3.321e-03 relative`; `DEQUANT_OK`.
      Cross-check: implied per-coordinate gradient RMS 5.045e-05 against 5.912e-05
      predicted independently from a 286.1M-param gradient clipped to global norm 1.0.

- [x] G3  Sabotage check: the self-test FAILS when the map or the dequant is broken.
      EVIDENCE: `sabotage=none exit=0`, `sabotage=map exit=1`, `sabotage=dequant exit=1`.

- [x] G4  Step 1 measured on ALL 11 onset rungs, per region, alpha from each
      checkpoint's own step through the optimizer's own `_sched`.
      EVIDENCE: 11 rows in the results table, `alpha=3.50` on every rung (past the
      `t_alpha` 1600 ramp, capped at 3.5). Core slow/fast 0.1751 -> 0.2370.

- [x] G5  Step 2 measured on the consecutive rung intervals: ||dW_core||, the
      directional autocorrelation, and their product.
      EVIDENCE: 10 dW intervals, 9 with `ac`. Range +0.4341 to +0.5738; composite
      spread 1.577x. The measure does not separate — reported as a FAILURE, not
      omitted.

- [x] G6  The arms re-scored by the optimizer-state measure.
      EVIDENCE: 10 arms, 23 checkpoints. 7 clean arms ranked, 3 penalised arms ranked
      separately. `b10-slotembed-nojit` has NO checkpoint (the run was stopped before
      step 2000) and is reported absent rather than silently dropped.

- [x] G7  Verdict stated against the pre-registered predictions, including failures.
      EVIDENCE: scorecard of 8, ONE holds. P8 was pre-declared decisive and failed.
      Filed under `failures/`.

- [x] G8  Unit test for the new module, and it fails when the module is broken.
      CHECK: `pytest tests/test_optstate_probe.py -q -p no:randomly`
      EVIDENCE: `19 passed`. Four sabotages each caught: ignore eps_inside (1 failed),
      noncore includes core (1 failed), unknown layout returns zeros (1 failed),
      region_of ignores _orig_mod (2 failed). Restored: 19 passed.

- [x] G9  Whole suite still green, and the shared-build refactor did not break
      `jac_ladder.py`.
      EVIDENCE: `pytest tests/ -q -p no:randomly` -> `327 passed, 1 xfailed` (was 308
      passed, 1 xfailed; +19 new). End-to-end rerun of `jac_ladder.py` on
      ROLL_step_1850 reproduces the pre-refactor `rms=2.5234`, `sigma=1127.648`.

- [x] G10 Pre-registration committed BEFORE the run, method amendments dated, results
      filed, parent document corrected in place.
      EVIDENCE: `f9a712a` pre-registration, `54e5517` Method amendments 1 and 2,
      `816edfa` results. `python scripts/verify_template.py` -> `ok`.

- [x] G11 At least one thing NOT verified is named in the report.
      EVIDENCE: the results file has a six-item "What this does NOT show", led by
      "the coherence measure is post-hoc and untested out of sample".

11 of 11 checked. No ABANDON lines.
