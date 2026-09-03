# Gates — is the core map under-determined on the slot manifold?

Result: `lab/experiments/failures/2026-08-24-tul-core-underdetermined.md`. Two offline
probes, one GPU ladder whose control is running.

- [x] A1  Per-slot embedding rows measured on BOTH slotembed arms at both checkpoints,
      CENTRED (the rows share a common mean by construction).
      EVIDENCE: 6 rows measured. Centred eff rank: `b10-slotembed` 27.15 -> 34.04,
      `s0-slotembed` 42.92 -> 42.87, `s0-stack` 22.37 -> 22.55. Centred pairwise cosine
      is −0.015 everywhere, which is exactly −1/(n−1) for 64 unstructured vectors.

- [x] B1  SELF-TEST. The FIRST form of this gate was itself wrong: `frac_g >= 0.99`
      through a 99 %-energy projector cannot fail, because that projector spans 935 of
      1024 directions. Replaced by "every energy curve reaches exactly 1.0 at k = in_dim".
      EVIDENCE: `SELF-TEST PASS` on both the slot and the token pass, 11 rungs each.

- [x] B2  Input effective rank and the g / u / m2 energy curves on all 11 rungs, per core
      MLP linear.
      EVIDENCE: 12 linears x 11 rungs. Slot eff rank 11.22 to 27.63; gap@32 +0.099 to
      +0.165.

- [x] B3  The same on the TOKEN path with the SAME weights.
      EVIDENCE: 11 rungs. Token eff rank 25.10 to 84.80; healthy-rung ratio 3.36. The
      collapse reproduces on 1024 positions: 75.16 -> 27.50.

- [x] B4  Verdict against the pre-registered predictions, failures included.
      EVIDENCE: 2 of 7 hold, and both are the weak ones. P4, P5 and P6 fail in the
      OPPOSITE direction to the prediction. Filed under `failures/`.

- [x] C1  Decision on the granularity ladder, stated with its reason.
      DECISION: run it, control first and sequentially. Reason: the token-path pass
      weakens the premise but cannot kill it — it evaluates A1-trained weights in an A0
      configuration and says nothing about a model TRAINED with finer spans. `span_cap`
      32 -> 12 is a measured 1.78x on the target quantity (14.39 -> 25.68), and
      `max_slots` 128 + `span_cap` 12 fits at batch 6 (peak 14.49 GB measured) where the
      parent document's OOM was only ever tested at batch 10 and 12, so the ladder can be
      run WITHOUT the packer's early-row-end confound.
      EVIDENCE: `g6-ctrl` launched 17:58:33, 4000 steps at batch 6. The fine arm runs
      only if this control takes over — two arms in this campaign have already been lost
      to a control that refused to fail.

- [x] T1  Unit tests for the new modules, sabotage-checked.
      EVIDENCE: `tests/test_subspace_probe.py` -> 14 passed. Five sabotages each caught:
      energy curve returns ones (2 failed), eff_rank returns dimension (2 failed), eps
      moved inside the sqrt (1 failed), eigenvectors not reordered (1 failed), rows not
      centred (2 failed). Restored: 14 passed.

- [x] T2  Whole suite green; template ok.
      EVIDENCE: `pytest tests/ -q -p no:randomly` -> `341 passed, 1 xfailed`.
      `verify_template.py: ok`.

- [x] T3  At least one thing NOT verified named in the report.
      EVIDENCE: the results file has a five-item "What this does NOT show", led by
      "nothing here is causal" and including the padding confound on the `span_cap`
      number.

8 of 8 checked. No ABANDON lines. The ladder itself is a separate, running experiment and
will be pre-registered before its fine arm starts.
