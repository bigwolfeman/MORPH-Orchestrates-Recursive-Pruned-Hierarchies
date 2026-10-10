# Agent Note: Phase 2 tracker (hinge pass, variable cell count, speed and decode)

Status: proposed

## Problem

Phase 1 (2026-10-08) made LXTUL training 2.3x faster (`lxtul_pointer_fast`, 26.7k tok/s, quality
inside the eager spread; merge 74a334b6). Wolfe (2026-10-10) set phase 2 to three open items, plus
one he raised: "I feel like we still haven't realized the actual TUL savings from decoding the
latent." His own low-hanging list is out of scope until he brings it. Work spans several notes and
agents, so one checklist tracks it. Each item links to the note that owns its detail; this file
holds only status.

## Proposal

Status keys: [ ] open, [~] in progress, [x] done (with the commit or filing), [-] dropped (with why).

### A. The gain hinge's fixed pass (Wolfe's decision)

The hinge draws its pass from the CPU generator and restores it, so it regularises ONE pass per
run, chosen by the seed (`transformer.py` `_t_gain`; filing
`lab/experiments/mixed/2026-10-08-fast2-package-paired.md`). Every winner number trained that way.

- [x] A1 Read the drawn pass directly (2026-10-10, eager `lxtul_pointer`, 50 steps, a wrapper
      logging every `torch.randint` drawn inside `_tul_core`): 54 of 54 draws are pass 6 on
      seed 1 (the winner's seed) and 54 of 54 are pass 3 on seed 2 (`n_grad_iters` 8). The pass
      is fixed per run and differs between seeds, so two seeds of one recipe regularise
      different passes. Scripts and logs: `/home/wolfe/morph-scratch/perf/hingepass/`.
- [ ] A2 Wolfe decides: keep one pass per run, a fresh pass per step, or every grad pass.
- [ ] A3 If per step: device-side draw (one-hot over passes or one graph per pass) that keeps
      `graph_step` capturable; byte gate with the draw fixed; paired check (3 pairs) against
      the fixed pass.
- [ ] A4 Interaction with C4 (skipping the inactive hinge): decide A first.

### B. Variable cell count (owner note: [`2026-10-05-lxtul-variable-cell-count.md`](../architecture/2026-10-05-lxtul-variable-cell-count.md))

- [ ] B1 Index-free cell seeding, router bias and write (the note's design blockers), M a
      per-forward argument; test that M = 4 reproduces the fixed-4 forward with matched seeds.
- [ ] B2 Train-time draw of M that keeps `graph_step`: one graph per M value or a masked M_max
      layout; memory at M_max on the 5090.
- [ ] B3 Prereg + two 5k arms (fixed 4, Poisson 4), both on `lxtul_pointer_fast`.
- [ ] B4 Inference sweep M in {1, 2, 4, 8, 16, 32, 64, 128}: K1-K6, CE, router pick vs oracle,
      cell diversity, tok/s. Filed.

### C. Speed and decode (owner note: [`2026-10-08-lxtul-training-speed-plan.md`](../architecture/2026-10-08-lxtul-training-speed-plan.md))

- [~] C1 Realise TUL's decode saving ("think once, decode cheap"): the loop runs once per span,
      so per-token decode cost should be the prelude + coda path, far below a model that loops
      every token. Today the KV-cached generators refuse the fan (`tul_generate_cached.py`
      `_check_supported`: `tul.fan_k > 0`), so LXTUL decodes with the eager recompute-per-step
      generator. Steps:
  - [~] C1a Per-token compute accounting (train and decode) for LXTUL, plain MORPH and the Parcae
        testbed, and the minimal set of features the cached generator must learn for
        `lxtul_pointer` (research agent, 2026-10-10).
  - [ ] C1b Decode bench today: eager LXTUL vs cached plain vs Parcae, tok/s at batch 1 and batch N.
  - [ ] C1c Cached + graphed LXTUL decode, bit-equal tokens to `generate_tul` (the existing
        cached generators' contract). Design so B's variable M fits.
  - [ ] C1d Any training-side saving from the same idea (train-only span decoder rows, slot rows
        through prelude/coda), from C1a's numbers.
- [ ] C2 LM-head slot rows (Lemma 3: needs `pointer_cell_key: false`; the coda keeps K/V at slot
      rows). About 0.34 TF per step.
- [ ] C3 Drop the logits recompute GEMM in the token head (keep bf16 logits, about +0.76 GB).
- [ ] C4 Skip the hinge backward on inactive steps (71 % of steps; needs a CUDA conditional node
      under `graph_step`). After A.
- [ ] C5 The 4-stream HC carrier: 1 stream was worth 60 ms per step at FAST. An architecture
      change, so it needs a quality arm, not only a speed bench.

### D. Literature: JEPA-Anything (arXiv 2609.20800)

- [~] D1 Read the paper against MORPH's existing implementation of its method (`tul.fan_opf`,
      arm F, `tul.py` ~596) and that arm's filed results; say what, if anything, is untested
      (research agent, 2026-10-10). Wolfe's prior: an optional later ablation on the latent
      loss, not the base.

## Alternatives considered

- **Track in a hosted page with a shared database.** Visible from any device, but agents and
  future sessions read the repo, and a second copy of the status drifts. Kept in the repo.
- **One note per item with no tracker.** The owning notes exist for B and C; without one list,
  the order constraints (A before C4, C1c designed for B) are not written anywhere.

## Acceptance criteria

- Every box is [x] or [-] with a commit, filing or reason.
- Each experiment in it has a prereg in `docs`/`lab` experiments before its run.
- The note moves to `implemented/` (or `archived/`) when phase 2 closes.

## Risks

- B and C1c both change the fan's shapes; building C1c for fixed M first would need rework.
- C5 changes the architecture; its speed gain is measured, its quality cost is not.
