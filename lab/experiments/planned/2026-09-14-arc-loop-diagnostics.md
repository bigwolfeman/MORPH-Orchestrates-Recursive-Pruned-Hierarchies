# Planned: four zero-training diagnostics from the lit-mining batch (AA score, spectral
gap, sink mass, basin map)

Status: planned

Date: 2026-09-14. Built from
[`docs/references/looping-depth/2026-09-13-lit-mining/`](../../../docs/references/looping-depth/2026-09-13-lit-mining/)
(`00-brief.md`, `A_theory.md` §3-4, `F_scale.md` §5) — the four "concrete arm for us"
instruments those four papers proposed, none previously run on this tree.

## Background

The lit-mining brief's headline numbers (all norm_match, paired bootstrap): the slot loop
earns 0.0003-0.002 nats on tokens across eleven-plus arms (K1-K6), K3-K6 is ~0 or
negative, one pass sets the scale and the rest rotate (rank of the update 181 -> 9),
consecutive-pass cosines are -0.2 to -0.6 (cancellation), every per-pass target given is
met in one pass. The four papers each give a DIFFERENT mechanistic reason iteration could
still matter (or legitimately not), and each gives one cheap instrument this tree has
never run:

1. **Anil et al. 2211.09961** (path independence) — Algorithm 1, the Asymptotic Alignment
   (AA) score: does the loop converge to the SAME place regardless of where it starts?
2. **Wu/Zhang/Cao 2606.00605** (power method) — the per-pass map's top-two-eigenvalue
   ratio predicts how many iterations are needed; measure sigma1/sigma2 per row and
   correlate against that row's own K1-K6.
3. **Wang et al. 2609.01343 (SMELT)** §6.4 — a second visit through repeated layers
   reduces attention-sink mass even when the loop earns nothing in nats; measure sink mass
   per pass.
4. **Lai et al. 2609.04963** (fractal basins) — Appendix B's basin map: perturb the
   entry state on a 2D slice, decode every iteration, record settling time; a flat field
   is their own "pre-bifurcation, purely contractive" signature, not a null result.

## Question

Do any of these four INDEPENDENT instruments find structure the K-curve, rank and cosine
instruments already on this tree cannot see — or do all four agree with the existing
"one pass does everything" reading, from four different angles?

## Method

Two checkpoints, both at step 5000, CPU-only build (`model.use_kernels=false`,
`model.tg_scoped_kernels=false` — the TG-scoped Triton path does not know about the
monkey-patches these instruments make; eager everywhere, correctness over speed on an
eval-only script):

| label | config | what it is |
| --- | --- | --- |
| `strict` | `tul_slot_spandec_strict` | the slot loop, strict TG geometry (the slot is the ONLY cross-span channel), span-decoder target |
| `plain` | `notul_norm_match_20k` | the plain looped core, Parcae noise entry (`core_state_init: noise`), no TUL |

Four scripts, `lab/divergence/{aa_score,jacobian_gap_vs_k,sink_mass_per_pass,basin_map}.py`,
each `--ckpt LABEL=CONFIG=PATH`, `--rows`/`--pairs`, `--device`, `--out <json>`, sharing
loader plumbing in `lab/divergence/_diag_common.py` (checkpoint load, row packing via
`_rows.py`'s stream-index-preserving cut, forced-depth per-row token CE, a QR-based
top-2 Jacobian power iteration, a scipy-free Spearman rho). Every instrument that rebuilds
the front calls `model._tul_tg_kwargs(layout)` first (2026-09-13's own bare-front bug).

96 rows for instruments 1-3 (validation stream, `skip_samples=0`, the probe family's
convention — not the trainer's held-out shard); 6 (row, slot) pairs, 41x41 grid, for
instrument 4. Full run plan:

* **AA score** — run the loop from its real entry to the model's own eval depth
  (once-converged), then again from (a) a DIFFERENT row's converged state (batch-rolled)
  and (b) `N(0, e.std()^2)` matched to the entry's own scale (twice-converged), same
  context both times. Cosine in `model._readout` space, over `layout.slot_valid` (slot
  loop) or every token position (plain — the plain entry, `_NoiseInit`, already draws
  independently per position, so per-position is the natural granularity there).
* **Jacobian gap vs K** — per ROW (batch=1): top-2 singular values of the pass map
  (`_apply_core_step` at iteration 0) via `_diag_common.jacobian_top2` (subspace/orthogonal
  iteration on `J^T J`, QR-reorthonormalised every step, Rayleigh-quotient readout — a
  hand-rolled single-vector Gram-Schmidt deflation was tried first and measured to
  converge sigma2 onto sigma1 on a synthetic 5/3 spectrum;
  `tests/test_loop_diagnostics.py::test_jacobian_top2_recovers_known_singular_values`
  pins the fix), restricted to the row's own active positions (`morph/training/
  core_jacobian.py`'s double-backward `Jv`/`J^Tw` identity). Paired with the SAME row's
  K1-K6 (forced-depth CE, `core_depth_sweep.py`'s per-row unit).
* **Sink mass per pass** — the window-branch hook `morph/model/attn_lift.py` already
  uses for `val/attn_slot_mass`, adapted to bucket by PASS (the loop calls every core
  layer once per iteration in a fixed order, so call-index // n_core is the pass number)
  and to read mass on ONE sink position per row (compact index 0 — the loop attends the
  GATHERED compact slot sequence, not the row's L_total positions; token position 0 for
  the plain core).
* **Basin map** — perturb a slot's ENTRY state (`core_init(e)`'s output at that slot)
  along two random orthonormal directions scaled by the entry's own norm; force depth
  1..T; decode the span decoder's first-token argmax (`tul_spandec.decode` at J=1, z
  only, no leakage). The plain model has no spandec and no slots, so its basin map
  perturbs a TOKEN position's entry and reads the ordinary tied LM head's argmax there —
  NOT the paper's setup verbatim, the closest analogue this model family has, reported
  as such and never conflated with the slot-loop reading.

CPU unit tests: `tests/test_loop_diagnostics.py`, a tiny MORPHTransformer fixture (slot
loop with spandec + strict geometry, and a plain noise-entry model), no checkpoint, no
tokenizer — one test per script's headline mechanism plus the shared math helpers
(`jacobian_top2` against a known synthetic spectrum, `spearman_rho`, the basin map's
settling-time/entropy). `python -m pytest tests/test_loop_diagnostics.py -q`.

## Predictions (frozen)

Wolfe's four predictions, with my own probability on each, reasoned from the four
papers' own mechanisms (A_theory.md/F_scale.md) and this tree's existing readings
(rank collapse 181->9, cancellation -0.2 to -0.6, K1-K6 <= 0.002 in eleven-plus arms) —
**not** from any number produced by these scripts. A tiny incidental smoke test (2 rows,
8 power iterations, a 9x9 grid) was run against both real checkpoints on CPU purely to
catch config/plumbing bugs before spending Spark time; its numbers are NOT used below and
are not quoted in this section (see "Not verified before launch").

- **P-1 (AA score, strict > 0.9, plain lower).** *Prior: 55%.* The strict arm's own
  numbers (rank collapse to ~9, fixed point reached by pass 6, gain 0.2-0.9, monotone
  convergence) are exactly Anil et al.'s "high-AA, informationless fixed point" profile —
  Anil's own framing (§ "a real tension with paper 1") is that path independence and an
  INFORMATIVE fixed point are dissociated, and this tree's brief already argues the strict
  loop's fixed point carries close to nothing beyond the entry. Held back from higher:
  the swap/noise re-init on THIS tree is a novel instrument nobody has run yet, and a
  strict-geometry model's loop is unusually short in wall-clock passes (attention over
  only ~64 compact slot positions), which could leave the loop still visibly
  entry-dependent at its OWN trained depth even if it would converge given more passes —
  AA is an asymptotic statement and this arm measures it at a possibly-small T. The
  "plain lower" half is the more novel bet: the plain checkpoint is Parcae's own NOISE
  entry, meaning its BASELINE once-converged run already starts from noise every time —
  its own two independent noise draws (once vs the AA-noise re-init) may look nearly
  identical by construction rather than because the loop is doing real path-dependent
  work, which would make "plain lower" wrong for a reason that has nothing to do with the
  mechanism this instrument is meant to probe. Flagged as the single biggest interpretive
  risk in this prereg.
- **P-2 (sigma1/sigma2 median > 3 on the slot loop, Spearman vs K1-K6 negative).**
  *Prior: 40%.* Wu/Zhang/Cao's mechanism predicts EXACTLY this shape if the core map's
  spectral gap is what is suppressing K-curve earning — and the tree's own `docs/
  cookbook/measuring-the-core-map.md` finding (the 2026-08-24 TUL-takeover audit) already
  reports the core blocks' amplifying directions ALIGN x2.9 across the onset, i.e. a
  narrowing effective gap is a real, previously-measured phenomenon in this exact
  architecture. What holds this below 50%: `sigma1/sigma2 > 3` is a specific numeric
  threshold picked without a prior measurement on THIS checkpoint at THIS iteration, and
  the median could easily land in [1.5, 3] and still show a real (if weaker) negative
  correlation — the DIRECTION of the correlation is where I have more confidence than
  the specific threshold.
- **P-3 (sink mass flat within 10% relative, passes 2-6).** *Prior: 75%.* This is the
  prediction most tightly coupled to instruments already on this tree: cancellation
  (-0.2 to -0.6) and near-zero K3-K6 both say passes 2-6 do almost nothing on this
  checkpoint, and "does almost nothing to the loss" is the more direct, already-measured
  claim that a flat sink-mass curve would corroborate from an independent angle (SMELT's
  own instrument, never applied here). The one way this could still surprise: SMELT's
  own finding is that sink REDUCTION can happen even while overall CE is flat (their
  "extractability, not new information" framing) — so there IS a documented precedent for
  this specific instrument moving while the K-curve doesn't, which is exactly why it is
  worth running rather than assumed.
- **P-4 (basin fields flat, fraction differing from centre < 5%).** *Prior: 70%.* Lai et
  al.'s own mechanism requires a near-bifurcation, weakly-unstable-saddle regime
  (multistable collapsing to monostable under training) to produce ANY basin structure at
  all, and this tree's own gain readings (0.2-0.9, monotone convergence to a fixed point,
  no reported instability anywhere in the slot-loop campaign) are their own paper's
  textbook description of the PRE-bifurcation, purely-contractive case where the paper
  itself predicts NO fractal structure. Held below 80%: the entry perturbation here uses
  the entry's OWN norm as the radius scale, which is a judgment call with no precedent on
  this checkpoint — a radius that is too large could push some grid points off the data
  manifold into a genuinely different (if uninteresting) regime and inflate the
  differing-fraction number for a reason unrelated to bifurcation structure.

## Binding

If **all four hold**, that is convergent evidence from four independent mechanisms (path
independence, spectral gap, attention re-weighting, basin topology) for the same
conclusion this tree already has from four OTHER independent instruments (K-curve, rank,
cosine, per-pass targets): the strict slot loop's passes 2-6 do essentially nothing on
this checkpoint, for reasons that are not "not enough iterations" or "not enough training"
but structural (no gap, no bifurcation, an already-fixed point).

If **any one fails** — especially P-2 or P-3, the two with genuine literature-documented
precedent for surprising this tree (a narrow-but-nonzero gap; sink reduction under a flat
K-curve) — that instrument becomes the next thing to chase: a failing P-2 says the map's
directions are NOT what is capping earning (contradicting the 2026-08-24 alignment
reading, worth reconciling); a failing P-3 would be the first EXTRACTABILITY-only signal
found on this checkpoint, distinct from every RELAY-shaped instrument already run.

## Not verified before launch

* **A tiny CPU smoke test against BOTH real checkpoints already ran**, purely to catch
  config/plumbing bugs (2 rows for instruments 1-3, an 8-iteration Jacobian, a 9x9 basin
  grid) before spending Spark time — NOT the frozen predictions' evidence base, and its
  numbers are not quoted here or used to set the probabilities above.
* **The AA-score noise-entry construction's calibration is unverified.** `N(0,
  e.std()^2)` matches the entry's OWN per-element std; whether that is "the same kind of
  perturbation" Anil et al. used (they used the solver's own zero/noise INITIALIZATION,
  not a re-init scaled to a converged state's own std) is a judgment call, flagged in
  P-1's own reasoning above.
* **The basin map's radius (`--radius 1.0`, one entry-norm) is unvalidated** against this
  checkpoint's own state geometry — no sweep was run to find where the field transitions
  from flat to structured, so a flat P-4 result cannot distinguish "no structure exists"
  from "the radius never reached it."
* **The plain-model basin map is a substitution, not the paper's design** (a token
  position's own next-token argmax, since there is no span decoder) — its numbers are
  reported and read separately, never averaged with the slot-loop numbers.
* **Instrument 2's Jacobian is measured at iteration 0 only** (`loop/core_gain_t0`'s own
  convention elsewhere on this tree), not averaged over the whole trajectory; a row whose
  gap opens or closes later in the loop would not be seen by this reading.
* **No wall-clock or memory budget was validated for the Spark's actual 96-row / 41x41
  run** beyond the CPU smoke at far smaller row/grid counts; the Spark run may need a
  smaller `--chunk` or a shorter grid if memory is tight, and that decision is made live
  during the run, not pre-committed here.
* **`--n-power-iter` (Jacobian) defaults to 20**, chosen from the synthetic-spectrum
  test's own convergence (60 iterations recovers a 5/3 gap to <1e-2 relative error), not
  calibrated against a REAL core-map spectrum, which may have a narrower gap needing more
  iterations to resolve — the reported `sigma1`/`sigma2` numbers should be read with that
  caveat, and the JSON records `rel_change`-free QR iterations rather than a convergence
  residual (a gap for a follow-up, not fixed here).

## Results

(filled after the Spark run)
