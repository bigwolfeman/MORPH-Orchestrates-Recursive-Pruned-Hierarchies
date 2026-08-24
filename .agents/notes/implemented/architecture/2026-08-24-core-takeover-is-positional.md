# Agent Note: the core takeover is a POSITION-space concentration, so no weight control reaches it

Status: implemented

## Problem

TUL arm A1 does not diverge, it TURNS AROUND: validation CE reaches a minimum somewhere
between steps 500 and 2000 and then climbs by 1.3 to 2.4 nats. Four runs at
`ademamix_alpha_cap` 3.5 do it, plus a fifth run made for this work. `tul_short.yaml` ships
`ademamix_alpha_cap: 1.0` and calls it "THE TUL DIVERGENCE FIX"; it is not one, it holds seed
0 for 20000 steps and let seed 1 die at 4140.

The [RCA](../../../../docs/experiments/results/2026-08-24-tul-takeover-rca.md) found the
signature — the core's per-block BACKWARD gain crossing 1, compounding over
`n_core x bptt_depth` = 24 weight-shared blocks — and a panel in which nothing stopped it.
What it could not say was WHERE the amplification lives, and therefore what to bound.

## Decision

**Do not enable any spectral control by default.** The evidence says a control on the core
weights' spectrum cannot reach this failure, and two of the four arms that tried made it
WORSE than doing nothing.

What ships instead is the ability to see the thing: `morph/training/core_jacobian.py`
measures the core map's Jacobian at the live operating point, `lab/divergence/jac_ladder.py`
walks a checkpoint ladder with it and adds the cotangent-concentration and spectral-gap
probes, and `CoreSpectralProjection` exists — off by default — because it is the honest
version of the instrument if anyone reaches for one again.

The measurements behind the decision, all on the same `onset-capture` ladder
([full record](../../../../docs/experiments/failures/2026-08-24-tul-takeover-cure.md)):

* The map barely changes size. Isotropic per-block gain +2.5 % across the whole onset.
* Its blocks' amplifying directions ALIGN, x2.9, tracking the core share step for step.
* The backward cotangent collapses from **13 effective slot positions to 2.5**, while the
  SAME weights on the token path — arm A0's code path, 1152 positions — keep 26 to 59.
* No core weight's spectral GAP opens. Median `sigma_1/sigma_2` 1.069 -> 1.132 and the worst
  gap FALLS, 2.647 -> 2.421.

The loop is power iteration — 24 applications of the same `J^T` — and power iteration
concentrates onto a direction. The direction it is concentrating in is POSITION space, which
is why A1 (64 slots) fails and A0 (1024 token positions) does not at the same weights and a
less protective optimizer setting.

## Alternatives considered

Each was implemented, run at the configuration that fails 4/4, and scored by the rule fixed
in the RCA.

* **Soft spectral penalty, cap 1.5 and cap 3.0.** REJECTED on measurement. It never pinned
  `sigma_max` (1.49 at step 300, 4.26 at 1800, against a cap of 1.5) and both arms ended 2.2
  to 2.7 nats above their own minima against the control's 0.62. A loss-side hinge is a tug
  of war and it can lose; an Adam-family optimizer normalises per coordinate, so raising
  `lambda` does not let it push harder, only own more of the update's direction.
* **Hard spectral projection, MLP only, cap 1.5.** REJECTED. It held `sigma_max` at exactly
  1.50 for its whole life and took over anyway. KEPT IN THE TREE, off by default, because it
  is strictly the better instrument if a spectral bound is ever wanted: the constraint holds
  by construction, nothing enters the loss, and it costs 6 % of throughput.
* **Hard projection over MLP plus attention.** REJECTED. The best arm of the four — it
  reaches validation CE 4.7418, better than the control's 4.7881 — and it still takes over.
* **A penalty on the spectral GAP, and one on the bulk spread.** The spread version was
  written, unit-tested and then DELETED, because measuring its target took twenty minutes
  and the target does not exist (the gap does not open), and because a random direction in
  1024 dimensions puts 1/1024 of its energy on the top singular vector, so a bulk statistic
  is blind to a dominant direction anyway.
* **`core_gain_clip`**, the forward carrier renormalisation. Rejected earlier by the RCA at
  every iteration range. It bounds the state, and the state is not the problem.
* **Cut `bptt_depth` 4 -> 2.** Not tested. It halves the exponent AND halves the power
  iteration's progress per backward, so it has the clearest mechanism behind it of anything
  untried — but it changes the credit-assignment window, which is a change to the METHOD.
* **Raise `tul.max_slots`.** The intervention the measurement points at, tested at the end of
  this work; see the experiment record for its result. It also changes the method (tokens per
  row moves 1033 -> 1161), so it is a design question, not a defect fix.

## Consequences

* `tul_short.yaml` is UNCHANGED. `ademamix_alpha_cap: 1.0` stays, with its limits now
  documented rather than implied: it is a coin flip, not a fix.
* Three real defects were found and fixed on the way, all of which corrupted comparisons:
  `train/loss` included the spectral penalty, so a penalised arm was not comparable to its
  control and the perplexity guard could fire on the regulariser; `CoreSpectralPenalty.sigmas()`
  ran 10 power iterations and under-read `spec/sigma_max`, the number this whole programme
  reads as the spectral norm; and `lab/divergence/jac_ladder.py`'s first version loaded no MLP
  weights at all.
* The block-gain abort criterion leads the shipped perplexity guard by 4940 steps on the
  measured control (1700 against 6640). It is still off by default because it forces
  `grad_probe_every=1`; the case for turning it on for TUL arms is now quantified.
* Anyone reaching for a spectral cap in this tree should read the experiment record first.
  Four arms, one control and one pre-fixed rule is enough evidence to stop.
