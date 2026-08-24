# Agent Note: the core takeover is a forward state collapse, so no weight control reaches it

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

**Do not enable any spectral control by default. Nothing found here is a cure; the best
lever is upstream, in the slot states.** A control on the core weights' spectrum cannot reach
this failure — two of the four arms that tried made it WORSE than doing nothing.

`tul.per_slot_embed` gives each slot INDEX its own input embedding row instead of adding one
shared `E_slot` to all of them, aimed at exactly the degeneracy measured above. It is the
largest single improvement found and it is NOT a cure: it holds seed 1 for the full 4000
steps (end core share 0.0223, gain 1.052 at r2 0.48 — the flat healthy signature — validation
CE monotone and finishing at its own minimum) and it takes over at seed 0, at step 2225.

What survives both seeds is worth having: the takeover is delayed from step 1150 to 2225,
the validation CE damage falls from +0.533 to +0.119, and the best CE reached is 0.78 and
0.46 nats BELOW the control's best-ever on the two seeds. `sigma_max` also falls as a
CONSEQUENCE — 2.88 and 2.45 at step 1500 against the control's 4.86 — with no spectral
control active, and only climbs once the takeover starts.

It is left OFF by default, and the second seed is the reason. A setting that holds one seed
of two is exactly what `ademamix_alpha_cap: 1.0` already is, and this recipe does not need a
second one of those presented as a fix. Had the second seed not been run, this note would
have said "cure".

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
* **And upstream of all of it, the forward states are already degenerate and the loop's
  effect on them flips sign at the onset.** The 50 valid slot states of a row occupy an
  effective rank of 1.7 to 4.8 in 1024 dimensions at EVERY rung; at healthy rungs the loop
  RAISES that rank across its iterations (x1.23 to x1.48) and by step 1850 it LOWERS it
  (x0.67), with the flip between 1750 and 1800 where the core share goes 0.021 to 0.372.
  The cotangent sits on the same top-3 slots at every core block.

The loop is power iteration — 24 applications of the same `J^T` — and power iteration
concentrates onto a direction. But it sharpens a concentration it is HANDED: the slot states
are near-parallel before the loop touches them, because a slot's input is one shared
`E_slot` plus a span bag-mean and a mean over many embeddings concentrates. That is why A1
(64 near-parallel slots) fails and A0 (1024 diverse token positions) does not, at the same
weights and a LESS protective optimizer setting. It is also why halving `bptt_depth` from 4
to 2 — four orders of magnitude off the backward compounding — cuts the harm by 64 % and
does not cure: the FORWARD still loops 6 to 8 times regardless.

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
* **Raise `tul.max_slots`.** NOT RUN — a measured OOM at batch 12 and at batch 10, and
  probably close to a no-op anyway, since a typical row uses 57 of the 64 slots and only
  7.7 % saturate.
* **Cut `bptt_depth` 4 -> 2.** RUN, as the pre-registered discriminator. It cuts the
  validation CE damage by 64 % (+0.192 against the control's +0.533) and does not prevent the
  takeover — which is what favours the forward reading over the pure backward one.
* **Per-slot input embeddings** (`tul.per_slot_embed`). The best lever found and not a cure —
  see the Decision above. Implemented, tested, off by default.

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
* The earliest indicator this programme has is now a FORWARD one and costs one no-grad pass:
  the ratio of the slot states' effective rank out of the loop to their rank into it. It
  crosses 1 between steps 1750 and 1800, before the core share moves. It is measured by
  `lab/divergence/jac_ladder.py --state-probe` and is NOT yet wired into the trainer as an
  abort criterion, which is the obvious next piece of plumbing.
