# Planned: does the slot apparatus pay for itself, and does the restriction buy anything?

Status: planned
Configs: `tul_a1` (full TUL), `tul_a3` (no slots, no core), `tul_a1_noaux` (NEW, the missing 2x2 cell)
Wolfe's call, 2026-08-28: run it at 4500 steps.

## Why these three, and why now

Two nights of TG variants moved ce_main by at most 0.112 nats (TG3b) and the span-specificity
panel then showed the restriction destroys the plan's content 20x at matched aux losses
(65.1% -> 3.0%). Before another variant is built, two prior questions have to be settled.

**Q1 — has the slot apparatus EVER paid?** `docs/ablation-ledger.md`, seed 0, batch 14,
2026-08-18: A0 (no TUL) 3.2805, A1 (full TUL) 3.2243, **A3 (no slots AND no core) 3.2407 at
2.76x A0 throughput.** A3 lands 0.016 nats from full TUL — far inside the 0.036-0.16 seed
spread this campaign has measured — at less than half the cost. But A1r diverged 2/2, so no
noise floor exists and every one of those numbers is n=1. The most consequential comparison
in the project has never been run with a second seed.

**Q2 — does the RESTRICTION buy anything?** TG2's one durable win was removing the takeover
(0 of 4 seeds against a control firing 3 of 4). TG2 changed the objective AND the mask
together. If zeroing the aux losses alone cures the takeover, the restriction is pure cost:
it charges 0.14-0.22 nats and erases the plan's span-specificity, and returns nothing.

`tul_a1_noaux` is also the missing cell of the specificity 2x2:

    |                  | aux ON              | aux OFF                |
    | restriction OFF  | ctrlworth-s3 65.1%  | THIS ARM               |
    | restriction ON   | tg1-s1        3.0%  | tg2/tg3b/cap64 0.1-0.6%|

## Predictions (frozen 2026-08-28 10:05, before any of the six runs starts)

Metric **ce_main** at step 3000 (matched to the control band 4.4586-4.5459) and at 4500.
Two seeds per arm; every claim is on the two-seed MEAN, because within-arm spread this
campaign is 0.036-0.16 nats and seed-matched n=1 is not readable.

**D1 (the slot apparatus is worth less than the loop's own line).** |A1 mean − A3 mean| at
step 3000 is **≤ 0.05 nats**. The ledger's n=1 gap was 0.016.

**D2 (and it costs more than double).** A3's steps/sec is **≥ 2x** A1's, measured from the
training logs.

**D3 (A1 is unstable).** At least 1 of 2 A1 seeds fires the takeover rule or trips the
divergence guard. A1's family has taken over 3 of 4 control seeds and A1r diverged 2/2.

**N1 (aux-off cures the takeover WITHOUT the restriction).** 0 of 2 `a1_noaux` seeds fire the
takeover rule. This is the prediction that decides whether the restriction has any remaining
justification.

**N2 (the restriction, not the aux loss, is what erased the content).** `a1_noaux`
specificity is **≥ 3.0%** — above tg1-s1, the best restricted arm. The multiplicative reading
of the 2x2 (restriction ~22x, aux ~10x) puts it near 6.5%; the falsifiable claim is that
removing the restriction keeps content that no restricted arm retains.

**N3 (aux-off does not cost CE).** `a1_noaux` mean ce_main@3000 is within 0.05 of the control
band's midpoint, 4.5023 — i.e. removing the aux losses is free or better on the main metric.

## Decision rule

- **D1 HOLDS and D2 HOLDS** → the slot apparatus does not pay at this scale. Keep the cheap
  part (tokens skipping the core, which is A3) and drop the slots, OR move the whole
  question to long context where a compressed plan has a job. Do not build another
  short-context TUL variant.
- **D1 FAILS** (A1 beats A3 by > 0.05) → the slots ARE earning their keep and the ledger's
  n=1 reading was noise. That reopens everything and is the single most welcome outcome here.
- **N1 HOLDS** → the restriction is retired. It costs 0.14-0.22 nats, erases 20x of plan
  specificity, and its only claimed win is reproduced without it.
- **N1 FAILS** (a noaux seed takes over) → the restriction IS doing stability work, and the
  aux-loss account of the takeover is incomplete — which the TG4b-s1 takeover already hinted
  at (that arm had aux OFF and took over with end share 1.0000).
- **N2 FAILS** (noaux specificity < 3.0%) → the emit loss, not the mask, is what writes
  span-specific content, and the 65.1% control number is an artefact of its aux objective
  rather than evidence about the restriction. My headline finding from the specificity panel
  would need rewriting, and I would rather find that here than defend it later.

## Method

Six runs: `tul_a3`, `tul_a1`, `tul_a1_noaux` x seeds 1,2. **4500 steps**, batch 6,
`use_kernels=false`, `ademamix_alpha_cap=3.5`, `grad_probe_every=1`, ckpt every 500 —
otherwise identical to every arm in this campaign so the control band stays comparable.
Sequential on the 5090 (UPS). Worth passes (now including the `plan SHUFFLED` condition) at
steps 3000 and 4500. Takeover from `score_arms.py`, rule unchanged.

NOT controlled, and named rather than hidden: **A3 is not iso-parameter.** It sets
`n_core: 0`, which removes the core's weights entirely, so it is a smaller AND shallower
model — `tul_a3.yaml` says so itself. That is inherent to a compute floor. It cuts against
A1, not for it: if a strictly smaller model matches A1 at >2x throughput, the apparatus is
not earning its parameters either. Parameter counts are printed at build and will be
reported per arm.

A3 also has no slots, so it has no plan and no worth passes — its rows in the worth table
are structurally absent, not missing data.
