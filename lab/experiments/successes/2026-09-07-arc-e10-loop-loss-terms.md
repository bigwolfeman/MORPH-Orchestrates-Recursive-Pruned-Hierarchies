# Success: ARC E10 — two loop loss terms under the detonation assay (terminal fixed point; directional gain hinge)

Status: success
Date: 2026-09-07 (frozen before any smoke; Wolfe: "try the top 3 or 4")
Arc: `2026-09-04-loop-contribution-arc.md`. Note:
`.agents/notes/proposed/architecture/2026-09-07-loop-stability-mechanisms-from-the-literature.md`.
Shares E9's assay and E9's three control draws.

## Question

E7's Jacobian says the loop's failure is a map whose top singular direction runs to 95
while its typical gain reads 0.85, and whose state drifts rather than settles. Two
published mechanisms target that directly with one loss term each: a terminal
fixed-point objective (2608.18222: a settling loop is depth-safe; removing the term
induces drift) and a Jacobian penalty along the top singular direction (STARS 2605.26733:
‖Jv‖² by a power-iteration JVP; here by finite difference along a power-iterated
direction). Does either remove the early detonation on the plain loop?

## Method

The E9 assay: `notul`, warmup 0, 1200 steps, six draws per arm (seeds 301–306), abort rule
`preclip/total > 1e4` at step ≥ 200, control = E9's three `notul_carry_ctx_wu0` draws.

- **E10a** `notul_fp_wu0.yaml`: `core_fixed_point_lambda 1.0` (λ · mean over samples of
  ‖h_T − h_{T−1}‖² / ‖h_T‖² at each sample's last iteration).
- **E10b** `notul_pgain_wu0.yaml`: `core_gain_lambda 0.01`, `core_gain_target 20`,
  `core_gain_eps 0.02`, `core_gain_direction power` (hinge on the finite-difference gain
  along a direction updated by one power step per training step).

Readouts per draw: verdict and first crossing, max `preclip/total`, val at 400/800/1200,
`train/fixed_point` and `train/core_gain_est` / `core_gain_max` traces. Output
`arc/results/2026-09-07-arc-e10/`. Runner `arc/run_e10.sh` after E9. ~13 min per draw,
~2.6 h for twelve.

Cost: E10a adds one elementwise term per finishing sample; E10b adds two extra core-step
applications at one iteration per step (about 1/3 of a core pass at mean 6).

## Predictions (frozen)

- **P10a.** At most 1 of 6 fixed-point draws detonates: **35 %**.
- **P10b.** At most 1 of 6 directional-hinge draws detonates: **40 %**.
- **P10c.** On the hinge arm, `core_gain_est` on the surviving draws sits below 20 by
  step 600 and stays there (the hinge holds σ_max): **55 %**.
- **P10d.** On the fixed-point arm, `train/fixed_point` falls below 0.05 by step 600
  (the loop settles) on the surviving draws: **50 %**.
- **P10e (the price).** Surviving draws' val CE at 1200 is within 0.05 of the surviving
  control draws' (or of the ramped notul at 1200 if no control survives): **45 %** for
  each arm.
- **P10f.** The three E9 control draws detonate ≥ 2 of 3 (shared with P9b): **70 %**.

## Binding

- P10a or P10b TRUE (with P10f TRUE) ⇒ that term is a mechanism-level hold on the early
  detonation; it goes to 5000 ramped steps against notul on the 480 rows (CE at the
  trained depth, the depth sweep, and the Jacobian sweep) before any default changes.
  If both hold, the cheaper one (E10a) is the candidate default and E10b the belt.
- Both FALSE and E9 FALSE ⇒ the early detonation is not a map-side property at all at
  this LR; the ternary trigger and the optimizer are next, and the bounded write
  (2605.18797) is the remaining structural candidate.
- P10c TRUE and P10b FALSE ⇒ σ_max is not the quantity either (the hinge held it and the
  run still detonated); read the probe record for what moved.

## Not verified before launch

Both terms under torch.compile on the GPU (the smoke is the first run); the λ choices
(1.0 and 0.01) are scale guesses from the E7 Jacobian, not measured; the power direction
is shared across the batch and updated once per step, so it lags a fast-rotating map;
the fixed-point term's gradient can be satisfied by a trivially contractive map (E1's
lesson) — the 5000-step follow-up, not the assay, reads that.

### Method amendment, 2026-09-07 06:25 (E10b inert; E10c added; predictions untouched)

E10b's first draw (seed 301) survived as AMBIGUOUS (max `preclip/total` 1.76e3 at 468) with
the penalty at exactly 0.0 on all 1200 steps: the reading along the shared, once-per-step
power direction sat at 1.1–1.5 (max 4.1 in the first 100 steps), never near the target of
20. Every step is a new batch, so a direction refined only ACROSS steps and averaged over
samples reads close to the typical gain, not σ_max. E10b as configured is therefore
INERT, and its six draws are re-designated as six additional warmup-0 CONTROL draws (the
plain recipe, penalty 0), which the assay can use for P10f / P9b. P10b and P10c are
unreadable on E10b. E10c replaces it: `notul_pgain2_wu0.yaml`, the same probe with
`core_gain_power_iters 2` (two finite-difference power steps at the SAME input before the
reading, warm-started from the buffer) and STARS' plain form (`core_gain_target 0`, λ 1e-3,
penalty λ·g²). Six draws, seeds 401–406, queued after E11 in `arc/run_e10c.sh`. The
prediction for E10c is P10b as written (≤ 1 of 6 detonates, 40 %); P10c reads on
`core_gain_est` with no fixed threshold (the σ_max band on the plain loop is unmeasured;
E7's slot-loop readings were 11–95).

## Results (2026-09-07 10:30; `arc/run_e10.sh` on worktree 4a8bae5 for E10a/E10b, `arc/run_e10c.sh` at 4a8bae5 for E10c; files and `scores.md` in `results/2026-09-07-arc-e10/`, scorer `score_e10.py`)

Draws: E10a seeds 301–306 (04:45–06:01), E10b seeds 301–306 (06:01–07:01, penalty 0.0 on
every step: controls), E10c seeds 401–406 (09:18–10:30). Control pool for this assay =
the six E10b draws + E9's one `notul_carry_ctx_wu0` draw (E9 was cut after 2 of 2 widened
draws detonated, so only one of its three planned control draws ran).

| arm | detonated | first crossings | max `preclip/total` on survivors | val at 800 on survivors |
|---|---|---|---|---|
| E10a fixed point (λ 1.0) | **0 of 6** | – | 11–18 (all at step 612–613) | 5.564–5.875 (mean 5.72) |
| controls (E10b penalty-zero + E9 ctx) | **4 of 7** | 201, 224, 266, 768 | 9.3, 135, 1.76e3 (AMBIGUOUS) | 5.907–6.307 (mean 6.13) |
| E10c within-step power penalty (λ 1e-3 · g², 2 power iters) | **2 of 6** | 240, 315 | 14–188 | 5.534–6.298 (mean 5.97) |

The val runs eval at 400 and 800 only (1200 steps end at 1199), so "val at 1200" reads at 800.

E10a: `train/fixed_point` at step 600 is 0.0054–0.0245 on all six draws (0.005–0.06 at
1199; the trace peaks 0.18–0.57 inside the first 20 steps and is below 0.05 from then on).

E10c: the penalty is LIVE (weighted 0.0012–0.0016 on healthy steps, never zero). On the
four survivors the power reading `loss/core_gain_est` has median 1.08–1.13, p95 1.20–1.34,
max 1.8–2.7; `core_gain_max` (the largest sample) tops out at 1.9–2.8. On the two detonated
draws the reading rises AHEAD of the tripwire: seed 406 reads 1.30 at 215, 5.73 at 220, 8.37
at 225, and `preclip/total` crosses 1e4 at 240; seed 404 reads 3.28 at 275, 6.12 at 300,
and crosses at 315. The one-step block-gain probe `preclip/core_block_gain` sits at
0.97–1.63 through both onsets and never reads them. The penalty at the onset is
λ · g² = 0.008 (seed 404) and 0.036 (seed 406): three orders below the CE.

Wall clock per 1200-step draw: E10a 12.7 min, E10b 14.7 min (two extra core applications
with the penalty at zero), E10c 16.3 min (four extra core applications): 1.29x E10a.

Scored:

- **P10a TRUE** (0 of 6; prior 35 %).
- **P10b FALSE** (2 of 6 on E10c; E10b unreadable, inert).
- **P10c not readable as written** (E10b inert; the amendment moved it to a threshold-free
  reading): the within-step reading sits at 1.1–1.3 on healthy steps and climbs to 3–8 in
  the 20–40 steps before a crossing. The direction SEES the onset; λ 1e-3 does not act on it.
- **P10d TRUE** (all six below 0.05 at 600; prior 50 %).
- **P10e**: E10a FALSE in the good direction (the fixed-point survivors read 0.41 nats
  BELOW the surviving controls at 800, six different seeds against four, so the 6.5 %
  per-run spread applies but does not cover 0.41); E10c TRUE (within the control band).
- **P10f TRUE by substitution** (4 of 7 controls, the prereg's "≥ 2 of 3" pool did not
  fully run; the September base rate is 17 of 24).

## Verdict

Success on the half that matters and failure on the other. The terminal fixed-point term is
the first loss term in this tree that holds the early detonation on the plain loop with no LR
ramp: 0 of 6 against 4 of 7 controls (Fisher exact one-sided p = 0.049), with a settling
trace below 0.05 from step 20 and a val CE that is not worse (0.41 nats better on the
survivors). The binding rule's follow-up already ran: E11 (`successes/2026-09-07-arc-e11-
fixed-point-ramped.md`) finds the term FREE under the shipped ramp at 5000 steps
(+0.0009 [−0.0022, +0.0040] at the trained depth).

The directional gain penalty fails twice. Once per step along a batch-averaged direction
(E10b) it reads the typical gain (1.1–1.5) and never fires. With two finite-difference
power steps inside the step (E10c) it reads a leading edge of the detonation 20–40 steps
early, but at STARS' λ 1e-3 the penalty is 0.008–0.036 at the onset and 2 of 6 draws
detonate. The binding's third branch ("P10c TRUE and P10b FALSE ⇒ σ_max is not the
quantity") does NOT fire: the hinge never held σ_max, so nothing about σ_max was tested.
What E10c does give is an instrument: the within-step power reading leads `preclip/total`
where the one-step block gain is blind.

Not verified: whether the power reading is σ_max or a rank-few proxy on the plain loop (no
offline Jacobian sweep was run on these draws; E7's 11–95 was the slot loop); a larger λ
(1e-1, 1.0) or a hinge at 2–3 on the E10c reading, which the leading edge suggests would
act; the reading's false-positive rate on the healthy spikes (max 2.7 on survivors against
3.3–8.4 at the onsets, one overlap at most); the E10c draws' 5000-step behaviour.

## Updated hypothesis

The early detonation is a settling failure that one relative-change term at the last
iteration removes (E10a, E11). The map's top-direction gain is a leading INDICATOR of the
onset (E10c), but a Jacobian penalty at STARS' weight is not a hold; if the penalty lane
stays open, the next draw is the E10c probe as a tripwire (reading > 3 for 5 steps) and a
hinge at 2.5 with λ 0.1, six draws, and it must beat 0 of 6 to matter. The lever moves to
shipping the fixed-point term (Wolfe's call, `.agents/notes/proposed/architecture/2026-09-07-
fixed-point-objective-as-the-loop-stability-term.md`) and porting it to `_tul_core` for the
block-loop rerun.
