# The TUL core takeover: the mechanism, and why no weight-space cure reaches it

Status: failure

The experiment asked whether bounding the core map's spectral norm cures the takeover. It
does not. The predictions that mattered — P4 (the cure holds at a second seed in the real
configuration) and P9 (the spectral cap stands alone) — are both falsified, so this is filed
under `failures/` whatever else it established. What it DID establish is the mechanism, in
much sharper form than the RCA had, and it eliminated an entire family of interventions with
a measurement that says why.

Pre-registration: [2026-08-24-tul-takeover-cure](../planned/2026-08-24-tul-takeover-cure.md)
(with two method amendments, both written before the arms they changed).
Mechanism this builds on: [the RCA](../results/2026-08-24-tul-takeover-rca.md).
Procedure: [measuring the core map's operator](../../cookbook/measuring-the-core-map.md).
Decision record: [the takeover is positional](../../../.agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md).
Figure: `docs/experiments/figures/tul_takeover_cure.png`.

## Summary

**The mechanism is now measured on the operator, not inferred from gradients.** The core is
weight-shared, so the backward applies the same `J^T` `n_core x bptt_depth` = 24 times. That
is power iteration, and it converges the cotangent onto the map's amplifying direction.
Across the onset the map's isotropic per-block gain moves **+2.5 %** while the ALIGNMENT of
its six blocks moves **x2.9** and the realized per-block gain moves **+34.5 %**. The map
barely changes. Its directions align.

**The alignment is in POSITION space, not feature space.** The backward cotangent collapses
from about **13 effective slot positions to 2.5** across the onset, tracking the core share
step for step. The SAME weights run through the token path — arm A0's code path, 1152
positions — keep 26 to 59. Meanwhile, on the same checkpoints, no core weight matrix's
spectrum explains anything: `sigma_max` per block rises only 2.35 -> 3.23, the spectral GAP
`sigma_1/sigma_2` has a median of 1.069 -> 1.132 with the WORST gap FALLING, and the bulk
spread of the spectrum does not move.

**Four interventions in feature space were tried and all four failed** at the configuration
that turns around 4 times out of 4. Including a hard projection that held `sigma_max` at
exactly 1.50 for the whole run. That is not four unlucky guesses; it is what the gap
measurement predicts, since a norm cap leaves every singular vector and every ratio
`sigma_i/sigma_j` untouched and therefore cannot slow an alignment at all.

**One cure works in the deterministic microcosm and does not transfer.** The soft spectral
cap at 1.5 prevents the takeover at batch 6 (control took over at 1866; the arm held to 2100
at equal CE) and loses at batch 12, where the drive is four times faster.

So: the takeover is understood, it is reproducible, it has a leading indicator, and it is
not yet cured. What follows is the evidence for each of those, in the order it was taken.

## What the failure actually is

Not a loss spike. A VALIDATION CE turnaround, and a slow one. From wandb
`adew-me/morph-tul`, `val/ce_tokens` at its minimum against its value at the end of the
run (`val/loss` for A0 and A3, which are not TUL arms and do not log `ce_tokens`):

| run | id | config | val min | at | val end | at | rise |
|---|---|---|---:|---:|---:|---:|---:|
| tul-a1 | `82easori` | alpha_cap 3.5, b14, s0 | 4.7917 | 1500 | 6.3780 | 4500 | **+1.586** |
| tul-a1 | `cyushbhr` | alpha_cap 3.5, b14, s0 | 4.8011 | 1000 | 6.1522 | 5500 | **+1.351** |
| tul-a1r | `8e49z6u8` | alpha_cap 3.5, b14, s1 | 5.1807 | 1500 | 6.6495 | 3000 | **+1.469** |
| tul-a1r | `8vdthy0r` | alpha_cap 3.5, b14, s1 | 5.8002 | 500 | 7.0830 | 2000 | **+1.283** |
| tul-a1r | `0ujvtukf` | **alpha_cap 1.0**, b12, s1 | 4.5164 | 2000 | 6.4471 | 4000 | **+1.931** |
| tul-a1 | `c23dwx4a` | alpha_cap 1.0, b12, s0 | 3.3005 | 18500 | 3.4782 | 19500 | +0.178 |
| tul-a0 | `l4apqgyo` | alpha_cap 3.5, b14, TUL off | 3.1749 | 19500 | 3.1749 | 19500 | +0.000 |
| tul-a3 | `4lb85o25` | alpha_cap 3.5, b14, no core | 3.1445 | 19500 | 3.1445 | 19500 | +0.000 |

Three facts set the target.

**At `alpha_cap` 3.5 it fails 4 out of 4**, which is why that setting is the control for
the deciding pair here. `tul_short.yaml` records the same thing as 5/5 with abort steps
2080, 3240, 4540, 5900 and 6200.

**`ademamix_alpha_cap: 1.0` is not a cure.** `tul_short.yaml` calls it "THE TUL DIVERGENCE
FIX". It holds seed 0 for 20000 steps and it does NOT hold seed 1, which turned around at
step 2000 and was aborted at 4140, 1.93 nats above its own minimum. At that setting the
failure is about a coin flip: two of three runs at `alpha_cap` 1.0 are healthy (including
one run here) and one died.

**The failure needs the looped core to be running on SLOTS.** A0 turns TUL off and loops
the core over 1024 token positions; A3 removes the core entirely. Both are healthy for
20000 steps at the LESS protective `alpha_cap` 3.5. Only A1 — the core looping over 64 slot
positions — fails.

## Method

Everything runs on one 5090, serially.

**The reproducible microcosm.** `--config-name tul_a1`, `training.deterministic=true`,
`model.use_kernels=false`, `training.batch_size=6`, seed 0, 2100 steps, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before the process starts. Two runs in this
configuration agree on all 300 probed steps across 85 series
([evidence](../results/2026-08-23-morph-bit-reproducible.md)), so a one-run comparison here is a
CONTROLLED comparison. It costs 2.28x throughput and half the batch, and it grows
`sigma_max` about eight times faster than the batch-12 recipe, so it is an accelerated
version of the failure rather than a different one.

**The real configuration.** `--config-name tul_a1r` — stock `tul_short` at batch 12 with
kernels on — `training.steps` per arm and `training.ademamix_t_beta3=20000` on every arm so
they share an optimizer schedule (`t_beta3` is null in `base.yaml` and therefore tracks
`training.steps`; see the RCA's harness defects). Kernels on means these are NOT
bit-reproducible, and the measured run-to-run spread on the gradient norm is 6.5 %, so only
differences much larger than that are readable.

**Verdict rule**, fixed in the RCA before any arm here ran: TAKEN OVER = core share above
0.5 on more than 30 % of the last 50 probed steps. Applied by
`lab/divergence/score_arms.py`, which also reports the block backward gain, the step at
which each criterion first fires, and each arm's own loss minimum and end.

**The operator.** `morph/training/core_jacobian.py` power-iterates `J^T J` at the live
operating point, with `J v` from the double-backward identity, in fp32 with autocast off,
restricted to the positions the iteration is actually updating. It reports `sigma_max`, the
convergence residual, and the Hutchinson typical gain `||J||_F / sqrt(n)`.
`lab/divergence/jac_ladder.py` walks a checkpoint ladder with one fixed batch and one fixed
depth draw per rung.

**Telemetry that was already there.** `spectral_penalty_log_every: 100` constructs the
penalty for LOGGING on every run, penalised or not, so `spec/sigma_max` exists for arms
that predate this work — including the two 20000-step arms whose comparison is the sharpest
single piece of evidence here and which cost nothing to obtain.
## The mechanism, measured on the operator

The RCA measured the per-block backward gain from GRADIENT norms and showed it crosses 1.
That says the backward amplifies. It does not say why, and two explanations fit the same
number: the map became expansive, or the map was always expansive in some direction and the
realized backward direction rotated into it. `morph/training/core_jacobian.py` measures the
map itself and separates them.

Four numbers, all at the same fixed batch and the same fixed depth draw on every rung of
the `onset-capture` ladder (300 power iterations; convergence residual at most 1e-7):

* **isotropic per-block gain** — `||J||_F / sqrt(n)` of one core block, the gain a generic
  direction sees.
* **realized per-block gain** — the RCA's fitted `preclip/core_block_gain`, the gain the
  actual backward cotangent sees.
* **alignment** — the whole step's isotropic gain divided by the PRODUCT of its six blocks'
  isotropic gains. At 1 the blocks' amplifying directions are generic with respect to each
  other; above 1 they agree, and the composition amplifies more than its factors.
* **sigma_max** — the worst case over all directions, i.e. the headroom alignment can climb
  into. Quoted per block as `sigma_max(step)^(1/6)`.

| step | isotropic | realized | **alignment** | sigma_max/block | core share |
|---:|---:|---:|---:|---:|---:|
| 1625 | 1.0292 | 1.053 | 1.104 | 2.616 | 0.016 |
| 1700 | 1.0315 | 1.057 | 0.986 | 2.349 | 0.012 |
| 1750 | 1.0349 | 1.066 | 1.136 | 2.816 | 0.021 |
| 1800 | 1.0427 | 1.329 | **1.469** | 3.149 | 0.372 |
| 1850 | 1.0467 | 1.434 | **1.919** | 3.226 | 0.890 |
| 1866 | 1.0552 | 1.238 | **2.850** | 3.225 | 0.961 |

**The map barely changes. Its directions align.** Over the onset the isotropic per-block
gain moves +2.5 %, from 1.0292 to 1.0552. Over the same 241 steps the alignment factor
moves x2.9, from 0.986 to 2.850, and it tracks the core share step for step: 1.136 at share
0.021, 1.469 at 0.372, 1.919 at 0.890, 2.850 at 0.961. The whole step's typical gain
follows alignment, not size: 1.19 to 3.93, a factor of 3.3.

Two things make that possible.

**Every core block is expansive already, and the headroom is large.** The isotropic
per-block gain is 1.02 to 1.08 at every rung, healthy included, while `sigma_max` per block
is 2.3 to 3.2. So a generic direction is amplified 5 % per block and the best direction is
amplified 300 %. A residual block cannot be made contractive — its Jacobian is `I + J_F`,
and `sigma_max(I + J_F) >= 1` unless `J_F` is contractive AND anti-aligned with the
identity — so expansiveness is not the pathology. The RATIO between the best direction and a
generic one is.

**The loop is power iteration.** The backward applies the SAME `J^T` `n_core x bptt_depth`
= 24 times, because the core is weight-shared. That is exactly the algorithm for finding a
matrix's top singular direction, and it converges at rate `(sigma_1/sigma_2)^k`. Nothing
else in the network does this: an ordinary stack of 24 distinct blocks applies 24 DIFFERENT
operators, and their top directions do not compose.

So the runaway is a positive feedback loop between the operator's spectrum and the loop's
own power iteration. The operator grows headroom (`sigma_max` per block 2.35 to 3.23,
+37 %); the weight-shared loop aligns the cotangent into it (alignment x2.9); the realized
per-block gain rises toward `sigma_max` and away from the isotropic 1.05; the core's
gradient is larger than every other region's by that gain raised to the 24th; the update
therefore chases that direction and grows the headroom further.

Arithmetic check, and what it does and does not close. The measured within-step alignment
takes the whole step's typical gain from 1.19 to 3.93, so over the four unrolled core steps
of a `bptt_depth` 4 backward that is `3.93^4 = 238` against a healthy `1.19^4 = 2.0`, a
factor of 120. The RCA's realized per-block gain reaches 1.43 to 1.88, i.e. `1.88^24 = 3e6`
against the observed pre-clip rise of ~7e7. The two agree in direction and the realized
number is the larger one, which is what the alignment reading predicts — the realized
cotangent is MORE aligned than a generic direction. What is measured here is alignment
WITHIN one core step, across its six blocks. Alignment ACROSS the four unrolled steps is the
same operator applied again and should compound, but it is not separately measured.

This is why the RCA's whole intervention panel failed. `core_gain_clip` rescales the FORWARD
state; it does not touch the spectrum, so the direction to align onto is still there.
Halving the learning rate slows the walk without changing where it walks to. `alpha_cap`
bounds one term of the optimizer's update, not the operator.

One qualification, added after the interventions ran. `sigma_max` per block does rise 2.35
to 3.23, but the RATIO `sigma_1/sigma_2` of the individual weight matrices does NOT — median
1.069 to 1.132, worst gap FALLING. So `sigma_2` rises with `sigma_1`, and the headroom that
opens is not a widening gap in any single map. Whatever is aligning is a property of the
COMPOSITION and of the positions the cotangent occupies, not of one matrix's spectrum. That
is the finding that killed every intervention below, and it is measured in the next two
sections.

Why arm A1 and not A0 is answered by the next section, which measures it.

## Where the concentration actually is: positions

Power iteration converges onto a DIRECTION, and the cotangent at a core block is a sum over
the active positions, so the number of positions bounds how sharply it can concentrate.
Measured as the participation ratio of the per-position cotangent norms and reported as an
EFFECTIVE number of positions (`lab/divergence/jac_ladder.py --rank-probe`, via
`register_full_backward_hook`, so it uses only public API):

| step | core share | slot path, effective positions (block 0 .. 5) |
|---:|---:|---|
| 1625 | 0.016 | 12.84 12.89 12.96 12.96 12.97 13.02 |
| 1700 | 0.012 | 13.38 13.56 13.61 13.67 13.75 13.87 |
| 1750 | 0.021 | 12.33 12.85 13.03 13.19 13.29 13.38 |
| 1800 | 0.372 | **6.85** 7.55 8.04 8.34 8.64 8.92 |
| 1850 | 0.890 | **3.73** 4.21 4.51 4.76 5.31 5.74 |
| 1866 | 0.961 | **2.49** 2.52 2.88 3.35 3.79 4.05 |

Of 57 valid slots the cotangent occupies about 13 effective positions while the run is
healthy and 2.5 once it has taken over — a factor of five collapse that tracks the core
share step for step, and that deepens toward block 0, which is the LAST block the backward
reaches and therefore the most-iterated.

The same weights on the token path — `slot_layout=None`, which is arm A0's code path over
1152 positions — never do this:

| step | token path, effective positions (block 0 .. 5) |
|---:|---|
| 1625 | 398.8 421.0 441.8 456.5 481.2 514.5 |
| 1750 | 23.8 60.9 97.6 133.8 179.8 215.4 |
| 1850 | 59.1 92.9 132.8 163.4 195.2 226.9 |
| 1866 | 25.6 53.8 100.8 142.7 183.5 220.3 |

At the block where the slot path is down to 2.5 positions, the token path is at 26 to 59 —
an order of magnitude more, at IDENTICAL weights. It is also noisier across rungs, because
its per-position cotangent is a much longer sum.

This is the measurement behind "A1 fails and A0 does not". The operator is the same. What
differs is how many independent terms the cotangent is a sum of — 57 slots against 1152
token positions — and therefore how sharply 24 applications of the same `J^T` can
concentrate it.

It also explains, after the fact, why every feature-space intervention failed: they all
constrain what the map does to a VECTOR, and the thing that is concentrating is which
POSITIONS carry the vector.

## What the cap sweep says, and what it does not

Projecting the core's linears onto `sigma_max <= cap` in the SICK state (`ROLL_step_1850`)
and re-measuring the operator says which family carries the amplification and how far it
has to shrink. The projection goes through each module's forward, so it acts on the
effective ternary map, and it re-measures afterwards and raises if it missed the cap.

Reference points: the sick state is `rms 2.5234, alignment 1.9188`; the three healthy rungs
are `rms 1.19-1.40, alignment 0.986-1.136`.

| scope | cap | linears projected | whole-step typical gain | alignment |
|---|---:|---:|---:|---:|
| — | none | 0 | 2.5234 | 1.9188 |
| mlp | 3.0 | 2 | 2.3153 | 1.7762 |
| mlp | 2.0 | 11 | 1.6921 | 1.3813 |
| **mlp** | **1.5** | 12 | **1.5001** | **1.2387** |
| mlp | 1.0 | 12 | 1.4643 | 1.2111 |
| attn | 3.0 | 1 | 2.5233 | 1.9187 |
| attn | 2.0 | 15 | 2.4249 | 1.8564 |
| attn | 1.5 | 18 | 2.2799 | 1.7560 |
| attn | 1.0 | 27 | 2.2766 | 1.7595 |
| all | 2.0 | 26 | 1.5246 | 1.2558 |
| **all** | **1.5** | 33 | **1.2424** | **1.0499** |
| all | 1.0 | 55 | 1.0320 | 0.9172 |

Three things fall out.

**The amplification lives in the MLP, not in attention.** Capping every CCA projection at
1.0 — 27 of them — moves the alignment from 1.919 to 1.760, 9 % of the way back. Capping
the twelve MLP linears at 1.5 moves it to 1.239, 76 % of the way back. This CORRECTS a
reading in the RCA: that table saw the attention projections moving most in RAW weight norm
and suggested they were the place to look. Raw weight norm is not the operator.

**1.5 is the knee.** From 3.0 to 1.5 the alignment falls 1.776 -> 1.239; from 1.5 to 1.0 it
falls only 1.239 -> 1.211. Almost everything the cap can buy is bought by 1.5, and tightening
further costs capacity for nothing. That is where the shipped value comes from.

**MLP alone gets most of the way; MLP plus attention gets all of it.** `all` at cap 1.5
lands at alignment 1.0499 and typical gain 1.2424, inside the healthy band on both. The
shipped cure caps the MLP only, because that is the configuration measured in training. The
`all` row says what extending `CoreSpectralPenalty`'s unused `include_attn` hook would be
worth, with a number rather than a guess.

The worst effective `sigma_max` at `ROLL_step_1850`, for reference: `core.0.mlp.0.gate_up`
3.371, `core.1.mlp.0.gate_up` 3.128, `core.4.attention.cca.gate.0` 3.018,
`core.5.mlp.0.gate_up` 2.937. The run's own log reports 3.30 at step 1800, which is the
same number from the same estimator, and is how the loader defect below was caught.

### And the sweep did NOT predict the training arms

Read the `all` row again: capping everything at 1.5 puts the frozen sick checkpoint's
alignment at 1.0499, inside the healthy band. `a35-proj15attn` does exactly that, every
step, and takes over anyway.

The sweep re-measures a FIXED set of weights. A run trained under the cap is free to rebuild
the same alignment inside the smaller budget, and the arms say it does — `a35-proj15` held
`sigma_max` at 1.50 for its whole life and its realized per-block gain reached 1.659,
HIGHER than its uncapped control's 1.402 at the same step. So the sweep answers "how much of
the sick state's alignment is attributable to weight magnitude" — 76 % for the MLP alone,
essentially all of it for MLP plus attention — and does NOT answer "will capping magnitude
prevent the alignment from forming". Those are different questions and only the second one
is the cure. Recorded because the first reading was mine, and the arms corrected it.
## The level of sigma_max is NOT the criterion — the RATE is

Both 20k-schedule arms at `alpha_cap` 1.0 logged `spec/sigma_max` (the logging-only
construction runs on every arm). They differ only by seed.

| step | `c23dwx4a` seed 0, healthy to 20k | `0ujvtukf` seed 1, aborted at 4140 |
|---:|---:|---:|
| 0 | 1.41 | 1.42 |
| 900 | 1.94 | 1.94 |
| 1600 | 2.11 | 2.06 |
| 1900 | 2.17 | 2.32 |
| 2100 | 2.23 | 2.93 |
| 2400 | 2.34 | 3.54 |
| 3000 | 2.58 | 4.87 |
| 4100 | 2.76 | 5.51 |
| 9900 | 3.21 | — |
| 19800 | 3.99 | — |

The two are indistinguishable to step ~1800. Then the failed seed KNEES: it first reaches 3.0 at
step 2200 and 5.51 by 4100. The healthy seed first reaches 3.0 at step 6300 and ends at
4.02 after 19900 steps.

So a healthy run reaches `sigma_max` 4.0 and is fine, while a failing one dies at 3.0-3.5.
The level is not the criterion. What separates them is the RATE: 1.3e-4 per step in the
healthy run against 1.4e-3 per step through the failed run's knee, ten times faster. That
is consistent with the alignment reading — the runaway is a feedback loop, and a feedback
loop shows up as a change of slope, not as a threshold crossing.

**Consequence for the cure, stated plainly.** A cap of 1.5 is far tighter than a healthy
run's own trajectory, which passes 1.5 before step 200 and ends near 4.0. The cap does not
work by holding the model where a healthy run would sit; it works by removing the headroom
the feedback loop needs. That makes the long-horizon CE cost a real open risk and not a
formality, which is what the 12000-step arm is for, and it makes the `cap 3.0` arm more
than a dose control: 3.0 still cuts the failed seed's runaway off at step 2200 while
leaving the healthy seed unconstrained until step 6300.

### The growth rate across every run that logged it

| run | config | d sigma_max / d step | outcome |
|---|---|---:|---|
| `c23dwx4a` tul-a1 s0 | b12, alpha_cap 1.0 | 1.3e-4 | healthy to 20000 |
| `0ujvtukf` tul-a1r s1 | b12, alpha_cap 1.0 | 1.3e-4 to step 1800, then **1.4e-3** | aborted 4140 |
| `onset-capture` | b6, deterministic, alpha_cap 3.5 | **1.05e-3** from step 0 | took over 1866 |
| `spec-scratch` (cure) | b6, deterministic, cap 1.5 | pinned at 1.50 from step ~400 | held to 2100 |
| `cure-a1r-ctrl` | b12, alpha_cap 1.0, seed 1 | 3.2e-4 | see below |

A smaller batch grows `sigma_max` faster, which is what a feedback loop driven by gradient
noise should do, and the deterministic microcosm at batch 6 is therefore an ACCELERATED
version of the failure rather than a different one. No single value of `sigma_max` predicts
the outcome: `onset-capture` took over at 3.3 while `c23dwx4a` sat at 4.0 and was fine.
## The one arm that worked, and did not transfer

In the deterministic microcosm — batch 6, `use_kernels=false`, seed 0, `alpha_cap` 3.5,
2100 steps, where two runs agree on all 300 probed steps across 85 series — the soft
spectral cap DOES cure the takeover.

| | control `onset-capture` | cure `spec-scratch` (cap 1.5, lambda 10) |
|---|---:|---:|
| core share at step 1866 | 0.602 | **0.011** |
| block gain, median of the last 50 | 1.303 | **0.968** |
| block-gain fit r2 | 0.898 | 0.161 |
| block-gain criterion fires | 1760 | **never** |
| core-share criterion fires | 1788 | **never** |
| `sigma_max` at step 1800 | 3.30 | 1.50 |
| train loss at 1800 | 5.1289 | 5.1348 |
| outcome | aborted at 1866 | ran to 2100 |

The intervention holds the block gain below 1 through the step at which the control took
over, at a cost of 0.006 nats. `sigma_max` is pinned at 1.50 from step 400 onward.

It does not transfer to batch 12. The reason is visible in the same telemetry: the soft
hinge holds when it engages before `sigma_max` gets away, and the drive at batch 12 is four
times faster. `onset-capture` grows `sigma_max` at 1.05e-3 per step; `a35-ctrl` reaches 5.69
by step 1600, faster still; the hinge at batch 12 was already at 2.86 by step 1200 and never
recovered the ground.

So the microcosm result is real and it is narrow: it says the mechanism is causal — an
intervention that holds the block gain below 1 does prevent the takeover — and it does not
say the intervention is a cure at the recipe's own scale.

## Four interventions, one control, one rule

All at `--config-name tul_a1r`, `ademamix_alpha_cap` 3.5 (which turns around 4/4 in the
table above), batch 12, kernels on, `ademamix_t_beta3` pinned to 20000 so every arm shares
an optimizer schedule. Scored at a common step 2050 by the rule fixed in the RCA: TAKEN OVER
= core share above 0.5 on more than 30 % of the last 50 probed steps.

| arm | what it constrains | end share | block gain | r2 | val CE min | val CE @2000 | rise | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `a35-ctrl` | nothing | 0.9523 | 1.402 | 0.91 | 4.7881 @1000 | 5.4116 | +0.623 | TOOK OVER |
| `a35-spec` | soft cap 1.5, MLP | 0.9979 | 2.038 | 0.97 | 5.4525 @500 | 8.1891 | **+2.737** | TOOK OVER, worse |
| `a35-cap30` | soft cap 3.0, MLP | 0.9982 | 2.003 | 0.96 | 5.1054 @1000 | 7.2813 | **+2.176** | TOOK OVER, worse |
| `a35-proj15` | HARD cap 1.5, MLP | 0.9529 | 1.659 | 0.95 | 4.8084 @1000 | 8.3046 | **+3.496** | TOOK OVER |
| `a35-proj15attn` | HARD cap 1.5, MLP + attention | 0.9148 | 1.571 | 0.94 | **4.7418 @1500** | 5.8500 | +1.108 | TOOK OVER |

**Two of these four rows are confounded, and the confound is mine.** The pre-clip probe
measures `p.grad` after the backward of the FULL objective, so on a penalised arm the
spectral penalty's own gradient — which lands entirely on core MLP weights — is inside
`preclip/core` and therefore inside the core share the verdict is computed from. It shows
up plainly in the pre-clip TOTAL, which is the budget the single global clip divides:

| step | ctrl | soft 1.5 | soft 3.0 | hard 1.5 | hard 1.5 +attn |
|---:|---:|---:|---:|---:|---:|
| 700 | 3.15 | 3.28 | 3.07 | 3.21 | 3.16 |
| 900 | 1.61 | **1.67e4** | **784** | 1.56 | 1.53 |
| 1100 | 1.35 | **1.61e5** | **2.56e7** | 1.35 | 1.32 |

The two HARD arms keep a normal gradient norm and a normal core share right up to their own
onset — they are clean tests. The two SOFT arms' core share reaching 0.998 is the
REGULARISER's gradient swamping the model's, not the core map taking over. Their validation
CE is penalty-free and still 2.2 to 2.7 nats above their own minima, so they still failed,
but they failed by a different route than the label suggests: a core-local penalty inflates
exactly the metric the takeover is judged by, and starves every non-core parameter of the
one global clip budget.

That is a third harness defect this work exposed, and it is unfixed: `_preclip_probe` cannot
separate the data gradient from a regulariser's without a second backward. Anyone adding a
loss term that is not uniform over the parameter tree must read `preclip/core_share` as
contaminated.

No firing STEP is quoted for these arms, on purpose. The abort criteria are defined over a
window of consecutive TRAINING steps at `grad_probe_every=1`, which is what the guard forces
when it is enabled; these arms probed every 25 steps to save time, so emulating a 200-step
window gives 8 samples and a 3-of-8 threshold rather than 60-of-200 — a different and far
noisier criterion. `lab/divergence/score_arms.py` now refuses to report a firing step when
the cadence is too coarse for the window, rather than reporting a number that looks like the
guard's and is not. Firing steps ARE quoted for the deterministic microcosm, which probes
every step.

Read the rows in pairs.

**Soft against hard.** The two soft arms end 2.2 to 2.7 nats above their own minima, against
the control's 0.62 — they are WORSE than doing nothing. Two reasons, and the second is the
one worth carrying away. A loss-side hinge is a tug of war and it lost, never pinning
`sigma_max` anywhere near its cap (1.49 at step 300, 2.86 at 1200, 4.26 at 1800, against a
cap of 1.5). And its gradient is CORE-LOCAL, so as the excess grows the penalty takes over
the single global clip budget and every parameter outside the core stops moving — the
regulariser feeds the failure it was added to prevent. A loss-side hinge is a tug of war and it lost: it never
pinned `sigma_max` anywhere near its cap (1.49 at step 300, 2.86 at 1200, 4.26 at 1800,
against a cap of 1.5). Once the excess is large its quadratic gradient dominates the loss
and the model optimises the regulariser instead of the data, which is what the validation
curve shows. The hard arms hold the constraint exactly — `a35-proj15` sat at `sigma_max`
1.50 from step 300 to its abort — and are much less destructive: `a35-proj15attn` reaches
the BEST validation CE of any arm here, 4.7418, better than the control's 4.7881.

**MLP against MLP plus attention.** Adding the CCA projections delays the fire by 50 steps
and more than halves the CE damage. It still takes over.

**None of them cures.** The best case is an intervention that costs nothing, reaches a
better validation CE than the control, and still takes over.

**And the hard cap made the realized gain WORSE.** `a35-proj15` pinned `sigma_max` at 1.50
and its realized per-block backward gain reached **1.659**, against the uncapped control's
1.402 at the same step. That is not noise around the control, it is the wrong direction, and
it has a reading: if the dynamics are driving toward some composed gain, capping each
block's magnitude leaves rotating the blocks' top subspaces INTO each other as the only way
to get there. On that reading a per-block spectral cap does not merely fail to slow the
alignment — it converts magnitude growth into alignment growth. The alignment factor is not
measured on these arms' checkpoints, so this is a reading of one number and not a result.

### Why none of them could have

A projection `W -> cW` is a UNIFORM rescale: it leaves every singular VECTOR of `W`
untouched and every ratio `sigma_i / sigma_j` untouched. Power iteration converges onto the
top direction like `(sigma_1 / sigma_2)^k`, so a spectral cap cannot slow the alignment. All
it does is lower the gain once aligned — which is why the hard arms are less harmful and
equally fatal.

The obvious repair is to attack the gap rather than the norm. That does not work either, and
the reason is a measurement rather than an argument. `sigma_1 / sigma_2` per core linear,
deflated power iteration, same ladder:

| step | median gap | worst gap | worst linear |
|---:|---:|---:|---|
| 1625 | 1.069 | 2.647 | `core.3.attention.cca.gate.0` |
| 1750 | 1.086 | 2.531 | `core.3.attention.cca.gate.0` |
| 1850 | 1.128 | 2.446 | `core.4.attention.cca.gate.0` |
| 1866 | 1.132 | 2.421 | `core.4.attention.cca.gate.0` |

The median gap moves 6 % across the entire onset and **the worst gap falls**. No core
weight matrix is opening a dominant direction. Whatever is aligning, it is not the spectrum
of any single map.

## The instruments, and what each cost to build

`morph/training/spectral_penalty.py` now holds two controls over the core linears, both off
by default, both enumerated by ONE shared helper so they cannot drift apart about what "the
core linears" are:

* `CoreSpectralPenalty` — the soft hinge `lambda * sum relu(sigma - cap)^2`. Pre-existing;
  now with `include_attn` actually implemented (it was a parameter that was accepted and
  ignored), and with its measured failure mode in the docstring.
* `CoreSpectralProjection` — `W <- W * min(1, cap/sigma)` after the optimizer step. New. The
  constraint holds by construction, nothing is added to the loss so `train/loss` stays
  comparable across arms, and it is cheaper: no autograd through the power iteration.
  Measured cost, `verify` off, 6 % of throughput (1.95 against 2.08 steps/s).

Two module-tree traps the unit tests caught before any GPU time was spent, either of which
would have made the projection silently do nothing: `MortarLinear` HOLDS NO WEIGHT (it
delegates to an inner `CMSBlockLinear`, and asking it for `.weight` raises), and the
trainable leaf is behind the ternary parametrization at `parametrizations.weight.original`,
so writing to `.weight` would be discarded on the next forward.

`verify=true` re-measures after each projection and RAISES if it missed the cap, because the
projection rests on the effective map being homogeneous in the raw weight — true of the
`TernarySTE` parametrization in use, NOT true of `CMSBlockLinear.enable_ternary`, whose
scale is a frozen buffer. It earned its place on its first real run by refusing a projection
that landed 13 % high. The cause turned out to be the estimator, not the assumption: two
power iterations from a random start under-read `sigma_max` by 11 % (1.2674 against a
converged 1.4293). The constructor now converges every vector once.

**One instrument was built and then removed.** A penalty on the spread of `||W v||^2` over
random directions, aimed at flattening the spectrum. Two facts killed it within twenty
minutes, both measurements rather than arguments: no core weight's spectral gap is opening
(the table above), so the target does not exist; and a random vector in 1024 dimensions puts
`1/1024` of its energy on the top singular vector, so a bulk statistic is blind to a single
dominant direction anyway — the unit test that showed it "working" had to plant a spike of
twice the matrix's Frobenius norm to make it visible. The code is gone. Shipping a knob that
has a measurement saying it cannot work is worse than not having one.

### The control that refused to fail

`cure-a1r-ctrl` was the FIRST control run for this experiment: seed 1, `alpha_cap` 1.0,
6000 steps, no penalty — the configuration in which wandb `0ujvtukf` turned around at step
2000 and was aborted at 4140. It did not fail. Validation CE fell monotonically to 3.7732 at
step 4500 and ended at 3.7878; the core share ended at 0.0105 and the block gain at 1.045;
`sigma_max` reached 2.67, tracking the HEALTHY seed-0 run rather than the dying seed-1 one.

It is reported, not discarded. At `alpha_cap` 1.0 the failure is roughly a coin flip — two
of three runs at that setting are healthy — and that is exactly why the deciding pair moved
to `alpha_cap` 3.5, which fails 4 out of 4. The move was made and committed while this
control was still at step 3500 and before any cure arm at the new configuration had started
(pre-registration, "Method amendment").
## What this exposed

**A silent measurement defect in my own tooling, found by a number that disagreed.** The
first version of `lab/divergence/jac_ladder.py` built a bare `MORPHTransformer` and called
`load_state_dict(strict=False)`. The core MLP is ternarised by a weight PARAMETRIZATION
applied after construction, so the unquantised model's key is `..._cms.weight` while the
checkpoint's is `..._cms.parametrizations.weight.original`; `torch.compile` also puts
`_orig_mod.` in the checkpoint's path and not in the script's. Every MLP tensor was dropped
in silence and stayed at random init. It surfaced because the cap sweep reported that NO
core linear exceeded 2.0 while the run's own log had `sigma_max` at 3.30. The fix is to
apply the quantization transforms first and then use the trainer's own `load_checkpoint`,
which RAISES when a checkpoint tensor finds no home. Every number the first ladder produced
was discarded rather than adjusted.

The general lesson is the one that keeps recurring in this programme: `strict=False` on a
model whose structure is built up in stages is a silent-data-loss switch. `load_checkpoint`
already knew this — its comment calls the same defect "a near-empty resume — latent
theater" — and the analysis script did not use it.

**`CoreSpectralPenalty.include_attn` was a parameter that was accepted and ignored.** A
config could ask for a scope it did not get. Implemented, tested and left off.

**`sigma_max` was already logged on every run and nobody had looked.** The logging-only
construction (`spectral_penalty_log_every: 100`, `lambda` 0) has been in the trainer long
enough that the two 20000-step arms carry it. Their comparison — a healthy seed and a dying
one, differing only by seed — is the sharpest single piece of evidence in this document and
it cost zero GPU time to obtain.
## Not verified

* **That position count is CAUSAL.** The concentration is measured and it separates A1 from
  A0 at identical weights, but the arms that act on it are at the end of this document and
  they change the method, not a defect.
* **Alignment ACROSS the unrolled steps.** What is measured is alignment within one core
  step, across its six blocks. The backward passes through `bptt_depth` = 4 such steps with
  the same operator, and the realized per-block gain (1.43 to 1.88) is above what the
  within-step number alone accounts for. The remainder is presumably the same effect
  compounding; it is not separately measured.
* **Why the cotangent picks the slots it picks.** Nothing here says WHICH slots end up
  carrying it, or whether they are the low-carrier-norm ones (where an RMSNorm Jacobian is
  large), the deep-Poisson-depth ones, or the ones the loss weights most. That is the
  measurement that would turn "more positions" from a lever into a fix.
* **Anything at `alpha_cap` 1.0, the SHIPPED setting.** The deciding pair had to move to 3.5
  because at 1.0 the failure is a coin flip — two of three runs at that setting are healthy,
  including one run here that was expected to fail and did not. No spectral control was
  tested at 1.0 in the real configuration, and a single arm there would prove nothing.
* **The long horizon.** No arm here ran past 7000 steps. `tul_short` runs 20000 and the
  deploy recipe 100000.
* **The deploy recipe.** Everything here is dense TUL at `seq_len` 1024. `base.yaml` runs
  prune, carve and ReMoE routing, none of which has been run with any of this.
* **n = 1 per arm outside the deterministic microcosm.** Kernels-on runs are not
  bit-reproducible and the measured run-to-run spread on the gradient norm is 6.5 %. The
  seed-1 control at `alpha_cap` 1.0 is the standing reminder.
* **Generation quality.** Every number here is teacher-forced CE. `gen_every` was 0 on every
  arm to buy GPU time, so nothing here says anything about what these models GENERATE.
* **The shipped `CoreSpectralProjection` at length.** It is unit-tested, smoke-tested on the
  real model with `verify=true`, and run for 2000 to 7000 steps in three arms. It is off by
  default and nobody should turn it on without reading this document.
