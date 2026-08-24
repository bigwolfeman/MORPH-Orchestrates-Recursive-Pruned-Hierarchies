# The TUL core takeover: a forward state collapse that no weight-space cure reaches

Status: failure

Filed by outcome, not by mood. The experiment ASKED whether bounding the core map's spectral
norm cures the takeover, and the two predictions that carried it — P4 and P9 — are falsified:
four weight-space interventions all failed and two were worse than doing nothing. The
diagnosis that replaced them, and the one arm that does hold, are the late predictions P10
to P12, each written before the arm it names.

Pre-registration: [2026-08-24-tul-takeover-cure](../planned/2026-08-24-tul-takeover-cure.md)
(three method amendments and three late predictions, each timestamped before its arm).
Mechanism this builds on: [the RCA](../results/2026-08-24-tul-takeover-rca.md).
Procedure: [measuring the core map's operator](../../cookbook/measuring-the-core-map.md).
Decision record: [the takeover is a forward state collapse](../../../.agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md).
Figure: `docs/experiments/figures/tul_takeover_cure.png`.

## Summary

**The slot states are near-degenerate by design, and at the onset the loop starts making
them worse instead of better.** The 50 valid slot states of a row occupy an effective rank of
1.7 to 4.8 in a 1024-dimensional space, with a mean pairwise cosine of +0.39 to +0.71, at
EVERY checkpoint including healthy ones — a slot's input is one SHARED `E_slot` plus a span
bag-mean, and a mean over many token embeddings concentrates. What changes at the onset is
the SIGN of what the core loop does to that rank: at healthy rungs it RAISES it across its
iterations (x1.23 to x1.48, cosine falling), and by step 1850 it LOWERS it (x0.67, cosine
rising). The flip is between steps 1750 and 1800, where the core share goes 0.021 to 0.372.
It is a FORWARD quantity, it needs one no-grad pass, and it is the earliest indicator in this
programme.

**Giving each slot its own input embedding is the largest improvement found, and is still
not a cure.** One config key, `tul.per_slot_embed`, aimed at exactly the degeneracy above and
pre-registered before it reported. At seed 1 it holds for the whole 4000 steps — end core
share **0.0223** against the control's 0.9999, block gain **1.052** against 2.445, validation
CE monotone and finishing at its own minimum. At seed 0 it takes over at step 2225. What
survives both seeds: the takeover is DELAYED from step 1150 to 2225, the validation CE damage
falls from +0.533 to +0.119, and the best CE reached is 0.78 and 0.46 nats BELOW the
control's best-ever 4.8528 on the two seeds. It is left off by default, because a setting
that holds one seed of two is exactly what `ademamix_alpha_cap: 1.0` already is.

**Four weight-space interventions failed first, and two were worse than doing nothing.**
Including a hard projection that pinned `sigma_max` at exactly 1.50 for a whole run and came
out with a realized per-block gain HIGHER than its uncapped control's. No core weight
matrix's spectral gap opens across the onset (median 1.069 -> 1.132, worst gap FALLING), so
there was nothing in the weight spectra for a norm control to grip. Fixing the slot states
brought `sigma_max` down as a consequence — 2.88 at step 1500 against the control's 4.86 —
without any spectral control being active, which is what a cause looks like next to a
symptom.

**Halving the backward's depth helps and does not cure.** `bptt_depth` 4 -> 2 takes 24
applications of the same `J^T` to 12 — four orders of magnitude off the compounding — and
cuts the validation CE damage by 64 %, from +0.533 to +0.192, while the run still turns
around. That is the pre-registered discriminator, and it is what moved the reading from
"backward power iteration" to "forward collapse, sharpened by backward power iteration".

So the takeover is diagnosed further than it was, it is reproducible, it now has a forward
leading indicator that costs one no-grad pass, an entire family of interventions is
eliminated with a measurement that says why, and the best intervention found doubles the
time to failure and improves CE on both seeds without preventing it. It is not cured.

Every claim in this document that a single arm would have supported was checked against a
second one. The headline above said "cured" for thirty minutes, on one arm, until the second
seed came back.

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
([evidence](2026-08-23-morph-bit-reproducible.md)), so a one-run comparison here is a
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

## The forward is already degenerate, and the loop's effect on it flips sign

The probe above measures the BACKWARD. Running the same capture forward — the operating
points the Jacobian probe already collects — and asking what the slot states look like as
GEOMETRY gives the sharpest number in this document.

Effective rank (participation ratio of the squared singular values) of the 50 valid slot
states of a row, in 1024 dimensions, per loop iteration; and their mean pairwise cosine:

| step | effective rank, iterations 0 -> 7 | mean pairwise cosine, 0 -> 7 | core share |
|---:|---|---|---:|
| 1625 | 2.82 2.52 3.53 4.01 4.78 4.51 4.45 **4.18** | +.517 ... **+.390** | 0.016 |
| 1700 | 2.25 2.34 2.40 2.60 3.27 3.45 3.41 **2.77** | +.606 ... **+.504** | 0.012 |
| 1750 | 2.04 1.84 1.91 2.28 2.95 3.27 3.40 **2.98** | +.640 ... **+.482** | 0.021 |
| 1800 | 2.40 2.16 2.13 2.28 2.53 2.53 2.57 **2.42** | +.561 ... **+.585** | 0.372 |
| 1850 | 2.71 1.69 1.94 1.88 1.93 1.89 1.83 **1.81** | +.582 ... **+.697** | 0.890 |
| 1866 | 3.18 2.46 2.97 2.98 2.97 3.11 2.74 **2.77** | +.537 ... **+.479** | 0.961 |

**The slot states are near-degenerate everywhere, healthy included.** Fifty vectors in a
1024-dimensional space with an effective rank between 1.7 and 4.8 and a mean pairwise cosine
between +0.39 and +0.71. That is not the disease; it is the design. A slot's input is one
SHARED `E_slot` plus the bag-mean of its span's token embeddings, and a mean over many
embeddings concentrates, so the 50 slots start nearly parallel and nothing downstream is
given a reason to separate them.

**What changes at the onset is the SIGN of what the loop does to that rank.**

| step | rank, last iteration / first | cosine, last − first | reading |
|---:|---:|---:|---|
| 1625 | 1.48 | −0.127 | the loop DIVERSIFIES |
| 1700 | 1.23 | −0.102 | diversifies |
| 1750 | 1.46 | −0.158 | diversifies |
| 1800 | 1.01 | **+0.023** | neutral — the flip |
| 1850 | **0.67** | **+0.115** | the loop COLLAPSES |
| 1866 | 0.87 | −0.058 | collapsing |

The flip sits between steps 1750 and 1800. The core share over the same two rungs goes
0.021 to 0.372. This is the earliest indicator in this whole programme — earlier than the
core share, earlier than the block-gain fit's r2, and it is a FORWARD quantity, measurable
with one no-grad pass.

**And the cotangent sits on a stable sink.** The top three slots carrying the backward are
the SAME at every one of the six core blocks (agreement 1.0 at five of six rungs), and the
single top slot's share of it rises from 0.18 at step 1625 to 0.54 at 1850.

### What this does to the reading

The picture that fits all of it: the slot states are handed to the loop already
near-parallel; a weight-shared loop applied 6 to 8 times is the oversmoothing regime, and
whether it separates or collapses them is a property the weights drift across; once it
collapses them, the gradient has almost nothing to distinguish slots by, concentrates on a
few, and the backward's 24 applications of the same operator sharpen that concentration
further.

Under that picture the alignment factor and the cotangent's participation ratio are both
SYMPTOMS of a forward state collapse, and every weight-norm intervention failed because it
was treating a symptom's symptom.

Two arms test that reading and both support it. The `bptt_depth` one is the cheap
discriminator: halving the backward's applications from 24 to 12 — four orders of magnitude
off the compounding — reduced the harm by 64 % and did not prevent it, and the FORWARD still
loops 6 to 8 times regardless of `bptt_depth`. The next section is the expensive one: an
intervention aimed at the slot states' degeneracy and at nothing else holds the run, and
brings `sigma_max` down as a side effect.

## The intervention aimed at the measured cause

Pre-registered as P12 at 15:07, one minute after the arm launched and before it reported
anything, on the strength of the forward measurement above.

`tul.per_slot_embed: true` with `per_slot_embed_std: 1.0` replaces the ONE shared `E_slot`
added to every slot with one row per slot INDEX, seated at the embedding-table mean plus
deterministic jitter. It injects up to `min(max_slots, d_model)` of guaranteed input
diversity into exactly the quantity that was measured to be degenerate. It changes nothing
else: same optimizer, same batch, same schedule, same control.

| arm | end core share | block gain | r2 | val CE min | at | val CE @3500 | rise | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `b10-ctrl` | 0.9999 | 2.445 | 0.97 | 4.8528 | 1500 | 5.3855 | +0.533 | TOOK OVER |
| `b10-bptt2` | 0.9992 | 2.784 | 0.98 | 4.9124 | 1500 | 5.1046 | +0.192 | TOOK OVER |
| **`b10-slotembed`** | **0.0223** | **1.052** | 0.48 | **4.0747** | **3500** | **4.0747** | **0.000** | **HELD** |

Its validation CE is monotone for the whole run — 5.5336, 4.9235, 4.5847, 4.4717, 4.4296,
4.0923, 4.0747 — with the minimum at the LAST eval. It never turns around. The core ends
holding 2.2 % of the pre-clip gradient where the control holds 100 %, and the per-block
backward gain sits at 1.052 with a fit r2 of 0.48, which is the healthy signature (flat and
noisy) rather than the geometric one.

And it is not a trade. The cured arm's validation CE, 4.0747, is **0.78 nats below the
control's best-ever value** of 4.8528. It is the best model produced anywhere in this work.

`sigma_max` of the core MLP tells the same story from the side, without being constrained at
all — no spectral control is active on this arm:

| step | `b10-ctrl` | `b10-bptt2` | `b10-slotembed` |
|---:|---:|---:|---:|
| 500 | 1.76 | 1.79 | 1.87 |
| 1000 | 2.45 | 4.19 | 2.81 |
| 1500 | 4.86 | 4.36 | **2.88** |
| 2000 | 5.21 | 4.71 | **2.92** |
| 3000 | 5.82 | 6.25 | **3.05** |
| 3900 | 6.73 | 7.26 | **4.76** |

Four arms spent the day trying to force `sigma_max` down and every one of them failed or
made things worse. Fixing the slot states' input diversity brought it down as a
CONSEQUENCE — flat at 2.9 to 3.1 through the whole window where the control climbs from 4.9
to 5.8 — with no spectral control active. That is what a cause looks like next to a symptom.
Note also that it does eventually climb, to 4.76 by step 3900, so this is not a run in which
`sigma_max` stopped growing; it is one in which the growth stayed on the healthy trajectory
(compare the healthy 20000-step run in the table above, which reaches 4.02 only at step
19900 — this arm is faster than that, and holding).

### The second seed took over — P13 falsified

The confirmation arm was launched at 15:40 and pre-registered before it reported. It does
not confirm.

| arm | seed | first share > 0.5 | end share | block gain | r2 | val CE min | at | val CE end | rise | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b10-ctrl` | 1 | **1150** | 0.9999 | 2.445 | 0.97 | 4.8528 | 1500 | 5.3855 | +0.533 | TOOK OVER |
| `b10-slotembed` | 1 | 3625 | 0.0223 | 1.052 | 0.48 | 4.0747 | 3500 | 4.0747 | 0.000 | held |
| `s0-slotembed` | 0 | **2225** | 0.9998 | 2.501 | 0.97 | 4.3929 | 3000 | 4.5115 | +0.119 | **TOOK OVER** |

So `per_slot_embed` is NOT a cure. It is in the same category as the incumbent
`ademamix_alpha_cap: 1.0` — it holds one seed and loses the other — and this document would
have claimed otherwise on one arm if the second had not been run.

What survives the second seed, and it is not nothing:

* **It delays.** The takeover moves from step 1150 to 2225, roughly double.
* **It reduces the harm.** Validation CE ends +0.119 above its own minimum against the
  control's +0.533, a 78 % reduction, and the `bptt_depth` arm's +0.192.
* **It is a better model on BOTH seeds.** Best validation CE 4.0747 (seed 1) and 4.3929
  (seed 0), against the control's best-ever 4.8528 — 0.78 and 0.46 nats better.
* **It slows the drive it was not aimed at.** `sigma_max` of the core MLP at step 1500:
  control 4.86, seed-1 arm 2.88, seed-0 arm 2.45. It only climbs once the takeover starts
  (seed 0: 2.45 at 1500, 4.71 at 2500, 6.38 at 3500), which is the ordering a cause-to-symptom
  reading predicts.

And the diagnosis is untouched by this. The forward state degeneracy is measured on
checkpoints, not inferred from an arm; the four spectral controls still failed; the
`bptt_depth` discriminator still landed where it landed. What the second seed changes is the
claim about the FIX, from "the cure" to "the largest single improvement found, and still not
enough".

### What this is, and what it is not

Two arms, 4000 steps each, one configuration (`alpha_cap` 3.5, batch 10), kernels on and
therefore not bit-reproducible. One held and one did not.

It also is not free of confounds. It adds `max_slots x d_model` = 65k parameters, 0.02 % of
286M, so the arms are not iso-parameter. And the jitter at seating means the arm does not
start from bit-identical weights.

It is NOT enabled by default, and the second seed is why. A setting that holds one seed of
two is exactly what `ademamix_alpha_cap: 1.0` already is, and this recipe does not need a
second one of those presented as a fix. What it IS worth having is a large CE gain on both
seeds and a doubled time-to-failure, which is why the code ships off rather than not at
all.

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

## The discriminator: halving the number of applications

Pre-registered at 14:51, before the verdict, as the arm that separates two readings of the
same measurements. `model.bptt_depth` 4 -> 2 takes the backward from 24 applications of the
same `J^T` to 12. Same batch, same optimizer settings, same control.

| arm | end core share | block gain | r2 | val CE min | val CE @3500 | rise | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `b10-ctrl` | 0.9999 | 2.445 | 0.97 | 4.8528 @1500 | 5.3855 | +0.533 | TOOK OVER |
| `b10-bptt2` | 0.9992 | **2.784** | 0.98 | 4.9124 @1500 | 5.1046 | **+0.192** | TOOK OVER |

**It did not prevent it.** The core still ends holding 99.9 % of the pre-clip gradient, with
a per-block gain that is HIGHER than the control's, and the loss still turns around. What it
did is reduce the harm by 64 %: +0.192 nats against +0.533 at the same step.

The prediction written before the result was that a backward-power-iteration disease should
make this arm unmissable, because `2.78^12 = 1.1e5` against `2.45^24 = 2.3e9` is four orders
of magnitude off the compounding. It is not unmissable. So:

* **The backward compounding is real and it is not the whole story.** Removing four orders
  of magnitude from it leaves a run that still turns around.
* **The core's SHARE is not the damage.** A gain above 1 compounded even 12 deep still swamps
  a coda that is compounded not at all, so the share saturates either way. Anything that
  reads the share as a severity measure — including this document's verdict rule — is
  reading a saturating quantity.
* **The competing reading gains ground.** The slot states are 57 vectors built from one
  shared `E_slot` plus a span bag-mean, so they start near-parallel, and the FORWARD applies
  the same six blocks to them 6 to 8 times regardless of `bptt_depth`. If the states are
  losing rank across the loop, the backward's concentration is a symptom and halving the
  backward should do roughly what it did: help, and not cure.

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
## What the measurements say to do next

1. **Confirm the cure, then turn it on.** One 4000-step arm at one seed is what stands
   behind `per_slot_embed` today. What would justify making it a default in
   `tul_short.yaml`: a second seed at the same configuration, and one full 20000-step arm
   against `tul-a0` (val CE 3.1749) to show the CE gain is not an early-schedule artifact.
   Both are ordinary runs, roughly 40 minutes and 3 hours on a 5090.
2. **Separate the two things `per_slot_embed` changes.** It adds per-index parameters AND
   jitters them at seating. `per_slot_embed_std: 0.0` keeps the parameters and starts every
   row equal, so the forward at step 0 is identical to the shared version; running that arm
   says whether the cure is the learnable per-slot capacity or the broken symmetry at init.
   It is one config key and one 40-minute run, and it is the sharpest remaining question.
3. **Find out WHY those slots.** The cotangent sits on the same top-3 slots at every core
   block, with the top slot's share rising 0.18 -> 0.54. Nothing here says whether they are
   the low-carrier-norm ones, the deep-Poisson-depth ones, or whatever the coda's
   token-to-slot attention weights most. One forward and one backward on checkpoints that
   already exist.
4. **Watch the rank ratio.** The loop's effect on the slot states' effective rank crosses 1
   between steps 1750 and 1800, before the core share moves. It costs one no-grad forward
   and it is a better abort criterion than anything currently shipped. It is measured by
   `lab/divergence/jac_ladder.py --state-probe` and is NOT yet wired into the trainer.

What the evidence says NOT to do: bound the size of the core weights. Four arms, one control,
one pre-fixed rule, and a spectral-gap measurement that says why. `max_slots` 64 -> 128 is
also not worth the memory — it OOMs at batch 12 and at batch 10, and a typical row uses 57 of
its 64 slots, so the budget is rarely the binding constraint.
## Not verified

* **The cure at n > 1.** One arm, one seed, 4000 steps, one configuration, kernels on and
  therefore not bit-reproducible. It is reported because at that setting the failure is 5 of
  5, not because one arm is enough on its own.
* **WHICH half of the cure works.** `per_slot_embed` adds per-index parameters AND jitters
  them at seating. Nothing here separates "the model can now tell its slots apart" from "the
  symmetry was broken at step 0".
* **That the state collapse is CAUSAL rather than merely upstream.** The cure is aimed at it
  and works, which is the strongest evidence here, but it is one arm and the intervention
  changes the architecture rather than isolating the variable.
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
