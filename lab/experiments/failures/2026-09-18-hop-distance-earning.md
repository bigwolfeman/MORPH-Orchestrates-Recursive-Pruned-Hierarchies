# Experiment: does the slot loop earn depth only where the content is HOPS away?

Status: failure
Date: 2026-09-18
Owner: Claude (session f9558148), sub-agent HopProbe, for Wolfe.

Frozen before any real-checkpoint number exists. At filing time the probe has run on no
checkpoint on any host; the only numbers in this tree are the 17 CPU unit tests of
`tests/test_hop_distance_probe.py` and the Hydra compose of the three configs below.

Arc: [`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).
Nearest prior record: [`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md).

## Question

Twelve slot-loop arms read a token K1−K6 of about +0.002 nats while the plain looped model
reads 0.033. Every stability lever is on both, so the flat K-curve is not a stability
story. The hypothesis on trial is mechanical: **a looped transformer earns depth through
GATHER** — pass k reads pass k−1 states of OTHER positions, one more attention hop per
pass — so a pass can only be worth something to a token whose needed content is more than
one hop away.

Under the strict geometry (`morph/model/tul_layout.py::tg_strict_allow`) the routes are
exactly:

| stage | relation |
| --- | --- |
| prelude | a token sees its own span's tokens; a cell sees its own span and its own earlier cells |
| loop | cell k at pass t reads cells k−`loop_reach` … k at pass t−1 (`loop_reach` 0 = every earlier cell) |
| coda | a token of span j sees its own span plus the prefix cells of earlier slots — all (`tg_coda_prefix_reach: all`) or only cell j−1 (`prev`) |

So on the SHIPPED strict arm (`all`, `loop_reach` 0) span i's content reaches a token of
span j in ONE hop at every depth. That arm cannot show hop-graded earning even if the
hypothesis is completely right, which makes it the null control and the instrument check,
not the test. The test is `prev` + `loop_reach` 1, where span j−h reaches cell j−1 only at
pass h−1.

The question in one line: **is per-token depth earning graded by the hop distance of the
span that token needs, and does CE(d) plateau at d ≈ h−1 on the arm where that distance is
forced?**

## Hypothesis

Depth earning is depth DEPENDENCE created by reachability. A token earns from pass k only
when the content it needs cannot arrive in fewer than k hops. On `strict` every span is one
hop from every later token, so every bin reads flat and the twelve flat K-curves are
explained by the geometry rather than by the loop failing to learn. On `prev_reach1` the
far bins must earn, and their CE(d) must stop falling at about the depth the chain needs.

If the far bins on `prev_reach1` read flat too, gather is not the mechanism and the
hypothesis is dead — on the one arm built so that gather is the ONLY route.

**The prior, in theorem form.** Zhu et al., *Reasoning by Superposition: A Theoretical
Perspective on Chain of Continuous Thought*, NeurIPS 2025
([pdf](https://proceedings.neurips.cc/paper_files/paper/2025/file/72c363c2a573ca2128bd176d3317696b-Paper-Conference.pdf)),
prove that a two-layer transformer with D continuous-thought steps solves directed-graph
reachability when D is the graph diameter, each thought holding a superposition of BFS
frontiers. That is exactly "depth equals hop distance", and it is why the plateau in P3 is
predicted at d ≈ h−1 rather than at some arbitrary depth: on `prev_reach1` the cell chain
IS the graph, the loop pass IS the continuous thought, and h−1 is the diameter a token of
span j must cross to reach span j−h. The theorem says the depth is sufficient; it says
nothing about whether a model trained on web text learns to use it, which is what this
experiment measures. The numeric predictions below are unchanged by the citation.

## Predictions

Numbers, not directions. `K1−K6` is (mean CE at forced depth 1) − (mean CE at forced depth
6) over the tokens of a bin, paired over rows with the sweep's bootstrap; "flat band" means
the point estimate lies in [−0.010, +0.010] nats.

**P1 — `slot-spandec-strict` @ 5000 (reach `all`, loop_reach 0): every hop bin is flat.**
For every bin h ∈ 1…6, K1−K6 lies in the flat band, and
`max_h (K1−K6) − min_h (K1−K6) < 0.015` nats. The bins do not rise monotonically with h
(Spearman ρ between h and K1−K6 over the six bins, |ρ| < 0.6).

**P2 — the localiser has teeth on the same checkpoint.** The corruption profile is NOT
flat: mean ΔCE(1) > 0.05 nats, ΔCE(1) > ΔCE(6), and the own-span reading (the h = 0
forward, which corrupts each span's first half and scores its second) is the largest of the
seven, > 0.15 nats. **If ΔCE(1) < 0.02 nats the localiser is inert, h\* is noise, and P1
and P3 cannot be read at all** — that is a protocol failure, filed in `failures/`, not a
null result about depth.

**P3 — `slot-spandec-strict-prev-reach1` @ 5000 (reach `prev`, loop_reach 1): the far bins
earn, and they plateau.**
* bins h ≥ 3: K1−K6 ≥ +0.020 nats, and at least 4x the h = 1 bin's value;
* bins h = 1 and h = 2: inside the flat band (their content is reachable at pass 0 or 1);
* the plateau, in the h = 5 and h = 6 bins: `CE(h−1) − CE(6) ≤ 0.20 · (CE(1) − CE(6))`.
  The curve has spent most of its fall by the depth the chain needs and little after.

**P4 — `slot-spandec-strict-reach1` @ 5000 (reach `all`, loop_reach 1): flat, like P1.**
The coda still reads every cell directly, so a reach limit INSIDE the loop forces no hop.
Every bin in the flat band and the spread under 0.015. If this arm shows P3's grading, the
grading is not about hops and the whole reading is rejected.

**P5 — the planted copy pair.** g = 0 (the rare source planted inside the scoring token's
own span) is the positive control: `CE_control − CE_planted(g=0) ≥ 0.5` nats at every
depth on every arm. **If that benefit is < 0.1 nats the model cannot copy a rare id at all,
the planted instrument is inert on this checkpoint (a floor effect at ln V ≈ 10.8), and no
g ≥ 1 cell says anything** — stated in advance so a flat planted table is not read as
evidence. Given a working g = 0:
* on `strict` and `reach1`: `benefit(g, 6) − benefit(g, 1)` in [−0.05, +0.05] for every g;
* on `prev_reach1`: `benefit(g, 6) − benefit(g, 1) ≥ +0.10` nats for g ≥ 3, and inside
  [−0.05, +0.05] for g = 1.

**Falsification, stated both ways.** P1 or P4 failing means the localiser is binning on
something other than hop distance (row position is the obvious candidate) and the
instrument is rejected before P3 is read. P3 failing means depth earning is not
gather-limited even where gather is the only route, and the gather hypothesis is dead.

## Method

### The instrument

`lab/divergence/hop_distance_probe.py`, built in this change, two instruments in one file.

1. **Corruption localiser.** For c = 0…H (H = 6) one forward replaces the tokens of every
   span i ≡ c (mod H+1) with tokens taken from another row of the same batch. `slot_layout`
   is a forward ARGUMENT, so the boundaries, the cells, the masks and every shape are
   byte-identical to the clean forward and only the token content moves. A token of span j
   is scored in the H phases that do not corrupt its own span, and in phase c its nearest
   corrupted span sits (j − c) mod (H+1) spans back, so each token gets exactly one reading
   at each distance 1…H. ΔCE(h) = that CE minus the clean CE at the same position and the
   same depth; h\* = argmax_h ΔCE(h). The clean per-token K-curve is then binned by h\*.
   Three readings are dropped and each is named in the JSON: a corrupted token itself, a
   token whose LABEL sits in a corrupted span, and a token of span j at a distance h > j
   (the phase corrupted nothing that token can causally read — left in, that wrap-around
   drags h\* down for the first H spans of every row). Only tokens with a finite reading at
   every distance 1…H are binned, so a binned token has span index ≥ H.
   The confound that remains, named: phase c corrupts every span at distance ≡ h (mod H+1)
   at once, so ΔCE(h) is h together with h+7, h+14 … Distance 0 is unmeasurable this way
   and gets its own forward (corrupt each span's first half, score the second); it is
   reported apart and kept OUT of the argmax.
2. **Planted copy pair** (`--planted`). A token id absent from the whole batch is written
   at a random position of span s−g AND as the first token of span s+1; the CE of
   predicting that repeat is read at the LAST token of span s, the position whose label it
   is (`pack_tul_row`: `labels[tok_pos] = ids[i+1]`). The scoring positions are identical
   for every g and for the no-source control, so the copy benefit is paired at the token
   level. Sites are spaced H+2 spans apart so no two plants share a span. This instrument
   assumes nothing about what a token needs and does not depend on the localiser.

Depth forcing is `core_depth_sweep.py`'s (`tul.slot_mean_depth`, `slot_max_depth`, and
`slot_depth_fixed` on a k-fixed arm). Depth 0 is refused: `_sample_slot_depths` reads
`tc.slot_mean_depth or self.cfg.mean_depth`, so 0 silently becomes the trained mean. The CE
map comes from `core_depth_sweep.ce_maps`, i.e. from `model.tul_forward_ablated`, so the TG
masks are the model's own — a bare `_tul_front` scores a strict arm from an unrestricted
prelude and flips the sign of the reading.

Gate: `tests/test_hop_distance_probe.py`, 17 CPU tests over the schedule, the wrap-around
drop, the plant (through the REAL packer) and the aggregation.

### Runs

Probe settings for every arm: `--rows 480 --batch 6 --depths 1,2,3,6,9,12,16 --hops 6
--planted --seed 0`, seq 1024, on the Spark or the 3070 (`docs/cookbook/running-probes-on-
the-second-host.md`), never on the 5090.

| arm | config | checkpoint | role |
| --- | --- | --- | --- |
| `slot-spandec-strict` | `tul_slot_spandec_strict` | on disk, step 5000 | P1, P2, P5 — null control + instrument check |
| `slot-spandec-strict-prev-reach1` | `tul_slot_spandec_strict_prev_reach1` | **must be retrained** | P3 — the test |
| `slot-spandec-strict-reach1` | `tul_slot_spandec_strict_reach1` | **must be retrained** | P4 — the second control |

The two reach arms ran on 2026-09-12 (K1−K6 0.013–0.016 against 0.002 for plain strict) and
their checkpoints were purged in the 2026-09-13 retention pass; only the configs remain.

### The retrain, which Wolfe must approve before anything runs

Two 5,000-step runs, configs UNCHANGED, one at a time on the 5090 (a run is live at filing
time, so these are queued, not launched). No training was started by this change.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/wolfe/morph-to PYTHONUNBUFFERED=1
cd /home/wolfe/morph-to
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
Q=/home/wolfe/morph-scratch/arc

mkdir -p $Q/slot-spandec-strict-prev-reach1
$PY -m morph.training.train --config-name tul_slot_spandec_strict_prev_reach1 \
  hydra.run.dir=$Q/slot-spandec-strict-prev-reach1/hy \
  training.grad_probe_path=$Q/slot-spandec-strict-prev-reach1/probe.jsonl \
  wandb.name=slot-spandec-strict-prev-reach1

mkdir -p $Q/slot-spandec-strict-reach1
$PY -m morph.training.train --config-name tul_slot_spandec_strict_reach1 \
  hydra.run.dir=$Q/slot-spandec-strict-reach1/hy \
  training.grad_probe_path=$Q/slot-spandec-strict-reach1/probe.jsonl \
  wandb.name=slot-spandec-strict-reach1
```

Checkpoints land in `/home/wolfe/morph-to/checkpoints/morph/<wandb.name>/step_5000.pt`.

### How the record closes

Fill Results and Verdict from the three JSONs plus their npz, then file under
`successes/` if P1–P5 held as written and `failures/` otherwise — including the two
protocol failures P2 and P5 name in advance. The predictions above are not edited after a
run; a method change gets a numbered `## Amendment N` with its date.

If the reading changes what ships, an Agent Note goes in the same change.

## Amendment 1 — 2026-09-18, the Zhu et al. prior

Added the paragraph "The prior, in theorem form." to Hypothesis. It cites Zhu et al.,
NeurIPS 2025, as the reason the P3 plateau is predicted at d ≈ h−1 and not at an arbitrary
depth. **No prediction was changed, added or removed**, and the frozen commit that carries
P1–P5 unedited is `e8e252e`.

Ordering, stated because it matters: the paragraph was written AFTER the 24-row 3070 smoke
on `slot-spandec-strict` @ 5000 had run (`EXIT=0`, output in
`/mnt/bigdata/morph-instruments/results/hop_smoke_strict_5000.*`). The smoke is a
functional check of the probe at 1/20 of the pre-registered row count, not a run of this
experiment, and the citation is a literature prior rather than a reading of it — but the
order is on the record so a reader can judge that for themselves. The smoke's own numbers
are reported when this file closes, beside the 480-row run they do not replace.

## Results

Three runs of `lab/divergence/hop_distance_probe.py` on the 3070 (MORPH checkout `99bd7c9`,
480 rows, batch 4, depths 1,2,3,6,9,12,16, hops 6, planted, seed 0), 2026-09-19.
Artifacts: `../results/2026-09-18-hop-distance-probe/` (one JSON and one `.txt` log per arm;
the 24-row smoke that validated the scorer is superseded by the 480-row strict run). The
retrained arms `prev-reach1` and `reach1` reached step 5000 healthy on the 5090 queue
(whole-arm K1−K6 +0.0150 and +0.0013 on the runner's sweep). Intervals are the row
bootstrap over 480 rows.

**Localiser (P2), all three arms:** own-span 0.53 / 0.53 / 0.53, ΔCE(1) 0.19 / 0.19 / 0.19,
ΔCE(6) 0.003 / 0.003 / 0.001. Teeth on every arm.

**Per-bin K1−K6** (tokens binned by the span the localiser says they depend on most):

| h | strict (reach all, loop 0) | reach1 (reach all, loop 1) | prev-reach1 (reach prev, loop 1) | prev-reach1: where the fall happens |
|---|---|---|---|---|
| 1 | +0.0019 [0.0014, 0.0023] | +0.0014 [0.0008, 0.0019] | +0.0109 [0.0094, 0.0123] | flat |
| 2 | +0.0031 [0.0025, 0.0037] | +0.0017 [0.0011, 0.0023] | −0.0169 [−0.0189, −0.0147] | depth hurts (4.480 → 4.497) |
| 3 | +0.0025 [0.0019, 0.0032] | +0.0044 [0.0036, 0.0053] | +0.0640 [0.0610, 0.0671] | pass 2 (4.550 → 4.485), flat after |
| 4 | +0.0010 [0.0003, 0.0017] | +0.0020 [0.0011, 0.0029] | +0.0535 [0.0503, 0.0566] | pass 3 (4.599 → 4.558) |
| 5 | +0.0004 [−0.0005, 0.0012] | −0.0005 [−0.0014, 0.0003] | +0.0288 [0.0255, 0.0320] | between pass 3 and 6 (4.533 → 4.498) |
| 6 | −0.0007 [−0.0015, 0.0001] | −0.0032 [−0.0041, −0.0023] | −0.0203 [−0.0239, −0.0168] | depth hurts (4.505 → 4.526) |
| spread | 0.0038 | 0.0076 | 0.0843 | |
| Spearman(h, K1−K6) | −0.83 | −0.49 | −0.14 | |

Token counts per bin on `prev-reach1`: 183k, 73k, 53k, 44k, 35k, 33k. The token-weighted mean
of the bins is +0.016, the runner's whole-arm reading.

**Planted copy pair** (benefit = CE_control − CE_planted, 2932 sites; > 0 means the copy was
used). g = 0 positive control: strict 0.107, reach1 0.125, prev-reach1 0.162 (all depths
within 0.01). On strict and reach1 every g from 1 to 6 is readable at depth 1 (0.063 → 0.025
falling with distance) and constant across depth (|benefit(6) − benefit(1)| ≤ 0.005 for every
g). On `prev-reach1`:

| g | d1 | d2 | d3 | d6 | d9 | d16 | benefit(6) − benefit(1) |
|---|---|---|---|---|---|---|---|
| 1 | +0.102 | +0.088 | +0.084 | +0.078 | +0.076 | +0.073 | −0.024 |
| 2 | +0.107 | +0.070 | +0.056 | +0.047 | +0.044 | +0.045 | −0.060 |
| 3 | 0.000 | +0.064 | +0.052 | +0.037 | +0.034 | +0.033 | +0.037 |
| 4 | 0.000 | 0.000 | +0.040 | +0.033 | +0.030 | +0.031 | +0.033 |
| 5 | 0.000 | 0.000 | 0.000 | +0.027 | +0.024 | +0.028 | +0.027 |
| 6 | 0.000 | 0.000 | 0.000 | +0.017 | +0.019 | +0.025 | +0.017 |

The zeros are exact: the probe computes every (g, depth) cell (`_planted`, no mask), and
when the source is out of reach the control and planted forwards are bit-identical under the
strict geometry. A copy g spans back first becomes readable at depth g − 1.

Scorecard:

- **P1: two clauses hold, one fails on the letter.** Every strict bin is in the flat band
  (largest |K1−K6| 0.0031) and the spread is 0.0038 < 0.015. Spearman is −0.83, outside the
  |ρ| < 0.6 clause: the bins FALL by 0.004 nats from h = 1 to 6. The clause guarded against a
  rise with h (row position); the trend has the opposite sign and lives inside the band.
- **P2 HOLDS** on every arm.
- **P3: the shape holds, the numbers do not.** h = 3 and 4 clear +0.020 and 4x the h = 1 bin
  (5.9x, 4.9x); h = 5 clears +0.020 but not 4x (2.6x); h = 6 is NEGATIVE. The near bins are
  not in the flat band (h = 1 +0.011, h = 2 −0.017). The plateau clause is unreadable: it
  needs CE at depths 4 and 5, which the sweep did not include. What the bins do show is the
  predicted arrival depth: the fall for h = 3 is complete at pass 2, for h = 4 at pass 3.
- **P4 HOLDS.** `reach1` is flat in every bin (spread 0.0076, ρ −0.49), and its planted
  benefits do not move with depth. The staircase is not row position.
- **P5: the control is weak, the shape holds, the bar is missed.** g = 0 reads 0.107 to
  0.162, above the 0.1 inert floor and far below the 0.5 the bars assumed. On strict and
  reach1 every g sits inside [−0.05, +0.05] (holds). On `prev-reach1`, g ≥ 3 is positive at
  +0.017 to +0.037 (bar +0.10: missed) and g = 1 is −0.024 (inside the band: holds). The
  arrival depth g − 1 was not a numbered clause and is the cleanest reading in the table.

## Verdict

Failure by the letter: P3 and P5 miss their frozen numbers and P1 misses one clause. Not
inconclusive on the question. The gather hypothesis said depth earning appears only where
the geometry forces content to travel through the loop, arriving at pass h − 1; the
`prev-reach1` staircase (h = 3 at pass 2, h = 4 at pass 3, h = 5 by pass 6; planted copies
exactly invisible until pass g − 1) is that prediction, and the two controls are flat at the
same rows with the same localiser. This is the first slot-loop arm in the tree whose K-curve
has structure, and the family's +0.002 came from geometries that never forced a hop.

What the frozen bars got wrong, and what the method could not separate:

1. **Magnitude falls with h and turns negative.** The bars assumed a far-bin gain that does
   not decay (≥ +0.020 at every h ≥ 3). The carry decays with hops, and at h = 6 and h = 2
   depth HURTS. The same shape appeared in the toy the same day
   (`../successes/2026-09-19-toy-eliminate6-hop-distance.md`: an untrained six-hop carry decays
   to 0.668 by hop 4 and training repairs it). Whether the loss at h = 2 and h = 6 is a
   fixed-capacity cell being overwritten by farther content (dilution) or a map that degrades
   near content with every pass is not distinguishable here: both predict the same K-curve.
2. **The plateau clause needs depths 4 and 5.** Not swept; unreadable, not falsified.
3. **The planted signal is weak.** A single rare token gives g = 0 a 0.11 to 0.16 benefit,
   so the g ≥ 3 arrivals of 0.02 to 0.04 are a fifth of the control and the +0.10 bar was set
   for an instrument five times stronger than the one that ran.

Next planned experiment: `../planned/2026-09-19-hop-distance-plateau-and-dilution.md`
(every depth 1 to 8 on `prev-reach1`, a two-token planted copy, and the near-bin dilution
read directly).

## Updated hypothesis

The strict slot loop carries content one cell per pass and loses part of it at every hop; a
token h spans from its source earns at pass h − 1 an amount that falls with h (0.064, 0.054,
0.029 at h = 3, 4, 5) and below zero by h = 6 at 5000 steps. Depth earning on this family is
gather-limited AND carry-limited: the geometry decides whether a hop is forced, the per-pass
carry decides how much of it arrives. The levers that follow are the ones that raise the
carry per hop and stop near content from being overwritten (width, a per-pass write gate on
the cell, a target that rewards far content), and every slot-loop arm should be scored on
the per-hop K-curve, since a whole-arm mean of +0.015 was hiding +0.064 and −0.020.
