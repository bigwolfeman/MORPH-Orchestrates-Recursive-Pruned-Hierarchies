# Agent Note: the LayerNorm common mode in latent targets and slot cells

Status: proposed

Interim record, written 2026-10-02 15:10 CDT at Wolfe's request ("We need to save a doc on
this. This is important interp work."). The probe is still running. Every number below
comes from `/home/wolfe/morph-scratch/lnprobe/summary_96rows.txt` (96 eval rows, 96
separate fit rows for every fixed statistic), made by
[`lab/divergence/ln_common_mode_probe.py`](../../../../lab/divergence/ln_common_mode_probe.py).
The coda-CE readings (H2 with `--follow`) and the literature section are not in yet.

## Problem

The stage-1 build (2026-10-02) found that the frozen plain model's prelude output, pooled
over a span and LayerNormed with no affine, is about 98 % one shared direction. I then
claimed that every latent arm since 2026-09-30 uses a target like that, and that this
explains three things: the CE tax of the latent pull (H1), a teacher blinded by the common
mode (H2), and false "collapse" readings in our cell instruments (H3). Wolfe added a
puzzle: the latent pull is what makes the slot loop earn depth (detached weight 1: K1-K6
+0.0333; rank-only head, no pull: +0.0052; ungraded fan: +0.0042), so what does the pull
actually do to the cells?

Terms used below:

- **Common direction `u`**: the unit mean of the pooled LN targets on the fit rows.
- **Cell direction `v`**: the unit mean of the slot cells over all passes, per arm.
- **Mean-vector energy share**: the share of a hidden state's squared norm that its
  per-channel mean holds. 1.0 means every token is the same vector.
- **Centred**: the per-coordinate mean over the batch is subtracted first.

## What the probe measured so far

### A. The massive channels belong to the plain model, not to the slot-loop arms

| model | max / median channel mean | mean-vector energy share | top channels |
| --- | --- | --- | --- |
| plain (5k) | 61.2 | 0.911 | 194: -110, 899: -105, 1018: -101, 905: +91 |
| ungraded fan | 7.5 | 0.099 | 621, 660, 132, 370 (about 2) |
| latent-selected arms (8) | 4.7 to 11.1 | 0.066 to 0.213 | 461, 534, 35, 660 recur (about 3 to 4) |

The plain model grows massive activations by step 5000 (four channels near |110|; 91 % of
each token's energy is the shared mean). No slot-loop arm does. The slot-loop arms share
some channels with each other (461, 534, 35, 660), and none with the plain model's four.

### A. So the latent-selected arms' own targets are NOT 98 % one direction

| target | mean pairwise cosine | centred PR | variance per coordinate |
| --- | --- | --- | --- |
| plain prelude, LN (the stage-1 target before standardising) | 0.994 | 8.0 | 0.0058 |
| latent-selected arms' EMA twin, LN (8 arms) | 0.37 to 0.66 | 8.0 to 27.3 | 0.34 to 0.63 |

**Correction to my claim.** The latent-selected arms pull toward their OWN EMA prelude
twin, and that target has a moderate common mode (cosine 0.37 to 0.66), not a 98 % one.
The 98 % reading is true for the stage-1 target (the frozen plain prelude), which the
stage-1 build now standardises. The arms with weight 10 or a teacher-followed forward have
the most common-mode targets (cosine 0.59 to 0.66, PR 8 to 14). The weight-1 and rank-only
arms have the least (0.37 to 0.55, PR 22 to 27).

### H2. The common mode does not blind the teacher (offline picks)

The teacher's pick under the shipped distance agrees with the pick under a per-coordinate
standardised distance on 95 % to 99 % of slots, at every pass, in all eight arms. With the
`u` component removed, agreement is 93 % to 98 %. The share of the between-cell distance
variance that lies along `u` has a median of 0.002 to 0.12. So the teacher ranks cells on
content, not on the common mode. Router-teacher agreement is the same under every distance
(for example detached weight 1, pass 1: 0.518 shipped, 0.522 standardised).
**H2 is refuted offline.** The coda CE when the loop follows each teacher is still to come.

### H1. The pull's residual is not shared across slots

For every arm, the head's mean output matches the target's mean (cosine 0.999). The
residual `g(cell) - z` has a shared part of 0.002 to 0.003 in the weight-1 and weight-10
arms (0.05 to 0.11 in the teacher-followed arms), and 2 % to 7 % of it lies along `u`. The
exit loss's gradient on the cells is also not shared across slots (coherence 0.001 to
0.006, except the teacher-followed arms at about 0.12). So the pull does not mostly push
every cell toward one constant vector. **H1 as I stated it is not supported.** The CE tax
needs another explanation.

### H3. Cross-slot cosine is almost all common mode; within-slot cosine partly

| arm | within-slot cell cosine raw / centred | cross-slot cosine raw / centred |
| --- | --- | --- |
| ungraded fan | 0.127 / 0.074 | 0.196 / 0.004 |
| detached weight 1 | 0.976 / 0.665 | 0.950 / 0.001 |
| joint weight 1 | 0.958 / 0.813 | 0.799 / 0.001 |
| rank-only head | 0.965 / 0.960 | 0.170 / 0.008 |
| detached weight 10 | 0.945 / 0.503 | 0.935 / 0.002 |

Within-slot effective rank does not change under centring (it is centred by construction).
**H3 is supported for cosines.** A raw cross-slot cosine near 0.9 means nothing about
content: centred, it is 0.001 to 0.015 in every arm. Within-slot raw cosines overstate how
alike the four cells are in the pulled arms (0.976 raw, 0.665 centred), but not in the
rank-only arm, whose four cells really are near-copies (0.960 centred).

### The puzzle. In the pulled arms, passes 2 to 6 grow one shared direction

Per pass, the cells' norm, the cosine of the mean cell to `v`, and the between-slot and
within-slot parts of the centred cells (RMS):

| arm | pass | norm | cos to `v` | between-slot | within-slot | ridge R^2 to target |
| --- | --- | --- | --- | --- | --- | --- |
| detached weight 1 | 0 | 34.0 | 0.03 | 2.7 | 33.9 | 0.063 |
| | 1 | 26.3 | 0.05 | 2.6 | 26.1 | 0.077 |
| | 2 | 125.9 | 0.96 | 33.8 | 21.6 | 0.091 |
| | 5 | 216.1 | 0.97 | 52.8 | 30.0 | 0.092 |
| rank-only head | 1 | 20.4 | 0.05 | 3.2 | 20.1 | 0.094 |
| | 2 | 42.2 | 0.37 | 36.8 | 13.3 | 0.093 |
| | 5 | 79.4 | 0.40 | 72.7 | 10.6 | 0.101 |
| ungraded fan | 1 | 32.8 | 0.03 | 4.5 | 32.5 | |
| | 6 | 76.3 | 0.25 | 40.3 | 61.8 | |

In every latent-pulled arm (detached and joint, weights 1 and 10, teacher-followed), pass
2 is a jump: the step is 3.7x to 4.7x the cells' norm, 86 % to 93 % of it along `v`, and
the cells end up at cosine 0.93 to 0.97 to `v`. Passes 3 to 6 keep growing the norm along
`v` with shrinking steps (relative step 0.6, 0.4, 0.3). At the same time the between-slot
part grows 10x to 20x (the slots become distinct from each other). The ridge probe to the
standardised target gains only from pass 0 to pass 2 and is flat after.

My reading, not yet tested: the pull makes the loop's later passes pump one shared
direction, which is scale growth (the same family as the `loop/core_gain_t0` failure mode
in the root CLAUDE.md). The K1-K6 "contribution" in these arms may partly be the coda
reading cell scale or position along `v` rather than new span content. The rank-only head
grows `v` much less (cosine 0.40) and its four cells collapse to near-copies (within-slot
11 against 20 at pass 1). The ungraded fan grows slowly and keeps its four cells apart.

The probe also has a `--vablate` mode (remove the `v` component from the cells and
measure the coda CE). That reading is the direct test of this, and is still running.

## Proposal

1. Finish the probe: H2 with `--follow` (coda CE when the loop follows the standardised
   teacher), and the `v`-ablation on detached weight 1. If removing `v` costs little CE,
   the depth those arms earn is scale, not content.
2. Report every cell cosine centred from now on. Raw cross-slot cosines are retired as
   evidence of collapse.
3. Keep the stage-1 standardisation of the frozen plain targets (built 2026-10-02). It is
   needed there: that target has variance 0.006 per coordinate under LN.
4. Do not add standardisation to the latent-selected arms' EMA targets on the grounds of
   H1 or H2: both are not supported. Their targets are not dominated by a common mode.

## Alternatives considered

- **Standardise the EMA targets now and rerun the rank-only arm** (my first proposal, sent
  to Wolfe at 14:15). H2 offline says the pick would change on 1 % to 5 % of slots, so the
  rerun would cost 75 minutes of GPU to move almost nothing. Dropped unless `--follow`
  says otherwise.
- **Treat the massive channels in the plain model as a defect to fix.** They are the known
  massive-activation pattern (Sun et al. 2024) and the plain model may use them as a bias
  or attention sink. Not a target of this note.

## Acceptance criteria

- The `--follow` and `--vablate` readings are added with paired CIs.
- The literature entries (massive activations, target normalisation in JEPA-family
  methods) are linked from here.
- Past filings that used raw cross-slot cosine as evidence of collapse are listed, with
  the centred value next to them.

## Risks

- 96 rows. The agreement rates and cosines are stable across passes and arms, but the
  ridge R^2 values are small and have no CI.
- `v` is per-arm (the mean of that arm's own cells), so "cosine to `v`" is high by
  construction once a common component dominates. The between-slot growth is the reading
  that does not depend on that choice.
- The plain model's massive channels make the frozen plain targets a different object
  from the EMA targets. Results from stage 1 do not transfer to the latent-selected arms
  without that caveat.
