# Agent Note: the LayerNorm common mode in latent targets and slot cells

Status: proposed

Started 2026-10-02 15:10 CDT as an interim record at Wolfe's request ("We need to save a
doc on this. This is important interp work."); completed the same day at 15:40 with the
coda-CE readings, the cell ablation and the literature. Every number below comes from
[`lab/experiments/results/2026-10-02-layernorm-common-mode/summary_96rows.txt`](../../../../lab/experiments/results/2026-10-02-layernorm-common-mode/summary_96rows.txt)
(the JSONs beside it; 96 eval rows = the exploration ledger's first 96 rows, and 96
separate fit rows for every fixed statistic), made by
[`lab/divergence/ln_common_mode_probe.py`](../../../../lab/divergence/ln_common_mode_probe.py)
on the step-5000 checkpoints, one GPU hold per checkpoint (`run_gpu.sh`, `run_vab.sh` in
the results folder hold the exact commands). Each arm is ONE seed. Sanity check: the
probe's "follow the shipped teacher" reading on detached weight 1 is +0.0022 nats, the
same as the ledger's teacher reading on 480 rows (+0.0022 [+0.0017, +0.0028]).

Literature: [massive activations, outlier and rogue dimensions](../../../../docs/references/attention/massive-activations/massive-activations.md)
and [target normalisation in latent prediction](../../../../docs/references/regularization-objectives/target-normalisation/target-normalisation.md).

## Problem

The stage-1 build (2026-10-02) found that the frozen plain model's prelude output, pooled
over a span and LayerNormed with no affine, is about 98 % one shared direction. I then
claimed (the orchestrator; "I" in this note is the orchestrator, the measurements are a
subagent's) that every latent arm since 2026-09-30 uses a target like that, and that this
explains three things: the CE tax of the latent pull (H1), a teacher blinded by the common
mode (H2), and false "collapse" readings in our cell instruments (H3). Wolfe added a
puzzle: the latent pull is what makes the slot loop earn depth (detached weight 1: K1-K6
+0.0333; rank-only head, no pull: +0.0052; ungraded fan: +0.0042), so what does the pull
actually do to the cells?

Terms used below:

- **Common direction `u`**: the unit mean of the pooled LN targets on the fit rows.
- **Cell direction `v`**: the unit mean of the slot cells (HC stream mean, the loop's
  carrier) of the 96 eval rows over all passes, per arm. For the ungraded fan: over the
  exit cells of forced depths 1 to 6.
- **Mean-vector energy share**: the share of a hidden state's squared norm that its
  per-channel mean holds. 1.0 means every token is the same vector.
- **Centred** (H3): the per-coordinate mean over all exit cells of the 96 eval rows is
  subtracted first.
- **Pass index `t`**: 0-based. Pass index 2 is the THIRD pass. **Forced depth `K`**:
  1-based, the number of passes run; the exit of depth `K` is pass index `K - 1`.
- **Pulled arms**: the latent-selected arms whose exit loss sends gradient into the cells
  (detached and joint, weights 1 and 10, the full-read arm, the two teacher-followed
  arms). The rank-only head arm is NOT pulled (its head reads detached cells).

## Measurements (2026-10-02)

### A. The massive channels belong to the plain model, not to the slot-loop arms

| model | max / median channel mean | mean-vector energy share | top channels |
| --- | --- | --- | --- |
| plain (5k) | 61.2 | 0.911 | 194: -110, 899: -105, 1018: -101, 905: +91 |
| ungraded fan | 7.5 | 0.099 | 621, 660, 132, 370 (about 2) |
| latent-selected arms (8) | 4.7 to 11.1 | 0.066 to 0.213 | 461, 534, 35, 660 recur (about 3 to 4) |

The plain model grows massive activations by step 5000 (four channels at |mean| 91 to 110 per token; 91 % of
each token's energy is the shared mean; the top four channels hold 44 % of it). This is
the pattern of Sun et al. 2024: a few fixed channels far above the rest (61x the median
channel mean). This probe reads per-channel MEANS over tokens, not Sun et al.'s
per-activation criterion (magnitude above 100 and about 1000x the median), so "massive
activations" is the family, not a checked match. No slot-loop arm has such channels (top
four channels: 0.8 % to 2.4 % of the energy). The top channels
461, 534, 35, 660 recur in the weight-10, teacher-followed and full-read arms; detached
weight 1, joint weight 1 and the rank-only head have other top channels. None of the
slot-loop arms has the plain model's four in its top ten. The live front and the EMA twin
of each arm have the same top channels (the twin tracks the front).

Why the plain model and not the strict slot-loop arms: unknown. Not measured. A candidate:
the strict geometry resets the prelude at every span, so no token sees a long context,
and massive activations sit at the start token and delimiters (Sun et al.).

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
arms have the least (0.37 to 0.55, PR 22 to 27). The stage-1 build read a centred PR of
2.9 to 5.3 on the plain target; this probe reads 8.0 on 4986 slots (different rows and
row count; same direction of result). The online floor (`fan_lsel_enc_gamma` 0.1 on the
per-coordinate std) is inactive on these targets: their per-coordinate std is 0.58 to
0.79.

### H2. The common mode does not blind the teacher (offline picks)

The teacher's pick under the shipped distance agrees with the pick under a per-coordinate
standardised distance on 95 % to 99 % of slots, at every pass, in all eight arms. With the
`u` component removed, agreement is 93 % to 98 %. The share of the between-cell distance
variance that lies along `u` has a median of 0.002 to 0.12. So the teacher ranks cells on
content, not on the common mode. Router-teacher agreement is the same under every distance
(for example detached weight 1, pass index 1: 0.518 shipped, 0.522 standardised).

A point of algebra the claim missed: `argmin_i |g_i - z|^2` does not change when one vector
is subtracted from both `g_i` and `z`. "Centring" the target alone cannot change the pick.
The common mode can steer the pick only through each cell's own disagreement along `u`
(the `along_u` share above, small) or through the per-coordinate scale (the standardised
distance, 95 % to 99 % the same pick).

**The coda CE when the loop follows each teacher** (`--follow`: `lsel_follow="teacher"`
at every pass and the exit, `lsel_distance` swapped; nats per token, 96 rows, 101877
tokens, paired row bootstrap, 95 % CI):

| arm | shipped teacher vs router | standardised vs shipped teacher | own-stats standardised vs shipped | `u` removed vs shipped |
| --- | --- | --- | --- | --- |
| detached weight 1 | +0.0022 [+0.0010, +0.0034] | +0.0000 [-0.0002, +0.0003] | +0.0004 [-0.0004, +0.0012] | +0.0002 [-0.0001, +0.0005] |
| rank-only head | +0.0048 [+0.0039, +0.0056] | +0.0001 [-0.0001, +0.0003] | +0.0008 [-0.0001, +0.0016] | +0.0002 [-0.0001, +0.0006] |
| joint weight 1 | +0.0051 [+0.0039, +0.0064] | +0.0001 [-0.0001, +0.0004] | +0.0023 [+0.0015, +0.0031] | +0.0001 [-0.0002, +0.0004] |
| detached weight 10 | +0.0034 [+0.0024, +0.0044] | +0.0003 [-0.0000, +0.0005] | +0.0006 [-0.0000, +0.0014] | +0.0002 [-0.0002, +0.0006] |

No common-mode-free teacher beats the shipped teacher on any arm; the own-stats variant is
worse on joint weight 1. Every teacher stays worse than the router.
**H2 is refuted**, offline and on the coda CE. The router-teacher agreement stall
(0.33 to 0.40 at pass index 2 and later on the router-followed arms, down from 0.41 to
0.57 at pass indices 0 and 1) and "following the teacher is worse than
following the router" need another explanation; the common mode is not it.

### H1. The pull's residual is not shared across slots

For every arm, the head's mean output matches the target's mean (cosine 0.999). The
residual `g(cell) - z` has a shared part of 0.002 to 0.003 in the weight-1 and weight-10
arms (0.05 to 0.11 in the teacher-followed arms), and 2 % to 7 % of it lies along `u`. The
exit loss's gradient on the cells is also not shared across slots (coherence 0.001 to
0.006, except the teacher-followed arms at about 0.12). So the pull does not mostly push
every cell toward one constant vector. **H1 as I stated it is refuted** (measured on all
eight latent-selected arms at the exit, eval mode, one batch of 96 rows).

Why, in one line: the head `g` ends in a bias, and its mean output matches the target's
mean to 3 % to 5 % (relative norm) on the router-followed arms. The constant part of the
target is fit by that bias, so it leaves no residual for the cells to chase. What is left
for the cells is the span-specific part of `z`, which the head does not predict (head
R^2 on the standardised target at the exit: -0.16 to +0.07 across the arms). The CE tax of
weight 10 over weight 1 (0.042 detached, 0.104 joint, from the filings) is real, but it is
not a common-mode effect. Not measured here: what that tax IS.

### H3. Cross-slot cosine is almost all common mode; within-slot cosine partly

Exit cells (the last pass's candidates; the ungraded fan at depth 6), HC stream mean:

| arm | within-slot cell cosine raw / centred | cell spread raw / centred | cross-slot cosine raw / centred | within-slot rank | cross-slot PR |
| --- | --- | --- | --- | --- | --- |
| ungraded fan | 0.127 / 0.074 | 1.385 / 1.524 | 0.196 / 0.004 | 1.87 | 15.2 |
| rank-only head | 0.965 / 0.960 | 0.166 / 0.179 | 0.170 / 0.008 | 2.15 | 11.9 |
| joint weight 1 | 0.958 / 0.813 | 0.188 / 0.408 | 0.799 / 0.001 | 2.45 | 13.2 |
| detached weight 1 | 0.976 / 0.665 | 0.142 / 0.597 | 0.950 / 0.001 | 2.29 | 7.8 |
| joint weight 10 | 0.945 / 0.589 | 0.224 / 0.719 | 0.911 / 0.001 | 1.95 | 7.4 |
| detached weight 10 | 0.945 / 0.503 | 0.218 / 0.843 | 0.935 / 0.002 | 2.06 | 7.7 |
| full read (detached) | 0.948 / 0.497 | 0.226 / 0.849 | 0.928 / 0.002 | 2.03 | 10.3 |
| teacher-followed joint | 0.988 / 0.929 | 0.099 / 0.236 | 0.796 / 0.008 | 2.68 | 5.2 |
| teacher-followed detached | 0.992 / 0.889 | 0.078 / 0.297 | 0.901 / 0.015 | 2.70 | 5.5 |

The ranks and the PR are identical raw and centred (they centre by construction).
**H3 is supported for cosines and for `cell_spread`, refuted for ranks.** A raw
cross-slot cosine near 0.9 says nothing about content: centred, it is 0.001 to 0.015 in
every arm. Within-slot raw cosines overstate how alike the four cells are in the pulled
arms (detached weight 1: 0.976 raw, 0.665 centred), but not in the rank-only arm, whose
four cells are near-copies after centring too (0.960). `cell_spread` (logged as
`lsel_cell_spread`) divides by the RMS of the slot mean, which carries the common mode, so
it reads 2x to 4x low on the pulled arms.

The common mode that does this is in the CELLS (the loop's carrier), not in the target
(see the puzzle below).

The logged `val/slot_cell_pairwise_cos` and `val/slot_pairwise_cos` read the cells after
`_readout` (`lm_mixer` + `final_norm`), not the raw carrier. The same reading in that space
(a second run, `readout/` in the results folder, same rows):

| arm | within-slot cosine raw / centred | cell spread raw / centred | cross-slot cosine raw / centred |
| --- | --- | --- | --- |
| ungraded fan | 0.369 / 0.219 | 0.957 / 1.225 | 0.372 / 0.001 |
| rank-only head | 0.985 / 0.979 | 0.102 / 0.124 | 0.314 / 0.001 |
| joint weight 1 | 0.947 / 0.898 | 0.201 / 0.288 | 0.483 / 0.001 |
| detached weight 1 | 0.900 / 0.706 | 0.283 / 0.546 | 0.703 / 0.004 |
| detached weight 10 | 0.834 / 0.566 | 0.374 / 0.735 | 0.704 / 0.004 |

Same verdict in the logged instrument's space: the raw cross-slot cosine (0.31 to 0.70) is
all common mode, and the within-slot cosine overstates sameness in the pulled arms. The
Thought Register's "rank 1.24 of 4, cos 0.94" was not re-measured (no register checkpoint
on disk): its rank stands as read; its cosine is unknown after centring.

### The puzzle. The pulled loops write a fixed bias direction at pass index 2; the depth they earn is not in it

**What the passes do to the cells.** Per pass index `t`: the cells' RMS norm, the mean
cosine of a cell to `v`, the between-slot and within-slot RMS of the centred cells, and a
ridge probe (fit rows to eval rows) from the router's winner cell to the standardised
target:

| arm | t | norm | cos to `v` | between-slot | within-slot | ridge R^2 |
| --- | --- | --- | --- | --- | --- | --- |
| detached weight 1 | 0 | 34.0 | 0.03 | 2.7 | 33.9 | 0.063 |
| | 1 | 26.3 | 0.05 | 2.6 | 26.1 | 0.077 |
| | 2 | 125.9 | 0.96 | 33.8 | 21.6 | 0.091 |
| | 5 | 216.1 | 0.97 | 52.8 | 30.0 | 0.092 |
| rank-only head | 1 | 20.4 | 0.05 | 3.2 | 20.1 | 0.094 |
| | 2 | 42.2 | 0.37 | 36.8 | 13.3 | 0.093 |
| | 5 | 79.4 | 0.40 | 72.7 | 10.6 | 0.101 |
| ungraded fan (depth K = t + 1) | 0 | 32.8 | 0.03 | 4.5 | 32.5 | |
| | 5 | 76.3 | 0.25 | 40.3 | 61.8 | |

**The step at pass index 2** (the step = this pass's candidates minus the state the pass
started from; size relative to that state, its share along `v`, and its coherence = how
much of it is the SAME vector for every cell of every slot), against each arm's K1-K6 from
its filing (480 rows):

| arm | step size | share along `v` | coherence | K1-K6 |
| --- | --- | --- | --- | --- |
| detached weight 1 | 4.72 | 0.93 | 0.90 | +0.0333 |
| full read (detached) | 3.70 | 0.86 | 0.83 | +0.0170 |
| joint weight 1 | 4.04 | 0.79 | 0.75 | +0.0096 |
| detached weight 10 | 3.66 | 0.86 | 0.84 | +0.0095 |
| joint weight 10 | 3.10 | 0.86 | 0.81 | +0.0067 |
| rank-only head | 1.33 | 0.21 | 0.17 | +0.0052 |
| ungraded fan (depth 2 to 3) | 0.56 | 0.21 | 0.08 | +0.0042 |
| teacher-followed joint | 3.36 | 0.81 | 0.81 | -0.2294 |
| teacher-followed detached | 4.02 | 0.88 | 0.88 | -0.1701 |

Every pulled arm takes this step; the two arms without a pull do not. The one-key pair
(joint weight 1 vs the rank-only head, which differ only in whether the head reads
detached cells) gives 4.04 vs 1.33. Later steps shrink (relative 0.2 to 0.6). Over the
seven router-followed arms the step size and K1-K6 rank almost the same (Spearman 0.96;
0.90 over the five pulled ones). One seed per arm, so this is a correlation.
Router-teacher agreement also drops at this pass, in every latent-selected arm (detached
weight 1: 0.52 at pass index 1, 0.36 at 2).

What the pull does NOT do: it does not push the cells along `v` at the exit (the exit
loss's gradient has 0.3 % to 2.3 % of its energy along `v`, H1). The step builds up
through the loop, not at the exit.

Correction to the interim reading: the between-slot growth at pass index 2 (slot content
appears) is NOT specific to the pull. The rank-only arm has it too (3.2 -> 36.8). What is
specific to the pull is the large shared component on top of it.

**The direct test: does the coda read `v` as content or as a bias?** (`--vablate`,
Sun et al.'s mean-vs-zero test applied to the written slot cells, at each forced depth K.)
`mean` sets every written cell's amplitude along `v` to its mean over the eval rows (the
per-input variation removed, the bias kept); `zero` removes the component along `v`. Delta
= reading minus shipped at the same depth, nats per token, 96 rows, 95 % CI. "K6 - K1" is
the K-curve under each write.

| arm | K6 `mean` | K6 `zero` | K6 - K1 shipped | K6 - K1 under `mean` | K6 - K1 under `zero` |
| --- | --- | --- | --- | --- | --- |
| detached weight 1 | +0.0001 [-0.0003, +0.0006] | +0.0117 [+0.0096, +0.0140] | -0.0326 [-0.0360, -0.0295] | -0.0329 [-0.0365, -0.0298] | -0.0246 [-0.0284, -0.0209] |
| full read (detached) | +0.0001 [-0.0001, +0.0003] | +0.0064 [+0.0051, +0.0077] | -0.0159 [-0.0179, -0.0138] | -0.0158 [-0.0178, -0.0138] | -0.0098 [-0.0120, -0.0075] |
| joint weight 1 | +0.0005 [+0.0001, +0.0008] | +0.0140 [+0.0124, +0.0155] | -0.0097 [-0.0113, -0.0080] | -0.0095 [-0.0112, -0.0077] | +0.0041 [+0.0020, +0.0062] |
| detached weight 10 | +0.0001 [-0.0001, +0.0003] | +0.0069 [+0.0056, +0.0083] | -0.0094 [-0.0114, -0.0076] | -0.0092 [-0.0112, -0.0074] | -0.0026 [-0.0048, -0.0005] |
| joint weight 10 | +0.0001 [-0.0001, +0.0003] | +0.0086 [+0.0072, +0.0099] | -0.0065 [-0.0075, -0.0054] | -0.0072 [-0.0083, -0.0060] | +0.0017 [-0.0000, +0.0035] |
| rank-only head | +0.0057 [+0.0047, +0.0068] | +0.0270 [+0.0249, +0.0291] | -0.0052 [-0.0063, -0.0041] | +0.0003 [-0.0011, +0.0019] | +0.0216 [+0.0192, +0.0240] |
| ungraded fan | +0.0006 [+0.0003, +0.0009] | +0.0056 [+0.0045, +0.0067] | -0.0044 [-0.0053, -0.0036] | -0.0038 [-0.0048, -0.0030] | +0.0013 [-0.0001, +0.0027] |

The written cells at K6 sit at cosine 0.87 to 0.91 to `v` in the pulled arms, with a mean
amplitude of 130 to 255 and an amplitude std of 65 to 118 (rank-only: 0.42, 26, 21;
ungraded fan: 0.18, 16, 24).

Readings:

1. **In every pulled arm, `v` is a fixed bias for the coda.** Its large per-input
   variation (std 65 to 118) carries nothing the coda uses: setting it to its mean costs
   at most 0.0005 nats, and the K-curve is unchanged. Removing the bias costs 0.006 to
   0.014 nats. This is the massive-activation signature (Sun et al.: mean-ablation
   harmless, zero-ablation harmful), but written by the slot loop into its own carrier,
   not by the prelude.
2. **So the depth the pulled arms earn is NOT the common mode.** With `v`'s variation
   removed, detached weight 1 still earns -0.0329 from K1 to K6. The earning is in the
   rest of the cell.
3. **In the rank-only arm it is the other way round.** Its whole K1-K6 rides on the
   per-input amplitude along its `v` (-0.0052 shipped, +0.0003 with the amplitude set to
   its mean). Without a pull, the loop's depth signal is ONE scalar per cell; with a pull,
   the depth signal lives off the shared direction, and the shared direction becomes a
   bias.
4. **Removing the bias hurts depth earning in the pulled arms** (joint weight 1: K6 - K1
   goes from -0.0097 to +0.0041). The coda's read of the later passes depends on the bias
   being there.

**What this explains, and what it does not.** The answer to Wolfe's puzzle ("if the pull's
direction is mostly the common mode, why does it make the loop earn depth?") is that the
premise is false for these arms: the pull does not act along the target's common mode
(H1), and the cells' large shared direction is a bias that the depth earning does not use.
What the pull changes is WHERE the loop's per-pass signal lives: in the pulled arms it is
in the non-shared part of the cell; without the pull it is one amplitude. Why the pull
does that, and why pass index 2: **unknown**. Untested candidates: (a) the head reads
`LN(cell)`, which is blind to scale, so a pull through it cannot use an amplitude channel
and rewards content that survives normalisation; (b) the epivol diversity term is charged after
the first `fan_repel_passes: 2` passes only (pass indices 0 and 1, per the `TULConfig`
comment "repel after passes 1..this"), so pass index 2 is the first pass where moving every
cell the same way costs nothing. The ungraded fan carries the same term and takes no such
step, so (b) can at most set WHEN the pull's step happens, not WHETHER; (c) the bias may
be how the loop keeps the written cells far from the coda's one-pass fixed point. None of
these was measured.

### Past filings that used a raw cosine or `cell_spread` for a verdict

Found by a read-only search subagent (2026-10-02); the lines marked "checked" were re-read
by me. Only the step-5000 latent-selected arms and the ungraded fan were re-measured
centred; for every other filing the centred value is NOT known, so the column says so.
Ranks and PRs in this tree are centred (`effective_rank`, `fan_stream_stats`,
`fan/stream_rank_t*`), so a verdict that a rank supports on its own stands.

| filing | the verdict | instrument | centred value | status |
| --- | --- | --- | --- | --- |
| [`2026-10-01-latent-selected-loop-weight1-and-read-all.md`](../../../../lab/experiments/failures/2026-10-01-latent-selected-loop-weight1-and-read-all.md) l. 140 (checked) | the full read's four final candidates are "near-copies" (spread 0.22) | `cell_spread` | 0.849 spread, within-slot cosine 0.497 (this note, same checkpoint) | **weakened**: centred, the four candidates are not near-copies; "reading all four adds near-copies" does not explain the full read's cost |
| [`2026-09-30-latent-selected-loop.md`](../../../../lab/experiments/failures/2026-09-30-latent-selected-loop.md) l. 129 | J-5 "no collapse onto the winner" (spread 0.0997 > 0.05) | `cell_spread` | teacher-followed joint 0.236, detached 0.297 | stands, with a larger margin |
| [`2026-08-27-warmup-sigreg-ntpdrop.md`](../../../../lab/experiments/failures/2026-08-27-warmup-sigreg-ntpdrop.md) l. 160-166 (checked) | the slot "collapse" is ARITHMETIC (bag-mean plus embedding anisotropy), from slot cosine +0.39 to +0.71 | raw cross-slot cosine | not measured | the arithmetic argument is about the raw cosine itself, so it stands as a statement about the cosine; it is not evidence of low content |
| [`2026-08-28-tul-fm1-live-arm.md`](../../../../lab/experiments/failures/2026-08-28-tul-fm1-live-arm.md) l. 132 | SIGReg took target pairwise cosine 0.63 -> 0.00 | raw target cosine | not measured | a raw cosine drop to 0 is what removing a common mode alone does; read with care |
| [`2026-09-13-arc-thought-register.md`](../../../../lab/experiments/failures/2026-09-13-arc-thought-register.md) l. 396-397 (checked), restated in [`2026-09-19-lxtul-fan-streams.md`](2026-09-19-lxtul-fan-streams.md) and `docs/slot-cells-distinct-vs-blurred.md` | the register's cells are copies: cell rank 1.2445 of 4, cosine 0.9432 | `slot_cell_eff_rank` + raw `slot_cell_pairwise_cos` | not measured (no checkpoint on disk) | stands on the rank alone; the cosine's size is unknown centred |
| [`2026-09-19-lxtul-fan4.md`](../../../../lab/experiments/failures/2026-09-19-lxtul-fan4.md) l. 181-213 | norepel collapses (`stream_cos_t6` 0.941) | raw `fan/stream_cos_t*` + `fan/stream_rank_t*` 1.23 to 1.28 | not measured | stands on the rank; the same filing shows a raw cosine of -0.33 at rank ~1.05 (a sign split), so the cosine was never the reliable half |
| the "5.7598 / 0.7104" slot rank / cosine defect: `2026-09-13-the-thought-register.md`, `morph/model/tul_vq.py`, `docs/tul-code-spec.md`, and the 2026-09-13 register-reader, bootstrap and cond4 filings | a row's slot states are near copies | `slot_eff_rank` + raw `slot_pairwise_cos` | not measured | already **retracted** for a different reason (a bare-front probe before 7a24adf; corrected 13.8466 / 0.5201, `docs/tul-loop-contribution-history.md` l. 958, checked); the 0.52 cosine is raw too |

One earlier filing already centred its cosine: [`2026-08-24-tul-core-underdetermined.md`](../../../../lab/experiments/failures/2026-08-24-tul-core-underdetermined.md)
(l. 130: "The centred pairwise cosine is -0.015 everywhere"). The standing rule "Never pass
or fail an arm on a cosine" (`docs/tul-loop-contribution-history.md` l. 1862) is why most
of these verdicts also carry a rank.

## Proposal

1. **Close H1 and H2 for the latent-selected arms.** Their EMA targets are not dominated by
   a common mode, and a common-mode-free teacher does not change the coda CE. Do not add
   target standardisation to these arms on the grounds of H1 or H2. The CE tax of weight
   10 and the teacher-worse-than-router gap stay open, with the common mode excluded.
2. **Keep the stage-1 standardisation of the frozen plain targets** (built 2026-10-02,
   `tul.latent_pre_target_norm: standard`). It is needed there: the plain prelude's LN
   target has 0.006 variance per coordinate and 0.994 mean cosine. Results from stage-1
   arms on that target do not transfer to the EMA-target arms without this caveat.
3. **Retire raw cosines as collapse evidence.** Every cell or slot cosine is reported
   centred (per-coordinate mean over the batch removed) beside the raw one; `cell_spread`
   is reported with a centred denominator. Ranks and PRs need no change. Past filings that
   used a raw cosine for a verdict are listed above.
4. **Treat the pulled arms' shared cell direction as a loop-written bias, not as
   collapse and not as depth.** It is mean-ablation-free and zero-ablation-costly; the
   K-curve does not use its variation.
5. **Next measurement for the puzzle (not run):** the per-direction split of the K-curve.
   Project each pass's winner cell onto the top centred PCA directions (fit rows) and
   mean-ablate them one band at a time at the write; find which band carries the K1-K6 in
   detached weight 1 and whether the rank-only arm's amplitude channel is `v` itself. And
   a one-key training pair to test candidate (b): the pulled arm with
   `fan_repel_passes: 6`.

## Alternatives considered

- **Standardise the EMA targets now and rerun the rank-only arm** (the orchestrator's first
  proposal, sent to Wolfe at 14:15). H2 offline says the pick would change on 1 % to 5 % of
  slots, and `--follow` says the coda CE does not move: on all four arms the standardised
  and `u`-removed teachers sit within 0.0003 of the shipped teacher with CIs that include
  0, and the own-stats teacher is equal or worse (+0.0023 on joint weight 1). Dropped: 75
  minutes of GPU to move nothing.
- **Treat the massive channels in the plain model as a defect to fix.** They are the
  known massive-activation pattern (Sun et al. 2024), which acts as a fixed bias the model
  relies on. Not a target of this note; the stage-1 standardisation already handles them
  in the target.
- **Remove the loop-written bias direction from the pulled arms' cells** (a centring or a
  projection at the write). Rejected for now: zero-ablation costs 0.006 to 0.014 nats and
  removes part of the K-curve. The coda relies on it.

## Acceptance criteria

- Met 2026-10-02: the `--follow` and `--vablate` readings are in, with paired CIs; the
  literature entries are linked; the past filings are listed (section above).
- Open: the per-direction K-curve split (Proposal 5) is filed as its own planned
  experiment before any training run on this question.
- Open: any new latent-target arm states which target it pulls toward and reads that
  target's mean cosine and centred PR in its prereg, so a plain-prelude result is never
  read as an EMA-target result again.

## Risks

- 96 rows, one seed per arm, step-5000 checkpoints only. The H2 and ablation CIs are row
  bootstraps; they do not cover seed variation. The Spearman 0.96 is over seven single-seed
  arms.
- The ridge R^2 values (0.06 to 0.16) have no CI.
- `v` is per-arm (the mean of that arm's own cells), so "cosine to `v`" is high by
  construction once a common component dominates. The between-slot growth and the
  `mean`/`zero` ablations do not depend on that choice of `v` being "right"; they test
  the direction that is there.
- The H3 tables read the carrier (HC stream mean) and, in a second table, the `_readout`
  space of the logged instruments. Both on the exit cells only.
- The `--vablate` `mean` reading uses the eval rows' own mean amplitude (no separate fit
  rows). A mean over 96 rows (about 5000 written cells) is a constant for every input, so
  this does not leak per-input information.
