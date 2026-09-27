# Planned: does a committing credit make LX's hypotheses differ per span? (arms c, a, b, fan6)

Status: failure

Date: 2026-09-26 22:07 (frozen before any of these four runs trained; the only GPU contact was a
one-step eager memory trace and a 30-step compiled smoke per config, 2026-09-26 19:16-19:39,
whose val numbers at step 30 are not read here).
Parents: [`../failures/2026-09-26-lx-selection-ceiling.md`](../failures/2026-09-26-lx-selection-ceiling.md)
(span-level selection over fp01's fixed codes buys +0.0005 to +0.0008 nats over a
token-shuffle null; the mixture beats the best single rollout by 0.025 at 5k),
[`../successes/2026-09-26-slot-channel-arms-ab.md`](../successes/2026-09-26-slot-channel-arms-ab.md)
(a2 = the write-all fan + fp01: gap to plain +0.2181 at 5k, a2 - fp01 -0.0472, K1-K6
+0.0057, worth zero 0.200). Notes: `.agents/notes/proposed/architecture/2026-09-26-lx-hard-credit.md`,
`.agents/notes/proposed/architecture/2026-09-26-lx-fan.md`.

## Question

LX trains the exact mixture `-log (1/K) sum_k exp(-CE_k(span))`. Its gradient weights
rollout k by the posterior `softmax_k(-CE_k)`. The codes start small and alike, so the
posterior starts near uniform and every rollout gets the same gradient: nothing breaks the
symmetry, and the rollouts end as an ensemble. Multiple Choice Learning says a committing
(winner-take-all) credit makes hypotheses specialize. a2 already commits, but over CELLS
(`fan_all_wta_lambda 1.0`). Wolfe's call (2026-09-26): test the three placements.

- **(c) `lxtul-e4probe-fp01-hard`**: fp01 with hard credit over the K=4 rollouts
  (`tul.code_enum_credit: hard`, eps 0.05). One factor against fp01.
- **(a) `lxtul-lxfan4-fp01`**: the codes on a2's write-all fan, soft mixture, NO cell WTA,
  epivol kept, no probe head. Against a2 it changes the code, WTA and the head.
- **(b) `lxtul-lxfan4-wta-fp01`**: (a) plus a2's cell WTA (1.0, select eps 0.05), winner per
  (rollout, slot). Against (a): one factor (WTA). Against a2: the code and the head.
- **(fan6) `lxtul-lxfan6-fp01`**: (a) at 6 cells. Against (a): one factor (cell count).

## Hypothesis

H: a committing credit on the ROLLOUT axis makes the rollouts carry span-level
differences. On (c), the span-coherent part of the selection ceiling (observed Bayes - oracle
minus its token-shuffle null, depth 6) rises from fp01's +0.0005 to at least +0.005.

## Predictions (mine, orchestrator, written before any run)

All CE comparisons are paired on the same 480 rows at depth 6 (`paired_vs_ruler.py`,
`core_depth_sweep.py`), 5000 steps. The cross-arm seed spread is 0.0137 nats.

- **P-c1 (H)**: (c)'s span-coherent selection ceiling >= +0.005 at depth 6. 30 %.
- **P-c2**: (c) - fp01 <= +0.0137 (hard credit does not cost the mixture read more than the
  seed spread). 55 %.
- **P-c3**: (c)'s mixture gain over its best single rollout > fp01's 0.025. 50 %.
- **P-c4**: (c)'s K1-K6 > fp01's +0.0125. 25 %.
- **P-a1**: (a) - a2 <= -0.0137. 35 %.
- **P-a2**: (a)'s K1-K6 > a2's +0.0057. 45 %.
- **P-b1**: (b) - (a) <= -0.0137. 30 %.
- **P-b2**: (b) - a2 <= -0.0137. 40 %.
- **P-f1**: (fan6) - (a) <= -0.0137. 35 %.
- **P-all**: no arm detonates (tripwire `preclip/total > 1e4` at step >= 200, sustained). 80 %.

## Method

- Code at `f685685` (hard credit, LX-Fan WTA, the memory fixes), trained by
  `/home/wolfe/morph-scratch/abc/runner.sh` in queue order c, a, b, fan6; the configs'
  own recipe (5000 steps, batch 6, seq 1024, fp01's losses). Same data order and seed as
  fp01 and a2.
- Per arm, the runner's readouts: `core_depth_sweep.py` depths 1-6 on 480 rows; the gap to
  plain 5k (`paired_vs_ruler.py`, ruler plain-panel-norm-match 5k); `lxtul_e_stage2_score.py`
  against fp01 5k; the notul pair; `worth_profile.py` zero/shuffle on 192 rows.
- Added after the runner, on the same sweeps: `paired_vs_ruler.py` with fp01 5k as ruler
  (P-c2), a2 5k as ruler (P-a1, P-b2), and (a) as ruler (P-b1, P-f1).
- (c) only: `lx_selection_ceiling.py` on its stage-2 npz at depths 1 and 6, with the
  token-shuffle null (P-c1), and the mixture-over-best-single gain from the same scorer (P-c3).
- Diagnostics, not graded: `tul/enum_code_win*`, `tul/enum_win_entropy`, `fan/oracle_gap`,
  `fan/stream_rank_t*`, per-cell worth on (a) and (b).
- A detonation guard kills a run whose probe trips the sustained tripwire; that arm files as
  a failure on P-all with its anatomy.

## Reading rules

- Success on H only if P-c1 holds. P-c2 failing with P-c1 holding means hard credit makes
  the hypotheses differ but the read pays for it; that is not a success of the arm.
- (a), (b), fan6 are read on their gaps; no arm passes or fails on a cosine or a rank.
- A difference inside +-0.0137 is no difference.

## Results (filed 2026-09-27 11:19)

Code `ce1b7dc`, runner `/home/wolfe/morph-scratch/abc/runner.sh`, seed 1, 5000 steps, 480
rows at depth 6. Artifacts: [`../results/2026-09-26-lx-credit-arms/`](../results/2026-09-26-lx-credit-arms/).

| arm | trained | gap to plain | vs ruler (same tokens) | K1-K6 | K3-K6 | worth zero / shuffle |
|---|---|---|---|---|---|---|
| fp01 (ref, filed) | 5000 | +0.265 | | +0.0126 | +0.0003 | |
| a2 (ref, filed) | 5000 | +0.2181 | | +0.0057 | +0.0005 | 0.200 / 0.190 |
| (c) fp01-hard | 5000, HEALTHY (max 581) | +0.2769 [+0.2615, +0.2941] | vs fp01 +0.0118 [+0.0094, +0.0143] | +0.0100 [+0.0093, +0.0108] | -0.0002 [-0.0004, -0.0000] | 0.188 / 0.163 |
| (a) lxfan4 | DETONATED 3311 (guard kill) | none | none | none | none | none |
| (b) lxfan4-wta | 5000, HEALTHY (max 270) | +0.2193 [+0.2038, +0.2368] | vs a2 +0.0014 [-0.0010, +0.0037]; vs fp01 -0.0458 [-0.0484, -0.0433] | +0.0162 [+0.0153, +0.0172] | +0.0020 [+0.0017, +0.0024] | 0.200 / 0.170 |
| fan6 | DETONATED 1545 (guard kill) | none | none | none | none | none |

**(c) selection ceiling** (`lx_selection_ceiling.py` on the arm's stage-2 npz, fp01 read from
the same npz as control; 25,278 spans, 501,106 tokens):

| | fp01 d6 | (c) d1 | (c) d6 |
|---|---|---|---|
| credit KL(credit, uniform) | (0.45, filed) | 0.6633 | 0.6426 |
| Bayes - oracle | +0.0413 | +0.0505 | +0.0499 |
| token-shuffle null Bayes - oracle | 0.0404 | 0.0459 | 0.0451 |
| span-coherent part (observed - null) | +0.0008 | +0.0046 | +0.0048 |
| mixture gain over best single (fixed - Bayes) | +0.0246 | +0.0497 | +0.0468 |
| argmax at d1 == argmax at d6 | 0.8189 | 0.8909 | |

(c)'s single rollouts read 4.4102 / 4.4281 / 4.4003 / 4.3993 against its Bayes read 4.3526;
wins at val spread over the four codes (`enum_win_entropy` 0.975; wins 0.29/0.23/0.26/0.22).

**(b) diagnostics** (val at 5000, not graded): `fan/oracle_gap` 0.0876, `fan/stream_rank_t1`
2.92, `t6` 2.54, `slot_cell_eff_rank` 2.72, `enum_width_gain_best` 0.024. The stage-2 scorer
and the notul pair exit 1 on (b): the scorer needs the parallel head, which LX-Fan refuses.

**Detonation anatomy** ([`detonation_anatomy.txt`](../results/2026-09-26-lx-credit-arms/detonation_anatomy.txt),
25-step medians of the per-step probe):

- (a) lxfan4: the pass-0 write grew early and stayed large (`core_gain_t0` 14.2 at step 1811,
  21 at 2811), the hinge reading sat at 0.91-0.96, while the per-slot maximum climbed
  1.29 -> 1.57 -> 4.53 and pass 1's gain rose 0.63 -> 0.90 -> 3.04 in the last 100 steps; the
  fixed-point residual went 0.04 -> 2.16. First `preclip/total > 1e3` at 3245.
- fan6: `core_gain_t0` 1.4 -> 13 -> 42 from step 1045 to 1540; the hinge reading crossed 1.0 at
  1275 and reached 1.5, charging 44-49 per step for ~250 steps (the C1 "hinge fight"). First
  `preclip/total > 1e3` at 1518.
- (b) at the same steps as (a): `core_gain_t0` 2.7-3.6, hinge 0.976-0.978, per-slot max
  1.18-1.24, penalty < 0.002, never above 1.0.

## Clause by clause

- **P-c1 (H), FAIL, narrowly.** (c)'s span-coherent part is +0.0048 at depth 6 (+0.0046 at
  depth 1) against a bar of +0.005 (fp01 +0.0008 in the same read). Six times fp01's, and still
  under the bar. No CI on the observed-minus-null difference (the null has 5 draws), so the
  0.0002 miss is inside the instrument's resolution: I do not read it as "near success" either.
- **P-c2, holds on the point estimate.** +0.0118 <= +0.0137; the CI upper end is +0.0143.
- **P-c3, holds.** 0.0468 against fp01's 0.0246 in the same read.
- **P-c4, fails.** K1-K6 +0.0100 < +0.0125; K3-K6 is -0.0002.
- **P-a1, P-a2, fail** (no checkpoint: (a) detonated at 3311).
- **P-b1, not readable** ((a) has no checkpoint); filed as a failure.
- **P-b2, fails.** (b) - a2 = +0.0014 [-0.0010, +0.0037]: no difference.
- **P-f1, fails** (fan6 detonated at 1545).
- **P-all, fails.** 2 of 4 detonated; both are the soft-credit fans.

## Verdict

Failure on H and on P-all. Hard credit on the rollout axis does what MCL predicts to the
rollouts, it specializes them: single rollouts lose 0.05 nats against the Bayes read (fp01:
0.025), the credit KL rises 0.45 -> 0.64, and the span-coherent selection value rises from
+0.0008 to +0.0048. It does not reach the bar, it costs +0.0118 nats of mixture CE, and the
decision moves EARLIER: the winning rollout at depth 1 is the winner at depth 6 on 89 % of
spans (fp01 82 %), and K3-K6 is zero. The commit happens in pass 1, not over passes.

The one positive is on the cell axis. (b) matches a2's gap (+0.0014, no difference) with 2.8x
a2's depth use (K1-K6 +0.0162 vs +0.0057) and 4x its K3-K6 (+0.0020 vs +0.0005); the
largest K3-K6 of any strict slot arm at 5k in the ledger rows I checked (fp01 +0.0003, nofp
+0.0009). One seed; not a prediction I made.

Stability, n = 1 per arm: both soft-credit fans detonated and both WTA fans (a2 at 5k and 10k,
(b) at 5k) did not. In (a) and fan6 the pass-0 write grew to 14-42x before the end, as pk4's
did (pk4 lived at 20x); in (b) it stayed near 3x. Whether the WTA term holds that growth is not
measured: this is a correlation over four runs.

## Updated hypothesis

Commit pressure specializes whatever axis it acts on, and it acts in pass 1. On the rollout
axis that buys ensemble diversity but no depth. On the cell axis, with the codes on top, the
loop earns more depth at the same CE. Next reads, in order: (1) (b)'s per-cell worth and its
selection structure over cells (does the depth use live in one cell?); (2) a second seed of
(b), since its depth use is the only positive; (3) the soft fans' pass-0 growth as the
detonation route, read with `xhc_carrier_anatomy.py`-style per-pass norms before any fix.
