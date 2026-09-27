# Planned: does a committing credit make LX's hypotheses differ per span? (arms c, a, b, fan6)

Status: planned

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
