# Agent Note: A graded-continuation target for the slot loop

Status: proposed

Date: 2026-09-17. Spec: [`docs/tul-code-spec.md`](../../../../docs/tul-code-spec.md) §17.2.
Prereg: [`lab/experiments/planned/2026-09-17-lctul-graded-target.md`](../../../../lab/experiments/planned/2026-09-17-lctul-graded-target.md).
Parent: [`2026-09-17-lctul-target-slot-loop.md`](2026-09-17-lctul-target-slot-loop.md) (the
code-target family). Wolfe, 2026-09-17: "didn't we also have an arm that was grading
plausible continuations with a grader?"

## Problem

Every target the slot loop has ever been given is a deterministic function of the past:
M-next, the span decoder, the horizon targets, the staged targets, the fitted `oracle_z`,
the per-pass ladder, and now the frozen VAE code `E(span s+1)` regressed with L2. Every
one of them is met in ONE pass. Arm A at 10k reads `code_target_cos_l0` 0.06, `l1` 0.135,
`l6` 0.132: pass 1 does all of it and passes 2-6 add nothing, and the exit cosine has sat
at 0.17 since step 1500 on a cell whose effective rank is 16 against the codes' 75
(`lab/experiments/results/2026-09-17-lctul-target/code_target_mean_tul-code-target_10000.txt`).

The reason is structural, not a tuning failure. A conditional mean is a POINT. The map's
first pass sets the scale and the rest rotate (`morph-loop-is-a-power-iteration`), so one
pass is enough to land on a point, and nothing about a point rewards a second pass. The
code-only prereg says this in its own binding: "P-C6 fails on both → one pass again; the
depth question moves off predicted targets entirely (a target that is computed, not
predicted)".

The two earlier graders in the record both failed for the same reason, from the other
side. `disc` (2026-09-12) asked "is this slot's next span below the BATCH MEDIAN CE?",
which is a property of the TEXT and not of `z`, so the head scored it from the context
alone. `critic` asked whether pass `t` beat pass `t-1` on the real coda loss, and the
coda loss needs the true span, so the label was a tie whenever the two states decoded the
same tokens
([`lab/experiments/failures/2026-09-12-arc-core-token-and-critic.md`](../../../../lab/experiments/failures/2026-09-12-arc-core-token-and-critic.md),
[`2026-09-12-core-token-gradient-and-within-context-critic.md`](2026-09-12-core-token-gradient-and-within-context-critic.md)).

## Proposal

Make the target a COMPUTED object that depends on the loop's own state.

`tul.code_grade: true`, on top of the code-target family (strict slot loop, `TULCodeProj`
cells, frozen `E`, frozen VAE coda). On a graded step:

1. The loop's predicted cells (detached) condition the frozen coda, which SAMPLES `K`
   candidate continuations of span `s+1`, at temperature `code_grade_temp`.
2. A grader that never sees span `s+1` ranks the `K` candidates by mean log-probability
   per token.
3. The frozen `E` encodes the winner. The loop is pushed toward `E(best)` — either by the
   same `2(1 - cos)` regression (`code_grade_loss: best`) or by an InfoNCE over the `K`
   candidate codes with the winner as the class (`code_grade_loss: pref`), which pushes
   toward the winner AND away from the losers.

Three properties this has and no earlier target had. The target is a function of the
loop's current state, because the candidates are drawn from the cell the loop just wrote.
It is not a function of the past alone, because the draw is stochastic and the ranking is
over draws. And it admits refinement: the target is an ARGMAX over proposals, so a pass
that proposes better candidates gets a better target, which is the one shape where pass
`t+1` can beat pass `t` on the same objective.

**The grader must not be the proposing coda with the same cell.** If the grade were the
candidate's log-probability under the cell that generated it, the argmax would be the
cell's own greedy decode, the target would be a fixed point of the current cell and the
term would teach nothing. So the grader reads the cell as ZERO. Two shipped graders:

- `code_grade_grader: "coda_past"` (default) — the frozen coda reading `E`'s TRUE codes at
  every slot EXCEPT the graded slot's own cell, which is zeroed. Those other codes are
  codes of PAST spans, so the grade is context aware and the truth of span `s+1` is
  unreachable. Realised in two passes over the slot-index parity, because one pass cannot
  both zero cell `s` (to grade span `s+1`) and keep it (as context for span `s+2`).
- `code_grade_grader: "coda_zero"` — every cell zeroed, one pass. Under the strict
  geometry the cells are the only cross-span channel, so this grader is context BLIND: it
  scores fluency inside the candidate alone. It is the cheap control that says how much
  of the effect needs context.

Ranking is unbiased by either choice: all `K` candidates of a slot are graded under
identical conditions, so a constant handicap cancels.

**What makes this affordable: the strict geometry.** Under `tg_geometry: strict` the
prelude is span-local and a coda token reads its own span plus EARLIER prefix cells
(`tg_strict_allow`), with the conv, the value shift and the retention carry reset per
segment. Substituting candidate tokens into span `s+1` therefore changes the model's
output only at span `s+1`'s own positions. ONE forward over a row decodes a candidate
token for EVERY span of that row at once, and the spans cannot see each other. So the
number of graded slots `S` is FREE; the cost levers are `K` (candidates),
`code_grade_tokens` (decode steps) and `code_grade_rows` (rows graded per step).

### Cost, arithmetic and not a guess

One unit `u` = one transformer block applied at one sequence position. At `d_model` 1024,
prelude 4 / core 6 / coda 4, `L` 1024, `B` 6, 64 slots, mean depth 6:

- A training step of the code-only arm (no coda at train): prelude `4·6·1024` = 24,576 u
  plus core `6·6·6·64` = 13,824 u = 38,400 u forward, about 115,200 u with the backward.
- One sampling forward over ONE row (eval, every cell given, eval depth 8): prelude 4,096
  + core `6·8·64` = 3,072 + coda 4,096 = 11,264 u.
- A graded step costs `(J + 2) · rows · K + 2 · rows` such passes: J decode steps and 2
  parity grading passes per candidate copy, plus 2 passes that grade the TRUE span for the
  `code_grade_true_rank` instrument. At `J` 16, `rows` 1, `K` 4 that is 74 row-passes =
  833,536 u = **7.2 training steps**. Amortised at `code_grade_every: 8` that is 0.90 extra
  steps per step, a rate of **0.53×** the code-only arm.

That 0.5× ceiling is what fixes the defaults, and it buys roughly four graded slots per
step-equivalent out of ~300 valid slots in the batch. The ordinary `code_target` L2 term
stays available at its own weight to carry the rest (`tul_code_grade.yaml` has it OFF at
weight 0; `tul_code_grade_l2.yaml` has it ON at 1.0).

### What was ruled out on cost

Generating with the real generator (`morph/inference/tul_generate.py`) is a
recompute-per-token pass over the whole row, so `S·K·J` decode steps of prelude + coda at
`S` 8, `K` 4, `J` 16 is about 4.2 M u — **36× a training step**. There is no version of
that which fits. The parallel-span decode is not an optimisation of it; it is the only
form in which the idea is affordable at all, and it exists only because the geometry is
strict.

## Alternatives considered

- **Grader: a second frozen plain checkpoint.** The most honest judge, and the one whose
  plausibility score means what the words say. Rejected for VRAM: the arms sit near 26 GB
  on a 31.4 GB card with ~1 GB of slack (root `CLAUDE.md`), and a second full model plus
  its activations does not fit. Not shipped as a refused knob either, because a knob that
  raises is a stub with a nicer name. If the Spark ever runs the panel, this is the first
  thing to add.
- **Grader: the proposing coda with its own cell.** Rejected on the argument above: the
  argmax is the cell's own greedy decode, so the target is a fixed point and the term is
  vacuous. This is the failure mode that killed `disc`, in a new costume.
- **Best-only (`best`) vs preference (`pref`).** Both shipped. `best` is the minimal
  change from `code_target` and keeps the cosine instruments comparable. `pref` is the one
  that uses the losers, and L2-to-a-mean is exactly what arm A's hedging says is the wrong
  shape. Default `pref`.
- **Conditioning the candidates on nothing (sample from the coda with a zero cell).** That
  would make the target independent of the loop's state — the property the whole arm
  exists for. Rejected.
- **Re-running the boundary rule on the candidate.** Would change the span's length and
  therefore every slot position after it, forcing a repack per candidate per decode step
  and destroying the parallel-span property. The candidate instead fills the TRUE span's
  token window. The span's LENGTH leaks; it is constant across the `K` candidates of a
  slot, so it cannot bias the ranking.
- **Decoding `J` tokens of a longer span.** `E` pools the whole span, so a partly
  substituted span would mix candidate tokens with true tokens and the target would carry
  the truth. Rejected: only slots whose true span is at most `code_grade_tokens` long are
  graded, and the covered fraction is logged (`code_grade_slot_frac`).
- **Grading every step on a slot subsample.** Same total compute per graded slot as
  grading fewer rows more often, because the pass is over a whole row either way. Kept
  `code_grade_rows` + `code_grade_every` as the two levers and grade every eligible slot
  of the chosen rows, which is strictly more signal for the same compute.

## Acceptance criteria

1. `tul.code_grade: false` is bit-identical to the code-only arm (same loss, same logits,
   same parameter set).
2. The candidates are real token sequences drawn from the frozen coda, at most
   `code_grade_tokens` long, occupying the true span's token window.
3. Changing the TRUE tokens of span `s+1` leaves every grade unchanged — the grader cannot
   see what it is grading against.
4. `E` and the coda receive exactly zero gradient from the term; the term reaches the loop.
5. The reported loss reconciles: `loss = code_grade_weighted + code_target_weighted +
   gain_reg_weighted`.
6. Training mode and the RNG are restored after a graded step.
7. Measured rate at least 0.5× the code-only arm at the shipped defaults.

## Risks

- **The signal is thin.** Four graded slots per step-equivalent. If the term needs a dense
  gradient this arm cannot give it one, and the honest answer is that the compute does not
  exist on this hardware rather than that the idea failed.
- **The eligibility filter biases toward short spans.** Spans longer than
  `code_grade_tokens` are never graded. `code_grade_slot_frac` is the reading; if it comes
  back below ~0.3 the defaults are wrong.
- **The first candidate token is cell-blind.** It is drawn from the boundary TOKEN
  position, the only trained emit head on this arm, and under the strict geometry that
  position cannot read slot `s`'s cell. `J - 1` of the `J` tokens are cell-conditioned.
- **`E(best)` may be no closer to `E(true)` than `E(worst)` is.** That is the grader's
  sanity question and it is instrumented (`code_grade_cos_best_true` against
  `code_grade_cos_worst_true`). If they are equal the grader is not selecting for anything
  the code measures, and the arm is refuted at its own instrument rather than at its CE.
- **The grader cannot be sanity-checked by a difference.** The candidates are drawn from
  the model that grades them, so `E_p[log p] = -H` sits at or above the truth's `-CE` by
  Jensen: `code_grade_true` is BELOW `code_grade_mean` however good the grader is, and the
  CPU fixture reads exactly that at init (best -2.84, mean -3.19, worst -3.55, true -5.16).
  The sanity reading is `code_grade_true_rank`, the fraction of non-degenerate candidates
  the real continuation outscores: 0.5 for a calibrated grader, near 0 when it ranks real
  text last. I wrote the difference first and the fixture caught it.
- **Degenerate candidates.** A repetition loop is fluent and scores well
  (`genppl-needs-a-diversity-guard`). Guarded by a distinct-2 floor on the candidate's own
  tokens, and the flagged fraction is logged.
