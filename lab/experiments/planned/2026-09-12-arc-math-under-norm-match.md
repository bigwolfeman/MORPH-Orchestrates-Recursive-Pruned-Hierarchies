# Planned: math under the shipped ternary rule — E16 and E17 were run on a starved core

Status: planned

Date: 2026-09-12 03:32 CDT (frozen before launch; the runner
`/home/wolfe/morph-scratch/arc/run_mathpanel.sh` is written and was NOT started).
Arc: [`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).

Evidence it stands on:

* **The two math panels were run under `absmean`.** ARC E16 (Olympiad,
  [record](../successes/2026-09-08-arc-e16-olympiad-curriculum-panel.md)) read `oly-mask`
  **0.18 nats BEHIND** the plain model on clean math, paired, and 2.4 accuracy points behind
  on the answer tokens. ARC E17 (Sudoku-Extreme,
  [record](../successes/2026-09-08-arc-e17-sudoku-depth-grid.md)) read `sud-mask` **0.155**
  behind the plain model at its own trained depth, flat from 1,500 through 6,000. Both ran
  `tul_m12_mnext_mask`.
* **`absmean` starves the looped core.** Measured on web text
  ([the ternary-rule note](../../../.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md)):
  under the absmean base the loop contributed **0.033** past pass 1 and 0.0009 past pass 3;
  the same recipe under `norm_match` reads **0.185** [0.182, 0.188] and **0.0139**
  [0.0131, 0.0146], with branch out/in ratios 0.86–1.10 against 0.19–0.24. `norm_match` has
  been the shipped rule since 2026-09-09.
* **The slot's TARGET changed after those panels too.** The span decoder replaced the
  order-free M-next bag and moved the mask arm 0.072 nats on web text, with the slot
  channel's worth rising 0.115 → 0.182
  ([record](../successes/2026-09-11-arc-span-decoder.md)).
* **Standing rule (Wolfe, 2026-09-12):** a flat loop on a corpus is NOT a verdict about that
  corpus. E16 and E17 closed the math lane for `absmean` + M-next. They did not close it for
  the recipe that ships.
* **The Olympiad holdout is contaminated.** 41 % of the original eval holdout is verbatim in
  training and bands 6-13 are 23-47 % unique. Score on `holdout_clean/` (1,906 docs) ONLY.

## Question

Does the slot loop still lose on math once the core is not starved (`norm_match`), the slot
has a target that asks for a whole span (`tul.spandec`), and the loop is the only cross-span
channel (`tul.tg_geometry: strict`)?

## Hypothesis

The E16/E17 deficit is a property of the absmean core and the order-free target, not of math.
Under the shipped rule the deficit narrows. It is a separate and weaker claim that it goes
negative; nothing in the web-text record supports that, so the prediction below is about the
SIZE of the gap and about whether the loop's depth instruments move at all on math, not about
winning.

## Method

Four arms, 6,000 steps each, run by `/home/wolfe/morph-scratch/arc/run_mathpanel.sh`
(modelled on `run_e17.sh` + `run_e16_clean.sh`; `arc_lib.sh`, the sustained tripwire, the
`gpu_busy` guard, a 12-step smoke before every draw). THE TWO CONTROLS RUN FIRST.

| arm | config | data | notes |
| --- | --- | --- | --- |
| `oly-notul-nm` | `notul_oly` | `oly_data` | plain control. The CONFIG is unchanged from 2026-09-08 — it already inherits `norm_match` through `notul` → `base`. Only the wandb/run NAME gains `-nm`, so it does not collide with the old checkpoints. |
| `sud-notul-nm` | `notul_sud` | `sudoku_data` | the same, for Sudoku. |
| `oly-spandec-strict` | `tul_oly_spandec_strict` | `oly_data` | `tul_slot_spandec_strict` + the data mixin. |
| `sud-spandec-strict` | `tul_sud_spandec_strict` | `sudoku_data` | the same, for Sudoku. |

The data mixin is composed AFTER the arm (`defaults: [<arm>, <data>, _self_]`), so it owns
seq (512 / 182), micro_batch (12 / 32), the effective batch (24 / 128), the 6,000 steps, the
curriculum stages, `ckpt_grad_iters: 4`, `ckpt_every: 1500`, and — for Sudoku — `max_slots: 0`
(→ `seq_len // 8` = 22; 64 slots at seq 182 OOMed the E17 mask arm's `prefix_project`). That
is exactly what `tul_sud_mask.yaml` / `tul_oly_mask.yaml` get, from the same two files, so
the new arms sit on the E16/E17 recipe on every axis except the three under test. A test
asserts the compose order wins on those keys
(`tests/test_tul_strict_geometry.py::test_the_math_arms_compose_and_build`).

Readout: `lab/divergence/olympiad_sweep.py` at every stage-end checkpoint (1500 / 3000 / 4500
/ 6000) at depths 1,2,3,6,9,12,16, on `holdout_clean/` for Olympiad and on the held-out board
shard for Sudoku, into
`/home/wolfe/morph-scratch/arc/results/2026-09-12-math-norm-match`. The slot arms also get
`worth_profile.py --rows 192` at 6,000 (`--modes auto` adds `all_slots`), where `all_slots`
must EQUAL `zero` under strict — a check on the geometry, not a result.

The rate rule is OFF by default in this runner (`RATE_FLOOR=0`) and that is deliberate: 8,086
tok/s is a seq-1024 web number and means nothing at seq 182 or 512. Set it only against a
measured control on the same data.

## Predictions (frozen)

Probabilities are the builder's, written before any GPU step.

- **P-a (the Sudoku gap narrows).** `sud-spandec-strict` minus `sud-notul-nm`, paired at
  depth 6 on the held-out boards at 6,000, is SMALLER than E17's **+0.155**. **60 %.**
  Reasoning: two of the three changes (the rule, the target) moved web text in the right
  direction by 0.07–0.18; the third (strict) costs CE on web text by construction. Residual:
  30 % it is the same to within 0.02, 10 % it is worse.
- **P-b (but still positive).** That gap is still **above 0.00** — the loop arm is behind the
  plain model. **75 %.** Reasoning: no slot-loop arm on any corpus has beaten its plain
  control at matched steps, and strict removes a route that was worth ~0.10 on web text.
- **P-c (the Olympiad gap narrows).** `oly-spandec-strict` minus `oly-notul-nm` on
  `holdout_clean` at 6,000 is smaller than E16's **+0.18**. **55 %.** Lower than P-a because
  the Olympiad curriculum changes the data distribution across stages and the deficit there
  grew with band difficulty.
- **P-d (Sudoku is where depth should show if anywhere).** `sud-spandec-strict` token K1−K6 on
  the held-out boards above **0.01** at 6,000. **35 %.** Reasoning: a Sudoku board is the one
  corpus in this tree where a span (one grid row) genuinely constrains later spans, and E17
  read the mask arm flat from depth 2 up — but it read that under a starved core, so the
  reading is not transferable. Residual: 50 % in [0.000, 0.010].
- **P-e (K3−K6 stays at zero on both).** Both slot arms' K3−K6 below **0.005**. **80 %.** No
  arm on any corpus has moved K3−K6 past 0.002.
- **P-f (the plain controls reproduce).** `sud-notul-nm` and `oly-notul-nm` land within
  **0.05** nats of their 2026-09-08 twins at 6,000 on the same holdout. **70 %.** They differ
  only by the ternary rule, which moved web text 0.010 at 20k and cost 0.1+ at 5k — so a
  MOVE here is informative in itself and is the first thing to read.
- **P-g (survival).** All four arms reach 6,000 with no sustained tripwire. **85 %.**
- **P-h (`all_slots` == `zero` on both slot arms).** **A CHECK, not a prediction.** If they
  differ, the strict geometry is not firing on the math shapes and the panel is void.

## Binding

If P-a and P-c both hold and the gaps land under 0.05, math goes back on the list and the
next arm is a longer math run at the shipped recipe.

If the gaps are unchanged to within 0.02 — the recipe change bought nothing on math — then
E16/E17's verdict stands for the shipped recipe too, and the math lane closes for the slot
loop as currently defined. That is a real outcome and is worth the four runs.

If P-f fails — the PLAIN controls move by more than 0.05 under `norm_match` — then the
2026-09-08 panels are not comparable to anything current and the whole math record needs
re-basing before any arm is read.

## Not verified before launch

* **Nothing has run.** No smoke, no step, no sweep. The card was busy for the whole build
  window and the runner was deliberately not started.
* **`tul_slot_spandec_strict` itself has never trained**, on math or on web text, so the math
  arms inherit an unmeasured parent.
* **The span decoder's J = 32 is mostly mask on Sudoku** (a grid row plus its newline is ~10
  tokens). That is waste, not error, and its cost at micro_batch 32 and 22 slots has not been
  measured.
* **Memory.** The E17 mask arm OOMed at 64 slots on seq 182; `max_slots: 0` fixes that, but
  the span decoder adds a `[B, S, J, V]` readout that the E17 arms never carried, at
  micro_batch 32. The smoke's peak GB is the first thing to read.
* **`worth_profile.py` on math shapes** has not been run; it was built for the web panel.
* **Sudoku solve-rate by rating bucket**, E17's headline readout, is produced by the same
  sweep but is not predicted above.

## Results

(pending)

## Verdict

(pending)
