# Planned: math under the shipped ternary rule — E16 and E17 were run on a starved core

Status: failure

Date: 2026-09-12 03:32 CDT (frozen before launch; the runner
`/home/wolfe/morph-scratch/arc/run_mathpanel.sh` is written and was NOT started).
Arc: [`2026-09-04-loop-contribution-arc.md`](../planned/2026-09-04-loop-contribution-arc.md).

Evidence it stands on:

* **The two math panels were run under `absmean`.** ARC E16 (Olympiad,
  [record](../failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md)) read `oly-mask`
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

All four arms ran on the 5090 (`arc/run_mathpanel.sh`, 2026-09-13, 03:00 to 15:09), 6,000
steps each, every `MATH DONE` line `verdict=HEALTHY`. Sweeps at 1500/3000/4500/6000 with
`olympiad_sweep.py` (Olympiad: `holdout_clean`, 1,906 docs; Sudoku: `eval_holdout`, 3,000
boards), `worth_profile.py` at 6,000 on the slot arms. Files:
[`results/2026-09-12-math-norm-match/`](../results/2026-09-12-math-norm-match/) (sweep and
worth JSONs, trimmed run logs as `.txt`). The paired gaps below are doc-paired at depth 6
(sum of per-doc CE over sum of per-doc tokens, 2,000 bootstrap draws over docs, computed
from the sweeps' `doc_ce_sum` / `doc_n_tokens`).

Token CE at 6,000 on the holdout, by forced depth:

| arm | d=1 | d=2 | d=3 | d=6 | d=9 | d=16 | K1−K6 [95 % CI] | K3−K6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `oly-notul-nm` (plain) | 0.8652 | 0.8026 | 0.8028 | **0.8005** | 0.7985 | 0.8503 | **+0.0647** [+0.0602, +0.0691] | +0.0024 |
| `oly-spandec-strict` | 1.0276 | 1.0210 | 1.0190 | **1.0166** | 1.0161 | 1.0256 | **+0.0111** [+0.0085, +0.0136] | +0.0024 [+0.0014, +0.0034] |
| `sud-notul-nm` (plain) | 0.5095 | 0.5086 | 0.5084 | **0.5085** | 0.5087 | 0.5098 | +0.0011 [+0.0009, +0.0012] | −0.0001 |
| `sud-spandec-strict` | 0.7636 | 0.7597 | 0.7590 | **0.7586** | 0.7589 | 0.7599 | **+0.0049** [+0.0047, +0.0052] | +0.0003 [+0.0002, +0.0004] |

Gaps and the 2026-09-08 twins (absmean rule) on the same holdouts:

| reading | value |
| --- | --- |
| `sud-spandec-strict` − `sud-notul-nm`, paired @6 | **+0.2502** [+0.2469, +0.2534] |
| E17 `sud-mask` − `sud-notul` @6 (unpaired means, same files) | +0.2136 (0.7122 − 0.4986); the prereg's bar was +0.155 |
| `oly-spandec-strict` − `oly-notul-nm`, paired @6 | **+0.2161** [+0.2015, +0.2313] |
| E16 `oly-mask` − `oly-notul` @6 on `holdout_clean` | +0.1728 (0.9065 − 0.7337); the prereg's bar was +0.18 |
| `sud-notul-nm` vs E17 `sud-notul` @6 | 0.5085 vs 0.4986: **+0.0099** (norm_match slightly worse) |
| `oly-notul-nm` vs E16 `oly-notul` @6 | 0.8005 vs 0.7337: **+0.0668** (norm_match worse, past the 0.05 bar) |
| `oly-notul-nm` K1−K6 vs E16 `oly-notul` | +0.0647 vs +0.0238: the plain loop earns 2.7x MORE depth under norm_match |
| Sudoku solve rate @6 | plain 0.8 %, slot 0.0 % (E17 headline; not predicted) |
| Olympiad answer accuracy @6 | plain 87.5 %, slot 84.6 % |

Worth profile at 6,000 (mean CE change when the cells are replaced; negative = the model
gets BETTER without them):

| arm | zero | shuffle | wrong_seed | all_slots | offsets 0 / 1 / 2 / 3 / 4+ under zero |
| --- | --- | --- | --- | --- | --- |
| `oly-spandec-strict` | **−0.1393** | −0.0067 | −0.0970 | −0.1393 | −2.23 / −0.60 / +0.07 / +0.01 / +0.09 |
| `sud-spandec-strict` | **−0.1903** | −0.0004 | +0.1670 | −0.1903 | −1.87 / −0.84 / −0.06 / +0.21 / +0.18 |

On both corpora the coda is better off with the cells ZEROED, and the damage sits at the
first two tokens after the slot (−1.9 to −2.2 nats at offset 0) while later offsets are
helped (+0.18 to +0.21 at offsets 3 to 4 on Sudoku). `wrong_seed` on Sudoku is +0.167:
the cells carry board-specific content the later offsets use, and the same cells mislead
the first token of the next row. `shuffle` is near zero on both, so the coda reads its
own slot's cell, not a bag.

## Verdict

- **P-a FALSE.** The Sudoku gap is +0.2502, larger than +0.155 (and larger than the
  +0.2136 recomputed from E17's own files).
- **P-b TRUE.** The gap is positive.
- **P-c FALSE.** The Olympiad gap is +0.2161, larger than +0.18 (recomputed +0.1728).
- **P-d FALSE.** Sudoku slot K1−K6 is +0.0049, inside the 50 % residual band
  [0.000, 0.010], not above 0.01.
- **P-e TRUE.** K3−K6 is +0.0024 (Olympiad) and +0.0003 (Sudoku), both below 0.005.
- **P-f FALSE on one of two.** Sudoku plain moved +0.010 (inside 0.05); Olympiad plain moved
  +0.067 (outside). The Olympiad E16 numbers are not comparable to the current recipe.
- **P-g TRUE.** Four of four healthy to 6,000, no tripwire.
- **P-h TRUE (the check).** `all_slots` equals `zero` on both slot arms to four decimals;
  the strict geometry fires on the math shapes.

Status: failure (P-a, P-c, P-d, P-f failed). Binding clause: the gaps did not narrow, they
grew by about 0.04 on both corpora under the shipped recipe, so E16/E17's verdict stands
for the shipped recipe too and the math lane closes for the slot loop as currently defined.
The one reading in the goal's own currency: `oly-spandec-strict` is the first slot arm on
any corpus to read token K1−K6 above 0.005 with a CI clear of it (+0.0111), a sixth of the
plain loop beside it (+0.0647), and its K3−K6 (+0.0024) equals the plain model's; on
Sudoku the plain loop is flat and the slot loop reads +0.0049, all in the first two passes.

## Updated hypothesis

Two things moved and one did not. The ternary rule moved the PLAIN loop's depth use on
Olympiad (K1−K6 0.024 → 0.065) while costing it 0.067 nats at 6,000, which is the same
"price paid early, loop share up" trade the web panel measured at 5k versus 20k. The slot
arms did not close their gap; they widened it, and the worth profile says why the CE is
where it is: the cells hurt the first two tokens of every next span by about 2 nats and
help the rest by 0.1 to 0.2. That offset-0 damage is the same on web text (the strict
ruler's zero worth at offset 0), on Olympiad and on Sudoku, so it is a property of the
strict cell write plus the span-decoder target, not of a corpus. It is not explained here;
the candidate mechanisms are the span decoder training z toward the WHOLE next span while
the coda needs the first token most, and token-state dropout at the cells. Test: the
register panel's worth profiles (already queued) will read the same offsets; if the
offset-0 damage is still ~2 nats with M cells, the write is the fault, not the capacity.
Do not queue another math training arm for the slot loop until the offset-0 damage is
understood on web text, where the runs are cheaper.

(pending)
