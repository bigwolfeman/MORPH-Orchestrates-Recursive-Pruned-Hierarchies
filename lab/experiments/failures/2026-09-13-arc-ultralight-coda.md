# A one-block coda: the cells are not a substitute for coda capacity

Status: failure

Date: 2026-09-13 (frozen before any GPU step; the 5090 is running the math panel). Arc:
[`2026-09-04-loop-contribution-arc.md`](../planned/2026-09-04-loop-contribution-arc.md). One-factor
partner: `slot-spandec-strict`
([`2026-09-12-arc-strict-geometry.md`](2026-09-12-arc-strict-geometry.md)).

## The fact

The coda has headroom it is not spending on `z`. Perturbing the slot state by 10 % moves
the loss by **0.0005** nats; a gradient-fitted `z` is worth **0.9 to 2.6** nats to the
same coda while the loop's own `z` is worth **+0.002** over its entry
([`slot-z-optimize-reader-has-capacity`], 2026-09-10 — and read it with
[`fitted-z-used-the-answer`], 2026-09-12: that headroom is HINDSIGHT, the fitted `z` saw
the scored tokens). Four coda blocks over 1152 positions can reconstruct most of what the
cells were supposed to carry from the tokens themselves, so nothing forces the coda to
read the cells hard.

## Question

If the coda is cut to ONE block while the block budget is conserved, does the model lean
on the slot cells — and does the loop's contribution move?

## Hypothesis

The cells are a channel the coda can afford to ignore. Take away the coda's capacity and
the cheapest remaining route to a next-span token is the cell. If that is right, the
`all_slots` worth rises above the partner's **0.1865** and the token K-curve moves off
**+0.0016**. If the CE gets much worse and NOTHING else moves, the coda's capacity was
load-bearing for the token path and the cells were never the bottleneck.

This is a cheap control, not a proposal: nobody wants to ship a one-block coda. It exists
to tell the register panel whether "the coda ignores z" is a real mechanism.

## Method

ONE arm, one factor against `slot-spandec-strict`: `n_prelude: 7`, `n_core: 6`,
`n_coda: 1`. **The block count is conserved at 14**, so the model has the same depth
budget and the same parameter count class; only the split moves. 5,000 steps, seq 1024,
batch 6, seed 1, ramp 1,000, `norm_match`, `core_fixed_point_lambda` 1.0,
prune/carve/route off, retention off, `tul.tg_geometry: strict`,
`tg_coda_prefix_reach: all`, `loop_reach: 0`, `mux_beta: 0`, `spandec` on at J 32.

| arm | config | one factor |
| --- | --- | --- |
| `slot-ultralight-coda` | `tul_slot_ultralight_coda.yaml` | `n_prelude: 7`, `n_coda: 1` |

Gate: a build-and-backward test in `tests/test_tul_slot_register.py`
(`test_an_ultralight_coda_builds_and_runs_a_forward_and_backward`) plus the config
composition check, because the one thing that could stop this arm is structural — the TG
kwargs, the per-layer x0/bigram injections and `slot_cell_inject_keep` all index the
coda's blocks by position.

Readout: the same six instruments as the register panel, with `worth_profile`'s
`all_slots` and the token K-curve carrying the weight.

## Predictions (frozen)

- **U-1 (the cells get used more).** `slot-ultralight-coda`'s `all_slots` worth is **above
  0.25** nats, against the partner's 0.1865. **55 %.** Reasoning: the worth metric is how
  much the model loses when the whole slot channel is cut, and a weaker coda has fewer
  alternatives. Against it: a weaker coda may also be worse at READING the cells, which
  lowers the worth for the opposite reason — the metric cannot tell those apart, which is
  why U-2 exists.
- **U-2 (the loop moves).** Token **K1−K6 above 0.005**, against the partner's +0.0016.
  **25 %.** Reasoning: the coda's capacity is not obviously what makes the PASSES
  redundant — the passes were flat on a Parcae core, under four entries, and at four
  depths. Leaning on the cells is not the same as leaning on the loop.
- **U-3 (CE cost).** Depth-6 CE, token-paired on 480 rows, is **worse** than
  `slot-spandec-strict` by **0.05 to 0.30** nats. **60 %.** Reasoning: the coda is where
  the token prediction happens; one block over 1152 positions is a real cut, and the three
  blocks moved to the prelude cannot substitute because the prelude never sees the cells.
  Residual: 25 % worse by more than 0.30, 15 % within 0.05.
- **U-4 (rate).** It clears the partner's 11,759 tok/s at step 200. **70 %.** The block
  count is conserved and the prelude runs under a narrower relation than the coda does.

## Binding

If **U-1 holds and U-2 fails** — the model leans on the cells and the passes still read
zero — that separates the two mechanisms cleanly: the cells are a usable channel whose use
is set by the coda's alternatives, and the loop's flatness is independent of both. The
register panel should then be read as a CHANNEL experiment, not a depth experiment.

If **U-1 and U-2 both hold** — leaning on the cells makes the passes matter — the coda's
capacity was masking the loop all along, and the honest next arm is the register at a
reduced coda.

If **U-1 fails** — the worth does not rise — the cells are not a substitute for coda
capacity at all, and "the coda ignores z because it doesn't have to read it" is dead as an
explanation.

## Not verified before launch

* **No GPU step.** The config composes and a CPU model with `n_coda: 1` builds and
  backprops; nothing beyond that.
* **The 7/6/1 split is one point, not a sweep.** 6/6/2 and 5/6/3 were not built.
* **Conserving the block count is a choice.** It keeps depth and parameters comparable and
  it makes the arm TWO changes (coda down, prelude up) rather than one. A 4/6/1 arm would
  be one change and a different model size. The confound is named, not removed.
* **The prelude cannot see the slot cells** under the strict geometry, so the three moved
  blocks cannot substitute for the coda's job. That is the mechanism behind U-3 and it
  means this arm is expected to be WORSE. It is a control.
* **`worth_profile`'s `all_slots` equals `zero` by construction on a strict arm**, so U-1
  is a reading of one quantity under two names.

## Results

The arm ran on the 5090 (`arc/run_recon.sh`, 2026-09-13 15:34 to 16:33), 5,000 steps,
`DONE slot-ultralight-coda exit=0 verdict=HEALTHY last=4999 Final val_loss=4.4365`.
Sweeps at 2,500 and 5,000 (`core_depth_sweep.py`, depths 1,2,3,6,9,12,16, 480 rows),
`worth_profile.py --rows 192 --modes auto` and the depth state probe at 5,000. Files:
[`results/2026-09-13-register/`](../results/2026-09-13-register/) — the sweep, worth and
state-probe JSONs, the trimmed run log as `run_slot-ultralight-coda.txt`, and the paired
gaps in `paired_gaps_5000.txt`.

Token CE at 5,000 by forced depth (480 rows):

| arm | d=1 | d=2 | d=3 | d=6 | d=9 | d=16 | K1−K6 [95 % CI] | K3−K6 [95 % CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `slot-ultralight-coda` | 4.3564 | 4.3552 | 4.3549 | **4.3547** | 4.3549 | 4.3558 | **+0.00169** [+0.00143, +0.00195] | +0.00017 [+0.00005, +0.00029] |
| `slot-spandec-strict` (partner) | 4.3490 | 4.3479 | 4.3476 | **4.3474** | 4.3476 | 4.3486 | +0.00161 [+0.00131, +0.00188] | +0.00017 [+0.00000, +0.00030] |

At 2,500 the arm reads K1−K6 +0.00073 [+0.00050, +0.00096] and K3−K6 −0.00009. The span
decoder's own K1−K6 is +0.00349 [+0.00295, +0.00409] against the partner's +0.00314.

Paired depth-6 CE on the sweep's 480 rows, `span_budget_profile.py --config
budget_web_full --depths 6` (gap = span − full, so positive means the ultralight arm is
worse):

| pairing | gap [95 % CI] |
| --- | --- |
| `slot-ultralight-coda` − `slot-spandec-strict` | **+0.0073** [+0.0043, +0.0099] |

Both arms pack the same 501,106 scored tokens (`prefix_k` 2, `L_total` 1152), so the
pairing is exact. By offset the cost is concentrated early: +0.0150 at offset 0, +0.0400
at offset 1, +0.0155 at offset 2, and +0.0033 over everything at offset 8+.

Worth profile at 5,000 (mean CE rise when the slot cells are replaced; higher = the model
needs them more):

| arm | zero | shuffle | wrong_seed | all_slots | zero @ offset 0 |
| --- | --- | --- | --- | --- | --- |
| `slot-ultralight-coda` | **0.1823** | 0.1498 | 0.0582 | **0.1823** | **+0.9493** |
| `slot-spandec-strict` (partner) | 0.1865 | 0.1739 | 0.0426 | 0.1865 | +0.7567 |

`all_slots` equals `zero` to four decimals on both, as the strict geometry requires.

Slot-state geometry from the run's own final `[VAL]` (the shipped post-`7a24adf` probe,
`val/slot_eff_rank` over a row's slot states, trainer val recipe, 19 periodic evals then
the final one — the same row set the ruler's log used):

| arm | `val/slot_eff_rank` | `val/slot_pairwise_cos` |
| --- | --- | --- |
| `slot-ultralight-coda` | **12.5820** | **0.4596** |
| `slot-spandec-strict`, corrected baseline | 13.85 | 0.520 |

Rate at step 200: **10,759 tok/s** against the partner's 11,759.

## Verdict

- **U-1 FALSE.** `all_slots` is **0.1823**, below the 0.25 bar and 0.0042 BELOW the
  partner's 0.1865. The whole slot channel is worth slightly LESS to a one-block coda than
  to a four-block one.
- **U-2 FALSE.** Token K1−K6 is **+0.00169** [+0.00143, +0.00195], a CI that does not
  reach 0.005 and overlaps the partner's +0.00161. The passes read the same as ever.
- **U-3 FALSE.** The paired depth-6 gap is **+0.0073** [+0.0043, +0.0099], inside the
  15 % residual band (worse by less than 0.05), not the predicted 0.05 to 0.30. Cutting
  three coda blocks and paying them to the prelude costs seven thousandths of a nat.
- **U-4 FALSE.** 10,759 tok/s against 11,759. Moving blocks from the coda to the prelude
  did not make the step cheaper; under the strict geometry the prelude runs same-span-only
  relations that the fused path cannot serve either.

Status: failure (four of four scorable predictions false). Binding clause: **U-1 fails,
so "the coda ignores z because it doesn't have to read it" is dead as an explanation.**
The cells are not a substitute for coda capacity; a coda with a quarter of the blocks
leans on them no harder.

The one number that did move is the SHAPE of the worth, not its size. At offset 0 the
one-block coda loses +0.9493 nats when the cells go, against the partner's +0.7567, while
its total worth is lower — so a weak coda depends on the cell more at the first token of a
span and less everywhere after it. `wrong_seed` tells the same story from the other side:
0.0582 against the partner's 0.0426, and +0.9743 at offset 0 against +0.618. The weak coda
is more easily misled by the wrong span's cell, which is what a reader with fewer
alternatives looks like.

For the record beside the register panel: the matched-compute plain control
(`plain-coda-matched`, 14 block-passes per token at depth 1) is 0.2536 nats ahead of
`slot-spandec-mask`, and the strict ruler ties that mask arm at −0.0001
([`2026-09-12-arc-strict-geometry.md`](2026-09-12-arc-strict-geometry.md)), so this arm
sits about **+0.261** nats behind matched-compute plain. That is chained arithmetic
through the ruler, not a fresh pairing.

## Updated hypothesis

The coda's capacity is not the gate on the slot channel. Removing three quarters of it
moved the token CE by 0.007 nats, left the worth of the whole channel flat and left the
K-curve on the same +0.0016 that twelve arms before it read. Two things follow.

First, the reading of the register panel is settled by elimination, not by hope: it is a
CHANNEL experiment. Whatever sets how hard the coda reads a cell, it is not how much
compute the coda has to reconstruct the cell's content from tokens — a one-block coda
cannot do that reconstruction and still does not read the cell harder.

Second, the offset-0/offset-rest split is the live signal. A weak coda needs the cell more
at exactly the position the cell was designed for (the first token after a boundary, worth
+0.95 here against +0.76) and less at every later offset, which is where the total is made.
The next question is not "does the coda have room to read z" but "why does a cell stop
paying after offset 1", and the instrument that separates those is the per-offset worth,
not the K-curve. A 6/6/2 or 5/6/3 split would say whether the offset-0 rise is monotone in
the missing capacity; nothing in this arm's design forbids it and it was not run.

One honest caveat on the CE reading: the arm is two changes, not one. The coda lost three
blocks and the prelude gained three, so "the coda's capacity is not the gate" is measured
at a conserved block count. A 4/6/1 arm — one change, a smaller model — was not built and
would separate the two.
