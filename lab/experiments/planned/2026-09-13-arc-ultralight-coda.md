# Planned: a one-block coda, to find out whether the coda is the thing ignoring z

Status: planned

Date: 2026-09-13 (frozen before any GPU step; the 5090 is running the math panel). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). One-factor
partner: `slot-spandec-strict`
([`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md)).

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

(to be filled after the run; predictions above are frozen)
