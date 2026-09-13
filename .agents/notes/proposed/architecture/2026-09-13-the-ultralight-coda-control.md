# Agent Note: a one-block coda, as a control on "the coda ignores z"

Status: proposed

## Problem

"The coda has enough capacity to ignore the slot cells" is a standing explanation for the
flat loop and it has never been tested. The readings behind it: perturbing the slot state
by 10 % moves the loss **0.0005** nats; a gradient-fitted `z` is worth **0.9 to 2.6** nats
to the same coda while the loop's own `z` is worth **+0.002** over its entry (2026-09-10)
— and that headroom is HINDSIGHT, because the fitted `z` saw the scored tokens
(`fitted-z-used-the-answer`, 2026-09-12). Four coda blocks over 1152 positions can
reconstruct much of what the cells were meant to carry from the tokens themselves, so
nothing FORCES the coda to read the cells hard.

## Proposal

`morph/configs/tul_slot_ultralight_coda.yaml`: one factor against `slot-spandec-strict` —
`n_prelude: 7`, `n_core: 6`, `n_coda: 1`. The block count is conserved at 14, so depth
budget and parameter class are comparable and only the split moves.

This is a CONTROL, not a proposal to ship. Nobody wants a one-block coda. It exists to
tell the Thought Register panel whether "the coda ignores z" is a real mechanism, and it
costs one 5,000-step run.

The reading that matters is the pair: if `all_slots` worth rises above the ruler's
**0.1865** AND token K1−K6 moves off **+0.0016**, the coda's capacity was masking the
loop. If the worth rises and the K-curve does not, the cells are a channel whose use is
set by the coda's alternatives, and the loop's flatness is independent of both — which
would mean the register panel should be read as a CHANNEL experiment, not a depth one.

## Alternatives considered

* **Dropout or noise on the coda's token input.** `tul.token_state_dropout` already does a
  version of this at 0.15 and every arm carries it. Raising it was rejected: it degrades
  the token path without removing the coda's CAPACITY, which is the thing being tested.
* **Cutting the coda without conserving the block count** (4/6/1). One change instead of
  two, and it is the cleaner factor. Rejected because it also shrinks the model, so a
  worse CE would have two explanations. Conserving the count makes the arm two changes
  (coda down, prelude up) and that confound is named rather than removed.
* **Masking the coda's attention to the tokens of its own span.** That is `tg_restrict` —
  the mask arm — which has been run since E4 and is worse everywhere
  (`tul-bottleneck-is-the-next-test`). Not new, not this.
* **Freezing the coda.** Rejected: a frozen coda cannot LEARN to read the cells, so the
  arm would test the wrong thing.

## Acceptance criteria

1. The config composes and a model with `n_coda: 1` builds and runs forward and backward —
   the TG kwargs, per-layer x0/bigram injections and `slot_cell_inject_keep` all index the
   coda's blocks by position. **Met**, guarded by a build-and-backward test.
2. `all_slots` worth above 0.25 against the ruler's 0.1865. **Not met — unrun.**
3. Token K1−K6 above 0.005. **Not met — unrun.**

Prereg: `lab/experiments/planned/2026-09-13-arc-ultralight-coda.md`.

## Risks

* **The arm is expected to be WORSE on CE**, by 0.05 to 0.30 nats. The coda is where the
  token prediction happens, and the three blocks moved to the prelude cannot substitute
  because the prelude never sees the slot cells under the strict geometry. That is the
  mechanism, not a surprise, and it is why this is a control.
* **`worth_profile`'s `all_slots` EQUALS `zero` by construction on a strict arm**, so the
  headline reading is one quantity under two names.
* **The worth metric cannot separate "the coda leans on the cells more" from "the coda
  reads the cells worse".** A weaker coda could lower the worth for the opposite reason.
  That is why the K-curve is read beside it and why neither is read alone.
* **7/6/1 is one point, not a sweep.** 6/6/2 and 5/6/3 were not built.
* **Nothing has run on a GPU.** The config composes; a tiny CPU model builds and
  backprops. That is all.
