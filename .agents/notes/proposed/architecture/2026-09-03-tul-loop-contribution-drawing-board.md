# Agent Note: TUL from the top — how to make the loop contribute under "think once per span, decode cheaply"

Status: proposed

## Problem

TUL's purpose is fixed: the core loops ONCE per span on the slot, and tokens decode
cheaply (prelude → coda, never through the core). The 2026-09-03 "paid loop" ship
(`2026-09-03-ship-the-paid-loop-cut-the-arms.md`) inverted that purpose by putting every
token through the core; Wolfe rejected it the same evening ("totally wrong"). The slot-only
tree is intact at `d9e04e6`. What the inversion left behind is a set of grounding numbers
on the one question that drove it: **the loop does not contribute** on the cheap-decode
design. This note is the drawing board for that question, written before any code.

Grounding numbers (all seed-1 MORPH at d 1024, seq 1024, 20k steps unless noted):

| arm | K1 − K6 (loop earning) | K3 − K6 (earning on trained support) | note |
|---|---|---|---|
| plain per-token loop, flat LR (`notul-l2nc`) | 0.120 | 0.015 | `successes/2026-08-31-loop-killer-bisect.md` |
| plain per-token loop, 1000-step ramp (`notul-20k-wu`) | 0.04 | — | `failures/2026-09-02-warmup-20k-pair.md` |
| slot-only TUL, every arm (A0/A1/A3/A4, gate, TG, gist) | ≤ 0.011 | — | `docs/ablation-ledger.md` |
| per-token loop + slots ("paid", rejected) | 0.168 at 5k, 0.100 at 20k | — | `successes/2026-09-02-*` |
| FM planner with the TRUE span target planted | worth 0.0000 | — | mean-pooled targets are empty |
| gist mux (slot content forced load-bearing) | worth_shuffle 0.05–0.07 | — | +0.34 nats token CE |

Read together: even the plain per-token loop earns ~0.015 nats on the depths it was
trained at, and 0.04 on the shipped ramp schedule. The slot loop earning ≤ 0.011 is not a
TUL-specific failure; it is the same near-zero, seen through a narrower pipe. **There was
no depth-earning to transfer to a slot in this regime.** That is the first-principles
fact the redesign has to start from.

## Proposal

What "loop contribution" requires, stripped down:

1. **Depth must be worth something somewhere.** A loop that adds nothing to a plain model
   cannot add anything through a slot. Condition: the plain per-token loop must earn on
   trained support (K3 − K6, not K1 − K6, which mostly measures how out-of-distribution
   K1 is).
2. **The thought must hold something the cheap path cannot recompute.** The slot sees the
   same causal prefix the next span's tokens see; its only edge is compute (6·T layers
   against the tokens' 6). If the token path can reach the same loss on its own, the loss
   gradient never needs the slot and the slot stays empty (the "plan is empty" finding,
   0.07 nats of its span; `worth_shuffle` ≤ 0.005 on every prefix arm).
3. **The gradient must reach the loop through the tokens' loss.** Through attention K/V
   alone it is weak; every arm that added its own slot supervision (gist, mux, SIGReg)
   made the slot load-bearing and paid for it in token CE.

The approach, in order, each step gated by a number before the next:

- **Step 0 — find a regime where depth earns.** Before any TUL arm: a plain MORPH run
  where K3 − K6 ≥ 0.10 nats. Levers, cheapest first: (a) the flat schedule instead of the
  ramp (0.120 vs 0.04 on K1 − K6; the ramp was adopted for detonation safety and it also
  removed the loop's earning — the detonation guard and depth-earning need separating,
  e.g. ramp then a higher flat LR, or the abort rule instead of the ramp); (b) data where
  depth is known to pay (code / math mixes, longer training); (c) a narrower model at the
  same loop depth so the loop is a larger share of the compute. Without Step 0 every TUL
  number is noise around zero, as it was in August.
- **Step 1 — the ceiling test (an oracle, one run, no new architecture).** In the
  cheap-decode forward, replace the slot's looped state with the frozen boundary-position
  state of a DEEP model (the kept `tul-a2-20k-wu` checkpoint, or a plain deep model from
  Step 0), and let the shallow token path attend it. This measures the most any thought
  computed from the prefix with extra depth can give shallow tokens. If it is ~0, no slot
  loop can earn on this data with this token path, and Step 2 is mandatory. If it is
  large, the problem is learning the thought, and Step 3 is next.
- **Step 2 — budget asymmetry: make the decoder need the thought.** Cheap decode has to
  be cheap in compute, not only in name: tokens through 2 + 2 layers (or prelude only)
  while the slot gets 6·T. With the token path starved, reducing the loss requires
  routing through z, which is the gradient pressure the August arms never had (their
  token path was 6 of 12 layers and decoded fine alone). Pair with attention routing so
  cross-span context reaches tokens ONLY through slots (tokens attend their own span plus
  slots; the TG restriction of `docs/tul-tg-spec.md` was one form of this — re-read its
  result before repeating it).
- **Step 3 — teach the thought, without a slot-side objective on tokens.** The candidate
  the record has not tried: distil the deep model's boundary state (Step 1's oracle) into
  the slot loop's output as a detached regression target. It is a JEPA-style target that
  is NOT a mean-pooled span (empty) and NOT the model's own shallow state (dmorph hs,
  cosine 0.14, hurt CE). The gate is Step 1's number: distillation can at most recover
  the ceiling.
- **Step 4 — fix the loop's input.** The slot state collapsed to rank 1.7–4.8 because its
  input was one shared `E_slot` plus a bag mean; per-slot embeddings doubled the time to
  failure. Seed the slot from the boundary token's actual prelude state (the
  `boundary` seed) and keep the per-slot embedding; measure the slot-state rank at the
  onset, the instrument already exists (`morph/training/core_jacobian.py`).

Acceptance metric for every arm, fixed now: K3 − K6 at the slot on the same 480 rows,
`worth_zero` / `worth_shuffle` at the token side, and the token CE against the cheap-decode
control at matched wall clock. A TUL arm "contributes" when the slot's loop earns on
trained support AND the tokens' CE at matched wall clock beats the control. Nothing else
counts.

## Alternatives considered

- **Keep the per-token loop and call the slot a bonus.** Rejected by Wolfe: it is plain
  MORPH with extra positions and defeats the purpose of TUL (cheap decode).
- **More slot-side objectives first (gist, mux, SIGReg variants).** Already measured: they
  make the slot load-bearing at a token-CE price, and they do not answer whether depth
  earns at all. Step 0 and Step 1 come first.
- **Skip Step 0 and go straight to Step 2.** Tempting because Step 2 is the design change;
  rejected because without a regime where depth earns, Step 2's numbers cannot be
  distinguished from noise (the August lesson, 8 GPU-hours re-deriving a known RCA).

## Acceptance criteria

Step 0 produces a plain run with K3 − K6 ≥ 0.10 on 480 rows; Step 1 produces one ceiling
number with a bootstrap CI; each later step is a preregistered experiment under
`lab/experiments/planned/` that cites the step's gate number.

## Risks

- Step 0 may fail on OpenWebText at d 1024: depth may simply not pay at this scale on
  this data, and the honest result is "TUL needs a regime we do not train in yet".
- The flat schedule brings back the detonation risk (`lab/divergence/DIVERGENCE-README.md`);
  the abort rule (preclip/total > 1e4 at step ≥ 200, 0 false positives in 44 runs) is the
  guard to run under, not the ramp.
- Step 2's starved token path lowers the control's CE too; the comparison must be at
  matched wall clock against a control with the SAME cheap token path and no slot.
