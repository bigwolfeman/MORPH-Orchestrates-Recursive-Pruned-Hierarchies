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

---

## Revision 2026-09-03 (evening) — branch `tul/think-once` from `d9e04e6`, the MUX reading, and the round-1 panel

Written after re-reading MUX (arXiv 2607.18264, full text) and every slot-loop filing.
No code and no run yet. Three corrections to the sections above, then the panel.

### Correction 1 — the recipe confound is NOT what killed the slot loop

The loop-killer bisect (`successes/2026-08-31-loop-killer-bisect.md`) found the token
loop's earning was killed by GLA × spectral cap, and the leak audit found full-BPTT loop
claims were `retention_carry` exploitation. Both were found AFTER the slot campaign, so
every slot-loop number from August carries at least one of GLA on, the cap, the leak, or
warmup 0. But `tul-20k` (`successes/2026-08-31-tul-vs-notul-20k.md`) already re-ran the
gist design (mask + MUX own + boundary seed + full-BPTT slot loop) under the fixed recipe
(retention off, cap 0, carry none) on the flat schedule: the slot loop earned K1−K6 =
0.015 at 20k while the token loop earned 0.207 on the same recipe, and the arm finished
0.357 nats behind the plain looped model. The fixed recipe did not unlock the slot loop.

### Correction 2 — what the ramp does and does not do (Wolfe's hypothesis, stated fairly)

Wolfe (2026-09-03): "It is totally possible the LR warmup resolves this and the original
TUL design was fine." Two different failures are in play and the ramp has only been
measured on one:

| failure | arms | mechanism (measured) | ramp status |
|---|---|---|---|
| paid-axis detonation | every arm on the flat winner recipe | early transient, steps 200–775, ternary trigger, core map organizes expansive | CURED, 0/9 draws (`DIVERGENCE-README.md` §A) |
| slot-loop takeover | A1 family (bag mean, emit/plast 0.5/0.5, no mask) | slow turnaround at 1000–3000; fuel is the emit/plast gradient into the shared core; removing the aux (TG2 0/2, v1a2b 0/4, tul-20k 20k clean) removes it | NEVER TESTED — every slot-loop arm ran under `warmup: 0` (base.yaml flipped to 1000 at d9e04e6, the last slot commit) |

So the ramp on the original A1 is an honest open cell, and cheap. My priors, to be frozen
in the prereg: A1 survives 2/2 under the ramp (60%); its slot loop still earns ≤ 0.02
nats on trained support (80%); its plan is still worth ≤ 0.04 nats (75%); it beats the
cheap-decode floor A3 by less than its slot compute is worth (75%). Reason: the ramp keeps
the core map near identity (`successes/2026-09-03-warmup-core-map.md`), which is why it is
stable, and it cut the plain loop's earning 0.207 → 0.04 for the same reason. Stability
and emptiness are different problems; the ramp addresses the first.

### Correction 3 — Step 0 is not "find a regime"; it is "give the loop a job" (MUX)

Step 0 above asked for a plain-model regime where K3−K6 ≥ 0.10. Re-reading MUX changes
that. The paper's payload is the LOCAL loss alone (γ=0 still beats every latent baseline,
Table 6; Fig. 3 rises 32.7 → 48.2% as local targets cover 0 → 6 tokens), and the loop we
already have is the paper's own parallel variant: MUX* generates K=24 latent tokens with
T=3 Jacobi iterations (App. E.1) — a per-sample loop over the latent block — and it beats
the sequential model (Table 1, 58.0 vs 56.7 ID). What MUX gives the slot loop is a job
with headroom that the loop can earn on directly, measured WITHOUT any reader:
`mux_local` at the slot as a function of forced depth. That is the instrument the August
campaign never had — every loop-worth number went through the coda.

Two MUX targets exist in the tree and they are different designs:

- `mux_target: own` — the slot is a lossless record of the span it terminates (memory).
  Deterministic target; a one-pass encoder reached `mux_rel` 0.51 with `n_core: 0`
  (GL1b). I expect the loop to earn ~0 on it: nothing to think about.
- `mux_target: next` — the slot is the multiplexed FORECAST of the span about to be
  decoded. This is "think once per span": the thought holds the next span in
  superposition (the paper's §4.3 parallel-hypotheses property), the cheap coda
  demultiplexes it token by token. The target has headroom (v1a2b, detached head, β 0.1,
  bag-mean seed, warmup 0: `mux_local` 6.83 vs unigram 7.32 at 3500 steps) and it is the
  one objective that ever moved loop worth (v1a2b: 5–8x control, 4/4 seeds, 0/4
  takeovers; `failures/2026-08-27-warmup-sigreg-ntpdrop.md`). Never run through the tied
  head at β=1 (the paper's recipe), never with the boundary seed, never under the winner
  recipe or the ramp, and its depth-earning was never measured.

MUX settings to keep from the paper (Table 9): β=1.0, τ=1.0, geometric ρ=0.9, loss
through the model's own tied unembedding (no detach), random chunking (our boundary
jitter `gate_truncate_p` is the same augmentation). Caveat the paper states: losslessness
of ρ=0.9 is proven for S ≤ 11 in fp32; our spans reach 32, so late tokens carry
0.9^31 ≈ 0.04 of the first token's weight. ρ is a knob for round 2, not round 1.

### Round-1 panel (proposed; predictions to be frozen in `lab/experiments/planned/` before launch)

Common recipe for every arm: the winner recipe (`model.retention=false`,
`training.spectral_project_cap=0`, `retention_carry: none`) + the ramp
(`training.warmup=1000`, flat 1e-4 after), ternary on, `ademamix_alpha_cap 3.5`,
`ademamix_t_beta3 3500` pinned, seq 1024, batch 6, seed 1, **5000 steps** (so the kept
`notul-20k-wu/step_5000.pt` and `tul-a2-20k-wu/step_5000.pt` are rulers on the same
recipe), `eval_every 250`, `grad_probe_every 1` + tripwire, `tul.eval_ablations=true`,
checkpoints at 2500 and 5000. One trainer on the 5090, sequential.

| # | arm | base config at d9e04e6 | isolates | est. |
|---|---|---|---|---|
| R0 | **A3-wu** — coreless, no slots | `tul_a3` + recipe | the cheap-decode FLOOR (8 layers/token). Every TUL arm must beat this at matched token compute | 12 min |
| R1 | **A1-wu** ×2 seeds — the original TUL | `tul_a1` + recipe | Wolfe's hypothesis: takeover rule, slot K-sweep, plan worth, CE vs R0 | 2 × 25 min |
| R3 | **M-own** — A1 + boundary seed + emit 0/plast 1 + MUX own, β 1, through the head, NO mask, full BPTT | `tul_g0c0` + `tul.tg_restrict=false` | does the loop earn on a deterministic memory target? (`mux_local(own)` vs depth) | 30 min |
| R4 | **M-next** — as R3 with `mux_target: next` | `tul_g0c0` + `tg_restrict=false` + `mux_target=next` | THE ARM: does the loop earn on a forecast? (`mux_local(next)` vs depth); does the coda read it (worth_zero/shuffle, attn_lift); CE vs R0/R3 | 30 min |
| R5 | **M-next-mask** — R4 + `tg_restrict` | `tul_g0c0` + `mux_target=next` | the plan as the only route: price vs R4, worth, depth | 35 min (eager) |
| R6 (optional) | **M-own-mask** = the gist loop under the ramp | `tul_g0c0` as is | the direct "does the ramp rescue tul-20k's design" datum; own vs next under the mask | 35 min |

Rulers on the same recipe at step 5000 (48 rows; re-sweep at 480): notul-wu K1 3.9879 /
K6 3.9488; A2 (paid loop) K1 4.1380 / K6 4.0812.

Readouts per arm, all on the same 480 fixed rows, paired by row:
1. `core_depth_sweep.py` depths 1..8 → token CE and span-first CE vs slot depth (K1−K6,
   K3−K6) — PLUS the new column `mux_local` per depth (the slot's own loss vs depth), with
   a paired bootstrap CI over rows. This column is the decisive number for "can the loop
   think"; it needs a ~40-line change to the sweep (report the mux loss the forward
   already computes).
2. `slot_path_worth.py` (plan-off / loop-off / shuffle), `worth_profile.py` (offset-in-span
   profile: first-token spike vs span-wide carrier), `val/attn_lift`.
3. `score_arms.py::fires` (takeover rule) and the tripwire (detonation).
4. `jac_ladder.py` on the slot path at 2500/5000 (map gain; is the slot map near identity
   under the ramp as the token map is).
5. Token CE vs R0 at matched steps AND at matched wall clock (queue-log epochs), and vs
   notul-wu step_5000.

Decision rule (proposed, to freeze): the slot loop "thinks" iff an arm's `mux_local` K1−K6
on 480 rows has a bootstrap CI above 0 AND its K3−K6 > 0.01. The thought "pays" iff that
arm's token CE beats R0 at matched wall clock. Round 2 is conditional: a thinking arm goes
to 20k against `notul-20k-wu` and a 20k R0; no thinking arm ⇒ the loop cannot earn on
slots at this depth/scale with any target we have, and the program moves to loop
composition itself (MORPH-general: the token loop saturates at K3–K4 too) or to a slot
stack that is deep without weight sharing.

### What the final design should do differently (speculation, to be tested by the panel)

1. **The thought is a forecast, not a memory.** `mux_target: next` on the looped slot
   state, trained through the tied head (the paper's Eq. 3–4). A memory is a one-pass job;
   a forecast has headroom and is what "decode cheaply" needs. Prop. 9 makes the slot
   states non-collapsing by construction — the rank collapse the takeover campaign
   measured (1.7–4.8 of 1024) is the failure this target is proven to prevent.
2. **The slot input is the boundary tap plus a per-slot embedding** (the two measured
   levers: pooling law −0.47 slope; per-slot embed doubled time-to-failure).
3. **The reader is the prefix** (Block Transformer; TG's own ablation: in-context prefix
   costs 0.4 PPL against per-layer cross-attention). Full attention by default; the mask
   is an arm (R5) whose price is measured 0.09–0.36 nats, not the design.
4. **Boundary jitter as augmentation** (the paper's random chunking is its best chunking;
   `gate_truncate_p` already does this on the gate arm) — round 2.
5. **Post-ramp LR for the core.** The ramp trades earning for stability by holding the
   core map near identity. A core-only LR boost after step 1000 (param groups; small code)
   is the next knob if R4 thinks but earns little — round 2, with the tripwire on.
6. **Decoding.** Think-once decoding needs a slot cache: run the core once when a
   boundary is emitted, then prelude + coda per token over a KV cache. The v1 generator
   recomputes the row per step. Engineering after the science, not before.
7. **The ceiling is MORPH's loop, not TUL's.** The token loop saturates by K4 on every
   recipe measured; TUL cannot earn more depth than the shared core composes. The Raven
   attention work is orthogonal to this panel and may be the real unlock.
