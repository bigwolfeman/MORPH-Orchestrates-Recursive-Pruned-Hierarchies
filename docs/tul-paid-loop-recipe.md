# The paid loop: how TUL came to earn its depth, and the recipe that trains it

A literate record, written 2026-09-03 00:00 at the end of the campaign that produced it.
It is the "lock-in" Wolfe asked for: what worked, why it worked, and where each claim's
number lives, so that the next person (or the next session) does not have to re-derive
any of it. Every number here is read from a filed experiment; the file is named next to
it. Two slots are still open at the time of writing and are marked OPEN.

Read this with `docs/tul-spec.md` (what TUL is) and
`lab/divergence/DIVERGENCE-README.md` (what to do when a run blows up) at hand.

## 1. The claim, in one paragraph

MORPH's looped core earns depth only when the positions whose loss you measure are the
positions that run the loop. Under TUL as specified, tokens skipped the core and only the
slot positions looped, so the loop was a free ride for every token: it earned 0.015 nats
at 20k and TUL lost to the plain looped model by 0.357 nats. Sending the tokens through
the same per-sample core, with the slots still present in the sequence (arm A2,
`tul.tokens_through_core`), brought the earning to 0.17 nats at 5k and made A2 the best
5k checkpoint of every arm. Paying the loop reopened an early-training detonation that
was a property of the whole recipe, not of TUL; it lives in steps 200 to 775, ternary
QAT is the surface it shows on, and a 1000-step linear LR ramp closes it (0 of 3 draws
detonated, 0.14 nats better at 2500) at the price of a smaller loop earning at that step,
whose fate at 5k and 20k is the open question.

## 2. Why the loop was a free ride, mechanically

Start from the forward as `morph/model/transformer.py` runs it. `_forward_tul` builds one
shared position axis of tokens and slots (`morph/model/tul_layout.py` packs it: one slot
after each span, boundary rule `.;!?` plus newline plus dashes, min span 4, cap 32). The
prelude runs on every position. Then the spec said: gather the slot positions, loop the
6-block core on them for a Poisson number of iterations, scatter them back, and hand the
coda the prelude output for the tokens (`input_norm(prelude)`, the `n_core == 0` path).
The coda runs on everything and the loss is token cross-entropy.

Follow the gradient. Token CE at position t depends on the coda, on the prelude at t, and
on the slot states that t attends to. It depends on the core only through those slot
states. A slot's state is one `E_slot` plus the bag-mean of its span's token embeddings,
looped. So the core's entire contribution to the loss is whatever the coda can extract
from a handful of looped bag-means. Three measurements said that channel carries almost
nothing:

- The loop's own depth curve on tul-20k: K1 3.7721 to K6 3.7568, 0.015 nats, saturated
  by depth 2 (`lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md`).
- The slot's plan, ablated at eval, is worth about 0.07 nats of its span; the model was
  trained to predict ONE token from it (memory `morph-tul-plan-is-empty`).
- The write-side ladder tried to fix the slot's INPUT (boundary seed R0, content seed
  W1, bound seed W2) and moved the loop's earning from 0.011 to 0.001 and 0.002
  (`lab/experiments/failures/2026-09-01-write-side-ladder.md`). The core ignored its
  input rank. Rank was never the lever.

Meanwhile the plain looped model with no slots (notul, `notul_bg0c0`) earned 0.220 at
4500 and 0.207 at 20k on the token axis. Same core, same weights class, same recipe.
The one difference was that its tokens ran the loop and its loss was measured on them.
That is the whole explanation: earning follows payment.

## 3. Arm A2: pay the loop, keep the slots

The fix is small because `_core_region` had already been extracted verbatim from the old
inline loop ("pure code motion", the docstring at `transformer.py:~1568`). A2 sets
`tul.tokens_through_core: true`, and `_forward_tul` takes the first branch:

```python
if tc.tokens_through_core:
    # tokens AND slots run the ordinary per-SAMPLE core
    x_coda = self._core_region(x, x0, bigram_emb, input_ids, attn_kwargs=tg_attn_kwargs)
```

Everything the loop already did for a plain sequence it now does for the packed row:
Poisson depth per sample (not per slot), the DiagonalInjection, the hyper-connection
carrier. The slots are still there, still seeded by the bag-mean, still attended by the
tokens as ordinary positions; the only change is who runs the core. A2 isolates one
variable against the notul twin: the presence of slots.

Config lineage (`morph/configs/`): `tul_a2` = `tul_g0c0` (retention off, cap 0) +
`tokens_through_core: true` + `tg_restrict: false` (the restriction is unspecified with
A2 and raises) + `mux_beta: 0` + `use_kernels: true`. `tul_g0c0` = `tul_l2` minus GLA
minus the cap; `tul_l2` = `tul_l1` + cap; `tul_l1` sits on `tul_short` (seq 1024, TUL
from step 0). Every arm in the campaign is one YAML delta from its parent, on purpose.

What it measured (`lab/experiments/successes/2026-09-01-a2-paid-loop.md`, 5000 steps,
batch 6, seed 1, flat LR 1e-4):

| cell | arm | K1 minus K6 | K6 CE | final val |
|---|---|---|---|---|
| paid + slots | A2 | 0.1685 | 4.1711 | 4.2315 |
| paid, no slots | R1 (notul twin, retry) | 0.1937 | 4.7548 (weak draw) | 4.7666 |
| free + slots | R0 | 0.0113 | 4.4689 | ~4.55 |
| free + slots, best seed | W1 | 0.0007 | 4.3395 | ~4.42 |

A2 earns 87% of the notul twin's depth with slots interleaved and has the best CE of
any 5k checkpoint. The restricted variant A2s (same-span-or-slot attention inside the
core) detonated 0 of 2 and never produced a number
(`lab/experiments/failures/2026-09-02-a2s-restricted-paid-loop.md`), so A2 won the 20k
slot by walkover.

Two things A2 is NOT. It is not a leak: corrupting every token after a cut moves the CE
before the cut by exactly 0 nats, in bf16, at both checkpoints, both cuts, three depths,
in the all-positions and the boundary mode
(`lab/experiments/successes/2026-09-02-a2-future-leak-probe.md`). And it is not the
takeover: the A1 slot-loop failure (val CE turning around after step 500 to 2000, forward
slot-state rank collapse, `lab/divergence/takeover-campaign.md`) does not occur on A2,
because the cotangent at the core is a sum over ~1152 positions per row instead of 50.

The eval-side lever for A2 is `model.cfg.mean_depth`, not the slot depth; the slot
depth knobs are inert on A2. `lab/divergence/_build.py::DepthLever` picks the knob per
arm, and `a2_depth_sweep.py` / `future_leak_probe.py` go through it.

## 4. The detonation, and what it turned out to be

Paying the loop reopened a failure that looked like the β1=0 AdEMAMix detonations of
Task #276: the pre-clip gradient norm spikes from step ~200 to ~330 and compounds under
clipping until the loss is a corpse by step 2000. The shipped div-guard (ppl > 1000
from step 2000) killed every one of them at step 2040. It happened on 5 of 7 paid draws
at first, then on notul too (R1 1 of 2), and on notul-20k's own recipe it had simply not
happened in that one draw.

What it is not, each with the experiment that closed it:

- Not stale second moments: the Task #276 cure is active in the fused kernel; the M2G
  onset capture found no stale-m2 signature
  (`lab/experiments/failures/2026-09-02-m2g-onset-capture.md`).
- Not TUL: notul-20k, R1, A2, A2s, and every gamma draw share the recipe (retention
  off, cap 0, ternary on, alpha_cap 3.5, flat 1e-4, warmup 0) and the ~70% per-draw
  rate.
- Not a leak (section 3).
- Not the ternary scale's contagion, at least not in a way you can fix by slowing
  gamma: a slow EMA on gamma (β 0.99) detonated 2 of 3 and cost the healthy draw 0.30
  nats (`failures/2026-09-02-gamma-ema-paid-validation.md`); a hard freeze detonated 3
  of 3 and never learned at all, val flat near 7.4, because a gamma pinned at the step-0
  mean|W| bounds every effective weight at that scale while the shadow weights grow
  (`failures/2026-09-02-gamma-freeze-discriminator.md`). The live per-forward gamma is
  load-bearing.

What it is, measured:

- Ternary QAT is the trigger surface: ternary-off draws were 3 of 3 healthy against the
  ~70% base rate.
- It is an early transient. Over every probe file on disk, 17 of 17 detonations cross
  `preclip/total` = 1e3 between step 200 and 775 and reach 1e4 within 146 steps; 44
  healthy runs, five of them past 20k, never exceed 830
  (`lab/experiments/results/detonation_onset_scan.csv`). That gives an abort rule with
  zero false positives: `preclip/total > 1e4 at any step ≥ 200`. It is in
  `DIVERGENCE-README.md` and in every runner since as a bash tripwire; it is not in the
  trainer yet.
- The healthy paid map is expansive and drifting outward. Power iteration on the core
  step's Jacobian (`lab/divergence/jac_ladder.py`, method in
  `docs/cookbook/measuring-the-core-map.md`) gives, at the first loop iteration, a
  typical gain of 1.05 at step 2500 and 1.13 at 5000, worst direction 55 and 97. At the
  diverged checkpoints the whole-step typical gain is 9 to 159 while each block's is 1.0
  to 1.3: a low-rank blowup carried by ten to twenty directions aligned across the six
  weight-shared blocks (`failures/2026-09-02-a2-core-jacobian-ladder.md`; filed as a
  failure by protocol because the alignment prediction came in against its prior).
  Wolfe's phrase for it was "the aggressive minimum is unstable", and that is what the
  instrument shows.

## 5. The warmup, and what it costs

The recipe ran a flat 1e-4 with `training.warmup: 0`, kept flat on purpose so that every
experiment in the ledger is comparable. The danger window is exactly the interval a
normal recipe spends ramping. `morph/training/optimizer.py` already had the ramp:

```python
if step < warmup_steps:
    return lr_max * step / max(1, warmup_steps)
```

`training.warmup=1000` on `tul_a2`, everything else unchanged
(`lab/experiments/failures/2026-09-02-a2-warmup-and-seq512.md`):

| | detonations | final val at 2500 | K1 minus K6 at 2500 |
|---|---|---|---|
| flat (clean A2) | ~70% per draw | 4.6776 | 0.1209 |
| warmup 1000, three draws | 0 of 3 | 4.5391 / 4.5440 / 4.5413 | 0.046 / 0.049 / 0.046 |

The ramp closes the window and is 0.14 nats better at the same step with a 0.005 spread
across draws. It also leaves the loop earning 40% of what the flat schedule earned at
2500; the warmup model at depth 1 already beats the flat model at depth 6. Whether that
is the earning killed or merely late is undecidable at 2500, because clean A2's own
earning grew from 0.12 to 0.17 between 2500 and 5000.

The file is in `failures/` by protocol (the earning prediction missed), and that is the
right filing: the recipe question was answered in the direction that matters and the
earning question was not.

Two things the warmup result does not license. It does not show a late detonation is
impossible: the map drifts outward while the loss falls, and only one run past 20k
(notul-20k) is on this recipe. And it does not stand alone: any 20k comparison on the
ramped schedule needs the notul baseline rerun on the same schedule, or the ledger's
0.357 and 0.207 stop meaning anything (Wolfe, 2026-09-02 22:33).

## 6. The recipe, as of 2026-09-03 00:00

`--config-name tul_a2` plus `training.warmup=1000`, with the panel flags used by every
arm in the campaign (`training.batch_size=6 training.seed=1
training.ademamix_alpha_cap=3.5 training.ademamix_t_beta3=3500`), seq_len 1024,
retention off, spectral cap 0, ternary QAT on (backbone, 127.8M params), int6 embed QAT,
β1=0 AdEMAMix, flat 1e-4 after the ramp, `export
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and the 1e4 tripwire on the probe.

Things that are OUT, with the reason, so they do not come back by accident:

| lever | why it is out |
|---|---|
| spectral cap / projection | four variants failed against the takeover, the cap kills depth-earning, and a uniform rescale cannot slow an alignment (`lab/experiments/failures/2026-08-24-tul-takeover-cure.md`, memory `l2cap-depth-earning-was-the-leak`) |
| GLA / retention as a stabilizer | cost 0.18 nats and 0.08 of earning in the bisect (`successes/2026-08-31-loop-killer-bisect.md`); the acausal carry was a learned leak (memory `morph-not-causal-retention-carry`). OPEN: whether it earns its place under warmup, see section 7 |
| ternary-gamma EMA or freeze | section 4 |
| dense warmup before ternary | ternary weights organize differently (Wolfe, 2026-09-02) |
| `core_gain_clip` | clamps the realized magnitude, not the map's gain |
| seq-length curriculum as the stabilizer | deferred to Wolfe's later warmup-on-length work; one stability probe would not have been worth its GPU |
| slot-loop-only TUL (A0/A1/A3 as the production arm) | section 2 |

## 7. OPEN at the time of writing

- **GLA under warmup.** `2026-09-02-a2-gla-under-warmup.md`, three draws queued
  behind wu5k. A frozen rule decides whether the 20k pair runs with retention: at least
  two healthy draws, mean val at 2500 at most 4.4915 (0.05 below the warmup mean), mean
  earning at least 0.026.
- **Earning at 5k under warmup: MEASURED 2026-09-03 00:16, reduced, not delayed.**
  `failures/2026-09-02-a2-warmup-5k-earning.md`: wu5k healthy to 5000, final val 4.1807
  (clean A2 4.2315), K1−K6 0.058 against clean A2's 0.1685 at the same step, saturated by
  depth 5, depth-8 tail flat. The warmup model's depth-1 CE (4.152) beats clean A2's
  depth-6 CE (4.171). The loop is a smaller, saturating contributor under the ramp; the
  recipe is better anyway. If the pair confirms it at 20k, mean depth can come down.
- **The matched 20k pair.** `2026-09-02-warmup-20k-pair.md`: A2 and notul, both on the
  ramp, same seed, tripwired; the honest prior for "A2 beats notul at 20k" is 40%,
  because the notul twin was 0.05 nats ahead at 5k on the flat schedule.
- **Trainer-side abort-and-retry** at the 1e4 rule with checkpoint rollback and reseed.
  Bash does it today; the trainer does not.
- **dmorph**, the no-loop TUL with a flow-matching objective at matched wall-clock and
  matched tokens (`.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md` on
  branch `feat/db-objective-l2`): a dedicated build session, because the wall-clock
  claim needs its kernel first.
- **Raven**, a GLA variant Wolfe wants to test once TUL is stable (2026-09-03 00:10),
  as the second candidate for whatever GLA was buying. No design notes in the repo yet.

When the three runs land, sections 6 and 7 get their numbers and the winner's config is
updated on this branch. The decision record that points here is
`.agents/notes/implemented/architecture/2026-09-02-paid-loop-warmup-recipe.md`.
