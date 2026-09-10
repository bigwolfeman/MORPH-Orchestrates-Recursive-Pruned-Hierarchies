# Planned: the M-next MUX loss on every pass of the slot loop

Status: failure

Date: 2026-09-10 (frozen before launch; Wolfe on the exit-only supervision: "this is likely
where our issue actually lives. that one arm needs to be run for sure"). Arc:
`2026-09-04-loop-contribution-arc.md`. Follows the two credit-assignment arms that already
ran — [`slot-mnext-progressive`](../failures/2026-09-10-arc-slot-mnext-progressive.md) and
[`slot-mnext-per-pass-lora`](../failures/2026-09-10-arc-slot-mnext-per-pass-lora.md) — and
the probe they were built on,
[`results/2026-09-10-slot-gradient-probe/README.md`](../results/2026-09-10-slot-gradient-probe/README.md).
Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).

## Question

The slot loop gets gradient ONLY at its exit state. Two doors reach it: the token CE
through the coda and the two prefix cells (`tul.W_prefix`), and the MUX loss computed ONCE
on the exit state (`transformer.py::_forward_tul`, `mux_loss = self._tul_mux_loss(h_slots,
...)`). No intermediate pass carries a loss of its own.

Three measurements from 2026-09-10 say the same thing about that shape. The per-pass
cotangent is FLAT — shares 0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183 on the ruler,
which is what ONE gradient looks like passing through a near-identity map, not what six
supervised passes look like. The coda's CE with the loop's exit equals its CE with the
loop's ENTRY to 0.0015 nats on the ruler and 0.0002–0.006 on every arm read
(`slot_z_optimize.py`). And every token K-curve is at or under 0.0006, with the forecast
K-curve at 0.0067.

Does giving every pass its own supervised target — the SAME M-next target the exit state
already carries, on a live carry, with the weights summing to 1 — change what the loop
does? This is the first arm whose mechanism the probes point at directly: the two credit
arms before it changed WHICH passes carry the one exit gradient (`progressive_p`) or gave
the passes their own parameters (`pass_lora_rank`), and both read flat. This one changes
what the passes are asked for.

## Hypothesis

H-mep-1 (for): the loop is empty because only its last state is ever scored. Supervise
every pass and the map has to make pass `t+1` better than pass `t` on the slots that reach
it — the only way to lower the average of `T+1` terms is for the trajectory to improve —
so the forecast K-curve steepens first and, if the coda reads the state at all, the token
curve follows.

H-mep-0 (against, and it is not the usual null): per-pass supervision toward ONE target is
a deep-supervision recipe, and deep supervision is known to make EARLY exits good. That is
the same direction as depth-independence, not the opposite of it. The averaged objective is
minimised just as well by a map whose FIRST pass reaches the target and whose later passes
sit near the identity — which is exactly the failure mode already measured (core blocks
move the slot state 2–20 % per pass). On this reading the arm lowers `mux_local` at depth 1
and the K-curve gets FLATTER, not steeper, while the token curve does not move at all.

Both readings predict the per-pass ladder `loop/mux_pass_t*` moves. They disagree about its
SHAPE, which is why the ladder is a readout and not a prediction.

## Method

`tul_slot_mnext_mux_every_pass` = `tul_slot_mux_norm_match` (the ruler `slot-mux-norm-match`)
plus ONE change: `tul.mux_every_pass: true`.

The mechanism, as implemented. `_tul_core` already keeps a live-carry trajectory for the
staged target; under this knob it keeps one entry per pass (`_db_traj[j]` is the masked
carry after pass `j`) plus, alongside it, the per-pass supervision mask `_mep_keep[j-1] =
slot_valid & (depths >= j)`, and returns both. Nothing is detached anywhere: the entries are
outer-graph states, so a term at pass 1 backpropagates through pass 1's core application AND
into the prelude, and the term at pass `j` reaches every earlier pass. `_forward_tul` then
builds `T` terms — `_tul_mux_loss(traj[j], slot_keep=mep_keep[j-1])`, the configured target
(`mux_target: next`) at each one — plus the ruler's own final term on every valid slot, and
averages them (uniform weights summing to 1, so `mux_beta` keeps its meaning). Stats
(`mux_local`, `mux_rel`, `mux_kl`, `mux_n_supervised`) come from the FINAL term, so the
sweep columns stay comparable with every arm before this one. The ladder is TRAINING only:
an eval forward computes the single final-state term, which is what leaves
`core_depth_sweep.py`'s forced-depth `mux_local` column unchanged.

Interactions, each one checked rather than assumed. The gain hinge probes one random grad
iteration at the detached operating point and does not care which passes carry a loss:
untouched. The terminal fixed-point term applies at each slot's last pass: untouched. The
progressive prefix is OFF on this arm; if it is ever on, a detached prefix pass must not
carry a term (it would train nothing through that slot and still take a share of the
average), so the mask carries `~prefix` and a test holds it. The coda still reads the
loop's FINAL state — only the loss changes.

`mux_every_pass: false` is bit-identical to the pre-change tree, verified directly rather
than by reading the diff: the tiny CPU slot-loop model at seed 3/7 gives loss
`9.30903148651123` and sha256
`ee9ffa170414e8311cf714b65b989bfe1bfbf240906466e2d959b213e447bea2` over all 208 gradient
tensors both at master `f4a284f` and with the knob off after the change. The knob is refused
under the paid loop, at `mux_beta <= 0`, beside `db_loop`, beside `mux_stage_own_iters` and
beside `cond_layers` (the think-once stack would read the final term through a stack the
per-pass terms never see). Three sabotage runs fail the suite: dropping the `~prefix` mask,
dropping the per-pass terms, detaching the trajectory.

Everything else is the ruler: M-next MUX at beta 1 through the tied head, prelude entry,
hinge lambda 100 at target 0.9, `core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0,
`ternary_scale_mode: norm_match`, seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000,
`tg_scoped_kernels`.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues (the ruler read 12,429).

Readout: `core_depth_sweep.py` at checkpoints 2,500 and 5,000, depths 1,2,3,6,9,12,16,
reporting both the token K-curve and the `mux_local` forecast K-curve on 480 rows;
`worth_profile.py` and `slot_state_probe.py` at 5,000 (the runner's `slot` kind does both);
`slot_gradient_probe.py` and `slot_z_optimize.py` at 5,000 by hand — the gradient probe
carries the cancellation ratio and the per-pass cotangent share, and the z-optimisation
probe carries the exit-minus-entry CE, which are the two quantities the mechanism claims to
move. New wandb keys to read alongside: `loop/mux_pass_terms`, `loop/mux_pass_t{1..T}`,
`loop/mux_pass_final` — the SHAPE of that ladder over training is the arm's own instrument,
and neither hypothesis is scored on it.

The ruler's numbers, cited once: 480-row CE at depth 6 = **4.3290** at step 5,000; token
K1−K6 +0.0001 [−0.0000, +0.0002], K3−K6 −0.0000; `mux_local` K1−K6 +0.0067 [+0.0053,
+0.0081], K3−K6 +0.0005; wall clock 52 min 38 s for 5,000 steps; combined cancellation ratio
0.520; per-pass cotangent share 0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183; z-optimisation
`ce_loop` 4.1173 with the loop ENTRY worth +0.0015.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at step
  >= 200): **82 %**. Reasoning: the ruler ran HEALTHY with a max preclip of 78.7 at step 245,
  and this arm adds SHORT gradient paths (a term at pass `j` reaches the prelude through `j`
  applications) beside the long one rather than lengthening anything. The detonation modes on
  this tree are the backward product through the loop and a first-pass scale runaway; a
  direct term at pass 1 pushes on the second. The 18 % is for that: pass 1 now carries its
  own loss on top of the chain, and `loop/core_gain_t0` is the leading indicator.
- **P-b (tokens).** Token K1−K6 at 5,000 above 0.03 (the plain prelude-entry ruler
  `plain-panel-norm-match`): **15 %**. Above 0.10 (the plain NOISE-entry value from E19's
  `parcae-entry`): **5 %**. Reasoning: the coda reads the exit state ONCE and reads it
  through a path whose CE is indifferent to it (exit minus entry 0.0002–0.006 nats on every
  arm). Nothing in this change touches the reader. The 15 % is the one route that exists: if
  the per-pass terms make the exit state genuinely different from the entry, the reader's
  indifference has something new to be indifferent to, and it could break either way.
- **P-c (forecast K-curve).** `mux_local` K1−K6 above 0.02 at 5,000 (the ruler: 0.0067):
  **30 %**. K3−K6 above 0.002 (the ruler: +0.0005): **20 %**. Reasoning: this is the target
  the mechanism supervises, and it is the one curve on this loop that is not already zero. But
  the mechanism cuts BOTH ways and I will not pretend otherwise — supervising pass 1 toward
  the same target directly rewards a state that is already good after one pass, which is the
  definition of a flat K-curve. H-mep-1 needs the map to improve the trajectory; H-mep-0 gets
  the same loss by making the first pass good and the rest near-identity, which is what the
  forward instruments already measure it doing. Under even money because the two effects have
  opposite signs and nothing in the tree tells me which is larger.
- **P-d (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above 0.02
  nats (the ruler: +0.0015; every arm read so far: 0.0002–0.006): **20 %**. Reasoning: a
  13x move on a number that has never left a narrow band across six arms with very different
  loops. The mechanism can plausibly make the exit state move further from the entry, but
  "moves further" and "moves in a direction the coda's readout uses" are different claims,
  and only the second one shows here.
- **P-e (per-pass cotangent).** Share at pass 1 below 0.10 (the ruler: 0.168): **8 %**.
  Reasoning: I think the framing behind this bar is backwards for THIS mechanism, and say so
  before the run. `slot_gradient_probe.py` runs `model.train()`, so the every-pass ladder
  fires inside the probe: pass 1's output then receives its OWN term's cotangent on top of
  the chain from the exit, which mechanically RAISES its share. A reading below 0.10 needs
  the chain contribution to collapse by more than the direct term adds. What a working
  per-pass signal actually looks like here is a flat-or-higher pass-1 share with the
  per-pass dW norms and the cancellation ratio moving — which P-f scores.
- **P-f (cancellation).** Combined cancellation ratio above 0.60 (the ruler: 0.520; the
  progressive arm read 0.497, the per-pass-LoRA arm is filed beside it): **40 %**. Reasoning:
  every pass is now asked for the SAME thing at its own state, which is the most direct
  reason six shared-weight updates would point the same way, and it is the first arm that
  asks it. Against: the states differ across passes, so the same target still induces
  different updates, and the two arms that attacked this quantity from the training side both
  moved it DOWN or not at all.
- **P-g (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **50 %**.
  Reasoning: the objective gains `T` terms whose weight comes out of the single exit term, so
  the exit state is optimised less directly for the same `mux_beta`. That is a real change to
  what the model spends its steps on, and 5,000 steps is a short horizon. I expect a small
  move rather than a large one, in either direction, and I am NOT treating a CE gap at 5,000
  steps as a verdict (Wolfe 2026-09-09: short-horizon CE cannot rank looped against
  unlooped).
- **P-h (cost).** Wall clock within 1.3x of the ruler's 52 min 38 s: **90 %**. Reasoning: the
  arithmetic is small — `T` extra `[B, S, V]` readouts at B=6, S=64, V=49169 is about 0.46
  TFLOP per step with the backward, against roughly 50 TFLOP for the step, so under 1 %. The
  risk is MEMORY, not FLOPs: each term retains a 75 MB fp32 logit tensor and its log-softmax
  output, so eight extra terms hold on the order of 1.2 GB more, and if that pushes the
  allocator the step time moves with it. The smoke's peak is the first reading of this.

## Binding

- P-c TRUE (`mux_local` K1−K6 > 0.02) and the ladder `loop/mux_pass_t*` decreasing with the
  pass index ⇒ per-pass supervision is the lever: the next arm carries it to a 20k horizon
  and combines it with the ONE forward-side lever that moved anything (the fixed depth's
  bowl), before any further lever search.
- P-c FALSE with `mux_local` at depth 1 BETTER than the ruler's ⇒ H-mep-0: the arm bought
  early-exit quality and paid for it in depth. That is a real finding about deep supervision
  on this loop and it retires "the passes are unsupervised" as an explanation. The next
  question is a target that CANNOT be met in one pass, not another way to deliver the same
  target.
- All flat (token and forecast curves inside the ruler's confidence intervals) ⇒ the third
  and last credit-assignment attack has failed. Combined with the two filed arms, the honest
  next step is not another arm on this loop: it is whether ~50 slot cells carrying a
  rank-2-to-5 state can support a depth-6 map at all, which is the binding the two earlier
  arms already wrote down.

## Not verified before launch

The arm has not run on a GPU: everything above is a CPU build, a CPU test suite and a config
compose check. `torch.compile` behaviour of the per-pass loss terms is untested until the
runner's 12-step smoke — the terms are a Python list comprehension over a trace-time-constant
range with no data-dependent branch, but that is an argument, not a measurement. The memory
cost of keeping the trajectory live at the REAL shape (B=6, S=64, L_total=1152, up to 8
passes) is unmeasured; the estimate above is arithmetic on tensor sizes, and the smoke's peak
is the first real number. Uniform weights are a choice, not a tuned value: no weighting
sweep (later passes weighted higher, an exponential ramp) is planned before this draw. The
per-pass terms are computed at the REALISED Poisson depths, so the number of terms varies per
step and the deepest term is supervised on few slots — the effect of that variance on the
gradient's noise is unmeasured. `loop/mux_pass_*` has been read on the tiny CPU model only.

## Results

Filed 2026-09-10 17:20. `slot-mnext-mux-every-pass` (commit `bf87aeb`, `tul.mux_every_pass`):
HEALTHY to 4,999, tripwire max 815 at step 245 (ruler 78.7, threshold 1e4), 12,063 tok/s at
step 200, smoke peak 12.62 GB (ruler 10.1: the retained per-pass logits), wall clock 52.1 min
(ruler 52.6, 0.99x). Trainer `[VAL 4750]` 4.4359 (ruler 4.4486); runner final val_loss 4.3682
(ruler 4.3775).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16:

| step | tokens K1−K6 | tokens K3−K6 | `mux_local` K1−K6 | `mux_local` K3−K6 | CE@6 (ruler) | MUX@1 / MUX@6 (ruler @6) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,500 | −0.0001 [−0.0002, −0.0000] | −0.0001 | −0.0002 [−0.0009, +0.0005] | −0.0013 [−0.0017, −0.0010] | 4.7359 | 6.8676 / 6.8678 |
| 5,000 | −0.0000 [−0.0001, +0.0001] | −0.0000 | +0.0001 [−0.0005, +0.0008] | −0.0007 [−0.0010, −0.0003] | 4.3061 (4.3290) | 6.7698 / 6.7696 (6.7724) |

The MUX at depth 1 (6.7698) is where the ruler sits at depth 6 (6.7724): the per-pass loss
bought the ruler's whole forecast quality in ONE pass and left nothing for passes 2–6.
Worth profile at 5,000: zero +0.046 [+0.038, +0.053], +0.025, +0.020, +0.019, +0.012;
shuffle +0.043, +0.033, +0.025, +0.022, +0.015. State probe: |h| 111 at depth 1 → 131 → 137
at 6 → 140 at 16; relative distance from depth 1 0.22 / 0.29 / 0.35, cos 0.99 / 0.99 / 0.98.

Gradient probe at 5,000 (12 rows, depth 6; the ladder fires inside the probe because it runs
in train mode): combined cancellation ratio **0.771** (ruler 0.520), MUX alone **0.850**
(ruler 0.550) with per-pass cosines to the total 0.70 / 0.83 / 0.89 / 0.93 / 0.94 / 0.85
(ruler 0.33 / 0.27 / 0.37 / 0.63 / 0.75 / 0.67); token CE alone 0.497. Per-pass cotangent
share at pass 1 (token CE) 0.17, flat as before. z-optimisation probe: ce_loop 4.1015, entry
+0.0016, zero +0.0113, shuffle +0.0116, fitted z −0.915 (random start −0.821), loop z rank 13.0.
Artifacts: `lab/experiments/results/2026-09-10-slot-mnext-mux-every-pass/`; npz and probe
JSON under `ignored/experiment-artifacts/`.

## Verdict

P-a TRUE. P-b FALSE. P-c FALSE (0.0001). P-d FALSE (0.0016). P-e FALSE (0.17). P-f TRUE
(0.771 > 0.60) — and it is the toy study's signature of the FAILURE mode, not of learning:
`lab/toy_slot_loop/WRITEUP.md` reads stuck runs at 0.95 and learning runs at 0.29–0.75, and
predicted this arm "flat or worse, with its ratio rising" before the arm finished. P-g TRUE
(0.023 better). P-h TRUE (0.99x). Binding clause 2 applies (H-mep-0): the arm bought
early-exit quality and paid for it in depth; "the passes are unsupervised" is retired as an
explanation. Every pass was asked for the SAME target and every pass answered the same way
(MUX cosines 0.7–0.94), so the passes became one pass.

## Updated hypothesis

Per-pass supervision toward one fixed target is depth-INDEPENDENCE training. What the toy
found to work is a target that changes with the pass: the slot's OWN span at the non-final
passes and the NEXT span at the exit (staged, 5 of 5 seeds vs 2 of 5 exit-only), so that the
intermediate passes have a reachable job that differs from the exit's. That mechanism exists
in the tree as `tul.mux_stage_own_iters` (E3, never run); arm `slot-mnext-staged` is queued.
The toy's second finding bounds all of this: under MORPH's own geometry, where the prelude and
the coda attend across spans, the toy's iterative task behaved like its one-pass control; the
loop only earned depth when it was the ONLY cross-span path. On the real model that is the
mask arm (`tg_restrict`), the one slot arm that ever read token K1−K6 above 0.02.
