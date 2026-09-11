# Agent Note: Credit assignment in the slot loop

Status: proposed

Date: 2026-09-10. Arms:
[`slot-mnext-progressive`](../../../../lab/experiments/failures/2026-09-10-arc-slot-mnext-progressive.md),
[`slot-mnext-per-pass-lora`](../../../../lab/experiments/failures/2026-09-10-arc-slot-mnext-per-pass-lora.md)
and
[`slot-mnext-mux-every-pass`](../../../../lab/experiments/failures/2026-09-10-arc-slot-mnext-mux-every-pass.md),
all one-factor arms on the ruler `slot-mux-norm-match`
(`morph/configs/tul_slot_mux_norm_match.yaml`).

## Problem

The slot loop is stable and empty. Under the shipped ternary rule the M-next arm reads a
token K-curve of +0.0001 nats from depth 1 to depth 6 over 480 rows and a forecast
(`mux_local`) K-curve of +0.0067, with K3-K6 at 0.0005. Every FORWARD-side lever tried
since has read flat too: the noise entry (+0.002), the terms off (+0.003), the reread
(+0.0003), a fixed depth (a bowl, at a worse CE), a wider slot channel, a different coda.
The panel's structural finding is that the core blocks move the slot state by 2-20 % per
pass while the same blocks move token states by 86-110 %.

The per-pass gradient probe
([`lab/experiments/results/2026-09-10-slot-gradient-probe/README.md`](../../../../lab/experiments/results/2026-09-10-slot-gradient-probe/README.md))
closed the obvious backward explanation and opened a new one:

1. The cotangent does NOT vanish through the loop under the prelude entry. Per-pass shares
   on `slot-mux-norm-match` are 0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183. There is no
   vanishing-gradient problem to fix.
2. The MUX pays for the loop 7.3x over the token CE (3.5e-3 vs 4.8e-4 into the loop state).
   Without the MUX the shared core receives about 1 % of the prelude's gradient norm.
3. **The passes fight each other.** `|sum_t dW_t| / sum_t |dW_t|` on the shared core
   weights is 0.520 combined, 0.601 for the token CE alone and 0.550 for the MUX alone.
   Six orthogonal equal-norm vectors read 0.41; six aligned ones read 1.0. Per-pass cosines
   to the total run from 0.27 to 0.75, and pass 6 carries 36 % of the norm sum while passes
   2-4 carry 10-12 % each.

Reading (3) as the live hypothesis: gradient reaches every pass, and what arrives asks one
weight-shared map to be six different things at once. Wolfe's framing on 2026-09-10 —
"the slot loop fails on gradients in the loop", "the loop issue is credit assignment".

Nothing here claims (3) is the CAUSE. It is equally consistent with the map being
near-inert for a reason upstream of the gradient, in which case near-orthogonal per-pass
updates are what you would expect from six noisy readings of nearly nothing. That
ambiguity is what the two arms are built to separate.

## Proposal

Two one-factor arms on the same ruler, from the two published methods that attack
per-pass credit assignment from opposite directions. Neither changes the loss, the entry,
the target, the depth draw or the recipe.

**Arm 1, `slot-mnext-progressive` (`tul.progressive_p: 0.5`).** Bansal, Schwarzschild et
al. 2022, the Deep Thinking progressive loss. With probability 0.5 a slot of realised depth
`T_i >= 2` draws `k_i` uniform in `[1, T_i - 1]` and its first `k_i` passes have their
OUTPUT detached, so those passes contribute no weight gradient through that slot. The map
is then trained to improve states it did not itself produce, and cannot co-adapt with its
own trajectory or settle at the identity. Asks: is the disagreement a training artefact?

- Implementation: `_tul_core` draws a per-slot `[B, S]` prefix length and applies one
  `torch.where(prefix, h_new.detach(), h_new)` per pass. `k_i <= T_i - 1` keeps every
  slot's LAST pass in the gradient window, so the terminal fixed-point term and the exit
  state always sit on a grad pass. The gain hinge is masked to GRAD passes only — its
  penalty shapes the core weights, and measuring at a detached position would constrain the
  map exactly where the objective was cut away. Zero new parameters, zero extra core
  applications.
- Detaching the pass INPUT as well was implemented, measured and REMOVED: for `t >= 1` it
  is redundant (the previous pass's output detach already makes the carry a constant at
  that slot, and a prefix is a contiguous initial run), and at `t == 0` it would also cut a
  GRAD slot's own read of a neighbouring slot's entry state, which the progressive loss
  never asked to remove. The slots share one sequence; a prefix slot still serves K/V to
  grad slots, and that read belongs to a grad pass.

**Arm 2, `slot-mnext-per-pass-lora` (`tul.pass_lora_rank: 32`).** Bae et al. 2024, Relaxed
Recursive Transformers. The core blocks stay weight-shared and pass `t` adds its own
rank-32 additive branch `y_t = sublayer(x) + B_t (A_t x)`, `B` zero-init. Asks: if the
passes genuinely want to be different maps, does giving them the capacity make the loop
earn depth?

- Implementation: one `PassLoRA` module per core block (`morph/model/mhc.py`), stacked
  parameters `A [T, r, C]` / `B [T, C, r]` per target, indexed by `pass_idx` — an index
  into a parameter, not a Python branch, so `torch.compile` sees no data-dependent control
  flow. `pass_idx` is threaded from `MORPHTransformer._apply_core_step` as `iter_idx`.
- Cost, measured by building both models: 6,291,456 parameters (268,223,095 ->
  274,514,551, +2.35 %) and +1.52 % of a core block's MLP FLOPs per pass.
- Granularity differs from the paper and is named as such: one delta per SUBLAYER (the
  attention branch, the SwiGLU) rather than per linear. It cannot express a delta acting
  inside the SwiGLU nonlinearity or on q/k/v before the score.

Both arms carry a bit-identity contract at their off value, verified against the
pre-change tree rather than by reading the diff: `progressive_p: 0` and
`pass_lora_rank: 0` reproduce loss `9.1006078720092773` and sha256 `32b174ef…2853f8` over
all 208 gradient tensors of the tiny CPU model at master `c429e22`, and the zero-init
deltas reproduce the same numbers with the module BUILT.

### Arm 3, `slot-mnext-mux-every-pass` (`tul.mux_every_pass: true`)

Added 2026-09-10 after arms 1 and 2 read flat. They both left the LOSS alone and changed how
the one exit gradient is delivered. This one changes what the passes are asked for.

**Problem.** The slot loop is scored ONLY at its exit state, through two doors: the token CE
via the coda and the two prefix cells, and the MUX loss computed ONCE on the exit state
(`_forward_tul`). No intermediate pass carries a loss. Every measurement of that shape says
the same thing: the per-pass cotangent is FLAT (0.168 / 0.168 / 0.166 / 0.160 / 0.154 /
0.183 on the ruler — one gradient through a near-identity map), the coda's CE with the exit
equals its CE with the ENTRY to 0.0015 nats on the ruler and 0.0002–0.006 on every arm read
(`slot_z_optimize.py`), and every token K-curve is at or under 0.0006. Wolfe: "this is
likely where our issue actually lives. that one arm needs to be run for sure."

**Proposal.** Put the CONFIGURED MUX target (`mux_target: next`) on the state after EVERY
pass of a LIVE carry — no detach anywhere — for the slots whose realised depth reaches that
pass, plus the ruler's own final term on every valid slot. Average the terms with uniform
weights summing to 1, so `mux_beta` keeps its meaning, and report stats from the FINAL term
so `mux_local` / `mux_rel` stay comparable with every earlier arm.

- Implementation: `_tul_core` keeps the same live-carry trajectory the staged target uses,
  one entry per pass, and RETURNS the per-pass supervision mask beside it
  (`slot_valid & (depths >= j)`, ANDed with `~prefix` when the progressive draw is on). A
  return value, not an attribute: rebuilding the mask in the caller would duplicate the
  conditions (`progressive_p`, `self.training`) that decide it. Off, no trajectory is kept
  and no mask is built. Zero new parameters, zero extra core applications; the cost is `T`
  extra `[B, S, V]` fp32 readouts per step and their backward.
- TRAINING only. An eval forward computes the single final-state term, which is what leaves
  `core_depth_sweep.py`'s forced-depth `mux_local` column identical to the ruler's — the
  arm's own readout must not be changed by the arm.
- Refused where a per-pass term has no meaning: the paid loop, `mux_beta <= 0`, `db_loop`
  (detached carry, the opposite contract), `mux_stage_own_iters` (two stages, two targets),
  `cond_layers` (the think-once stack would read the final term through a stack the per-pass
  terms never see) and SCSE (the carried state is a deviation, not the slot state).
- The honest tension, written down before the run: per-pass supervision toward ONE target is
  a deep-supervision recipe, and deep supervision makes EARLY exits good. That is the same
  direction as depth-independence. The averaged objective is minimised just as well by a map
  whose first pass reaches the target and whose later passes sit near the identity — which is
  the failure already measured. The arm is worth running because it is the first one that
  touches the loss the passes see at all, not because the sign is obvious.

### Arm 4, `slot-mnext-staged` (`tul.mux_stage_own_iters: 3`)

Added 2026-09-10 after arm 3 filed and read flat exactly as the toy study below predicted
for a single-job attachment. This one is the toy's own answer to what does NOT read flat.

**The toy study.** `lab/toy_slot_loop/WRITEUP.md` (2026-09-10, the arch server's RTX 3070)
built a task that FORCES iteration by construction (S3 group composition, one span's label
at the head of the next, strict geometry so the slot loop is the only cross-span route) and
ran six loss attachments at 5 seeds each. `staged` (own-span target at every non-final pass,
next-span target at the exit) solved the chain on 5 of 5 seeds; `exit` (the ruler's own
attachment) and `deep_coda` 2 of 5; `mux_all` (arm 3's shape) 1 of 5; `progressive` (arm 1's
shape) 0 of 5. The backward shows why: `staged`'s per-pass cosine to the total core-weight
update runs +0.89 / +0.96 / +0.92 / +0.86 / +0.75 and then **-0.51** at the last pass — two
jobs separating — while `mux_all` runs +0.98 / +0.99 / +0.99 / +0.99 / +0.99 / +0.94, one
instruction repeated six times. The toy also inverted a reading this note's Problem section
took at face value: cancellation correlates NEGATIVELY with earning (-0.604 across the grid;
`summary`, the one-pass control where no loop can earn anything, reads a HIGHER ratio than
`compose`), and arm 3's measured 0.771 (up from the ruler's 0.520) is that signature.

**Proposal.** `tul.mux_stage_own_iters` already exists in the tree for exactly this shape
(arc E3, `morph/model/transformer.py` `_forward_tul`, ~lines 3800-3830) and was built and
CPU-tested on 2026-09-04 but never run on a GPU — its own configs
(`tul_to_mnext_y2_stage2/3.yaml`) predate `norm_match` and have no checkpoint directory.
`slot-mnext-staged` = the ruler `slot-mux-norm-match` plus `mux_stage_own_iters: 3`: the
state after iteration 3 supervised toward the span the slot terminates, the final state
toward the next span, mean of the two. k=3 (not "every non-final pass", the toy's actual
winner) is E3's original choice and the nearest one-factor test the existing knob can
express without new code; the toy's stronger form stays open if this arm moves anything.
Prereg: `lab/experiments/planned/2026-09-10-arc-slot-mnext-staged.md`. Filed 2026-09-10 18:27 under `lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md`: the toy's three signatures transferred (forecast K1−K6 +0.067, pass-6 cosine −0.22, cancellation 0.442) and the exit forecast stayed at the ruler's value; the later passes do a 1.87-nat own→next conversion the exit does not need. Horizon read `slot-mnext-staged-20k` filed 2026-09-11 08:47 under `lab/experiments/successes/2026-09-10-arc-slot-mnext-staged-20k.md`: HEALTHY to 20k; token-paired gap to the plain 20k model +0.192 / +0.160 / +0.163 / +0.161 at 5k / 10k / 15k / 20k (flat from 10k); every staged signature horizon-invariant (forecast K1−K6 0.06, tokens 0.002, pass 3 half the core gradient, entry-vs-exit +0.006). The overhead, not the loop, is what costs the CE.

**The honest tension.** The toy's own Q4 result is the reason to expect this arm to read
flat despite the toy's clean staged-target win: under the toy's PERMISSIVE geometry (the one
that matches MORPH — a prelude and coda that attend across spans, not the strict one built
to force the loop to matter), the same iterative task behaved like the one-pass control.
Holding the loss attachment fixed at exit-only and switching geometries moved the loop's own
contribution from +0.514 to +0.106 K1-K6 nats. If that transfers, no loss attachment —
staged included — fixes the real slot loop while the prelude and coda can already compose
across spans on their own, and the cheap test for that is not another loss arm: it is the
`tg_restrict` family already in the tree (the mask arm), read as a K-curve.

### Arms 5-8 (2026-09-10, after the k=3 stage arm was queued)

Four more one-factor arms, added the same afternoon. Arms 5-7 sit on the stage arm
`slot-mnext-staged`; arm 8 sits on the ruler. Two of them are loss/readout code
(`tul.mux_stage_all`, `tul.mux_readout`) and two are config only.

**Arm 5, `slot-mnext-staged-all` (`tul.mux_stage_all: true`, `mux_stage_own_iters: 1`).**
The toy's winning attachment IN FULL. `mux_stage_own_iters` alone supervises ONE
intermediate pass; the toy's `staged` cell supervised every non-final pass, which is the
cell that solved the chain on 5 of 5 seeds. Under this knob `mux_stage_own_iters` names the
FIRST supervised pass and the own-span term is applied at passes k .. T-1 on the slots whose
realised depth reaches each pass, averaged; the next-span term still supervises the FINAL
state alone and the loss is still `0.5 * (own + next)`, so `mux_beta` and the reported
`mux_local` keep their meaning.

- Implementation: it turns on the SAME live trajectory and the SAME per-pass keep masks
  `mux_every_pass` already collects in `_tul_core` and returns. One trajectory in the tree,
  not two. Training only, so the depth sweep's two final-state columns are unchanged.
- Off is bit-identical to the pre-change tree: loss `9.359314918518066` and sha256
  `189911591a8f60ee2f9dee7932461bca63b2bf5433d2355270763fe0461aefc0` over all 208 gradient
  tensors of the tiny CPU model at master `fe42d85` and after the change. Five sabotage runs
  fail the suite (drop the mask, supervise the final state too, detach the trajectory, sum
  instead of average, do not collect the trajectory).
- Prereg: `lab/experiments/planned/2026-09-10-arc-slot-mnext-staged-all.md`.

**Arm 6, `slot-mnext-staged-fullread` (`tul.mux_readout: full`).** Not a loss arm: a READER
arm on the same staged base. The MUX term reads the HC carrier's stream MEAN and the coda's
own reader (`prefix_project`) does not, and the audit measures the loop's update surviving
that mean at 0.139 of its per-stream norm on one arm. `full` normalises each stream before
the average, which is the tied head applied per stream with the logits averaged. It belongs
to this note only as a sibling; its problem, its alternatives and its risks are in
[`2026-09-10-slot-loop-readout-and-attention-defects.md`](2026-09-10-slot-loop-readout-and-attention-defects.md),
with arm 8.

Filed 2026-09-10 20:33 under `lab/experiments/failures/2026-09-10-arc-slot-mnext-staged-fullread.md`: the per-stream head moved the exit forecast 0.016 nats and the reader nothing (z-opt entry +0.0046, worth zero +0.061 vs staged 0.078); tokens 0.070 behind the ruler on one seed; cancellation 0.382 with pass 4 negative. F2 was a defect on paper and inert in practice.

**Arm 7, `slot-mnext-staged-mask` (`tul.tg_restrict: true`, scope "all").** The toy's OTHER
finding, and the larger of the two. Its question 4 held the loss attachment fixed and swapped
the geometry: under `permissive` — MORPH's own, a prelude causal over everything and a coda
reading a causal chain of prefix cells — the iterative task behaved like the one-pass control
(one core pass reached 0.283 nats and 66 % accuracy) and the loop's contribution fell from
+0.514 to +0.106 K1-K6. Under `strict`, where the loop is the only cross-span path, the same
attachment solved the chain. On the real model that geometry is `tg_restrict` at the default
scope, which masks the prelude AND the coda. This arm puts the staged attachment in it.

Filed 2026-09-10 19:30 under `lab/experiments/failures/2026-09-10-arc-slot-mnext-staged-mask.md`: tokens K1−K6 +0.0033 (the mask's historic 0.02-0.06 did not reproduce under norm_match with the staged loss), forecast K1−K6 +0.060 with the exit 0.021 worse than the ruler, z-opt entry +0.0235 (first above the bar), cancellation 0.385 with two negative passes; CE 0.108 behind the ruler.

The one-factor follow-up `slot-mux-mask-norm-match` (the mask on the ruler, no staged term; prereg `lab/experiments/planned/2026-09-10-arc-slot-mux-mask-norm-match.md`) filed 2026-09-11 01:18 under `lab/experiments/failures/2026-09-10-arc-slot-mux-mask-norm-match.md`: tokens K1−K6 +0.0009, so the ternary rule (`absmean` → `norm_match`), not the staged loss, removed the mask's historic token dependence; every mask K-curve in the record was measured on the starved core. Worth zero +0.554 and z-opt entry +0.0159 say the coda reads the slot hard under the mask and reads what pass 1 made; cancellation 0.556 with the ruler's cotangent shares says the mask leaves the passes alone.

Its no-MUX twin `slot-loop-mask-norm-match` (prereg `lab/experiments/planned/2026-09-10-arc-slot-loop-mask-norm-match.md`) filed 2026-09-11 02:07 under `lab/experiments/failures/2026-09-10-arc-slot-loop-mask-norm-match.md`: the coda's token loss as the ONLY loss under the same mask reads tokens K1−K6 −0.0001, exit == entry (z-opt −0.0002), the state moves 0.05 of its norm through six passes, and the core receives 0.5 % of the prelude's gradient norm (0.12 vs 21.9). H-starve: the token CE cannot train the loop even as the only path, so the MUX was load-bearing for every movement the loop ever made; the slot's target must be a per-slot supervised quantity, not the coda's loss routed through the slot.

`slot-mux-fixed-point-off` (the ruler with `core_fixed_point_lambda` 0; prereg `lab/experiments/planned/2026-09-10-arc-slot-mux-fixed-point-off.md`) filed 2026-09-11 03:10 under `lab/experiments/failures/2026-09-10-arc-slot-mux-fixed-point-off.md`: the toy's strongest lever is a state BOUND on the real model, not a contribution lever. Without it the last pass moves the state 0.44 of its norm (0.065 with it), the trajectory expands 2.2x over six passes instead of contracting, and the step-245 gradient event is 18x larger; tokens, forecast past pass 3, worth and entry-vs-exit all sit where the ruler put them. The term stays shipped on the plain-model detonation record.

`slot-mnext-gradpass` (attack 6, gradient-conditioned passes; note `2026-09-10-gradient-conditioned-slot-passes.md`) filed 2026-09-11 05:01 under `lab/experiments/failures/2026-09-10-arc-slot-mnext-gradpass.md`: the feature acts, the loop takes one descent step on its own span and holds, and the exit forecast is 0.017 nats better than the ruler's (K1−K6 +0.0236, K3−K6 +0.0016); tokens +0.0014, entry-vs-exit +0.0075. The best exit on the ruler family, one pass deep. `slot-mnext-parcae-core` filed the same night under successes: all flat on a plain Parcae core, the core is exonerated.

- Config only; no code. Kernels are copied from every mask arm since E16 and are not a free
  choice: `tg_restrict` forces `use_kernels: false`, and `tg_scoped_kernels: true` is what
  keeps the rate above the 8,086 floor (E4's fully-eager twin on this recipe read 9,168 tok/s;
  E16 measured the flag at 1.63x) AND keeps the gain hinge paired with the ruler's, because
  the eager finite difference reads 0.94 on a map whose gain is 0.87.
- The mask's own prior on this tree is the reason not to read a large token K-curve as
  success: E16 read token K1-K6 0.406 on clean math at 0.18 nats BEHIND the plain model, and
  E17 read every Sudoku bucket flat from T = 2 to 16. The mask buys forced dependence.
- Prereg: `lab/experiments/planned/2026-09-10-arc-slot-mnext-staged-mask.md`.

**Arm 8, `slot-mux-hca-fix` (`model.core_hca_compress_ratio: 16`).** Not a loop arm at all:
a correctness fix on the ruler, run because "not the cause" should rest on a repaired arm
rather than on a control that differs in four ways. Three of the six core blocks have been
delivering about half their attention output on every slot arm this campaign scored. Its
problem, its alternatives and its risks are in
[`2026-09-10-slot-loop-readout-and-attention-defects.md`](2026-09-10-slot-loop-readout-and-attention-defects.md),
with arm 6.

Filed 2026-09-11 11:54 under `lab/experiments/successes/2026-09-10-arc-slot-mux-hca-fix.md`: the fix reached the model (audit rerun: 4 blocks, |out_comp| 17.7-68.9 on core blocks 1/3/5) and the branch deficit closed (attention slot/token 0.73 / 0.83 / 1.08 against the ruler's 0.37 / 0.43 / 0.32); tokens K1−K6 +0.0004, forecast +0.0078, audit K0−K6 0.0025, z-opt entry +0.0027. F1 is closed as a lever on a repaired arm; CE 0.030 better at 5k, so the ratio is a correctness fix for `tul_short.yaml`, Wolfe's call.

Shipped 2026-09-11: `model.core_hca_compress_ratio: 16` is now `tul_short.yaml`'s default,
so every slot-loop arm carries the fix. `tul_slot_mux_hca_fix.yaml` is retired (no control
left to run it against). Agent Note:
[`2026-09-11-hca-compressed-branch-dead-on-the-slot-sequence.md`](../../implemented/bug-fix/2026-09-11-hca-compressed-branch-dead-on-the-slot-sequence.md).

## Alternatives considered

- **DEQ / implicit differentiation** (Bai et al. 2019). Solve for the fixed point and
  differentiate through it with the implicit function theorem, so the backward is a single
  linear solve instead of a product over passes — which would make "which pass gets credit"
  a non-question. Rejected for now on two grounds. First, the probe says the backward
  product is NOT the problem here: the per-pass cotangent is flat, so the thing implicit
  differentiation fixes is not broken. Second, MORPH's loop is a per-sample Poisson depth
  of mean 6 with a MUX loss read at the realised depth, not an iteration to convergence;
  making it a DEQ changes the objective, the depth draw and the stability story at once,
  which is three factors, not one.
- **HRM / TRM deep supervision with detached segments.** Supervise the state after each
  segment and detach between segments. Rejected as an ARM because this tree already ran its
  shape twice: `tul.db_loop` (detached carry, per-iteration local MUX) and
  `tul.mux_stage_own_iters` (staged targets, live carry) are both in the tree, and the
  progressive loss is the strictly more informative version of the same idea — a random cut
  rather than a fixed one, which is what forces the map to handle arbitrary states instead
  of only the segment boundaries it was trained on. Its detached-segment half is a special
  case of `progressive_p` with a deterministic `k`.
- **Universal Transformer timestep embedding.** Give each pass an explicit index signal.
  Already run: `tul.core_stage_cond: "iter"` (arc E2) and the think-once `iter` / `iter_all`
  arms. E2's finding was that per-iteration conditioning earns nothing past iteration 3
  once the loop is stable (`iter-all-r` K3-K6 +0.0001), and the 2026-08-30 cond-zero probe
  found iteration conditioning poisons depth-earning during formation. Per-pass LoRA is the
  same intent with a different lever — a per-pass WEIGHT delta rather than a per-pass
  activation shift — which is why the two are refused together in config
  (`tul.pass_lora_rank` with `tul.core_stage_cond` raises).
- **Coconut-style curriculum** (progressively replacing reasoning steps with latents).
  Rejected: Wolfe has ruled out staged-recipe curricula on this tree as unfriendly recipes,
  the same call that rejected dense-then-ternary warmup. A method whose result depends on a
  schedule nobody can reproduce from the config is not a result.
- **Deep supervision through the CODA at every pass** (the token CE, not the MUX, read at
  each pass). This is the version that would test whether the READER can use an intermediate
  state, which is the closer question to the measured failure. Rejected on cost: the coda is
  three blocks over the full 1,152-position sequence plus a full-vocabulary CE, so a term per
  pass is ~6x the coda's FLOPs and its activations, against ~1 % of a step for the MUX
  readout over 64 slot cells. If the MUX ladder moves anything, this is the follow-up worth
  the price.
- **A token CE per pass on a sub-sample of positions** (say 1/6 of the rows per pass), which
  buys the coda-side signal at the MUX's price. Rejected as a FIRST arm: it changes the
  gradient's variance and its target at once, and the sub-sample size is a tuned number
  nobody has a prior for on this tree. It is the natural fallback if the coda-side question
  survives the MUX ladder.
- **The staged target `tul.mux_stage_own_iters`** (already in the tree): the state after
  iteration k toward the span the slot terminates, the final state toward the next span. Two
  fixed stages, two different targets, and only one intermediate state supervised. It answers
  "does an intermediate state have a job", not "does every pass have one", and its mixed
  targets make `mux_local` a mixture. Refused beside `mux_every_pass` for that reason.
- **The detached DB-loop `tul.db_loop`** (already in the tree): evenly spaced supervised
  iterations with the carry DETACHED, so each term trains exactly ONE core application. That
  is the HRM/TRM shape and it deliberately removes the trajectory the arm is asking about. A
  live carry is the whole point here: a term at pass `j` must reach every earlier pass.
- **Doing nothing on the loop and attacking the state's rank instead.** The levers panel's
  structural finding is that ~50 slot cells sit at effective rank 1.7-4.8 in 1024
  dimensions. This is the real competing explanation and it is deliberately NOT one of
  these two arms: it changes what the loop is fed rather than how the loop is trained, and
  mixing it in would make the credit-assignment question unanswerable. It is the binding
  outcome if both arms read flat.

## Acceptance criteria

Per-arm predictions are frozen in the two planned files and are the binding record. At the
level of this note:

- Each arm's off value is bit-identical to the pre-change tree — loss, base weights and
  every base gradient. Verified, cited above.
- Each arm's mechanism is proven to ACT by a test that fails when the mechanism is removed.
  Done: the progressive output detach and the hinge mask each fail a sabotage run, and the
  pass-LoRA zero-init and index-threading each fail a sabotage run.
- The pass-LoRA parameters are invisible to ternary QAT, the CMS prune, the MORTAR carve
  and the deploy packer, and this is a property of their TYPE (plain `nn.Parameter` on a
  non-Linear module), not of an exclusion list. Two tests hold it.
- Either arm clears `mux_local` K1-K6 > 0.02 at 5,000 steps (the ruler: 0.0067), or the
  arm reports flat and says so.
- The per-pass gradient probe at 5,000 reports the cancellation ratio on the SHARED core
  weights for every arm, against the ruler's 0.520.
- `mux_every_pass: false` is bit-identical to the pre-change tree: loss `9.30903148651123`
  and sha256 `ee9ffa17…7bea2` over all 208 gradient tensors of the tiny CPU model at master
  `f4a284f` and after the change. Three sabotage runs fail the suite (dropping the `~prefix`
  mask, dropping the per-pass terms, detaching the trajectory).
- The every-pass ladder does not change the EVAL readout: an eval forward's `mux_local` is
  equal, tensor for tensor, with the knob on and off. A test holds it.

## Risks

- **Both arms read flat.** This is the modal outcome by the base rate of this tree: every
  forward-side lever since 2026-09-04 has read flat, and these are the first two
  training-side ones. That result is still worth having — it would move the failure past
  the backward and onto the map's near-inertness, and it retires "credit assignment" as
  the working explanation rather than leaving it as an untested intuition.
- **The LoRA arm detonates.** It adds a per-pass degree of freedom to the map's gain while
  the hinge still samples ONE random grad iteration per step, which arc E2 showed is the
  wrong instrument for a per-iteration map. `loop/core_gain_t0` is the leading indicator
  (~500 steps of warning on E7 and E13). Survival is predicted at 70 %, the lowest of the
  two.
- **A CE gain on the LoRA arm gets misread.** +2.35 % parameters usually buys a small CE
  at a fixed step count. That is a capacity reading. Only the K-curves and the cancellation
  ratio speak to depth (Wolfe 2026-09-09: short-horizon CE cannot rank looped against
  unlooped, or one density against another).
- **The deltas absorb the disagreement without changing the shared map.** H-lora-1 could be
  "true" in the sense that the per-pass parameters move and the loop still learns nothing
  new, because the shared weights end up exactly as scattered as before. The cancellation
  ratio measured on the shared weights alone is the instrument that catches this, and it is
  a frozen prediction on that arm.
- **The MUX ladder flattens the curve it is meant to steepen.** Deep supervision toward one
  target rewards a good FIRST pass, which is depth-independence. If `mux_local` at depth 1
  improves and K1-K6 shrinks, the arm has bought early-exit quality and paid in depth. That
  is a real finding and the prereg's second binding clause names it, but it is the outcome
  most likely to be misread as "the arm did nothing".
- **The memory cost is arithmetic, not a measurement.** Each per-pass term retains a 75 MB
  fp32 logit tensor and its log-softmax output at B=6, S=64, V=49169; eight extra terms hold
  on the order of 1.2 GB more on a card the ruler already runs at ~11 GB peak. The smoke's
  peak is the first real number.
- **No arm has run on the GPU.** Everything above is CPU builds, CPU tests and config
  compose checks. `torch.compile` on the new index path and the new mask path is untested
  until the runner's 12-step smoke.

Filed 2026-09-11 14:53 under `lab/experiments/failures/2026-09-11-arc-span-budget.md`: the cross-span BUDGET with no slot cells (Parcae core, `model.span_mask` row vs span) is 0.3994 nats [0.3838, 0.4162] at 5k and rising (0.163 at 2,500); flat 0.31 at offset 8+, 0.96 at the span's first position, 0.28 at offset 0. The prefix write's worth (0.093) is about a quarter of it. The loop is unchanged under the cut (K1−K6 +0.0301 vs +0.0279). Ceiling recorded; next is the route split on the mask arm.
