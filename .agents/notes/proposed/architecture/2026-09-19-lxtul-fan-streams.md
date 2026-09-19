# Agent Note: LXTUL — K latent streams, a repulsion, a gate and an oracle

Status: proposed

## Problem

The TUL slot loop does not earn depth. Twelve arms read token K1−K6 inside
[−0.0001, +0.0033] against the plain looped model's +0.033, and pass 1 does 89–95 % of the
work. Every target tried so far has a ONE-STEP OPTIMUM: each is the conditional mean of
some future quantity, so nothing in training asks pass 2 to differ from pass 1.

The 2026-09-18 latent-exploration survey
([`docs/references/looping-depth/latent-exploration/2026-09-18-latent-exploration-survey.md`](../../../../docs/references/looping-depth/latent-exploration/2026-09-18-latent-exploration-survey.md))
maps nine papers onto {one stream, K streams} x {deterministic, stochastic}. Three cells
are measured on this tree and all three are flat: the strict slot loop, LCTUL and LCTUL-D,
and the Thought Register (`tul.slot_cells: 4`, 2026-09-13 —
[`2026-09-13-the-thought-register.md`](2026-09-13-the-thought-register.md), record
[`lab/experiments/failures/2026-09-13-arc-thought-register.md`](../../../../lab/experiments/failures/2026-09-13-arc-thought-register.md)),
which collapsed to effective rank 1.2445 of 4 at within-slot cosine 0.9432 — closer
together than the 0.8443 it starts at. The fourth cell is empty.

The register is not that cell and the survey says exactly why. It had K streams and
nothing else: no term that pushed them apart, no selector allowed to pick between them,
and no instrument that could have said whether picking would have helped. PLR (arXiv
2601.03153) supplies the closed form for the collapse — under an L-Lipschitz shared map
mean pairwise stream distance obeys `D(T) = L^(2T) D(0)`, and MORPH's core is measured
contractive at gain 0.89 — and its ablation says the GATE is the largest single
contributor, ahead of the repulsion.

Worse, the register's one positive number was unreadable. It beat the strict ruler by
0.0217 nats on paired depth-6 CE, and it differed from that ruler by TWO things: the cells
and a four-times-wider prefix write. The record attributes the win to the width, and the
clean width control has never run. Any new K-stream arm that keeps the 1:1 cell write
inherits that confound.

## Proposal

Build the empty cell as `tul.fan_k` (family name LXTUL, beside LCTUL), arm
`slot-spandec-strict-fan4`, config `morph/configs/tul_slot_spandec_strict_fan4.yaml`, one
factor against `tul_slot_spandec_strict` at `prefix_k: 4`.

**Reuse the register, do not rebuild it.** `tul.fan_k` ALIASES `tul.slot_cells` in
`morph/training/tul_setup.py::build_tul_runtime` (setting both raises). The per-stream
learned trigger (`TULSlotRegister`: K queries pooling K different vectors from the span's
own prelude states, `W_o` zero-init so step 0 is the ruler), the within-slot loop relation
(`slot_cell_relation`, delivered as `tg_relation`), the cell-level layout in `_tul_core`,
the cost accounting and every register refusal are therefore the register's, unchanged.
`fan_k: 0` builds nothing and the forward is bit-identical.

**Add the three things it did not have**, all in `morph/model/tul_fan.py`:

1. `tul.fan_repel_lambda` x the mean pairwise cosine among a slot's K streams, charged
   after each of the first `tul.fan_repel_passes` passes (default 2). EARLY, because the
   theorem makes the collapse exponential in depth: repelling at pass 1 fights `L^2` and
   at pass 6 fights `L^12`. Read off the trajectory `_tul_core` already builds for the
   other per-pass readers. Every OTHER pass, the seed included, is still measured under
   `no_grad` and reported as `fan/stream_cos_t{t}`.
2. `tul.fan_mix` — `mean` (the control, no parameter) or `softmax` (`TULFanMix`: one
   shared `d_model -> 1` linear scores each stream, a softmax over K mixes them). The
   gate is zero-init, so at step 0 a softmax arm IS the mean arm, bit for bit.
3. The oracle-over-stream, `MORPHTransformer._tul_fan_oracle`, at every val: the coda is
   re-run once per stream through the SAME `prefix_project`, and `fan/oracle_ce` is the
   per-span minimum, token-weighted, against `fan/single_ce` (stream 0) and
   `fan/mixed_ce`.

**Close the width confound by construction.** The fan does NOT take the register's 1:1
cell write. It mixes the K streams to ONE state at the register's mean seam — upstream of
the MUX, the span decoder, SIGReg and the energy, so every reader grades what the coda
gets — and writes that one state through the ordinary single-source `prefix_project`. At
`prefix_k: 4` the fan's coda input has exactly the shape a `prefix_k: 4` ruler's has, so
`prefix_k == slot_cells` is relaxed for a fan model and for a fan model only.

The falsifier is stated in
[`lab/experiments/planned/2026-09-19-lxtul-fan4.md`](../../../../lab/experiments/planned/2026-09-19-lxtul-fan4.md):
if `fan/oracle_ce` does not beat `fan/single_ce` by more than 0.022 nats — the register's
own measured width gain, the only width number this lineage has on disk — the streams are
copies and the width branch closes.

## Alternatives considered

**Reuse `tul.row_contrast_lambda` as the diversity term.** Rejected, and the survey names
it: that term separates a ROW's SLOTS from each other, and its own docstring records that
at `slot_cells > 1` it reads the cells' MEAN. It cannot see the within-slot axis the fan
lives on. Its register pairing (`tul_slot_register_m4_contrast.yaml`) has never run and
would not have measured this.

**Keep the register's 1:1 cell write and add a width control run.** Rejected on cost and
on precedent. It needs a second 5,000-step arm to interpret the first, and the control it
needs (`slot-spandec-strict-pk8`) has been queued since 2026-09-13 and never reached the
card. Mixing to one state removes the confound for free and makes the oracle a
within-model comparison with no packer difference either.

**GRAM's stochastic seed (a learned Gaussian guidance around the slot seed).** Rejected
for this arm, not in general. GRAM's own ablation scores "stochasticity only" 94.88 on
Sudoku and 50.27 on N-Queens, so randomness alone is not the mechanism; and MORPH's
stochastic/one-stream cell is already measured flat (LCTUL's thinker is context-blind, the
past worth 1 % of its flow loss). A stochastic fan is the natural follow-up IF the
deterministic fan's oracle gap is real, and pointless if it is not. The Parallel-TTS
sampling gate on the existing `slot-spandec-strict` checkpoint
([`2026-09-18-sample-oracle-gate.md`](../../../../lab/experiments/planned/2026-09-18-sample-oracle-gate.md))
asks the cheaper version of the same question first.

**Per-pass additive noise on the streams.** Rejected: it is the stochastic axis again, it
fights the slot-loop gain constraint that keeps this recipe from detonating
(`lab/divergence/BREAK-GLASS-IN-CASE-OF-DIVERGENCE-THE-SLOT-LOOP-GAIN-CONSTRAINT.md`), and
an injected perturbation makes "the streams differ" true by construction and therefore
unmeasurable.

**LTF's GFlowNet over the latent trajectory.** Rejected as the wrong size of step. It
replaces the objective, the sampler and the depth policy at once, so a flat result would
name no cause; and its own sampler, free to choose depth under an accuracy reward, chose
1.9 steps on math word problems. If the fan's oracle gap is real, a learned selector over
streams is the next increment and LTF is the increment after that.

**K = 2 instead of 4.** Weighed and declined. PLR's optimum is M = 2 and M > 2 degrades,
and LTC's budget split at B = 8 prefers 2 thoughts x 4 answers over 4 x 2 by 1.97 points,
so K = 4 is outside both measured optima. The honest reason to run 4 anyway is that
MORPH's register arm was 4 and this arm should be ONE factor from it. K = 2 is the cheap
follow-up if P-1 holds.

## Acceptance criteria

1. `tul.fan_k: 0` is bit-identical to the tree before the key — loss, logits, total
   gradient and the state-dict key set, pinned in `tests/test_tul_fan.py` against the
   numbers `tests/test_tul_prefix_source.py` already carries for the strict ruler at
   `prefix_k 4`, measured before this change.
2. The repulsion reads 1.0 for identical streams, exactly 0 for orthogonal ones, carries
   gradient on the first `fan_repel_passes` passes and NONE on the seed or the later
   passes, and ignores pad slots.
3. The softmax mixture equals the mean at equal logits; a one-hot gate reproduces feeding
   that stream alone, checked end to end against the oracle's own single-stream replay.
4. `fan/oracle_ce` equals the token-weighted per-span minimum over the K single-stream
   tables the instrument builds, and sits at or below `fan/single_ce` by construction.
5. Unknown `tul.fan_*` keys raise through `KNOWN_TUL_KEYS`.
6. The experiment's own bar is P-1 in the prereg: a within-model oracle gap above 0.022
   nats at 5,000 steps. Below it, this note moves to `rejected/` with that number on the
   `Status:` line.

## Risks

- **The repulsion is the weakest component in the one paper that measured it.** PLR's own
  ablation calls its KL repulsion its smallest contributor and says too much of it hurts.
  `fan_repel_lambda: 0.1` is a guess, and the arm carries `fan4-norepel` so a flat result
  cannot be blamed on the term without evidence.
- **`oracle_ce <= mixed_ce` is NOT true by construction**, contrary to how the instrument
  is easy to describe: a convex combination of K streams can beat every one of them. Only
  `oracle_ce <= single_ce` is guaranteed. The tests assert the guaranteed one and the
  prereg reads the gap against `single_ce`, not against the mixture.
- **The reader confound (`the-reader-was-the-limit`).** Every oracle number is a property
  of a (stream, reader) PAIR: the same cell measured −4.6413 nats against a frozen coda
  and +0.1768 against one given 10k steps to adapt. Here the coda trains alongside the
  streams, so the reading is "what this coda can use", which is the right question for
  this arm and is not a statement about what the streams contain.
- **The oracle's own selection bias.** Picking the best stream per span after seeing that
  span is optimistic by construction. `fan/oracle_pick0` is the alarm — `torch.min`
  returns the first minimal index, so identical streams drive it to 1.0 — and the survey's
  split-span version (choose on the first half, score on the second) is the honest
  follow-up if P-1 holds.
- **Cost.** The core runs over 4x64 = 256 cells instead of 64; the register measured
  9,582 tok/s against its partner's 11,759. The oracle adds four coda passes per val
  batch. Neither has run on the 5090 at seq 1024: the CPU wire check at seq 256 is the
  only end-to-end evidence, and the runner's 12-step smoke is the real gate.
- **A trace-time branch per realised depth.** The repulsion and the per-pass readings loop
  over the trajectory, whose length is the batch's realised max depth. That is the
  existing `mux_every_pass` pattern and the same recompile surface, but it is a surface.
