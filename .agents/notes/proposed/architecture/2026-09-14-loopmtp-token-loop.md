# Agent Note: LoopMTP on the plain token loop — a horizon-indexed target per iteration

Status: proposed

## Problem

Every per-pass target this tree has built was met in ONE pass. Progressive losses, per-pass
LoRA, an oracle gradient trajectory, staged denoising targets, gradient-conditioned passes
and a token map inside the core all read K3−K6 ≤ 0.002 nats
([`per-pass-targets-are-met-in-one-step`](2026-09-13-parked-loop-directions.md) and the
panels it lists), and the plain loop's own K1−K6 is 0.170 at 20k with pass 1 doing nearly
all of it. The common shape of every one of those failures: each pass was asked for THE SAME
THING, so a map that achieves it once has nothing left to do.

LoopMTP (arXiv 2608.03624, Shomali, Frey, Berghaus, Koehler et al. 2026) asks each pass for
a different thing. Wolfe read it on 2026-09-13 and said it "directly addresses seemingly
everything we need it to". At 260M parameters — MORPH's own scale — it reports 10.5 %
perplexity and +8.08 % relative task accuracy over a param-matched non-looped baseline
(Sec 4.2), and a per-iteration ground-truth rank up to 35.6x better with the signal than
without (Fig 3). That is the first candidate in this arc whose evidence is at our scale.

## Proposal

Two knobs on the PLAIN looped model (`morph/model/transformer.py::_core_region`), both off
by default and bit-identical when off, mapping MORPH's core iteration `t` onto the paper's
loop iteration `t`.

**1. `model.loopmtp_weight` — the soft MTP target (Eq 12-13).** After core iteration `t`
(`t ≥ 2`; `loopmtp_free_first` is the paper's "the first iteration is reserved as an
unconstrained representation"), every position `i` gets a cosine loss to the DETACHED tied
output embedding of the token `t` steps ahead:

    L_align^(t) = mean over valid i of  1 − cos(proj(x_i^(t)), sg[E_{u_{i+t}}])
    L_align     = 1/(T−1) · Σ_{t=2..T} L_align^(t)

`labels[i]` is already `u_{i+1}` in this tree, so `u_{i+t}` is `labels[i + t − 1]`, the same
shift `_mtp_apply` uses. The last `t − 1` positions of a row have no such token and are
masked, which is the paper's `1/(S − t)` normaliser written so it stays right when the row
also carries `-100` padding. The whole term is TRAINING ONLY, like
`core_fixed_point_lambda`, so val CE stays the number every ladder arm is compared on, and
it enters the loss through the existing `_core_aux` mechanism as `loopmtp_weighted`.

**2. `model.core_readout: last|gated` — the aggregator (Eq 9-11).** `gated` makes the coda
read a content-conditional mix of ALL T iterates instead of the last:

    g^(t)      = softplus(W_g x^(t) + β_t · 1_d)
    g̃^(t)      = g^(t) / (Σ_s g^(s) + ε)        elementwise, over ITERATIONS
    z          = Σ_{t=1..T} g̃^(t) ⊙ x^(t)

One `[d, d]` `W_g` shared by every iteration, T scalar biases, both zero-initialised, so at
init `z` is the uniform mean of the T iterates — the paper's "All (uniform)" variant, its
second-best in Fig 4 left. The paper's ponder regulariser (`loopmtp_ponder_weight`, default
0.05, KL of the per-iteration gate mass against uniform) comes with it and is always
COMPUTED on a gated arm so a run has a live gate-collapse instrument, and weighted into the
loss only when its λ is non-zero.

Both knobs require full BPTT (and, until the 2026-09-14 amendment below, a fixed depth)
(`bptt_depth ≥ mean_depth`), and REFUSE at build otherwise, along with a TUL model, SCSE and
a coreless model.

**Arms** (`lab/experiments/planned/2026-09-14-arc-loopmtp-token-loop.md` holds the frozen
predictions): `norm-match-20k-d3-loopmtp` against the existing depth-3 ladder rung, and
`norm-match-20k-d6-loopmtp` against a new `norm-match-20k-d6fixed` control. The Poisson top
rung cannot be the control — the paper's T is a constant.

**Instruments.** `lab/divergence/loopmtp_iteration_probe.py` reads the paper's own three:
the median rank of `u_{i+k}` from iteration `t`'s state (Fig 3), the consecutive- and
first-iterate cosines (Fig 2 bottom), and the per-iteration gate mass (Fig 4 right). It
scores a control arm through `--collect-only`, which turns the per-iteration collection on
without adding a parameter or changing a forward op.

### Deviations from the paper, and why

- **A shared state-side projection (`loopmtp_proj: linear`, the arms' setting).** The paper
  applies no projection: its `x^(t)` is the state that feeds the LM head directly, so a
  cosine against an unembedding row is a comparison inside the head's own space. MORPH puts
  three coda blocks, `lm_mixer` and `final_norm` between the core and the head, so the raw
  core state does not live in that space and the paper-literal form would be a strictly
  harder constraint than the one the paper measured. The projection is ONE shared
  RMSNorm → Linear at identity init (so at init it is a pure rescale and cosine is
  unchanged), never per-iteration — a per-iteration projection could satisfy T targets by
  rotating the read-out instead of by making the iterates differ. `loopmtp_proj: none` keeps
  the paper-literal form available as an ablation.
- **The gate acts per Hyper-Connection stream.** MORPH's carrier is `[B, S, n, C]`. `W_g`
  acts on the last axis and the normalisation runs over iterations at every `(stream,
  channel)`, so the aggregate is a valid carrier the coda consumes unchanged. On a plain
  `[B, S, C]` residual the same code is the paper's form verbatim.
- **`x^(0)` and the iteration index enter MORPH's own way.** The paper concatenates a
  normalised `(t−1)/T` and a re-normalised token embedding and projects. MORPH already
  re-injects the source every iteration through `DiagonalInjection` plus the per-layer
  `ChannelInject` term, and already threads `iter_idx` into the MLP. Those are not replaced.
- **Training only, for both terms.** The paper's total loss is one objective. Here val CE
  must stay comparable to every other arm in the arc, so `L_align` and `L_ponder` are
  computed in `training` only, and `train/loss` subtracts the weighted terms exactly as it
  already does for `mtp_weighted` and `fp_weighted`.
- **λ_align at T = 6 is not a paper value.** Table 2 gives 0.01 / 0.01 / 0.05 / 0.15 for
  T = 3 / 5 / 7 / 9. T = 3 uses the paper's 0.01; T = 6 uses 0.05, the larger of the
  bracketing pair, because under-weighting the signal is the failure that would answer the
  question wrongly. Not swept.
- **Inference cost is unchanged and is not a saving.** Every token still pays T core passes,
  and the aggregator needs all T states, so a gated model cannot early-exit or stream the
  loop. LoopMTP's only route to less compute is that a SMALLER T becomes good enough; the
  mechanism itself buys nothing at inference and adds two `[d, d]` matmuls per iteration.


### Amendment (2026-09-14): the Poisson draw

Two corrections after the first arms ran. (1) `mean_depth == max_depth` never was a fixed
depth on this tree: `_sample_depths` is Poisson(mean) clamped to [1, max], so the
"fixed-depth" arms drew partial depths and the first LoopMTP smoke died in the compile
warmup. `model.depth_fixed: true` (26ef770) is the real constant-T knob and the three
fixed arms set it. (2) At Wolfe's direction LoopMTP now runs under the recipe's own draw
(a5a1265): the active set is a depth-sorted prefix at every iteration, so the gate, the
alignment maps and the ponder term read prefix states and each row gates over and is
aligned on exactly its own realised passes, with `beta` spanning `max_depth`. At a fixed
depth every prefix is the whole batch and the arithmetic is unchanged per row. The arm
`norm-match-20k-loopmtp` pairs with the Poisson top rung directly (predictions P-10 to
P-15 in the prereg), which is the arm the recipe would ship; the "Poisson rung as the
depth-6 control" objection below applies to the FIXED arms only.

## Alternatives considered

- **The amortized slot-loop port of the same idea** (`2026-09-14-arc-horizon-passes`, in
  build in parallel): pass `t` predicts the span `t` spans ahead. That is the design that
  preserves TUL's intent — think once per span, decode cheaply per token — and it is the one
  that would actually reduce compute. It is also the one running on the mechanism that has
  been null eleven-plus times, and its horizons run out of row: at ~50 slots per 1024 tokens
  a `t`-ahead span for large `t` leaves the row. Building the token-loop version FIRST tests
  the paper's mechanism where the paper measured it, on a lineage with a live 0.170-nat
  K-curve, so a null on the slot port can be read against a known-good control instead of
  being the twelfth undiagnosable null. Not exclusive: both are being built.
- **Hard-CE full-vocabulary MTP heads** (Gloeckle et al. 2024), which this tree already has
  as `model.mtp_heads` / `notul_mtp4` (ARC E8). That arm puts four `[d, d]` heads on the
  CODA read-out, all reading the SAME final state, each with a full-vocabulary CE at weight
  1.0. Three differences: it supervises the OUTPUT, not the iterations, so it cannot
  differentiate passes by construction; it projects to the vocabulary, which the LoopMTP
  authors say "hurt performance at a small scale" (Sec 4.3) and which is the expensive
  formulation Wolfe's "MTP needs 3B+" belief is actually about; and its heads can satisfy a
  lookahead target by rotating the read-out, which a shared cosine target against frozen
  embeddings cannot. `notul_mtp4` is kept and untouched — the knobs are independent and can
  run together — but it is not this arm and its result does not predict this one. It also
  already cost a measured lesson: at weight 1.0 MTP heads opened the first-iteration SCALE
  growth mode under a passing hinge (E13/E14), which is one reason λ_align here is ≤ 0.05.
- **The Poisson rung as the depth-6 control.** Cheaper (it exists) and wrong: the aggregator
  gates exactly T states and each iteration owns a horizon, so rows with different draws
  would carry different target sets and different gate supports. The build refuses it, and
  `norm-match-20k-d6fixed` is the extra run that buys the comparison.
- **Running the gate and the target as separate arms (the 2×2).** Cleaner attribution, two
  more 20k runs. Rejected for now: the paper ships them together and the first question is
  whether the combination reproduces at all. The knobs factor, and the "target only" cell
  leaves the forward bit-identical to its control, so the factorial is one config each if
  the panel reads positive. Named as a confound in the prereg.
- **Aligning the CODA read-out per iteration instead of the core state** (run the coda T
  times). It would remove the projection deviation by construction and it costs T coda
  passes per step, roughly +50 % forward on a 3:6:3 model, on an arm whose whole purpose is
  to make the model CHEAPER. Rejected on cost.
- **A per-iteration projection instead of a shared one.** Lets each iteration rotate into its
  own target space, which is exactly the escape hatch that would let the iterates stay
  identical while the loss falls. Rejected: it would make P-5 and P-6 of the prereg
  unfalsifiable.

## Acceptance criteria

- `tests/test_loopmtp.py` green, including: knobs-off is bit-identical to the base commit
  `fd2daae` measured by running the pre-change `morph/` tree in a second process on the same
  CPU fixture; knobs-on shares every base weight with the same-seed control; the alignment
  term is positive and enters the loss; no gradient reaches the tied embedding table through
  the alignment TARGET while the LM-head path still trains it; iteration `t` reads
  `labels[i + t − 1]` and nothing else; the row end is masked; the gate normalises over
  iterations and the aggregate is the sum over all T iterates; every refusal fires.
- Eight source-level sabotages each caught by a named test (dropped `sg[E]`, an off-by-one
  horizon, ignoring `free_first`, normalising the gate over the feature axis, aggregating
  only the last iterate, dropping the row-end mask, collecting the pre-step state, dropping
  the fixed-depth refusal).
- The three configs compose through Hydra and `build_morph_config`.
- The prereg's P-1 decides the ship depth; P-5 and P-6 decide whether the MECHANISM
  reproduces here, independently of whether it pays.
- Promotion to `implemented/` requires a filed result under
  `lab/experiments/successes/` or `failures/`, not a green test suite.

## Risks

- **The K-curve is confounded on a gated arm.** Forcing the depth narrows the aggregate as
  well as shortening the loop, so K1−KT will rise for a reason that is not depth earning.
  The prereg says so and leans on the iteration probe instead; a reader who scores a gated
  arm's K1−KT against a "last" arm's without reading that paragraph will draw a wrong
  conclusion. `core_depth_sweep.py`'s docstring carries the warning.
- **Budget.** The paper's gains are at 6.8B tokens with Muon+AdamW at LR 1.9e-3; these arms
  see ~123M tokens on AdEMAMix at 1e-4. A null here is evidence about this budget, not about
  the method, and must be filed that way.
- **An auxiliary term on the core under ternary.** MTP heads at weight 1.0 already opened a
  scale mode on the slot loop (E13/E14). λ_align here is 0.01–0.05 and the arms run the
  1000-step ramp, but the detonation tripwire (`preclip/total > 1e4` at step ≥ 200) is the
  only guard and it has never been exercised against this term.
- **Memory.** The gate keeps two carrier-shaped tensors per iteration. Arithmetic says
  ~0.6 GB on top of a ~10 GB control at batch 6 / seq 1024; it is arithmetic, not a
  measurement, and the 5090 has about 1 GB of slack once the desktop is up.
- **Nothing here has run on a GPU.** The configs compose, the model builds, a forward and a
  backward run on CPU. Every rate, every memory figure and both instruments' behaviour on a
  real checkpoint are unverified.
