# Planned: LoopMTP on the token loop — does a horizon-indexed target make the passes differ

Status: planned

Date: 2026-09-14 (frozen before any GPU step of the three new arms; the depth-3 rung and
the Poisson depth-6 rung already exist). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Parent ladder:
[`2026-09-13-arc-depth-ladder-ship.md`](2026-09-13-arc-depth-ladder-ship.md). Design note:
[`2026-09-14-loopmtp-token-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-14-loopmtp-token-loop.md).
Paper read: [`D_objective.md` §1](../../../docs/references/looping-depth/2026-09-13-lit-mining/D_objective.md).

## Question

Every per-pass target this tree has tried was met in ONE pass: progressive losses, per-pass
LoRA, an oracle gradient trajectory, staged denoising targets, gradient-conditioned passes
and a token map inside the core all read K3−K6 ≤ 0.002 nats on the slot loop, and the plain
loop's own K1−K6 sits at 0.170 at 20k with pass 1 doing nearly all of it. Every one of those
targets asked each pass for THE SAME THING. LoopMTP (arXiv 2608.03624, Shomali et al. 2026)
asks each pass for a DIFFERENT thing: iteration `t` is aligned to the output embedding of
the token `t` steps ahead (Eq 13), iteration 1 is left unsupervised on purpose, and the
read-out is a learned content-conditional mix of all T iterates (Eq 9-11) rather than the
last one. At 260M parameters — MORPH's own scale — they report 10.5 % perplexity and +8.08 %
relative task accuracy over a param-matched non-looped baseline, and a per-iteration
ground-truth rank up to 35.6x better with the signal than without (Sec 4.2, Fig 3).

The ship question, in Wolfe's words from 2026-09-13: *"I would just be happy to have
something that reduces compute enough that I can ship."* So: **does horizon guidance let
depth 3 match or beat plain depth 6 at 20,000 steps?** If it does, the ship depth is 3 at
26 block-passes per token instead of 44.

## What is already known

- The plain norm-match model at Poisson depth 6 reads forced-depth CE 3.6217 / 3.4990 /
  3.4672 / 3.4516 at eval depths 1 / 2 / 3 / 6 at 20k, K1−K6 0.1702, rising from 0.136 at
  5k (`results/2026-09-09-norm-match-recipe-reads/sweep_norm-match-20k_20000.json`).
- Training depth moves CE and eval depth does not: a depth-`d`-TRAINED rung sits well under
  the depth-6 model's forced-depth-`d` number (memory
  `training-depth-moves-ce-eval-depth-does-not`). The d1/d2/d3 ladder measures exactly that
  and is running under its own prereg.
- Rates on this lineage: depth 6 Poisson 9,901 tok/s, depth 1 27,610 tok/s (step-200 lines
  in `arc/norm-match-20k/run.log`, `arc/plain-depth1/run.log`). Peak memory 10.08 GB and
  9.04 GB.
- The existing `notul_mtp4` arm (ARC E8) is the OTHER multi-token idea and it is not this
  one: four full-vocabulary CE heads on the CODA read-out, all reading the same final
  state, equal weight 1.0. It tests whether a lookahead target on the OUTPUT helps; LoopMTP
  puts a different lookahead on each ITERATION of the loop, by cosine against a frozen
  embedding rather than by a vocabulary projection — the expensive formulation is the one
  the paper says hurts at small scale (Sec 4.3).

## Method

Three new arms on the `notul_norm_match_20k` lineage (Parcae noise entry, `norm_match`
ternary, seq 1024, batch 6, 1,000-step LR ramp, 20,000 steps, checkpoints every 5,000,
retention off, prune/carve/route off, `core_fixed_point_lambda` 0.0). All four numbers below
came from composing each config through Hydra and `build_morph_config`, not from reading the
YAML.

| arm | config | depth (mean=max=bptt) | `core_readout` | λ_align | λ_ponder | control |
| --- | --- | --- | --- | --- | --- | --- |
| `norm-match-20k-d3-loopmtp` | `notul_norm_match_20k_d3_loopmtp` | 3 | gated | 0.01 | 0.05 | `norm-match-20k-d3` (exists) |
| `norm-match-20k-d6fixed` | `notul_norm_match_20k_d6fixed` | 6 | last | 0 | 0 | — (it IS a control) |
| `norm-match-20k-d6-loopmtp` | `notul_norm_match_20k_d6_loopmtp` | 6 | gated | 0.05 | 0.05 | `norm-match-20k-d6fixed` |

`norm-match-20k-d6fixed` exists because the Poisson top rung CANNOT be a LoopMTP control:
the paper's T is a constant, every iteration carries its own horizon target and the
aggregator gates exactly T states, so a per-sample Poisson draw would give different rows
different target sets. The build REFUSES `mean_depth != max_depth` with either knob on.

λ_align 0.01 is the paper's own T=3 value (Table 2: 0.01 / 0.01 / 0.05 / 0.15 for
T = 3 / 5 / 7 / 9). The paper gives no T=6; 0.05 is a judgment call, recorded in the config
header and in "Not verified" below.

**Read-outs.** Each checkpoint (5k, 10k, 15k, 20k) gets the runner's plain readout,
`core_depth_sweep.py --depths 0,1,2,3,6,9,12,16 --rows 480`. Between-arm CE is token-paired
on the 480 rows (`span_budget_profile.py --full A.npz --span B.npz`). The 20k checkpoint
of every LoopMTP arm and of `norm-match-20k-d6fixed` also gets
`lab/divergence/loopmtp_iteration_probe.py --rows 48`, which reads the paper's own three
instruments: the median rank of `u_{i+k}` from iteration `t`'s state (Fig 3, both through
the model's LM head and under the alignment loss's own cosine score), the consecutive- and
first-iterate cosines (Fig 2 bottom), and the per-iteration gate mass (Fig 4 right). The
control is scored with `--collect-only`, which turns the per-iteration collection on without
adding a parameter or changing a forward op.

**What the K-curve means on a gated arm.** Forcing the depth changes what the aggregator
sees as well as how many iterations ran: at forced depth `d` the gate normalises over the
`d` iterates that exist, at `d = 0` the aggregate is the entry carrier, and above the
trained T iteration `t > T` reuses `beta_T`. So K1−KT on a gated arm mixes two effects and
is NOT the same quantity as K1−KT on a "last" arm. That is why the iteration probe is the
decisive instrument here and the K-curve is the ship number.


### Method amendment (2026-09-14, after the d3-loopmtp smoke failed at c90b9f1)

The tree's `_sample_depths` is Poisson(mean_depth) clamped to [1, max_depth], so
`mean_depth == max_depth` never was a constant T: a mean-6 / max-6 model runs about a
third of its rows at depth 1-5, and LoopMTP's constant-T guard fired in the compile
warmup (and would have fired on the first real step). Fix: a real `model.depth_fixed`
knob (every row runs max_depth iterations), required by the LoopMTP build check, honoured
by the compile warmup, set on `_d3_loopmtp`, `_d6fixed` and `_d6_loopmtp`. Because the
ladder rung `notul_norm_match_20k_d3` keeps the clamped draw, a new one-factor control
`notul_norm_match_20k_d3fixed` (depth_fixed, LoopMTP off) is added and queued; the
LoopMTP delta at depth 3 is read against IT, and P-1 (the ship question) stays against
the Poisson-6 top rung as written. Reason: the arms cannot start otherwise; no prediction
is changed.

## Predictions (frozen)

- **P-1 (the ship question: d3 + LoopMTP vs plain depth 6).** CE(d3-loopmtp@20k) −
  CE(d6fixed@20k), token-paired on 480 rows, is **≤ +0.02 nats** (d3 with LoopMTP matches
  plain depth 6 within the ladder's own ship threshold). **30 %.** Reasoning: the ladder's
  own P-3 already gives d3 a 60 % chance of landing within 0.02 WITHOUT LoopMTP, so this is
  mostly asking whether LoopMTP costs nothing; the alignment term takes capacity away from
  the next-token objective, and every horizon-indexed target this tree has tried was met in
  one pass. Residual: 50 % in (+0.02, +0.08], 20 % worse than +0.08.
- **P-2 (a, d3-loopmtp vs its OWN control d3).** CE(d3-loopmtp@20k) − CE(d3@20k),
  token-paired, is **negative** (LoopMTP helps at depth 3). **40 %.** Reasoning: the paper's
  gain is real at 260M but at 6.8B tokens; our 20k × 6 × 1024 is ~123M tokens, ~55x less, and
  an auxiliary term usually costs CE before it pays. Residual: 45 % in [0, +0.03], 15 % worse
  than +0.03.
- **P-3 (b, d6-loopmtp vs d6fixed).** CE(d6-loopmtp@20k) − CE(d6fixed@20k), token-paired, is
  **negative**. **35 %.** Lower than P-2 because λ_align is 5x larger here and is not a
  paper-swept value at this T.
- **P-4 (c, the K-curve).** `norm-match-20k-d6-loopmtp` reads **K1−K6 ≥ 0.25** at 20k, i.e.
  at least 1.5x the plain Poisson model's 0.170. **45 %.** And `norm-match-20k-d3-loopmtp`
  reads **K1−K3 ≥ 0.10** (the plain depth-6 model's forced K1−K3 is 0.154, a depth-3-TRAINED
  rung's is unknown). **40 %.** Reasoning: a gated read-out mechanically raises K1−KT
  because depth 1 also narrows the aggregate — this prediction is partly about that
  artefact, which is why P-5 and P-6 are the ones that test the mechanism. Residual on the
  first: 35 % in [0.17, 0.25), 20 % below 0.17.
- **P-5 (d, the horizon probe — the mechanism).** On `norm-match-20k-d6-loopmtp` at 20k, for
  every `t` in 2..6, the median rank of `u_{i+t}` read from iteration `t` is **lower
  (better) than** the median rank of the same `u_{i+t}` read from iteration 1, under the
  alignment loss's own cosine score. **80 %.** Under the model's LM head read-out
  (`rank_head`), the same statement for at least 4 of the 5 iterations: **55 %.** Reasoning:
  the first is close to "the auxiliary loss trained"; the second asks whether the horizon
  information is legible to the model's own head, which the loss never demanded. On
  `norm-match-20k-d6fixed` (no alignment term) the same comparison holds for **at most 1**
  of the 5 iterations: **70 %.**
- **P-6 (e, consecutive cosine — did the passes separate).** `cos(x^(t), x^(t+1))` averaged
  over `t` is **at least 0.05 lower** on `norm-match-20k-d6-loopmtp` than on
  `norm-match-20k-d6fixed`. **60 %.** Reasoning: this is the paper's Fig-2-bottom signature
  and the closest thing to this tree's own "consecutive pass updates cancel at cosine −0.2
  to −0.6" finding. Residual: 40 % that the two arms' bands overlap, which would say the
  target changed what the passes ARE GRADED ON without changing what they DO.
- **P-7 (f, rate).** Step-200 tok/s of `norm-match-20k-d6-loopmtp` is **≥ 0.90x**
  `norm-match-20k-d6fixed`'s. **70 %.** Arithmetic: the alignment path adds T `[d, d]`
  projections and the gate adds T more, ~0.5 TFLOP against the step's ~30 TFLOP
  forward+backward, so under 2 % of the FLOPs; the rest would be memory traffic on the T
  retained gate tensors. `norm-match-20k-d6fixed` itself is **≥ 9,000 tok/s** (the Poisson
  rung reads 9,901 and a fixed depth does the same mean work with a simpler active set):
  **75 %.** `norm-match-20k-d3-loopmtp` is **≥ 12,000 tok/s**: **65 %** (the ladder predicts
  d3 ≥ 13,000 and this arm pays the LoopMTP overhead on top).
- **P-8 (g, memory).** Peak allocated of `norm-match-20k-d6-loopmtp` is **under 12.5 GB** at
  batch 6 / seq 1024. **80 %.** Arithmetic: the control peaks near 10 GB; the gate keeps
  2 × T carrier-shaped bf16 tensors, 6 × 6 × 1024 × 4 × 1024 × 2 B ≈ 0.3 GB each, so ~0.6 GB
  on top, and the T iterates themselves are already retained under full BPTT. All three arms
  complete 20,000 steps without an OOM: **85 %.**
- **P-9 (the gate does not collapse).** `loss/loopmtp_ponder` on both LoopMTP arms stays
  **below 0.20** for the whole run (uniform is 0, a one-hot gate on T=6 is log 6 = 1.79).
  **70 %.** The ponder term at 0.05 is pulling against collapse by construction; this
  predicts that pull is enough.

## Binding

If **P-1 holds**, depth 3 with LoopMTP is the ship depth and the next run is the
ternary + MORTAR + route conjunction at that depth. If P-1 fails but **P-2 or P-3 holds**,
LoopMTP is a free improvement at whatever depth the ladder picks and goes into the recipe
without changing the depth decision. If P-1, P-2 and P-3 all fail but **P-5 and P-6 hold**,
the mechanism works and the objective does not: the passes differ and the difference is
worth nothing on next-token CE, which is the strongest version of "the loop's geometry and
next-token prediction are disjoint" this tree will have measured, and the next build is
item 3 of
[`2026-09-13-parked-loop-directions.md`](../../../.agents/notes/proposed/architecture/2026-09-13-parked-loop-directions.md)
(full-bandwidth latent feedback). If **P-5 fails** as well, LoopMTP does not reproduce here
at all and the finding is about scale or data, not about the design — record it and move to
the parked menu.

## Not verified before launch

- **No GPU step of any of the three arms exists.** Every rate and memory number above is
  arithmetic, or is a different arm's measured log. The configs compose through Hydra and
  the model builds and runs a forward and a backward on CPU; nothing more.
- **The two mechanisms are confounded.** Both LoopMTP arms turn the gated aggregator AND the
  alignment target on together, which is how the paper ships it. If the panel reads positive
  we will not know which half did it. The knobs factor (`core_readout: gated` with
  `loopmtp_weight: 0`, and `loopmtp_weight > 0` with `core_readout: last`), and the second
  of those leaves the forward bit-identical to its control, so the factorial is one config
  each — it is not queued.
- **λ_align is not swept.** One value per arm. 0.01 at T=3 is the paper's; 0.05 at T=6 is
  not a paper value at that T (the paper gives 0.01 at T=5 and 0.05 at T=7) and was chosen
  as the larger of the bracket.
- **λ_ponder 0.05 is the paper's default and is never ablated here.** It is on in both
  LoopMTP arms and in neither control.
- **The state-side projection is a deviation from the paper** (see the design note). The
  paper compares the raw iterate to an unembedding row because its iterate feeds the LM head
  directly; MORPH has three coda blocks, `lm_mixer` and `final_norm` in between. The arms
  run `loopmtp_proj: linear`, one shared identity-init RMSNorm → Linear. `loopmtp_proj:
  none` is the paper-literal ablation and is not queued.
- **`core_depth_sweep.py` has not been run end to end on a LoopMTP checkpoint** — that needs
  a GPU and the validation shard. What IS tested on CPU is the lever it uses
  (`model.cfg.mean_depth = d`, including the `d = 0` rung) on a gated model, and that the
  gate normalises over the available iterations at every forced depth.
- **`loopmtp_iteration_probe.py` has not been run on a real checkpoint.** Its rank helper is
  hand-checked on CPU (a score matrix whose argmax is the target gives rank 1; whose
  argmin is the target gives rank V) and its collection path is exercised on a tiny model;
  the loader, the autocast path and the memory at seq 1024 are unverified.
- **Seed.** The lineage composes `training.seed: 0`. The ladder prereg says "seed 1"; that
  claim was not reproduced here and the composed value is what will run unless the runner
  overrides it. MORPH runs decorrelate in ~11 steps and n=1 comparisons carry a 6.5 % median
  spread, so a 0.02 threshold is inside that spread's reach.
- **20k is one horizon and seq 1024 is one length.** Nothing here says the ordering holds at
  100k or at seq 4096, and the `base.yaml` conjunction (TST × prune/carve/route) is
  untouched by these arms.
- **The paper's own gains are at 6.8B tokens with Muon+AdamW at LR 1.9e-3.** These arms see
  ~123M tokens on AdEMAMix at 1e-4. A null here is evidence about this budget, not about the
  method.

## Results

(to be filled after the runs; predictions above are frozen)
