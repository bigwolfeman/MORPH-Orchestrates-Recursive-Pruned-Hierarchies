# Planned: loop contribution on Huginn-3.5B (a recurrent-depth model trained to use depth)

Status: planned
Date: 2026-09-04 (frozen before the sweep; eval only; Wolfe's call: "we should take huginn
3.5b and test loop contribution on it")

## Question

Every stable MORPH loop, token or slot, earns nothing past iteration 3 on OpenWebText
(arc branch (a): `failures/2026-09-04-arc-e1-*`, `-e2-*`), and its earning is a flat
refinement of the predictable part of the loss (`failures/2026-09-04-arc-e0-*`). Is that
a property of web text, or of how MORPH is trained (mean depth 6, max 8, 328M positions,
ternary, the gain hinge)? Huginn-0125 (Geiping et al. 2025, arXiv 2502.05171) is the
same shape at scale — 2 prelude + 4 recurrent + 2 coda layers, 3.5B parameters, trained
on 800B tokens with the recurrence sampled r ~ U(4, 64) (mean ~32, truncated backprop
over the last 8) — and it reports per-step gains on reasoning benchmarks up to 64 steps
while the paper's language-modeling loss curve saturates after a handful of steps. This
measures, on the SAME rows and with the SAME instruments as our arms, how much a
recurrent-depth model that was trained to use depth earns from extra iterations on
web text, and where.

## Method

`lab/huginn/huginn_depth_sweep.py` (new): load `tomg-group-umd/huginn-0125` (bf16,
`trust_remote_code`) on the 5090; 480 rows of OpenWebText validation text, 1024 tokens
each under Huginn's own tokenizer (the same source documents the MORPH sweeps use, drawn
in the same order; the tokenization differs, so rows are matched by document, not by
token); forward at `num_steps` ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64} on identical
batches; per-token CE from the logits; per-row and per-offset-in-span sums with the arc's
`_earning.EarningProfile` (spans cut by MORPH's boundary rule rebuilt on Huginn's
tokenizer with `boundary_lut_from_tokenizer`); paired-bootstrap CIs over rows for
K1−K3, K3−K6, K6−K16, K16−K32, K32−K64; `score_arc_e0.py` for the offset profile and
the row-difficulty correlation at (1, 6) and (3, 32). Output
`lab/experiments/results/2026-09-04-huginn-loop-contribution/`. Eval only; no training.

Comparison rulers (ours, same instruments, 480 rows): notul-20k-wu K1−K6 0.041, K3−K6
0.002 (20k), a2 K1−K6 0.104; every slot arm K3−K6 ≤ 0.0014.

## Predictions (frozen)

- **H1.** Huginn's token CE on web text falls monotonically with `num_steps` from 1 to
  32 (each successive pair's paired CI above 0): **80%**.
- **H2.** Huginn's K3−K6 > 0.05 nats: **70%** (3 steps is far below its trained
  regime; MORPH's is 0.002).
- **H3.** Huginn's K16−K32 > 0.01 nats: **55%**; K32−K64 > 0.005: **35%** (saturation
  near the training mean; the paper's LM curve flattens early).
- **H4.** Huginn's earning profile over offset-in-span is flat to ±15 % of its mean at
  (1, 6): **55%**; the span's first token earns the least, as on every MORPH model:
  **50%**.
- **H5.** Spearman(row CE₁, row earning K1−K6) ≤ +0.10: **50%** (earning does not
  concentrate on hard rows).
- **H6.** At its trained depth, Huginn's CE on these rows is below 3.0 nats: **60%**
  (a 3.5B model at 800B tokens on web text; MORPH's plain 20k model reads 3.49 at
  depth 1, 3.45 at 6).

## Decision rule (binding)

- H2 TRUE ⇒ depth earning past iteration 3 on web text EXISTS for a model trained to use
  it: the arc's closing option (i), "data where depth pays", is downgraded and the
  training regime (deep recurrence at train time, long training) becomes the candidate
  cause of MORPH's flat loops. The next MORPH prereg trains the plain loop at a deep
  recurrence draw (mean 16+) with truncated backprop, under the abort rule.
- H2 FALSE and H1 FALSE ⇒ web text is depth-flat for a 3.5B model trained on it too;
  the closing rule's option (i) stands and the loop question moves off web text.
- H4/H5 read the SHAPE either way: a flat profile on Huginn says the flat shape is a
  property of next-token prediction on web text, not of MORPH.

## Not verified before launch

Huginn's remote code under transformers 5.15 (written for 4.x; the load is the first
check); its `num_steps` forward on a labelled batch; the boundary rule rebuilt on a
65k-vocab tokenizer; memory at 64 steps and batch 3 × 1024 (the recurrent block's
activations are not checkpointed at eval).
