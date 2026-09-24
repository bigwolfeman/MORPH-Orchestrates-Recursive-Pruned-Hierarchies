# Planned: LXTUL-GK, the stochastic slot loop trained on K prior rollouts (multi-sample bound)

Status: failure

Date: 2026-09-23 13:27 (frozen before any build or GPU step). Parent:
[`2026-09-23-lxtul-g-panel.md`](2026-09-23-lxtul-g-panel.md). Note:
[`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md).
Wolfe's go: 2026-09-23 ("Should I build the multi-sample arm ... Answer: yes").

## Question

lxtul-g trained every pass on a posterior step that sees the next span and deployed the
prior: the prior's rollouts are out of distribution for the coda (5k prior-sample token CE
5.0956 at depth 6 against the ruler's 4.3474; span-decoder CE rising with depth). b1 (beta
1) collapsed the posterior (KL 0.16 nats per slot at step 2100). If instead the model
trains on the deployed object, K prior rollouts per row read by the per-token Bayesian
weighting, with the loss the multi-sample bound

    L = - sum over spans s of log (1/K) sum_k exp( sum_{j in s} log p(tok_j | z_k) ),

does the stochastic loop become useful to the coda, does width (K) earn, and does depth
earn as an outcome?

## Hypothesis

Training on prior rollouts removes the exposure gap by construction: the coda only ever
sees what the loop produces. The soft per-span credit (the reweighting's gradient) pays a
rollout for being the one that explains the span, so the loop is paid to propose distinct
continuations: exploration from the objective, with no diversity term and no per-pass job.
No posterior and no KL in the first build: with the prior as the proposal the bound needs
neither.

## Arms

On `tul_slot_spandec_strict` (seq 1024, batch 6, seed 1, norm_match, ramp 1000, fixed
point 1.0 on the deterministic part), 5000 steps, recon runner, at the build's commit.

| arm | config | one factor |
|---|---|---|
| `lxtul-gk4` | `tul_slot_spandec_strict_gk4.yaml` | the ruler + the gram step (prior only) trained on the K = 4 multi-sample bound |
| `lxtul-gk1` | `tul_slot_spandec_strict_gk1.yaml` | lxtul-gk4 with K = 1 (one prior rollout: a noisy loop trained consistently; the width control) |

Partners with sweeps on disk: the ruler @5000, lxtul-g @5000, fan4-all @5000.

## Instruments

`lab/divergence/lxtul_g_probe.py` at 5000: `ce_prior@1`, `ce_iw@N` for N in {1, 4, 16},
the identity null, `ce_zero`, sigma/r; its depth corner (`ce_prior@1` and `ce_iw@4` at
forced depths 1 and 6). The runner's sweep (token K-curve on one seeded prior sample),
worth profile, state probe. Paired CE against the ruler on identical tokens.

## Predictions (frozen)

- **P-1 (the gap is gone).** lxtul-gk4 `ce_iw@4` better than lxtul-g's `ce_prior@1`
  by more than 0.5 nats: **85 %.**
- **P-2 (no price).** lxtul-gk4 `ce_iw@4` paired against the ruler at depth 6 within
  +0.020: **45 %**; better than the ruler: **25 %.**
- **P-3 (width earns).** lxtul-gk4 `ce_prior@1 - ce_iw@4` above 0.010: **40 %**; and
  larger than lxtul-gk1's same gap: **40 %.**
- **P-4 (the noise survives).** lxtul-gk4 prior sigma/r at 5000 above 0.02: **55 %.**
- **P-5 (depth as an outcome).** lxtul-gk4 `ce_iw@4` at forced depth 1 minus at depth 6
  above +0.005: **30 %**; token K1-K6 on the seeded single prior sample above +0.005:
  **20 %.**
- **P-6 (training width beats K = 1).** lxtul-gk4 `ce_iw@4` better than lxtul-gk1's
  `ce_iw@4` (paired): **55 %.**
- **P-7 (cost).** lxtul-gk4 tok/s at step 200 at or above 0.5x the ruler's (~11.7k):
  **70 %.**

## Verdict rules

No clause on a cosine. P-1 failing means the build is wrong, not the idea: check the
loss against the probe's `ce_iw@4` on the same rows first. P-3 and P-6 holding with P-5
failing files as: width earns, depth does not emerge. P-4 failing (sigma to the floor)
with P-3 failing files as: the objective prefers a deterministic loop; exploration is not
paid on web text at this scale.

## Method

Opus builder on a worktree at HEAD; review, gate tests and sabotages by me before merge;
then the runner. Artifacts under `../results/2026-09-23-lxtul-gk/`.

## Method amendment (2026-09-23 16:03, after the gk4 smoke failed, before any gk4 step)

Predictions unchanged. What happened and what changed:

- **Build and review.** Merged as 1416d4d. My review added two tests the builder's suite
  lacked (the loop's depth table is the base draw tiled rollout-major; identical rollouts
  give the K = 1 gradient on every parameter). Two of my five sabotages (depth
  `repeat_interleave`, the front fed by rollout 0 only) passed the builder's suite and fail
  only these tests.
- **The gk4 smoke ran out of memory** (runner 15:46:13, the compile warmup's first backward,
  24.6 GB in the process). The runner skipped it and started gk1 at 1416d4d, so **gk1 runs
  first**. A CUDA allocator trace of one training step on the Spark put the K = 4 step at
  26.64 GB against 12.68 GB at K = 1. The largest item was a pre-existing waste in
  `TULSlots.prefix_project`: its broadcast matmul expanded `W_prefix` to `[B, S, K, C, C]`
  and saved it for the backward (1.5 GB at K = 1, 6.0 GB at K = 4). Fixed in 911ef4e for
  every arm (forward bit-identical, gradient summation order only). The span decoder on
  the K rollouts' exit states was the next item (~4.4 GB); its decode is checkpointed at
  K > 1 in 43bb234. The K = 4 step now traces at 15.17 GB.
- 911ef4e also fixes a regression 1416d4d put on master: the GK keywords `n_rep` and
  `checkpoint_blocks` went to every forward and broke wrappers with the old signature,
  among them the instrument `lab/divergence/slot_path_worth.py::token_tax`.
- **gk4 is re-queued at 43bb234**, after gk1. The two arms therefore run at different
  commits. The K = 1 arithmetic differs between them only in the W_prefix gradient's fp32
  summation order (1e-11 to 2e-8 relative on the pinned fixtures).
- **P-7 now carries the recompute.** The coda blocks and the span decode are recomputed in
  the backward at K > 1. P-7 is scored on the tok/s the runner logs at step 200, as frozen.

## Method amendment (2026-09-23 18:59, after the first gk4 run, before its readouts are scored)

Predictions unchanged. **The first gk4 run is confounded by a build defect.** The build
claimed the K rollouts of a row differ in their Gaussian steps alone. They did not:
`training.dropout` 0.1 places `nn.Dropout` after every MLP and in every block, and on the
expanded batch each rollout row drew its own mask in the core and the coda. The bound
could therefore earn width from dropout, a randomness absent at eval. The run's training
log shows the signature: prior sigma/r fell to 2.8e-4 (gk1: 3.1e-4) while
`tul/gk_width_gain` rose from 0.0008 at step 1340 to 0.0065 at 5000. Every GK test ran at
model dropout 0.0, so no test could see it.

Fixed in 7d44ed7: the core and coda dropout of a K > 1 GK model draws one mask per base
row and shares it across the rollouts (`morph/model/rollout_dropout.py`, with a test that
the K rollouts are bit-identical under dropout 0.3 when their noise seeds match).

- **Arm `lxtul-gk4-shared`**: `tul_slot_spandec_strict_gk4.yaml` at 7d44ed7, queued now.
  P-1..P-7 are scored on this arm.
- **The first run (`lxtul-gk4`, 43bb234)** is kept and filed as confounded. Its readings
  are reported beside the rerun's, never as the P-clauses' evidence.
- **gk1 is unaffected**: at K = 1 there is one rollout per row and nothing to share.

## Results (filed 2026-09-23 21:24)

Artifacts: [`../results/2026-09-23-lxtul-gk/`](../results/2026-09-23-lxtul-gk/) (sweeps at 2500
and 5000, worth, slot state, the probe JSONs, `paired_5000.json`, run logs). Probe:
`lab/divergence/lxtul_g_probe.py` on the Spark (65db1f0), 192 validation rows (200,719
tokens). Sweeps: the runner's, 480 rows (501,106 tokens), one seeded prior sample. Paired
CIs bootstrap over 1,024-token blocks (`lab/divergence/sweep_score.paired`). The ruler is
`slot-spandec-strict` @5000 at depth 6 (sweep 4.3474, K1−K6 +0.0016).

Training (final val, tok/s at step 200; ruler 11,759 tok/s):

| arm | commit | final val | tok/s | x ruler |
|---|---|---|---|---|
| lxtul-gk1 | 1416d4d | 4.4238 | 10,629 | 0.904 |
| lxtul-gk4 (confounded) | 43bb234 | 4.4277 | 4,999 | 0.425 |
| lxtul-gk4-shared | 7d44ed7 | 4.4266 | 5,594 | 0.476 |

The eval probe (dropout off):

| arm | ce_prior@1 | ce_iw@4 | ce_iw@16 | ce_zero | width gain @4 | worth(zero) | prior sigma/r |
|---|---|---|---|---|---|---|---|
| lxtul-gk1 | 4.2661 | 4.2660 | 4.2660 | 4.4494 | 0.00007 | 0.1833 | 0.00034 |
| lxtul-gk4 (confounded) | 4.2616 | 4.2616 | 4.2616 | 4.4507 | 0.00006 | 0.1891 | 0.00029 |
| lxtul-gk4-shared | 4.2578 | 4.2577 | 4.2577 | 4.4469 | 0.00005 | 0.1892 | 0.00026 |

The identity null (`ce_iw_identity@4`) equals `ce_prior@1` to 4 decimals on every arm: the
four samples are the same sample.

Paired readings on the probe's tokens:

| reading | point | 95 % CI |
|---|---|---|
| gk4-shared ce_iw@4 − lxtul-g ce_prior@1 | −0.7533 | [−0.7664, −0.7413] |
| gk4-shared ce_iw@4 − ruler d6 | −0.0044 | [−0.0079, −0.0007] |
| gk1 ce_iw@4 − ruler d6 | +0.0039 | [+0.0004, +0.0076] |
| gk4 (confounded) ce_iw@4 − ruler d6 | −0.0005 | [−0.0043, +0.0033] |
| gk4-shared ce_iw@4 − gk1 ce_iw@4 | −0.0083 | [−0.0116, −0.0050] |
| gk4-shared ce_iw@4, depth 1 − depth 6 | +0.0035 | [+0.0029, +0.0041] |
| gk4-shared ce_prior@1 − ce_iw@4 | +0.0000 | [+0.0000, +0.0001] |
| gk4-shared ce_zero − ruler d6 | +0.1848 | [+0.1766, +0.1926] |

Paired readings on the sweep's tokens:

| arm | d6 − ruler d6 | K1−K6 | K3−K6 |
|---|---|---|---|
| lxtul-gk1 | +0.0038 [+0.0014, +0.0062] | +0.0037 [+0.0034, +0.0041] | +0.0003 [+0.0002, +0.0004] |
| lxtul-gk4 (confounded) | −0.0012 [−0.0035, +0.0011] | +0.0035 [+0.0032, +0.0039] | +0.0001 [−0.0000, +0.0003] |
| lxtul-gk4-shared | −0.0033 [−0.0056, −0.0010] | +0.0036 [+0.0032, +0.0039] | −0.0001 [−0.0002, +0.0000] |

The slot-loop floor for K1−K6 across twelve earlier arms is [−0.0001, +0.0033]. All three
GK arms sit at its top edge, and all their depth effect is in passes 1 to 3.

Scorecard (on lxtul-gk4-shared):

| clause | reading | verdict |
|---|---|---|
| P-1 ce_iw@4 better than lxtul-g ce_prior@1 by > 0.5 | −0.7533 | held |
| P-2 ce_iw@4 within +0.020 of the ruler; better than the ruler | −0.0044 [−0.0079, −0.0007] | held, both clauses (n = 1, see below) |
| P-3 width gain > 0.010; larger than gk1's | 0.00005 vs 0.00007 | failed, both clauses |
| P-4 prior sigma/r > 0.02 | 0.00026 | failed |
| P-5 ce_iw@4 d1 − d6 > +0.005; K1−K6 > +0.005 | +0.0035; +0.0036 | failed, both clauses |
| P-6 ce_iw@4 better than gk1's | −0.0083 [−0.0116, −0.0050] | held (n = 1, see below) |
| P-7 tok/s ≥ 0.5x the ruler | 0.476x | failed |

## Verdict

Failure. P-1 holds by a wide margin: training on prior rollouts removes the exposure gap,
as the design said it would. Everything the design was FOR fails. The learned noise
switches itself off (sigma/r 0.1 at init to 0.00026, P-4), so the K rollouts are one
rollout (identity null equal to one sample, width gain 0.00005, P-3), and depth reads the
same as the deterministic ruler's (P-5). The verdict rule for P-4 and P-3 failing together
applies: **the objective prefers a deterministic loop; exploration is not paid on web text
at this scale** under a learned Gaussian step.

Why the noise dies has a known answer. The multi-sample bound is a smooth (log-mean-exp)
aggregate. Its gain from spread is second order in sigma, while the cost of noise on the
mean rollout is first order, so gradient descent drives sigma to the floor. XM (arXiv
2607.27372, App. F.1;
[`docs/references/tul-latent-emission/explorative-modeling/`](../../../docs/references/tul-latent-emission/explorative-modeling/explorative-modeling.md))
derives this and uses a hard min to pay for spread at first order. It fits all three arms.

The two held clauses (P-2's second clause, P-6) are NOT width or depth effects. With
identical rollouts the K = 4 gradient equals the K = 1 gradient (pinned by
`test_identical_rollouts_give_the_k1_gradient_on_every_parameter`), so gk4-shared and gk1
train the same function after the noise collapses, at different commits and with
different early noise. Their 0.0083 gap and gk4-shared's 0.0033 to 0.0044 lead over the
ruler are each one run against one run. MORPH runs decorrelate within 11 steps at a fixed
seed, and the plain seed floor measured on 2026-09-22 is about 0.004. Read them as
run-to-run spread until a seed twin says otherwise.

Cost: the K = 4 step recomputes the coda and the span decode in the backward and runs the
loop and coda on 4x the rows: 0.476x the ruler's rate, for no width.

## Updated hypothesis

Closed: a learned Gaussian step inside the slot loop, trained by the ELBO (LXTUL-G) or by
the smooth multi-sample bound (LXTUL-GK). No more learned-sigma arms. Randomness in the
loop is not what the coda is missing on web text at 5k. Two open lines, neither a new
Gaussian arm:

1. Reasoning by Superposition (arXiv 2505.12514,
   [`reasoning-by-superposition`](../../../docs/references/looping-depth/latent-exploration/reasoning-by-superposition/reasoning-by-superposition.md)) keeps a frontier in ONE
   deterministic continuous state and pays one hop per pass, with no noise. A probe for
   whether our trained cells already carry a superposed frontier is being built
   (next prereg before it runs).
2. A proof-first design for loop contribution in Lean
   (`lab/theory/`, in progress) before any further arm.
