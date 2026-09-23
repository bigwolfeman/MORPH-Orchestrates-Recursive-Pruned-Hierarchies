# Planned: LXTUL-GK, the stochastic slot loop trained on K prior rollouts (multi-sample bound)

Status: planned

Date: 2026-09-23 13:27 (frozen before any build or GPU step). Parent:
[`2026-09-23-lxtul-g-panel.md`](../failures/2026-09-23-lxtul-g-panel.md). Note:
[`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-23-lxtul-gram-stochastic-loop.md).
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
