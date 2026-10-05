# Planned: what if the LXTUL prelude and loop see the full token history? (pair)

Status: planned

Date: 2026-10-05 02:23 CDT, before any smoke or run. Wolfe (2026-10-05): "we need to test the prelude and loop
seeing full token history." Note:
[`2026-10-05-prelude-full-token-history`](../../../.agents/notes/proposed/architecture/2026-10-05-prelude-full-token-history.md).

| arm | config | prelude | loop | coda |
| --- | --- | --- | --- | --- |
| full history, open coda | `lxtul_hist` | every earlier token, no segment resets | seeds pooled from history-aware prelude states | reads its tokens' prelude states, so history also reaches it WITHOUT the loop |
| full history, coda reads embeddings | `lxtul_hist_embed` | the same | the same | reads token embeddings + cells only: the loop is the only cross-span route (exact-0 perturbation test with cells zeroed) |

Both compose `lxtul.yaml`; they differ in `tul.tg_strict_prelude: causal` (+ `tul.coda_token_input:
embed`), `training.steps` 5000 and `wandb.name`.

## Question

The open-coda arm is the CE ceiling of this family and a loop-contribution stress test: with
a bypass, does the loop still earn? The embeddings arm asks the real question: given seeds
that saw the whole past, does the loop deliver more of it to the coda than today's own-span
seeds? Cost to watch: the coda loses the prelude's own-span processing (4 blocks).

## Predictions (mine, orchestrator)

References: LXTUL 5k seed 1 / 2: gap +0.272 / +0.282, K1-K6 +0.0215 / +0.0172, far-bigram
bucket gap +1.150 / +1.154; tok/s at this checkpointing about 12400.

- **H-1** open coda: gap to plain 5k below +0.150. 60 %.
- **H-2** open coda: K1-K6 below +0.0086 (the loop goes idle when a bypass exists). 65 %.
- **H-3** embeddings: gap below +0.262. 40 %.
- **H-4** embeddings: K1-K6 above +0.0108, CI above 0. 55 %.
- **H-5** embeddings: far-bigram bucket gap at least 0.10 below +1.150. 35 %.
- **H-6** both: tok/s at least 10000. 65 %.
- both: no detonation. 80 % each.

Pass rules: open coda, H-1 (the ceiling is real); embeddings, H-3 and H-4.

## Method

5000 steps each, seed 1, `model.ckpt_grad_iters: 4`, `runner_steps.sh`, queued right after the
snap pair. Mechanism check and 60-step smoke first (`smoke_queue.sh`). Readouts per arm
(`arm_after.sh`): depth sweep, gap vs plain 5k, worth, ledger, LayerNorm probe and
`--vablate`, exact-recall probe with per-bucket loop K1-K6, repetition eval.
