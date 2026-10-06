# Planned: what if the LXTUL prelude and loop see the full token history? (pair)

Status: failure

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

## Results

References as preregistered: LXTUL 5k seed 1 / seed 2 K1-K6 +0.0215 / +0.0172, gap to plain 5k +0.272 / +0.282, zero-cell worth +0.176 / +0.154, far-bigram bucket gap +1.150 / +1.154. Readouts: 480-row depth sweep, `paired_vs_ruler.py` gap vs plain 5k, `worth_profile.py` on 192 rows, the exploration ledger, `exact_recall_gap.py` vs plain 5k, the LayerNorm probe's `--vablate` on 96 rows (K6-K1 change when the cells' shared-direction amplitude is set to its mean). Artifacts: [`../results/2026-10-05-lxtul-full-history-pair/`](../results/2026-10-05-lxtul-full-history-pair/).

| reading | open coda (prelude sees full history) | embeddings coda (coda reads raw embeddings) |
| --- | --- | --- |
| tripwire | HEALTHY, max 434 | AMBIGUOUS, max 3.9e3 |
| K1-K6 | +0.0140 [+0.0132, +0.0147] | +0.0134 [+0.0128, +0.0141] |
| gap to plain 5k | +0.2308 [+0.2175, +0.2461] | +0.4046 [+0.3891, +0.4222] |
| zero-cell worth / shuffle | +0.033 / +0.015 | +0.173 / +0.141 |
| ledger CE | 4.3109 | 4.4850 |
| far-bigram bucket gap | +1.040 | +1.248 |
| mean-ablation K6-K1 change | -3.7 % | -1.2 % |
| tok/s, peak | 12295, 20.1 GB | 12214, 20.3 GB |

| prediction | held |
| --- | --- |
| H-1 open coda gap < +0.150 | no (+0.231) |
| H-2 open coda K1-K6 < +0.0086 | no (+0.0140) |
| H-3 embeddings gap < +0.262 | no (+0.405) |
| H-4 embeddings K1-K6 > +0.0108 | yes |
| H-5 embeddings far-bigram gap >= 0.10 below +1.150 | no (worse, +1.248) |
| H-6 tok/s >= 10000 | yes, both |
| no detonation | yes, both |

## Verdict

Failure on both pass rules. The open coda, the family's CE ceiling, closes only 0.04 of the gap: a
prelude that sees the whole past still leaves +0.231 to plain, so most of the gap is not a lack of
history at the prelude. The cells are then nearly worthless (worth 0.033) yet the loop still
earns K1-K6 +0.014, so depth earning here is not the cells' cross-span content. The embeddings coda
loses the prelude's own-span processing and costs 0.13 nats; full-history seeds do not make the
loop deliver more.

## Updated hypothesis

The gap is not mainly missing history at the front: even with full history in the prelude the
strict geometry's coda (4 blocks over its own span plus slot cells) trails plain by 0.23.
