# Experiment: LCTUL-D first arm — a discrete code and a masked denoiser

Status: failure

Date: 2026-09-16 (frozen before any GPU step; a CPU contract suite and a Spark 12-step smoke
are the only runs that exist at filing time). Note:
`.agents/notes/proposed/architecture/2026-09-16-lctul-d-discrete-code-masked-denoiser.md`.
Spec: `docs/tul-code-spec.md` §16.

## Question

Does a thinker whose loss is the conditional entropy of a DISCRETE code (a masked denoiser)
learn the conditional given the past, where the flow thinker learned 1 % of it, and does its
k-round sampler show a k-curve?

## Hypothesis

The flow thinker's context-blindness is a property of the objective (velocity regression on
a high-entropy continuous code), not of the model or the data. On the same geometry, E, coda
and data, a categorical objective on symbols puts the past into the gradient, and the
unmasking sampler's rounds resolve within-span symbol dependence.

## Predictions (frozen)

Arm `tul-code-d` (`tul_code_d.yaml`: `tul_code` + `code_discrete`, C 512, G 4, N 8,
`code_sub_p` 0.3, `code_cfg_drop` 0.1, 20k steps, batch 6, seq 1024, phases 2 at 2k and 3 at
10k), one seed, against `tul-code-cfg` (the flow arm with the same null condition) and the
strict ruler at matched steps.

- P-D1 (codes used): `tul/vq_perplexity` ≥ 64 at 5k and never < 16 after step 500.
- P-D2 (code read): `val/ce_tf` at 5k ≤ strict ruler `val/loss` at 5k − 0.15.
- P-D3 (conditional thinker, the primary): at 20k the denoiser ELBO in nats per span with
  the past minus with the null condition on every row (the flow probe's twin, `cfg_drop` 0.999
  vs 0.001) ≥ 10 % of the with-past number. Flow arm: 1.1 %.
- P-D4 (a real sample at k = 1): at 20k the 8-draw marginal at k = 1 is within 0.03 of k = 2
  (the mean-vs-sample step of the flow arms, 0.05 to 0.11, is absent).
- P-D5 (depth resolves dependence): at 20k `ce_marginal(k=1) − ce_marginal(k=8)` ≥ 0.02 with
  the paired bootstrap CI clear of 0.
- P-D6 (the coda tolerates samples): at 20k `val/loss` (sampled, k 8) − `val/ce_tf` ≤ 0.30
  (flow arm at 20k: 0.63 against the ruler, phase-3 coda).
- P-D7 (rate): tok/s ≥ 0.8× `tul-code-cfg`'s at the same shape.

## Method

Queue line at the head of `recon_arms.txt` after the running stage. Amended 2026-09-16
19:35 (reason: the line was inserted): commit 7262ed8, line
`tul-code-d:tul_code_d:tul-code-d:code:5000,10000,15000,20000:20000:1:.../results/2026-09-16-lctul-d:7262ed8:`,
Spark watcher `spark_code_probes_d.sh` (marginal ks 1,2,4,8 at every checkpoint; subspace,
samples and flow at 20k). Predictions unchanged. Amended again 19:50 (reason: the first
queue line carried no EXTRA, so the arm started under `tul_code`'s 5000-step default with
phases at 500 / 2500; it was killed at step ~40, no checkpoint written, and re-queued with
`training.steps=20000 training.ademamix_t_beta3=20000 training.ckpt_every=5000`, the same
extras as `tul-code-cfg`; phases then sit at 2000 / 10000). Probes: the marginal k-sweep (k 1, 2, 4, 8) at 5k / 10k / 15k /
20k on the Spark; the context-share reading at 20k from `code_flow_probe.py` with
`tul.code_cfg_drop=0.001,tul.code_cfg_scale=1.0` and `0.999` (it reports the denoiser ELBO
through the `code_fm_*` keys on a discrete model; convert `code_fm_raw` to nats per span);
span samples and the semantic probe at 20k. Pairing: `paired_vs_ruler.py` on the ruler's
sweep JSON.

## Results

The arm ran from 19:50 to 21:53 on 2026-09-16 (wandb `yonzqfnz`, last synced step 9320,
last val line 9750) and was killed at step ~9900 on Wolfe's call to replace it with the
coarse-plan arm `tul-code-dplan` (`2026-09-16-lctul-dplan-semantic.md`). No 10k checkpoint
exists; every 20k prediction is read at the 5k checkpoint, the only one written. Artifacts:
`lab/experiments/results/2026-09-16-lctul-d/` (context-share pair, continuity JSONs, the 5k
marginal sweep, `readings-5k.txt`).

| Prediction | Reading at 5k | Holds |
| --- | --- | --- |
| P-D1 perplexity ≥ 64 at 5k, never < 16 after 500 | 234.9 at 5k (297 of 512 codes used); minimum after step 500 is 43.5 at 7480 | yes |
| P-D2 `val/ce_tf` ≤ ruler − 0.15 | 3.960 against the strict ruler's 4.378 (−0.418) | yes |
| P-D3 context share ≥ 10 % (primary) | ELBO rel 0.6252 with the past, 0.6251 without: 0.02 % (flow arm 1.1 %) | no |
| P-D4 k = 1 within 0.03 of k = 2 | 8-draw marginal 4.781 at k 1, 4.833 at k 2 (0.053) | no |
| P-D5 k1 − k8 ≥ 0.02, CI clear | −0.162 [−0.176, −0.146]: MORE rounds are WORSE | no (inverted) |
| P-D6 sampled − tf ≤ 0.30 | 5.650 − 3.960 = 1.69 at 5k (phase 2; phase 3 never ran) | no at 5k |
| P-D7 tok/s ≥ 0.8× the flow arm | 21,858 against 30,350 = 0.72× | no |

The continuity probe (`readings-5k.txt`) reads the 5k codes the same way it read the flow
arm's: consecutive slots are barely closer than random slots (cosine 0.075 against 0.040;
symbol match 0.046 against 0.027) and lag 1 = lag 2 = lag 4. The code is a document
signature with no sequential structure, on both arms.

## Verdict

Failure. The primary prediction P-D3 fails harder than the flow arm it was built to beat:
the masked denoiser puts 0.02 % of its loss on the past at 5k where the flow thinker put
1.1 % at 20k. The categorical objective did not put the past into the gradient. The 72-bit
code (8 symbols of 512) is read well by the coda (P-D1, P-D2 hold) but is not predictable
from the past, so the k-curve inverts: every extra unmasking round moves the sample further
from the encoder's code (k 1 is the best round on CE at every checkpoint the sweep saw).
What the method could not distinguish: whether the 20k reading would have moved, and
whether the rounds would have helped on GENERATION rather than CE (the Jensen argument in
`2026-09-16-lctul-dplan-semantic.md` says CE cannot reward a sampled latent over a mean).
The next planned file answers both with a coarser code and a semantic score.

## Updated hypothesis

Context-blindness is not a property of the flow objective. It follows from the code's
rate: 72 bits per span of a signature the past does not determine, when the whole cross-span
budget is 0.40 nats per token. A code the past CAN determine must be small (the plan arm:
4 symbols of 64, 24 bits) and must be scored on what it does to the generated span, not on
the CE of a sampled latent.
