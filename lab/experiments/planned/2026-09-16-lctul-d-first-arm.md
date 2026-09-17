# Experiment: LCTUL-D first arm — a discrete code and a masked denoiser

Status: planned

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

(to fill)

## Verdict

(to fill)

## Updated hypothesis

(to fill)
