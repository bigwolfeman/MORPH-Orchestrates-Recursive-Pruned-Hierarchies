# dmorph v1 panel — results (seed 1, 5,000 steps)

Prereg: `lab/experiments/failures/2026-09-03-dmorph-v1-panel.md` (with its three Method
amendments). Design: `.agents/notes/rejected/architecture/2026-09-03-dmorph-v1.md`.
Runs (wandb `adew-me/morph-tul`): `dmorph-ctl-s1-5k`, `dmorph-tok-s1-5k` (second attempt;
the first is `dmorph-tok-s1-5k.ABORTED-r1.log`, amendment 3), `dmorph-hs-s1-5k`.
Recipe: `dmorph_{ctl,tok,hs}.yaml`, seq 1024, batch 6, warmup 1000, `dmorph.source_std
1.0` on the two dmorph arms (amendment 2). Logs are in this directory.

## Log reads (`python lab/dmorph/score_logs.py dmorph-*-s1-5k.log`, last 4 evals: 4250–5000)

| run | clean head CE | one-pass noisy head `dm_ce` | ladder CE | ladder cos | tok/s (steps ≥ 1000) | wall clock for 5k |
|---|---|---|---|---|---|---|
| ctl | 4.2096 | – | – | – | 28,856 | 19.0 min |
| tok | 4.3588 | 3.8027 | 6.3956 | 0.179 | 19,997 | 29.4 min |
| hs  | 4.6140 | 4.1392 | 4.9108 | 0.141 | 19,888 | 29.5 min |

Wall clock is the runner's START→DONE span (compile included in all three). The
matched-wall-clock step for a dmorph arm is 5000 × 19.0 / 29.4 ≈ 3230; the eval at 3250
reads tok 4.5937, hs 4.8167, against ctl's 4.2096 at 5000.

Per-band one-pass CE of the tok arm at the final eval: band 0 (t ∈ [0, 0.25), the high-noise
read) 5.3824, band 1 4.7722, band 2 3.7910, band 3 0.9257. The grid mean 3.73 is carried by
the autoencoding bands; the band-0 read is 1.05 nats WORSE than the clean head.

tok arm greedy accuracy on the same positions at the final eval: clean head 0.2896,
ladder head 0.2255.

hs arm four-condition worth at the final eval (CE at the emit position through the same
readout): clean target 5.3351, ladder 4.5768, zero 10.8030, shuffle 7.1649; costs relative
to clean: ladder −0.7583, zero +5.4679, shuffle +1.8298.

## Checkpoint reads at step 5000 (GPU, after the runs)

hs four-condition worth, 40 val batches, bootstrap CI (`hs-worth-step5000.json`, `python
lab/dmorph/worth_scorer.py --ckpt checkpoints/morph/dmorph-hs-s1-5k/step_5000.pt --config
dmorph_hs --override dmorph.source_std=1.0 --batches 40`): clean target 5.5503 [5.4091,
5.7080], ladder 5.1675 [5.0015, 5.3616], zero 10.8030, shuffle 7.5012 [7.3583, 7.6418];
costs against clean: ladder −0.3829 [−0.4507, −0.3152], zero +5.2527, shuffle +1.9509;
ladder cosine to the target 0.1437 [0.1413, 0.1464]; ladder greedy accuracy at the emit
position 0.3072; the clean HEAD at the emit position 5.4154 [5.2617, 5.5887]. The ladder
output is NOT a copy of the target (cosine 0.14) and reads 0.38 nats better than the
target through the same readout — a readout whose temperature was trained on ladder
outputs, not on unit targets, so "better than clean" is a readout artefact, not a
refinement claim; the run's overall clean head is 0.40 nats behind the control.

tok decodability on the trained table with the reshaped source (`tok-decodability-step5000.json`,
`python lab/dmorph/decodability.py --config dmorph_tok --override dmorph.source_std=1.0
--ckpt checkpoints/morph/dmorph-tok-s1-5k/step_5000.pt`): `t* = 0.85` (band 3), training
mass above it 13.6 %. Prediction P6 was resolved TRUE at init under the matched source
(Method amendment 2: the source was reshaped to `source_std 1.0` BEFORE the panel, which is
the trap doing exactly what P6 said it would); with the reshaped source the trained table
keeps 13.6 % of the mass in the autoencoding region, and that region carries the band-3
one-pass CE of 0.93.

## Prereg resolution (`failures/2026-09-03-dmorph-v1-panel.md`)

| P | prior | read | result |
|---|---|---|---|
| P1 tok clean head within ±0.02 of ctl at matched tokens | 55 % | 4.3588 vs 4.2096 (+0.149) | FALSE |
| P2 ctl beats both arms at matched wall-clock | 65 % | ctl 4.2096 vs tok 4.5937 / hs 4.8167 at step 3250 | TRUE |
| P3 tok ladder greedy accuracy below the clean head | 70 % | 0.2255 vs 0.2896 | TRUE |
| P4 hs ladder cos > 0.9 AND worth cost ≤ 0.01 | 60 % | cos 0.1437; cost −0.38 | FALSE (cosine) |
| P5 both arms beat A2's 480-row 3.4701 at matched wall-clock | 60 % | 4.36 / 4.61 at 5k (A2's number is a 20k read; no 5k dmorph number is near it) | FALSE |
| P6 σ*-trap mass > 40 % with uniform t | 75 % | resolved TRUE at init (amendment 2); 13.6 % after the reshape | TRUE |

Three of six on the majority side → `failures/`.

## The FPF run (dmorph v1.1, `failures/2026-09-03-dmorph-fpf-tok.md`)

`dmorph-tok-fpf-s1-5k` (`dmorph_tok_fpf.yaml`: `fpf_p 0.5`, `recur 0`, everything else
the tok arm's), launched 15:06 after the hs run, exit 0, 35.5 min against the tok arm's
29.4. `python lab/dmorph/score_logs.py --last 3` (evals 4500, 4750, 5000):

| read | v1 tok | FPF tok |
|---|---|---|
| clean head CE | 4.3588 | 4.3920 |
| one-pass noisy head `dm_ce` (null carry) | 3.8027 | 3.8251 |
| ladder CE, 0 held-time updates | 6.3956 | 6.7974 |
| ladder CE, 2 held-time updates | – | 6.5226 |
| residual→correctness AUROC (r2) | – | 0.4678 |
| tok/s, steps ≥ 1000 | 19,997 | 17,104 (0.855×) |

Final-eval extras (step 5000): ladder greedy accuracy r0 0.2261, r2 0.2277, clean head
0.2873; mean residual r2 0.0561; per-band one-pass CE 5.478 / 4.647 / 3.808 / 0.947 (the
same σ*-trap shape as v1). The rollout ran: `dm/fpf_frac` averaged 0.474 over the logged
steps with mean `t_start` 0.224; the carry projection `W_s` at the step-2500 checkpoint
has Frobenius norm 24.1 against 7.8 for the velocity head, so the carry was learned and
used. The FPF ladder ran 0.2–0.4 nats BEHIND v1's ladder from step 1250 to the end on
every eval, and the residual AUROC sat below 0.5 on 15 of 16 evals (0.43–0.48; one read
of 0.62 at step 500).

| P | prior | read | result |
|---|---|---|---|
| P1 FPF ladder r0 ≥ 1.0 nat below v1's ladder | 65 % | +0.40 (worse) | FALSE |
| P2 FPF ladder r0 within 0.3 of its own one-pass head | 55 % | 6.80 vs 3.83 | FALSE |
| P3 held-time r2 ≤ r0 − 0.02 | 50 % | 6.5226 vs 6.7974 (−0.27) | TRUE |
| P4 residual AUROC r2 ≥ 0.65 | 55 % | 0.468 | FALSE |
| P5 clean head within ±0.05 of v1 | 70 % | +0.033 | TRUE |
| P6 tok/s ≥ 0.85× v1 | 70 % | 0.855× | TRUE |
