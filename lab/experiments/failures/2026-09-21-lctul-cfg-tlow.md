# Planned: LCTUL cfg arm with the flow noise schedule shifted to HIGH noise (`tul.code_t_logit_mean −1`)

Status: failure

Date: 2026-09-21 (frozen before any GPU step of the arm). Arc: the LXTUL-P note
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
rung P2's first half (a pass with a job, on the existing flow thinker, before the target
swap). Parent: `tul-code-cfg`
([`failures/2026-09-15-tul-code-conditioned-thinker.md`](../failures/2026-09-15-tul-code-conditioned-thinker.md)).
Wolfe, 2026-09-21: "If LCTUL with flow matching is the same thing, then why doesn't ours
work? We need to test those changes."

## Question

LCM's denoiser and our flow thinker are one mechanism (a noised target at a drawn noise
level, the clean target as the per-level loss). Ours is context-blind: its flow loss moves
1 % when the past is removed (0.3125 with the past against 0.3161 with the trained null on
`tul-code-cfg` @ 20k), and guidance at sample time, which only amplifies a learned context
gap, moved the one-draw CE by < 0.01 nats across w ∈ [1, 3]. The one dial LCM found that
flips a denoiser from regressing to contrasting is the training noise schedule (their
Table 5: a schedule concentrated at low noise reads best ℓ2 and worst CA, "akin to a
Base-LCM"; a schedule spread to high noise reads best CA). Our t is uniform on [0, 1]; on
the straight-line path with a unit source the log-SNR is `2·logit(t)`, so half of training
sits above SNR 1 where the noisy input already carries the answer. Does shifting the
schedule to high noise make the thinker read the context?

## Hypothesis

At high noise the input tells the field nothing, so the only way to lower the loss there is
to read the past; a schedule that spends 84 % of training below t = 0.5 forces it. The
target is still a verbatim code of which the context explains 5–15 % (the residual-ratio
reading 1.7–1.9, `lab/theory/lctul_euler_depth/`), so the CEILING on what any schedule can
give is low; the arm measures whether the field moves toward that ceiling, not whether the
code becomes the span.

## Method

One factor over `tul_code_cfg` (CFG drop 0.1, scale 2.0, renormed truth cell):
`morph/configs/tul_code_cfg_tlow.yaml` sets `tul.code_t_logit_mean: −1.0`,
`tul.code_t_logit_std: 1.0` and nothing else: `t = sigmoid(−1 + ε)`, median 0.269
(λ = −2, SNR 0.135), E[t] = 0.3033, P(t < 0.5) = 0.841. The draw consumes the same one
`torch.rand` as the uniform branch (the normal quantile of the same uniform), so the arm
and its parent stay aligned on every later random draw of a step; `None` is bit-identical
to the tree before the knob (tested). The sampler's Euler grid stays uniform. New
instrument: `val/code_ca` (LCM's contrastive accuracy on the sampled code: in-batch
retrieval of the slot's own true code among the batch's valid slots, the two temporal
neighbours excluded) against `val/code_ca_chance`; `train/code_t_mean` reads the schedule.
Same recipe as the parent: 20,000 steps from scratch through the three LCTUL phases,
`training.ademamix_t_beta3=20000`, `ckpt_every=5000`, the runner (SWEEP@5k/10k/15k/20k,
PROFILE, STATEPROBE, WORTH) into `/home/wolfe/morph-scratch/arc/results/2026-09-21-lctul-tlow/`.
Readouts: the context-blind flow probe (`code_flow_probe`, `code_cfg_drop` 0.999 vs 0.001
on the 20k checkpoint, the parent's instrument); the paired one-draw read against the
parent's and the strict ruler's `tokens.npz` at 20k; the k-curve of the 8-draw marginal
(`val/ce_marginal` at k = 1 and 16, the parent's probe on the Spark or the 3070); `val/code_ca`
and `val/code_ca_chance` from wandb.

## Predictions (frozen)

Parent (`tul-code-cfg` @ 20k) for reference: context-blind gap 0.0036 (1 %); sample
residual 1.81 / 1.84 (rank-128 head); 8-draw marginal k = 1 → 16: 4.380 → 4.375 (−0.005);
one draw vs the strict ruler +0.648; one draw vs `tul-code-20k` +0.022; ce_tf 0.32;
phase-3 rate 19.7k tok/s. `val/code_ca_chance` is about 1/(valid slots in a batch − 2),
roughly 0.003 at batch 6 × 64 slots. Probabilities are the builder's.

- **P-1 (healthy).** 20,000 steps, tripwire HEALTHY, phase-3 rate at or above **15,000**
  tok/s. **85 %.**
- **P-2 (schedule live).** `train/code_t_mean` inside **[0.28, 0.33]** over phases 2–3 (the
  analytic 0.3033). **95 %.** The sanity clause: if this fails the knob did not reach the
  draw.
- **P-3 (THE ARM'S REASON: the field reads the context).** The context-blind flow probe on
  the 20k checkpoint moves by **more than 3 %** of the with-past loss when the past is
  removed (parent 1 %). **45 %.** Against it: the target's predictable fraction is one
  tenth, and a field can meet a high-noise loss with the code's unconditional mean.
- **P-4 (CA above chance).** `val/code_ca` at 20k above **3 × `val/code_ca_chance`**.
  **50 %.** No parent number exists (the instrument is new); chance is the floor a
  conditional-mean sample reads.
- **P-5 (the marginal earns steps).** 8-draw marginal `k = 16 − k = 1` at or below
  **−0.020** (parent −0.005). **35 %.** The theorem: an affine field gives one scalar per k;
  guidance on a field that reads the context is what makes it curve.
- **P-6 (one draw vs the parent).** Paired one-draw CE at k = 8, tlow − cfg, at or below
  **−0.020** (parent − tul-code-20k was +0.022). **35 %.**
- **P-7 (the ceiling stands).** One draw vs the strict ruler at 20k at or above **+0.45**
  (parent +0.648). **80 %.** A verbatim target caps what the sampler can carry; this
  clause is the reason the SONAR swap (rung P3) is queued regardless of P-3.
- **P-8 (E untouched).** ce_tf at 20k at or below **0.40** (parent 0.32): the schedule does
  not reach the encoder. **85 %.**

## Binding

If **P-3 holds and P-7 holds**: the schedule was a real lever and the verbatim target is the
ceiling; rung P3 (the same config with the target swapped to SONAR embeddings of the next
span) goes next, composed FROM this arm's config.

If **P-3 fails**: on a verbatim target no schedule makes the field read the past; rung P3
still runs, but from the parent's uniform schedule with the shift as a second factor to
re-test on the semantic target, and the note's Risks gain the line "the schedule is not a
lever on a lossless code".

If **P-5 or P-6 lands in its tail** (gains of 0.05 or more): the sampler was a larger part
of the limit than the target; sweep `code_t_logit_mean` (−2, −0.5) and the width before
the target swap.

If **P-2 fails**: stop, read the config path, and re-queue; nothing else in the panel is
readable.

## Not verified before launch

- No GPU step of the arm; the runner's smoke is the first. The wandb series
  `val/code_ca`, `val/code_ca_chance`, `train/code_t_mean` are emitted by the model and
  named in the trainer's key lists (tested on a CPU eval forward), never yet seen in a run.
- CA's cost is arithmetic (one [n, n] cosine per val forward, also on the marginal's K
  sampled forwards), not measured.
- The context-blind probe and the Spark k-curve probe scripts the parent used are to be
  located in `lab/experiments/results/2026-09-15-tul-code-cond/` at readout time; not
  re-run in this session yet.
- The draw's `None` branch is a config read inside `draw_flow_t`, a Python-level constant
  on a frozen field (the same class as the `code_discrete` branch beside it), not a
  build-time bound method.

## Results

Run: 20,000 steps at `6210c1f` on the 5090 through `run_recon.sh`, START 04:33 local
2026-09-21, DONE 07:35, tripwire HEALTHY (`preclip/total` max 143 at step 11454), final
val_loss 4.4686, phase-1 rate 33,915 tok/s at step 200, phase-3 rate 19,157 tok/s (parent
19,742), peak 13.22 GB. wandb `bzltw7vw`. Artifacts in `../results/2026-09-21-lctul-tlow/`:
`sweep_tul-code-cfg-tlow_{5000,10000,15000,20000}.json` (+ the 20k `tokens.npz`),
`worth_..._20000.json`, `slot_state_..._20000.json`, `run_tul-code-cfg-tlow.txt`,
`wandb_series_tul-code-cfg-tlow.json`, `code_flow_tul-code-cfg-tlow_20000_{blind,past}.{json,txt}`
and `code_marginal_sweep_tul-code-cfg-tlow_20000.{json,txt}` (Spark, worktree `MORPH-0921`
at `82868b4`, the parent's 96 rows), `paired_tlow_vs_parent_20000.json` (ruler = the
parent's 20k sweep at k = 1), `paired_tlow_vs_ruler_20000.json` (ruler =
`slot-spandec-strict-20k` at depth 6).

**The schedule reached the draw.** `train/code_t_mean` over phases 2–3 (900 points): mean
**0.3030**, min 0.2752, max 0.3355; analytic 0.3033.

**The new instrument.** `val/code_ca` climbed 0.0027 (step 250) → 0.0049 (6750) → 0.0085
(19750) against `val/code_ca_chance` 0.0032–0.0033: **2.7 × chance** at 20k. The parent has
no CA reading (the instrument is new), so this is the FIRST above-chance in-batch retrieval
of a sampled MORPH code; LCM's Base-LCM read 70–80 % on sentences.

**Context-blind flow probe at 20k (Spark, 96 rows, 2 repeats).** With the past
(`code_cfg_drop` 0.001) `code_fm_rel` **0.3921 ± 0.0025**; blind (0.999) **0.3994 ±
0.0023**; gap 0.0073 = **1.9 %** of the with-past loss (parent 0.0036 = 1 %). Bands 0–3:
0.4513/0.3598/0.2927/0.3789 with the past, 0.4626/0.3639/0.2961/0.3851 blind.

**8-draw marginal (Spark, 96 rows).** k = 0 (encoder) 0.3239; k = 1 → 16: 4.3634, 4.3541,
4.3532, 4.3548, 4.3566; **k = 16 − k = 1: −0.0068 [−0.0089, −0.0048]** (parent 4.3795 →
4.3748, −0.0047 [−0.0070, −0.0024]). The k = 1 marginal is 0.016 below the parent's on the
same rows.

**Paired one-draw reads (501,106 tokens, 490 blocks).** vs the parent at k = 1: k = 1
**−0.0095 [−0.0123, −0.0068]**, k = 2 −0.0149, k = 3 −0.0117, k = 6 −0.0041, k = 9 −0.0005,
k = 12 +0.0013, k = 16 +0.0029. vs the strict ruler at depth 6: k = 1 **+0.6384 [+0.6310,
+0.6462]** (parent +0.648), k = 16 +0.6508; the encoder code (k = 0) −3.4704. Runner sweep at
20k: K1−K6 −0.0054 [−0.0067, −0.0039], K3−K6 −0.0076: more Euler steps HURT a single draw
past k = 2.

**Teacher-forced CE.** `val/ce_tf` 0.3551 at 19750 (min over the run 0.3151; parent 0.32).
Worth profile at 20k: zero 0.0558, shuffle 0.0114.

**Scoring.**

- **P-1: HOLDS.** HEALTHY, phase-3 19,157 tok/s ≥ 15,000.
- **P-2: HOLDS.** `code_t_mean` 0.3030 inside [0.28, 0.33].
- **P-3 (the arm's reason): FAILS.** 1.9 % against 3 %. The past is worth twice what it was
  worth to the parent and still under the bar.
- **P-4: FAILS.** 2.7 × chance against 3 ×.
- **P-5: FAILS.** −0.0068 against −0.020 (parent −0.0047).
- **P-6: FAILS.** −0.0095 [−0.0123, −0.0068] against −0.020 (parent − tul-code-20k was
  +0.022, so the sign flipped and the interval is clear of zero).
- **P-7: HOLDS.** +0.638 against +0.45. The verbatim ceiling stands.
- **P-8: HOLDS.** 0.355 ≤ 0.40.

Four of eight hold. The four that fail all moved in the predicted direction by roughly
half their bar.

## Verdict

**Failure** on the arm's reason. Shifting the flow noise to high noise (E[t] 0.30 against
0.50) is a real but small lever on a verbatim target: the field's context dependence
doubles (1 % → 1.9 %), a sampled code is retrievable in-batch at 2.7 × chance, one draw reads
0.010 nats better than the parent and the k-curve bends 0.002 more, while the ceiling
against the strict ruler stays at +0.64 nats. None of it reaches the bars the prereg set,
and the parent's diagnosis stands: on a code whose predictable-from-context fraction is a
tenth, no schedule makes the field read the past. Binding branch taken: "P-3 fails: rung P3
still runs, but from the parent's uniform schedule with the shift as a second factor to
re-test on the semantic target". The sonar arm was BUILT composed from this config
(`tul_code_cfg_tlow_sonar.yaml`); per this binding it should be re-cut over
`tul_code_cfg` (uniform t) with the shift as a second factor, and it is HELD in any case on
its own C-1 miss (`planned/2026-09-21-lctul-tlow-sonar.md`). The note's Risks gain the line
"the schedule is not a lever on a lossless code".

## Updated hypothesis

A wide, high-noise schedule gives the field more of its loss at levels where the context is
the only signal, and the field takes it (the gap doubled). The size of what it can take is
bounded by the target: E's code is a near-lossless copy of the next span (ce_tf 0.32–0.36),
and the context explains a tenth of it, so the ceiling on the context-blind gap is the
target's, not the schedule's. The lever on the target is rung P3 (SONAR) or a code that IS
the conditional entropy (LCTUL-D). The schedule shift is kept as a second factor for that
test, not as a rung.

**Addendum 2026-09-21 (round-trip probe, Spark, `code_roundtrip_tul-code-cfg-tlow_20000.{json,txt}`,
48 rows = 2,449 slots, not a pre-registered clause).** Chance cosine between two slots'
true codes on this arm: 0.1349 [0.1268, 0.1434] (tul-code-20k 0.1016). The k = 8 sample's
cosine to its own truth: **0.1457 [0.1392, 0.1527]**, above chance by 0.011 with the
intervals touching (tul-code-20k's k = 8 sample sat AT chance, 0.098 vs 0.102); its
round-trip gap −0.006 (nothing to lose). The k = 1 sample: 0.2232 (tul-code-20k 0.2221).
The TRUE code decoded and re-encoded lands at **0.892** on this arm against **0.291** on
tul-code-20k (gap 0.108 vs 0.709): this encoder/decoder pair keeps a code on a round trip
where the earlier pair lost most of it. Which factor did that — the CFG training (tlow's
parent `tul-code-cfg` has no round-trip reading) or the schedule — is UNMEASURED; run
the probe on `tul-code-cfg_step_20000.pt` (on the Spark) before crediting the schedule.
