# Experiment: which objective writes to the core, and do they fight?

Status: **failure — the hypothesis was refuted, the measurement worked.** O2 failed
and O5 came back as neither of the two modes I predicted. Filed here because
"success" means the predictions held, not that the run produced a number.
Requested by Wolfe: *"our real root is gradient assignment on the core."*

## Question

Every gradient number this campaign has produced comes from
[`train.py::_preclip_probe`](../../../morph/training/train.py), which reads
`p.grad` **after the backward of the summed objective**. One backward of
`L = L_main + L_plast + L_emit` leaves `g_main + g_plast + g_emit` in one tensor.
That sum is identical whether the terms are aligned, orthogonal, or cancelling.

So the instrument cannot separate the three things the literature separates, and
they take opposite fixes:

| mode | signature on the core | fix |
| --- | --- | --- |
| conflict | `cos(g_a, g_b) < 0` | project — PCGrad, CAGrad |
| domination | `cos ≈ 0`, one norm ≫ the other | rescale — GradNorm, MGDA, FAMO |
| starvation | `cos > 0`, one norm tiny | decouple, or delete the shortcut |

This decides the next architecture move. If the core's objectives CONFLICT,
adding a fourth (MTP) makes it worse. If the coda's route is aligned but tiny,
adding targets is exactly right.

## Method

[`lab/divergence/objective_split.py`](../../divergence/objective_split.py). One
forward+backward per objective on the same batch under the same RNG seed, then
per region: each objective's gradient norm and the cosine between every pair.
Train mode, so the Poisson depth draw and token-state dropout are inside the
gradient being decomposed. Validation batches at `skip_samples=60_000`, the same
fixed set `slot_path_worth.py` uses, so the numbers sit next to the P panel.

Groups: `main` (the coda's ordinary token CE), `plast` (last token of the span
predicting the next span's first token — the free token path), `emit` (the slot's
own prediction of that same token), and `mux` where an arm carries the head.
A group whose configured weight is 0 is EXCLUDED from the objective and reported
at unit weight with a `*`, direction only.

### Two gates, both of which refuse the whole panel

1. **Additivity.** `Σ_g (W_g/W)·g_g` must equal the full CE gradient to 2e-2
   relative. The reduction is `Σ w·CE / Σ w`, so a group-only pass carries a
   different denominator; each group is rescaled by `W_g/W`, and this test is
   what proves the rescale right.
2. **Determinism.** The same objective run twice must reach self-cosine ≥ 0.99.
   MORPH's bag-mean scatter uses atomics and a 4 % per-step gradient error from
   that source is already on record
   ([memory](../../divergence/takeover-campaign.md)). A cross-objective cosine of
   0.05 means nothing if an objective against ITSELF only reaches 0.9.

Both gates are proven to BITE by `--sabotage`, which drops the rescale or
reseeds the self-run; `tests/test_objective_split.py` asserts each then exits
non-zero. A gate that cannot fail is not a gate.

### Checkpoints

| label | checkpoint | why |
| --- | --- | --- |
| control | `ctrlworth-s2` steps 1000 / 2000 / 3000 | healthy, and the evolution |
| taken over | `ctrlworth-s1` step 3000 | this seed takes over at ~2800 |
| MUX arm | `tul-v1a2b-s2` step 3000 | highest plan worth measured, 0.0330 |
| broken arm | `tul-warmup-s0` step 3000 | ppl_tok 305; does conflict appear only when an arm breaks? |

## Predictions (frozen 2026-08-27, before any run)

Read on the **core** region. Reference: control plan worth 0.0164 nats at step
3000; `ce_emit`'s worth to its own target 2.6564 nats;
`ce_plast − ce_emit = −0.2191`; core pre-clip share > 0.9 at takeover.

- **O0 (validity):** both gates pass on every checkpoint. If either fails,
  nothing below is readable and the panel is void — not "reported with caveats".
- **O1 (it is NOT conflict):** on the healthy control at step 3000,
  `cos(g_main, g_emit) > −0.05`. The campaign's evidence is that the plan is
  EMPTY, not fought over.
- **O2 (the coda's route is tiny):** on the same checkpoint,
  `‖g_main‖ / ‖g_emit‖ < 0.25` on the core. The indirect route is worth 0.0164
  nats against `ce_emit`'s 2.6564 to its target.
- **O3 (the takeover is emit-driven):** on `ctrlworth-s1` at step 3000, `emit`'s
  share of the summed core gradient norm is **> 0.5**, and higher than it is on
  the healthy control at the same step.
- **O4 (MUX is aligned, not accidental):** on `tul-v1a2b-s2`,
  `cos(g_mux, g_main) > 0` on the core. If it is negative, the head's measured
  plan-worth gain came from fighting the objective and the arm is not the
  precedent I have been treating it as.
- **O5 (the decision):** O1 and O2 together ⇒ domination or starvation, and
  adding targets (an MTP-shaped chain) is indicated. `cos(g_main, g_emit) < −0.05`
  ⇒ conflict, MTP is contraindicated, and gradient projection comes first.

**O5 is the one that matters.** It is the only prediction here that changes what
gets built next. O1–O4 exist to make O5 readable.

## Risks and confounds recorded up front

- **`plast` and `main` are both token-position losses.** Tokens skip the core
  entirely; their gradient reaches it only through the coda's attention over slot
  positions. So a small `‖g_main‖` on the core is expected and is NOT by itself
  evidence of starvation — the norm ratio in O2 is a description, and the COSINE
  is what carries the diagnosis.
- **Batch 6 is small for a cosine.** The per-batch spread is reported next to the
  accumulated value, and a verdict whose spread crosses zero is reported as
  undecided rather than as its sign.
- **Weight 0 is not absence of an objective in the arm's history.** The v1a-2b
  arms ran `emit_weight=0` for their whole life, so their `emit*` direction is a
  counterfactual, not a term they were ever trained on.
- **This probe measures a gradient at a point, not a trajectory.** It cannot say
  which objective CAUSED the state the checkpoint is in.


---

# Results

Run 2026-08-27. Artifacts: `/home/wolfe/morph-scratch/split32` (4 batches) and
`split_confirm` (16 batches). Scorer:
[`score_split.py`](../../divergence/score_split.py).

## O0 — the gates, and what they caught

**Both sabotage runs exited 1**, so the gates bite: `--sabotage scale` fails
additivity, `--sabotage seed` fails determinism.

They then caught two real defects before any verdict was read.

1. **Self-cosines above 1.0** (1.0156, 1.0128). A cosine cannot exceed one, so
   this was pure accumulation error: the region vectors hold tens of millions of
   fp32 entries reaching 1e3, and a plain fp32 dot loses more than the 1e-2
   differences being resolved. Every reduction is float64 now, and self-cosine
   became exactly **1.000000** — which also settles a standing worry: this
   forward IS bit-reproducible, so the bag-mean atomics do not contaminate it.
2. **bf16 broke additivity.** The first run REFUSED on 4 of 6 checkpoints, with
   relative errors 2.5e-2 to 8.3e-2 against a 2e-2 tolerance. The same checkpoint
   in fp32 gives **2.9e-5, a 2600x improvement**, so the miss was entirely bf16
   rounding — amplified because the core's gradient is a small component of a
   large backward. The tolerance was NOT loosened to fit the data; the
   measurement was made accurate enough to meet it.

   **This generalises past this probe.** `_preclip_probe` runs in bf16, so every
   core-share number this campaign has published carries several percent of
   rounding noise. It does not overturn a 0.9-against-0.02 share claim. Nothing
   finer than a few percent should be read off those series.

Final panel: additivity 3.8e-5 to 1.4e-4, determinism 1.000000 everywhere.

## The checkpoint I called healthy was not

O1/O2/O5 were first read off `ctrl-s2` at step 3000. The fine probe puts that
seed's takeover at **step 2970** — 549 of 3500 steps over share 0.5, shipped rule
fires at 2970. The worklist already said seed 2 takes over near 3000 and I picked
it anyway. Re-read on `ctrl-s3`, which never takes over (1 transient step in
3500). Both readings are kept below, because the contrast between them is the
most informative thing in this experiment.

## Verdicts

| | prediction | result | |
| --- | --- | --- | --- |
| **O1** | `cos(g_main, g_emit) > −0.05` on the healthy control | 95 % CI **[−0.040, +0.033]** @3000, **[−0.036, +0.044]** @2000 (n=16) | **HELD**, and tightly |
| **O2** | `‖g_main‖/‖g_emit‖ < 0.25` | **0.579 / 0.844 / 0.822** at steps 1000/2000/3000 | **FAILED** |
| **O3** | emit share > 0.5 on the taken-over control | **0.677** | **HELD** |
| **O4** | `cos(g_mux, g_main) > 0` on the MUX arm | s2 **+0.072** CI [+0.008, +0.136]; s3 **+0.160** CI [+0.072, +0.248] | **HELD**, both seeds exclude zero |
| **O5** | the decision | neither conflict, nor domination, nor starvation | **hypothesis refuted** |

## O5 — what the split actually says

Healthy regime, clean seed `ctrl-s3`:

```
step 1000   ||g_main||/||g_emit|| 0.579   cos(main,emit) noisy, unreadable
step 2000                         0.844   cos  +0.004  95% CI [-0.036,+0.044]
step 3000                         0.822   cos  -0.003  95% CI [-0.040,+0.033]
```

The two objectives are **orthogonal to within ±0.04**, and `g_main` is
**58-84 % the size of `g_emit`**. The core is not starved, not dominated, and not
fought over. It receives a large gradient from the coda's objective, in a
direction unrelated to its own target — and the plan is still worth 0.015 nats.

**So gradient assignment is not the root in the healthy regime.** The gradient is
there. What is missing is HEADROOM: `‖g‖` is the local slope, and the achievable
gain through the slot was already measured at 0.019 nats, 0.070 with every token
masked. That kills three candidate fixes at once — taxing the token path
(nothing to recover), PCGrad/CAGrad (nothing to project at cos ≈ 0), and
GradNorm/MGDA (nothing to rebalance at ratio 0.8).

Gradient assignment DOES go pathological, but only inside the takeover:

```
ctrl-s2 @3000  core share 0.85   cos(main,emit) +0.664   norms 100x step 2000
ctrl-s1 @3000  core share 0.99   cos(main,emit) +0.621   emit share 0.677
```

Both gradients rotate into one direction and blow up together. That is the
signature of a shared expanding eigendirection, not of the objectives agreeing —
consistent with the `ρ(J_core) > 1` reading in
[the iterative-map note](../../../.agents/notes/implemented/architecture/2026-06-19-iterative-map-dynamics.md).
The high cosine is a symptom of the instability.

## O4 and the MUX arm

The head's gradient into the core is **positively aligned with the coda's
objective on both seeds**, with confidence intervals excluding zero. It is also
NEGATIVELY aligned with `plast` — the free token path's own objective — at
−0.070, CI [−0.135, −0.005] on seed 3 and the same sign but not significant on
seed 2. Read carefully, that is the head pushing the core AGAINST the shortcut,
which is the mechanism the arm was built for. Two seeds, one significant: a
hypothesis, not a result.

`‖g_mux‖` is 2.4x and 1.6x `‖g_main‖`, so the head is the largest real gradient
into the core on that arm.

## Updated hypothesis

The disease is not credit assignment in the multi-task sense. It is that the
coda's objective has almost no achievable gain through the slot, whatever
gradient it sends. The only lever the measurement leaves is a NEW objective with
its own headroom — which is what MUX is, and what an MTP-shaped chain would be a
better-formed version of.

**The next experiment must test headroom directly, not slope**: freeze everything
but the core, train on `ce_main` alone, and watch plan worth. If it saturates
near 0.02 the architecture bounds it; if it climbs, this is an optimization
failure and the MTP plan is the wrong fix. That experiment does not exist yet.

## Not verified

- Every per-batch cosine spread still crosses zero at 16 batches. The verdicts
  above rest on the mean's confidence interval, not on any single batch.
- The probe reads a gradient at a point. It says which fix the landscape admits,
  NOT which objective caused the state the checkpoint is in.
- `ctrl-s3` is one seed. O1's bound is not replicated on another clean seed.
