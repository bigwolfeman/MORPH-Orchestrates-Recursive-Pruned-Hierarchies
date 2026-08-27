# Experiment: which objective writes to the core, and do they fight?

Status: **planned. Probe built and unit-tested (8 tests); no checkpoint has been
run through it.** Predictions frozen 2026-08-27 before the first run.
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
