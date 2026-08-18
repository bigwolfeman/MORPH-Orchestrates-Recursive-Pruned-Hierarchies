# TUL arms — the first complete comparison (2026-08-18)

A1 finished. Two pre-registered questions can now be answered, one cannot, and one
metric points the other way. All four are below.

Runs: `tul-a0-acap1` (278 min) and `tul-a1-acap1` (177 min), both 20000 steps, batch 14,
seq 1024, one epoch, `ademamix_alpha_cap=1.0`, `token_state_dropout=0.15` (spec).
Queue `ignore/tul_logs/run_tul_arms2.sh`, commit `4650cb1`. wandb project `morph-tul`.

## The numbers

| arm | cap | final val CE | ppl | note |
|---|---|---|---|---|
| A0 (stored, first pass) | 3.5 | 3.2736 | 26.41 | reference only — NOT the comparison |
| **A0c — dense baseline** | 1.0 | **3.2805** | 26.59 | the number A1c is read against |
| A3 — compute floor | 3.5 | 3.2407 | 25.55 | see the caveat below |
| **A1c — the method** | 1.0 | **3.2243** | 25.89 | `val/ce_tokens_final`, the comparable metric |

A1c is quoted on `val/ce_tokens`, which is CE over token positions only (ordinary +
`t_last`). That is the metric §4 defines as comparable to a baseline's token CE; A1's
`val/loss` is the weighted double label and is not comparable to A0's. Both readings
agree here: `val/loss_final` is 3.2273, `val/ce_tokens_final` 3.2243.

* **A1c beats A0c by 0.0562 nats** at 10.86 layer passes per token (measured,
  `val/layer_passes_per_token_final`).
* The cap cost the baseline 0.0069 nats (A0c 3.2805 against the stored A0 3.2736).

## 1. The cap prevented the takeover — it did not merely delay it

This was the stated risk: the cap was verified to 4800 steps and these runs are 20000.
A1c logged 200 gradient-share points and **the core share crossed 0.5 exactly once**
(peak 0.8647, a single point, the transient shape from RCA §21), ending at 0.0380 with
`train/grad_norm` 0.8612. No `[ABORT]`. Held over 4× longer than the evidence for it.

## 2. The "Works" gate cannot be evaluated, and that is the honest state

Spec §7.3 sets it as **`plan_nats > A1r spread`**. Measured `val/plan_nats_final` =
**+0.0270** — positive, so removing the slots from the coda's sequence *does* cost the
model something, i.e. the coda is using the plan. But **A1r never completed** — it
aborted at step 3240 — so there is no spread, and +0.0270 cannot be called larger than a
number nobody has measured. The gate is unevaluated, not passed.

What it costs to close: one replicate of `tul-a1-acap1` at a second seed, ~3 h.

## 3. The plan loses to the trivial channel at the job it exists for

`val/first_tok_counterfactual_final` = **−0.1196**, defined `ce_plast − ce_emit`, where
positive means the plan helps. So:

* predicting a span's first token **from the previous token**: CE **2.890**
* predicting it **from the slot**: CE **3.010**

The plan is 0.12 nats *worse* than just reading `t_last` at the first token of a span —
the one prediction the latent plan is built to make. It is not a contradiction with
§2: `plan_nats` is averaged over all span tokens and is small and positive, while this
is the first token alone. Read together they say the slot contributes a little on
average and is beaten by the cheap channel where it should be strongest.

## 4. Beating A0 is not the gate anyway

Spec §7.3, last line: "Beating A0 on `val/ppl_tokens` is NOT a gate at this scale (§1)."
The 0.0562 nats is real and measured, and it is not what the method is being judged on.

## Caveats

* **A3 ran at cap 3.5**, so "A1c clears the A3 floor by 0.0164" carries exactly the
  two-variable problem the A0 re-run was done to remove. An A3 at cap 1.0 (~1 h 40 m,
  it has no core) would fix it.
* **n = 1 per arm.** Every number here is a single run. The 2026-08-17 lesson was that
  single runs of a bimodal process measure trajectory, not method. These arms are not
  bimodal in the same way — both finished cleanly — but 0.0562 nats has no error bar.
* **Generation was not read.** §7.3 also gates on rep4@512 ≤ A0's and no span-length
  collapse. `val/span_mean_span_final` is 19.88 with `span_cap_frac` 0.2809, which is in
  range, but rep4 and distinct-3 have not been computed.
* A0 logs no `layer_passes_per_token`, so the compute ratio against A1c's 10.86 is not
  measured here, only A1c's own figure.
