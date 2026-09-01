# Agent Note: Slot-channel recovery — the ladder to close TUL's 0.357-nat gap

Status: proposed

Date: 2026-09-01. Follows the 20k head-to-head
([../../../../lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md](../../../../lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md)):
TUL on the winner recipe loses the token axis by 0.357 nats (last-5 mean 3.846
vs 3.489) and the gap widens with training. Binding left TUL default-off;
this note is the recovery program.

## Problem

Under TG-restrict, cross-span information must flow through slots. The slot
write is a bag-mean: measured slot states have effective rank 1.7–4.8 in a
1024-d space, pairwise cosine +0.39 to +0.71
(lab/experiments/failures/2026-08-24-tul-takeover-cure.md). A mean of N
embeddings concentrates by construction, so the channel destroys
distinguishability before training sees it — the same class of defect JPmHC
fixed in the HC mixer by making the map isometric BY CONSTRUCTION (Cayley)
instead of hoping training finds the invariant. Separately, TUL-spec (2x2,
docs/ablation-ledger.md) showed the only gradient that writes span CONTENT
into slots is the emit/plast aux pair, which is also the takeover fuel — so
the content-writing signal we have is the one we cannot afford.

Unknown: how much of the 0.357 is (a) information the mask deletes that slots
fail to carry, (b) capacity/params spent on slot machinery, (c) optimization
tax. The ladder measures (a) first, then attacks the write.

## Proposal

The experiment ladder. Run in order; each rung gates the next. Every rung gets its own frozen prereg
in `lab/experiments/planned/` before launch (this note is the program, not
the prereg). All trained arms use the winner recipe (retention off, cap 0,
carry none) with AdEMAMix horizons scaled to run length
(`t_alpha`, `t_beta3` ≈ run length; the 20k pair ran 4500-step horizons —
mis-scheduled for 83% of the run). Flat LR stays (base.yaml recipe choice).

### E1 — Mask-surgery decomposition (eval-only, no training)

- **Question.** What does the TUL visibility mask alone cost a model that
  never adapted to it?
- **Hypothesis.** The mask is the dominant term: masked-noTUL degradation
  ≥ 0.25 of the 0.357 gap. Prediction to freeze at prereg: 60%.
- **Method.** Load `notul-20k/step_20000` on the eager path (the depth sweep
  already does this). Build span boundaries on each eval batch with
  `BoundaryRule.cut`. Feed a pairwise `tg_allow` mask through the attention
  forward (`attention.py:805` already accepts `tg_allow`). Two variants:
  - E1a same-span-only (harsh floor);
  - E1b same-span + one carrier position per prior span (the boundary token
    stands in for the slot — the fair analogue of TUL's read).
  Score CE overall AND stratified by within-span offset (first tokens of a
  span carry the cross-span dependence; mirror the stratified re-score in
  lab/tul/arms-result.md).
- **Pseudocode.**
  ```
  model = build(notul_bg0c0, use_kernels=false); load(step_20000)
  for batch in eval_set:
      spans = BoundaryRule.cut(batch.ids)              # same rule as TUL
      allow_a = same_span_pair_mask(spans) & causal    # [B,L,L]
      carriers = last_token_of_each_span(spans)
      allow_b = allow_a | attends_to(carriers) 
      for name, allow in {none: None, a: allow_a, b: allow_b}:
          ce[name] += forward(batch, tg_allow=allow).ce_per_offset
  report ce, ce_by_offset, delta_vs_none
  ```
- **Expected conclusions.** Δ(E1b) ≈ 0.25–0.35 ⇒ the write channel is the
  whole problem; E2/E3 justified. Δ(E1b) ≤ 0.10 ⇒ the gap is mostly
  mechanism/optimization tax; promote E4 (distill) and demote E3.
- **Cost.** Eval-only on an existing checkpoint. ~1 GPU-hour + harness code.

### E2 — Bound-superposition seed: static rank check (no training)

- **Question.** Does HRR-style binding fix the rank collapse by construction?
- **Hypothesis.** `seed_bound = (1/sqrt(N)) * sum_k R_k @ e_{t_k}` with fixed
  random orthogonal R_k (VSA binding, Plate HRR) lifts seed effective rank
  toward min(N, d) and drops pairwise cosine below 0.1. Prediction: 80% —
  this is near-theorem, the check guards implementation error.
- **Method.** No model. Take real spans from the loader, compute bag-mean
  seeds vs bound seeds from the SAME embedding table (tul-20k checkpoint's),
  run the existing rank/cosine probe on both populations.
- **Pseudocode.**
  ```
  E = load_embeddings(tul-20k/step_20000)
  R = [random_orthogonal(d, seed=k) for k in range(span_cap)]   # frozen
  for span in sample_spans(loader, n=2000):
      s_bag[i]   = mean(E[span])
      s_bound[i] = sum(R[k] @ E[span[k]]) / sqrt(len(span))
  report eff_rank(s_bag), eff_rank(s_bound), pairwise_cos(both)
  ```
- **Expected conclusions.** Pass ⇒ E3 launches. Fail ⇒ the binding
  implementation is wrong (fix before any training) — the theory does not
  fail quietly here.
- **Cost.** CPU-scale, < 1 hour total.

### E3 — Bound-seed arm (trained, the Cayley-analogue rung)

- **Question.** Does an information-preserving write recover CE?
- **Hypothesis.** With the seed injective-by-construction, the coda can read
  span content from slots and the loop has distinguishable states to
  iterate. Predictions to freeze: recovers ≥ 0.10 of the 0.357 gap (40%);
  trained slot-state effective rank ≥ 3x the tul-20k reference (65%); no
  takeover under the TG2-style objective (70%).
- **Known prior art to cite in the prereg:** slot_seed mode changes alone
  were CE-neutral (TG4a: 0.003 nats) and one caused a takeover (TG4b) —
  but neither fixed rank; the rank gate (E2, and re-measured trained) is
  what separates this arm from those.
- **Method.** New `tul.slot_seed: bound` mode in `tul.py`: per-offset frozen
  random orthogonal R_k applied before the span sum (Cayley-parameterized
  LEARNED R_k is a follow-up arm, not this one — fixed rotations first,
  zero new trained params, zero new stability surface). Config
  `tul_g0c0` + the mode, 20k steps, panel flags, scaled horizons. Twin
  reference: rerun `tul_g0c0` unchanged at scaled horizons so the seed
  effect is not confounded with the recipe fix.
- **Pseudocode (the write, replacing the bag-mean):**
  ```
  # in TULSlots.span_seed — today: seed = E_slot + mean_k(e_k)
  seed = E_slot + (1/sqrt(N)) * sum_k rotate(e_k, R[k])   # R frozen buffers
  ```
- **Expected conclusions.** CE recovers ⇒ the write was the bottleneck;
  learned-Cayley R_k and read-side decoding become the follow-ups. Rank
  recovers but CE does not ⇒ capacity or read is binding ⇒ E5 (matrix
  slots / R3). Takeover reappears ⇒ file with forensics; binding changed
  the loop's input geometry.
- **Cost.** 2 x ~4.5 h (arm + horizon-matched twin).

### E4 — Teacher-distill across the mask (bounded content gradient)

- **Question.** Can we train slots to carry exactly the masked information
  without the aux-loss takeover fuel?
- **Hypothesis.** Teacher = same weights, unrestricted visibility; student =
  TG-restrict path; `L += lambda * D(h_student_mid, sg(h_teacher_mid))` at
  token positions recovers ≥ 0.15 nats on span-first tokens. Prediction to
  freeze: 45%. Gradients are bounded by the stopgrad teacher; this does NOT
  regress the latent onto embeddings (the banned pattern) and is the
  mechanism already sketched in
  [2026-08-18-tul-teacher-distill.md](2026-08-18-tul-teacher-distill.md).
- **Method.** Continued training from `tul-20k/step_20000` (5k steps) before
  any from-scratch commitment; teacher forward on a half-frequency schedule
  to cap the ~2x compute. Mid-coda layer target per the distill note's
  Phase 0 probe.
- **Pseudocode.**
  ```
  for step in continue(tul-20k, 5000):
      h_s = forward_tg_restricted(batch)               # student, grads on
      if step % 2 == 0:
          with no_grad(): h_t = forward_unrestricted(batch)   # teacher
          loss = ce(h_s) + lam * mse(mid(h_s), mid(h_t))[token_positions]
      else: loss = ce(h_s)
  ```
- **Expected conclusions.** Works ⇒ the content-writing objective exists
  without takeover fuel; combine with E3's write in one arm. Null ⇒ the
  student cannot match a teacher whose information it cannot see — evidence
  the write itself (E3/E5) is the only lever.
- **Cost.** ~2.5 h continued run.

### E5 — Matrix slots (gated; the R3 on-ramp)

Only if E1 says information-cost AND E3 recovers rank but not CE: the slot
state becomes an outer-product memory `S = sum_t k_t v_t^T` (capacity d x r,
fast-weights form), routed writes per
[../feature/2026-08-31-raven-memory-arms.md](../feature/2026-08-31-raven-memory-arms.md).
Not specified further here — R3's note owns it, behind the contraction gate.

## Alternatives considered

- **Re-add GLA for absolute CE** — rejected: BG0C0 beat every GLA variant on
  absolute CE, not only depth-earning (loop-killer bisect, filed).
- **Re-enable emit/plast aux losses with a stronger cap** — rejected for the
  first rungs: they are the proven takeover fuel (TG-round arms); E4 is the
  bounded replacement for the same gradient.
- **Learned Cayley R_k first** — deferred in favor of fixed random
  orthogonals: identical invariant, zero new trained parameters, and it
  separates "binding helps" from "training the binding helps".
- **Chase the K3–K6 deep-composition flatness** — out of scope by standing
  veto; this program targets the TUL-vs-noTUL gap only.
- **LR decay in the recipe fix** — excluded: flat LR is the deliberate
  base.yaml recipe; only the AdEMAMix horizons scale with run length.

## Acceptance criteria

- E1 filed with the decomposition number and per-offset strata; it names
  which lane (write vs optimization) the ladder follows.
- E2 rank gate passes before E3 spends GPU-days.
- E3/E4 each recover their frozen fraction of the gap on the horizon-matched
  twin comparison, with no div-guard abort and takeover watch clean.
- Program-level: a TUL arm within 0.15 nats of the horizon-matched noTUL
  baseline at 20k, or a filed chain of failures that names the binding
  constraint and hands off to R3.

## Risks

- E1 harness: `tg_allow` plumbing exists but has only been exercised under
  `tg_restrict=True` builds; the noTUL eager forward may need a small shim.
  Mitigate: byte-identity check (allow=None == no-arg forward) before any
  masked number is read.
- E1b's carrier-token proxy is not exactly a slot read; report E1a/E1b as a
  bracket, not a point.
- Binding changes the loop input geometry; TG4b showed seed changes can
  reawaken the takeover on a base that was clean. Takeover telemetry
  (core share, block gain) stays on for every trained arm.
- Sweep noise floor ±0.01 nats; all CE claims quoted against it.
- n=1 runs: MORPH run comparisons decorrelate fast; prefer within-run and
  matched-twin readouts over cross-run absolutes.
