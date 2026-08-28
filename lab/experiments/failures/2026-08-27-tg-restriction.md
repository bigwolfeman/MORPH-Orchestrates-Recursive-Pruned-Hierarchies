# Planned: TG restriction — close the token shortcut, make the slot load-bearing

Status: failure (P2; P1,P3,P4 held)
Spec: ../../../docs/tul-tg-spec.md   Paper: arXiv 2512.25026 (Thought Gestalt)
Written and committed BEFORE the implementation existed or any arm ran.

## Question

Does restricting token attention to the current span — leaving prior context
reachable only through slot positions (TG's architecture, adapted to TUL) — make
the slot latent and the core loop load-bearing, at parity-ish CE?

## Hypothesis

The plan is empty because it is optional: full causal attention over past tokens is
a cheaper path than the slot. Every objective-side fix failed (aux weights, MUX,
token tax, warmup, SIGReg) because none removed the alternative. Removing the
alternative architecturally forces the MAIN CE loss to route context through the
slot, which (a) fills the plan, (b) gives the core loop a job, and (c) removes the
need for the emit objective — the takeover's fuel (objective-split O5).

## Predictions (frozen — do not edit during runs)

Control band (already banked, tul_a1 seeds 4/5/6 @3500): ce_main 4.459–4.546,
loop worth −0.0002..+0.0042, plan worth 0.0124–0.0164, takeover fires 3 of 4
control seeds by ~step 3000 under the standing rule.

- **P1 (survival):** TG1 ce_main at step 3500 ≤ 4.85 for at least one seed
  (≤ +0.30 nats over the control band's upper edge). The restricted model pays a
  learning-to-use-the-channel cost at this budget; more than 0.6 nats over band
  (> 5.15) on BOTH seeds means the channel cannot carry the context at this scale
  and TG3 (soft restriction) is the next rung.
- **P2 (the point):** TG1 loop worth ≥ 0.05 nats on every surviving seed
  (control: ≤ 0.004; current best-known: 0.009). Plan-off worth will be enormous
  BY CONSTRUCTION under the restriction — report it, but it decides NOTHING.
- **P3 (takeover fuel):** TG2 (plast_weight=0, emit_weight=0) fires the standing
  takeover rule on 0 of 2 seeds.
- **P4 (TG's recipe suffices):** TG2 ce_main within 0.05 nats of TG1's per matched
  seed — the aux losses add nothing once the architecture forces the channel.

## Decision rule

- P1 AND P2 hold → thesis alive; fund one long run (≥ 10k steps) before any wider sweep.
- P1 holds, P2 fails → the restriction works but the LOOP still adds nothing;
  that is a separate, honest negative for the loop and must be filed as such.
- P1 fails on both seeds (> 5.15) → run TG3 once before any verdict.
- Anything ambiguous → the next rung is ONE longer run, not more seeds (n=1 run
  comparisons are noisy; within-run measurements decide).

## Method

Base `tul_a1`, plus `tul.tg_restrict=true`. 3500 steps, batch 6, seq 1024,
`training.ademamix_alpha_cap=3.5`, `model.use_kernels=false`, `eval_every=250`,
`ckpt_every=500`, `gen_every=0`, grad probe cadence as in the 0827 arm panel
(fine enough for the takeover rule's 20-sample refusal). Seeds {1, 2} per arm,
sequential on the 5090 (UPS). Arms: TG1, TG2. TG3 only per the decision rule.
Scoring: `slot_path_worth` for plan/loop worth; `score_arms.py::fires` for the
takeover rule; ce_main from the eval lines. wandb: full config incl. tg_restrict.

Known confound, accepted and recorded: TG arms do not build the compressed-branch
compressor/indexer (param delta reported at build). A "slot-comp branch without the
window restriction" control would isolate it; not run initially (waste), listed as
a follow-up if P1/P2 read out ambiguous.

## Results (filled 2026-08-27, runs tg1 s1/s2 + tg2 s1/s2, artifacts in ../results/2026-08-27-tg-restriction/)

**METRIC CORRECTION 2026-08-28.** The first fill of this table read loop/plan worth off
the `loss` column of the worth JSONs. The campaign's convention — and the metric the
control band and P2's 0.05 line were written against — is **`ce_main`**. Both columns
are shown below; `ce_main` is the one that scores. For the TG2 arms the two columns are
identical by construction (`plast_weight=emit_weight=0` ⇒ the weighted loss IS ce_main),
so only the TG1 numbers moved. Corrected numbers, all at ce_main:

| arm | ce_main@3000 | ce_main@3500 | loop worth 3000→3500 | plan worth 3000→3500 | takeover |
|---|---|---|---|---|---|
| tg1-s1 | 4.8104 | 4.7154 | +0.0009 → +0.0060 | +0.0637 → +0.0870 | held (end share 0.024) |
| tg1-s2 | DIVERGED, abort @2920 | — | — | — | TOOK OVER @1258 (end share 0.9986, gain 2.05, r2 0.98) |
| tg2-s1 | 4.8763 | 4.7607 | +0.0276 → +0.0363 | +0.1232 → +0.1545 | held (end share **0.0020**) |
| tg2-s2 | 4.7121 | 4.6224 | +0.0378 → +0.0237 | +0.0516 → +0.0334 | held (end share **0.0035**) |

Superseded (wrong metric, `loss` column): tg1-s1 loop +0.0019→+0.0078, plan +0.111→+0.142;
tg2 rows unchanged.

Control band (muxconf-ctrl s4/5/6 @3000): ce_main 4.459–4.546, loop −0.0002..+0.0042,
plan 0.0124–0.0164, takeover fires 3 of 4 control seeds by ~3000.

- **P1 HELD.** tg1-s1 ce_main 4.7154 ≤ 4.85. The restricted model survives at
  +0.17..+0.27 nats over the band with tokens blind past their own span.
- **P2 FAILED.** tg1-s1 loop worth +0.0060 < 0.05 (ce_main). TG2's 0.024–0.036 is
  6–9× the control band's upper edge and the campaign's largest measured loop worth,
  but no seed reached the 0.05 line.
- **P3 HELD.** 0 of 2 TG2 seeds fire; end shares 0.0020/0.0035 are the LOWEST core
  shares measured in this campaign, with block gain ≈ 1.0 — with the emit/plast
  gradients removed there is no expanding eigendirection at all. tg1-s2 (aux ON)
  took over on the classic mechanism, completing the contrast within one panel.
- **P4 HELD** on the only matched pair: |4.7607 − 4.7154| = 0.0453 ≤ 0.05 (seed 2
  has no matched pair — tg1-s2 diverged).

## Verdict

failure (P2 — the pre-registered loop-worth line was missed; 3 of 4 predictions held).
Plan worth rose from the control band's 0.0124–0.0164 to 0.033–0.155, and removing the
aux losses under the restriction eliminates the takeover entirely (P3). The LOOP is
still not earning 0.05 nats. TG2's loop worth (0.024–0.036) is suggestive but
sub-threshold and seed-inconsistent in direction (s1 rising, s2 falling 3000→3500).

**CONFOUND CORRECTION 2026-08-28 — the plan-worth rise is NOT evidence the plan holds
more.** This verdict originally read "The RESTRICTION moves the plan … 2× to 10×". That
inference does not survive inspection of what the ablation removes. `no-plan` zeroes
what `prefix_project` writes into the shared sequence. Under `tg_restrict` a token's
ONLY route to any earlier span is that slot path, so zeroing it removes ALL cross-span
information. The UNRESTRICTED control keeps full causal attention and recovers the same
information directly from earlier tokens, so its plan worth is low **by construction**,
whatever its plan contains. A restricted arm must therefore show a higher plan worth
than the control even if the two plans carry byte-identical content.

What the number does support: within the restricted family, the plan path is load-bearing
for the coda — removing it costs 0.033–0.155 nats and nothing else recovers that. What it
does NOT support: any claim about the plan's information CONTENT relative to the control.
Only a direct read of `z` separates those, which is what
`lab/divergence/plan_content_probe.py` was built to do
(pre-registration: ../planned/2026-08-28-plan-content.md). Treat the content question as
OPEN until that probe reports.

This is the second confound of one family found in this campaign: **the ablation's
FALLBACK differs between the things being compared.** The other is the slot-seed
confound recorded below.

**The number that matters most, and that P1 was too permissive to catch.** At MATCHED
step 3000 the restriction COSTS 0.17–0.42 nats of ce_main against the control band
(TG arms 4.7121–4.8763 vs control 4.4586–4.5459). Our 3000-step rung is 18.8M tokens
(3500 = 21.9M), which sits INSIDE the 12M–50M token range where TG demonstrates its
advantage. So at TG's own data scale we measure a large loss PENALTY where TG reports
a gain — a sign disagreement, not a magnitude one. P1's +0.30-nat survival line was
written to ask "does it survive", and it cannot distinguish "survives and is on track"
from "survives and is going the wrong way".

## Updated hypothesis

The token-shortcut hypothesis is half right: closing the shortcut makes the PLAN
load-bearing and (with single-objective training) removes the takeover, but loop
depth on the slot still buys < 0.05 nats at 3500 steps. Either the loop needs the
longer run the decision rule prescribes (loop worth was still rising in tg2-s1), or
refining an already-contextualized span summary is a task the loop cannot add much
to at this scale. Next per the decision rule: ONE long TG2 run (≥ 10k steps), not
more seeds; discriminator = loop worth trajectory.

## Post-hoc: what the TG paper's own numbers say (added 2026-08-28, full paper read)

Converted from the paper's Appendix B tables (PPL → nats, ln ratio vs matched GPT-2):

| sweep | TG's advantage (nats) |
|---|---|
| data scaling, N≈85M: 12M / 20M / 30M / 50M tokens | 0.0279 / 0.0266 / 0.0362 / 0.0339 |
| param scaling, D=50M: 0.34M / 1.3M / 5.4M / 21.3M params | 0.0702 / **0.1387 / 0.1148 / 0.0720** |
| vs the SIMPLE "GPT-2 + sentence-boundary tokens" baseline, 50M | 0.0213 |

Three facts that bound what this method can pay us:

1. **The whole prize is ~0.03 nats** at TG's own 85M configuration, and 0.021 nats over
   a boundary-token baseline that costs nothing to build. Our restriction currently
   COSTS 0.17–0.42 nats at matched step 3000 — 6–14× the entire prize.
2. **TG's advantage DECLINES with model size in their own sweep**: 0.139 → 0.115 →
   0.072 nats as params go 1.3M → 5.4M → 21.3M. We are at 282M, 13× past their largest
   point. Their own trend extrapolates to ≲0.03 nats at our scale.
3. **The fitted scaling exponents are the same** (α ≈ 0.152 TG vs 0.149 GPT-2); the paper
   says the gain is "primarily due to an intercept shift." An intercept shift does not
   compound with training, so a longer single-pass run cannot convert our deficit.
   CAVEAT, stated honestly: TG's points are CONVERGED models (multi-epoch on a fixed
   12–50M-token subset, early stopping); ours are single-pass and far from converged.
   Fact 3 excludes "the advantage compounds", NOT "we are undertrained".

**Ablation table (Table 1, 30M tokens, 85.6M params) — what it buys us:**

| variant | PPL | what it means for TUL |
|---|---|---|
| TG baseline | 29.8 | — |
| Detach sentence reps at memory write | **35.0** | the mechanism, by 10×. Gradient flow through the memory chain. TUL has this (slots in-sequence, not detached). |
| In-context (prefix) memory instead of cross-attn | 30.2 | **TUL's exact architecture, costs 0.4 PPL.** We do NOT need per-layer cross-attention. |
| Self→Cross layers | 29.4 | +34% params for −0.4 PPL. Not worth it. |
| No stream curriculum | 30.5 | backprop-depth control |
| Max sentence length 32 (vs 64) | 30.4 | we are AT 32 |
| Sentence rep from last layer (vs mid layer 7) | 30.3 | minor |
| No context seeding | 30.2 | we do not do this |
| No EOS down-weighting | 30.4 | TG2 already does this (weights 0) |

Sum of all our remaining deviations ≈ 1.5–2 PPL ≈ 0.05–0.07 nats — it does NOT explain
our 0.17–0.42 nat deficit. No TG ablation lifts the token restriction while keeping the
memory, so the paper cannot tell us what the restriction costs on its own.

## Post-hoc: the pooling measurement that DOES explain it

`lab/divergence/pooling_probe.py` on `tg2-s1/step_3500.pt` (artifact: `pooling-tg2-s1.json`):

    eff_rank(slot inputs, centred) = 27.62 over 342 slots
    mean deviation 0.3027 vs ||E_slot|| 0.2378 -> signal/constant = 1.273
    slope of log(dev) vs log(L) = -0.470 (r2 0.922)   [-0.5 = plain-mean pooling law]

    span len      n   mean dev
     4-5         36     0.5162
    24-32       119     0.2102

**The plain-mean pooling law is confirmed at r²=0.92.** Our slot seed loses 2.5× of its
signal from short spans to long ones, against a shared constant of comparable size. TG's
sentence vector is a TAP — `m_t = W_sent · H^(ℓs)_{i_EOS}`, the EOS position's hidden
state at mid layer 7 — which does not dilute with length at all.

That asymmetry explains the one ablation we look worst on: TG GAINS by raising sentence
length 32→64, while our pooling law says the same change would LOSE us signal. We cannot
copy that knob until the seed stops being a mean.

## Consequences for the next arm

- **Do not run 10k single-pass steps.** Facts 1–3 above say the prize is ~0.03 nats and
  does not compound; the deficit is 6–14× that.
- **Next arm is provenance (TG4):** replace `E_slot + mean_j embed(t_j)` with TG's own
  `W_sent · h(boundary token)` tapped from a mid prelude layer, on top of TG2. This is
  the largest un-tested deviation and the only one with a measured mechanism behind it.
- **Do not raise `span_cap` to 64** until the seed is a tap — our own pooling law says it
  hurts us where it helped TG.
- **The strategic caveat:** TG has NO loop. Its entire measured effect is ~0.03 nats from
  sentence memory plus gradient flow. Even a perfect TG replication does not deliver the
  ~0.1 nats the TUL LOOP thesis needs. TG is evidence for the PLAN, not for the LOOP —
  and TG2 has already banked the takeover result (0/2). Its plan-worth rise over the
  control is confounded by the restriction itself — see the CONFOUND CORRECTION in the
  Verdict — so it is not a banked result and the plan-content probe decides it.
