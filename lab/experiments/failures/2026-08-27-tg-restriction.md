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

| arm | ce_main@3500 | loop worth 3000→3500 | plan worth 3000→3500 | takeover |
|---|---|---|---|---|
| tg1-s1 | 4.7154 | +0.0019 → +0.0078 | +0.111 → +0.142 | held (end share 0.024) |
| tg1-s2 | DIVERGED, abort @2920 | — | — | TOOK OVER @1258 (end share 0.9986, gain 2.05, r2 0.98) |
| tg2-s1 | 4.7607 | +0.0276 → +0.0363 | +0.123 → +0.155 | held (end share **0.0020**) |
| tg2-s2 | 4.6224 | +0.0378 → +0.0237 | +0.052 → +0.033 | held (end share **0.0035**) |

Control band (muxconf-ctrl s4/5/6 @3000): ce_main 4.459–4.546, loop −0.0002..+0.0042,
plan 0.0124–0.0164, takeover fires 3 of 4 control seeds by ~3000.

- **P1 HELD.** tg1-s1 ce_main 4.7154 ≤ 4.85. The restricted model survives at
  +0.17..+0.27 nats over the band with tokens blind past their own span.
- **P2 FAILED.** tg1-s1 loop worth +0.0078 < 0.05. (TG2's 0.024–0.036 is 3–4× the
  control and the campaign's largest ever, but P2 was written for TG1 and 0.05 was
  the line; no seed reached it.)
- **P3 HELD.** 0 of 2 TG2 seeds fire; end shares 0.0020/0.0035 are the LOWEST core
  shares measured in this campaign, with block gain ≈ 1.0 — with the emit/plast
  gradients removed there is no expanding eigendirection at all. tg1-s2 (aux ON)
  took over on the classic mechanism, completing the contrast within one panel.
- **P4 HELD** on the only matched pair: |4.7607 − 4.7154| = 0.0453 ≤ 0.05 (seed 2
  has no matched pair — tg1-s2 diverged).

## Verdict

failure (P2 — the pre-registered loop-worth line was missed; 3 of 4 predictions held).
The RESTRICTION works: plan worth rose an order of magnitude (0.012–0.016 → up to
0.155) — the slot channel is load-bearing for the first time in the campaign — and
removing the aux losses under it eliminates the takeover entirely (P3) at no CE cost
(P4). The LOOP is still not earning 0.05 nats. TG2's loop worth (0.024–0.036, 3–4×
control) is suggestive but sub-threshold and seed-inconsistent in direction
(s1 rising, s2 falling 3000→3500).

## Updated hypothesis

The token-shortcut hypothesis is half right: closing the shortcut makes the PLAN
load-bearing and (with single-objective training) removes the takeover, but loop
depth on the slot still buys < 0.05 nats at 3500 steps. Either the loop needs the
longer run the decision rule prescribes (loop worth was still rising in tg2-s1), or
refining an already-contextualized span summary is a task the loop cannot add much
to at this scale. Next per the decision rule: ONE long TG2 run (≥ 10k steps), not
more seeds; discriminator = loop worth trajectory.
