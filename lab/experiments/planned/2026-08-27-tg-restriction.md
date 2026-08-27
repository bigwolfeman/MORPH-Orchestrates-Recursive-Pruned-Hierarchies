# Planned: TG restriction — close the token shortcut, make the slot load-bearing

Status: planned
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
