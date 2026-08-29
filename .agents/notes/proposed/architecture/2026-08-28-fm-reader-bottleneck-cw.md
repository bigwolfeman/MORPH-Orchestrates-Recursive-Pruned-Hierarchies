# Agent Note: The FM plan bottleneck is the reader — force reliance with the compaction window

Status: proposed

## Problem

FM1 (run `h4s01ngk`, filed `lab/experiments/failures/2026-08-28-tul-fm1-live-arm.md`)
ended the "can the plan know the future" question: co-trained with SIGReg, the flow
planner retrieves the true next-span target at top-1 0.66–0.77 within-row (chance
0.02, copy floor ~0.20) at every eval. P1's information-cap explanation is dead —
the cap was target geometry (pairwise cos 0.63 on the frozen backbone, 0.00 after
250 steps of SIGReg), not missing information.

And the coda still pays exactly zero nats for it: `val/plan_worth_shuffle` =
0.0000 ± 0.0003 at all 17 evals. The bottleneck is the reader, not the writer.
Token CE is fully satisfied by the token context while the tokens are still in
the sequence; `token_state_dropout=0.15` is a probabilistic tax on a structural
problem (the CW spec said this before FM existed).

## Proposal

Train FM1 under the compaction window: `tul.coda_token_cut=576` on the FM1 arm
(`tul_fm1_cw.yaml`). The coda then sees only the last 448 tokens plus every slot;
content before the cut has no path to the loss except through a plan. A 70%-
accurate plan cannot be worth zero under that regime — either it pays, or
prefix-reading itself is broken and the run exposes it. This is Wolfe's
"last spans as tokens + previous latent z's" design, at the screened cut value.

One code fix ships with it: `_forward_tul`'s gather branch must mask slot-position
labels to −100 before scoring (the eval screen `tul_forward_cw_arms` already does
this; the training path did not — unpatched, CW training silently reinstates the
emit loss at weight 1.0 and double-counts every span's first token).

## Alternatives considered

- **Raise `token_state_dropout` (0.15 → 0.5+).** Rejected: still a probabilistic
  tax; FM1 showed 0.15 produces literally zero reliance, and the CW spec's screen
  already measured the structural version cleanly (CW1 > CW2, CI excludes 0).
- **Seed 2 of plain FM1.** Rejected: every load-bearing number was far from its
  threshold in both directions; a second seed cannot change the verdict.
- **Literal "last 2 spans" cut (C ≈ 984).** Rejected for the first run: ~40
  scored tokens per row starves the CE signal at 4500 steps. C=576 is the
  screened precedent with 448 scored tokens. Push harder only if worth fires.
- **JEPA-style latent loss on the coda.** Rejected here: regressing onto slot
  state is the banned pattern (LCM, CoCoMix, BT §4.2 — see CLAUDE.md).

## Acceptance criteria

Pre-registered in `lab/experiments/planned/2026-08-28-tul-fm1-cw.md`: the arm
earns continuation iff `val/plan_worth_shuffle` ≥ 0.01 nats sustained (≥3
consecutive evals) — the same gate FM1 sat at 0.0000 on. `val/ce_tokens` under
CW is scored on tokens ≥ cut only and is NOT comparable to A3/FM1 numbers; the
within-run ablations (worth_zero / worth_shuffle / copy_gap) are the metrics.

## Risks

- CE trains on 448/1024 positions per row — slower learning per step; 4500 steps
  may undershoot. Mitigation: the gate is an ablation delta, not an absolute CE.
- The gathered coda skips the §5 half-weighting (`layout=None` path, plain CE).
  FM1's `plast_weight=1.0` makes this a no-op for THIS arm; any CW arm with
  non-default weights inherits a silent objective change. Recorded here so it is
  a known sharp edge, not a surprise.
- `plan_worth` under CW recomputes the coda per ablation — eval cost triples on
  those batches. Accepted: eval is 20 batches every 250 steps.

Related: [2026-08-28-tul-fm-arc](2026-08-28-tul-fm-arc.md) (the arc this decides
the next step of), `docs/tul-compaction-window-spec.md` (mechanism + screen).

---

## Addendum 2026-08-28 (post FM1-CW + oracle probe + research sweep): the principled fix

FM1-CW failed its gate (`failures/2026-08-28-tul-fm1-cw.md`) and the oracle probe
(`successes/2026-08-28-oracle-prefix-probe.md`) showed even PERFECT content at the
prefix is worth 0.0000 nats — the content channel's entire signal is an
is-something-there energy cue (+0.0003, identical for shuffled plans and scaled
noise). Three research passes (repo docs, vlt record, LeVJEPA paper) converge:

1. **The failure family is posterior collapse, not attention sinking.** He 2019:
   the collapsed optimum (latent ignored) is stable from initialization because
   the decoder's cheap channel out-gradients the latent pathway; loss curves
   cannot show it. The attention-sink hypothesis was measured and REJECTED for
   the related takeover symptom (2026-08-24: core attention diffuse, sink was a
   consequence of state collapse) — and the FM1-CW register phenomenon shows the
   coda attends slots fine (0.81 nats of position value); attention is not the
   blocked resource. The read DECODING has no gradient reason to exist.
2. **Every mechanism we tried removes alternatives; none trains the reader.**
   Dropout (tax), CW (deletion), TG (attention rerouting) — all inflate position
   reliance, none created content reliance. The one mechanism that EVER produced
   decodable slot content is direct slot supervision: the emit CE (2x2 result:
   65.1% specificity with aux on vs 0.1–0.6% off). FM1 turned it off.
3. **The detach starves the read path.** TG paper Table 1: detaching the
   memory-write gradient costs 10x PPL. FM1's z is fully detached AND unread —
   consistent, since nothing on either side of W_prefix carries a gradient that
   rewards decoding.
4. **CoCoMix's shape is the prescription: loss alone near-null, insertion alone
   near-null, BOTH together win.** FM1 has insertion without a read loss.

**Proposal FM2 = FM1 + the emit CE turned back on** (`emit_weight: 0.5`,
`plast_weight: 0.5` — the spec §5 weights). At the slot position the only
content is `W_prefix(z)` (+E_slot); the emit label is the next span's first
token; minimizing it REQUIRES learning to decode the plan. Gradients stop at
W_prefix/E_slot/coda (z stays detached, no core loop exists), so the takeover
fuel finding (TG P3: emit gradients detonated the CORE LOOP via BPTT) does not
apply — the organ is absent. The reader learns to decode y-like vectors; the
planner independently learns to emit y-like vectors; they meet in the space
SIGReg shaped. Watch `val/first_tok_counterfactual`: it has been NEGATIVE at
every checkpoint ever measured (prev token beats slot); FM2 succeeding means it
crosses zero, and `worth_shuffle` finally moves.

Alternatives weighed for this addendum: per-layer cross-attention injection
(Optimus wins with it, but Block Transformer Fig 3f measured it 0.18 nats WORSE
than prefix at BPE scale, and our prefix already IS layer-visible attendable KV —
the Optimus "losing baseline" was add-once-to-input, which we do not do);
un-detaching the planner (restores an iterated-map gradient through the 6-step
Euler ladder — the exact disease class we removed); train-time dropout sweep
(deferred: the record shows dropout exposes content only when content is already
being read — sequence it AFTER the reader exists).
