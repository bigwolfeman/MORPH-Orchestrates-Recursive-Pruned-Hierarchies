# Planned: TG3b — ask the soft-restriction question from a base that can survive it

Status: planned
Config: `morph/configs/tul_tg3b.yaml`   Spec: ../../../docs/tul-tg-spec.md §6
Prior: ../failures/2026-08-27-tg-restriction.md, ./2026-08-28-tg-round2-seed-and-softness.md

## Why this arm exists

TG3 could not answer its own question. It was built to ask whether the hard restriction is
simply TOO TIGHT to carry context at this scale — i.e. it targets the 0.17–0.42 nat ce_main
DEFICIT against the control band, which is the number standing between TUL and the
baseline. But TG3 is `tul_tg1`-based, so it inherits the plast/emit objectives that the O5
objective-split result and the TG1/TG2 panel both name as the takeover's fuel.

Measured 2026-08-28: **tg3-s1 TOOK OVER at step 842** — end core share 0.9986, block
backward gain 1.953 at r² 0.97 (an expansive core map, the `ρ(J_core) > 1` mechanism from
`.agents/notes/implemented/architecture/2026-06-19-iterative-map-dynamics.md`) — validation
rose 1.659 nats from its step-750 minimum, and the divergence guard aborted the run at step
2040 with no step-3000 checkpoint. The arm died of a cause unrelated to the question asked.

The takeover record is consistent across the whole campaign:

| base | aux losses | arms | takeovers |
|---|---|---|---|
| tul_tg1 | ON | tg1 s1/s2, tg3 s1 | tg1-s2 @1258, tg3-s1 @842 |
| tul_tg2 | OFF | tg2 s1/s2, tg4a s1/s2 | **0 of 4** (shares 0.0020 / 0.0035 / 0.0011 / 0.0010) |

**Softening the mask did not rescue the takeover. Removing the aux losses did.** So the
soft-restriction question has to be asked from the `tul_tg2` base.

Disclosure: this arm is my own judgment call as task master, not something the user asked
for. It replaces GPU time that would otherwise idle after the queue drains, and it is
chained AFTER the plan-content probes so it cannot delay the result that redirects
everything. TG4b is left in the queue untouched.

## A bonus that TG3 could not have delivered

TG3b is a `bag_mean` arm. Tonight's TG4a result established that loop worth is comparable
only WITHIN one `slot_seed`, so the loop column means on TG3b exactly what it meant on TG2
and on the control. The 0.05-nat loop line becomes scoreable again, which it is not on
TG4a/TG4b.

## Predictions (frozen 2026-08-28 02:55, before this arm has run one step)

Metric is **ce_main** from `slot_path_worth.py` at step 3000. Control band 4.4586–4.5459.
TG2's own seeds: ce_main@3000 4.8763 / 4.7121 (mean **4.794**); plan worth 0.1232 / 0.0516
(mean **0.0874**); loop worth 0.0276 / 0.0378.

**S1 (the aux-loss story holds).** 0 of 2 TG3b seeds fire the takeover rule. TG3b is
tul_tg2-based, and every tul_tg2-based seed so far has held. A divergence here refutes the
"aux losses are the fuel" account and is the single most interesting way this arm can fail.

**S2 (softening recovers CE).** Mean ce_main@3000 over TG3b's two seeds is BELOW TG2's mean
of 4.794. Predicted on the MEAN, not seed-matched: MORPH runs decorrelate at a fixed seed
and the observed within-arm spread this campaign is 0.036–0.16 nats, so a seed-matched
comparison at n=1 is not readable.

**S3 (the confound correction's real test).** Mean plan worth@3000 over TG3b's two seeds is
BELOW TG2's mean of 0.0874. This is prediction T2 re-asked on a base that can reach step
3000. If plan worth is largely a function of what the mask FORBIDS rather than what the
plan CONTAINS, then giving tokens a direct route to the previous span MUST lower it.

**S4 (the loop still misses its line).** Neither seed reaches loop worth ≥ 0.05 on ce_main
at step 3000 or 3500, read from the plain `no-loop` column — which is meaningful here
because TG3b is a `bag_mean` arm.

## Decision rule

- **S3 FAILS** → the confound correction of 2026-08-28 is WRONG. Plan worth tracks content
  after all, the round-1 verdict's original wording was right, and it gets reinstated in
  `../failures/2026-08-27-tg-restriction.md` and `docs/ablation-ledger.md` with this arm
  cited. I would rather find this than defend the correction.
- **S1 FAILS** (a TG3b seed diverges) → the aux-loss account of the takeover is incomplete;
  mask softness is itself takeover-relevant. That reopens a question we thought closed and
  outranks everything else in the worklist.
- **S2 HOLDS by more than 0.10 nats** → the hard restriction WAS too tight, the deficit is
  attackable by mask design, and the TG track's CE story is NOT closed.
- **S2 FAILS** → mask softness is not the lever either. Combined with tonight's finding
  that the SEED is not the lever, the TG track's CE story IS closed, and A7 (FlexAttention,
  cashing the measured 10.96× sparsity) is the only remaining item worth GPU time.

## Method

Two seeds (1, 2), 3500 steps, batch 6, `use_kernels=false`, `ademamix_alpha_cap=3.5`,
`grad_probe_every=1`, ckpt every 500 — identical to every other arm in this campaign so the
control band stays comparable. Sequential on the 5090 (UPS). Worth passes at steps 3000 and
3500. Takeover from `score_arms.py`, rule unchanged.

Chained to start only after: the round-2 queue drains, the three plan-content probes run,
and the tg4a-s1 worth backfill completes. If the plan-content probe reports EMPTY, read that
result FIRST — it may retire S3 as a question worth asking.

NOT controlled: nothing new. TG3b differs from TG2 by the mask alone, which is the whole
point and is what makes the comparison clean — unlike TG3, which differed from TG4a/TG4b in
both mask softness and objective count.
