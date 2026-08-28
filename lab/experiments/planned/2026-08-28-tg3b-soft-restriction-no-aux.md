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

## Results (filled 2026-08-28 07:15). ALL FOUR PREDICTIONS HELD.

| arm | ce_main@3000 | @3500 | loop worth 3000/3500 | plan worth 3000/3500 | end core share |
|---|---|---|---|---|---|
| tg3b-s1 | 4.6767 | 4.5703 | 0.0342 / 0.0233 | 0.0407 / 0.0372 | 0.0009 held |
| tg3b-s2 | 4.6872 | 4.5918 | 0.0344 / 0.0385 | 0.0365 / 0.0354 | 0.0010 held |
| **mean** | **4.6820** | **4.5811** | 0.0343 / 0.0309 | **0.0386** | — |
| TG2 (reference) | 4.8763 / 4.7121 → **4.794** | — | 0.0276 / 0.0378 | 0.1232 / 0.0516 → **0.0874** | 0.0020 / 0.0035 |

**S1 HELD.** 0 of 2 fired; core shares 0.0009 / 0.0010, the campaign's lowest pair.

**S2 HELD, and by more than the decision rule's 0.10 threshold.** Mean ce_main@3000 4.6820
against TG2's 4.794 — **an improvement of 0.112 nats**, the largest single move this
campaign has produced without divergence. The deficit against the control band roughly
HALVES: TG2 sat 0.248–0.335 nats behind, TG3b sits **0.136–0.223** behind.

**S3 HELD — and this is the test of the 2026-08-28 confound correction.** Mean plan
worth@3000 fell from TG2's 0.0874 to **0.0386**, a 56% drop, when the ONLY change was
giving tokens a direct route to the previous span. The correction predicted exactly this:
plan worth tracks what the mask FORBIDS, not what the plan CONTAINS. Had plan worth held up
or risen, the correction was wrong and the round-1 verdict's original wording would have
been reinstated. It did not. **The correction stands, and it is now tested rather than
merely argued.**

**S4 HELD.** Loop worth 0.0342 / 0.0344 at step 3000 and 0.0233 / 0.0385 at 3500 — all
under 0.05, on a `bag_mean` arm where the column means what it has always meant. The loop
has now missed its line under every seed, objective, and mask variant tried.

## Verdict

**success** (4 of 4 predictions held). Per this file's own decision rule, S2 holding by more
than 0.10 nats means: **the hard restriction WAS too tight, the deficit IS attackable by
mask design, and the TG track's CE story is NOT closed.**

That REVERSES the expected-value assessment written into `lab/divergence/TG-WORKLIST.md` at
02:50, which called the track "close to exhausted" on the evidence then available (seed
doesn't matter, objectives don't matter, and TG3 had just died). The missing arm was the one
TG3 was supposed to be. Mask geometry is a live lever; seed and objective count are not.

## What this does NOT show

TG3b is still 0.136–0.223 nats behind the control band at matched step 3000. Halving a
deficit is not closing it. And the loop — the thing TUL is named for — still earns under
0.05 nats here, so the CE gain comes from giving TOKENS more context, not from the plan or
the loop doing more work. Read alongside the plan-content probe, whose first panel was
REFUSED by its own memorization gate.
