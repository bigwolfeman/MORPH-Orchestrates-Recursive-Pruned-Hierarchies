# Planned: LXTUL-E Stage 2, where the committed read points and who pays for width

Status: planned

Date: 2026-09-24 10:07 (frozen before the build finished and before any Stage 2 GPU step).
Parent: [`../failures/2026-09-24-lxtul-e-stage1.md`](../failures/2026-09-24-lxtul-e-stage1.md).
Design note: [`2026-09-23-provable-loop-contribution.md`](../../../.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md).
Wolfe's go: 2026-09-24 ("Let's run those.").

## Question

Stage 1 trained the loop against two readers of the next span: the coda (one token at a
time, the deployed path) and the parallel head (the whole span at once, training only).
Width and depth paid the parallel head (0.114, 0.018) and barely paid the coda (0.0056
over the ruler, K1−K6 0.002). Two things in the data point at why:

1. The head's loss is mostly on tokens it cannot predict. Its CE is ~7.0 from span offset
   4 on (near unigram), and Stage 0's width value there is 0.05–0.13 against 0.31–0.39 at
   offsets 1–2. On offsets 0–3, e4's head already reads K1−K6 **0.0478** and width
   (e1 − e4) **0.2552**; over all 32 offsets it reads 0.018 and 0.114.
2. Nothing tells us whether the head's gradient helps or hurts the deployed coda.

Stage 2 asks: (a) does pointing the committed read at the span's first 4 tokens raise the
loop's depth value, on the head and on the coda? (b) with the head's gradient cut, does the
coda earn more or less from width and depth when it is the only reader that shapes the
loop?

## Hypothesis

(a) The 28 unpredictable offsets add gradient noise to z without a width or depth signal.
Removing them concentrates the head's pull on the positions where the codes matter, so the
loop's depth job on those positions grows, and some of it reaches the coda.
(b) The head's pull competes with the coda's. With it gone the coda shapes z alone. The
Stage 1 evidence is that the parallel head in place of the teacher-forced span decoder
cost the coda 0.0078 (e1 vs ruler), so a loop the coda alone shapes should not be worse
for the coda. Whether the coda then uses width or depth more is open.

## Arms

On e4 (`tul_slot_spandec_strict_e4`: the strict ruler recipe, K = 4 enumerated codes
re-added every pass at r 0.1, 4 rollouts, exact mixture), 5000 steps from scratch, seed
1, one trainer at a time on the 5090. Each changes ONE thing.

| arm | config | the one factor |
|---|---|---|
| `lxtul-e4j4` | `tul_slot_spandec_strict_e4j4.yaml` | `tul.spandec_parallel_span_cap: 4`: the head predicts only the first 4 tokens of the next span |
| `lxtul-e4probe` | `tul_slot_spandec_strict_e4probe.yaml` | `tul.spandec_parallel_detach: true`: the head trains on the DETACHED exit state, a probe that cannot shape the loop |

Reference on disk: `lxtul-e4` @5000 (Stage 1). Ruler: `slot-spandec-strict` @5000.

## Instrument

`lab/divergence/lxtul_e_stage2_score.py` (built for this stage; it imports the Stage 1
scorer's readings) on the same 480 validation rows, forced depths 1..6: par and coda
mixture CE, width gains (best single code minus the mixture), exit separation, K1−K6,
paired block-bootstrap CIs against e4. Par comparisons between arms with different head
token sets use the SHARED tokens only (e4 restricted to span offsets 0–3 for j4). Plus
`core_depth_sweep`, `worth_profile`, `lxtul_e_notul_pair.py`-style pairing to notul, and
generation samples, as Stage 1.

## Predictions

e4j4:

- **J-1 (the committed read's depth value grows).** j4's par K1−K6 on its own tokens
  (offsets 0–3) >= 0.068 (e4's 0.0478 + 0.02), CI clear of 0.0478: **30 %.**
- **J-2 (depth reaches the deployed path).** j4's coda K1−K6 >= 0.005 (e4 0.0020), CI
  clear of zero: **25 %.**
- **J-3 (the coda gains).** j4 coda mix − e4 coda mix @6 <= −0.003, CI clear of zero:
  **40 %.**
- **J-4 (width on the deployed path).** j4's coda width gain (best code) @6 >= 0.0275
  (e4's value): **50 %.**

e4probe:

- **Q-1 (the head's pull helped the coda).** probe coda mix − e4 coda mix @6 >= +0.005,
  CI clear of zero: **35 %.** (The opposite sign, probe better by >= 0.005: 25 %.)
- **Q-2 (the coda uses width when it pays alone).** probe coda width gain (best code) @6
  >= 0.0275: **40 %.**
- **Q-3 (depth on the deployed path).** probe coda K1−K6 >= 0.005, CI clear of zero:
  **15 %.**
- **Q-4 (the head's pull shaped what a committed reader finds).** probe par mix − e4 par
  mix @6 >= +0.05 on shared tokens: **70 %.**

Diagnostic, not scored: exit separation d6/d1 on both arms (e4: 1.99 abs, 1.67 rel).

## Verdict rules

Stage 2 is a success if J-1 or J-2 holds AND the arm that meets it is not worse than e4 on
the coda at depth 6 (arm − e4 coda mix CI upper < +0.005). J-2 or Q-3 holding is the
positive this line of work exists for: depth value on the deployed path. If neither arm
moves coda K1−K6 above 0.005 and the separation ratio stays near 2, the loop's integration
job is real but the deployed reader does not pay for it, whoever shapes it; the next move
is the reader, not the objective.

No clause passes on a cosine.

## Method

Build on branch `lxtul-e-stage2` (Opus subagent; orchestrator reviews and merges). Smoke
each config for 30 steps. Then e4j4, then e4probe, 5000 steps each, then the scorer,
sweeps, worth, notul pairing and samples. Artifacts: JSON and `.runlog.txt` files in
`../results/2026-09-24-lxtul-e-stage2/`.
