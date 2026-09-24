# Planned: LXTUL-E Stage 2, where the committed read points and who pays for width

Status: failure

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
[`../results/2026-09-24-lxtul-e-stage2/`](../results/2026-09-24-lxtul-e-stage2/).

## Results (filed 2026-09-24 14:20)

Artifacts: [`../results/2026-09-24-lxtul-e-stage2/`](../results/2026-09-24-lxtul-e-stage2/)
(`stage2_score.json`, sweeps, worth profiles, `gen_samples.json`, `notul_pair.runlog.txt`,
the run logs of both arms, the scorer and the chain). Chain at 40289e2. Both arms ran
5000 steps with exit 0 and no spike (e4j4 7,543 tok/s, e4probe 6,136). Scorer self-check
against each model's own forward: max |dev| under 1e-6. 480 rows, 501,106 coda tokens;
head tokens 486,031 (e4, e4probe) and 96,690 (e4j4, offsets 0–3).

| clause | reading, 95 % CI | verdict |
|---|---|---|
| J-1 j4 par K1−K6 (own tokens) >= 0.068 | +0.0238 [+0.0208, +0.0271]; e4 on the same tokens +0.0478 | failed |
| J-2 j4 coda K1−K6 >= 0.005 | +0.0017 [+0.0014, +0.0020] | failed |
| J-3 j4 coda mix − e4 coda mix @6 <= −0.003 | **+0.0668 [+0.0643, +0.0695]** | failed, opposite sign |
| J-4 j4 coda width gain @6 >= 0.0275 | +0.0238 [+0.0225, +0.0250] | failed |
| Q-1 probe coda mix − e4 coda mix @6 >= +0.005 | **−0.0120 [−0.0144, −0.0098]** | failed; the opposite sign (25 %) held |
| Q-2 probe coda width gain @6 >= 0.0275 | +0.0287 [+0.0271, +0.0303] | held on the point; the CI straddles |
| Q-3 probe coda K1−K6 >= 0.005 | +0.0011 [+0.0009, +0.0013] | failed |
| Q-4 probe par mix − e4 par mix @6 >= +0.05 (shared) | +0.1337 [+0.1296, +0.1381] | held |
| diagnostic: exit separation d6 / d1, abs | e4 1.99, j4 2.13, probe 1.33 | – |

More readings:

- Coda @6 against the ruler: e4 −0.0056, e4j4 **+0.0612** [+0.0585, +0.0638], e4probe
  **−0.0176** [−0.0198, −0.0154]. Against notul on identical tokens (491,520): e4 +0.2657,
  e4j4 +0.3330, e4probe +0.2542.
- j4 vs e4 par @6 on the shared tokens (offsets 0–3): −0.1310 [−0.1384, −0.1235]. j4's
  par width gain (best code minus the mixture) is 1.42 nats at depth 6 (e4 0.317): each
  code alone is a poor reader, so the four codes split the spans between them.
- Probe par K1−K6 +0.0038 (e4 +0.0180). Worth(zero) total: e4 0.248, j4 0.173, probe 0.213.
- Samples: top-k rep4 0.021 (j4) and 0.025 (probe), real text 0.025; greedy 0.76 and 0.78;
  prose reads like the Stage 1 arms.

## Verdict

Failure: neither J-1 nor J-2 held. By the verdict rule's second clause, neither arm moves
the coda's K1−K6 above 0.005 (0.0011–0.0020 on all three arms), so the loop's integration
job is real but the deployed reader does not pay for it, whoever shapes the loop. The next
move is the reader, not the objective.

What the numbers say:

- **The parallel head's gradient costs the deployed coda.** Cut it (the probe) and the coda
  improves by 0.012; the probe's 4-rollout coda is the best strict-TUL coda on this panel
  (0.0176 better than the ruler, 0.254 behind notul). Point it at the first 4 tokens and
  the coda loses 0.067: the committed read on the span's opening tokens pulls the exit
  state hard toward a four-way split of span openings, and the coda pays for it.
- **The head's gradient is what drives the loop's integration.** Rollout separation grows
  x1.99 (e4) and x2.13 (j4) with depth when the head trains the loop, x1.33 when it does
  not. The probe's committed read is 0.134 worse and its depth value falls to 0.004.
- **Focusing the head halved its depth value.** On offsets 0–3, j4 earns 0.024 from depth
  against e4's 0.048 on the same tokens, while its CE there is 0.131 better: the codes
  took over the job the passes did.
- **The deployed path gets width, not depth, in every arm.** The coda's width gain is
  0.024–0.029; its K1−K6 is 0.001–0.002 with or without the head's pull.

Unverified: one seed per arm; the probe's coda gain over e4 (0.012) is outside the ~0.004
seed floor measured on earlier arms, but the floor was not re-measured here; the coda
number is the 4-rollout mixture at 3.1x the ruler's layer passes, and the probe's best
single rollout is ~0.011 behind the ruler.

## Updated hypothesis

The loop earns depth only for a reader whose gradient trains it and that cannot get the
content elsewhere (the committed head). The coda reads the exit state as a one-pass
feature and takes width from the rollouts, but no objective on the loop has yet made the
coda's own prediction improve with passes. A reader-side change is next: the coda must
need something only later passes make.
