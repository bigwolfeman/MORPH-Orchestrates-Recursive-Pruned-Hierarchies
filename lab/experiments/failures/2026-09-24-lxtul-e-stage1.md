# Planned: LXTUL-E Stage 1, a loop that carries an enumerated code, read in parallel

Status: failure

Date: 2026-09-24 (frozen before any Stage 1 GPU step beyond the builder's 30-step smoke).
Design and priors:
[`2026-09-23-provable-loop-contribution.md`](../../../.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md)
(its P-1..P-5 are carried here unchanged; P-1's threshold is now fixed by Stage 0). Gate:
[`2026-09-23-lxtul-e-stage0.md`](../successes/2026-09-23-lxtul-e-stage0.md), B0 = 0.1031.
Build: 017d472. Wolfe's go: 2026-09-23 ("lets try it").

## Question

If the slot loop carries one of K = 4 enumerated codes, re-added at a fixed size every pass,
and a committed parallel reader of the next span is trained on the exact mixture over the
four rollouts, does width survive training, and does the loop's depth get a job (integrating
the code into a separated per-code state) that the parallel read pays for?

## Hypothesis

From the Lean results: width pays only a committed reader (T3), a re-supplied code is
amplified by the passes as Σ λᵗ (T5, linear case), and a teacher-forced coda hedges (T3), so
its K-curve and its width gain stay flat.

## Arms

On the strict ruler recipe (`tul_slot_spandec_strict`: seq 1024, batch 6, seed 1,
norm_match, ramp 1000, fixed point 1.0), 5000 steps from scratch, at 017d472, one trainer at
a time on the 5090.

| arm | config | one factor |
|---|---|---|
| `lxtul-e1` | `tul_slot_spandec_strict_e1.yaml` | the ruler with the parallel head in place of the span decoder; K = 1, no code |
| `lxtul-e4` | `tul_slot_spandec_strict_e4.yaml` | e1 + K = 4 enumerated codes re-added every pass (r 0.1), 4 rollouts, exact mixture for the head and the coda |

Partner on disk: the ruler `slot-spandec-strict` @5000.

## Instrument

`lab/divergence/lxtul_e_stage1_score.py` on the 480 validation rows (terms in its
docstring): par CE and coda CE at forced depths 1..6; width gains (one code minus the
mixture); exit separation between codes and its depth-6 / depth-1 ratio; paired e4 vs e1 and
e4 vs ruler. Plus the runner's `core_depth_sweep` and `worth_profile` on both arms.

## Predictions (frozen; the note's, with P-1's threshold filled in)

- **P-1 (width survives).** e4's parallel mixture CE beats e1's parallel CE by at least
  max(0.01, 0.5 · B0) = **0.0515** nats per head token, paired CI clear of zero: **50 %.**
- **P-2 (the coda hedges).** e4's coda width gain (one code alone minus the 4-rollout read)
  is at most 0.005: **85 %.**
- **P-3 (width creates a depth job).** e4's parallel-read K1−K6 >= 0.02 with the CI clear
  of zero: **35 %.** e1's parallel-read K1−K6 <= 0.005: **60 %.**
- **P-4 (the teacher-forced bypass).** Coda token K1−K6 <= 0.005 on both arms: **80 %.**
- **P-5 (no price).** e4's coda CE under the 4-rollout read, paired against the ruler at
  depth 6, within +0.02: **55 %.**
- **Diagnostic, not scored:** e4's exit-separation ratio, depth 6 over depth 1 (linear
  theory 4.6 at gain 0.89).

## Verdict rules

From the note. What refutes the account: P-1 holds, P-3's first clause fails, and the
separation ratio is below 1.3 (the reader reads the one-pass separation; integration is not
a job). Or P-2 fails by more than 0.02 with P-1 failing (the teacher-forced coda is not the
hedging reader T3 treats it as). P-3's first clause holding is the loop-contribution
positive the design exists for; it is read on a TRAINING-ONLY head, and says nothing about
the deployed token path unless P-2 or P-4 also moves. No clause passes on a cosine.

## Method

e1 then e4, then the scorer and the sweeps. Artifacts: JSON and run logs in
[`../results/2026-09-24-lxtul-e-stage1/`](../results/2026-09-24-lxtul-e-stage1/).

## Results (filed 2026-09-24 04:10)

Artifacts: [`../results/2026-09-24-lxtul-e-stage1/`](../results/2026-09-24-lxtul-e-stage1/)
(`stage1_score.json`, both sweeps and worth profiles, `notul_pair.runlog.txt`, the run logs of
both arms, the scorer and the chain). Chain at a045b47. Both arms ran 5000 steps with exit
0 and no spike. Scorer self-check against each model's own forward: max |dev| 1.1e-6. 480
rows, 501,106 coda tokens, 486,031 head tokens, 490 blocks.

| clause | reading, 95 % CI | verdict |
|---|---|---|
| P-1 e1 par − e4 par mix @6 >= 0.0515 | +0.1138 [+0.1106, +0.1174] | held |
| P-2 e4 coda width gain (best code) @6 <= 0.005 | +0.0275 [+0.0261, +0.0289] | failed |
| P-3 e4 par K1−K6 >= 0.02 | +0.0180 [+0.0165, +0.0196] | failed (CI wholly below 0.02) |
| P-3 e1 par K1−K6 <= 0.005 | +0.0054 [+0.0047, +0.0061] | failed |
| P-4 coda K1−K6 <= 0.005, both arms | e4 +0.0020, e1 +0.0011 | held |
| P-5 e4 coda mix − ruler coda @6 <= +0.02 | −0.0056 [−0.0079, −0.0033] | held |
| diagnostic: e4 exit separation d6 / d1 | 1.99 abs (0.266 → 0.530), 1.67 rel | – |

By depth (forced 1..6), e4:

| depth | par mix | par width gain (best code) | coda mix | coda width gain | exit sep abs |
|---|---|---|---|---|---|
| 1 | 6.8058 | 0.259 | 4.3437 | 0.0211 | 0.266 |
| 2 | 6.7924 | 0.300 | 4.3422 | 0.0257 | 0.370 |
| 3 | 6.7891 | 0.312 | 4.3418 | 0.0261 | 0.424 |
| 6 | 6.7878 | 0.317 | 4.3418 | 0.0275 | 0.530 |

e1 over the same depths: par 6.9070 → 6.9016, coda 4.3563 → 4.3552.

Against notul (`plain-panel-norm-match` @5000 at depth 6, its training mean), on identical
tokens joined by `lab/divergence/lxtul_e_notul_pair.py`: the coda set (491,520 tokens) reads
notul 4.0750, ruler coda +0.2716, e1 coda +0.2796, e4 coda mix +0.2657. The head set
(476,712 next-span tokens) reads notul 4.073, e1 par 6.900, e4 par mix 6.786; at span offset
0, notul 3.484 and e4 par 4.115; at offsets 2 and higher, notul 4.10–4.17 and e4 par
6.69–7.08.

Compute: e1 14.6 layer passes per token at 11,762 tok/s; e4 44.6 at 6,157 tok/s.

## Verdict

Failure as written: three clauses of six held, and the one the design exists for (P-3,
width creates a depth job the parallel read pays for) missed its threshold by 0.002 with
its CI wholly below it. Neither refutation rule fires: the separation ratio is 1.99 (the
rule needs < 1.3), and P-2 failed with P-1 holding (the rule needs P-1 failing too).

What the numbers say:

- **Width survives training and the loop integrates the code.** The rollouts' exit states
  move apart with every pass (0.27 → 0.53), and both readers' width gains grow with depth
  (head 0.26 → 0.32, coda 0.021 → 0.028). This is the depth job T5 describes. Its price is
  small: 0.018 nats on the head's mixture, 0.002 on the coda, and three quarters of the
  head's share (0.0134) is in pass 2.
- **The loop adds almost nothing over a code bolted onto the frozen ruler.** Stage 0's head
  on the ruler's frozen cells reached 6.791 under the same mixture; e4, whose loop carries
  the code from pass 1, reaches 6.788 at depth 6 and 6.806 at depth 1. Width is worth ~0.11
  nats per span token either way (B0 0.103, e1 − e4 0.114).
- **The teacher-forced coda does not hedge (P-2's premise was wrong).** It reads each code's
  rollout differently: one code alone costs 0.0275 over the 4-rollout read. So the coda
  mixture beats the ruler by 0.0056 and e1 by 0.0134, at 3.1x the layer passes, and the
  deployed read is that same mixture, run as 4 rollouts.
- **The parallel reader stays 2.7 nats per token behind notul.** It is near notul only at
  span offset 0 (+0.63); from offset 2 on it is ~2.9 behind. The product cap (T3) is the
  dominant term (e1 par − the ruler's teacher-forced decoder: 2.42), and width closes 5 %
  of it.

Unverified: one seed per arm; 5000 steps; the r 0.1 code scale was not swept. The e4 coda
number is the deploy read's log loss by the chain rule of the mixture (the label-free
forward, `_enum_mixture_logprobs`, gives each position the Bayes-weighted mixture over
rollouts); `tests/test_tul_lxtul_e.py::test_coda_mixture_is_the_brute_force_per_span_logsumexp_and_the_deploy_read`
pins that equality on a small model, and it was not re-measured on the step-5000 checkpoint.

## Updated hypothesis

An enumerated code re-added every pass gives the loop a real integration job (separation
doubles with depth, width gains grow with it), but the job is worth 0.02 nats because the
reader that pays for width (the parallel head) is capped far below the token path, and the
reader that is the deployed path (the coda) pays only 0.002 for depth. The lever is not
more integration: it is a reader whose width value is large AND that is the deployed path.

## Addendum 2026-09-24 04:45: generation samples

`gen_samples.json` and `gen_samples.runlog.txt` (`scripts/tul_samples.py` at ce68156: 12
prompts, 256 tokens, seed 1234, top-k 50 at T 0.8 and greedy; e4 decodes through the
deploy read) and `par_samples.runlog.txt` (`lab/divergence/parallel_span_samples.py`,
greedy whole-span decodes of the parallel head, 2 rows × 4 spans). Real-text anchor at the
same length: rep4 0.025, distinct3 0.948.

| model | top-k rep4 | top-k distinct3 | greedy rep4 | greedy distinct3 |
|---|---|---|---|---|
| notul | 0.125 | 0.824 | 0.920 | 0.076 |
| ruler | 0.061 | 0.898 | 0.647 | 0.309 |
| e1 | 0.017 | 0.951 | 0.775 | 0.195 |
| e4 | 0.030 | 0.922 | 0.702 | 0.254 |

All four read as fluent-ish 5000-step web text under top-k; none is better by eye. Every
TUL arm repeats less than notul at both decodes (notul greedy loops on one sentence); this
is a diversity reading, not a quality one, and notul's CE is 0.27 lower. The parallel head
decodes the next span's first token or two plausibly (`S. official and`, `5 million`) and
then falls into `the the the` and `....`; e4's four rollouts differ only in those first
tokens and in punctuation.
