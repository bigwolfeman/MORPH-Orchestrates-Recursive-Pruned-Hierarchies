# Span-only model: vision, results, and the variants to probe

Status: live tracking doc, started 2026-10-26. Update it when a span-line run is filed. Numbers are copied from the filings named in
each row. The runs live in the Parcae testbed worktrees (`/home/wolfe/parcae-wt-<name>`, branches `lxtul/<name>`, local only). CE is
nats per token on the 480 minfan-5k rows unless a row says otherwise. K1-K6 = CE at loop depth 1 minus CE at depth 6 of the same model.

## The vision

Wolfe, 2026-10-24 and 2026-10-26:

- Spans matter in themselves, not only for training ms/step. A span (a sentence-sized unit, 4 to 32 tokens, cut at `.;!?`, newline
  and dashes) is the unit for very long agent memory, for thinking in whole sentences, and for fast decoding. Spans are "strictly
  more ideal than NTP".
- The span-only model keeps the prelude, the loop and the coda in span space, and decodes only its spans. The whole model gets
  span-level credit assignment.
- The span loop has to think about the future, not only summarize the past.
- The bar: K1-K6 clearly above the span controls, about 0.01 or more.

Why it matters for the cloud run: the budget is 6B to 60B tokens in 3 days on 4x5090 or 2xH100, a 3B model pruned to 1B. A token-rate
loop cannot reach 6B tokens at 3B. A loop that runs once per span costs a fraction of one. Some CE and some loop contribution are an
acceptable price for that saving.

Standing rules for this line: never lower backprop depth. No MoE core. No token-through-loop designs. No per-pass or pass-indexed
aux targets, no depth curriculum, no learned-sigma arms. One seed per span run (the control's K1-K6 seed spread is 0.0002).

## What a span-only model is, from first principles

```
span-only model
├── cut         the boundary rule makes spans; causal, so the generator cuts the same way
├── encode      tokens of span s -> one span vector z_s (pool of a light token encoder)
├── think       a looped core over the span sequence (Poisson depth, full BPTT); ALL depth lives here
├── forecast    the state of span s must hold what span s + 1 needs (the loop's future job)
├── render      a light decoder writes span s + 1's tokens from span s's state and its own earlier tokens
└── credit      the main CE reaches the loop ONLY through the states the decoder reads
                (any other path from the past to a token is a bypass, and the loop loses its share)
```

Loop contribution = share x serial benefit.

- Share is the part of the loss that has to go through the loop. A side channel (a token window that crosses span edges, a decoder
  that reads many span states, a pointer over every past token) gives the gradient a cheaper path, and the loop's share falls.
- Serial benefit is how much a second, third, sixth pass adds. A job that one pass can finish (pooling, a linear scan, a summary)
  earns no depth however much load it carries.

## Where the line stands (2026-10-28, 03:15)

```
share:   side channel 0.002-0.011 nats  ->  series 0.12-0.14 nats (zeroing 0.20-0.25)   SOLVED by the series cut
serial:  K1-K6 +0.0015  ->  +0.003 with a one-span read                                 1/3 of the 0.01 bar
         read width 5 and 2-4 loop cells per span: still +0.003 at 5k (cells fill in pass 1)   width is NOT the limit
         training length: split read 5 at 160k steps (983M tokens): +0.0110 [+0.0104, +0.0116]   OVER THE BAR (2026-10-27)
         flat twin at 160k (6 layers, one pass): ties the loop, +0.0006 [-0.0016, +0.0027], 1.38x faster   (expected at this stage, Wolfe)
compression: one 1024-d state per span as the ONLY path from the past (no pointer) still
             gives K1-K6 +0.0032 and CE 4.32 (0.12 behind a pointer-free model with a 64-token window)   NOT a wall at 5k
copy:    the pointer head is worth 0.31 (side channel) to 0.405 (series) nats; spans keep meaning, not exact tokens
latent:  a KL-priced code per span sits on the free-bits floor (1.0-1.6 nats); 0.07-0.10 judge bpb behind the series model (2026-10-28)
         a fixed-rate FSQ code is unpredictable (prior learns 0.6 of 11 nats/span): worse; feature-fed loop + residual: 0.025 bpb behind
```

## Results ledger

### Stage 1. Span-CALM: span-only with a VAE latent interface (2026-10-20)

Two stages. Stage 0 trains a span VAE. Stage 1 trains a span-only backbone to predict the next span's latent with an energy-score
head (no token CE). The span is then decoded from the predicted latent. Filed: `parcae-wt-spancalm`
`docs/experiments/failures/2026-10-20-parcae-span-calm-5k.md` (f3f28cd, correction f268fc0). The CE is an 8-sample estimate of an
implicit sampler, biased high, and is not comparable to an exact token CE.

| Arm | CE@1 | CE@6 | K1-K6 | Notes |
|---|---|---|---|---|
| frozen decoder, VAE rate ~597 nats/span (s1) | 8.9399 | 8.9224 | +0.0176 | depth repairs off-manifold samples |
| tuned decoder, rate ~597 (s1) | 4.5572 | 4.5548 | +0.0024 | null 4.98 |
| tuned decoder, rate-low (rlow-dec s1) | 4.4303 | 4.4272 | +0.0031 | round 2, not filed; raw `abq/spancalm2/eval` |
| co-trained decoder (sc3-cot s1) | 4.2205 | 4.2178 | +0.0027 | round 3, not filed; best span-only CE |
| head + co-train, rate-xlow (s1) | 4.2873 | 4.2864 | +0.0009 | round 4, not filed |

Reading: the interface failed. The backbone's part (null minus CE) was about 0.43 nats per token in every arm; the gains came from the
span-local decoder. A latent with ~600 nats per span is more than the backbone can predict. Token-level reference: gpt8 + pointer 3.692.

### Stage 2. Capacity inversion: a light token path under a looped span stack, side channel (2026-10-23 to 10-26)

`capinv_b`: 1 encoder + 3 decoder token layers at d512, window 64 (crosses span edges), a d1024 span stack at Parcae depth, decoder reads
the last 8 span states as K/V, pointer head. 116 ms/step (the 1+3 ruler with no span stack: 43 ms). Prereg
`parcae-wt-capinv docs/experiments/planned/2026-10-23-parcae-capacity-inversion-5k.md` (4e1fe0f); its results were never filed (raw
in-run CE in `abq/night1023/gpu_todo.log`).

| Arm (all side channel) | CE@6 | K1-K6 | Load on the span channel | Filing |
|---|---|---|---|---|
| capinv control, 1+3 (s1 / s2) | 3.8865 / 3.8863 | +0.0015 / +0.0013 | worth 0.0020-0.0051 over its ruler | not filed |
| 4 draft cells per span | 3.8867 | +0.0020 | draft zeroing +0.0029 | draft e259ca7, not supported |
| 2 draft cells | 3.8868 | +0.0016 | +0.0016 | same |
| 4 cells + next-token target | 3.8943 | +0.0021 | +0.0024 | same |
| summary forecast, no cells | 3.8967 | +0.0010 | n/a | same |
| pass chain, last-state read | 3.8856 | +0.0014 | chain zeroing +0.0028 | passchain 81a6286, not supported |
| pass chain, full read | 3.8858 | +0.0023 | +0.0046 | same |
| scan plan + 8 summaries | 3.8810 | +0.0011 | plan zeroing +0.0678 | scanplan b7a73a5, not supported |
| scan plan only | 3.8778 | +0.0015 | plan zeroing +0.0784 | same (prediction 2 supported) |
| control, no pointer | 4.1988 | +0.0004 | stack worth 0.0008 | pointer-free pair e259ca7 |

Reading: as a side channel the span stack carries almost nothing (0.002 to 0.011 nats), because the 64-token window and the 8-span
read do its job more cheaply. The scan plan carries load (0.078) but finishes in one pass. Without the pointer the model loses 0.31
nats, and the span stack recovers 0.0008 of it: far context is mostly copy content.

### Stage 3. Series: the span stack as the only cross-span path (2026-10-26)

`tok_scope: span`: a token attends only to the tokens of its own span. Cross-span information reaches it only through the span states of
completed spans, the scan plan and the pointer head. Block Transformer / MegaByte shape. Filed successes in `parcae-wt-series`: 002b1b0
(series), 171a660 (read-one-span).

| Arm | CE@6 | K1-K6 | Span-state / plan zeroing @6 | Span stack worth over series ruler | ms/step |
|---|---|---|---|---|---|
| series ruler, no span stack | 4.0434 | 0 | n/a | 0 | 30 |
| series, decoder reads 8 span states | 3.9183 | +0.0014 | read +0.2162 | 0.1251 | 74 |
| series cut alone (span window 3) | 3.9203 | +0.0018 | read +0.2103 | 0.1232 | 82 |
| series scan plan only | 3.9080 | +0.0032 | plan +0.2370 | 0.1354 | 114 |
| series scan plan + summaries | 3.9070 | +0.0026 | plan +0.1678, read +0.0107 | 0.1364 | 116 |
| **series, decoder reads 1 span state** | **3.9157** | **+0.0029** | read +0.1952 | | 74 |
| **same, no pointer (span-only cross-span path)** | **4.3206** | **+0.0032** | read +0.2269 | | 71 |

Paired (bootstrap 95 %): read 1 minus read 8, dK +0.0015 [+0.0012, +0.0018], dCE -0.0027. Read 1 minus series scan plan only, dK
-0.0003 [-0.0007, +0.0001]. No-pointer read 1 minus the pointer-free side-channel control, dK +0.0027 [+0.0024, +0.0031], dCE
+0.1219.

Reading: the series cut solves share. The decoder's multi-span read was a second bypass; a one-span read doubles K1-K6 and improves CE.
The scan's earlier lead was its narrow read. Every narrow-read arm sits near +0.003. Zeroing at depth 1 reads within 0.005 of depth 6:
the span stack's value is still present after one pass.

#### Stage 3b. Forecast width: loop cells and a split read (2026-10-27)

Read-one-span plus C loop cells per span (each enters as the span's summary plus a learned offset, loops with the span core, read only by
the next span), or one loop state read as 5 K/V rows (`read_split`). Filed failure `parcae-wt-series` e97b5b7
(`docs/experiments/failures/2026-10-26-parcae-series-span-cells-5k.md`). Paired, 480 rows, bootstrap 95 %.

| Arm | CE@6 | K1-K6 | decoder zeroing @6 | cells zeroing @6; its depth part | ms/step |
|---|---|---|---|---|---|
| read-one-span, seed 1 / seed 2 | 3.9157 / 3.9176 | +0.0029 / +0.0030 | +0.195 / +0.198 | n/a | 74 |
| **split read 5 (one state, 5 read rows)** | **3.9145** | **+0.0030** | +0.224 | rows 2-5: +0.167; **+0.0017** | **75** |
| 2 cells | 3.9178 | +0.0028 | +0.218 | +0.037; -0.0002 | 102 |
| 4 cells | 3.9184 | +0.0028 | +0.222 | +0.045; +0.0002 | 140 |
| 4 cells, no pointer | 4.3158 | +0.0030 | +0.250 | +0.047; -0.0002 | 136 |

Paired: read-one-span seed spread dCE@6 +0.0019 [+0.0002, +0.0038], K1-K6 0.0001. 4 cells minus split read 5: dCE@6 +0.0039
[+0.0023, +0.0054]. 4 cells no pointer minus read-one-span no pointer: dCE@6 -0.0048 [-0.0065, -0.0030].

Reading: cells raise share (the decoder leans on the read harder) and carry 0.04 nats, all of it present after pass 1; no serial
benefit, at 1.4-1.9x the step time. The split read is free, and its extra rows carry over half of the arm's K1-K6: the later passes'
work reaches the decoder through a wide read of one state. Generation samples (nucleus 0.95): no arm loops; span arms read gen-PPL
93-107 under the plain token loop vs 59 for plain itself, the same order as the 0.2-nat CE gap. Greedy loops in every pointer arm.

#### Stage 3c. Training length: split read 5 at 160k steps (2026-10-27)

One run, seed 1, 160k steps (983M tokens, 3 h 25 min at 81.8k tok/s, LR flat to 80k then linear cooldown). Filed success `parcae-wt-series`
2ccf2c1 (`docs/experiments/successes/2026-10-27-parcae-series-split-read-160k.md`). `val/k1_k6` on all 480 rows every 5k steps:

```
K1-K6  5-20k +0.0026 | 25-50k +0.0043 | 55-80k +0.0052 | cooldown 85-105k +0.0066 | 110-135k +0.0086 | 140-160k +0.0106
CE@6   4.07 (5k) -> 3.76 (80k) -> 3.3878 (160k)
final sweep: K1-K6 +0.0110 [+0.0104, +0.0116]; K2-K6 +0.0028; K3-K6 +0.0008; K4-K6 inside 0   (5k: +0.0030 [+0.0027, +0.0032])
```

Reading: the 5k +0.003 was a training-length reading. The span loop's contribution grows 3.7x and crosses the 0.01 bar with its whole interval,
and it grows faster while the LR cools (+0.0054 over the cooldown vs +0.0029 over the flat half). The loop uses about 4 of its mean 6 passes.
Caveats: one seed; K1-K6 compares the loop with itself at depth 1, so whether the passes buy CE over a flat span core trained at its own depth is
the flat-twin question (variant A3).

Flat twins at 5k (filed failure `parcae-wt-series` e2897f6): CE@6 loop 3.9145, flat 6 layers once 3.9153 (53 ms), flat 36 layers once 3.9130
(154 ms, 2.1x the loop at equal linear-layer FLOPs). The flat twin trained at one pass beats the loop scored at one pass by 0.0021 [0.0006,
0.0037]: two thirds of the loop's 5k K1-K6 is that handicap. The 160k flat twin decides whether the loop's +0.0110 is a CE gain.

Flat twin at 160k (filed failure `parcae-wt-series` cc291fd, `docs/experiments/failures/2026-10-27-parcae-series-split-read-flat-160k.md`):
the parameter-matched flat core (6 layers, one pass), same data order and init draws, 2 h 33 min at 113.2k tok/s.

```
paired, 480 rows   depth:    1        2        3        4        6        8
looped CE                 3.3988   3.3906   3.3885   3.3880   3.3878   3.3879
flat CE (one pass)        3.3883 at every depth
flat - loop              -0.0104  -0.0023  -0.0002  +0.0004  +0.0006 [-0.0016, +0.0027]  +0.0005
val CE@6 gap, mean over 16 readings: -0.0004 (constant LR half), -0.0004 (cooldown half); no trend
```

Reading: the two span cores tie on CE, and the flat one trains 1.38x faster. Wolfe 2026-10-27: this is expected behavior for the span loop
at this stage, not a verdict on its depth. One seed per arm. The compute-matched flat twin (36 layers, about 6.8 h) is written and held for
Wolfe's go (`abq/flat36-160k/run.sh`).

### Stage 4. Span latent: a KL-priced code per span, trained end to end (2026-10-27/28)

The series split read 5 model with each span's pooled feature replaced by a sampled code: posterior from the span's own tokens, prior from the
loop state of the span before, the decoder reads the span's own code (5 K/V rows), one loss (the negative ELBO, free bits 1 nat per span per
level). Three hierarchies (one code; one level per loop pass; a slow group code per 4 spans) x the decoder residual (the loop's state) on/off.
Filed failure `parcae-wt-latent` c001ef6 (`docs/experiments/failures/2026-10-27-parcae-span-latent-grid-5k.md`); design note
`.agents/notes/proposed/architecture/2026-10-27-span-latent-hierarchy.md`. 5k, seed 1, 480 rows and 480 prompts:

```
                      IWAE@6   rate (nats/span)       judge bpb, sampled p0.95   MAUVE
series reference      3.9145*  -                      1.895 [1.873, 1.919]      0.44
one code, residual    4.027    1.20                   1.971                     0.34
one code, code only   4.064    1.30                   1.965                     0.34
per-pass, residual    4.126    1.0-1.2 x 6 levels      1.991                     0.42
nested, residual      4.032    span 1.27, group 1.39  1.974                     0.36
* exact CE, not a bound          (REAL text 0.890 bpb; every latent arm trails the reference by 0.07-0.10 bpb, CIs exclude 0)
```

Reading: every code sits on the free-bits floor. The ELBO prices a code nat 1:1 against the decoder's free token path, and a code-fed loop sees
each past span through about 1 nat (`val/span_state_cos` 0.53-0.74 against 0.37 for the reference). The residual does not change sampled bpb
(+0.005 [-0.020, +0.030], main prediction not supported). Per-pass levels each buy about 0.74 nats for their 1-nat price (ELBO K1-K6 -0.07).
Stage 4b, the code interface (filed failure `parcae-wt-latent` ee82f74, `docs/experiments/failures/2026-10-28-parcae-span-code-fsq-feature-5k.md`;
Wolfe's codebook idea): a fixed-rate FSQ code (64k codes, at most 11.07 nats per span, exact per-dimension categorical prior on detached codes),
and a loop fed the pooled span features instead of codes.

```
                          bound@6          code cost/span   judge bpb, sampled   prompt_dep
series reference          3.9145 exact     -                1.895                0.055
feature-fed, residual     3.946 IWAE       1.01 (KL)        1.920 (+0.025 [0.001, 0.049] vs reference)   0.055
Gaussian code-fed, res.   4.027 IWAE       1.20 (KL)        1.971                0.040
FSQ, code only            4.293 exact      10.33 (code CE)  2.049 (+0.083 vs the Gaussian code only, main not supported)
```

Reading: the FSQ code is used hard by the mouth (code-off +0.87-0.95 nats per token) but the prior learns only 0.5-0.7 of its 11 nats per
span, so generation hands the mouth near-random codes. Feeding the loop the features opens the history channel (bound -0.08 with the
residual), and once the loop holds history the residual is worth -0.106 on the bound and -0.047 bpb. The feature-fed residual arm is the best
latent model, with the reference's prompt dependence. The missing piece is a code priced by its predictability (FSQ with the code CE into the
encoder at beta < 1, or Gaussian Quant); not yet planned.

### Context: token-resolution span loops (2026-10-23, not span-only)

Full-width token layers with a span loop beside them (`spanloop_local*`, span cap 8). CE@6 3.703 to 3.718, K1-K6 0.000 to 0.005. These
are 0.2 nats better on CE and cost about 12 of 14 layer-equivalents per token at full width. Preregs only
(`parcae-wt-capinv` 9fe025a, `parcae-wt-spancap` 4fbb9cf), raw numbers in `abq/night1023/gpu_todo.log`. They are not the span-only
line; they bound what a token path at full width buys.

## Laws found so far

1. Every cheaper path around the loop takes its share: the crossing token window, the multi-span decoder read, the pointer for copy
   content. Remove a bypass and the loop's share and depth go up (series: 12x the worth; one-span read: 2x the K1-K6).
2. Load is not depth. The scan plan carries 0.24 nats in series and its value is complete at pass 1. A job earns depth only if one pass
   cannot finish it.
3. One span state is enough to carry the past at 5k. Compression costs 0.12 nats against a 64-token window, not a collapse.
4. Exact copy does not fit in a pooled span state. A span-only model without a copy path pays 0.3 to 0.4 nats on web text.
5. Two-stage interfaces (VAE latent, frozen decoder) lose more than they give at this scale. End-to-end series training works.
6. Width is not depth. Extra loop positions per span (cells) take share and fill in one pass; a wider read of one loop state is free
   and is where the later passes' work shows up. The span loop's +0.003 is set by its job, not by its width (2026-10-27).
7. Training length moves the span loop where width does not: 5k K1-K6 +0.003 grows to +0.011 by 160k steps, fastest in the LR cooldown.
   A 5k reading of a span loop is a floor, not its value (2026-10-27).
8. A KL-priced span code sits on whatever free-bits floor you give it when the decoder has a token path inside the span. Hierarchy and a
   decoder residual do not move it (stage 4, 2026-10-28).
9. A span code trained only for the decoder holds what the context cannot predict. Feed the loop the span's features for history; the code
   is only worth its predictable part (stage 4b, 2026-10-28).

## Variants to probe (planning, 2026-10-26)

Ordered by what each one tests. Each needs a prereg before any run.

```
serial benefit (the open factor)
├── A. forecast width          RUN 2026-10-27, not supported (stage 3b): K1-K6 +0.003 in every arm; split read 5 is the new base
│   ├── 4 loop cells per span, read only by the next span      capinv_b_series_r1_cells4          +0.0028, 140 ms
│   ├── 2 loop cells                                            capinv_b_series_r1_cells2          +0.0028, 102 ms
│   ├── read width 5 from ONE loop state (wider mem_proj)       capinv_b_series_r1_split5          +0.0030, 75 ms, best CE
│   └── 4 cells, no pointer (span-only twin)                    capinv_b_series_r1_cells4_nopointer +0.0030, CE -0.005
├── A2. training length        RUN 2026-10-27, SUPPORTED: K1-K6 +0.0110 [+0.0104, +0.0116] at 160k (stage 3c)
├── A3. flat twins             5k RUN 2026-10-27, not supported: all three span cores tie (flat - loop +0.0008 [-0.0007, +0.0025]; 36-layer flat -0.0015,
│                              154 ms); the loop's 5k K1-K6 is mostly a depth-1 handicap. 160k param-matched: RUN, not supported, ties
│                              (+0.0006 [-0.0016, +0.0027], 1.38x faster; expected at this stage). 160k compute-matched (36 layers): written, held for Wolfe's go (6.8 h)
├── B. variable cell count     Wolfe's idea (2026-10-05): parked; cells fill in one pass, so more cells at inference has nothing to add yet
├── C. training signal         one-document 4096 rows (4x spans per row); span cap 8 (more span positions per step)
├── D. thinner renderer        decoder 3 -> 1 layer: the span state must hold more of the next span
└── E. span-level main loss    a whole-next-span decoder as THE loss of a span-only model (not an aux term)
copy (law 4)
├── F. span-level copy         a pointer over the past spans' tokens chosen by the span state, in place of the row-wide pointer
└── G. pointer-free scale      does the 0.4-nat copy gap close with tokens/param? (horizon test at matched tokens/param)
speed and decode
├── H. cost per token          layer-equivalents of each arm (`lxtul/flop_tally.py`); series read-one-span is 74 ms vs 116 for capinv
└── I. cached span decode      series decodes a span from fixed span states: the span stack runs once per span at decode
    J. DITTO decode phase       RUN 2026-10-27, SUPPORTED (`parcae-wt-series` 5f4fba9): 1000 steps at 0.1x LR on both 160k models, greedy 4-gram
                                repeat 0.72 -> 0.60, longest copy 82 -> 52 tokens, CE +0.06/+0.07 vs matched controls; still loops, ratio under-converged
latent interface (stage 4)
├── K. KL-priced span code      RUN 2026-10-28, not supported: every code on the free-bits floor, 0.07-0.10 bpb behind the reference
├── L. fixed-rate FSQ code and feature-fed loop   RUN 2026-10-28, not supported: FSQ +0.083 bpb; feature-fed + residual 0.025 behind the reference
└── M. predictability-priced code   FSQ with the code CE into the encoder (beta < 1) or Gaussian Quant, on the feature-fed residual base; not planned
```

## Open questions

- Is +0.003 a ceiling of this chassis at 5k, or of the job? Neither: training length moves it to +0.011 (A2). Width (A) does not.
- Does the loop beat a flat span core with the same objective? At equal parameters, a tie at 160k (A3; expected at this stage). At
  equal compute: the 36-layer twin at 160k, held.
- Does the curve keep rising past 160k, and how does it scale at the cloud's tokens per parameter? One seed so far.
- Does a span-only model need its own copy path, or does the gap shrink with scale?
- What is the CE gap to the token-rate references (3.70 to 3.72 at full width) at the cloud's tokens per parameter, not at 5k?

## Sources

- Memory: `span-loop-needs-a-future-job`, `span-index` (span-CALM, cloud budget, span savings literature review).
- TUL intent ("think once per span, decode cheaply"): [tul-design-history.md](tul-design-history.md).
- Loop contribution history: [tul-loop-contribution-history.md](tul-loop-contribution-history.md).
