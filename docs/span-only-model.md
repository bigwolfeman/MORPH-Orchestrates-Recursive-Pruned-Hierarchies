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

## Where the line stands (2026-10-27, evening)

```
share:   side channel 0.002-0.011 nats  ->  series 0.12-0.14 nats (zeroing 0.20-0.25)   SOLVED by the series cut
serial:  K1-K6 +0.0015  ->  +0.003 with a one-span read                                 1/3 of the 0.01 bar
         read width 5 and 2-4 loop cells per span: still +0.003 at 5k (cells fill in pass 1)   width is NOT the limit
         training length: split read 5 at 160k steps (983M tokens): +0.0110 [+0.0104, +0.0116]   OVER THE BAR (2026-10-27)
         open: does the loop beat a flat span core with the same objective? flat twins, 5k pilot then 160k
compression: one 1024-d state per span as the ONLY path from the past (no pointer) still
             gives K1-K6 +0.0032 and CE 4.32 (0.12 behind a pointer-free model with a 64-token window)   NOT a wall at 5k
copy:    the pointer head is worth 0.31 (side channel) to 0.405 (series) nats; spans keep meaning, not exact tokens
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
├── A3. flat twins             BUILT 2026-10-27 (capinv.span_passes / span_core_layers): param-matched (6 layers once) and compute-matched (36 once); 5k pilot, then 160k
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
```

## Open questions

- Is +0.003 a ceiling of this chassis at 5k, or of the job? Neither: training length moves it to +0.011 (A2). Width (A) does not.
- Does the loop beat a flat span core with the same objective at equal parameters, and at equal compute (A3)?
- Does the curve keep rising past 160k, and how does it scale at the cloud's tokens per parameter? One seed so far.
- Does a span-only model need its own copy path, or does the gap shrink with scale?
- What is the CE gap to the token-rate references (3.70 to 3.72 at full width) at the cloud's tokens per parameter, not at 5k?

## Sources

- Memory: `span-loop-needs-a-future-job`, `span-index` (span-CALM, cloud budget, span savings literature review).
- TUL intent ("think once per span, decode cheaply"): [tul-design-history.md](tul-design-history.md).
- Loop contribution history: [tul-loop-contribution-history.md](tul-loop-contribution-history.md).
