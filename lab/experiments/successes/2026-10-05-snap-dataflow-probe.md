# Planned: where does the information go in snap? A stage-by-stage data-flow probe

Status: success

Date: 2026-10-05 13:27 CDT, before the probe is built. Wolfe (2026-10-05): "the loop should be driving almost
all the contribution I feel like. These small contributions are so bizarre to me. I feel
like we are missing something that will be obvious when looking at the data moving through
the model."

## Question

Snap (`lxtul_snap` 5k) and LXTUL (5k seed 1) earn K1-K6 of +0.012 / +0.022 and their cells
are worth about 0.18 nats, against a cross-span budget of 0.40+. Follow one span's
information through every stage and find where it is lost, ignored or bypassed.

## Stages and readings (snap and LXTUL, same rows)

1. **Seed pool**: each cell's attention over its span's prelude tokens: entropy, the share on
   the boundary token and on the span's first token, the token types it picks.
2. **Loop, per pass**: relative change ||h_t - h_(t-1)|| / ||h_t||; cosine to pass 1; router
   winner switches between passes; cell spread before each reset.
3. **Loop, cross-slot reads**: per core layer, the attention mass a slot's cells put on
   earlier slots' cells versus their own slot's cells.
4. **Write**: norm of the winner's written vector and of the pseudo tokens against the coda's
   token-position states; the learned gate g; pseudo vertex mass and which tokens are picked
   (and whether they are copied later).
5. **Coda reads**: per coda layer and head, the attention mass a token puts on slot positions
   (winner vs pseudo vs zero positions) against its own span's tokens, by slot distance; and
   the value-weighted output norm from slot positions vs tokens.
6. **Depth at the read**: the same coda readings at forced depth 1 vs 6.
7. **Worth split**: zero the winner only, the pseudo tokens only, both; per recall bucket.

## Predictions (mine, orchestrator)

- **D-1** coda attention mass on ALL slot positions, averaged over tokens, layers, heads: under
  10 % (the coda mostly reads its own span). 70 %.
- **D-2** the seed pools put more than 30 % of their mass on one position (the boundary token
  or the span's first token) in most cells: the pool is near a sink. 45 %.
- **D-3** per-pass relative change after pass 2 under 5 %: the loop is near its fixed point
  after two passes, so K1-K6 can only come from passes 1-2. 60 %.
- **D-4** the loop's cells put under 20 % of their attention on earlier slots: the loop mostly
  re-reads its own span's seed, so it adds little cross-span computation. 50 %.
- **D-5** snap's pseudo tokens get more coda attention mass than its winner position. 55 %.

No pass rule: this is a diagnostic. It is filed as success if the readings identify a stage
that loses more than half of what enters it, and as a failure if nothing stands out.

## Results

Run on the 3070 at aa127cb, 96 validation rows per arm (batch 1, seed 1234), eval depth 6,
router-followed. Probe `lab/divergence/dataflow_probe.py`. Sanity: hooks leave CE
bit-equal; captured attention rows sum to 1 within 3.6e-7; recomputed attention outputs match
within 0.5 % (bf16); zero-everything worth reproduces the runner (snap +0.1843 vs +0.1852,
LXTUL +0.1769 vs about +0.176). Artifacts:
[`../results/2026-10-05-snap-dataflow-probe/`](../results/2026-10-05-snap-dataflow-probe/).

| stage | reading | snap | LXTUL |
| --- | --- | --- | --- |
| seed pool | normalised entropy; cells with > 30 % on boundary or first token | 0.92; 9.8 % | 0.92; 10.8 % |
| loop carrier | share of energy that DIFFERS between slots: entry, after pass 1, exit | 57 %, 0.4 %, 1.8 % | 54 %, 0.2 %, 1.7 % |
| loop carrier | RMS at entry / after pass 1 / exit (all 4 HC streams) | 1.13 / 1.04 / 1.05 | 1.12 / 1.06 / 1.06 |
| loop carrier | per-slot RMS at entry / after pass 1 | 0.85 / 0.068 | 0.82 / 0.046 |
| loop, per pass | change of the stream mean, p1..p6 | 14, 0.67, 0.95, 0.21, 0.05, 0.03 | 20, 0.74, 0.92, 0.25, 0.05, 0.02 |
| core reads | attention mass on earlier slots / own slot | 0.87 / 0.13 | 0.86 / 0.14 |
| core reads | shared share of what the earlier-slot read returns, layers 1-4 | 0.64-0.96 | 0.83-0.97 |
| write | coda-input RMS: winner / pseudo / tokens | 0.16 / 9e-5 / 0.55 | 0.17 / 0 / 0.58 |
| write | snap gate g (init 0) | 0.001-0.009 | n/a |
| coda reads | token attention on slot positions; winner / other positions / own span | 0.53; 0.20 / 0.32 / 0.47 | 0.58; 0.35 / 0.23 / 0.42 |
| depth | K1-K6 on these rows | +0.0135 | +0.0222 |
| worth | zero both / winner only / pseudo only | 0.184 / 0.112 / 0.037 | 0.177 / 0.177 / n/a |

| prediction | reading | held |
| --- | --- | --- |
| D-1 coda slot attention under 10 % | 53 % / 58 % | no |
| D-2 seed pool near a sink | 10 % / 11 % of cells | no |
| D-3 per-pass change under 5 % after pass 2 | pass 3 changes the stream mean by 95 %; under 5 % only at pass 5 | no |
| D-4 loop reads earlier slots under 20 % | 87 % / 86 % (but what it reads is mostly one shared vector) | no |
| D-5 pseudo attention above winner attention | 0.32 vs 0.20 | yes |

## Verdict

Success under the prereg's rule: one stage loses far more than half of what enters it. **The
loop's FIRST PASS replaces the slot-specific state with one vector shared by every slot of
every row.** At entry 54-57 % of the carrier's energy differs between slots; after pass 1,
0.2-0.4 %; at the exit 1.7-1.8 %. The RMS stays near 1.05, so the per-slot part shrinks
about 12-18x in absolute size. The shared vector lives in the DIFFERENCES between the 4
Hyper-Connection streams and cancels in the stream mean, which is why every earlier probe
(all of which read the stream mean, including the 2026-10-02 LayerNorm probe) saw a small
"shared direction" instead of a 98 % constant. The read side is wide open (the coda puts
about half its attention on slot positions), and the loop does read the past, but what its
cross-slot attention returns is mostly that same constant.

Other findings: snap's pseudo-token gate never opened (g 0.001-0.009); the pseudo positions
work anyway because RMSNorm's eps and the q/k norm turn a 9e-5 input into full-size keys, and
their picks recur later no more often than random span tokens (0.50 vs 0.55). In LXTUL the
"zero" positions are not dead at the read: the coda's conv and value shift carry the winner
into them (23 % of coda attention).

Not measured: whether the constant is inert or load-bearing (no causal ablation), whether
the per-slot content is still linearly decodable after pass 1, and which block or norm
creates the constant.

## Updated hypothesis

The loop's small and fragile contribution is a signal-to-constant problem: every pass
spends about 98 % of the carrier on a slot-independent vector, so the content the loop
refines is a 2 % ripple. Removing the constant before the per-pass norm (or locating and
removing its source) should raise the per-slot share and, if the hypothesis holds, the
loop's contribution.

## Follow-up: where the constant comes from (2026-10-05, no separate prereg)

Run after the verdict above, to locate the source; it had no predictions of its own, so it
is evidence for the next prereg, not a test. Probe `lab/divergence/carrier_constant_probe.py`
(3070 at f649ac6, 96 rows per arm; hooks bit-equal; the per-source decomposition sums to the
carrier exactly; reproduces 0.43 % / 0.19 % after pass 1). Artifacts beside the ones above
(`carrier_constant_*`).

- **Source: the core ATTENTION, not the norm, the injection, the MLPs or the reset.** The
  constant's direction is already 47-49 % of the entry state (cosine +0.91 / +0.94 to the
  final constant). In pass 1 the HC pre-map feeds the core attention an input that is
  67-94 % shared by layers 1-2 and about 99.9 % shared by layer 3, while the MLPs are fed
  the per-slot part (4-30 % shared). Core attention then writes a large, 99.8-99.9 %
  shared output (LXTUL layer 2: RMS 51; snap layers 2-3: RMS 3 and 24), concentrated into
  one HC stream (Hpost row about [0, 0, 3.9, 0]). The per-pass norm divides the 25-50x
  blow-up back out. Norm gain g is about 1.00; DiagonalInjection, x0 terms, MLP writes and
  the reset are small or per-slot. It is rebuilt every pass by the same layers.
- **Same vector:** cosine 0.97-0.98 between pass 1 and the exit; identical across the 4
  cells; one channel direction with a fixed sign per stream (+,-,-,+ / +,-,+,-), spread over
  many channels (top-32 channels 34-44 % of its energy), not one massive channel.
- **Content survives:** row-held-out ridge R^2 for the slot's own span (mean embedding):
  seed 0.397 / 0.385, after pass 1 0.350 / 0.355, exit 0.333 / 0.349. Next span: seed
  0.044 / 0.043, exit 0.054 / 0.055: the loop adds about +0.01 R^2 of next-span content.
- **Load-bearing in the trained models:** removing it before the per-pass norm costs
  +0.114 / +0.127 nats and turns K1-K6 negative; a same-size constant with permuted
  channels costs as much; keeping it out of the norm only (N(f-c)+c) costs +0.096 / +0.065.

Reading: from core layer 2 on, the loop's attention reads a near-constant input and writes
a near-constant output: a no-op the downstream has learned to rely on. Cross-slot retrieval
in the loop is therefore mostly a blur, and the per-slot work happens in the MLPs. Any fix
must be trained in (post-hoc removal breaks the trained model).
