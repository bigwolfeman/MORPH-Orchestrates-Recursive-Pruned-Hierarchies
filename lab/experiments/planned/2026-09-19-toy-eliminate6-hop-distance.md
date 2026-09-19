# Experiment: does the carry survive SIX hops, and can a fitted reader find what the head cannot?

Status: planned
Date: 2026-09-19
Owner: Claude (session f9558148), sub-agent ToyEliminate, for Wolfe.

Frozen before any `eliminate6` training run exists. At filing time the only numbers from
the new task are the self-check suite (`lab/toy_slot_loop/selfcheck.py`, 106 checks pass,
exit 0), the enumerated ceilings below, and one GPU smoke whose wall time and readings are
disclosed in Method.

Iteration 1: [`2026-09-18-toy-eliminate-deferred-commitment.md`](../failures/2026-09-18-toy-eliminate-deferred-commitment.md)
and its result [`2026-09-18-toy-eliminate/README.md`](../results/2026-09-18-toy-eliminate/README.md).
Study: [`lab/toy_slot_loop/`](../../toy_slot_loop/).

## Question

`eliminate` answered its question and broke its own instrument. Every one of its 30 trained
cells solved the task, so the escape rate measured nothing; the alive set turned out to be
carried as the EXCLUSIONS rather than as a superposition of candidate embeddings; and the
load-bearing surprise was that a linear probe read the alive set at 1.000 on a RANDOM-INIT
network the moment an attention hop could reach it. The conclusion drawn was "carrying a
set through this loop costs nothing and needs no training; what training buys is the
readout."

That conclusion rests on a two-hop chain. **This iteration asks whether it survives six.**
Each pass is another nonlinear mixing into a fixed-width state that is also carrying
everything else, so a fact five hops back has been re-encoded five times. If the carry
degrades with hop distance and training repairs it, then "carrying costs nothing" was a
two-hop artefact and the honest statement is "carrying costs nothing for one or two hops".
If the carry is free at six hops too, the loop's problem really is the readout alone.

Second question, also from iteration 1: a tied-head readout cannot separate "the head
cannot read it" from "the state does not hold it". A per-slot linear reader FITTED to the
exit state on held-out rows, with the model frozen, separates them, and it is reported as a
value CE so it sits next to the coda's own reading and next to the enumerated ceilings.

## The task: `eliminate6`

`lab/toy_slot_loop/tasks.py`. 22 disjoint ids: 8 candidate symbols (input), 6 distractor
symbols (input), 8 answer symbols (OUTPUT ONLY, `answer(c) = 14 + c`). One instance per
row, 8 spans of 3 cells:

| span | content | alive set after it |
|---|---|---|
| 0, 1 | the six DISTINCT candidates, three per span, in a uniformly random order | 6 |
| 2 … 6 | [elimination][distractor][distractor], five eliminations in a uniformly random order | 5, 4, 3, 2, 1 |
| 7 | three distractors; its HEAD carries the survivor's answer id as the label | — |

**Three deviations from the brief, all forced, all named.** The brief asked for six
candidates, four eliminations, a four-pass chain and the answer at the head of span 6.
(1) Four eliminations of six candidates leave TWO alive, so the answer would not be
determined and the requested ladder could not reach 0: it takes FIVE. (2) Six candidates
need two spans at `span_len` 3, and the answer slot is the last elimination span, so the
distance from the answer slot to span 0 is SIX, not four. (3) With a six-symbol alphabet
every instance's candidate set would be the whole alphabet, the candidate spans would carry
nothing, and the question would collapse into the exclusion encoding iteration 1 already
found; so the alphabet is eight and an instance uses six. The requested commitment ladder,
ln 6 → ln 5 → ln 4 → ln 3 → ln 2 → 0, is reproduced exactly.

A deeper chain also needs a deeper draw. The answer slot needs `d_k >= k` for k = 1…6 along
the prefix chain, which holds on about 39 % of rows under the toy's Poisson(6) and on about
69 % under Poisson(8), so `eliminate6` runs at **mean depth 8, max 10** and `run_cell.py`
REFUSES to start it shallower without `--allow_shallow_depth`.

### The ceilings, enumerated over all 20,160 configurations

`tasks.eliminate6_ceilings()` enumerates; `selfcheck.py::t_eliminate6_ceilings` checks each
against its closed form (worst deviation 1.42e-12) and checks the sampler against the
enumeration by Monte Carlo (H(survivor | all five eliminations) measured 1.0973 against
ln 3 = 1.0986 on 40,000 rows). Chance is ln 8 = 2.0794.

**Reachability floor** — the lowest value CE at forced depth d. One fact per hop:

| d | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| what the answer slot can hold | e5 | e4,e5 | e3…e5 | e2…e5 | all five | + span 1 | + span 0 |
| best possible value CE | 1.9459 | **1.7918** | **1.6094** | **1.3863** | **1.0986** | **0.5493** | **0.0000** |

The depth price is measured, not argued: `selfcheck.py::t_eliminate6_depth_requirement`
differentiates the value logit at the head of span 7 with respect to the raw cell
embeddings and reads exactly zero at every span closer than the forced depth allows and
non-zero at and above it, for T = 1, 2, 4, 6.

**Commitment ceiling** — a predictor that is a function of the candidate spans and the
first k eliminations only:

| k | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| best possible value CE | 1.7918 (ln 6) | 1.6094 (ln 5) | 1.3863 (ln 4) | 1.0986 (ln 3) | 0.6931 (ln 2) | 0.0000 |

**Point-carry ceiling** — collapses to ONE candidate after k eliminations and still reads
the five eliminations locally: 0.9155, 0.8789, 0.8240, 0.7324, 0.5493, 0.0000.

**A warning the previous task did not need.** These ladders OVERLAP: reachability at depth
d equals commitment at k = d − 1 for d = 1, 2, 3, 4. So a plateau alone no longer names the
failure mode, unlike `eliminate`, where 1.386 / 0.924 / 0.693 / 0 were distinct. The
K-curve shape is what separates them: a depth-starved model IMPROVES with forced depth up
to the depth it learned, a committed one does not move.

## The instruments

`candidate_mass`, `membership_probe` and `twin_divergence` are now spec-driven, so
`eliminate` and `eliminate6` share one implementation. Roles for `eliminate6`: `cand` =
slot 1 (whole candidate set from pass 1), `mid` = slot 4 (third elimination, set complete
at pass 4), `answer` = slot 6. Probe depth 8.

Changes made for this iteration, with what they cost:

1. **The membership probe fits three facts, not one**: `alive` (as before), `in_set` (is
   the symbol one of the instance's candidates — carried by the candidate spans) and `dead`
   (killed by an elimination inside the window — one hop per elimination span). Reading the
   answer slot's probe at pass t IS the hop-distance ladder, because pass t reaches exactly
   hop t.
2. **Every probe and reader fit is seeded per (role, pass, fact)**, so it is reproducible
   and independent of how many fits preceded it. The 2026-09-18 version was not: re-running
   those cells would move their probe numbers by up to 0.047 on a deliberately
   under-trained 20-step fit. Those numbers were saturated at 0.500 / 1.000, so no
   conclusion of iteration 1 moves, but they are not bit-comparable with a re-run.
   `candidate_mass` and `twin_divergence` ARE bit-identical across the refactor, verified
   old-against-new on a seeded untrained model: 6 of 6 role readings identical.
3. **`adapted_reader` is new**: one linear map per (slot, forced depth), fitted to the exit
   state on one draw of rows with the model frozen and scored on a disjoint draw, reported
   as a value CE in nats. By forced depth at the answer slot (d = 1…6, 8) and by role slot
   at the probe depth.
4. **A balanced accuracy with an empty class returns None, not NaN.** NaN is not valid JSON
   and it silently poisons every comparison downstream; it hid a real difference for one
   debugging round here.
5. **Random-init cells are first-class**: 3 seeds per width, not one afterthought.

## Predictions (frozen 2026-09-19, before any cell ran)

"Solved" means value accuracy > 0.9 on the 256-row probe draw, the same escape rule as
every grid in this study.

- **P1. The basin is back and the attachment axis separates.** 55 % that plain `exit`
  escapes on 2 or fewer of 5 seeds at d96 (against 5 of 5 on `eliminate`); 50 % that
  `exit`+fixed-point escapes strictly more often than `exit`; 45 % that `staged` escapes on
  3 or more of 5. 40 % that some cell reads 0/5 while another reads 3/5 or better, which is
  what "the axis separates" means. 15 % that nothing escapes anywhere, which would leave the
  instruments to carry the study again.
- **P2. Stuck seeds stop part-way along the chain.** 60 % that the median stuck seed's value
  CE at depth 6 lies in [1.0986, 1.7918], i.e. it learned between one and four hops. 25 %
  that any stuck seed sits below 0.6931.
- **P3. Stuck seeds are depth-starved, not committed.** 70 % that the median stuck seed's
  value CE falls by at least 0.10 nats between forced depth 1 and forced depth 6. A flat
  K-curve at a plateau would mean commitment instead, and the two ladders overlap, so this
  is the clause that separates them.
- **P4. THE HOP LADDER AT RANDOM INIT — the headline.** At the answer slot of an untrained
  network, the `dead` probe at hop h is fitted on a state that has been re-encoded h times.
  60 % that its balanced accuracy DECAYS monotonically in h (allowing 0.01 of noise),
  70 % that it is at least 0.95 at hop 0 and 1, 50 % that it is below 0.75 at hop 4 and
  50 % that it is below 0.65 at hop 6, at d96. For `in_set`, whose source spans sit at hops
  5 and 6, 55 % that the random-init accuracy at hop 6 is below 0.70. **And the contrast:
  65 % that the trained cells hold the ladder ABOVE random init by at least 0.05 at every
  hop >= 3.** If that holds, iteration 1's "carrying costs nothing" needs the qualifier "for
  one or two hops", and carrying across depth is itself something training buys.
- **P5. The exclusion encoding replicates.** 70 % that at every role and pass of every
  solved seed the largest of the three masses is the one on the reachable DEAD candidates.
  Only 30 % that any solved seed reads, at role `mid` (alive set 3 from pass 4), entropy
  above 0.5 of a possible ln 3 = 1.0986 with dead-mass below 0.3 — the superposition
  signature, which `eliminate` failed in 20 of 20 cells.
- **P6. The fixed-point entropy effect replicates.** 65 % that at role `cand` the
  `exit`+fixed-point arm's mean readout entropy exceeds plain `exit`'s by at least 0.20
  nats at BOTH widths. On `eliminate` the gap was 0.989 against 0.352 of a possible 1.0986.
- **P7. The fitted reader on a RANDOM-INIT loop.** Written down before seeing it, as asked.
  80 % that at forced depth 1 its value CE is 1.79 or worse (the depth-1 floor is 1.7918);
  60 % that at depth 4 and beyond it is BELOW chance minus 0.2, i.e. below 1.88, so it does
  recover something from an untrained loop; but 75 % that at depth 6 it is still above 0.8,
  i.e. an untrained loop does NOT hand a linear reader the answer even when every fact is
  formally inside the window. 50 % that its CE at depth 6 is within 0.25 of its CE at depth
  4, which would say the last two hops deliver nothing to a linear reader.
- **P8. The reader beats the coda on stuck seeds.** 70 % that on stuck seeds the fitted
  reader's value CE at depth 6 is at least 0.15 nats below the coda's, i.e. the state holds
  more than the tied head exposes; 40 % that the gap exceeds 0.5 nats. On solved seeds both
  are at 0.000 and the comparison says nothing.
- **P9. Twin divergence is the geometry, again.** 90 % that in 30 of 30 trained cells the
  drop pass is None at role `cand` and 2 at role `mid`, and 70 % that it is 4 at role
  `answer`. The first elimination is at span 2, so the geometric drop pass is the slot
  index minus 2: 4 for slot 6, 2 for slot 4, never for slot 1. One instance per row, so the
  cross-instance leak that fired on two random-init cells in iteration 1 cannot happen here.
  The `answer` clause gets a lower probability than the others because `drop_pass` is a
  cosine crossing 1 - 1e-3, and a one-token change four hops back may be attenuated below
  that threshold: the cosine row, not the crossing, is the data.
  *(Corrected 2026-09-19, after the smoke and before the grid: as first written this said
  "3 at role `mid`", which is simply wrong arithmetic — slot 4 minus span 2 is 2, and the
  instrument's own `expected_drop_pass` field said 2 all along. The smoke made the slip
  visible; its readings are disclosed in Method.)*
- **P10. Width, now that it is readable.** 50 % that d192 escapes on at least two more cells
  of 15 than d96. Iteration 1 could not read this because every cell escaped.

### The decision rule this grid exists to settle

If the random-init hop ladder is flat and high (0.90 or better at every hop) and the
random-init reader reaches below 0.5 nats at depth 6, the carry is free at six hops too and
iteration 1's conclusion stands unqualified: the loop's problem is the readout. If the
ladder decays with hops and the trained cells lift it, then carrying across depth is
learned, "carrying costs nothing" was a two-hop artefact, and the thing to test on the real
model is whether ITS carry decays with hop distance — which is the sibling hop-distance
probe already running on real checkpoints.

### Reading rules, fixed in advance

- Escape rate is the primary metric; nothing below 0.02 nats of value CE and nothing below
  0.05 of probability mass or probe accuracy is a result at 5 seeds.
- The instruments are read on solved and stuck seeds separately. Pooling averages a
  superposition against a collapse.
- Random-init cells are a baseline, never a data point about training.
- A probe accuracy is evidence when it is LOW. A high one is not evidence of use.

## Method

Host: the 3070 (`ssh 3070`), venv `/home/wolfe/venvs/toyloop` (torch 2.14.0+cu130), working
copy `/mnt/bigdata/toyloop/toy_slot_loop`. The box is also running two hop-distance probes
for another agent; the toy used 569 MiB on the previous grid, every job runs under
`nice -n 10`, one cell at a time, and nothing outside `/mnt/bigdata/toyloop/` is touched.

Grid: `eliminate6` x attachment {`exit`, `exit` + `fixed_point_lambda 1.0`, `staged`} x
d_model {96, 192} x 5 seeds, plus random-init (steps 0) x 2 widths x 3 seeds. 36 runs,
4,000 steps, batch 128, strict geometry, mean depth 8, max depth 10, and every other knob
at the toy's 2026-09-10 default. At d192 the head count and feed-forward width scale with
it: n_heads 4 -> 8 (head dim 24 held), d_ff 256 -> 512. The K-curve runs at forced depths
1, 2, 3, 4, 5, 6, 8, 10 and the fitted reader at 1, 2, 3, 4, 5, 6, 8.

Cut on purpose: no `mux_all`, no `mux_all_detach`, no `deep_coda`, no `progressive`, no
permissive geometry, no longer horizon, and no sweep of the depth draw.

### Everything the smoke printed, disclosed before filing

One 300-step cell, `exit` at d96, mean depth 8, ran to time the grid and catch a crash. It
is n = 1 at 300 of 4,000 steps, and it is reported in full so a reader can discount any
prediction it bears on.

| quantity | reading |
|---|---|
| value CE @6, accuracy, escape | 1.5641, 0.337, none |
| K1-K6 | +1.1586 |
| `candidate_mass` at (`cand`, pass 1) | survivor 0.147, dead 0.000, other 0.853, entropy 0.221 of a possible ln 6 = 1.7918 |
| `membership_probe` at (`cand`, pass 1) | alive 1.00, in_set 1.00, dead None (no elimination is ever reachable from slot 1) |
| fitted reader value CE, depth 1 / depth 8 | 2.043 / 1.497 |
| `twin_divergence` drop pass, cand / mid / answer | None / 2 / 5 |
| wall clock | 12 s training, 40 s for the whole cell |

Two of those bear on predictions. The reader's 2.043 at depth 1 is consistent with P7's
"1.79 or worse". The `answer` drop pass of 5 against a geometric 4 is why P9's `answer`
clause carries a tolerance caveat; at 300 steps a one-token change four hops back moves the
cosine by less than 1e-3.

Artifacts: one JSON per cell under
`ignored/experiment-artifacts/2026-09-19-toy-eliminate6/`, aggregated with
`python lab/toy_slot_loop/aggregate.py <dir>` into
`lab/experiments/results/2026-09-19-toy-eliminate6/`.
