# Experiment: when early commitment is expensive, does the toy slot loop carry a SET?

Status: planned
Date: 2026-09-18
Owner: Claude (session f9558148), sub-agent ToyEliminate, for Wolfe.

Frozen before any `eliminate` training run exists. At filing time the only numbers in this
tree from the new task are: the self-check suite (`lab/toy_slot_loop/selfcheck.py`, all
checks pass, exit 0), the enumerated ceilings below, one 12-step CPU smoke that proved the
pipeline runs, and one GPU smoke whose wall time is quoted in Method. No trained cell has
been read.

Prior study: [`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md) (2026-09-10).
Arc: [`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).

## Question

The toy settled two things on 2026-09-10. On `summary`, whose target is computable in one
pass, no attachment moves the K-curve: 30 of 30 cells read K1−K6 = +0.0000. On `compose`,
an S_3 scan that formally needs six passes, `staged` and `exit`+fixed-point solve the chain
on 5 of 5 seeds and plain `exit` on 2 of 5. In `compose` the optimal intermediate state is
a POINT: one group element, the running register. Nothing in the study has ever asked the
loop to carry a state that is not a point.

The superposition literature asks exactly that. Zhu et al. (NeurIPS 2025, *Reasoning by
Superposition*) show a looped transformer solving graph reachability by holding a
SUPERPOSITION of frontier nodes in one continuous state and pruning it with later evidence,
and that this emerges from outcome supervision alone, with no supervision on the set.
Rizvi-Martel et al. (2026, *The Illusion of Superposition?*) report the other half: the
same capacity collapses to a single candidate under pretraining or fine-tuning, and latent
reasoning contributes little below roughly 128–256 hidden dimensions.

MORPH's real slot cells sit at effective rank about 14 in 1024 dimensions and read 95 % of
their gain in pass 1; the healthy plain loop reads 89 %. The Thought Register arm
(2026-09-13) seeded four cells apart and they collapsed to rank 1.24 of 4. So the repo has
already reproduced the collapse half of that literature, on its own stack, and has never
run the other half: a task whose optimum requires the set.

**The question: when early commitment is provably expensive, does THIS loop (a shared core
block over slot cells under strict geometry) (a) solve the task, (b) carry the alive set as
a superposition of candidate embeddings, as some other encoding, or not at all, (c) defer
commitment pass by pass, and (d) depend on width?**

## The task: `eliminate`

`lab/toy_slot_loop/tasks.py`. 18 disjoint ids: 6 candidate symbols (input), 6 distractor
symbols (input), 6 answer symbols (OUTPUT ONLY, `answer(c) = 12 + c`, a fixed bijection).
`compose` and `summary` keep their 12-id S_3 vocabulary and are bit-identical under the
change (verified: loss, MUX, core weight gradient and embedding gradient match the previous
commit to the last printed digit on 12 of 12 task x attachment combinations).

One instance is four spans of three cells; a row is 8 spans, so two independent instances:

| span | content |
|---|---|
| s+0 | three DISTINCT candidates — the alive set, size 3 |
| s+1 | [first elimination][distractor][distractor] — alive set 2 |
| s+2 | [second elimination][distractor][distractor] — alive set 1 |
| s+3 | three distractors; its HEAD carries the survivor's answer id as the label |

The survivor is uniform over the three candidates and the elimination order is uniform, so
from span s+0 alone the answer is uniform over 3, after s+1 uniform over 2, after s+2
determined. Every other token position carries the ordinary next-symbol label. The answer
never appears as an input token. MUX targets: the next-span target exists only at the
answer slot s+2; the `staged` own-span target is the span's own elimination symbol, at
s+1 and s+2.

### The ceilings, enumerated

Every number below is computed by `tasks.eliminate_ceilings()`, which enumerates the 120
equiprobable ordered triples (survivor, first eliminated, second eliminated) and takes a
conditional entropy. `selfcheck.py::t_eliminate_ceilings` checks each against its closed
form and checks the SAMPLER against the enumeration by Monte Carlo (measured
H(survivor | e1, e2) = 1.3855 against ln 4 = 1.3863 on 60,000 instances).

**Reachability ceiling** — the lowest value CE at forced depth d, the K-curve's floor.
Under strict geometry the answer slot s+2 reaches spans s+2−d … s+2 after d passes, so
depth buys exactly the facts inside that window. The measured proof is the same shape as
`compose`'s: `selfcheck.py::t_eliminate_depth_requirement` differentiates the value logit
at the head of span 3 with respect to the raw cell embeddings and reads
**0.000e+00 at depth 1 and 1.010 at depth 2**.

| d | 1 | 2 | 3 | 6 | 8 |
|---|---|---|---|---|---|
| best possible value CE | **1.3863** (ln 4) | **0.0000** | 0.0000 | 0.0000 | 0.0000 |

**Commitment ceiling** — the lowest value CE for a predictor that is a function of spans
s+0 … s+t only, i.e. one that commits at span s+t and then ignores every later
elimination. This is the price of early commitment stated as an entropy.

| commits after | s+0 | s+1 | s+2 |
|---|---|---|---|
| alive set | 3 | 2 | 1 |
| best possible value CE | **1.0986** (ln 3) | **0.6931** (ln 2) | 0.0000 |

**Point-carry ceiling** — the realistic plateau for a loop that collapses its state to ONE
candidate id at span s+t but still reads the later eliminations locally at the answer slot.
It is below the commitment ceiling because those eliminations are still readable.

| collapses at | s+0 | s+1 | s+2 |
|---|---|---|---|
| best possible value CE | **0.9242** = (2/3)·ln 4 | **0.6931** = ln 2 | 0.0000 |

Chance is ln 6 = 1.7918. So `eliminate` has FOUR plateaus, and the plateau a stuck run
sits at says where it committed: 1.386 = depth starved, 0.924 = collapsed at the candidate
span, 0.693 = collapsed after the first elimination, 0.000 = carried the set.

## The instruments

All three read the slot state after t = 0 … 6 passes (t = 0 is the entry state, zero
passes) at three slot roles inside an instance: `cand` = s+0, `elim1` = s+1, `answer` =
s+2. Each reading is scored against that slot's reachable window at that pass, not against
the whole instance.

1. **`candidate_mass`** — the tied answer head's softmax restricted to the three answer ids
   of the instance's own candidates: mean mass on the survivor, on candidates the window
   already proves dead, and on alive non-survivors, plus the entropy and the top mass. The
   entropy is load-bearing: the mean mass on the survivor is 1/3 both for a 3-way
   superposition and for a uniformly random commitment, so only the entropy separates them.
   This reads whether the set is carried as a superposition of candidate EMBEDDINGS.
2. **`membership_probe`** — a linear probe (6 logistic outputs, "symbol c is still alive in
   this window") fitted post hoc on the FROZEN model's states from one draw and scored on a
   disjoint draw. Balanced accuracy over all six symbols, plus `acc_within_set`, alive
   against already-eliminated inside the instance's own three candidates. This reads whether
   the set is carried AT ALL, in any encoding.
3. **`twin_divergence`** — pairs of rows identical through span s+0 and differing ONLY at
   the head of span s+1 (the survivor and the first eliminated candidate swap roles): the
   cosine between the two rows' slot states per role and per pass, the first pass it drops
   below 1 − 1e-3, and the value CE of both twins.
4. `k_curve`, `write_contribution`, `participation_rank` and the per-pass gradient probe
   run unchanged.

**An instrument caveat measured before filing, which shapes how the probe may be read.**
On an untrained 12-step CPU model the membership probe already reads `acc_within_set` 0.94
at (`elim1`, pass 1). A random network preserves linearly decodable information, so a high
probe accuracy is NOT evidence that the loop uses the set. The probe's job is the opposite:
a LOW accuracy is evidence the set was destroyed. Two random-init cells (steps 0, one per
width) are therefore part of the grid as the probe's own baseline.

## Predictions (frozen 2026-09-18, before any cell ran)

Bars are on the 4,000-step strict-geometry grid. "Solved" means value accuracy > 0.9 on the
256-row probe draw, the same escape rule the 2026-09-10 grid used.

- **P1. Escape rate beats `compose`.** 55 % that plain `exit` escapes on at least 3 of 5
  seeds at d_model 96, against `compose`'s 2 of 5; 75 % that `exit` + fixed-point and
  `staged` each escape 5 of 5, matching `compose`. Reason: the chain here is 2 passes deep,
  not 6, so the one-pass basin is shallower. 20 % that every one of the 30 cells escapes, in
  which case the escape axis is uninformative and the study's answer comes entirely from the
  encoding instruments.
- **P2. Escape is fast.** 60 % that the mean escape step over escaped seeds is below 750
  (`compose` at `exit` read 625 and its fixed-point arm 1400).
- **P3. Stuck seeds land on a derived plateau.** 70 % that every stuck seed's value CE at
  depth 6 sits within 0.05 nats of one of 1.3863, 0.9242, 0.6931. 60 % that at least one
  stuck seed across the 30 cells sits at 0.9242 or 0.6931 rather than at 1.3863, which would
  be a measured point-carry solution. Falsified if a stuck seed sits below 0.6931 − 0.05
  without reaching accuracy 0.9: the plateau derivation would then be wrong.
- **P4. The K-curve is bounded below by the reachability ceiling, and depth past pass 2 is
  worth nothing.** 80 % that on solved seeds K1−K6 ≥ 1.386 and K2−K6 = 0.000 ± 0.02. A
  K1−K6 BELOW 1.386 on a seed at value CE 0.000 would mean the depth-1 forward beats the
  enumerated ceiling, i.e. something outside the loop is answering.
  *(Corrected 2026-09-18, after the smoke and before the grid. As first written this
  predicted K1−K6 = 1.386 ± 0.02. That was a derivation error, not a forecast: the ceiling
  bounds the depth-1 CE from BELOW, and a model trained at Poisson depth has no reason to be
  optimal at a depth it is not trained at — `compose` trained at fixed depth 8 read 12.987 at
  depth 1. The smoke made the error visible; the corrected clause is the one the derivation
  supported all along, and the smoke numbers it would be graded against are disclosed in
  Method.)*
- **P5. The set is carried as a superposition at the pruning slot.** 40 % that on solved
  seeds `candidate_mass` at (`elim1`, pass 1) reads entropy above 0.50 (against ln 2 = 0.693
  for a perfect 2-way superposition and 0 for a commitment) with mass on the
  already-eliminated candidate below 0.15. This is the study's headline prediction and it is
  the one I expect to be wrong, because every previous MORPH arm collapsed.
- **P6. Commitment is deferred, not immediate.** 45 % that the mass on the reachable dead
  candidate falls monotonically with pass at role `elim1` and that the top mass at role
  `answer` rises above 0.8 only at pass 2 or later.
- **P7. The probe finds the set where the mass does not.** 65 % that on solved seeds
  `membership_probe` `acc_within_set` at (`elim1`, pass 1) is above 0.85 while the mass
  entropy there is below 0.35, i.e. the set is carried in some encoding the tied answer head
  does not expose. The null clause: 75 % that `acc_balanced` at (`elim1`, pass 0), where the
  candidate span is NOT reachable, is below 0.60.
- **P8. `staged` pushes the state to advertise the DEAD candidate.** 60 % that `staged`
  reads mass on the reachable eliminated candidate above 0.50 at (`elim1`, pass 0) and above
  0.25 at (`elim1`, pass 1), against `exit` below 0.15 at (`elim1`, pass 1). The own-span
  target here is the span's OWN elimination, so unlike `compose` the staged target is a
  commitment pressure pointed away from the alive set. If `staged` still solves 5 of 5 while
  reading a high dead-mass, then solving does not need a clean superposition.
- **P9. twin_divergence is exactly the geometry.** 95 % that in 30 of 30 cells the drop pass
  is None at role `cand` (cosine 1.0 at every pass), 0 at `elim1` and 1 at `answer`. That is
  a structural fact, so its value is the magnitude: 50 % that the cosine at (`answer`,
  pass 1) is below 0.90 on solved seeds and above 0.98 on seeds stuck at 1.386.
- **P10. Width matters, weakly.** 50 % that d_model 192 escapes on at least two more cells
  of 15 than d_model 96. Rizvi-Martel et al. report latent reasoning contributing little
  below 128–256 dims; 96 is below that band and 192 inside it. Falsified if 192 escapes on
  fewer or the same cells, which would say width is not the constraint at this scale.

### The falsifier this study exists to expose

If solved seeds read early commitment on `candidate_mass` (entropy at (`elim1`, pass 1)
below 0.25 of a possible 0.693, top mass above 0.85) while `membership_probe`
`acc_within_set` there stays above 0.85, then the alive set is carried as a POINT-plus-
context encoding and not as a superposition of candidate embeddings. The conclusion would
then be that superposition is one encoding among several rather than the mechanism, and the
repo's Thought Register result stops being evidence that the loop cannot hold a set.

### Reading rules, fixed in advance

- The escape rate is the primary metric. Nothing below 0.02 nats of value CE is a result at
  5 seeds, and nothing below 0.05 of probability mass or probe accuracy is a result.
- A CE at 4,000 steps ranks nothing across cells that differ in parameter count, so d_model
  96 and 192 are compared on escape rate and on the instruments, never on CE alone.
- Every table reports mean and [min, max] over the 5 seeds.
- The instruments are read on SOLVED seeds separately from stuck seeds. Pooling them would
  average a superposition against a collapse.

## Method

Host: the arch server's RTX 3070 (`ssh 3070`), venv `/home/wolfe/venvs/toyloop`
(torch 2.14.0+cu130), working copy `/mnt/bigdata/toyloop/toy_slot_loop` (the box's `/home`
is full). Every job runs under `nice -n 10`, one cell at a time, detached with `setsid`.

Grid: `eliminate` x attachment {`exit`, `exit` + `fixed_point_lambda 1.0`, `staged`} x
d_model {96, 192} x 5 seeds = 30 cells, 4,000 steps, batch 128, strict geometry, and every
other knob at the toy's 2026-09-10 default (mean depth 6, max 8, prelude entry, injection
decay 0.45, full BPTT, AdamW lr 2e-3 with a 200-step warmup and cosine decay). At d_model
192 the head count and the feed-forward width scale with it: n_heads 4 -> 8 (head dim 24
held) and d_ff 256 -> 512. Two extra cells at steps 0, one per width, give the membership
probe its random-init baseline. 32 runs.

Cut on purpose: no `mux_all`, no `mux_all_detach`, no `deep_coda`, no `progressive` (all
four were measured on `compose` and none of them is the question here), no permissive
geometry, no depth sweep, no longer horizon.

### Everything the smoke printed, disclosed before filing

The predictions above were written before any GPU cell ran. Two 300-step smoke cells then
ran, to time the grid and to catch a crash, and they are reported here in full so a reader
can discount any prediction they bear on. They are n = 1, one seed, at 300 of 4,000 steps.

| smoke | value CE @6 | acc | K1−K6 | escape step | train s | cell s |
|---|---|---|---|---|---|---|
| `staged`, d96 | 0.0031 | 1.000 | +1.8784 | 250 | 7 | ~13 |
| `exit`, d192 | 0.0006 | 1.000 | +1.6159 | 250 | 10 | 17 |

Both solved the task at the FIRST probe (step 250), which makes P1's 20 % clause — that
every cell escapes — the likely outcome and the escape-rate axis close to dead. Their
instrument readings: at (`cand`, pass 0) mass survivor/dead/other 0.343/0.000/0.657 with
entropy 0.395 (`staged`) and 0.337/0.000/0.663 with entropy 0.401 (`exit`); at (`elim1`,
pass 1) 0.000/1.000/0.000 with entropy 0.000 (`staged` — total commitment to the DEAD
candidate, which is what its own-span target asks for) and 0.328/0.305/0.367 with entropy
0.174 (`exit` — near-uniform MEAN masses under a per-row commitment, exactly the case the
entropy column exists to separate). `membership_probe` `acc_within_set` read 1.0 at
(`elim1`, pass 1) in both. `twin_divergence` drop passes were None/0/1 in both.

Artifacts: one JSON per cell under
`ignored/experiment-artifacts/2026-09-18-toy-eliminate/`, aggregated with
`python aggregate.py <dir>` into `lab/experiments/results/2026-09-18-toy-eliminate/`.
