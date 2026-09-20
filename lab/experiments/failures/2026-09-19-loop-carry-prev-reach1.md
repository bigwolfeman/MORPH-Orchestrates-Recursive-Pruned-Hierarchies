# Loop carry: if a cell KEEPS what it read from its neighbour, does the far hop stop decaying?

Status: failure

Date: 2026-09-19. Commit: the sha in the queue lines below. Arms
`slot-spandec-strict-prev-reach1-carry-sum` and `-carry-gate`, 5,000 steps, seq 1024,
batch 6, ramp 1,000, `norm_match`, strict geometry, coda reach `prev`, `loop_reach 1`,
span decoder on. Written and committed BEFORE either arm ran on any GPU. Nothing here is
fitted to a result: the only numbers measured so far are the CPU wire check's at the foot
of the Method, and they are labelled as such.

## Question

The hop-distance probe's second pass
(`../failures/2026-09-19-hop-distance-plateau-and-dilution.md`) settled the mechanism
behind the reach-1 ruler's staircase. With the loop's cross-cell read cut after pass 0, so
that nothing new arrives, content the loop had ALREADY carried in decays under the cell's
own later passes — a planted copy two spans back falls 0.148 → 0.035 nats of benefit
between depth 1 and depth 6, and the h = 2 bin loses 0.083 nats — while the cell's OWN
span, re-supplied every pass by the per-layer x0 / bigram injection, is refined instead
(0.181 → 0.291). The ruler therefore earns +0.064 at h = 3, +0.054 at h = 4, +0.029 at
h = 5 and −0.020 at h = 6: the farther the source, the more passes its content spends
decaying before the coda reads it.

**If neighbour content is re-supplied every pass the way own-span content is, does the
decay stop, and does the far hop start earning?**

`tul.loop_carry` (`.agents/notes/proposed/architecture/2026-09-19-loop-carry-reinjection.md`,
built at commit `24d8565`) is that mechanism and nothing else: a per-cell state that
accumulates core layer 0's window-branch read and re-injects it, RMS-matched to the
carrier, at the entry of every later pass.

## Hypothesis

The decay is an artefact of WHERE the carried content sits, not of the map. Own-span
content survives six passes because it is re-added at every layer of every pass; neighbour
content enters once and is then only re-processed. Re-supplying it removes the asymmetry,
so the planted g = 2 pair under `--cut-after 1` should stop decaying, and the bins whose
source is far enough that the content must survive several passes (h = 5, h = 6) should
gain the most. The near bins should be untouched or slightly worse: the carry is one more
thing competing for the cell's capacity, and g = 1's uncut decay (0.181 → 0.137) is
already displacement by later arrivals.

`gate` should not beat `sum` at 5,000 steps. Its only advantage is weighting a new read
against what it holds, the weighting starts at exactly ½ everywhere, and it carries 2.1 M
extra parameters that nothing in the objective pushes on directly.

I do not expect the carry to widen the per-pass reach and the code refuses to let it: the
arrival depth `h − 1` is a property of the attention relation, which is untouched. A
violation there would mean the capture is not the tensor this note says it is.

## Predictions (frozen)

The five clauses are the Agent Note's Acceptance criteria, verbatim, read against the
ruler's own readings in `../failures/2026-09-19-hop-distance-plateau-and-dilution.md`.
All on `slot-spandec-strict-prev-reach1` geometry at 5,000 steps, 480 rows, the same
instrument. The headline arm is `sum`; `gate` is read against `sum`.

- **P1 (the mechanism clause).** Planted g = 2 under `--cut-after 1`: benefit at depth 6
  within 0.03 of its depth-1 value (ruler: 0.148 → 0.035). If this fails the carry is not
  doing what it is for.
- **P2 (the far hops).** Per-hop K1−K6: h = 5 ≥ +0.045 and h = 6 ≥ +0.010 (ruler +0.029
  and −0.020); h = 3 and h = 4 not below the ruler's minus 0.010.
- **P3 (own-span content not paid for it).** h = 1 within 0.005 of the ruler and the g = 0
  pair benefit within 0.05.
- **P4 (the contracts).** Bit-identity at `loop_carry: none` (the
  `tests/test_tul_prefix_source.py` constants) and a contract test that under `sum` at
  forced depth T the carry at cell k equals the sum of its T reach reads.
- **P5 (stability).** No detonation and `loop/core_gain_t0` under the ruler's band over
  the 5,000 steps; the smoke prints the same peak memory within 5 %.

**Filing rule.** `successes/` only if P1, P2, P3 and P5 all hold on the `sum` arm. P4 is
already measured (see the Method's last paragraph) and is a gate on the build, not on the
run: if it ever fails the run is void, not a failure.

**Rejection of the reading.** If P1 holds and P2 fails, the decay was not what limited the
far hops and the note's Problem section is withdrawn in favour of whatever the per-hop
table then shows. If P1 fails while the arm's whole-arm CE improves, the improvement is
NOT attributed to the carry's stated mechanism and no further carry arm is queued until a
new instrument separates the two.

## Method

**Arms.** One config, two queue lines, one factor apart:

| name | overrides on `tul_slot_spandec_strict_prev_reach1_carry.yaml` | isolates |
| --- | --- | --- |
| `...-carry-sum` | none (`tul.loop_carry: sum` is in the yaml) | the carry, with no new parameter |
| `...-carry-gate` | `tul.loop_carry=gate` | the learned write weighting, +2.1 M params |

The standing partner is `slot-spandec-strict-prev-reach1` at 5,000 steps — the ruler whose
hop table the predictions quote. It is ONE factor away from the `sum` arm: same seed, same
packer, same prefix width, same parameter set (the `sum` carry adds none). The `gate` arm
is 2.1 M parameters wider, so it is read against `sum` and not against the ruler; that
confound is named in the Agent Note's Build notes.

**Training.** The 5090 runs ONE trainer through the queue in
`/home/wolfe/morph-scratch/arc/run_recon.sh`. Two lines are appended at the end of
`recon_arms.txt`, kind `slot`, sweeps 2500,5000, last 5000, smoke 1, outdir
`/home/wolfe/morph-scratch/arc/results/2026-09-19-loop-carry`. The runner's own 12-step
smoke is the first GPU execution of this code — no GPU run of any kind preceded this file.

**Scoring, in reading order.**

1. `lab/divergence/hop_distance_probe.py` on the 3070, checkpoint step 5000, 480 rows,
   batch 4, hops 6, `--planted --planted-len 2`, depths `1,2,3,4,5,6,7,8,12,16` — the
   per-hop K-curve and the planted pair's decay row (P2, P3's g = 0 clause).
2. The same probe with `--cut-after 1`, depths `1,2,3,6` — P1, the mechanism clause. This
   is the run that decides the note.
3. `carry/rms_t{1..6}` and `carry/gate_mean_t{1..6}` from the trainer, read per pass and
   NEVER summed (`depth-summing-instruments-hide-pass-trades`): a carry whose RMS stops
   growing after pass 2 has stopped keeping anything.
4. `loop/core_gain_t0` and the sustained tripwire over the whole run — P5.
5. The runner's own `core_depth_sweep` at 2,500 and 5,000 and `worth_profile` at 5,000, as
   context. The whole-arm K1−K6 is NOT the score: a whole-arm mean of +0.015 hid +0.064
   and −0.020 in the probe's first pass.

**Amendments.** The Predictions section is frozen. If the method has to change it is
amended here with the date and the reason; if the change makes the run unable to answer
the question, the run is rejected and a new planned file is opened.

**What has actually been run, and none of it is a result.** CPU only, at commit
`24d8565`. `tests/test_tul_loop_carry.py` 31 passed; `test_tul_prefix_source` 54,
`test_tul_strict_geometry` 52, `test_tul_fan` 21, `test_tul_setup_keys` 8,
`test_tul_forward` 40, `test_checkpoint_compat` 5, `test_slot_gain_reg` 9,
`test_tg_restrict` 21, `test_tul_slot_register` 54 — all passed (P4 is inside the first
two). Both configs compose through Hydra and `build_tul_runtime`. A wire check on the real
config at `data.seq_len=256`, batch 1, `CUDA_VISIBLE_DEVICES=""`: 292.5 M parameters
(294.6 M on `gate`), forward 2.1 s, backward 2.0 s, every gradient finite,
`carry/rms_t1..t6` 0.112 → 0.508 and `carry/inject_ratio` exactly 1.0 at every pass, on a
random model over synthetic spans — which predicts nothing about a trained one.

**Unverified at the time of writing:** no GPU run, `torch.compile` untested, peak memory
untested (P5's memory clause is read off the runner's smoke, not from anything here).

## Results

Read 2026-09-20 (sum: runner readouts in `results/2026-09-19-loop-carry/`; gate: the
runner's readouts LOAD_FAIL on an override arm (`runner-smoke-skips-extra`), so its sweeps,
worth profile and state probe ran by hand on the Spark from the same template and are
copied into the same directory; hop probes on the 3070 at commit `908fe61`, 480 rows,
batch 4, hops 6, `--planted --planted-len 2`). The ruler's pair rows are the same
instrument's `hop2_prev-reach1-pair-{alldepths,cut1}_5000.json` in
`results/2026-09-19-hop-distance-pass2/`. Scored with the scratch `hop_score.py` that
scored the ruler.

**Whole-arm depth sweep at 5,000 (480 rows, token CE), context only.**

| arm | d1 | d3 | d6 | K1−K6 | K3−K6 |
|---|---|---|---|---|---|
| ruler `prev-reach1` | 4.3742 | 4.3620 | 4.3578 | +0.0163 | +0.0042 |
| `carry-sum` | 4.4958 | 4.4561 | 4.4554 | +0.0404 | +0.0008 |
| `carry-gate` | 4.4167 | 4.4094 | 4.4081 | +0.0086 | +0.0013 |

Both carry arms sit behind the ruler at depth 6 (sum +0.098, gate +0.050). The sum arm's
larger K1−K6 is a worse depth 1, not a better depth 6: on this arm the carry is injected
only from pass 2 (`transformer.py`, `_cy = _carry_state if (... and t > 0)`), and the
planted rows below show that NO neighbour content reaches a cell at depth 1.

**Per-hop K1−K6 (pair, all depths, 5,000; sum arm; gate arm pending the 3070 run).**

| h | ruler | `carry-sum` | sum per depth (d1, d2, d3, d6, d16) |
|---|---|---|---|
| 1 | +0.0109 | +0.1033 [+0.0995, +0.1071] | 4.5436, 4.4417, 4.4420, 4.4404, 4.4532 |
| 2 | −0.0169 | +0.0846 [+0.0794, +0.0898] | 4.6473, 4.5641, 4.5655, 4.5627, 4.5622 |
| 3 | +0.0640 | −0.0209 [−0.0248, −0.0169] | 4.4455, 4.4921, 4.4818, 4.4664, 4.4407 |
| 4 | +0.0535 | −0.0742 [−0.0793, −0.0690] | 4.1809, 4.2491, 4.2465, 4.2550, 4.2900 |
| 5 | +0.0288 | −0.0878 [−0.0947, −0.0809] | 4.4917, 4.5692, 4.5660, 4.5794, 4.6279 |
| 6 | −0.0203 | −0.0957 [−0.1024, −0.0890] | 4.5357, 4.6209, 4.6173, 4.6314, 4.6807 |

The sum arm inverts the ruler's staircase (Spearman(h, K1−K6) = −1.000). The near bins
gain everything at pass 2, which is the first pass the carry is injected, and the far bins
are best at depth 1 and get monotonically worse with every pass after 2.

**Planted pair, benefit at the target cell (nats), sum arm vs ruler.**

| row | ruler d1 → d2 → d3 → d6 | `carry-sum` d1 → d2 → d3 → d6 |
|---|---|---|
| g = 2, `--cut-after 1` (P1) | +0.148 → +0.073 → +0.050 → +0.036 | +0.000 → +0.108 → +0.147 → +0.210 |
| g = 1, `--cut-after 1` | +0.181 → +0.245 → +0.271 → +0.290 | +0.003 → +0.090 → +0.124 → +0.170 |
| g = 0, `--cut-after 1` | +0.566 → +0.577 → +0.582 → +0.585 | +0.679 → +0.666 → +0.653 → +0.624 |
| g = 2, uncut | +0.148 → +0.108 → +0.088 → +0.071 (d16 +0.062) | +0.000 → +0.108 → +0.097 → +0.090 (d16 +0.079) |
| g = 0, uncut, d6 | +0.558 | +0.659 |

**Trainer instruments.** `carry/rms_t1..t6` at step 4980: sum 4.97, 22.5, 39.0, 54.2, 66.3,
75.4; gate 9.43, 29.3, 48.0, 65.2, 79.0, 89.3 (both 0.11 → 0.50 at step 0). The gate's
`carry/gate_mean_t1..t6` 0.38, 0.28, 0.27, 0.27, 0.27, 0.27. `loop/core_gain_t0`: ruler
max 1.95 (last 1.27), sum max 2.82e5 at 4921 (last 2.46e5), gate max 573 at 4967 (last
454). `preclip/total` max at step ≥ 200: ruler 33.9, sum 6.1e4 at 3652 (ONE row; the
two-row tripwire never fired), gate 53.5. Smoke peak memory: ruler 12.95 GB, sum 12.96 GB,
gate 12.96 GB.

**Scoring on the `sum` arm.**

- **P1: fails by the letter.** Under `--cut-after 1` the g = 2 benefit at depth 6 is
  +0.210 against +0.000 at depth 1: 0.210 apart, the clause asked for 0.03. The clause's
  premise is false on this arm: content two spans back arrives at pass 2, not pass 1,
  because the carry is the only route and it is injected from pass 2. Read for its intent
  (does a re-supplied read stop decaying?) the answer is yes: +0.108 at d2 → +0.210 at d6
  against the ruler's +0.148 → +0.036. That is recorded, not scored.
- **P2: fails, all four clauses.** h = 5 −0.088 (asked ≥ +0.045), h = 6 −0.096 (asked
  ≥ +0.010), h = 3 −0.021 and h = 4 −0.074 (asked not below the ruler's +0.064 / +0.054
  minus 0.010).
- **P3: fails.** h = 1 is +0.103 against the ruler's +0.011 (asked within 0.005); the g = 0
  benefit at depth 6 is +0.659 against +0.558 (asked within 0.05).
- **P4: holds** (measured at build, Method's last paragraph; a gate on the run, not a
  result).
- **P5: fails.** `loop/core_gain_t0` reaches 2.82e5 against the ruler's band of at most
  1.95; no detonation (the tripwire never fired) and the memory clause holds (12.96 vs
  12.95 GB).

**The `gate` arm, read against `sum`.** Depth-6 CE 4.4081, 0.047 better than sum and 0.050
behind the ruler; K1−K6 +0.0086. Its carry RMS grows the same way (89 at t6) and its gate
mean settles at 0.27 from pass 3 on, so the learned weighting does not bound the state.
`core_gain_t0` 573: the same scale mode, 500x smaller. Its hop tables (3070, ALLDEPTHS
exit 0 at 21:08 UTC, CUT1 exit 0 at 21:51 UTC, same instrument and rows):

| h | ruler | `carry-sum` | `carry-gate` | gate per depth (d1, d2, d3, d6, d16) |
|---|---|---|---|---|
| 1 | +0.0109 | +0.1033 | +0.0028 [+0.0015, +0.0040] | 4.2702, 4.2661, 4.2663, 4.2674, 4.2780 |
| 2 | −0.0169 | +0.0846 | +0.0530 [+0.0510, +0.0553] | 4.6372, 4.5853, 4.5822, 4.5841, 4.5899 |
| 3 | +0.0640 | −0.0209 | +0.0359 [+0.0337, +0.0383] | 4.6665, 4.6705, 4.6579, 4.6306, 4.5945 |
| 4 | +0.0535 | −0.0742 | −0.0331 [−0.0355, −0.0306] | 4.2660, 4.2948, 4.2974, 4.2991, 4.3016 |
| 5 | +0.0288 | −0.0878 | −0.0458 [−0.0487, −0.0430] | 4.5313, 4.5647, 4.5674, 4.5771, 4.5983 |
| 6 | −0.0203 | −0.0957 | −0.0464 [−0.0497, −0.0436] | 4.6796, 4.7131, 4.7159, 4.7260, 4.7475 |

Planted, gate: g = 1 uncut +0.173 → +0.124 (d1 → d6; ruler +0.181 → +0.137), cut-1
+0.173 → +0.131 (ruler +0.181 → +0.290); g = 2 uncut +0.024 → +0.065 (ruler +0.148 →
+0.071), cut-1 +0.024 → +0.092 (ruler +0.148 → +0.036); g = 0 flat at +0.56.

Read against `sum`: the gate halves the damage (h = 5 −0.046 against −0.088, h = 6
−0.046 against −0.096) and keeps a direct one-back read at depth 1 (g = 1 +0.173, where
`sum` has +0.003), but two-back content still arrives only from pass 2 (g = 2 +0.024 at
d1 against the ruler's +0.148) and the far bins still get worse with every pass after 2.
The h = 3 bin is the one place the gate beats the ruler's shape (it keeps falling to
d16, 4.5945, where the ruler settles by pass 3); it is bought with the far bins. P1 by
the letter fails on the gate too (d6 − d1 = +0.068 > 0.03) and by intent holds
(re-supply stops the decay: +0.069 → +0.092 under the cut). The verdict does not read
the gate (the filing rule reads `sum` only).

## Verdict

**Failure.** P1, P2, P3 and P5 fail on the `sum` arm; the filing rule needed all four.

## Updated hypothesis

1. The carry as built is not one factor from the ruler. It REPLACES the pass-1 read
   rather than adding to it: at depth 1 a carry cell holds no neighbour content at all
   (planted g = 1 and g = 2 both +0.000 at d1, h = 1 K1−K6 +0.103 from a depth-1 rung
   with nothing in it). Any next carry keeps the direct read at pass 1 and adds the
   re-supply on top.
2. Re-supply does stop the decay. That is the one clause of the Hypothesis that survived
   (g = 2 under the cut: +0.108 → +0.210 instead of +0.148 → +0.036). The question's
   second half, does the far hop start earning, is answered no, and the reason is not
   decay: with the read uncut, the state is a plain sum of every pass's read, so content
   h spans back is one part in T of what gets injected at pass T. The far bins lose to
   DILUTION and get worse with every pass after 2; the same dilution the ruler's near/far
   bins showed (`2026-09-19-hop-distance-plateau-and-dilution.md`), now made worse by
   accumulating it.
3. The state is unbounded and it runs. Carry RMS 0.5 → 75 over training and 5 → 75 across
   the six passes at 5k, and the first-iteration gain 1.24 → 2.8e5 on the same clock. The
   RMS match at the injection site bounds what is injected, not the state, and the gate
   (mean 0.27) scales the read but never subtracts. A next carry is bounded by
   construction: a running MEAN or an EMA with a fixed target RMS, so pass T injects a
   vector whose norm does not grow with T or with training.
4. `gate` did not beat `sum` by enough to matter (0.047 at depth 6, both behind the ruler)
   and the prereg's expectation that it would not stands.

No carry arm is queued from this file. The next planned file, if Wolfe wants one, is a
bounded carry that keeps the pass-1 read, scored on the same per-hop table with the
dilution reading as the thing to beat (h = 5 and h = 6 must not get worse with depth).
