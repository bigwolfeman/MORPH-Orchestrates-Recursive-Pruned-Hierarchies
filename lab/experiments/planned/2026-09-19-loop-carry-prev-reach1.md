# Loop carry: if a cell KEEPS what it read from its neighbour, does the far hop stop decaying?

Status: planned

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
