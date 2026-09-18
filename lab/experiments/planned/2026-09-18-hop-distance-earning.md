# Experiment: does the slot loop earn depth only where the content is HOPS away?

Status: planned
Date: 2026-09-18
Owner: Claude (session f9558148), sub-agent HopProbe, for Wolfe.

Frozen before any real-checkpoint number exists. At filing time the probe has run on no
checkpoint on any host; the only numbers in this tree are the 17 CPU unit tests of
`tests/test_hop_distance_probe.py` and the Hydra compose of the three configs below.

Arc: [`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).
Nearest prior record: [`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md).

## Question

Twelve slot-loop arms read a token K1−K6 of about +0.002 nats while the plain looped model
reads 0.033. Every stability lever is on both, so the flat K-curve is not a stability
story. The hypothesis on trial is mechanical: **a looped transformer earns depth through
GATHER** — pass k reads pass k−1 states of OTHER positions, one more attention hop per
pass — so a pass can only be worth something to a token whose needed content is more than
one hop away.

Under the strict geometry (`morph/model/tul_layout.py::tg_strict_allow`) the routes are
exactly:

| stage | relation |
| --- | --- |
| prelude | a token sees its own span's tokens; a cell sees its own span and its own earlier cells |
| loop | cell k at pass t reads cells k−`loop_reach` … k at pass t−1 (`loop_reach` 0 = every earlier cell) |
| coda | a token of span j sees its own span plus the prefix cells of earlier slots — all (`tg_coda_prefix_reach: all`) or only cell j−1 (`prev`) |

So on the SHIPPED strict arm (`all`, `loop_reach` 0) span i's content reaches a token of
span j in ONE hop at every depth. That arm cannot show hop-graded earning even if the
hypothesis is completely right, which makes it the null control and the instrument check,
not the test. The test is `prev` + `loop_reach` 1, where span j−h reaches cell j−1 only at
pass h−1.

The question in one line: **is per-token depth earning graded by the hop distance of the
span that token needs, and does CE(d) plateau at d ≈ h−1 on the arm where that distance is
forced?**

## Hypothesis

Depth earning is depth DEPENDENCE created by reachability. A token earns from pass k only
when the content it needs cannot arrive in fewer than k hops. On `strict` every span is one
hop from every later token, so every bin reads flat and the twelve flat K-curves are
explained by the geometry rather than by the loop failing to learn. On `prev_reach1` the
far bins must earn, and their CE(d) must stop falling at about the depth the chain needs.

If the far bins on `prev_reach1` read flat too, gather is not the mechanism and the
hypothesis is dead — on the one arm built so that gather is the ONLY route.

## Predictions

Numbers, not directions. `K1−K6` is (mean CE at forced depth 1) − (mean CE at forced depth
6) over the tokens of a bin, paired over rows with the sweep's bootstrap; "flat band" means
the point estimate lies in [−0.010, +0.010] nats.

**P1 — `slot-spandec-strict` @ 5000 (reach `all`, loop_reach 0): every hop bin is flat.**
For every bin h ∈ 1…6, K1−K6 lies in the flat band, and
`max_h (K1−K6) − min_h (K1−K6) < 0.015` nats. The bins do not rise monotonically with h
(Spearman ρ between h and K1−K6 over the six bins, |ρ| < 0.6).

**P2 — the localiser has teeth on the same checkpoint.** The corruption profile is NOT
flat: mean ΔCE(1) > 0.05 nats, ΔCE(1) > ΔCE(6), and the own-span reading (the h = 0
forward, which corrupts each span's first half and scores its second) is the largest of the
seven, > 0.15 nats. **If ΔCE(1) < 0.02 nats the localiser is inert, h\* is noise, and P1
and P3 cannot be read at all** — that is a protocol failure, filed in `failures/`, not a
null result about depth.

**P3 — `slot-spandec-strict-prev-reach1` @ 5000 (reach `prev`, loop_reach 1): the far bins
earn, and they plateau.**
* bins h ≥ 3: K1−K6 ≥ +0.020 nats, and at least 4x the h = 1 bin's value;
* bins h = 1 and h = 2: inside the flat band (their content is reachable at pass 0 or 1);
* the plateau, in the h = 5 and h = 6 bins: `CE(h−1) − CE(6) ≤ 0.20 · (CE(1) − CE(6))`.
  The curve has spent most of its fall by the depth the chain needs and little after.

**P4 — `slot-spandec-strict-reach1` @ 5000 (reach `all`, loop_reach 1): flat, like P1.**
The coda still reads every cell directly, so a reach limit INSIDE the loop forces no hop.
Every bin in the flat band and the spread under 0.015. If this arm shows P3's grading, the
grading is not about hops and the whole reading is rejected.

**P5 — the planted copy pair.** g = 0 (the rare source planted inside the scoring token's
own span) is the positive control: `CE_control − CE_planted(g=0) ≥ 0.5` nats at every
depth on every arm. **If that benefit is < 0.1 nats the model cannot copy a rare id at all,
the planted instrument is inert on this checkpoint (a floor effect at ln V ≈ 10.8), and no
g ≥ 1 cell says anything** — stated in advance so a flat planted table is not read as
evidence. Given a working g = 0:
* on `strict` and `reach1`: `benefit(g, 6) − benefit(g, 1)` in [−0.05, +0.05] for every g;
* on `prev_reach1`: `benefit(g, 6) − benefit(g, 1) ≥ +0.10` nats for g ≥ 3, and inside
  [−0.05, +0.05] for g = 1.

**Falsification, stated both ways.** P1 or P4 failing means the localiser is binning on
something other than hop distance (row position is the obvious candidate) and the
instrument is rejected before P3 is read. P3 failing means depth earning is not
gather-limited even where gather is the only route, and the gather hypothesis is dead.

## Method

### The instrument

`lab/divergence/hop_distance_probe.py`, built in this change, two instruments in one file.

1. **Corruption localiser.** For c = 0…H (H = 6) one forward replaces the tokens of every
   span i ≡ c (mod H+1) with tokens taken from another row of the same batch. `slot_layout`
   is a forward ARGUMENT, so the boundaries, the cells, the masks and every shape are
   byte-identical to the clean forward and only the token content moves. A token of span j
   is scored in the H phases that do not corrupt its own span, and in phase c its nearest
   corrupted span sits (j − c) mod (H+1) spans back, so each token gets exactly one reading
   at each distance 1…H. ΔCE(h) = that CE minus the clean CE at the same position and the
   same depth; h\* = argmax_h ΔCE(h). The clean per-token K-curve is then binned by h\*.
   Three readings are dropped and each is named in the JSON: a corrupted token itself, a
   token whose LABEL sits in a corrupted span, and a token of span j at a distance h > j
   (the phase corrupted nothing that token can causally read — left in, that wrap-around
   drags h\* down for the first H spans of every row). Only tokens with a finite reading at
   every distance 1…H are binned, so a binned token has span index ≥ H.
   The confound that remains, named: phase c corrupts every span at distance ≡ h (mod H+1)
   at once, so ΔCE(h) is h together with h+7, h+14 … Distance 0 is unmeasurable this way
   and gets its own forward (corrupt each span's first half, score the second); it is
   reported apart and kept OUT of the argmax.
2. **Planted copy pair** (`--planted`). A token id absent from the whole batch is written
   at a random position of span s−g AND as the first token of span s+1; the CE of
   predicting that repeat is read at the LAST token of span s, the position whose label it
   is (`pack_tul_row`: `labels[tok_pos] = ids[i+1]`). The scoring positions are identical
   for every g and for the no-source control, so the copy benefit is paired at the token
   level. Sites are spaced H+2 spans apart so no two plants share a span. This instrument
   assumes nothing about what a token needs and does not depend on the localiser.

Depth forcing is `core_depth_sweep.py`'s (`tul.slot_mean_depth`, `slot_max_depth`, and
`slot_depth_fixed` on a k-fixed arm). Depth 0 is refused: `_sample_slot_depths` reads
`tc.slot_mean_depth or self.cfg.mean_depth`, so 0 silently becomes the trained mean. The CE
map comes from `core_depth_sweep.ce_maps`, i.e. from `model.tul_forward_ablated`, so the TG
masks are the model's own — a bare `_tul_front` scores a strict arm from an unrestricted
prelude and flips the sign of the reading.

Gate: `tests/test_hop_distance_probe.py`, 17 CPU tests over the schedule, the wrap-around
drop, the plant (through the REAL packer) and the aggregation.

### Runs

Probe settings for every arm: `--rows 480 --batch 6 --depths 1,2,3,6,9,12,16 --hops 6
--planted --seed 0`, seq 1024, on the Spark or the 3070 (`docs/cookbook/running-probes-on-
the-second-host.md`), never on the 5090.

| arm | config | checkpoint | role |
| --- | --- | --- | --- |
| `slot-spandec-strict` | `tul_slot_spandec_strict` | on disk, step 5000 | P1, P2, P5 — null control + instrument check |
| `slot-spandec-strict-prev-reach1` | `tul_slot_spandec_strict_prev_reach1` | **must be retrained** | P3 — the test |
| `slot-spandec-strict-reach1` | `tul_slot_spandec_strict_reach1` | **must be retrained** | P4 — the second control |

The two reach arms ran on 2026-09-12 (K1−K6 0.013–0.016 against 0.002 for plain strict) and
their checkpoints were purged in the 2026-09-13 retention pass; only the configs remain.

### The retrain, which Wolfe must approve before anything runs

Two 5,000-step runs, configs UNCHANGED, one at a time on the 5090 (a run is live at filing
time, so these are queued, not launched). No training was started by this change.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/wolfe/morph-to PYTHONUNBUFFERED=1
cd /home/wolfe/morph-to
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
Q=/home/wolfe/morph-scratch/arc

mkdir -p $Q/slot-spandec-strict-prev-reach1
$PY -m morph.training.train --config-name tul_slot_spandec_strict_prev_reach1 \
  hydra.run.dir=$Q/slot-spandec-strict-prev-reach1/hy \
  training.grad_probe_path=$Q/slot-spandec-strict-prev-reach1/probe.jsonl \
  wandb.name=slot-spandec-strict-prev-reach1

mkdir -p $Q/slot-spandec-strict-reach1
$PY -m morph.training.train --config-name tul_slot_spandec_strict_reach1 \
  hydra.run.dir=$Q/slot-spandec-strict-reach1/hy \
  training.grad_probe_path=$Q/slot-spandec-strict-reach1/probe.jsonl \
  wandb.name=slot-spandec-strict-reach1
```

Checkpoints land in `/home/wolfe/morph-to/checkpoints/morph/<wandb.name>/step_5000.pt`.

### How the record closes

Fill Results and Verdict from the three JSONs plus their npz, then file under
`successes/` if P1–P5 held as written and `failures/` otherwise — including the two
protocol failures P2 and P5 name in advance. The predictions above are not edited after a
run; a method change gets a numbered `## Amendment N` with its date.

If the reading changes what ships, an Agent Note goes in the same change.
