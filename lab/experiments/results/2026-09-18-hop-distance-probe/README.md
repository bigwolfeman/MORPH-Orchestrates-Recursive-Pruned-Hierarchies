# Hop-distance probe artifacts (2026-09-19)

Record: `../../failures/2026-09-18-hop-distance-earning.md`. Instrument:
`lab/divergence/hop_distance_probe.py` (MORPH checkout `99bd7c9` on the 3070).

One JSON and one `.txt` log per arm, all at step 5000, 480 rows, batch 4, depths
1,2,3,6,9,12,16, hops 6, planted, seed 0:

- `hop_slot-spandec-strict_5000.*`: the null control (coda reach `all`, loop reach 0). P1, P2, P5.
- `hop_slot-spandec-strict-reach1_5000.*`: the second control (coda reach `all`, loop reach 1). P4, P5.
- `hop_slot-spandec-strict-prev-reach1_5000.*`: the test (coda reach `prev`, loop reach 1). P3, P5.

JSON layout per arm: `dce_profile` (mean ΔCE by hop, index 0 = own span), `complete_frac`,
`bins[h]` with `n_tokens`, `ce[depth]` and the bootstrapped `K1-K6` / `K3-K6`, and `planted`
with `control[depth]` and `g{0..6}.{ce,benefit}[depth]`. The `.log` files are gitignored
here, hence `.txt`.

Command (on the 3070, from `/mnt/bigdata/morph-instruments/MORPH`, see
`run_hop_reach.sh` and `run_hop_strict_after.sh` there):

```
python lab/divergence/hop_distance_probe.py \
  --ckpt <arm>=tul_<arm with _>=/mnt/bigdata/morph-instruments/<arm>_step_5000.pt \
  --rows 480 --batch 4 --depths 1,2,3,6,9,12,16 --hops 6 --planted --seed 0 \
  --out results/hop_<arm>_5000.json
```

Scored with the scratch scorer `hop_score.py` (P1-P5 as written in the record).
