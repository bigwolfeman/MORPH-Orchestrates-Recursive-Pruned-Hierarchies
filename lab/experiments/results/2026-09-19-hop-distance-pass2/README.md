# Hop-distance probe, second pass: artifacts (2026-09-19)

Record: `../../failures/2026-09-19-hop-distance-plateau-and-dilution.md`. First pass:
`../2026-09-18-hop-distance-probe/`. Instrument: `lab/divergence/hop_distance_probe.py` at
`c813b1a` (runs 1-4) and `cd4e111` (the `--row-offset` re-run), shipped by scp onto the 3070's
`99bd7c9` checkout; the md5 of the file that ran is the first line of `run_hop_pass2.txt`.

All runs: `slot-spandec-strict-prev-reach1` @ 5000 unless named, 480 rows, batch 4, hops 6,
`--planted`. One JSON and one `.txt` log each:

- `hop2_prev-reach1-pair-alldepths_5000.*`: depths 1-8, 12, 16, seed 0, `--planted-len 2`. P1, P2, P4.
- `hop2_prev-reach1-pair-cut1_5000.*`: depths 1, 2, 3, 6, `--planted-len 2 --cut-after 1`. P3.
  The cut re-bins the tokens (ΔCE beyond h = 2 is exactly 0); read within-run contrasts only.
- `hop2_prev-reach1-seed1_5000.*`: depths 1, 2, 3, 6, `--seed 1`, single token. The run as
  launched for P5; its bins are identical to seed 0 because the seed never chose the rows.
  Kept as the record of Method amendment 1.
- `hop2_prev-reach1-rows480_5000.*`: depths 1, 2, 3, 6, `--seed 1 --row-offset 480`, single
  token. The P5 reading on a disjoint later stretch of the validation stream.
- `hop2_strict-pair_5000.*`: `slot-spandec-strict` @ 5000, depths 1, 6, `--planted-len 2
  --skip-localiser`. P4's g = 0 clause on the null arm.
- `run_hop_pass2.txt`: the launcher log (`run_hop_pass2.sh` then `run_hop_pass2b.sh` on the
  3070), with the probe md5 and every START/DONE line.

JSON layout as in the first pass, plus `cut_after`, `planted_len` and `row_offset` in the arm
dict and `planted.planted_len`.
