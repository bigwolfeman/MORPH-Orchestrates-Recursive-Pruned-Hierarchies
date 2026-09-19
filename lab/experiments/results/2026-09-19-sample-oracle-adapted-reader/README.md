# Sample-oracle gate, second reading: adapted reader (2026-09-19)

Record: `../../failures/2026-09-19-sample-oracle-adapted-reader.md`. First reading (frozen
reader, strict ruler @ 5000): `../2026-09-18-sample-oracle-gate/`.

- `sample_oracle_uf_30000.json`: `lab/divergence/sample_oracle_probe.py` on
  `tul-code-target-uf/step_30000.pt` (config `tul_code_target_uf`, Spark, 96 rows, batch 4,
  N = 16, sigmas 0.3 and 1.0, entry depths 1 and 6, exit depth 6, seed 0, exit site
  `code_proj`). Per cell: `det`, `mean_samples`, `oracle_N`, `gain_N` with row bootstrap,
  `cos_exit` on entry cells.
- `sample_oracle_uf_30000.txt`: the run log (the `.log` is gitignored here).

Command (on the Spark, from `~/morph-to`):

```
python lab/divergence/sample_oracle_probe.py \
  --ckpt uf=tul_code_target_uf=/home/wolfe/morph-instruments/tul-code-target-uf_step_30000.pt \
  --rows 96 --batch 4 --samples 16 --sigmas 0.3,1.0 --entry-depths 1,6 --exit-depth 6 \
  --out /home/wolfe/morph-instruments/results/sample_oracle_uf_30000.json
```
