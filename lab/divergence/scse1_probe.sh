#!/bin/bash
# H23 part 2: prove the probe change is a no-op on baseline data, THEN probe Stage 1.
# Pre-registration: lab/experiments/planned/2026-08-25-scse-stage1-initial-deviation.md
#
# Step 1 is a REGRESSION GATE, not a formality. `forcing_bias` changed its anchor from
# `points[0]["h"]` to `points[0]["e"]`. Those are the same tensor for a baseline model
# (h = e.clone()), so re-probing an H21 checkpoint MUST reproduce the stored b_t bitwise
# to the precision it was written at. If it does not, the anchor change moved a number it
# had no business moving, and every H23 reading is void.
set -eu
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/scse1
B=/home/wolfe/morph-scratch/seedsweep
export PYTHONPATH=$PWD

$PY lab/divergence/drift_probe.py --self-test

echo "=== REGRESSION GATE: re-probe baseline seed 3 with the new anchor ==="
$PY lab/divergence/drift_probe.py --config-name tul_a1 \
    --ckpt-dir checkpoints/morph/seedsweep-s3 --out $S/regress_s3.json
$PY - <<'PYEOF'
import json, sys
old = json.load(open("/home/wolfe/morph-scratch/seedsweep/drift_s3.json"))
new = json.load(open("/home/wolfe/morph-scratch/scse1/regress_s3.json"))
o = {r["step"]: [x["b_rel"] for x in r["forcing_bias"]] for r in old}
n = {r["step"]: [x["b_rel"] for x in r["forcing_bias"]] for r in new}
assert o.keys() == n.keys(), f"rung sets differ: {sorted(o)} vs {sorted(n)}"
worst = 0.0
for s in sorted(o):
    for a, b in zip(o[s], n[s]):
        worst = max(worst, abs(a - b) / max(abs(a), 1e-30))
d0 = {r["step"]: r["forcing_bias"][0]["delta0_rel"] for r in new}
print(f"  worst relative change in b_t across all rungs: {worst:.3e}")
print(f"  delta0_rel on the BASELINE model: {sorted(set(round(v, 12) for v in d0.values()))}")
if worst > 1e-9:
    print("  GATE FAILED: the anchor change moved b_t on a model where the two anchors "
          "are the same tensor. Every H23 reading is void until this is explained.")
    sys.exit(1)
if any(v > 1e-9 for v in d0.values()):
    print("  GATE FAILED: a baseline model reports a non-zero initial deviation, so "
          "`delta0_rel` is not measuring what it claims.")
    sys.exit(1)
print("  GATE PASSED: b_t unchanged, and the baseline reports Delta_0 = 0 exactly.")
PYEOF

echo
for sd in 0 1 2 3; do
  d=checkpoints/morph/scse1-s$sd
  if [ ! -d "$d" ]; then echo "!! seed $sd: no checkpoint dir $d -- skipped"; continue; fi
  echo "=== probing SCSE Stage 1 seed $sd ==="
  $PY lab/divergence/drift_probe.py --config-name tul_a1 \
      --overrides "training.batch_size=6,model.use_kernels=false,model.core_init_scale=0.1" \
      --ckpt-dir "$d" --out "$S/drift_s$sd.json"
done

echo
echo "################ H23 SCORECARD ################"
$PY lab/divergence/score_h23.py --scse $S --base $B --seeds 0 1 2 3
