#!/bin/bash
# Full-SCSE experiment, part 2: the P1 validity gate.
# Pre-registration: lab/experiments/planned/2026-08-25-scse-full-method.md
#
# Step 1 is a REGRESSION GATE, not a formality. `drift_probe.step_at` and `forcing_bias`
# both grew an SCSE branch. Those branches key off `point["scse"]`, which is False on every
# capture from a non-SCSE model, so re-probing an H21 baseline checkpoint MUST reproduce the
# stored b_t bitwise. If it does not, the branch is firing where it must not and every P1
# reading below is void.
set -eu
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/scse2/probe
mkdir -p $S
export PYTHONPATH=$PWD

$PY lab/divergence/drift_probe.py --self-test

echo "=== REGRESSION GATE: re-probe an H21 BASELINE checkpoint with the patched probe ==="
$PY lab/divergence/drift_probe.py --config-name tul_a1 \
    --ckpt-dir checkpoints/morph/seedsweep-s3 --out $S/regress_s3.json
$PY - <<'PYEOF'
import json, sys
old = json.load(open("/home/wolfe/morph-scratch/seedsweep/drift_s3.json"))
new = json.load(open("/home/wolfe/morph-scratch/scse2/probe/regress_s3.json"))
o = {r["step"]: [x["b_rel"] for x in r["forcing_bias"]] for r in old}
n = {r["step"]: [x["b_rel"] for x in r["forcing_bias"]] for r in new}
assert o.keys() == n.keys(), f"rung sets differ: {sorted(o)} vs {sorted(n)}"
worst = max(abs(a - b) / max(abs(a), 1e-30)
            for s in sorted(o) for a, b in zip(o[s], n[s]))
print(f"  worst relative change in b_t on BASELINE data: {worst:.3e}")
if worst > 1e-9:
    print("  GATE FAILED: the SCSE branch moved a baseline number. P1 is void.")
    sys.exit(1)
print("  gate passed: the SCSE branch is inert on non-SCSE captures")
PYEOF

echo "=== probe both arms ==="
for sd in 1 2 3 4; do
  for arm in ctrl scse; do
    d=checkpoints/morph/scse2-$arm-s$sd
    if [ -d "$d" ]; then
      echo "-- $arm s$sd --"
      # The probe REBUILDS the model from config, so the SCSE arm must be told to build
      # SCSE. Loading an SCSE checkpoint into a baseline model would either fail on the
      # missing keys or, worse, load them into nothing and probe the wrong operator.
      OV="training.batch_size=6,model.use_kernels=false"
      if [ "$arm" = "scse" ]; then OV="$OV,model.scse_enabled=true"; fi
      $PY lab/divergence/drift_probe.py --config-name tul_a1 --overrides "$OV" \
          --ckpt-dir $d --out $S/$arm-s$sd.json
    else
      echo "-- $arm s$sd: no checkpoint dir, skipped --"
    fi
  done
done

echo "=== score against the pre-registered predictions ==="
$PY lab/divergence/score_scse.py --dir /home/wolfe/morph-scratch/scse2/runs --probe-dir $S
