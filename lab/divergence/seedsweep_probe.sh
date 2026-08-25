#!/bin/bash
# H21 part 2: probe every seed's checkpoints, then score against the pre-registration.
# Pre-registration: docs/experiments/planned/2026-08-24-tul-forcing-bias-predicts-divergence.md
#
# All four seeds are arm A1, so all four are probed on the slot path with the same config.
# The anchor is each seed's own h_0 = input_norm(x) on its own slot positions.
#
# The self-test runs FIRST and the script exits on its failure (set -e), because a probe
# that cannot reproduce its own identities cannot be trusted to read a checkpoint.
set -eu
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/seedsweep
export PYTHONPATH=$PWD

$PY lab/divergence/drift_probe.py --self-test

for sd in 0 1 2 3; do
  d=checkpoints/morph/seedsweep-s$sd
  if [ ! -d "$d" ]; then echo "!! seed $sd: no checkpoint dir $d -- skipped"; continue; fi
  echo "=== probing seed $sd ==="
  $PY lab/divergence/drift_probe.py --config-name tul_a1 \
      --ckpt-dir "$d" --out "$S/drift_s$sd.json"
done

echo
echo "################ H21 SCORECARD ################"
$PY lab/divergence/score_h21.py --dir $S --seeds 0 1 2 3
