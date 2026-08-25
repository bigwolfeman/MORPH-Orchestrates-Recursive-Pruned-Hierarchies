#!/bin/bash
# Stage 0, part 2: probe both arms' checkpoints, then score against the pre-registration.
# Pre-registration: docs/experiments/planned/2026-08-24-tul-forcing-bias-arm-control.md
#
# Each arm is probed on the code path it ACTUALLY runs, which is the comparison the
# experiment asks for:
#   A0 builds no TUL layer at all (284.0M params against A1's 286.1M) and loops the core over
#      every token position, so it is probed with --config-name tul_a0 --token-path.
#   A1 loops the core over ~57 slot positions, so it is probed on the slot path.
# The anchor in each case is that arm's own h_0 = input_norm(x) on its own positions.
set -eu
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/stage0
export PYTHONPATH=$PWD

$PY lab/divergence/drift_probe.py --self-test

$PY lab/divergence/drift_probe.py --config-name tul_a0 --token-path \
    --ckpt-dir checkpoints/morph/stage0-tul_a0 --out $S/drift_a0.json
$PY lab/divergence/drift_probe.py --config-name tul_a1 \
    --ckpt-dir checkpoints/morph/stage0-tul_a1 --out $S/drift_a1.json

$PY lab/divergence/score_stage0.py --a0 $S/drift_a0.json --a1 $S/drift_a1.json \
    --a0-log $S/tul_a0.log --a1-log $S/tul_a1.log
