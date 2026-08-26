#!/bin/bash
# H23: does SCSE Stage 1 (Delta_0 != 0) stop the A1 takeover?
# Pre-registration: lab/experiments/planned/2026-08-25-scse-stage1-initial-deviation.md
#
# Identical to lab/divergence/seedsweep.sh except model.core_init_scale=0.1.
# The control is that sweep's output; core_init_scale=0.0 is bit-identical to the code
# that produced it (tests/test_scse_core_init.py).
# Sequential (UPS: one trainer at a time against a loaded GPU).
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/scse1
mkdir -p $S
for sd in 0 1 2 3; do
  WANDB_DIR=/home/wolfe/morph-scratch PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  $PY -m morph.training.train --config-name tul_a1 \
    hydra.run.dir=$S/hy-s$sd \
    model.core_init_scale=0.1 \
    training.steps=3500 training.batch_size=6 training.seed=$sd \
    training.ademamix_alpha_cap=3.5 model.use_kernels=false \
    training.eval_every=250 training.gen_every=0 training.ckpt_every=500 \
    wandb.name=scse1-s$sd \
    > $S/s$sd.log 2>&1
  echo "seed $sd exit=$? at $(date +%H:%M:%S)"
done
echo "SCSE1 ALL DONE at $(date +%H:%M:%S)"
