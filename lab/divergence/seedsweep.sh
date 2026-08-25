#!/bin/bash
# Does the forcing bias predict WHICH A1 seeds diverge?
# Pre-registration: docs/experiments/planned/2026-08-24-tul-forcing-bias-predicts-divergence.md
# Sequential (UPS: one trainer at a time against a loaded GPU).
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/seedsweep
mkdir -p $S
for sd in 0 1 2 3; do
  WANDB_DIR=/home/wolfe/morph-scratch PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  $PY -m morph.training.train --config-name tul_a1 \
    hydra.run.dir=$S/hy-s$sd \
    training.steps=3500 training.batch_size=6 training.seed=$sd \
    training.ademamix_alpha_cap=3.5 model.use_kernels=false \
    training.eval_every=250 training.gen_every=0 training.ckpt_every=500 \
    wandb.name=seedsweep-s$sd \
    > $S/s$sd.log 2>&1
  echo "seed $sd exit=$? at $(date +%H:%M:%S)"
done
echo "SEEDSWEEP ALL DONE at $(date +%H:%M:%S)"
