#!/bin/bash
# Stage 0 of the SCSE plan, on the 5090.
# Pre-registration: docs/experiments/planned/2026-08-24-tul-forcing-bias-arm-control.md
#
# SEQUENTIAL, not concurrent. Concurrency was justified on the Spark by 25% GPU utilisation
# (launch-bound work, so two arms interleave). The 5090 does not have that headroom, and two
# trainers against a loaded GPU is the configuration that trips this workstation's UPS.
#
# Same build as checkpoints/morph/onset-capture, so these arms are comparable to each other
# AND to the existing A1 onset ladder.
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch
mkdir -p $S/stage0
for arm in tul_a0 tul_a1; do
  WANDB_DIR=$S PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  $PY -m morph.training.train --config-name "$arm" \
    hydra.run.dir=$S/stage0/hy-$arm \
    training.steps=1900 training.batch_size=6 training.seed=0 \
    training.ademamix_alpha_cap=3.5 model.use_kernels=false \
    training.eval_every=500 training.gen_every=0 training.ckpt_every=300 \
    wandb.name=stage0-$arm \
    > $S/stage0/$arm.log 2>&1
  echo "$arm exit=$? at $(date +%H:%M:%S)"
done
echo "STAGE0 BOTH ARMS DONE at $(date +%H:%M:%S)"
