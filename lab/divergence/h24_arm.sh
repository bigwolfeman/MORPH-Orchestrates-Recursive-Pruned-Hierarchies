#!/bin/bash
# H24 arm — does training with the core's HCA branch alive change the takeover?
# Pre-registration: docs/experiments/planned/2026-08-25-h24-hca-branch-arm.md
#
# PAIRED and INTERLEAVED (ctrl-s0, arm-s0, ctrl-s1, ...): a drift in machine state hits
# both members of a pair, not one whole arm. SEQUENTIAL: one trainer at a time against a
# loaded GPU (the UPS trips on GPU-100% plus CPU load together).
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/h24arm
mkdir -p $S

COMMON="training.steps=6000 training.batch_size=6 training.ademamix_alpha_cap=3.5
        model.use_kernels=false training.eval_every=250 training.gen_every=0
        training.ckpt_every=1000"

for sd in 0 1 2 3; do
  for arm in ctrl hca16; do
    if [ "$arm" = "ctrl" ]; then CFG=tul_a1; else CFG=tul_a1_hca16; fi
    tag=$arm-s$sd
    echo "START $tag $(date +%H:%M:%S)"
    WANDB_DIR=/home/wolfe/morph-scratch PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
    $PY -m morph.training.train --config-name $CFG \
      hydra.run.dir=$S/hy-$tag \
      $COMMON training.seed=$sd wandb.name=h24-$tag \
      > $S/$tag.log 2>&1
    echo "$tag exit=$? at $(date +%H:%M:%S)"
  done
done
echo "H24 ARM ALL DONE at $(date +%H:%M:%S)"
