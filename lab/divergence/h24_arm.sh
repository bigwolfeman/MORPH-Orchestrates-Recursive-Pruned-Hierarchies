#!/bin/bash
# H24 arm — does training with the core's HCA branch alive change the takeover?
# Pre-registration: docs/experiments/planned/2026-08-25-h24-hca-branch-arm-1seed.md
#
# ONE SEED [W, 2026-08-25]. Seed 0 is the strongest single signal in this campaign: it
# aborted at step 2040 in the seed sweep with a +1.17 nat rise while seeds 1-3 stayed
# inside the 0.168-nat healthy noise floor. The 4-seed design is rejected, not amended:
# docs/experiments/failures/2026-08-25-h24-hca-branch-arm-4seed.md.
#
# SEQUENTIAL: one trainer at a time against a loaded GPU (the UPS trips on GPU-100% plus
# CPU load together).
#
# The CONTROL may already be running from the rejected 4-seed launch. Pass `arm` to run
# only the arm, or no argument to run both.
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/h24arm
mkdir -p $S
WHICH=${1:-both}

COMMON="training.steps=6000 training.batch_size=6 training.ademamix_alpha_cap=3.5
        model.use_kernels=false training.eval_every=250 training.gen_every=0
        training.ckpt_every=1000 training.seed=0"

run () {   # run <tag> <config-name>
  echo "START $1 $(date +%H:%M:%S)"
  WANDB_DIR=/home/wolfe/morph-scratch PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  $PY -m morph.training.train --config-name $2 \
    hydra.run.dir=$S/hy-$1 $COMMON wandb.name=h24-$1 \
    > $S/$1.log 2>&1
  echo "$1 exit=$? at $(date +%H:%M:%S)"
}

[ "$WHICH" = "arm" ] || run ctrl-s0  tul_a1
run hca16-s0 tul_a1_hca16
echo "H24 ARM ALL DONE at $(date +%H:%M:%S)"
