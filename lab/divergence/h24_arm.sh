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
# only the arm, or `both` (default) for both.
#
# STEP BUDGET IS NOT A FREE KNOB. `morph/training/optimizer.py:152`:
#   t_beta3 = int(_tb) if _tb is not None else int(tr.steps)
# and `base.yaml` ships `ademamix_t_beta3: null`. So changing `training.steps` changes the
# optimizer's beta3 WARMUP HORIZON — the slow EMA the whole takeover story runs on. The
# first launch of this arm used 6000 against a seed sweep run at 3500 and the control then
# failed to take over at all. Second argument is the budget so it is always explicit and
# always visible in the log; 3500 matches the seed sweep and the onset ladder.
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
WHICH=${1:-both}
STEPS=${2:-3500}
S=/home/wolfe/morph-scratch/h24arm$STEPS
mkdir -p $S
echo "budget: training.steps=$STEPS  ->  ademamix_t_beta3=$STEPS (null in the yaml)"

COMMON="training.steps=$STEPS training.batch_size=6 training.ademamix_alpha_cap=3.5
        model.use_kernels=false training.eval_every=250 training.gen_every=0
        training.ckpt_every=500 training.seed=0"

run () {   # run <tag> <config-name>
  echo "START $1 $(date +%H:%M:%S)"
  WANDB_DIR=/home/wolfe/morph-scratch PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  $PY -m morph.training.train --config-name $2 \
    hydra.run.dir=$S/hy-$1 $COMMON wandb.name=h24-$1-$STEPS \
    > $S/$1.log 2>&1
  echo "$1 exit=$? at $(date +%H:%M:%S)"
}

[ "$WHICH" = "arm" ] || run ctrl-s0  tul_a1
run hca16-s0 tul_a1_hca16
echo "H24 ARM ALL DONE at $(date +%H:%M:%S)"
