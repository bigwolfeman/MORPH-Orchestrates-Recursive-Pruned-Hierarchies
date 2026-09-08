#!/usr/bin/env bash
# usage: run.sh <tag> <config-name> <micro> <eff> <steps> [extra hydra overrides...]
set -u
SP=/tmp/claude-1000/-mnt-BigAssDrive-00projects-00DeepNet-00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies/f9558148-13bd-4ad7-9000-9456505c04f9/scratchpad/perf
TAG=$1; CFG=$2; MICRO=$3; EFF=$4; STEPS=$5; shift 5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_DIR=/home/wolfe/morph-scratch
export MORPH_PERF_REGIONS=1
export MORPH_DEBUG_STEP=1
export OLYMPIAD_REPO=/mnt/sda1/Projects/00NN/Olympiad-AI
cd /home/wolfe/morph-to
rm -f $SP/probe/$TAG.jsonl
mkdir -p $SP/probe
/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python -m morph.training.train \
  --config-name "$CFG" \
  training.steps=$STEPS \
  curriculum.eff_batch=$EFF \
  "curriculum.stages=[{seq_len:512,context_len:512,micro_batch:$MICRO,steps:$STEPS}]" \
  training.grad_probe_path=$SP/probe/$TAG.jsonl \
  wandb.name=perf-$TAG \
  hydra.run.dir=$SP/hy/$TAG \
  "$@" > $SP/logs/$TAG.log 2>&1
E=$?
WALL=$SECONDS
rm -rf /home/wolfe/morph-to/checkpoints/morph/perf-$TAG
echo "EXIT=$E tag=$TAG wall_s=$WALL"
grep -E "^\[perf\]" $SP/logs/$TAG.log | tail -3
grep -E "^\[ *[0-9]+/" $SP/logs/$TAG.log | tail -2
python3 $SP/summ.py $TAG
