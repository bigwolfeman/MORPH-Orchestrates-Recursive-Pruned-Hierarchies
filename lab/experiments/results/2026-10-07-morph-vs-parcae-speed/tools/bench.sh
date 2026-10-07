#!/bin/bash
# bench.sh TAG CFG [overrides...]: 420-step MORPH bench under the GPU lock, timestamped log in runs/TAG.
TAG=$1; CFG=$2; shift 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
WT=${WT:-/home/wolfe/morph-wt-lsel}
R=/home/wolfe/morph-scratch/perf/r1007; OUT=$R/runs/$TAG; mkdir -p $OUT
cd $WT || exit 1
echo "[$(date +%T)] $TAG start" >> $R/bench.log
env WANDB_MODE=offline WANDB_DIR=$OUT PYTHONPATH=$WT flock /home/wolfe/morph-scratch/gpu.lock $PY -m morph.training.train \
  --config-path $WT/morph/configs --config-name $CFG \
  training.steps=420 training.eval_every=100000 training.n_eval_batches=1 training.ckpt_every=100000 \
  wandb.name=bench-$TAG hydra.run.dir=$OUT/hy "$@" 2>&1 | /home/wolfe/morph-scratch/perf/scripts/tsfilter.py > $OUT/run.log
rc=${PIPESTATUS[0]}
rm -rf "$WT/checkpoints/morph/bench-$TAG"
echo "[$(date +%T)] $TAG exit=$rc $(/home/wolfe/morph-scratch/perf/r1007/summ.py $TAG 2>&1 | tail -1)" >> $R/bench.log
