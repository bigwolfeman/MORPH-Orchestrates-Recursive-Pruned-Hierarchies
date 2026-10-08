#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_COMPILE_THREADS=2
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python; S=/home/wolfe/morph-scratch/perf/fast5k; T=$S/tree; LOCK=/home/wolfe/morph-scratch/gpu.lock
note(){ echo "[$(date '+%m-%d %T')] $*" >> $S/chain.log; }
cd $T
note "train start"
WANDB_MODE=offline WANDB_DIR=$S flock $LOCK $PY -m morph.training.train --config-name lxtul_pointer_fast \
  wandb.name=lxtul-pointer-fast-5k hydra.run.dir=$S/hy > $S/train.log 2>&1
note "train exit=$? $(grep -oE 'Final val_loss=[0-9.]+' $S/train.log | tail -1)"
C=$(ls -t $T/checkpoints/morph/lxtul-pointer-fast-5k/step_*.pt | head -1)
flock $LOCK env PYTHONPATH=. $PY lab/divergence/core_depth_sweep.py --ckpt "fast=lxtul_pointer=$C" \
  --depths 1,6 --rows 480 --batch 3 --out $S/sweep.json > $S/sweep.log 2>&1
note "sweep exit=$? ckpt=$(basename $C) $(grep -E 'K1-K6|depth=' $S/sweep.log | tr -s ' ' | tr '\n' ' ' | cut -c1-320)"
