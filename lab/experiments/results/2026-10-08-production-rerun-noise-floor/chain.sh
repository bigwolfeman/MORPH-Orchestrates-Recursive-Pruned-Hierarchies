#!/bin/bash
export TORCHINDUCTOR_COMPILE_THREADS=2   # UPS: cap compile CPU load (Wolfe 2026-10-08)
# Noise floor: production lxtul_pointer, seed 1, eager, same code path as the 2026-10-06 winner run
# (all new keys off; their off paths are gated byte-identical). Starts after the identity agent's gates.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python; WT=/home/wolfe/morph-wt-graph
D=/home/wolfe/morph-scratch/perf/graph/noise; LOCK=/home/wolfe/morph-scratch/gpu.lock
note(){ echo "[$(date '+%m-%d %T')] $*" >> $D/chain.log; }
until [ -f /home/wolfe/morph-scratch/perf/graph/identity/GATES_DONE ]; do sleep 30; done
cd $WT
note "train start $(git rev-parse --short HEAD)"
WANDB_MODE=offline WANDB_DIR=$D flock $LOCK $PY -m morph.training.train --config-name lxtul_pointer \
  wandb.name=lxtul-pointer-rerun hydra.run.dir=$D/hy > $D/train.log 2>&1
note "train exit=$? $(grep -oE 'Final val_loss=[0-9.]+' $D/train.log | tail -1)"
flock $LOCK env PYTHONPATH=. $PY lab/divergence/core_depth_sweep.py \
  --ckpt "rerun=lxtul_pointer=$WT/checkpoints/morph/lxtul-pointer-rerun/step_5000.pt" \
  --depths 1,6 --rows 480 --batch 3 --out $D/sweep.json > $D/sweep.log 2>&1
note "sweep exit=$? $(grep -E 'K1-K6|depth=' $D/sweep.log | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
