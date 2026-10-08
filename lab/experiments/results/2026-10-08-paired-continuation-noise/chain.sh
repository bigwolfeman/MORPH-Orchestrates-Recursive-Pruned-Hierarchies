#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_COMPILE_THREADS=2
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python; P=/home/wolfe/morph-scratch/perf/paired; T=$P/tree
CK=/home/wolfe/morph-wt-graph/checkpoints/morph/lxtul-pointer-rerun/step_2500.pt; LOCK=/home/wolfe/morph-scratch/gpu.lock
note(){ echo "[$(date '+%m-%d %T')] $*" >> $P/chain.log; }
cd $T
for a in a b; do
  note "train $a start"
  WANDB_MODE=offline WANDB_DIR=$P flock $LOCK $PY -m morph.training.train --config-name lxtul_pointer \
    training.resume=$CK training.steps=3500 wandb.name=paired-noise-$a hydra.run.dir=$P/hy_$a > $P/train_$a.log 2>&1
  note "train $a exit=$? $(grep -m1 -o 'Resumed[^,]*' $P/train_$a.log) $(grep -oE 'Final val_loss=[0-9.]+' $P/train_$a.log | tail -1)"
done
for a in a b; do
  C=$(ls -t $T/checkpoints/morph/paired-noise-$a/step_*.pt | head -1)
  flock $LOCK env PYTHONPATH=. $PY lab/divergence/core_depth_sweep.py --ckpt "$a=lxtul_pointer=$C" \
    --depths 1,6 --rows 480 --batch 3 --out $P/sweep_$a.json > $P/sweep_$a.log 2>&1
  note "sweep $a exit=$? ckpt=$(basename $C) $(grep -E 'K1-K6|depth=6' $P/sweep_$a.log | tr -s ' ' | tr '\n' ' ' | cut -c1-260)"
done
