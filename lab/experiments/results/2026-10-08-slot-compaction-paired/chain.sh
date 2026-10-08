#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_COMPILE_THREADS=2
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python; P=/home/wolfe/morph-scratch/perf/paired3; T=$P/tree
CK=/home/wolfe/morph-wt-graph/checkpoints/morph/lxtul-pointer-rerun/step_2500.pt; LOCK=/home/wolfe/morph-scratch/gpu.lock
FAST2="training.grad_probe_every=0 training.loop_cot_probe=false training.capturable_optimizer=true training.compile_core_dynamic=false model.graph_safe=true training.graph_step=true training.compile_blocks=true model.tg_fused_attention=true model.hc_fused_grad=true model.ternary_step_cache=true model.ckpt_grad_iters=0 model.slot_gain_no_ckpt=true model.fan_twin_compile=true model.fan_div_fast=true model.fan_lsel_pick_bf16=true model.fan_target_online=true model.slot_depth_stratified=true model.slot_compact=gather"
note(){ echo "[$(date '+%m-%d %T')] $*" >> $P/chain.log; }
cd $T
for a in g h i; do
  note "train $a start"
  WANDB_MODE=offline WANDB_DIR=$P flock $LOCK $PY -m morph.training.train --config-name lxtul_pointer $FAST2 \
    training.resume=$CK training.steps=3500 wandb.name=paired-cpt-$a hydra.run.dir=$P/hy_$a > $P/train_$a.log 2>&1
  note "train $a exit=$? $(grep -m1 -o 'Resumed[^,]*' $P/train_$a.log) $(grep -oE 'Final val_loss=[0-9.]+' $P/train_$a.log | tail -1)"
done
for a in g h i; do
  C=$(ls -t $T/checkpoints/morph/paired-cpt-$a/step_*.pt | head -1)
  flock $LOCK env PYTHONPATH=. $PY lab/divergence/core_depth_sweep.py --ckpt "$a=lxtul_pointer=$C" \
    --depths 1,6 --rows 480 --batch 3 --out $P/sweep_$a.json > $P/sweep_$a.log 2>&1
  note "sweep $a exit=$? ckpt=$(basename $C) $(grep -E 'K1-K6|depth=' $P/sweep_$a.log | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
done
