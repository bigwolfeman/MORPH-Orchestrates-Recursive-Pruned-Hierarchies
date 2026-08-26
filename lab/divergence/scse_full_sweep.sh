#!/bin/bash
# The FULL SCSE method against a matched tul_a1 control.
# Pre-registration: lab/experiments/planned/2026-08-25-scse-full-method.md
# Implementation:   docs/scse-spec.md
#
# Sequential by design (UPS: one trainer at a time against a loaded GPU — two threads at
# full CPU while the GPU is over 530 W trips it).
#
# PAIR ORDER, not arm order: ctrl-1, scse-1, ctrl-2, scse-2, ... so that an interrupted
# campaign still yields COMPLETE PAIRS. Scoring is paired by seed, so four control runs and
# one SCSE run would be worth nothing.
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/scse2/runs
mkdir -p $S

run () {                       # $1 = arm (ctrl|scse), $2 = seed
  local arm=$1 sd=$2 ov
  if [ "$arm" = "scse" ]; then ov=true; else ov=false; fi
  WANDB_DIR=/home/wolfe/morph-scratch PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  $PY -m morph.training.train --config-name tul_a1 \
    hydra.run.dir=$S/hy-$arm-s$sd \
    training.steps=3500 training.batch_size=6 training.seed=$sd \
    training.ademamix_alpha_cap=3.5 model.use_kernels=false \
    model.scse_enabled=$ov \
    training.eval_every=250 training.gen_every=0 training.ckpt_every=1000 \
    wandb.name=scse2-$arm-s$sd \
    > $S/$arm-s$sd.log 2>&1
  echo "$arm s$sd exit=$? at $(date +%H:%M:%S)"
}

for sd in 1 2 3 4; do
  run ctrl $sd
  run scse $sd
done
# Seed 0 is pathological in the CONTROL (it diverged in H21 and again in H23), so it is a
# divergence probe against a known outcome, not a CE pair. Excluded from P2/P3, run LAST.
run scse 0
echo "SCSE2 ALL DONE at $(date +%H:%M:%S)"
