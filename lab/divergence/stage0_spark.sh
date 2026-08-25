#!/bin/bash
# Stage 0 of the SCSE plan (.agents/notes/proposed/architecture/2026-08-24-scse-source-centered-core-loop.md):
# b_t / R_t as a function of TRAINING STEP, for both arms, in ONE build.
#
# Both arms, not just A0: checkpoints/morph/onset-capture/README.md records that two torch
# builds gave different trajectories from the same seed. The A1 ladder already in hand came
# from the 5090; comparing a Spark A0 ladder against it would confound the arm with the
# build. Running both here removes that. The extra arm is free -- the GPU is idle.
#
# Concurrent, not sequential: measured GPU utilisation is 25% at 0.12 sps, so this workload
# is launch-bound rather than compute-bound and two processes interleave instead of halving
# each other. 20 CPU cores, 95 GB RAM free, 18 GB peak per arm against 121 GB unified.
#
# ckpt_every=300 gives 6 rungs per arm at 2.2 GB each; disk is the binding constraint
# (56 GB free). Stage 0 wants a TREND across training, not a 25-step bracket on an onset.
set -u
cd ~/morph-perf
arm=$1
mkdir -p ~/out
WANDB_DIR=$HOME/out PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
nohup .venv/bin/python -m morph.training.train --config-name "$arm" \
  hydra.run.dir=$HOME/out/hy-$arm \
  data.dataset="$HOME/owt/openwebtext-train-*.arrow" \
  training.steps=1900 training.batch_size=6 training.seed=0 \
  training.ademamix_alpha_cap=3.5 model.use_kernels=false \
  training.eval_every=500 training.gen_every=0 training.ckpt_every=300 \
  wandb.name=stage0-$arm \
  > $HOME/out/$arm.log 2>&1 &
echo "$arm launched pid=$!"
