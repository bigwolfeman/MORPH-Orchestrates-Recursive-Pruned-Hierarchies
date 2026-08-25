#!/bin/bash
# H24 — does training with the core's HCA compressed branch alive stop the TUL takeover?
# Pre-registration: docs/experiments/planned/2026-08-25-h24-hca-branch-arm-binary.md
#
# THE SIGNAL IS BINARY: the divergence guard fires, or it does not. Everything here is
# chosen so the CONTROL reliably fires, because a control that survives answers nothing.
#
# REGIME = the one in docs/tul-divergence-rca.md §1, where A1 aborted at step 4540 and
# A1r (seed 1) at 3240 — "Two seeds fail the same way. This is structural, not seed luck."
# That means tul_a1 at batch 12, alpha_cap 3.5, PRODUCTION KERNELS, and the full 20k-step
# optimizer schedule.
#
# ademamix_t_beta3 IS PINNED. `morph/training/optimizer.py:152` falls back to
# training.steps when the key is null, which base.yaml ships — so shortening a run
# silently shortens the optimizer's beta3 warmup, the slow EMA this whole failure runs on.
# Pinning 20000 reproduces the RCA schedule exactly while stopping at 6000, which is past
# both documented abort steps. An earlier launch of this arm changed the budget instead of
# pinning the horizon and its control never took over at all.
#
# SEQUENTIAL and INTERLEAVED: one trainer at a time (the UPS trips on GPU-100% plus CPU
# load together), and ctrl/arm alternate per seed so machine drift hits both.
set -u
cd /home/wolfe/morph-perf
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python
S=/home/wolfe/morph-scratch/h24bin
mkdir -p $S

COMMON="training.steps=6000 training.ademamix_t_beta3=20000 training.batch_size=12
        training.ademamix_alpha_cap=3.5 training.eval_every=250 training.gen_every=0
        training.ckpt_every=1000"

run () {   # run <tag> <config-name> <seed>
  echo "START $1 $(date +%H:%M:%S)"
  WANDB_DIR=/home/wolfe/morph-scratch PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  $PY -m morph.training.train --config-name $2 \
    hydra.run.dir=$S/hy-$1 $COMMON training.seed=$3 wandb.name=h24-$1 \
    > $S/$1.log 2>&1
  echo "$1 exit=$? at $(date +%H:%M:%S)"
}

for sd in 0 1; do
  run ctrl-s$sd  tul_a1        $sd
  run hca16-s$sd tul_a1_hca16  $sd
done
echo "H24 BINARY ARM ALL DONE at $(date +%H:%M:%S)"
