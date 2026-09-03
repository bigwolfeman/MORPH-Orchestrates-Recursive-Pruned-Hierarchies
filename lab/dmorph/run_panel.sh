#!/usr/bin/env bash
# dmorph v1 panel (lab/experiments/failures/2026-09-03-dmorph-v1-panel.md): three arms x
# two seeds, 20k steps each, seq 1024 batch 6, ONE trainer at a time on the 5090.
#
# PREPARED, NOT LAUNCHED. Wolfe launches it. Nothing here starts on its own.
#
#   setsid nohup bash lab/dmorph/run_panel.sh </dev/null >/home/wolfe/morph-scratch/dmorph-panel.log 2>&1 & disown
#
# Order: ctl s1, tok s1, hs s1, ctl s2, tok s2, hs s2 — the control first so the bar is
# on disk before any arm finishes, and seed 1 of every arm before seed 2 of any.
# Abort rule from the prereg: preclip/total > 1e4 after step 200 is a detonation
# (lab/divergence/DIVERGENCE-README.md); the trainer's own divergence guard enforces it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MORPH_PY:-/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python}"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # mandatory on the 5090 (CLAUDE.md)
export WANDB_DIR="${WANDB_DIR:-/home/wolfe/morph-scratch}"
export WANDB_PROJECT=morph-tul
export WANDB_ENTITY=adew-me
STEPS="${DMORPH_STEPS:-20000}"
# Method amendment 2026-09-03: the reshaped source on both dmorph arms (prereg, amendment 2).
DMORPH_SRC="${DMORPH_SOURCE_STD:-1.0}"
TAG="${DMORPH_TAG:-}"
SEEDS="${DMORPH_SEEDS:-1 2}"     # Wolfe 2026-09-03: one seed each for the 5k panel (DMORPH_SEEDS=1)
# Which configs to run, in order. The FPF arm (dmorph_tok_fpf, v1.1) is launched alone:
#   DMORPH_ARMS=dmorph_tok_fpf DMORPH_SEEDS=1 DMORPH_STEPS=5000 DMORPH_TAG=-5k
ARMS="${DMORPH_ARMS:-dmorph_ctl dmorph_tok dmorph_hs}"
RESULTS="$ROOT/lab/experiments/results/2026-09-03-dmorph-v1-panel"
mkdir -p "$RESULTS"

gpu_is_free() {
  # Refuse to start while another compute process holds > 2 GiB (Wolfe's desktop
  # sits at ~1 GiB across two processes).
  local busy
  busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
         | awk '$1 > 2048 {c++} END {print c+0}')
  [ "$busy" = "0" ]
}

run_arm() {
  local cfg="$1" seed="$2"
  local name="${cfg//_/-}-s${seed}${TAG}"          # dmorph_tok_fpf -> dmorph-tok-fpf
  local extra=()
  if [ "$cfg" != "dmorph_ctl" ]; then extra=(dmorph.source_std="$DMORPH_SRC"); fi
  until gpu_is_free; do
    echo "[panel] $(date +%H:%M:%S) GPU busy, waiting 60 s before $name"
    sleep 60
  done
  # Checked AFTER the wait: a run finishing on the card while we waited must be skipped.
  if [ -f "$ROOT/checkpoints/morph/$name/step_${STEPS}.pt" ]; then
    echo "[panel] $name already at step $STEPS, skipping"
    return 0
  fi
  echo "[panel] $(date +%H:%M:%S) START $name"
  # `set -e` must NOT end the panel on one failed arm (2026-09-03: the tok abort took
  # the hs run with it). The trainer's exit code is captured and reported instead.
  local rc=0
  ( cd "$ROOT" && "$PY" -m morph.training.train --config-name "$cfg" \
      training.steps="$STEPS" training.seed="$seed" wandb.name="$name" "${extra[@]}" \
      2>&1 | tee "$RESULTS/$name.log"; exit "${PIPESTATUS[0]}" ) || rc=$?
  echo "[panel] $(date +%H:%M:%S) DONE  $name (exit $rc)"
}

for seed in $SEEDS; do
  for cfg in $ARMS; do
    run_arm "$cfg" "$seed"
  done
done
echo "[panel] all runs finished (arms: $ARMS; seeds: $SEEDS) $(date)"
