#!/bin/bash
# End-to-end num_stages/num_warps sweep on the 30B dominant GEMVs.
# Each config = a full build+bench process (env read at import). Parity stays green
# (schedule-only knobs never touch math). Logs the SUMMARY-JSON line per config.
set -u
cd /tmp/morph-30b-lever-wt
PY="/home/wolfe/.venv/bin/python"
run() {
  local tag="$1"; shift
  echo "==================== CONFIG: $tag ($*) ===================="
  env "$@" PYTHONPATH=$PWD $PY ignore/scale30b_bench_fast.py --gate-steps 8 --bench 64 \
      2>&1 | grep -E 'SUMMARY-JSON|BENCH\]|gate\]|roofline\] HBM' | sed "s/^/[$tag] /"
}
run "baseline"        # env unset = stages3/warps4 both
run "mortar_s4"   MORPH_MORTAR_STAGES=4
run "mortar_w8"   MORPH_MORTAR_WARPS=8
run "front_s4"    MORPH_FRONT_STAGES=4
run "front_w8"    MORPH_FRONT_WARPS=8
echo "SWEEP_DONE"
