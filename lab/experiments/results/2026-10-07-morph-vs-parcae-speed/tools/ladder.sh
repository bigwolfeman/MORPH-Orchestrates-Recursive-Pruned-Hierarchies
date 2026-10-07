#!/bin/bash
# Cumulative ladder from production lxtul_pointer toward Parcae's execution recipe.
cd /home/wolfe/morph-scratch/perf/r1007
until grep -q "morph_plain exit" prof_chain.log; do sleep 5; done
B=./bench.sh; C=lxtul_pointer
P1="training.grad_probe_every=0 training.loop_cot_probe=false"
P2="$P1 training.compile_blocks=true training.compile_core_dynamic=false"
P3="$P2 training.ternary=false"
P4="$P3 model.hc_streams=1"
P5="$P4 training.dropout=0.0"
$B L0_base $C
$B L1_noprobe $C $P1
$B L2_cblocks $C $P2
$B L3_noternary $C $P3
$B L4_hc1 $C $P4
$B L5_nodrop $C $P5
