#!/bin/bash
# LN common-mode probe driver (2026-10-02). One flock hold per checkpoint.
# Curfew: no new hold after 15:50 CDT; every hold is killed by 15:58.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WT=/home/wolfe/morph-wt-lsel; CK=/home/wolfe/morph-to/checkpoints/morph
PY=/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python; LOCK=/home/wolfe/morph-scratch/gpu.lock
OUT=/home/wolfe/morph-scratch/lnprobe; LOG=$OUT/driver.log
P=tul_slot_spandec_strict_fan4_all_fp01
log(){ echo "[$(date '+%T')] $*" >> $LOG; }
ARMS=(
 "det_w1 ${P}_lsel_det_rf_lam1 slot-spandec-strict-fan4-all-fp01-lsel-det-rf-lam1 --follow"
 "joint_rank ${P}_lsel_joint_rf_lam1_rank slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank --follow"
 "plain notul_panel_norm_match_s1r plain-panel-nm-ctrl-s1r"
 "fan ${P}_nowta slot-spandec-strict-fan4-all-fp01-nowta"
 "joint_w1 ${P}_lsel_joint_rf_lam1 slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1"
 "det_w10 ${P}_lsel_det_rf slot-spandec-strict-fan4-all-fp01-lsel-det-rf"
 "joint_w10 ${P}_lsel_joint_rf slot-spandec-strict-fan4-all-fp01-lsel-joint-rf"
 "det_tf ${P}_lsel_det slot-spandec-strict-fan4-all-fp01-lsel-det"
 "joint_tf ${P}_lsel_joint slot-spandec-strict-fan4-all-fp01-lsel-joint"
 "det_all ${P}_lsel_det_rf_all slot-spandec-strict-fan4-all-fp01-lsel-det-rf-all"
)
cd $WT
for a in "${ARMS[@]}"; do
  read -r lab cfg tag extra <<< "$a"
  [ -e $OUT/$lab.json ] && { log "$lab exists, skip"; continue; }
  now=$(date +%s); stop=$(date -d '15:50' +%s); kill=$(date -d '15:58' +%s)
  [ $now -ge $stop ] && { log "CURFEW: $lab and later not started"; break; }
  log "$lab wait lock"
  flock $LOCK bash -c "now=\$(date +%s); [ \$now -ge $stop ] && exit 99; \
    timeout --signal=KILL \$(( $kill - \$now )) env PYTHONPATH=. $PY lab/divergence/ln_common_mode_probe.py \
    --ckpt $lab=$cfg=$CK/$tag/step_5000.pt $extra --rows 96 --fit_rows 96 --batch 3 --out $OUT/$lab.json" \
    > $OUT/$lab.log 2>&1
  rc=$?; log "$lab exit=$rc $(tail -1 $OUT/$lab.log | cut -c1-100)"
  [ $rc -eq 99 ] && { log "CURFEW inside lock: stop"; break; }
done
log "driver done"
