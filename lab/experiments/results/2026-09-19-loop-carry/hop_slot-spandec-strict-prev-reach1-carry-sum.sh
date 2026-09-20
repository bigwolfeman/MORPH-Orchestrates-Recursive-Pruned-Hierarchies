#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/bigdata/morph-instruments/MORPH
echo "ALLDEPTHS start $(date -u +%H:%M:%S)"
PYTHONPATH=. ~/morph-venv/bin/python lab/divergence/hop_distance_probe.py --ckpt 'slot-spandec-strict-prev-reach1-carry-sum=tul_slot_spandec_strict_prev_reach1_carry=/mnt/bigdata/morph-instruments/slot-spandec-strict-prev-reach1-carry-sum_step_5000.pt' --rows 480 --batch 4 --hops 6 --planted --planted-len 2 --depths 1,2,3,4,5,6,7,8,12,16 --out ../results/hop2_slot-spandec-strict-prev-reach1-carry-sum-pair-alldepths_5000.json > ../results/hop2_slot-spandec-strict-prev-reach1-carry-sum-pair-alldepths_5000.log 2>&1
echo "ALLDEPTHS exit=$? $(date -u +%H:%M:%S)"
echo "CUT1 start $(date -u +%H:%M:%S)"
PYTHONPATH=. ~/morph-venv/bin/python lab/divergence/hop_distance_probe.py --ckpt 'slot-spandec-strict-prev-reach1-carry-sum=tul_slot_spandec_strict_prev_reach1_carry=/mnt/bigdata/morph-instruments/slot-spandec-strict-prev-reach1-carry-sum_step_5000.pt' --rows 480 --batch 4 --hops 6 --planted --planted-len 2 --cut-after 1 --depths 1,2,3,6 --out ../results/hop2_slot-spandec-strict-prev-reach1-carry-sum-pair-cut1_5000.json > ../results/hop2_slot-spandec-strict-prev-reach1-carry-sum-pair-cut1_5000.log 2>&1
echo "CUT1 exit=$? $(date -u +%H:%M:%S)"
echo "ALL DONE"
