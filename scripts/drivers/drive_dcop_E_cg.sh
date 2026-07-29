#!/usr/bin/env bash
# E alone: DDS-style short CG data step. Split out of drive_dcop_fdkcg.sh after the first attempt
# died of GPU CONTENTION (2026-07-24: another job held 41.8 of 49 GiB on the target card, our
# process had 6.45 GiB and could not get one more 482 MB sinogram). Pick a card with room:
#   GPU=0 bash scripts/drivers/drive_dcop_E_cg.sh
# expandable_segments cuts allocator fragmentation, which is what turns "tight" into "OOM" when
# the run shares a card.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [E] cg on GPU $CUDA_VISIBLE_DEVICES  $(date) ==="
$PY scripts/run_posterior3d.py --ckpt $CK --split val --run 0 --loss l2si \
    --dc_op cg --kappa 0 --out data/dcop_E_cg

echo "=== DONE  $(date) ==="
