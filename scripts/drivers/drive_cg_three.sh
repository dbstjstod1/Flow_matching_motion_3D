#!/usr/bin/env bash
# THE DELIVERABLE SET for the promoted data step: `cg` on three CQ500 val patients.
#
# Each patient gets its OWN motion draw (seed 3 / 7 / 11) rather than the same one three times, so
# the set samples anatomy AND motion. That is deliberately the harder read -- a per-patient spread
# here mixes both factors -- but repeating one trajectory would leave the whole set resting on a
# single motion draw, which is exactly the weakness the val-0 A/B already had.
#
# --kappa 0 REPRODUCES THE VALIDATED CONFIGURATION: every run in the 5-way data-step A/B had TV
# off. The script's own kappa default is 0.3 (inherited from the 2D by-eye streak sweep, tuned
# when the data step was `adj`), and cg-plus-TV is untested -- so it is pinned here rather than
# left to the default. Everything else is the default: N=50, PER=50, l2si, uniform K=2,
# cg_iters=5, cg_lam=0.
#
#   GPU=0 bash scripts/drivers/drive_cg_three.sh
# Roughly 70 min per patient on this box (~5 fwd+adj pairs per ODE step), so ~3.5 h total.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one () {
    local idx=$1 seed=$2
    echo "=== [cg] val patient $idx, motion seed $seed  $(date) ==="
    $PY scripts/run_posterior3d.py --ckpt $CK --split val --run "$idx" --seed "$seed" \
        --loss l2si --dc_op cg --kappa 0 --out "data/cg_v${idx}"
}

run_one 0 3
run_one 1 7
run_one 2 11

echo "=== THREE-PATIENT cg SET DONE  $(date) ==="
