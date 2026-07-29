#!/usr/bin/env bash
# Day-2 sweep, user's four directives (2026-07-24), serialized on ONE GPU:
#
#  [1] TV ON/OFF -- the promoted cg default with the TV triple simply left ON (kappa 0.3 /
#      tv_step 0.03 / tv_iters 5, the 2D winning setting) vs the validated kappa=0 run cg_v0.
#      Same patient/seed, so the ONLY difference is TV. theta-averaged readout EXCLUDED
#      (--theta_avg defaults to 1 now, per user).
#  [2] ORACLE estimator sweep, l2si, GT reference, FIXED 2500 iters/config (= the loop's own
#      N50xPER50 budget): does PER=50 converge; views 24 -> 48 -> 96 trend; lr sweep for a SAFE
#      value; and the band-UNlimited stock-NGP encoder (AI_Geocal's) vs our hashbl, at two lrs.
#  [3] cg_iters DOWN: 1 then 3, vs the cg_v0 baseline's 5. User expects 1 to win -- and it is
#      also the cheapest config, so if it does win the loop gets ~2x faster for free.
#
#   GPU=0 bash scripts/drivers/drive_sweep_day2.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--ckpt $CK --split val --run 0 --seed 3 --loss l2si"

echo "=== [1/4] TV ON (kappa 0.3 default triple), cg, val 0  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op cg --out data/cg_v0_tv

echo "=== [2/4] oracle estimator sweep (l2si, GT ref, 2500 iters/config)  $(date) ==="
$PY scripts/exp_est_sweep.py --suite oracle --ref gt --iters_per_config 2500 \
    --run 0 --seed 3 --out data/est_sweep

echo "=== [3/4] cg_iters=1, kappa 0, val 0  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op cg --cg_iters 1 --kappa 0 \
    --out data/cgiters1_v0

echo "=== [4/4] cg_iters=3, kappa 0, val 0  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op cg --cg_iters 3 --kappa 0 \
    --out data/cgiters3_v0

echo "=== DAY-2 SWEEP DONE  $(date) ==="
