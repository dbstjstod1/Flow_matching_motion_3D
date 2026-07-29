#!/usr/bin/env bash
# Blind posterior on CQ500 val patient 0, N=50 / PER=50 / uniform K=2 / tv_iters=15, 500k ckpt.
# lncc first (user's pick, backed by the 3D oracle), then l2si (the 2D winner) for the A/B.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=0

echo "=== [1/2] lncc  $(date) ==="
$PY scripts/run_posterior3d.py --ckpt $CK --split val --run 0 \
    --loss lncc --out data/posterior3d_val0_lncc_N50

echo "=== [2/2] l2si  $(date) ==="
$PY scripts/run_posterior3d.py --ckpt $CK --split val --run 0 \
    --loss l2si --out data/posterior3d_val0_l2si_N50

echo "=== BOTH DONE  $(date) ==="
