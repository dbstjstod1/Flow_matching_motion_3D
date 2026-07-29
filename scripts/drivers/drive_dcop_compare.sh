#!/usr/bin/env bash
# Three-way controlled comparison of the DATA step, everything else identical:
#   A  adj   raw-adjoint normalized soft step (the 2D recipe)      -- baseline, 31.10 dB / 0.788
#   B  sart  SART row/column-normalized algebraic update
#   C  sart + ASD-POCS adaptive TV coupling (dtvg = alpha*dp, shrink when dg > rmax*dp)
# Common: 500k ckpt, CQ500 val 0, l2si, N=50, PER=50, uniform K=2, n_samples=256.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=0
COMMON="--ckpt $CK --split val --run 0 --loss l2si"

echo "=== [A] adj (baseline)  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op adj --kappa 0 \
    --out data/dcop_A_adj

echo "=== [B] sart  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op sart --kappa 0 \
    --out data/dcop_B_sart

echo "=== [C] sart + ASD-POCS  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op sart --asd --asd_ng 20 \
    --out data/dcop_C_asdpocs

echo "=== ALL DONE  $(date) ==="
