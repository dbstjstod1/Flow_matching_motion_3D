#!/usr/bin/env bash
# D/E extension of the data-step comparison (A/B/C = drive_dcop_compare.sh; A adj won at
# 31.11 dB / 0.790, x_t 29.47, rot 0.23 deg). These two are the SPECTRAL preconditioners the
# low-frequency diagnosis actually calls for (SART's diagonal weights were not):
#   D  fdk  filtered-residual FDK-preconditioned step, eta ramp 0.1 -> 0.5 over t
#   E  cg   5 matched-adjoint CG iterations on the normal equations, warm-started (DDS)
# Common (IDENTICAL to A): 500k ckpt, CQ500 val 0, l2si, N=50, PER=50, uniform K=2, kappa 0.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-1}
COMMON="--ckpt $CK --split val --run 0 --loss l2si"

echo "=== [D] fdk  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op fdk --kappa 0 \
    --out data/dcop_D_fdk

echo "=== [E] cg  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op cg --kappa 0 \
    --out data/dcop_E_cg

echo "=== ALL DONE  $(date) ==="
